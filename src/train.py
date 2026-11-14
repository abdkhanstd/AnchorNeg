"""Bi-encoder training for misconception retrieval.

Modes
  zeroshot : no training, frozen encoder
  ce       : standard fine-tuning, linear classifier over misconception ids
  mnrl     : in-batch contrastive (query -> misconception), random batches
  ours     : answer contrast sets (question-grouped batches, reverse
             misconception -> response loss with the correct-answer view and
             sibling distractors as counterfactual negatives)
Flags select the components so that each ablation removes one of them.
"""
import argparse, json, os, time, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import data as D
import metrics as Mx

MODELS = {
    "bert": "bert-base-uncased",
    "deberta": "microsoft/deberta-v3-base",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}
def mpath(k):
    d = os.path.join(D.ROOT, "hf_models", k)
    return d if os.path.isdir(d) else MODELS[k]


QPREFIX = {"bge": "Represent this sentence for searching relevant passages: "}
QPREFIX["bge-small"] = QPREFIX["bge-large"] = QPREFIX["bge"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bge")
    p.add_argument("--mode", default="ours", choices=["zeroshot", "ce", "mnrl", "ours"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bq", type=int, default=16, help="questions per batch")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--hard", type=int, default=4, help="mined hard negatives per query (0 = none)")
    p.add_argument("--no_correct", action="store_true", help="drop correct answer from the input text")
    p.add_argument("--group", type=int, default=-1, help="question-grouped batches (default: on for ours)")
    p.add_argument("--rev", type=float, default=-1, help="weight of reverse loss (default 1 for ours, 0 otherwise)")
    p.add_argument("--rev_correct", type=int, default=1, help="correct-answer view as counterfactual negative")
    p.add_argument("--qlen", type=int, default=256)
    p.add_argument("--mlen", type=int, default=48)
    p.add_argument("--tag", default="")
    p.add_argument("--save", action="store_true")
    p.add_argument("--dump", type=int, default=1, help="save label/query embeddings for offline scoring")
    p.add_argument("--diff", type=float, default=0.0, help="beta: query = f(wrong) - beta * f(correct view)")
    p.add_argument("--rev_local", type=int, default=0, help="reverse softmax only over the answers of the same question")
    p.add_argument("--rev_all", type=int, default=0, help="unlabelled distractors also act as negative views")
    p.add_argument("--distill", type=float, default=0.0, help="weight of unseen-label consistency distillation")
    p.add_argument("--dk", type=int, default=128, help="labels sampled per step for distillation")
    p.add_argument("--dpool", default="unseen", choices=["unseen", "all"], help="label pool for distillation")
    p.add_argument("--neg_pool", default="all", choices=["all", "seen", "random"], help="label pool for mined hard negatives (random: uniform subset of the inventory with the size of the seen set)")
    p.add_argument("--remine", type=int, default=0, help="re-mine hard negatives with the current encoder every epoch")
    p.add_argument("--freeze_mis", type=int, default=0, help="misconception tower kept frozen at the pretrained encoder")
    p.add_argument("--out", default=os.path.join(D.ROOT, "results", "runs"))
    a = p.parse_args()
    if a.group < 0:
        a.group = int(a.mode == "ours")
    if a.rev < 0:
        a.rev = 1.0 if a.mode == "ours" else 0.0
    return a


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class Encoder(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.m = AutoModel.from_pretrained(name).float()
        self.cls = None

    def forward(self, batch):
        h = self.m(**batch).last_hidden_state
        msk = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
        return e


def encode(model, tok, texts, maxlen, bs=128, dev="cuda", normalize=True):
    model.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
            e = model(b).float()
            out.append(F.normalize(e, dim=-1) if normalize else e)
    return torch.cat(out)


def main():
    a = get_args()
    seed_all(a.seed)
    dev = "cuda"
    name = a.tag or (f"{a.model}_{a.mode}" + ("_nocorr" if a.no_correct else "")
                     + f"_h{a.hard}_g{a.group}_r{a.rev:g}_rc{a.rev_correct}" + ("_fz" if a.freeze_mis else "") + (f"_d{a.diff:g}" if a.diff else "") + ("_ra" if a.rev_all else "") + ("_rl" if a.rev_local else "") + ({"seen": "_npseen", "random": "_nprand"}.get(a.neg_pool, "")) + ("_dyn" if a.remine else "") + (f"_kd{a.distill:g}" + ("" if a.dpool == "unseen" else "all") if a.distill else ""))
    odir = os.path.join(a.out, name)
    os.makedirs(odir, exist_ok=True)
    ofile = os.path.join(odir, f"f{a.fold}_s{a.seed}.json")
    if os.path.exists(ofile):
        print("exists", ofile); return

    df, mis = D.load()
    folds = D.construct_folds(df)
    tr_df, te_df = df[folds != a.fold], df[folds == a.fold]
    wc = not a.no_correct
    tr_items, te_items = D.build_items(tr_df, wc), D.build_items(te_df, wc)
    pre = QPREFIX.get(a.model, "")
    for it in tr_items + te_items:
        it["wrong"] = [(pre + t, m, i) for t, m, i in it["wrong"]]
        it["correct"] = pre + it["correct"]
        it["views"] = [pre + v for v in it["views"]]
    te_q, te_y, te_ids = D.flat_queries(te_items)
    tr_q, tr_y, _ = D.flat_queries(tr_items)
    seen = set(tr_y.tolist())

    tok = AutoTokenizer.from_pretrained(mpath(a.model))
    model = Encoder(mpath(a.model)).to(dev)
    M = len(mis)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    mmodel = model
    if a.freeze_mis and a.mode in ("mnrl", "ours"):
        mmodel = Encoder(mpath(a.model)).to(dev).eval()
        for p_ in mmodel.parameters():
            p_.requires_grad_(False)
    if a.mode == "ce":
        model.cls = torch.nn.Linear(model.m.config.hidden_size, M).to(dev)

    if a.mode != "zeroshot":
        # mined hard negatives from the frozen encoder (same for every method that uses them)
        def mine():
            me = encode(mmodel, tok, mis, a.mlen)
            qe = encode(model, tok, tr_q, a.qlen)
            sc = qe @ me.T
            sc[torch.arange(len(tr_y)), torch.tensor(tr_y)] = -1e4
            if a.neg_pool in ("seen", "random"):
                keep = np.unique(tr_y)
                if a.neg_pool == "random":
                    keep = np.random.RandomState(1000 + 10 * a.seed + a.fold).choice(M, len(keep), replace=False)
                never = torch.ones(M, dtype=torch.bool, device=sc.device)
                never[torch.as_tensor(keep, device=sc.device)] = False
                sc[:, never] = -1e4
            top = sc.topk(50, dim=1).indices.cpu().numpy()
            model.train()
            return {t: top[i] for i, t in enumerate(tr_q)}

        hard_pool = None
        if a.hard > 0 and a.mode in ("mnrl", "ours"):
            hard_pool = mine()
        if a.distill > 0:
            t_me = encode(model, tok, mis, a.mlen)
            t_q = {t: e for t, e in zip(tr_q, encode(model, tok, tr_q, a.qlen))}
            pool = np.array(sorted(set(range(M)) - set(tr_y.tolist()))) if a.dpool == "unseen" else np.arange(M)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        n_q = len(tr_q)
        steps_per_epoch = max(1, n_q // max(1, int(a.bq * 2.3)))
        total = steps_per_epoch * a.epochs
        warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
        scaler = torch.amp.GradScaler()
        step = 0
        log = []
        for ep in range(a.epochs):
            if a.remine and ep > 0 and hard_pool is not None:
                hard_pool = mine()   # dynamic mining with the current encoder
            model.train()
            if a.group:
                order = np.random.permutation(len(tr_items))
                batches = []
                cur, cnt = [], 0
                for j in order:
                    cur.append(tr_items[j]); cnt += len(tr_items[j]["wrong"])
                    if len(cur) == a.bq:
                        batches.append(cur); cur = []
                if cur: batches.append(cur)
                # each batch: list of (text, mid, qidx) wrong views + correct views
                bdata = []
                for b in batches:
                    w = [(t, m, k) for k, it in enumerate(b) for t, m, _ in it["wrong"]]
                    c = [(it["correct"], k) for k, it in enumerate(b)]
                    x = [(v, k) for k, it in enumerate(b) for j, v in enumerate(it["views"])
                         if j not in it["wpos"] and v != it["correct"]] if a.rev_all else []
                    bdata.append((w, c, x))
            else:
                # random query batches with the same expected number of wrong views
                per = int(round(n_q / max(1, len(tr_items)) * a.bq))
                perm = np.random.permutation(n_q)
                bdata = []
                # question index for each flat query, so the reverse loss can still be computed
                qidx = np.concatenate([[k] * len(it["wrong"]) for k, it in enumerate(tr_items)])
                for s in range(0, n_q, per):
                    idx = perm[s:s + per]
                    w = [(tr_q[i], int(tr_y[i]), int(qidx[i])) for i in idx]
                    ks = sorted(set(int(qidx[i]) for i in idx))
                    c = [(tr_items[k]["correct"], k) for k in ks]
                    x = [(v, k) for k in ks for j, v in enumerate(tr_items[k]["views"])
                         if j not in tr_items[k]["wpos"] and v != tr_items[k]["correct"]] if a.rev_all else []
                    bdata.append((w, c, x))
            for w, c, x in bdata:
                texts = [t for t, _, _ in w]
                gold = torch.tensor([m for _, m, _ in w], device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    qb = tok(texts, padding=True, truncation=True, max_length=a.qlen, return_tensors="pt").to(dev)
                    qe = model(qb)
                    if a.mode == "ce":
                        loss = F.cross_entropy(model.cls(qe).float(), gold)
                    else:
                        need_c = a.diff > 0 or (a.rev > 0 and a.rev_correct)
                        if need_c:
                            cb = tok([t for t, _ in c], padding=True, truncation=True, max_length=a.qlen, return_tensors="pt").to(dev)
                            ce_raw = model(cb).float()
                            cpos = {k: i for i, (_, k) in enumerate(c)}
                            ci = torch.tensor([cpos[k] for _, _, k in w], device=dev)
                        if a.diff > 0:
                            qe = qe.float() - a.diff * ce_raw[ci]
                        qe = F.normalize(qe.float(), dim=-1)
                        cand = set(gold.tolist())
                        if hard_pool is not None:
                            for t, m, _ in w:
                                cand.update(np.random.choice(hard_pool[t], a.hard, replace=False).tolist())
                        cand = sorted(cand)
                        pos = {m: i for i, m in enumerate(cand)}
                        mb = tok([mis[m] for m in cand], padding=True, truncation=True, max_length=a.mlen, return_tensors="pt").to(dev)
                        if a.freeze_mis:
                            with torch.no_grad():
                                me = F.normalize(mmodel(mb).float(), dim=-1)
                        else:
                            me = F.normalize(model(mb).float(), dim=-1)
                        tgt = torch.tensor([pos[m] for m in gold.tolist()], device=dev)
                        loss = F.cross_entropy(qe @ me.T / a.tau, tgt)
                        if a.distill > 0:
                            S = np.random.choice(pool, min(a.dk, len(pool)), replace=False)
                            sb = tok([mis[m] for m in S], padding=True, truncation=True, max_length=a.mlen, return_tensors="pt").to(dev)
                            se = F.normalize(model(sb).float(), dim=-1)
                            tq = torch.stack([t_q[t] for t in texts])
                            t_logit = tq @ t_me[torch.as_tensor(S, device=dev)].T / a.tau
                            s_logit = qe @ se.T / a.tau
                            loss = loss + a.distill * F.kl_div(F.log_softmax(s_logit, 1), F.log_softmax(t_logit, 1),
                                                               log_target=True, reduction="batchmean")
                        if a.rev > 0:
                            # reverse direction: gold misconception must pick its own distractor
                            ge = me[tgt]                                  # (n, d)
                            views = qe
                            if a.rev_correct:
                                ce_ = F.normalize(ce_raw, dim=-1)
                                views = torch.cat([qe, ce_])
                                if x:
                                    xb = tok([t for t, _ in x], padding=True, truncation=True, max_length=a.qlen, return_tensors="pt").to(dev)
                                    xr = model(xb).float()
                                    if a.diff > 0:
                                        xr = xr - a.diff * ce_raw[torch.tensor([cpos[k] for _, k in x], device=dev)]
                                    views = torch.cat([views, F.normalize(xr, dim=-1)])
                            lg = ge @ views.T / a.tau                     # (n, n + nc)
                            same = (gold[:, None] == gold[None, :])
                            if a.rev_correct:
                                same = torch.cat([same, torch.zeros(len(w), views.shape[0] - len(w), dtype=torch.bool, device=dev)], 1)
                            if a.rev_local:
                                vk = [k for _, _, k in w] + ([k for _, k in c] + [k for _, k in x] if a.rev_correct else [])
                                vk = torch.tensor(vk, device=dev)
                                wk = torch.tensor([k for _, _, k in w], device=dev)
                                lg = lg.masked_fill(wk[:, None] != vk[None, :], -1e4)
                            lse_all = torch.logsumexp(lg, 1)
                            lse_pos = torch.logsumexp(lg.masked_fill(~same, -1e4), 1)
                            loss = loss + a.rev * (lse_all - lse_pos).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); step += 1
                if step % 50 == 0:
                    log.append((step, float(loss.detach())))
            print(f"ep {ep} step {step} loss {float(loss):.4f}", flush=True)
    train_time = time.time() - t0
    train_mem = torch.cuda.max_memory_allocated() / 2**30

    # evaluation
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    if a.mode == "ce":
        model.eval(); outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(te_q), 128):
                b = tok(te_q[i:i + 128], padding=True, truncation=True, max_length=a.qlen, return_tensors="pt").to(dev)
                outs.append(model.cls(model(b)).float())
        scores = torch.cat(outs).cpu().numpy()
    else:
        me = encode(mmodel, tok, mis, a.mlen)
        torch.cuda.synchronize(); t2 = time.time()
        if a.diff > 0:
            te_c = [it["correct"] for it in te_items]
            qraw = encode(model, tok, te_q, a.qlen, normalize=False)
            craw = encode(model, tok, te_c, a.qlen, normalize=False)
            cidx = torch.tensor([k for k, it in enumerate(te_items) for _ in it["wrong"]], device=dev)
            qe = F.normalize(qraw - a.diff * craw[cidx], dim=-1)
        else:
            qe = encode(model, tok, te_q, a.qlen)
        scores = (qe @ me.T).cpu().numpy()
        torch.cuda.synchronize()
        q_time = (time.time() - t2) / len(te_q) * 1000
        # answer-set scoring: all options of each test question (labels not used)
        allv = [v for it in te_items for v in it["views"]]
        vraw = encode(model, tok, allv, a.qlen, normalize=False)
        if a.diff > 0:
            corr_of_view = torch.repeat_interleave(craw, 4, dim=0)
            vraw = vraw - a.diff * corr_of_view
        ve = F.normalize(vraw, dim=-1)
        sv = (ve @ me.T).view(len(te_items), 4, -1) / a.tau          # (nq, 4, M)
        logp_ans = sv - torch.logsumexp(sv, 1, keepdim=True)          # log p(answer | m, question)
        qi = torch.tensor([k for k, it in enumerate(te_items) for _ in it["wrong"]], device=dev)
        ai = torch.tensor([p for it in te_items for p in it["wpos"]], device=dev)
        lp = logp_ans[qi, ai].cpu().numpy()                           # (n, M)
        del sv, logp_ans
    eval_time = time.time() - t1
    eval_mem = torch.cuda.max_memory_allocated() / 2**30
    pq = Mx.per_query(scores, te_y)
    is_seen = np.array([y in seen for y in te_y])
    res = dict(args=vars(a), name=name, n_test=len(te_y), n_unseen=int((~is_seen).sum()),
               all=Mx.summarize(pq),
               seen=Mx.summarize({k: v[is_seen] for k, v in pq.items()}),
               unseen=Mx.summarize({k: v[~is_seen] for k, v in pq.items()}) if (~is_seen).any() else None,
               train_time_s=train_time, train_peak_gb=train_mem, eval_time_s=eval_time, eval_peak_gb=eval_mem,
               ms_per_query=q_time if a.mode != "ce" else None,
               n_params=sum(p.numel() for p in model.parameters()))
    # top-1 confidence for calibration: softmax over scores / tau
    s = torch.tensor(scores)
    temp = a.tau if a.mode != "ce" else 1.0
    prob = torch.softmax(s / temp, 1)
    conf, pred = prob.max(1)
    np.savez_compressed(os.path.join(odir, f"f{a.fold}_s{a.seed}.npz"), ids=np.array(te_ids), gold=te_y,
                        rank=pq["rank"], seen=is_seen, conf=conf.numpy(), pred=pred.numpy(),
                        top25=np.argsort(-scores, 1)[:, :25].astype(np.int16))
    with open(ofile, "w") as f:
        json.dump(res, f, indent=1)
    print(name, a.fold, a.seed, json.dumps(res["all"]), "unseen", json.dumps(res["unseen"]))
    if a.mode != "ce" and a.dump:
        edir = os.path.join(D.ROOT, "results", "emb", name)
        os.makedirs(edir, exist_ok=True)
        tre = encode(model, tok, tr_q, a.qlen) if a.diff == 0 else None
        np.savez_compressed(os.path.join(edir, f"f{a.fold}_s{a.seed}.npz"),
                            me=me.cpu().half().numpy(), qe=qe.cpu().half().numpy(), ids=np.array(te_ids), y=te_y,
                            seen=is_seen, tre=None if tre is None else tre.cpu().half().numpy(), try_=tr_y,
                            ve=ve.cpu().half().numpy(), wq=np.array([k for k, it in enumerate(te_items) for _ in it["wrong"]]),
                            wp=np.array([p for it in te_items for p in it["wpos"]]))
    if a.mode != "ce":
        for lam in (0.25, 0.5, 1.0, 2.0):
            sc2 = scores / a.tau + lam * lp
            pq2 = Mx.per_query(sc2, te_y)
            d2 = odir + f"_as{lam:g}"
            os.makedirs(d2, exist_ok=True)
            r2 = dict(res, name=name + f"_as{lam:g}", all=Mx.summarize(pq2),
                      seen=Mx.summarize({k: v[is_seen] for k, v in pq2.items()}),
                      unseen=Mx.summarize({k: v[~is_seen] for k, v in pq2.items()}) if (~is_seen).any() else None)
            p2 = torch.softmax(torch.tensor(sc2), 1)
            c2, pr2 = p2.max(1)
            np.savez_compressed(os.path.join(d2, f"f{a.fold}_s{a.seed}.npz"), ids=np.array(te_ids), gold=te_y,
                                rank=pq2["rank"], seen=is_seen, conf=c2.numpy(), pred=pr2.numpy(),
                                top25=np.argsort(-sc2, 1)[:, :25].astype(np.int16))
            with open(os.path.join(d2, f"f{a.fold}_s{a.seed}.json"), "w") as f:
                json.dump(r2, f, indent=1)
            print(" answer-set lam", lam, json.dumps(r2["all"]))
    if a.save:
        wdir = os.path.join(D.ROOT, "weights", name, f"f{a.fold}_s{a.seed}")
        model.m.save_pretrained(wdir); tok.save_pretrained(wdir)


if __name__ == "__main__":
    main()
