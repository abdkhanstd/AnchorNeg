"""Generalized zero-shot intent retrieval with the same bi-encoder recipe as train.py.

Same encoder, loss, temperature, optimiser and mining procedure; only the data differ.
Pools: all (HN-All), seen (AnchorNeg), random (size-matched control); --hard 0 = in-batch.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import intent_data as I
from train import Encoder, encode, mpath, QPREFIX, seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="banking77")
    p.add_argument("--model", default="bge")
    p.add_argument("--pool", default="all", choices=["all", "seen", "random", "zeroshot"])
    p.add_argument("--hard", type=int, default=4)
    p.add_argument("--P", type=int, default=20)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--tau", type=float, default=0.05)
    a = p.parse_args()
    seed_all(a.seed)
    dev = "cuda"
    name = f"{a.data}_{a.model}_" + ("zeroshot" if a.pool == "zeroshot" else f"{a.pool}_h{a.hard}")
    odir = os.path.join(I.ROOT, "results", "intent", name)
    os.makedirs(odir, exist_ok=True)
    ofile = os.path.join(odir, f"f{a.fold}_s{a.seed}.json")
    if os.path.exists(ofile):
        print("exists", ofile); return

    trx, try_, tex, tey, labels = I.load(a.data)
    keep = [i for i, l in enumerate(labels) if l != "oos"]
    remap = {old: new for new, old in enumerate(keep)}
    labels = [labels[i] for i in keep]
    try_ = np.array([remap[y] for y in try_]); tey = np.array([remap[y] for y in tey])
    M = len(labels)
    groups = I.unseen_groups(labels)
    unseen = set(groups[a.fold].tolist())
    m = np.array([y not in unseen for y in try_])
    trx = [t for t, k in zip(trx, m) if k]; try_ = try_[m]
    seen_set = np.unique(try_)
    pre = QPREFIX.get(a.model, "")
    trq = [pre + t for t in trx]; teq = [pre + t for t in tex]

    tok = AutoTokenizer.from_pretrained(mpath(a.model))
    model = Encoder(mpath(a.model)).to(dev)
    t0 = time.time()
    if a.pool != "zeroshot":
        pool = None
        if a.hard > 0:
            me = encode(model, tok, labels, 32)
            qe = encode(model, tok, trq, 64)
            sc = qe @ me.T
            sc[torch.arange(len(try_)), torch.as_tensor(try_)] = -1e4
            if a.pool in ("seen", "random"):
                allowed = seen_set if a.pool == "seen" else np.random.RandomState(1000 + 10 * a.seed + a.fold).choice(M, len(seen_set), replace=False)
                bad = torch.ones(M, dtype=torch.bool, device=sc.device); bad[torch.as_tensor(allowed, device=sc.device)] = False
                sc[:, bad] = -1e4
            pool = sc.topk(a.P, dim=1).indices.cpu().numpy()
            del me, qe, sc
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = a.epochs * (len(trq) // a.bs)
        warm = max(1, int(0.1 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
        scaler = torch.amp.GradScaler()
        for ep in range(a.epochs):
            model.train()
            perm = np.random.permutation(len(trq))
            for st in range(0, len(perm) - a.bs + 1, a.bs):
                idx = perm[st:st + a.bs]
                gold = try_[idx]
                cand = set(gold.tolist())
                if pool is not None:
                    for i in idx:
                        valid = pool[i][pool[i] >= 0]
                        cand.update(np.random.choice(pool[i], a.hard, replace=False).tolist())
                cand = sorted(cand); pos = {c: j for j, c in enumerate(cand)}
                with torch.autocast("cuda", dtype=torch.float16):
                    qb = tok([trq[i] for i in idx], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
                    lb = tok([labels[c] for c in cand], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
                    q = F.normalize(model(qb).float(), dim=-1); l = F.normalize(model(lb).float(), dim=-1)
                    loss = F.cross_entropy(q @ l.T / a.tau, torch.tensor([pos[g] for g in gold], device=dev))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
            print("ep", ep, float(loss.detach()), flush=True)
    train_time = time.time() - t0
    me = encode(model, tok, labels, 32); qe = encode(model, tok, teq, 64)
    sc = (qe @ me.T).cpu().numpy()
    g = sc[np.arange(len(tey)), tey][:, None]
    rank = (sc > g).sum(1) + 1
    is_seen = np.array([y not in unseen for y in tey])
    top1_seen = np.array([sc_i.argmax() not in unseen for sc_i in sc])
    acc = (rank == 1).astype(float); rr = 1.0 / rank
    accS, accU = acc[is_seen].mean(), acc[~is_seen].mean()
    res = dict(args=vars(a), name=name, acc=acc.mean(), mrr=rr.mean(), r5=(rank <= 5).mean(), acc_seen=accS, acc_unseen=accU,
               hmean=2 * accS * accU / (accS + accU + 1e-12), seen_share=top1_seen.mean(), base_rate=is_seen.mean(),
               n_test=len(tey), n_unseen_labels=len(unseen), train_time_s=train_time)
    np.savez_compressed(ofile[:-5] + ".npz", rank=rank, seen=is_seen, top1_seen=top1_seen, y=tey)
    json.dump(res, open(ofile, "w"), indent=1)
    print(name, a.fold, a.seed, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
