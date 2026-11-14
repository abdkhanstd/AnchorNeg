"""Error analysis: seen/unseen MAP@25 and sibling confusion (top-1 equals the
misconception of another distractor of the same question)."""
import glob, os, sys
import numpy as np
import data as D

RUNS = os.path.join(D.ROOT, "results", "runs")


def sibling_stats(name, seed=0):
    ids, gold, pred, rank, seen, top = [], [], [], [], [], []
    for f in range(D.N_FOLDS):
        p = os.path.join(RUNS, name, f"f{f}_s{seed}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        ids += list(z["ids"]); gold += list(z["gold"]); rank += list(z["rank"]); seen += list(z["seen"])
        pred += list(z["pred"]) if "pred" in z else [-1] * len(z["gold"])
        top += list(z["top25"]) if "top25" in z else [np.full(25, -1)] * len(z["gold"])
    gold, pred, rank, seen = map(np.array, (gold, pred, rank, seen))
    qid = np.array([i.split("_")[0] for i in ids])
    sib = {}
    for q, g in zip(qid, gold):
        sib.setdefault(q, set()).add(int(g))
    conf, conf5 = [], []
    for q, g, p, t in zip(qid, gold, pred, top):
        others = sib[q] - {int(g)}
        conf.append(p in others)
        # sibling misconception ranked above gold within the top 25
        tl = list(t)
        gr = tl.index(g) if g in tl else 25
        conf5.append(any(o in tl[:gr] for o in others))
    has_sib = np.array([len(sib[q]) > 1 for q in qid])
    mp = np.where(rank <= 25, 1 / rank, 0)
    return dict(map_seen=mp[seen].mean(), map_unseen=mp[~seen].mean(),
                sib_top1=np.mean(np.array(conf)[has_sib]), sib_above=np.mean(np.array(conf5)[has_sib]))


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(os.listdir(RUNS))
    for n in names:
        s = sibling_stats(n)
        if s:
            print(f"{n:42s} " + " ".join(f"{k}={v:.4f}" for k, v in s.items()))
