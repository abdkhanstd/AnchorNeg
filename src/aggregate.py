"""Aggregate run results: mean over folds and seeds, bootstrap CIs, paired tests, ECE."""
import glob, json, os, sys
import numpy as np
from scipy import stats

import data as D

RUNS = os.path.join(D.ROOT, "results", "runs")


def load_run(name):
    """Concatenate per-query arrays over the 5 folds for each seed -> {seed: dict}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, name, "f*_s*.npz"))):
        b = os.path.basename(f)[:-4]
        fold, seed = int(b.split("_")[0][1:]), int(b.split("_")[1][1:])
        out.setdefault(seed, {})[fold] = dict(np.load(f))
    full = {}
    for s, fd in out.items():
        if len(fd) != D.N_FOLDS:
            continue
        cat = {k: np.concatenate([fd[i][k] for i in range(D.N_FOLDS)]) for k in fd[0] if k != "top25"}
        o = np.argsort(cat["ids"])  # align queries across runs
        full[s] = {k: v[o] for k, v in cat.items()}
    return full


def pq_metrics(rank):
    return dict(map25=np.where(rank <= 25, 1 / rank, 0), r1=(rank <= 1) * 1.0, r5=(rank <= 5) * 1.0,
                r25=(rank <= 25) * 1.0, mrr10=np.where(rank <= 10, 1 / rank, 0))


def ece(conf, correct, bins=15):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def summary(name, subset=None):
    runs = load_run(name)
    if not runs:
        return None
    per_seed = []
    for s, r in sorted(runs.items()):
        m = pq_metrics(r["rank"])
        msk = np.ones(len(r["rank"]), bool) if subset is None else (r["seen"] if subset == "seen" else ~r["seen"])
        d = {k: float(v[msk].mean()) for k, v in m.items()}
        if "conf" in r:
            d["ece"] = ece(r["conf"][msk], (r["rank"][msk] == 1).astype(float))
            d["brier"] = float(np.mean((r["conf"][msk] - (r["rank"][msk] == 1)) ** 2))
        per_seed.append(d)
    keys = per_seed[0].keys()
    return {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0)
            for k in keys} | {"n_seeds": len(per_seed)}


def paired(a, b, metric="map25", n_boot=10000, seed=0):
    """Paired comparison of run a vs run b on query-level metric averaged over common seeds."""
    ra, rb = load_run(a), load_run(b)
    sa = sorted(ra); sb = sorted(rb)
    xa = np.mean([pq_metrics(ra[s]["rank"])[metric] for s in sa], 0)
    xb = np.mean([pq_metrics(rb[s]["rank"])[metric] for s in sb], 0)
    assert (ra[sa[0]]["ids"] == rb[sb[0]]["ids"]).all()
    d = xa - xb
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)])
    w = stats.wilcoxon(xa, xb, zero_method="zsplit")
    return dict(diff=float(d.mean()), ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                p_wilcoxon=float(w.pvalue))


if __name__ == "__main__":
    for n in sorted(os.listdir(RUNS)):
        s = summary(n)
        if s:
            print(f"{n:45s} seeds={s['n_seeds']} " + " ".join(f"{k}={v[0]:.4f}±{v[1]:.4f}" for k, v in s.items() if k != "n_seeds"))
