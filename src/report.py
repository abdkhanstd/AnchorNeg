"""Collect all numbers for the paper: per-config means over seeds (5 folds pooled),
seen/unseen MAP@25, harmonic mean, seen-bias, calibration, cost, paired tests."""
import glob, json, os, sys
import numpy as np
from scipy import stats

import data as D
import aggregate as A

RUNS = os.path.join(D.ROOT, "results", "runs")
OUT = os.path.join(D.ROOT, "results", "summary")


def cost(name):
    js = [json.load(open(f)) for f in glob.glob(os.path.join(RUNS, name, "f*_s*.json"))]
    if not js or "train_peak_gb" not in js[0]:
        return {}
    g = lambda k: float(np.mean([j[k] for j in js if j.get(k) is not None])) if any(j.get(k) is not None for j in js) else None
    return dict(train_s=g("train_time_s"), train_gb=float(np.max([j["train_peak_gb"] for j in js])),
                ms_q=g("ms_per_query"), params_m=js[0]["n_params"] / 1e6)


def full(name):
    runs = A.load_run(name)
    if not runs:
        return None
    rows = []
    for s, r in sorted(runs.items()):
        m = A.pq_metrics(r["rank"])
        se = r["seen"]
        d = {k: float(v.mean()) for k, v in m.items()}
        d["map_seen"] = float(m["map25"][se].mean()); d["map_unseen"] = float(m["map25"][~se].mean())
        d["hmean"] = 2 * d["map_seen"] * d["map_unseen"] / (d["map_seen"] + d["map_unseen"])
        if "conf" in r:
            corr = (r["rank"] == 1).astype(float)
            d["ece"] = A.ece(r["conf"], corr); d["brier"] = float(np.mean((r["conf"] - corr) ** 2))
        rows.append(d)
    keys = rows[0].keys()
    res = {k: (float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows], ddof=1)) if len(rows) > 1 else 0.0) for k in keys}
    res["n_seeds"] = len(rows)
    res.update(cost(name))
    return res


def paired(a, b, metric="map25", subset=None):
    ra, rb = A.load_run(a), A.load_run(b)
    common = sorted(set(ra) & set(rb)) or None
    sa = common or sorted(ra); sb = common or sorted(rb)
    def vec(r, ss):
        x = np.mean([A.pq_metrics(r[s]["rank"])[metric] for s in ss], 0)
        return x
    xa, xb = vec(ra, sa), vec(rb, sb)
    assert (ra[sa[0]]["ids"] == rb[sb[0]]["ids"]).all()
    if subset is not None:
        msk = ra[sa[0]]["seen"] if subset == "seen" else ~ra[sa[0]]["seen"]
        xa, xb = xa[msk], xb[msk]
    d = xa - xb
    rng = np.random.RandomState(0)
    bs = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)])
    p = stats.wilcoxon(xa, xb, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
    return dict(diff=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p), n=len(d))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    allres = {}
    for n in sorted(os.listdir(RUNS)):
        r = full(n)
        if r:
            allres[n] = r
            print(f"{n:48s} s={r['n_seeds']} map={r['map25'][0]:.4f}±{r['map25'][1]:.4f} seen={r['map_seen'][0]:.4f} unseen={r['map_unseen'][0]:.4f} H={r['hmean'][0]:.4f} r1={r['r1'][0]:.4f} r5={r['r5'][0]:.4f} r25={r['r25'][0]:.4f}")
    json.dump(allres, open(os.path.join(OUT, "all.json"), "w"), indent=1)
