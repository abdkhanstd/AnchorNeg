"""Additional statistics: post-hoc recall, construct-cluster bootstrap, per-seed and
per-fold signs, distinct unseen misconception shares, lambda sensitivity."""
import numpy as np
import data as D
import offline as O
import aggregate as A
import report as R
from make_figs import seeds_of

B = "bge_mnrl_h4_g0_r0_rc1"
AN = "bge_mnrl_h4_g0_r0_rc1_npseen"
KD = "bge_mnrl_h4_g0_r0_rc1_npseen_kd1all"
HKD = "bge_mnrl_h4_g0_r0_rc1_kd1all"
ZS = "bge_zeroshot_h4_g0_r0_rc1"
PEN = [0, .01, .02, .03, .04, .05, .07, .1]
ENS = [0, .1, .2, .3, .4, .5]


def ranks_of(sc, y):
    g = sc[np.arange(len(y)), y][:, None]
    return (sc > g).sum(1) + 1


def posthoc_recall(name, kind, grid, zs=None):
    out = []
    for s in seeds_of(name):
        hs = O.crossfold(name, kind, grid, zs, seed=s)["chosen"]
        rk, ss = [], []
        for f in range(D.N_FOLDS):
            z = O.load(name, f, s); zz = O.load(zs, f, 0) if zs else None
            sc = O.variant(z, kind, hs[f], zz)
            rk.append(ranks_of(sc, z["y"])); ss.append(O.seen_mask(z, sc.shape[1])[sc.argmax(1)])
        rk = np.concatenate(rk); ss = np.concatenate(ss)
        out.append([(rk <= 1).mean(), (rk <= 5).mean(), (rk <= 25).mean(), ss.mean()])
    return np.mean(out, 0)


def cluster_boot(a, b, n=5000):
    df, _ = D.load()
    cons = dict(zip(df.QuestionId.astype(str), df.ConstructId))
    ra, rb = A.load_run(a), A.load_run(b)
    xa = np.mean([A.pq_metrics(r["rank"])["map25"] for r in ra.values()], 0)
    xb = np.mean([A.pq_metrics(r["rank"])["map25"] for r in rb.values()], 0)
    ids = next(iter(ra.values()))["ids"]
    c = np.array([cons[i.split("_")[0]] for i in ids])
    uc = np.unique(c); idx = {u: np.where(c == u)[0] for u in uc}
    d = xa - xb
    rng = np.random.RandomState(0)
    bs = []
    for _ in range(n):
        pick = rng.choice(uc, len(uc), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        bs.append(d[ii].mean())
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def signs(a, b):
    ra, rb = A.load_run(a), A.load_run(b)
    per_seed = {s: A.pq_metrics(ra[s]["rank"])["map25"].mean() - A.pq_metrics(rb[s]["rank"])["map25"].mean() for s in sorted(set(ra) & set(rb))}
    per_fold = []
    for f in range(D.N_FOLDS):
        da = np.mean([np.load(f"{A.RUNS}/{a}/f{f}_s{s}.npz")["rank"] for s in per_seed], 0)
        ma = np.mean([np.where(np.load(f"{A.RUNS}/{a}/f{f}_s{s}.npz")["rank"] <= 25, 1 / np.load(f"{A.RUNS}/{a}/f{f}_s{s}.npz")["rank"], 0).mean() for s in per_seed])
        mb = np.mean([np.where(np.load(f"{A.RUNS}/{b}/f{f}_s{s}.npz")["rank"] <= 25, 1 / np.load(f"{A.RUNS}/{b}/f{f}_s{s}.npz")["rank"], 0).mean() for s in per_seed])
        per_fold.append(ma - mb)
    return per_seed, per_fold


if __name__ == "__main__":
    print("cal recall/SS", posthoc_recall(B, "penalty", PEN))
    print("ens recall/SS", posthoc_recall(B, "zs_ens", ENS, ZS))
    for a, b in [(AN, B), (KD, B), (KD, HKD), (AN, "bge_mnrl_h0_g0_r0_rc1")]:
        print("cluster", a[-12:], b[-12:], cluster_boot(a, b))
        print("signs", signs(a, b))
    df, _ = D.load(); folds = D.construct_folds(df)
    for f in range(D.N_FOLDS):
        _, ytr, _ = D.flat_queries(D.build_items(df[folds != f])); _, yte, _ = D.flat_queries(D.build_items(df[folds == f]))
        u = np.unique(yte)
        print("fold", f, "distinct test labels", len(u), "unseen share", round(np.mean(~np.isin(u, ytr)), 3))
    for n in ["bge_mnrl_h4_g0_r0_rc1_npseen_kd0.5all", HKD, KD]:
        r = R.full(n)
        if r:
            print(n, r["n_seeds"], {k: round(r[k][0], 4) for k in ["map25", "r1", "r25", "map_seen", "map_unseen", "ece", "brier"]}, r.get("train_s"))
    for n in [B, AN]:
        r = R.full(n); print(n, "brier", round(r["brier"][0], 4), "train_s", r["train_s"], "gb", r["train_gb"])
