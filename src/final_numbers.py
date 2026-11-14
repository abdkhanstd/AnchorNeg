"""Numbers for the paper tables: post-hoc baselines per seed (cross-fold selection),
top-1 seen share, and summary rows."""
import json, os, sys
import numpy as np
import data as D
import offline as O
import report as R
from make_figs import seeds_of

OUT = os.path.join(D.ROOT, "results", "summary")


def posthoc(name, kind, grid, zs=None):
    rows = []
    for s in seeds_of(name):
        r = O.crossfold(name, kind, grid, zs, seed=s)
        rows.append([r["map"], r["seen"], r["unseen"]])
    return np.mean(rows, 0), np.std(rows, 0, ddof=1) if len(rows) > 1 else np.zeros(3)


def seen_share(name, kind="base", grid=None, zs=None):
    vals = []
    for s in seeds_of(name):
        hs = O.crossfold(name, kind, grid, zs, seed=s)["chosen"] if grid else [0] * D.N_FOLDS
        top, base = [], []
        for f in range(D.N_FOLDS):
            z = O.load(name, f, s)
            zz = O.load(zs, f, 0) if zs else None
            sc = O.variant(z, kind, hs[f], zz)
            top.append(O.seen_mask(z, sc.shape[1])[sc.argmax(1)]); base.append(z["seen"])
        vals.append((np.concatenate(top).mean(), np.concatenate(base).mean()))
    return np.mean(vals, 0)


if __name__ == "__main__":
    ZS = "bge_zeroshot_h4_g0_r0_rc1"
    B = "bge_mnrl_h4_g0_r0_rc1"
    res = {}
    for n in [B, "bge_mnrl_h4_g0_r0_rc1_npseen"]:
        res[n + "+cal"] = posthoc(n, "penalty", [0, .01, .02, .03, .04, .05, .07, .1])
        res[n + "+ens"] = posthoc(n, "zs_ens", [0, .1, .2, .3, .4, .5], ZS)
        res[n + "+ans"] = posthoc(n, "answerset", [0, .25, .5, 1, 2])
    res[ZS + "+ans"] = posthoc(ZS, "answerset", [0, .25, .5, 1, 2])
    for k, (m, sd) in res.items():
        print(f"{k:45s} map={m[0]:.4f}+-{sd[0]:.4f} seen={m[1]:.4f} unseen={m[2]:.4f} H={2*m[1]*m[2]/(m[1]+m[2]):.4f}")
    for n in sys.argv[1:]:
        print("seen-share", n, seen_share(n))
    print("seen-share cal", seen_share(B, "penalty", [0, .01, .02, .03, .04, .05, .07, .1]))
    print("seen-share ens", seen_share(B, "zs_ens", [0, .1, .2, .3, .4, .5], ZS))
