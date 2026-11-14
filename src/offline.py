"""Offline scoring variants on dumped embeddings.

For every test fold, a hyperparameter is chosen on the other folds only
(cross-fold selection), so the test fold's labels are never used for tuning.
"""
import os, sys, json
import numpy as np
import data as D

EMB = os.path.join(D.ROOT, "results", "emb")
TAU = 0.05


def load(name, fold, seed=0):
    p = os.path.join(EMB, name, f"f{fold}_s{seed}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def map25(scores, y):
    g = scores[np.arange(len(y)), y][:, None]
    r = (scores > g).sum(1) + 1
    return np.where(r <= 25, 1.0 / r, 0.0)


def base_scores(z):
    return z["qe"].astype(np.float32) @ z["me"].astype(np.float32).T


def seen_mask(z, M):
    m = np.zeros(M, bool)
    m[np.unique(z["try_"])] = True
    return m


def variant(z, kind, h, zs=None):
    s = base_scores(z)
    M = s.shape[1]
    if kind == "base":
        return s
    if kind == "penalty":            # calibrated stacking: lower the score of labels seen in training
        return s - h * seen_mask(z, M)[None, :]
    if kind == "zs_ens":             # score ensemble with the frozen encoder
        return (1 - h) * s + h * base_scores(zs)
    if kind == "answerset":          # add lambda * log p(answer | m, question) over all options of the question
        ve = z["ve"].astype(np.float32) @ z["me"].astype(np.float32).T / TAU   # (4 * nq, M)
        nq = ve.shape[0] // 4
        ve = ve.reshape(nq, 4, -1)
        lse = np.logaddexp.reduce(ve, axis=1)
        lp = ve[z["wq"], z["wp"]] - lse[z["wq"]]
        return s / TAU + h * lp
    raise ValueError(kind)


def crossfold(name, kind, grid, zs_name=None, seed=0):
    folds = [load(name, f, seed) for f in range(D.N_FOLDS)]
    zss = [load(zs_name, f, 0) for f in range(D.N_FOLDS)] if zs_name else [None] * D.N_FOLDS
    per = {h: [map25(variant(z, kind, h, zz), z["y"]) for z, zz in zip(folds, zss)] for h in grid}
    out, chosen = [], []
    for f in range(D.N_FOLDS):
        best = max(grid, key=lambda h: np.mean(np.concatenate([per[h][g] for g in range(D.N_FOLDS) if g != f])))
        chosen.append(best)
        out.append(per[best][f])
    allq = np.concatenate(out)
    seen = np.concatenate([z["seen"] for z in folds])
    return dict(map=allq.mean(), seen=allq[seen].mean(), unseen=allq[~seen].mean(), chosen=chosen)


if __name__ == "__main__":
    name = sys.argv[1]
    zs = "bge_zeroshot_h4_g0_r0_rc1"
    print(name, "base", crossfold(name, "base", [0]))
    print(name, "penalty", crossfold(name, "penalty", [0, .01, .02, .03, .04, .05, .07, .1]))
    print(name, "zs_ens", crossfold(name, "zs_ens", [0, .1, .2, .3, .4, .5], zs))
    print(name, "answerset", crossfold(name, "answerset", [0, .25, .5, 1, 2]))
