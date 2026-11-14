"""Summaries of the intent-retrieval runs: mean over the 5 unseen-intent folds and seeds."""
import glob, json, os
import numpy as np
import intent_data as I

DIR = os.path.join(I.ROOT, "results", "intent")
KEYS = ["acc", "mrr", "acc_seen", "acc_unseen", "hmean", "seen_share", "base_rate"]


def load(name):
    out = {}
    for f in glob.glob(os.path.join(DIR, name, "f*_s*.json")):
        j = json.load(open(f))
        out[(j["args"]["fold"], j["args"]["seed"])] = j
    return out


if __name__ == "__main__":
    for d in ["banking77", "clinc150"]:
        print("==", d)
        res = {}
        for cfg in ["zeroshot", "all_h0", "all_h4", "random_h4", "seen_h4"]:
            r = load(f"{d}_bge_{cfg}")
            if not r:
                continue
            res[cfg] = r
            seeds = sorted({s for _, s in r})
            per_seed = [[np.mean([r[(f, s)][k] for f in range(I.N_FOLDS) if (f, s) in r]) for k in KEYS] for s in seeds]
            m = np.mean(per_seed, 0); sd = np.std(per_seed, 0, ddof=1) if len(per_seed) > 1 else np.zeros(len(KEYS))
            print(f"{cfg:10s} n={len(r):2d} " + " ".join(f"{k}={m[i]:.3f}+-{sd[i]:.3f}" for i, k in enumerate(KEYS)))
        if "seen_h4" in res and "all_h4" in res:
            common = sorted(set(res["seen_h4"]) & set(res["all_h4"]))
            for k in ["acc", "acc_unseen", "hmean"]:
                dd = [res["seen_h4"][c][k] - res["all_h4"][c][k] for c in common]
                print(f"AnchorNeg - HN-All {k}: mean {np.mean(dd):.3f}, min {np.min(dd):.3f}, positive {np.mean(np.array(dd) > 0):.2f} of {len(dd)} fold-seed pairs")
