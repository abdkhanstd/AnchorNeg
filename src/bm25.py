"""BM25 lexical baseline on the same construct-grouped folds."""
import json, os, re
import numpy as np
from rank_bm25 import BM25Okapi
import data as D
import metrics as Mx


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def main():
    df, mis = D.load()
    folds = D.construct_folds(df)
    bm = BM25Okapi([toks(m) for m in mis])
    odir = os.path.join(D.ROOT, "results", "runs", "bm25")
    os.makedirs(odir, exist_ok=True)
    for f in range(D.N_FOLDS):
        tr = D.build_items(df[folds != f]); te = D.build_items(df[folds == f])
        _, tr_y, _ = D.flat_queries(tr)
        q, y, ids = D.flat_queries(te)
        sc = np.stack([bm.get_scores(toks(t)) for t in q])
        pq = Mx.per_query(sc, y)
        seen = np.isin(y, tr_y)
        res = dict(name="bm25", n_test=len(y), all=Mx.summarize(pq),
                   seen=Mx.summarize({k: v[seen] for k, v in pq.items()}),
                   unseen=Mx.summarize({k: v[~seen] for k, v in pq.items()}))
        json.dump(res, open(os.path.join(odir, f"f{f}_s0.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(odir, f"f{f}_s0.npz"), ids=np.array(ids), gold=y, rank=pq["rank"], seen=seen)
        print(f, res["all"])


if __name__ == "__main__":
    main()
