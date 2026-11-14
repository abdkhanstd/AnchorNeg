"""Retrieval metrics for single-relevant-item ranking."""
import numpy as np


def ranks_of_gold(scores, gold):
    """scores: (n, M) array, gold: (n,) ints -> 1-based rank of gold."""
    g = scores[np.arange(len(gold)), gold][:, None]
    return (scores > g).sum(1) + 1


def per_query(scores, gold):
    r = ranks_of_gold(scores, gold)
    return dict(
        rank=r,
        r1=(r <= 1).astype(float),
        r5=(r <= 5).astype(float),
        r25=(r <= 25).astype(float),
        mrr10=np.where(r <= 10, 1.0 / r, 0.0),
        # MAP@25 with one relevant item equals reciprocal rank truncated at 25
        map25=np.where(r <= 25, 1.0 / r, 0.0),
    )


METRICS = ["map25", "mrr10", "r1", "r5", "r25"]


def summarize(pq):
    return {k: float(np.mean(pq[k])) for k in METRICS}
