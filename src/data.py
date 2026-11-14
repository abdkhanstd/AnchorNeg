"""Eedi data loading, construct-grouped folds and text templates."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_raw", "eedi")
N_FOLDS = 5


def load():
    df = pd.read_csv(os.path.join(RAW, "train.csv"))
    mis = pd.read_csv(os.path.join(RAW, "misconception_mapping.csv"))
    mis = mis.sort_values("MisconceptionId").reset_index(drop=True)
    assert (mis.MisconceptionId.values == np.arange(len(mis))).all()
    return df, mis.MisconceptionName.tolist()


def construct_folds(df, n_folds=N_FOLDS, seed=0):
    """Assign each ConstructId to one fold so that no construct (and so no
    templated sibling question) appears in both training and test data."""
    rng = np.random.RandomState(seed)
    cons = df.ConstructId.unique()
    rng.shuffle(cons)
    # greedy balancing on number of labelled distractors
    cnt = df.groupby("ConstructId").apply(
        lambda g: sum(g[f"Misconception{c}Id"].notna().sum() for c in "ABCD"), include_groups=False)
    load_ = np.zeros(n_folds)
    fold_of = {}
    for c in sorted(cons, key=lambda c: -cnt[c]):
        f = int(np.argmin(load_))
        fold_of[c] = f
        load_[f] += cnt[c]
    return df.ConstructId.map(fold_of).values


def query_text(r, ans_letter, with_correct=True):
    """Text of the student-response view for answer `ans_letter`."""
    s = (f"Subject: {r.SubjectName}. Construct: {r.ConstructName}.\n"
         f"Question: {r.QuestionText}\n")
    if with_correct:
        s += f"Correct answer: {r[f'Answer{r.CorrectAnswer}Text']}\n"
    s += f"Student answer: {r[f'Answer{ans_letter}Text']}"
    return s


def build_items(df, with_correct=True):
    """One item per question: wrong views with labels plus the correct view."""
    items = []
    for _, r in df.iterrows():
        wrong = []
        for c in "ABCD":
            if c == r.CorrectAnswer:
                continue
            mid = r[f"Misconception{c}Id"]
            if pd.isna(mid):
                continue
            wrong.append((query_text(r, c, with_correct), int(mid), f"{r.QuestionId}_{c}"))
        if not wrong:
            continue
        views = [query_text(r, c, with_correct) for c in "ABCD"]
        wpos = [ "ABCD".index(i.split("_")[1]) for _, _, i in wrong]
        items.append(dict(qid=int(r.QuestionId), construct=int(r.ConstructId),
                          wrong=wrong, correct=query_text(r, r.CorrectAnswer, with_correct),
                          views=views, wpos=wpos))
    return items


def flat_queries(items):
    q, y, ids = [], [], []
    for it in items:
        for t, m, i in it["wrong"]:
            q.append(t); y.append(m); ids.append(i)
    return q, np.array(y), ids
