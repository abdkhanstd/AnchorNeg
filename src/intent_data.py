"""Generalized zero-shot intent retrieval data (BANKING77, CLINC150).

Labels are the intent names written as text ("activate my card"). Intents are
split into 5 groups; in fold f the intents of group f are unseen: their training
utterances are removed. Test utterances of all intents (official test split)
are evaluated, so seen and unseen intents compete in one ranking.
"""
import os, re
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(ROOT, "data_raw", "intent")
N_FOLDS = 5


def _names(readme, config=None):
    txt = open(os.path.join(DIR, readme), encoding="utf-8").read()
    meta = yaml.safe_load(txt.split("---")[1])
    infos = meta["dataset_info"]
    if isinstance(infos, list):
        infos = [i for i in infos if i.get("config_name") == config][0]
    for feat in infos["features"]:
        if "class_label" in feat.get("dtype", {}) if isinstance(feat.get("dtype"), dict) else False:
            names = feat["dtype"]["class_label"]["names"]
            return [names[k] if isinstance(names, dict) else names[int(k)] for k in sorted(names, key=int)] if isinstance(names, dict) else names
    raise ValueError("no class names")


def load(name):
    if name == "banking77":
        names = _names("banking77_README.md")
        tr = pd.read_parquet(os.path.join(DIR, "banking77_train.parquet")).rename(columns={"label": "y"})
        te = pd.read_parquet(os.path.join(DIR, "banking77_test.parquet")).rename(columns={"label": "y"})
    elif name == "clinc150":
        names = _names("clinc_README.md", "plus")
        tr = pd.read_parquet(os.path.join(DIR, "clinc_train.parquet")).rename(columns={"intent": "y"})
        te = pd.read_parquet(os.path.join(DIR, "clinc_test.parquet")).rename(columns={"intent": "y"})
        oos = names.index("oos")
        tr, te = tr[tr.y != oos], te[te.y != oos]
    else:
        raise ValueError(name)
    labels = [re.sub(r"[_]+", " ", n).strip() for n in names]
    return tr.text.tolist(), tr.y.values, te.text.tolist(), te.y.values, labels


def unseen_groups(labels, seed=0):
    rng = np.random.RandomState(seed)
    ids = np.array([i for i, l in enumerate(labels) if l != "oos"])
    rng.shuffle(ids)
    return np.array_split(ids, N_FOLDS)


if __name__ == "__main__":
    for n in ["banking77", "clinc150"]:
        a, b, c, d, L = load(n)
        print(n, len(a), len(c), len(set(b)), len(L), L[:4])
