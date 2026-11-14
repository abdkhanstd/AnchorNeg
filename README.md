# AnchorNeg: seen-anchored hard negatives for misconception retrieval

Code, configurations and per-run results for retrieving the misconception behind a student's wrong answer on the Eedi *Mining Misconceptions in Mathematics* data, under a construct-disjoint five-fold protocol in which many test misconceptions never occur as training positives.

**AnchorNeg** mines hard negatives only from labels that are positives somewhere in the training fold. With the standard recipe (negatives mined from the whole inventory), labels that never have a positive are only ever pushed down, and the fine-tuned retriever learns to prefer labels it has seen. **AnchorNeg+KD** adds a consistency term that keeps each response's score distribution over a random sample of the whole inventory close to the frozen encoder.

## Results (BGE-base-en-v1.5, 5 folds pooled = 4,370 responses, mean of 3 seeds)

| Method | MAP@25 | R@1 | R@25 | MAP@25 seen | MAP@25 unseen | top-1 seen share |
|---|---|---|---|---|---|---|
| Zero-shot | 0.182 | 0.099 | 0.540 | 0.137 | 0.239 | 0.526 |
| In-batch contrastive (k=0) | 0.277 | 0.144 | 0.756 | 0.247 | 0.315 | 0.497 |
| Hard negatives from full inventory (HN-All) | 0.302 | 0.175 | 0.734 | 0.347 | 0.245 | 0.761 |
| HN-All, random pool of the same size as the seen set | 0.290 | 0.162 | 0.732 | 0.320 | 0.253 | 0.722 |
| **AnchorNeg** | 0.311 | 0.175 | 0.761 | 0.314 | **0.308** | 0.571 |
| **AnchorNeg+KD** | **0.326** | **0.192** | **0.767** | 0.358 | 0.286 | 0.694 |

The base rate of seen gold labels is 0.557.

Generalized zero-shot intent retrieval (BGE-base, 5 unseen-intent folds x 3 seeds, top-1 accuracy overall / on unseen intents):

| Method | BANKING77 | CLINC150 |
|---|---|---|
| Zero-shot | 0.696 / 0.698 | 0.736 / 0.736 |
| In-batch (k=0) | 0.885 / 0.711 | 0.920 / 0.756 |
| HN-All | 0.823 / 0.351 | 0.871 / 0.472 |
| Random pool (size-matched) | 0.836 / 0.428 | 0.881 / 0.526 |
| AnchorNeg | 0.886 / 0.707 | 0.923 / 0.756 |

With MPNet as encoder, AnchorNeg reaches 0.886 / 0.678 and 0.932 / 0.778 (HN-All 0.833 / 0.375 and 0.892 / 0.552).

Dynamic mining (`--remine 1`, negatives re-mined from the current encoder every epoch), BGE-base, 3 seeds: HN-All 0.319 MAP@25 (unseen 0.269, seen share 0.742), AnchorNeg 0.316 (unseen 0.318, seen share 0.547), AnchorNeg+KD 0.324 (unseen 0.289). All runs, including BERT, DeBERTa-v3, MPNet and BGE-small backbones, the dose-response sweep over the number of negatives and the ablations, are in `results/runs/`.

## Layout

```
src/
  data.py          data loading, construct-grouped folds, input templates
  train.py         training and evaluation (all methods; see flags below)
  jobq.py          runs a list of train.py jobs, one per GPU
  bm25.py          BM25 baseline
  metrics.py       MAP@25, MRR@10, Recall@k
  aggregate.py     loads per-run outputs, ECE
  report.py        summary table, paired bootstrap and Wilcoxon tests
  offline.py       post-hoc variants on saved embeddings (calibrated stacking,
                   zero-shot ensemble, answer-set rescoring), cross-fold selection
  analysis.py      sibling-distractor confusion statistics
  final_numbers.py post-hoc rows and top-1 seen share
  extra_numbers.py construct-cluster bootstrap, per-seed and per-fold signs
  intent_data.py   BANKING77 / CLINC150 loaders and unseen-intent folds
  train_intent.py  the same recipe on generalized zero-shot intent retrieval
  intent_report.py summary of the intent runs
configs/
  intent_runs.txt  commands for the intent experiments (SCRIPT=train_intent.py)
  paper_runs.txt   every training/evaluation command behind the reported numbers
  history/         job lists of the exploratory runs
results/
  runs/            one folder per configuration: f<fold>_s<seed>.json (metrics, cost)
                   and .npz (per-response rank, gold id, seen flag, confidence, top-25)
  runs_v1/         answer-contrastive runs made with an earlier version of train.py
  runs_release/    the runs that produced the released checkpoints
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

Intent data: download the parquet files of `PolyAI/banking77` (default config) and `clinc/clinc_oos` (`plus` config) from the Hugging Face Hub as `data_raw/intent/{banking77,clinc}_{train,test}.parquet`, together with the two dataset cards as `banking77_README.md` and `clinc_README.md` (the intent names are read from them).

Eedi data: download `train.csv` and `misconception_mapping.csv` from the Kaggle competition *Eedi - Mining Misconceptions in Mathematics* and place them in `data_raw/eedi/`. The files we used have SHA-256

```
31fbea4fdb250fc67a8254aba0764100e89db71a43ea74d2f123ef1c95545187  train.csv
2b2fd9c031084b11fe486435c823452e5ef1a8edc1fa47ab7fa676ffe141a507  misconception_mapping.csv
```

Backbones are loaded from the Hugging Face Hub, or from `hf_models/<key>` if that folder exists (keys: `bge`, `bge-small`, `mpnet`, `bert`, `deberta`).

## Running

```bash
cd src
# HN-All baseline, fold 0
python train.py --model bge --mode mnrl --hard 4 --fold 0
# AnchorNeg
python train.py --model bge --mode mnrl --hard 4 --neg_pool seen --fold 0
# AnchorNeg+KD
python train.py --model bge --mode mnrl --hard 4 --neg_pool seen --distill 1 --dpool all --fold 0
# everything
python jobq.py ../configs/paper_runs.txt
# summaries and tests
python report.py
python final_numbers.py bge_mnrl_h4_g0_r0_rc1 bge_mnrl_h4_g0_r0_rc1_npseen
python extra_numbers.py
```

Key flags of `train.py`: `--neg_pool {all,seen,random}` (mining pool), `--hard k` (negatives per response, pool of the top 50 under the frozen encoder), `--distill λ --dpool all` (consistency term over 128 sampled labels), `--no_correct` (drop the correct answer from the input), `--mode ours --rev w --rev_all 1 --rev_local 1` (answer-contrastive reverse loss), `--remine 1` (dynamic mining), `--save` (write the checkpoint to `weights/`). Each run writes embeddings to `results/emb/` (not included, about 2 GB) that `offline.py` uses.

## Checkpoints

Fine-tuned BGE-base checkpoints for every fold (seed 0) of AnchorNeg and AnchorNeg+KD are attached to the GitHub release as fp16 safetensors. Fold *f* was trained on the other four folds, so each checkpoint should only be evaluated on its own fold. Pooled scores of the released checkpoints: AnchorNeg 0.311 MAP@25, AnchorNeg+KD 0.327 MAP@25.

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("anchorneg_bge-base_fold0")   # unzipped release asset
tok = AutoTokenizer.from_pretrained("anchorneg_bge-base_fold0")
```

Responses are encoded with the prefix `Represent this sentence for searching relevant passages: `, misconceptions without prefix, mean pooling and cosine similarity.

## Licence

MIT. The Eedi data are not redistributed here and remain under the competition's terms.
