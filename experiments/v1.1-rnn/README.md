# v1.1-rnn: the RNN control with the v1.1 change

> One change from v1.1: the RNN batch size goes from 64 to 16, the same as the v1.1 RSNN. Only the RNN is trained.

## What was done

In v1.1 only the RSNN got batch size 16; the RNN was copied from v1 (batch 64). The RNN was then no longer the RSNN's
non-spiking twin, so RSNN against RNN in v1.1 compared spiking and batch size at the same time. This run trains the RNN
with batch 16, so the v1.1 pair differs only in the spikes again. Batch 16 for both recurrent models is also what the
group repository's main branch (8549d78) uses.

| Setting | Value |
|---|---|
| RNN batch size | 16 (v1 and v1.1: 64) |
| RNN size | 17 ELU units, 752 parameters (as v1) |
| Everything else | As v1.1 |

| Step | Detail |
|---|---|
| Run | Run id `20261106_144857`, 9 target subjects × 10 repeats = 90 folds, the same folds and seeds as v1 and v1.1 (fold seed = 2026 + 100 × target + repeat) |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Two runs shared the GPU at the same time, so the times per fold are longer than for a run alone. Not run on ARC |
| Time | 35.7 min per fold on average, about 18.0 h for the run |
| Models trained | RNN only (`MODELS = ('rnn',)` in `benchmark.py`). EEGNet, the converted SNN and the RSNN of v1.1 are in `../v1/` and `../v1.1/` |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| RNN control, batch 16 (this run) | 0.437 ± 0.125 | 0.249 | 0.406 | 0.812 | 7/9 |
| RSNN, batch 16 (v1.1) | 0.386 ± 0.105 | 0.181 | 0.341 | 0.795 | 7/9 |
| RNN control, batch 64 (v1) | 0.411 ± 0.117 | 0.215 | 0.372 | 0.804 | 7/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

Session 1 (target's 20 % fine-tuning validation split, 9 development folds, saved checkpoints scored by
`../summary/score_saved.py`): RNN 0.450, v1.1 RSNN 0.421.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set has now been scored in many versions, so any choice between versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this run |
| `code/launch_shards.sh`, `code/shard.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, each under a sleep lock. They expect the pipeline layout `rssns_for_mi_EEG/src` next to them and a `.venv` two folders up |
| `results/20261106_144857/config.json` | Every setting of the run, including the parameter counts |
| `results/20261106_144857/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261106_144857/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261106_144857/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261106_144857/predictions/<model>/s<S>_r<R>.npz` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261106_144857/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261106_144857/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261106_144857/checkpoints/` | Trained weights (PyTorch `.pt`), pretrained and fine-tuned |
| `results/20261106_144857/comparison/` | Pipeline statistics: chance tests, per-subject summary, and paired tests and bootstrap intervals where a run has two models or more (this run has one model, so `paired_tests.csv` is empty; the comparison with the other models is in `../summary/`) |
| `results/20261106_144857/figures/` | Pipeline figures (accuracy per subject, confusion matrices, metric distributions) |
| `results/20261106_144857/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.1.diff` shows every line of `code/src/` that differs from `v1.1`, the version this run was built from.

Glossary and the overview of all versions: `../README.md`.
