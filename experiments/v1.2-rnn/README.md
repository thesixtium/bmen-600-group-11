# v1.2-rnn: the RNN control with the v1.2 change

> One change from v1.2: the RNN gets the same readout on 4 time windows as the v1.2 RSNN, and its size (14 units). Only the RNN is trained.

## What was done

In v1.2 only the RSNN got the 4-window readout (14 neurons); the RNN was copied from v1 (one window, 17 units). This run
gives the RNN the same readout, so the v1.2 pair differs only in the spikes again. The readout sees the mean activity of
every unit in each of 4 windows of 64 steps (14 × 4 = 56 readout inputs).

| Setting | Value |
|---|---|
| RNN readout | 4 time windows of 64 steps (v1: mean over all 256 steps) |
| RNN size | 14 ELU units, 746 parameters, as the v1.2 RSNN (v1: 17 units, 752) |
| Everything else | As v1.2 |

`config.json` → `parameter_counts` still shows `rnn_hidden: 17`: that summary function was not changed for this run. The
trained RNN has 14 units and 746 parameters (see the checkpoints: `input_proj.weight` 14 × 22, `readout.weight` 4 × 56).

| Step | Detail |
|---|---|
| Run | Run id `20261107_093450`, 9 target subjects × 10 repeats = 90 folds, the same folds and seeds as v1 and v1.2 |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Two runs shared the GPU at the same time, so the times per fold are longer than for a run alone. Not run on ARC |
| Time | 10.5 min per fold on average, about 5.4 h for the run |
| Models trained | RNN only. EEGNet, the converted SNN and the RSNN of v1.2 are in `../v1/` and `../v1.2/` |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| RNN control, 4 windows (this run) | 0.422 ± 0.115 | 0.230 | 0.400 | 0.807 | 7/9 |
| RSNN, 4 windows (v1.2) | 0.371 ± 0.090 | 0.162 | 0.345 | 0.790 | 6/9 |
| RNN control, 1 window (v1) | 0.411 ± 0.117 | 0.215 | 0.372 | 0.804 | 7/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

Session 1 (20 % split, 9 development folds): RNN 0.473, v1.2 RSNN 0.406.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set has now been scored in many versions, so any choice between versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this run |
| `code/launch_shards.sh`, `code/shard.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, each under a sleep lock. They expect the pipeline layout `rssns_for_mi_EEG/src` next to them and a `.venv` two folders up |
| `results/20261107_093450/config.json` | Every setting of the run, including the parameter counts |
| `results/20261107_093450/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261107_093450/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261107_093450/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261107_093450/predictions/<model>/s<S>_r<R>.npz` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261107_093450/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261107_093450/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261107_093450/checkpoints/` | Trained weights (PyTorch `.pt`), pretrained and fine-tuned |
| `results/20261107_093450/comparison/` | Pipeline statistics: chance tests, per-subject summary, and paired tests and bootstrap intervals where a run has two models or more (this run has one model, so `paired_tests.csv` is empty; the comparison with the other models is in `../summary/`) |
| `results/20261107_093450/figures/` | Pipeline figures (accuracy per subject, confusion matrices, metric distributions) |
| `results/20261107_093450/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.2.diff` shows every line of `code/src/` that differs from `v1.2`, the version this run was built from.

Glossary and the overview of all versions: `../README.md`.
