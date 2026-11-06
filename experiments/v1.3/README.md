# v1.3: mu and beta band inputs

> One change from v1: the RSNN and the RNN get the EEG split into the mu (8–13 Hz) and beta (13–30 Hz) bands, 44 inputs instead of 22.

## What was done

Motor imagery shows as a power drop in the mu and beta bands over the motor cortex. EEGNet learns such filters in its first layer, the RSNN has no filter layer. A fixed band-pass front end gives the recurrent models these bands directly. The hidden size drops to keep the parameter count matched. The readout is one window again (v1).

| Setting | Value |
|---|---|
| Front end | Butterworth band-pass, order 4, zero-phase, 8–13 Hz and 13–30 Hz per channel → 44 inputs |
| RSNN and RNN hidden neurons | 12, 736 parameters |
| Readout | One rate per neuron (as v1) |
| Batch size | 64 (as v1) |

| Step | Detail |
|---|---|
| Run | Run id `20261102_163908`, 9 target subjects × 10 repeats = 90 folds per model, the same folds and seeds as v1 (fold seed = 2026 + 100 × target + repeat) |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Not run on ARC |
| Models trained | RSNN, RNN control |
| Models reused | EEGNet (ANN), Converted SNN: their code did not change, so their predictions, metrics and histories were copied from v1 (`results/20261102_163908/SEEDED_FROM_V1.txt`). The statistics in `comparison/` then compare all models on the same 90 folds |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| RSNN | 0.428 ± 0.123 | 0.237 | 0.383 | 0.809 | 7/9 |
| RNN control | 0.416 ± 0.107 | 0.221 | 0.369 | 0.805 | 7/9 |
| EEGNet (ANN) (copied from v1) | 0.597 ± 0.153 | 0.462 | 0.582 | 0.866 | 9/9 |
| Converted SNN (copied from v1) | 0.569 ± 0.15 | 0.425 | 0.547 | 0.856 | 9/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set of the RSNN was scored in v1, v1.1, v1.2, v1.3 and v1.4, so the choice between these versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this version |
| `code/how_to_run.txt` | Run instructions of the pipeline |
| `code/launch_shards.sh`, `code/watch_run.py`, `code/seed_from_v1.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, and show progress. They expect the pipeline layout `rssns_for_mi_EEG/src` and a `.venv` two folders up |
| `results/20261102_163908/config.json` | Every setting of the run, including the parameter counts |
| `results/20261102_163908/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261102_163908/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261102_163908/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261102_163908/predictions/<model>/s<S>_r<R>.npz`, `predictions.csv` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261102_163908/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261102_163908/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261102_163908/checkpoints/` | Trained weights (PyTorch `.pt`) |
| `results/20261102_163908/comparison/` | Pipeline statistics: chance tests, paired Wilcoxon tests with Holm correction, bootstrap intervals, per-subject summary, figures |
| `results/20261102_163908/figures/` | Pipeline figures (training curves, accuracy per subject) |
| `results/20261102_163908/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.diff` shows every line of `code/src/` that differs from v1, the version this one was built from.

Glossary and the overview of all versions: `../README.md`.
