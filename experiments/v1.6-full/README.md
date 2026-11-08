# v1.6-full: per-subject normalization, full run

> The v1.6 change (`../v1.6/`, a session-1 screen) as a full 90-fold run scored on session 2: every subject's data are z-scored per channel with that subject's own session-1 statistics. Built on v1.4. EEGNet, converted SNN and RNN; the RSNN is not trained in this run.

## What was done

Each subject's EEG is scaled to microvolts, then z-scored per channel with the mean and SD of that subject's own
session 1 (over trials and time; labels are not used). Session 2 of a subject is scaled with its session-1 statistics,
so no session-2 information enters. This applies to every split and to both input kinds (22 channels for EEGNet, 44 mu
and beta band inputs for the recurrent models).

| Setting | Value |
|---|---|
| Input normalization | Per subject, per channel, session-1 statistics (v1.4: one fixed unit scale) |
| Recurrent models | As v1.4: 44 mu (8–13 Hz) and beta (13–30 Hz) band inputs, 12 units, 736 parameters, gradient clipping 1.0, initial recurrent spectral norm ≤ 0.9 |
| Folds | As v1: 5 pretraining and 3 validation subjects |

| Step | Detail |
|---|---|
| Run | Run id `20261107_085030`, 90 folds per model, the same folds and seeds as v1 to v1.4 |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Two runs shared the GPU at the same time, so the times per fold are longer than for a run alone. Not run on ARC |
| Time | EEGNet 3.3 min and RNN 10.9 min per fold on average, about 7.3 h for the run |
| Models trained | EEGNet, converted SNN (from this run's EEGNet), RNN. The RSNN was not trained; its v1.6 value is the session-1 screen in `../v1.6/` |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| EEGNet (ANN) | 0.609 ± 0.141 | 0.479 | 0.591 | 0.870 | 9/9 |
| Converted SNN | 0.578 ± 0.143 | 0.438 | 0.551 | 0.859 | 9/9 |
| RNN control | 0.423 ± 0.105 | 0.230 | 0.383 | 0.808 | 8/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`. In that file these rows have the version label `v1.6`.

ANN − SNN (accuracy lost in conversion): median +0.041, ANN better for 7 of 9
subjects, Holm-corrected p = 0.129. Against v1 (EEGNet 0.597, SNN 0.569) and v1.4 (RNN 0.434) the
change is small.

Session 1 (20 % split, 9 development folds, saved checkpoints): EEGNet 0.655, SNN 0.552,
RNN 0.452; RSNN screen 0.490 (`../v1.6/results/summary.json`).

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set has now been scored in many versions, so any choice between versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this run |
| `code/launch_shards.sh`, `code/shard.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, each under a sleep lock. They expect the pipeline layout `rssns_for_mi_EEG/src` next to them and a `.venv` two folders up |
| `results/20261107_085030/config.json` | Every setting of the run, including the parameter counts |
| `results/20261107_085030/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261107_085030/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261107_085030/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261107_085030/predictions/<model>/s<S>_r<R>.npz` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261107_085030/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261107_085030/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261107_085030/checkpoints/` | Trained weights (PyTorch `.pt`), pretrained and fine-tuned |
| `results/20261107_085030/comparison/` | Pipeline statistics: chance tests, per-subject summary, paired Wilcoxon tests with Holm correction, bootstrap intervals, figures |
| `results/20261107_085030/figures/` | Pipeline figures (accuracy per subject, confusion matrices, metric distributions) |
| `results/20261107_085030/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.4.diff` shows every line of `code/src/` that differs from `v1.4`, the version this run was built from.

Glossary and the overview of all versions: `../README.md`.
