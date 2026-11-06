# v1.4: stable recurrent training

> One change from v1.3: gradient clipping and a cap on the initial recurrent weights, for the RSNN and the RNN alike.

## What was done

The RNN diverged in some folds (very large training loss). Two standard fixes for recurrent networks were added: the gradient norm is clipped to 1.0 before each update, and the initial recurrent weight matrix is scaled so its largest singular value (spectral norm) is at most 0.9. `code/check_clipping.py` and `code/check_rnn_init.py` (with their JSON outputs) are the checks made before the run.

| Setting | Value |
|---|---|
| Gradient clipping | Global L2 norm 1.0 (RSNN and RNN) |
| Initial recurrent spectral norm | At most 0.9 (RSNN and RNN) |
| Front end, sizes, batch | As v1.3 |

| Step | Detail |
|---|---|
| Run | Run id `20261103_175004`, 9 target subjects × 10 repeats = 90 folds per model, the same folds and seeds as v1 (fold seed = 2026 + 100 × target + repeat) |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Not run on ARC |
| Models trained | RSNN, RNN control |
| Models reused | EEGNet (ANN), Converted SNN: their code did not change, so their predictions, metrics and histories were copied from v1 (`results/20261103_175004/SEEDED_FROM_V1.txt`). The statistics in `comparison/` then compare all models on the same 90 folds |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| RSNN | 0.432 ± 0.123 | 0.243 | 0.39 | 0.811 | 8/9 |
| RNN control | 0.434 ± 0.124 | 0.245 | 0.39 | 0.811 | 7/9 |
| EEGNet (ANN) (copied from v1) | 0.597 ± 0.153 | 0.462 | 0.582 | 0.866 | 9/9 |
| Converted SNN (copied from v1) | 0.569 ± 0.15 | 0.425 | 0.547 | 0.856 | 9/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set of the RSNN was scored in v1, v1.1, v1.2, v1.3 and v1.4, so the choice between these versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this version |
| `code/how_to_run.txt` | Run instructions of the pipeline |
| `code/launch_shards.sh`, `code/watch_run.py`, `code/seed_from_v1.sh`, `code/shard.sh`, `code/check_*.py` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, and show progress. They expect the pipeline layout `rssns_for_mi_EEG/src` and a `.venv` two folders up |
| `results/20261103_175004/config.json` | Every setting of the run, including the parameter counts |
| `results/20261103_175004/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261103_175004/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261103_175004/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261103_175004/predictions/<model>/s<S>_r<R>.npz`, `predictions.csv` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261103_175004/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261103_175004/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261103_175004/checkpoints/` | Trained weights (PyTorch `.pt`) |
| `results/20261103_175004/comparison/` | Pipeline statistics: chance tests, paired Wilcoxon tests with Holm correction, bootstrap intervals, per-subject summary, figures |
| `results/20261103_175004/figures/` | Pipeline figures (training curves, accuracy per subject) |
| `results/20261103_175004/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.3.diff` shows every line of `code/src/` that differs from v1.3, the version this one was built from.

Glossary and the overview of all versions: `../README.md`.
