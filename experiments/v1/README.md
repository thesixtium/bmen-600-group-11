# v1: the baseline run

> The pipeline as agreed in the plan, run once in full on all four models. Every later version changes the RSNN (and its RNN control) only.

## What was done

The pipeline code was run unchanged on all 90 folds. The RNN control is a non-spiking twin of the RSNN (recurrent neural network with the ELU (exponential linear unit) activation, same size, inputs and training), so a difference between them shows the effect of spiking alone.

| Setting | Value |
|---|---|
| Input | 22 EEG (electroencephalography) channels × 256 samples (4–40 Hz, 128 Hz, 0.5–2.5 s after the cue) |
| RSNN (recurrent spiking neural network) | 17 leaky integrate-and-fire neurons, all-to-all recurrence, 752 parameters (converted SNN: 776) |
| Readout | One spike rate per neuron over the whole trial |
| Batch size | EEGNet 16, RSNN 64, RNN 64 |
| Epochs | 500 pretraining, 100 fine-tuning, best validation weights restored |

| Step | Detail |
|---|---|
| Run | Run id `20261029_141407`, 9 target subjects × 10 repeats = 90 folds per model, the same folds and seeds as v1 (fold seed = 2026 + 100 × target + repeat) |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Not run on ARC |
| Models trained | EEGNet (ANN), Converted SNN, RSNN, RNN control |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

| Main observation | Evidence |
|---|---|
| The RSNN underfits: it does not fit its own pretraining data | Training accuracy on the pretraining subjects 0.29–0.34, training loss near ln 4 = 1.386 (`histories/`) |
| The weakness is shared by the RNN, so spiking is not the only cause | RNN 0.411 against RSNN 0.364 |
| The RNN diverges in some folds | Training loss above 5 in 8 of 90 folds (`histories/rnn/`) |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| EEGNet (ANN) | 0.597 ± 0.153 | 0.462 | 0.582 | 0.866 | 9/9 |
| Converted SNN | 0.569 ± 0.15 | 0.425 | 0.547 | 0.856 | 9/9 |
| RSNN | 0.364 ± 0.088 | 0.152 | 0.316 | 0.788 | 6/9 |
| RNN control | 0.411 ± 0.117 | 0.215 | 0.372 | 0.804 | 7/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set of the RSNN was scored in v1, v1.1, v1.2, v1.3 and v1.4, so the choice between these versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this version |
| `code/how_to_run.txt` | Run instructions of the pipeline |
| `code/launch_shards.sh`, `code/watch_run.py` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, and show progress. They expect the pipeline layout `rssns_for_mi_EEG/src` and a `.venv` two folders up |
| `results/20261029_141407/config.json` | Every setting of the run, including the parameter counts |
| `results/20261029_141407/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261029_141407/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261029_141407/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261029_141407/predictions/<model>/s<S>_r<R>.npz`, `predictions.csv` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261029_141407/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261029_141407/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261029_141407/checkpoints/` | Trained weights (PyTorch `.pt`) |
| `results/20261029_141407/comparison/` | Pipeline statistics: chance tests, paired Wilcoxon tests with Holm correction, bootstrap intervals, per-subject summary, figures |
| `results/20261029_141407/figures/` | Pipeline figures (training curves, accuracy per subject) |
| `results/20261029_141407/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

Glossary and the overview of all versions: `../README.md`.
