# Experiments v1 to v1.7

> Code, settings and results of every version of the local training runs: 5 full runs (v1 to v1.4, scored on the test set) and 3 screens (v1.5 to v1.7, session 1 only). Each folder has its own README.

## 1. Versions at a glance

```mermaid
flowchart LR
  v1[v1 baseline] --> v11[v1.1 batch 16]
  v1 --> v12[v1.2 4-window readout]
  v1 --> v13[v1.3 mu and beta inputs]
  v13 --> v14[v1.4 clipping and spectral cap]
  v14 --> v15[v1.5 screen: where is the gap]
  v14 --> v16[v1.6 screen: per-subject z-score]
  v14 --> v17[v1.7 screen: pooled split]
```

```
v1 --+-- v1.1  (batch 16)
     +-- v1.2  (4-window readout)
     +-- v1.3  (mu + beta inputs) -- v1.4 (clipping + spectral cap) --+-- v1.5 screen
                                                                     +-- v1.6 screen
                                                                     +-- v1.7 screen
```

| Folder | Kind | One change (from) | Models trained | Scored on | RSNN accuracy | EEGNet accuracy |
|---|---|---|---|---|---:|---:|
| `v1/` | Full run, 90 folds | Baseline, as planned | EEGNet, SNN, RSNN, RNN | Session 2 | 0.364 | 0.597 |
| `v1.1/` | Full run, 90 folds | RSNN batch 64 → 16 (v1) | RSNN | Session 2 | 0.386 | 0.597 (v1) |
| `v1.2/` | Full run, 90 folds | 4-window readout, 14 neurons (v1) | RSNN | Session 2 | 0.371 | 0.597 (v1) |
| `v1.3/` | Full run, 90 folds | Mu and beta band inputs, 12 neurons (v1) | RSNN, RNN | Session 2 | 0.428 | 0.597 (v1) |
| `v1.4/` | Full run, 90 folds | Gradient clipping 1.0, initial spectral norm ≤ 0.9 (v1.3) | RSNN, RNN | Session 2 | 0.432 | 0.597 (v1) |
| `v1.5/` | Screen | None: diagnostic of v1/v1.4 weights, fine-tuning only arm | EEGNet, RSNN, RNN | Session 1 | 0.485 (fine-tuned, 20 % split) | 0.631 |
| `v1.6/` | Screen, 9 folds | Per-subject z-score (v1.4) | EEGNet, RSNN, RNN | Session 1 | 0.490 (20 % split) | 0.648 |
| `v1.7/` | Screen, 9 folds | Pooled 80 / 20 pretraining split (v1.4) | EEGNet, RSNN, RNN | Session 1 | 0.529 (20 % split) | 0.686 |

Session-2 accuracies are means over 9 subjects (each the mean of 10 repeats). Converted SNN, v1: 0.569. Screen accuracies are on session-1 data and are **not comparable** with the session-2 column of the full runs.

> ⚠️ All numbers here are **preliminary**. They were made before the pipeline validation checklist passed, and the RSNN's test set was scored in 5 versions. The screens did not meet their decision rules, so no screen setting was carried into a full run.

## 2. Folder layout

| Path | What it holds |
|---|---|
| `v1/` … `v1.4/` | `README.md`, `code/` (pipeline source as run, local launch helpers), `results/<run id>/` (the complete run folder: config, folds, predictions, metrics, histories, checkpoints, statistics, figures, logs) |
| `v1.1/` … `v1.4/CODE_CHANGES_vs_<base>.diff` | Every source line that differs from the version it was built from |
| `v1.5/` … `v1.7/` | `README.md`, `code/` (screen scripts, the same in all three), `results/` (one JSON per job, logs, `summary.json`) |
| `summary/` | `compute_metrics.py` and its output `metrics_by_version.csv`: accuracy, Cohen's kappa, macro-F1, macro specificity, macro precision and subjects above chance for every model of v1 to v1.4 |

## 3. Use the results for analysis

| Task | Where |
|---|---|
| Recompute any test metric | `v*/results/<run id>/predictions/<model>/s<S>_r<R>.npz` (arrays `y_true` and `y_pred` of the 288 test trials, `scores` 288 × 4) or `predictions.csv` |
| Per-fold accuracy, kappa, macro-F1 | `results.csv` of the run |
| Confusion matrices | `confusion_matrices.csv` of the run |
| The pipeline statistics (chance, Wilcoxon with Holm correction, bootstrap) | `comparison/` of the run |
| Training curves | `histories/` of the run |
| All versions in one table | `summary/metrics_by_version.csv` (rerun: `python experiments/summary/compute_metrics.py` from the repository root) |

In v1.1 to v1.4, the models that did not change (see each README) were copied from v1, so every run folder holds all four models on the same 90 folds.

## 4. Environment

`requirements.txt` pins the exact package versions of every run (Python 3.12, torch 2.4.0).

```bash
# from the repository root
uv venv --python 3.12 .venv
uv pip install -r experiments/requirements.txt
```

`uv sync` is not used here: it needs a `pyproject.toml` and a `uv.lock`, and the repository has neither

#

## Glossary

| Term | Meaning |
|---|---|
| ANN | Artificial neural network, here EEGNet |
| SNN | Spiking neural network, here EEGNet converted to spiking neurons |
| RSNN | Recurrent spiking neural network: one all-to-all recurrent layer of leaky integrate-and-fire neurons |
| RNN control | Recurrent neural network with the ELU (exponential linear unit) activation, the same size and training as the RSNN, without spikes |
| EEG | Electroencephalography |
| Session 1 / session 2 | The two recording days of BCI IV 2a (Brain-Computer Interface Competition IV, dataset 2a). Session 2 is the test set |
| Fold | One target subject and one random draw of 5 pretraining and 3 validation subjects. 9 targets × 10 repeats = 90 folds |
| Development folds | Repeat 0 of each target: 9 folds used by the screens |
| Screen | A short experiment on session-1 data that ranks a setting without using the test set |
| Spectral norm | The largest singular value of a matrix. Above 1 a recurrent layer can amplify its own activity |
| Gradient clipping | Scaling the gradient down when its norm is above a limit, so one update cannot be very large |
| Z-score | (value − mean) / standard deviation |
| Macro-F1 | The F1 score (harmonic mean of precision and recall) averaged over the 4 classes |
| GPU | Graphics processing unit |
