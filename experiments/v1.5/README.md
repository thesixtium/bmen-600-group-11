# v1.5: where does the RSNN lose accuracy? (screen)

> Two diagnostic parts, no new setting: (A) the saved weights of all 90 folds scored on session 1, (B) fine-tuning without pretraining on the 9 development folds.

> ⚠️ **A screen, not a result.** It uses only session-1 data and only the 9 development folds (all 9 targets, repeat 0). The 20 % fine-tuning split also chose the fine-tuning checkpoint, so its scores are slightly optimistic for every model alike. Session 2 (the test set) was never used. A screen ranks settings. It is not a test result and must not be compared with the session-2 numbers of v1 to v1.4.

## What was done

The RSNN trails EEGNet. This screen asks in which phase the gap opens. Part A loads the saved pretrained and fine-tuned weights (EEGNet from v1, RSNN and RNN from v1.4) and scores them on session-1 data only: `code/eval_saved.py` → `results/saved_eval.csv`. Part B trains each model with fine-tuning only, from the same random start, and compares it with pretraining followed by fine-tuning. Part B also retrains the v1.4 setting with `--check` and confirms that the RSNN and RNN weights match the saved ones exactly.

| Step | Detail |
|---|---|
| Code | `code/screen.py` trains one model on one fold with the v1.4 pipeline code unchanged (`../v1.4/code/src`). `code/screen_common.py` loads the data and builds the fold for each variant. `code/jobs/v1.5.txt` lists every job |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 jobs in parallel (`code/run_screen.sh`) |
| Analysis | `code/analyze.py v1.5` summarizes parts A and B and applies two rules fixed before the run. It writes `results/summary.json` |
| Reference | RSNN and RNN: the saved v1.4 weights of repeat 0 (the screen code reproduces them exactly). EEGNet: its retrained v1.4 run from v1.5 (EEGNet training is not bit-exact on the GPU) |

## Part A: saved weights, 90 folds (session 1)

| Model | Pretrained, target session 1 | Subjects above chance | Pretrained, 20 % split | Fine-tuned, 20 % split |
|---|---:|---:|---:|---:|
| EEGNet (ANN) | 0.375 | 6/9 | 0.377 | 0.631 |
| RSNN | 0.340 | 6/9 | 0.344 | 0.485 |
| RNN control | 0.333 | 5/9 | 0.334 | 0.455 |

## Part B: fine-tuning only against pretraining + fine-tuning (20 % split, 9 folds)

| Model | Fine-tuning only | With pretraining | Folds where pretraining is better |
|---|---:|---:|---:|
| EEGNet (ANN) | 0.586 | 0.619 | 5/9 |
| RSNN | 0.400 | 0.494 | 9/9 |
| RNN control | 0.414 | 0.487 | 6/9 |

Rules (fixed before the run): "pretraining is the RSNN's first problem": **no**. "pretraining gives the RSNN nothing": **no**. Pretraining helps the RSNN in every fold, and the gap to EEGNet opens mainly in fine-tuning.

## Files

| Path | What it holds |
|---|---|
| `results/v1.4_*.json` | One file per model, fold and arm (`full` or `ftonly`): session-1 accuracies, best pretraining epoch and validation loss, largest training loss, run time |
| `results/saved_eval.csv` | Part A: session-1 scores of the saved weights, 270 rows (3 models × 90 folds) |
| `results/logs/` | One training log per job |
| `results/summary.json` | Output of `analyze.py`: means, paired differences, subjects better or worse, unadjusted Wilcoxon p, the rule |
| `code/` | `screen.py`, `screen_common.py`, `eval_saved.py`, `analyze.py`, `run_screen.sh`, `jobs/` (the same code in v1.5, v1.6 and v1.7) |

## Run it again

```bash
# from experiments/v1.5/code, with the Python that has torch 2.4.0
python screen_common.py          # downloads BCI IV 2a (MOABB) and builds code/cache/ once
PYTHON=python ./run_screen.sh v1.5  # finished jobs are skipped
python analyze.py v1.5
```

`code/cache/` holds the preprocessed data (about 340 MB). Do not commit it.

Part A needs the checkpoints in `../v1/results/` and `../v1.4/results/`.

Glossary and the overview of all versions: `../README.md`.
