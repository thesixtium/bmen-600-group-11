# v1.6: per-subject normalization (screen)

> Every subject is z-scored per channel with its own session-1 mean and standard deviation, before pretraining and fine-tuning. Compared with v1.4 on the 9 development folds, for EEGNet, the RSNN and the RNN.

> ⚠️ **A screen, not a result.** It uses only session-1 data and only the 9 development folds (all 9 targets, repeat 0). The 20 % fine-tuning split also chose the fine-tuning checkpoint, so its scores are slightly optimistic for every model alike. Session 2 (the test set) was never used. A screen ranks settings. It is not a test result and must not be compared with the session-2 numbers of v1 to v1.4.

## What was done

Subjects differ in signal amplitude. A network pretrained on 5 subjects then sees a new subject at another scale. Per-subject scaling removes that difference. It uses session-1 statistics only, so no test data is used. [DEVIATION: new preprocessing step, not in the plan]

| Step | Detail |
|---|---|
| Code | `code/screen.py` trains one model on one fold with the v1.4 pipeline code unchanged (`../v1.4/code/src`). `code/screen_common.py` loads the data and builds the fold for each variant. `code/jobs/v1.6.txt` lists every job |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 jobs in parallel (`code/run_screen.sh`) |
| Analysis | `code/analyze.py v1.6` compares with the v1.4 reference on the same folds and applies a decision rule fixed before the run. It writes `results/summary.json` |
| Reference | RSNN and RNN: the saved v1.4 weights of repeat 0 (the screen code reproduces them exactly). EEGNet: its retrained v1.4 run from v1.5 (EEGNet training is not bit-exact on the GPU) |

Decision rule (fixed before the run): the change helps if the RSNN is better on both measures in at least 7 of 9 folds and EEGNet is worse in at most 2 of 9.

## Results (session 1, 9 folds, v1.4 → v1.6)

| Model | After pretraining: target session 1 (288 trials) | Better / 9 | After fine-tuning: 20 % split | Better / 9 |
|---|---:|---:|---:|---:|
| EEGNet (ANN) | 0.353 → 0.400 | 4 | 0.623 → 0.648 | 6 |
| RSNN | 0.345 → 0.382 | 7 | 0.494 → 0.490 | 3 |
| RNN control | 0.353 → 0.390 | 7 | 0.487 → 0.452 | 2 |

Rule met: **no**. Best pretraining epoch (median): EEGNet (ANN) 9, RSNN 127, RNN control 13. Largest training loss: EEGNet (ANN) 2.5, RSNN 2.03, RNN control 4.3.

## Files

| Path | What it holds |
|---|---|
| `results/v1.6_full_*.json` | One file per model, fold: session-1 accuracies, best pretraining epoch and validation loss, largest training loss, run time |
| `results/logs/` | One training log per job |
| `results/summary.json` | Output of `analyze.py`: means, paired differences, subjects better or worse, unadjusted Wilcoxon p, the rule |
| `code/` | `screen.py`, `screen_common.py`, `eval_saved.py`, `analyze.py`, `run_screen.sh`, `jobs/` (the same code in v1.5, v1.6 and v1.7) |

## Run it again

```bash
# from experiments/v1.6/code, with the Python that has torch 2.4.0
python screen_common.py          # downloads BCI IV 2a (MOABB) and builds code/cache/ once
PYTHON=python ./run_screen.sh v1.6  # finished jobs are skipped
python analyze.py v1.6
```

`code/cache/` holds the preprocessed data (about 340 MB). Do not commit it.

Glossary and the overview of all versions: `../README.md`.
