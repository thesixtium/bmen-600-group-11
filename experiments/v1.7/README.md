# v1.7: pooled pretraining split (screen)

> Pretraining uses session 1 of all 8 non-target subjects, split 80 / 20 stratified by subject and class (1,843 training and 461 validation trials), instead of 5 training and 3 validation subjects. Compared with v1.4 on the 9 development folds, for EEGNet, the RSNN and the RNN.

> ⚠️ **A screen, not a result.** It uses only session-1 data and only the 9 development folds (all 9 targets, repeat 0). The 20 % fine-tuning split also chose the fine-tuning checkpoint, so its scores are slightly optimistic for every model alike. Session 2 (the test set) was never used. A screen ranks settings. It is not a test result and must not be compared with the session-2 numbers of v1 to v1.4.

## What was done

With 3 held-out validation subjects, the validation loss measures a new-subject effect and stops pretraining early. A pooled split validates on the same subjects the model trains on and uses all 8 subjects for training. [DEVIATION: changes the 5 / 3 subject split of the plan]

| Step | Detail |
|---|---|
| Code | `code/screen.py` trains one model on one fold with the v1.4 pipeline code unchanged (`../v1.4/code/src`). `code/screen_common.py` loads the data and builds the fold for each variant. `code/jobs/v1.7.txt` lists every job |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 jobs in parallel (`code/run_screen.sh`) |
| Analysis | `code/analyze.py v1.7` compares with the v1.4 reference on the same folds and applies a decision rule fixed before the run. It writes `results/summary.json` |
| Reference | RSNN and RNN: the saved v1.4 weights of repeat 0 (the screen code reproduces them exactly). EEGNet: its retrained v1.4 run from v1.5 (EEGNet training is not bit-exact on the GPU) |

Decision rule (fixed before the run): the change helps if the RSNN is better on both measures in at least 7 of 9 folds and EEGNet is worse in at most 2 of 9.

## Results (session 1, 9 folds, v1.4 → v1.7)

| Model | After pretraining: target session 1 (288 trials) | Better / 9 | After fine-tuning: 20 % split | Better / 9 |
|---|---:|---:|---:|---:|
| EEGNet (ANN) | 0.353 → 0.445 | 8 | 0.623 → 0.686 | 7 |
| RSNN | 0.345 → 0.383 | 6 | 0.494 → 0.529 | 5 |
| RNN control | 0.353 → 0.409 | 6 | 0.487 → 0.511 | 5 |

Rule met: **no**. Best pretraining epoch (median): EEGNet (ANN) 470, RSNN 456, RNN control 235. Largest training loss: EEGNet (ANN) 2.15, RSNN 1.89, RNN control 1.44e+05.

## Files

| Path | What it holds |
|---|---|
| `results/v1.7_full_*.json` | One file per model, fold: session-1 accuracies, best pretraining epoch and validation loss, largest training loss, run time |
| `results/logs/` | One training log per job |
| `results/summary.json` | Output of `analyze.py`: means, paired differences, subjects better or worse, unadjusted Wilcoxon p, the rule |
| `code/` | `screen.py`, `screen_common.py`, `eval_saved.py`, `analyze.py`, `run_screen.sh`, `jobs/` (the same code in v1.5, v1.6 and v1.7) |

## Run it again

```bash
# from experiments/v1.7/code, with the Python that has torch 2.4.0
python screen_common.py          # downloads BCI IV 2a (MOABB) and builds code/cache/ once
PYTHON=python ./run_screen.sh v1.7  # finished jobs are skipped
python analyze.py v1.7
```

`code/cache/` holds the preprocessed data (about 340 MB). Do not commit it.

Glossary and the overview of all versions: `../README.md`.
