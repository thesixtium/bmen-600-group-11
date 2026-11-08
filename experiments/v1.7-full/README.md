# v1.7-full: pooled pretraining split, full run

> The v1.7 change (`../v1.7/`, a session-1 screen) as a full 90-fold run scored on session 2: pretraining uses session 1 of all 8 non-target subjects, split 80 / 20, instead of 5 training and 3 validation subjects. Built on v1.4. All four models.

## What was done

For each fold, session 1 of all 8 non-target subjects is pooled and split 80 / 20 (stratified by subject and class,
`random_state` = fold seed) into pretraining and pretraining-validation trials. The fine-tuning and test splits of the
target are unchanged. `config.json` → `fold` still shows `n_pretrain_subjects: 5`, because the fold list (and so the
seeds) is the same as v1; the pooled split is made from it in `folds.py`.

| Setting | Value |
|---|---|
| Pretraining data | All 8 other subjects, 80 % train / 20 % validation (v1 to v1.6: 5 train subjects / 3 validation subjects) |
| Recurrent models | As v1.4: 44 mu and beta band inputs, 12 units, 736 parameters, gradient clipping 1.0, initial recurrent spectral norm ≤ 0.9 |
| EEGNet, SNN | As v1 |

| Step | Detail |
|---|---|
| Run | Run id `20261106_144835`, 90 folds per model, the same folds and seeds as v1 to v1.6 |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Two runs shared the GPU at the same time, so the times per fold are longer than for a run alone. Not run on ARC |
| Time | EEGNet 3.4, RSNN 21.9 and RNN 11.9 min per fold on average, about 18.8 h for the run |
| Models trained | EEGNet, converted SNN (from this run's EEGNet), RSNN, RNN |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| EEGNet (ANN) | 0.654 ± 0.161 | 0.538 | 0.645 | 0.885 | 9/9 |
| Converted SNN | 0.647 ± 0.159 | 0.529 | 0.636 | 0.882 | 9/9 |
| RSNN | 0.470 ± 0.138 | 0.294 | 0.443 | 0.823 | 8/9 |
| RNN control | 0.477 ± 0.135 | 0.303 | 0.454 | 0.826 | 9/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`. In that file these rows have the version label `v1.7`.

| Paired test (accuracy, 9 subjects) | Median difference | First model better | Holm-corrected p |
|---|---:|---:|---:|
| RSNN − SNN (the research question) | −0.162 | 0 of 9 | 0.020 |
| ANN − SNN (lost in conversion) | +0.003 | 9 of 9 | 0.020 |
| RNN − RSNN (effect of spiking) | +0.009 | 6 of 9 | 0.359 |

Source: `results/20261106_144835/comparison/paired_tests.csv` (exact two-sided Wilcoxon, Holm correction over the comparisons of
each metric). v1.7 is the best version for every model, but the RSNN stays below the converted SNN for all 9 subjects.

Session 1 (20 % split, 9 development folds, saved checkpoints): EEGNet 0.682, SNN 0.674,
RSNN 0.529, RNN 0.511.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set has now been scored in many versions, so any choice between versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this run |
| `code/launch_shards.sh`, `code/shard.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, each under a sleep lock. They expect the pipeline layout `rssns_for_mi_EEG/src` next to them and a `.venv` two folders up |
| `results/20261106_144835/config.json` | Every setting of the run, including the parameter counts |
| `results/20261106_144835/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261106_144835/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261106_144835/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261106_144835/predictions/<model>/s<S>_r<R>.npz` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261106_144835/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261106_144835/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261106_144835/checkpoints/` | Trained weights (PyTorch `.pt`), pretrained and fine-tuned |
| `results/20261106_144835/comparison/` | Pipeline statistics: chance tests, per-subject summary, paired Wilcoxon tests with Holm correction, bootstrap intervals, figures |
| `results/20261106_144835/figures/` | Pipeline figures (accuracy per subject, confusion matrices, metric distributions) |
| `results/20261106_144835/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.4.diff` shows every line of `code/src/` that differs from `v1.4`, the version this run was built from.

Glossary and the overview of all versions: `../README.md`.
