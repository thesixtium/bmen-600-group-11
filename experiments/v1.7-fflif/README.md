# v1.7-fflif: the RSNN of v1.7 without its recurrence (FF-LIF control)

> One change from `v1.7-full`: the recurrent weights of the RSNN are removed and the layer is widened to the same parameter budget (15 neurons, 739 parameters; RSNN 12 neurons, 736). Only FF-LIF is trained.

## What was done

FF-LIF (feedforward leaky integrate-and-fire) is the RSNN with its recurrent weight matrix W_rec removed and
nothing else changed. Each neuron still leaks, integrates its input over time and resets after a spike, so it keeps a
memory of its own past input, but it never receives a spike from another neuron:

```
RSNN:    u_t = 0.9 · u_(t−1) + W_in · x_t + b + W_rec · s_(t−1) − 1.0 · s_(t−1)
FF-LIF:  u_t = 0.9 · u_(t−1) + W_in · x_t + b                    − 1.0 · s_(t−1)
both:    s_t = 1 if u_t > 1.0 else 0;   output = W_out · mean_t(s_t) + b_out
```

Without the H × H matrix the same parameter budget buys more neurons. `choose_fflif_hidden_size` in `benchmark.py` takes
the largest hidden size whose parameter count is at most that of the converted SNN (776), the rule the RSNN uses. The
class is `FeedforwardLIFNet` in `models.py`; `run_fflif_benchmark` is a copy of the RSNN runner. The control is not in
the research plan; it was added to separate the effect of the recurrence from the other differences between the RSNN
and the converted SNN.

| Setting | RSNN of v1.7 | FF-LIF (this run) |
|---|---|---|
| Connections between neurons | all-to-all, trainable | none |
| Hidden size, parameters | 12, 736 | 15, 739 |
| Input | 44 mu and beta band inputs, standardized; pooled 80 / 20 pretraining split | same |
| Neurons, readout | LIF, β = 0.9, threshold 1.0, reset by subtraction; mean spike rate | same |
| Training | `RSNN_TRAIN`: 500 + 100 epochs, learning rate 0.001, batch 64, dropout 0.25 then 0.5; gradient clipping 1.0 | same |

| Step | Detail |
|---|---|
| Run | Run id `20261107_160945`, 90 folds, the same folds and seeds as `v1.7-full`, so every fold pairs with the RSNN fold of the same subject and repeat |
| Hardware | One local laptop GPU (NVIDIA RTX 4060), 3 shards in parallel (targets 1–3, 4–6, 7–9). Two runs shared the GPU at the same time, so the times per fold are longer than for a run alone. Not run on ARC |
| Time | 14.9 min per fold on average, about 7.6 h for the run. Removing W_rec does not make a fold faster: the 256 steps still run one after the other |
| Models trained | FF-LIF only (`MODELS = ('fflif',)`). The RSNN it is compared with is in `../v1.7-full/` |
| Test data | Target session 2, 288 trials (72 per class). Session 2 was not used for training, early stopping or checkpoint selection |
| Checks | With the same weights FF-LIF gives exactly the output of an RSNN whose W_rec is zero; no recurrent weight is in the checkpoints; 90 of 90 folds, 739 parameters in every fold, no NaN losses |

## Results (session 2, 90 folds)

| Model | Accuracy (mean ± SD over 9 subjects) | Cohen's kappa | Macro-F1 | Macro specificity | Subjects above chance |
|---|---:|---:|---:|---:|---:|
| FF-LIF control (this run) | 0.479 ± 0.151 | 0.306 | 0.452 | 0.826 | 8/9 |
| RSNN (v1.7) | 0.470 ± 0.138 | 0.294 | 0.443 | 0.823 | 8/9 |

Subject values are the mean of the 10 repeats. Chance threshold: 0.292 accuracy (binomial, n = 288, p = 0.25, α = 0.05). Source: `../summary/metrics_by_version.csv`, made from this folder by `../summary/compute_metrics.py`.

| RSNN against FF-LIF (v1.7, accuracy, 9 subjects) | Value |
|---|---:|
| Median paired difference, RSNN − FF-LIF | −0.001 |
| Subjects where the RSNN is better | 3 of 9 |
| Exact two-sided Wilcoxon p (one test, not corrected) | 0.359 |
| Session 1, 20 % split, 9 development folds: RSNN / FF-LIF | 0.529 / 0.489 |

Source: `../summary/table33.csv`, made by `../summary/build_tables.py`. The RSNN and FF-LIF are within 0.02 of each
other and the subjects go both ways: the trained recurrence adds no measurable accuracy here. "Not significant" does not
prove "equal": with 9 subjects only a large effect can be excluded. FF-LIF uses the RSNN's settings and was not tuned for
itself.

> ⚠️ **Preliminary.** These session-2 (test) numbers come from a local run before the pipeline validation checklist passed. The test set has now been scored in many versions, so any choice between versions is itself tuned on the test set. Use them as exploratory results, not as the final answer to the research question.

## Files

| Path | What it holds |
|---|---|
| `code/src/` | The pipeline source exactly as it ran for this run |
| `code/launch_shards.sh`, `code/shard.sh` | Local helpers used for the run: start 3 shards of 3 target subjects in parallel on one GPU, each under a sleep lock. They expect the pipeline layout `rssns_for_mi_EEG/src` next to them and a `.venv` two folders up |
| `results/20261107_160945/config.json` | Every setting of the run, including the parameter counts |
| `results/20261107_160945/folds.json` | The 90 folds (pretraining, validation, fine-tuning and test trial indices) |
| `results/20261107_160945/results.csv` | One row per model and fold: accuracy, kappa, macro-F1 |
| `results/20261107_160945/confusion_matrices.csv` | The 4 × 4 confusion matrix of every model and fold |
| `results/20261107_160945/predictions/<model>/s<S>_r<R>.npz` | True and predicted labels of every test trial. Use these to recompute any metric |
| `results/20261107_160945/metrics/<model>/` | Per-fold metric files (JSON) |
| `results/20261107_160945/histories/` | Loss and accuracy per epoch, pretraining and fine-tuning, per model and fold |
| `results/20261107_160945/checkpoints/` | Trained weights (PyTorch `.pt`), pretrained and fine-tuned |
| `results/20261107_160945/comparison/` | Pipeline statistics: chance tests, per-subject summary, and paired tests and bootstrap intervals where a run has two models or more (this run has one model, so `paired_tests.csv` is empty; the comparison with the other models is in `../summary/`) |
| `results/20261107_160945/figures/` | Pipeline figures (accuracy per subject, confusion matrices, metric distributions) |
| `results/20261107_160945/run_log*.txt`, `fold_times.jsonl`, `timings.json` | Logs of the 3 shards and the final pass, time per fold |

`CODE_CHANGES_vs_v1.7-full.diff` shows every line of `code/src/` that differs from `v1.7-full`, the version this run was built from.

Glossary and the overview of all versions: `../README.md`.
