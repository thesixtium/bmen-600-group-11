# BMEN 600 Group 11: Can one recurrent spiking layer replace dedicated spatial and temporal filters for cross-session motor imagery EEG decoding?

Final project repository (BMEN 600, Fall 2026, Schulich School of Engineering, University of Calgary). It contains the code,
saved results and documents for the final research paper, [`docs/Final Report/main.pdf`](docs/Final%20Report/main.pdf).

## Contents

1. [Research question](#research-question)
2. [Models compared](#models-compared)
3. [Dataset and access](#dataset-and-access)
4. [File map](#file-map)
5. [Setup](#setup)
6. [Running the analysis](#running-the-analysis)
7. [Where the paper's figures and tables come from](#where-the-papers-figures-and-tables-come-from)
8. [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
9. [Other folders (not part of the final pipeline)](#other-folders-not-part-of-the-final-pipeline)
10. [AI-assisted work](#ai-assisted-work)

---

## Research question

Brain–computer interfaces (BCIs) that decode imagined movements from the electroencephalogram (EEG) could let people with
complex mobility needs control a wheelchair or another assistive device, but low decoding accuracy limits their use. Motor
imagery information is both spatial (which electrodes) and temporal (how the rhythms change during the trial). EEGNet
extracts it with dedicated temporal, spatial and feature-summarizing convolution layers.

**Research question:** can a recurrent spiking neural network (RSNN), which has to extract spatial and temporal features
within a single recurrent layer, decode four-class motor imagery EEG from BCI Competition IV dataset 2a as accurately as
EEGNet when each subject is tested on a separate, later recording session?

The subject, not the trial, is the independent case. Each model is pretrained on other subjects, fine-tuned on the target
subject's session 1 and tested on its session 2, repeated 10 times per subject (90 folds shared by all models). The primary
measure is session-2 accuracy, averaged over repeats to one value per subject. Models are compared with exact Wilcoxon
signed-rank tests (Holm-corrected) and bootstrap 95% confidence intervals over the nine subjects.

The research question changed after the midterm plan. The plan asked whether an RSNN outperforms the converted SNN.
The final paper compares the RSNN with EEGNet and adds non-spiking and wider recurrent controls.

## Models compared

The paper and the code use different names for the same models:

| Paper | Code | Description | Trainable parameters |
|---|---|---|---|
| ANN (EEGNet) | `ann` | EEGNet-4,2 with ELU units | 796 |
| SNN (converted) | `snn` | Each fold's fine-tuned ANN converted to integrate-and-fire neurons, not trained | 776 |
| RNN-p | `rnn` | One all-to-all recurrent ELU layer, 17 units (non-spiking control) | 752 |
| RSNN-p | `rsnn` | One all-to-all recurrent leaky integrate-and-fire (LIF) layer, 17 neurons | 752 |
| RSNN-n | `rsnn_wide` | Same as RSNN-p with 2560 neurons, EEGNet's number of hidden units | 6,622,724 |
| not reported | `rnn_wide` | Same as RNN-p with 2560 units. Every fold diverged to NaN, so no results exist ([`lexi's notes`](rssns_for_mi_EEG/lexi's%20notes)) | — |

Parameter counts are computed in [`models.py`](rssns_for_mi_EEG/src/models.py) and saved in each run's `config.json`.

## Dataset and access

- **Dataset:** BCI Competition IV dataset 2a (Graz data set A; Brunner *et al.*, Graz University of Technology, 2008).
  9 subjects (S1–S9), 2 sessions on different days, 6 runs × 48 trials = 288 trials per session (72 per class), 5,184 trials
  in total. Classes: left hand, right hand, both feet, tongue. 22 EEG and 3 EOG channels at 250 Hz.
- **Source:** [dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf) and
  [competition website](https://www.bbci.de/competition/iv/), accessed through
  [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) as `BNCI2014_001`
  (accessed October 8, 2026).
- **How it is used:** the 22 EEG channels only (EOG excluded). No subject or trial is excluded, and artifact-marked trials
  are kept.
- **How to get it:** nothing has to be downloaded by hand. The first run of the training script
  ([`main.py`](rssns_for_mi_EEG/src/main.py)) calls [`download_data.py`](rssns_for_mi_EEG/src/download_data.py), which has
  MOABB download the recordings into `rssns_for_mi_EEG/data/` and reuse them afterwards. This first run needs an internet
  connection. The analysis script (`make_plots.py`) never downloads, and only reads this cache.
- **Terms:** the data are openly available. Cite the competition review (Tangermann *et al.*, 2012) and credit the recording
  group (Brunner *et al.*, 2008; Naeem *et al.*, 2006). MOABB lists the dataset under CC BY-ND 4.0. **The recordings are not
  redistributed** in this repository: `data/` must stay out of version control. Subjects are identified only by code; do
  not try to link them to other sources.

## File map

```
bmen-600-group-11/
├── README.md                     this file
├── requirements.txt              Python packages (see Setup)
├── tree.txt                      directory listing of the team's working copy
├── .gitignore
├── docs/
│   ├── Final Report/
│   │   ├── main.pdf              FINAL RESEARCH PAPER
│   │   ├── main.tex, references.bib    LaTeX source (IEEEtran)
│   │   └── figures/              figures included by main.tex (copies of the generated figures) + IEEEtran.cls
│   ├── BMEN_600_Midterm_Research_Plan_Group_11.pdf / .docx    midterm research plan
│   ├── Preliminary_Results.docx  draft of the midterm's preliminary-results section
│   └── code_documentation.md     notes on an EARLIER version of the code (see "Other folders")
├── rssns_for_mi_EEG/             ANALYSIS PROJECT: everything that produces the paper's results
│   ├── src/                      Python source (module table below)
│   ├── run_benchmark.slurm       SLURM job for the UCalgary ARC cluster (environment setup + training workers)
│   ├── submit_chain.sh           queues chained 24 h GPU jobs for one run id
│   ├── submit_chain_ann_snn.sh, submit_chain_rsnn.sh, submit_chain_rnn.sh,
│   │   submit_chain_rsnn_wide.sh, submit_chain_rnn_wide.sh      one submission script per model family
│   ├── how_to_run.txt            detailed ARC operating guide (monitoring, reruns, NaN troubleshooting)
│   ├── CHANGELOG.md              last code changes before the final runs
│   ├── lexi's notes              why rnn_wide has no results
│   ├── slurm_*.out               SLURM output of the team's jobs
│   ├── results/
│   │   ├── benchmark/<run id>/   one folder per training run; the paper run is 20261113_124400
│   │   └── data_statistics/      dataset figures + CSVs
│   ├── most_recent_results/      flat copy of every figure of the last analysis + RUN.txt (name of the run)
│   └── data/                     MOABB dataset cache (created on first run, not redistributed)
├── evaluation/                   early prototype of the statistics, synthetic data only
├── notebooks/
│   └── 01_preliminary_results.ipynb   exploratory dataset analysis for the midterm plan
└── literature/
    └── BMEN_600_Midterm.bib      references of the midterm plan
```

`rssns_for_mi_EEG/.idea/` (PyCharm settings) and `rssns_for_mi_EEG/.venv/` (a local virtual environment) are not needed
to run anything.

### Source modules (`rssns_for_mi_EEG/src/`)

| Module | Role |
|---|---|
| [`main.py`](rssns_for_mi_EEG/src/main.py) | **Training entry point.** Loads and caches the data, preprocesses it, builds the 90 folds, prepares the run folder and trains/converts the selected models. No plotting |
| [`make_plots.py`](rssns_for_mi_EEG/src/make_plots.py) | **Analysis entry point.** Tables, statistics, figures and CSVs from a saved run. Never trains |
| [`make_arch.py`](rssns_for_mi_EEG/src/make_arch.py) | Standalone script that draws the architecture figure (paper Fig. 2) |
| [`benchmark_config.py`](rssns_for_mi_EEG/src/benchmark_config.py) | Every hyperparameter, seed and constant, including quick mode |
| [`download_data.py`](rssns_for_mi_EEG/src/download_data.py), [`preprocess_data.py`](rssns_for_mi_EEG/src/preprocess_data.py) | Data loading (MOABB) and preprocessing |
| [`folds.py`](rssns_for_mi_EEG/src/folds.py) | The 90 shared folds and the per-fold data arrays |
| [`models.py`](rssns_for_mi_EEG/src/models.py), [`ann_to_snn.py`](rssns_for_mi_EEG/src/ann_to_snn.py) | EEGNet, RSNN, RNN and parameter counts, plus the ANN → SNN conversion |
| [`training.py`](rssns_for_mi_EEG/src/training.py), [`benchmark.py`](rssns_for_mi_EEG/src/benchmark.py), [`timing.py`](rssns_for_mi_EEG/src/timing.py) | Shared training loop, per-model runners, metrics, saving, timing |
| [`data_statistics.py`](rssns_for_mi_EEG/src/data_statistics.py), [`data_statistics_csv.py`](rssns_for_mi_EEG/src/data_statistics_csv.py) | 13 dataset / data-quality figures and their CSVs |
| [`benchmark_plots.py`](rssns_for_mi_EEG/src/benchmark_plots.py), [`training_plots.py`](rssns_for_mi_EEG/src/training_plots.py) | Accuracy, metric and confusion-matrix figures; loss curves, kept epochs, training stability |
| [`comparison.py`](rssns_for_mi_EEG/src/comparison.py), [`comparison_plots.py`](rssns_for_mi_EEG/src/comparison_plots.py) | Chance tests, paired Wilcoxon + Holm tests, bootstrap CIs and their figures |
| [`finetune_plots.py`](rssns_for_mi_EEG/src/finetune_plots.py), [`subject_analysis.py`](rssns_for_mi_EEG/src/subject_analysis.py), [`csv_export.py`](rssns_for_mi_EEG/src/csv_export.py) | Zero-shot vs. fine-tuned performance, subject difficulty and signal quality, CSVs behind the figures |

## Setup

| Item | Requirement |
|---|---|
| Language | Python 3.10–3.12 (the code uses `int \| None` syntax, and `torch==2.4.0` has no Python 3.13 wheels) |
| Training platform | University of Calgary ARC cluster, partition `gpu-v100`, one NVIDIA V100 GPU per job, SLURM. `run_benchmark.slurm` refuses to run on a CPU |
| Analysis platform | Any computer with the packages below |
| Packages | `torch==2.4.0`, `snntorch`, `moabb`, `mne`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` |
| Exact versions of the paper run | `rssns_for_mi_EEG/results/benchmark/20261113_124400/logs/pip_freeze_<chain>_job<g>.txt` |
| Paper | LaTeX with the IEEEtran class |

**On ARC,** nothing is installed by hand. Each job loads the `python/3.12.5` module, creates a fresh virtual environment in
`/tmp` and installs the packages above.

**On your own computer:**

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch==2.4.0
pip install snntorch moabb mne numpy scipy pandas scikit-learn matplotlib
```

The ARC jobs do not read [`requirements.txt`](requirements.txt); they install the list above. If you install from
`requirements.txt` instead, make sure `torch` and `snntorch` are included. `make_plots.py` needs both because it imports the
model code. The midterm notebook additionally uses `seaborn`.

## Running the analysis

Run every command from `rssns_for_mi_EEG/`. There are two entry points, which are run in this order:

```
1. src/main.py        TRAINING   data -> preprocessing -> 90 folds -> train/convert models -> per-fold results in results/benchmark/<run id>/
2. src/make_plots.py  ANALYSIS   saved run -> tables, statistics, figures, CSVs (never trains)
   src/make_arch.py   FIGURE     architecture diagram (independent of any run)
```

### Protocol, as implemented

| Step | Settings |
|---|---|
| Preprocessing (`preprocess_data.py`) | 22 EEG channels. 4–40 Hz band-pass on each continuous run, resampling to 128 Hz, epochs 0.5–2.5 s after each trial's event annotation (`CUE_OFFSET_S = 0.0`), giving 22 × 256 per trial. No artifact rejection, no baseline correction. Volts converted to µV |
| Folds (`folds.py`) | Seed `2026 + 100·subject + repeat`. Session 1 of 5 other subjects = pretraining (1,440 trials), of the remaining 3 = pretraining validation (864). Target session 1 split 80/20, stratified = fine-tuning train/validation (230/58). Target session 2 = test (288) |
| Training (`benchmark_config.py`, `training.py`) | Adam, learning rate 1e-3, cross-entropy, batch size 16. 500 pretraining epochs (dropout 0.25), then 100 fine-tuning epochs (dropout 0.5). Gradient clipping at a global L2 norm of 1.0. The lowest-validation-loss weights of each phase are kept |
| SNN (`ann_to_snn.py`) | BatchNorm folded, 99.9th-percentile normalization on the target's 230 fine-tuning training trials, 256 steps, prediction = most output spikes |
| RSNN / RNN (`models.py`) | Per-channel input standardization from the pretraining subjects only. One EEG sample per step (256 steps). LIF: β = 0.9, threshold 1, fast-sigmoid surrogate gradient (slope 25) |
| Metrics (`benchmark.py`) | Accuracy, Cohen's κ, macro-F1 and confusion matrix on the 288 session-2 trials |

### Step 1: Train the models (ARC)

```bash
cd ~/bmen-600-group-11/rssns_for_mi_EEG
sed -i 's/\r$//' run_benchmark.slurm submit_chain.sh submit_chain_*.sh     # first time only, after editing on Windows
chmod +x submit_chain.sh submit_chain_*.sh
# Set PROJECT_DIR (and --mail-user) at the top of run_benchmark.slurm for your own ARC account.

./submit_chain_ann_snn.sh 4                        # starts a new experiment and prints its Run ID
BENCH_RUN_ID=<id> ./submit_chain_rsnn.sh 4         # every other chain uses the SAME run id
BENCH_RUN_ID=<id> ./submit_chain_rnn.sh 4
BENCH_RUN_ID=<id> ./submit_chain_rsnn_wide.sh 4
```

Each script queues 4 parallel GPU chains of 4 chained 24-hour jobs (the argument sets the number of jobs). Each job runs 3
workers, so the 90 folds are split into 12 shards. Finished folds are skipped, so resubmitting with the same id resumes a
run. ANN and SNN share one chain because the SNN converts the ANN checkpoints of the same run. Training refuses to add to a
run whose settings differ from the current code. See [`how_to_run.txt`](rssns_for_mi_EEG/how_to_run.txt) for monitoring,
reruns and troubleshooting.

**Smoke test without SLURM** (2 epochs, one fold, not comparable to the protocol; works on a CPU):

```bash
BENCH_QUICK=1 BENCH_TARGETS=1 BENCH_REPEATS=0 BENCH_MODELS=ann,snn python src/main.py
```

`main.py` options are set with environment variables: `BENCH_MODELS` (subset of `ann,snn,rsnn,rnn,rsnn_wide,rnn_wide`),
`BENCH_RUN_ID`, `BENCH_SHARD`/`BENCH_N_SHARDS`, `BENCH_TARGETS`/`BENCH_REPEATS`, and `BENCH_QUICK=1` (quick mode; run id
prefixed `quick_`). `BENCH_PATIENCE` enables optional early stopping, which changes the protocol and was not used. On
Windows PowerShell, set the variables first (for example `$env:BENCH_QUICK="1"`) and then run `python src/main.py`.

Each run folder `results/benchmark/<run id>/` contains `config.json` and `folds.json` (all settings, parameter counts,
folds and seeds). For every model and fold it also contains `checkpoints/`, `histories/`, `metrics/` and `predictions/`
files (`s<S>_r<R>.*`). A metrics file marks a fold as done. The folder also holds `logs/` (worker logs and `pip freeze`
per job) and timing files.

### Step 2: Analyse a saved run

```bash
python src/make_plots.py --list-runs
python src/make_plots.py 20261113_124400 --load-data --finetune-inference      # every stage, as for the paper
```

Running `python src/make_plots.py` with no arguments does the same thing (`DEFAULT_ARGS` at the bottom of the script).
The stages run in this order:

| Stage | Needs | Writes (`<run>` = `results/benchmark/<run id>`) |
|---|---|---|
| `data_statistics` | `--load-data` (dataset cache in `data/`) | `results/data_statistics/*.png` + `*.csv` |
| `benchmark` | per-fold metrics and predictions | `<run>/figures/accuracy_per_subject.png`, `metric_distributions.png`, `confusion_matrices.png` |
| `training` | `histories/` (`logs/` for diverged folds) | `<run>/figures/training_loss_<model>.png`, `training_curves_overlay.png`, `training_kept_epochs.png`, `training_stability.png` |
| `comparison` | per-fold metrics and predictions | `<run>/comparison/*.csv`, `<run>/comparison/figures/*.png` |
| `csv` | the stages above | CSVs with the plotted numbers next to each figure |
| `finetune` | checkpoints + cached `finetune/pretrained_metrics.csv`, or `--finetune-inference` | `<run>/finetune/*` |
| `subjects` | per-fold metrics; `--load-data` for the signal-quality parts | `<run>/subject_analysis/*` |

To rebuild the result tables and figures from the saved per-fold files **without the EEG data**:

```bash
python src/make_plots.py 20261113_124400 --stages benchmark training comparison csv finetune subjects
```

`make_plots.py` first rebuilds `results.csv`, `predictions.csv` and `confusion_matrices.csv` from the per-fold files. It
overwrites the run's figures and tables in place and empties and refills `most_recent_results/`.

### Step 3: Architecture figure

```bash
python src/make_arch.py [output.png]       # default: src/model_architectures.png
```

## Where the paper's figures and tables come from

`<run>` = `rssns_for_mi_EEG/results/benchmark/20261113_124400`. Each figure has a CSV of the same name next to it.

| Paper item | Produced by | Output |
|---|---|---|
| Fig. 1: class-specific band power (topomaps) | `make_plots.py` stage `data_statistics` (`--load-data`) | `rssns_for_mi_EEG/results/data_statistics/class_topomaps.png` |
| Fig. 2: model architectures | `make_arch.py` | `model_architectures.png` |
| Fig. 3: session-2 accuracy per subject, chance markers | stage `comparison` | `<run>/comparison/figures/accuracy_slopes.png` |
| Fig. 4: median training and validation loss | stage `training` | `<run>/figures/training_curves_overlay.png` |
| Fig. 5: zero-shot vs. fine-tuned accuracy | stage `finetune` | `<run>/finetune/finetune_pre_post_aggregate.png` |
| Table I: models, hidden units, parameters | start of training (`models.model_summary`) | `<run>/config.json` → `parameter_counts` |
| Table II: paired differences in accuracy | stage `comparison` | `<run>/comparison/paired_tests.csv` (rows `metric = accuracy`: `median_diff`, `ci_low`, `ci_high`, `n_a_better`/`n_b_better`, `p_holm`) |
| Table III: per-class recall, feet→tongue confusion | stages `benchmark` + `csv` | `<run>/figures/confusion_matrices.csv` (`row_fraction`). "T pred." is the share of all predictions that were tongue, computed from the counts in this file |
| Mean ± SD of accuracy, κ, macro-F1 | stage `comparison` | `<run>/comparison/model_summary.csv` |
| Subjects above chance | stage `comparison` | `<run>/comparison/chance_tests.csv` |
| Kept epochs, loss spikes | stage `training` | `<run>/figures/training_kept_epochs.csv`, `training_stability.csv`, `training_loss_<model>_kept_epochs.csv` |
| Zero-shot accuracy, fine-tuning gains and tests | stage `finetune` | `<run>/finetune/finetune_pre_post_aggregate.csv`, `finetune_gain_aggregate.csv` |
| Kendall's W, variance split, Spearman correlations | stage `subjects` (`--load-data`) | `<run>/subject_analysis/subject_rank_consistency_summary.csv`, `subject_variance_decomposition.csv`, `subject_signal_vs_accuracy_correlations.csv` |

Two numbers in the Results section are not written by a script and were computed from the per-trial predictions in
`<run>/predictions.csv`: the 26% of ANN-misclassified trials that the recurrent models got right, and the 60% / 78%
correct-hand rates.

The paper's figure copies are in [`docs/Final Report/figures/`](docs/Final%20Report/figures/). The same figures are collected
in `rssns_for_mi_EEG/most_recent_results/` after every analysis.

## Reproducibility notes and limitations

- **Seeds.** Fold seed = `2026 + 100 × subject + repeat` (subject draw, fine-tuning split, initialization, shuffling).
  Statistics use seed 2026. PyTorch deterministic algorithms are not enabled, so retraining on a GPU may not give
  bit-identical weights. Re-running the analysis on the saved predictions uses the same seeds.
- **The paper run has no `COMPLETE` marker.** `main.py` writes `COMPLETE` only when every model, including `rnn_wide`, is
  finished. `rnn_wide` diverged, so `20261113_124400` has all 90 folds for `ann`, `snn`, `rsnn`, `rnn` and `rsnn_wide`, but
  no marker.
- **Epoch window.** Epochs start 0.5 s after MOABB's event annotation of each trial (`CUE_OFFSET_S = 0.0`), which the paper
  treats as cue onset.
- **Compute.** The full protocol ran as chains of 24-hour V100 jobs on ARC. Measured times are in each run's `timings.json`
  and `fold_times.jsonl`.
- **Earlier runs.** `results/benchmark/` also holds three earlier runs (`20261027_153619`, `20261105_142914`,
  `20261107_101310`), one of them incomplete. The paper does not use them.

## Other folders (not part of the final pipeline)

- **`evaluation/`:** standalone prototype of the midterm plan's statistics for three models (ANN, SNN, RSNN), demonstrated
  only on **synthetic** predictions. `outputs/synthetic_results.json` is not a project result. Its loader expects a
  different `.npz` layout than the one `benchmark.py` writes, so it does not read benchmark runs. The paper's statistics
  come from `src/comparison.py`. Tests: `cd evaluation && python -m pytest -q test_eval_stats.py`.
- **`notebooks/01_preliminary_results.ipynb`:** midterm exploration with MOABB's default 2–6 s window at 250 Hz (metadata,
  class balance, PSD, band-power topographies). It needs `seaborn`. Its first cell sets a local MOABB download folder that
  must be changed to a folder on your computer.
- **`docs/code_documentation.md`:** notes on an earlier, three-model version of the code. It does not match the final code
  in several places (RSNN batch size 64, no gradient clipping, unstandardized RSNN input, quick mode on). Use the source code
  and this README as the reference.
- **`literature/BMEN_600_Midterm.bib`:** midterm references. The final references are in `docs/Final Report/references.bib`.

## AI-assisted work

Claude (Anthropic) and Microsoft Copilot were used to generate and revise analysis code and to edit text. The team reviewed
all AI-generated code and tested the pipeline with short smoke runs before the full experiments. No AI tool labelled,
transformed or analysed the data directly. Details are in the "AI-Assisted Work" section of the final paper.