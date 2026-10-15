# BMEN 600 Group 11: Recurrent spiking networks for motor imagery EEG

Midterm research plan repository (BMEN 600, Fall 2026). Team: Pahan, Amir, Lex.

## Research question

Does a recurrent spiking neural network (RSNN) outperform a feedforward spiking neural network (SNN) at decoding motor imagery EEG from BCI Competition IV dataset 2a, under a subject-level, cross-session evaluation?

Three models are compared on identical folds: **ANN** (EEGNet), **SNN** (the trained EEGNet converted to a spiking network) and **RSNN** (one trained recurrent layer of LIF neurons, parameter-matched to the SNN). The primary measure is accuracy on each target subject's held-out session 2; the subject is the independent unit (9 paired observations). Full plan: [`docs/`](docs/).

## Project status

| Component | Status |
|---|---|
| Dataset loading, class balance, spectral and band-power exploration | Done (`notebooks/01_preliminary_results.ipynb`) |
| Evaluation code (metrics, chance tests, Wilcoxon + Holm, bootstrap) | Done, tested on **synthetic** predictions only (`evaluation/`) |
| Preprocessing and fold generation, EEGNet, conversion, RSNN, training on 90 folds | Not started (see project plan in the midterm PDF) |

**No model results exist yet.** Everything under `evaluation/outputs/` is synthetic and exists only to check that the statistics code recovers known effects.

## Dataset and access

- **Dataset:** BCI Competition IV dataset 2a (Graz): 9 subjects, 2 sessions each, 4 classes (left hand, right hand, feet, tongue), 22 EEG channels at 250 Hz, 288 trials per session.
- **Source:** [BCI Competition IV, dataset 2a description](https://www.bbci.de/competition/iv/desc_2a.pdf), accessed here through [MOABB](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html) (`BNCI2014_001`). Date accessed: October 8, 2026.
- **How to get it:** nothing to download by hand. MOABB downloads the data the first time a notebook calls `paradigm.get_data(...)`. Raw files are stored in `data/` (git-ignored) and **must not be committed or redistributed**.
- **Terms:** openly available. Credit the recording groups and cite Naeem et al., J. Neural Eng., 3(3), 2006 (see the research plan, Section 3). Subjects are coded A01-A09; do not try to link them to other sources.

## File map

```
bmen-600-group-11/
├── README.md                     this file
├── requirements.txt              Python packages
├── .gitignore
├── docs/
│   ├── BMEN_600_Midterm_Research_Plan_Group_11.pdf    submitted midterm plan
│   ├── BMEN_600_Midterm_Research_Plan_Group_11.docx   editable source
│   └── Preliminary_Results.docx                       draft of the preliminary results section
├── literature/
│   └── BMEN_600_Midterm.bib      references used in the plan
├── notebooks/
│   └── 01_preliminary_results.ipynb   CURRENT PROGRESS CHECK: load data, class balance, PSD, band-power maps
├── results/
│   └── preliminary/
│       └── noisy_trials.csv      per-subject indices of noisy trials
├── evaluation/
│   ├── eval_stats.py             metrics, chance tests, Wilcoxon, Holm, bootstrap
│   ├── test_eval_stats.py        unit and smoke tests
│   ├── evaluation.ipynb          walkthrough of every statistic on synthetic data
│   └── outputs/
│       └── synthetic_results.json    SYNTHETIC output, not project results
└── data/                         MOABB download location (git-ignored, created on first run)
```

Planned (not yet in the repo): `src/` for preprocessing, fold generation, ANN/SNN/RSNN code and training scripts; `results/` for per-fold prediction files.

## Setup

- **Platform:** local Jupyter (JupyterLab or VS Code) on Python 3.10 or newer. Final training will run on the University of Calgary ARC Cluster.
- **Install:**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- **Main packages:** MOABB, MNE, NumPy, SciPy, scikit-learn, pandas, matplotlib, seaborn. PyTorch and snnTorch will be added when the model code lands.

## Reproduce the current progress check

1. Complete the setup above and start Jupyter from the repository root.
2. Open `notebooks/01_preliminary_results.ipynb`.
3. In the first code cell, set the MOABB download directory to this repo's `data/` folder (the notebook currently contains a path from one team member's machine):

   ```python
   from pathlib import Path
   moabb.set_download_dir(str(Path("..").resolve() / "data"))
   ```
4. Run all cells. The first run downloads the dataset, so it takes longer and needs an internet connection.
5. Check against the expected output:
   - `X shape: (5184, 22, 1001)` (9 subjects x 2 sessions x 6 runs x 48 trials, 22 channels, 1,001 samples)
   - 1,296 trials in each of the four classes (25% each)
   - the grand-average PSD figure (peak near 10 Hz, 50 Hz line-noise peak) and the mu/beta band-power difference topographies, which are Fig. 2 of the research plan.

### Evaluation code check (synthetic data)

```bash
cd evaluation
python -m pytest -q test_eval_stats.py     # all tests should pass
jupyter notebook evaluation.ipynb          # run all cells; rewrites outputs/synthetic_results.json
```

## AI-assisted work

Microsoft Copilot helped write the code that generates the preliminary results, and Claude helped with text clarity and with the evaluation code and its documentation. Team members reviewed all AI-assisted code. Details are in the AI-Assisted Work section of the research plan.

## Authors

Pahan (preliminary-results data and section), Amir (literature review, evaluation code), Lex (research plan and repository). Per-member contributions are in the research plan.