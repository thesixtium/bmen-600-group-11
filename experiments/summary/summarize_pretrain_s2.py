"""Session-2 accuracy after pretraining only and after fine-tuning, per run and model.

Reads pretrain_s2_scores/<run>.json (written by score_pretrain_s2.py). Subject is the unit of analysis: the 10 repeats are
averaged into one value per subject, giving 9 paired values per model.
  gain        = fine-tuned - pretrained, per subject
  p (gain)    = exact two-sided Wilcoxon signed-rank test on the 9 gains, not corrected for multiple tests
  above chance = subjects whose mean accuracy is above the binomial threshold binom.isf(0.05, 288, 0.25) / 288
                = 84/288 = 0.2917, defined as in comparison.py:149-154 (binomial part only; the benchmark also needs
                the permutation test, so the fine-tuned counts can be higher than Table 32's)
Writes session2_by_phase.csv, session2_by_phase_per_subject.csv and session2_by_phase_gap.csv.
Usage (from the repository root): python experiments/summary/summarize_pretrain_s2.py
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, wilcoxon

HERE = Path(__file__).resolve().parent
ORDER = ["v1", "v1.1", "v1.2", "v1.3", "v1.4", "v1.6", "v1.7", "v1.1-rnn", "v1.2-rnn", "v1.0-fflif", "v1.7-fflif"]
LABEL = {"ann": "EEGNet", "rsnn": "RSNN", "rnn": "RNN", "fflif": "FF-LIF"}
N_TEST = 288
K = int(binom.isf(0.05, N_TEST, 0.25))  # as comparison.py: accuracy must be strictly above K / n
THRESHOLD = K / N_TEST

rows, subj_rows = [], []
for run in ORDER:
    res = json.loads((HERE / "pretrain_s2_scores" / f"{run}.json").read_text())
    for model, r in res["models"].items():
        assert r["finetune_check"]["folds_with_any_difference"] == 0, (run, model)
        pre = np.array([r["pretrained"]["per_subject"][str(t)] for t in range(1, 10)])
        ft = np.array([r["finetuned"]["per_subject"][str(t)] for t in range(1, 10)])
        gain = ft - pre
        p = wilcoxon(ft, pre, alternative="two-sided", method="exact").pvalue
        rows.append({
            "run": run, "model": LABEL[model],
            "pretrained_mean": round(pre.mean(), 3), "finetuned_mean": round(ft.mean(), 3),
            "gain_mean": round(gain.mean(), 3), "gain_median": round(float(np.median(gain)), 3),
            "gain_min": round(gain.min(), 3), "gain_max": round(gain.max(), 3),
            "subjects_gain_positive": int((gain > 0).sum()), "wilcoxon_p_gain": round(float(p), 4),
            "subjects_above_binomial_pretrained": int((pre > THRESHOLD).sum()),
            "subjects_above_binomial_finetuned": int((ft > THRESHOLD).sum()),
        })
        for t in range(1, 10):
            subj_rows.append({"run": run, "model": LABEL[model], "subject": t,
                              "pretrained": round(pre[t - 1], 4), "finetuned": round(ft[t - 1], 4)})

for name, data in (("session2_by_phase.csv", rows), ("session2_by_phase_per_subject.csv", subj_rows)):
    with open(HERE / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)

print(f"chance threshold: {K}/{N_TEST} = {THRESHOLD:.4f}")
print(f"{'run':11s} {'model':7s} {'pre':>6s} {'ft':>6s} {'gain':>6s} {'med':>6s} {'min':>6s} {'max':>6s} {'n+':>3s} {'p':>7s} {'chP':>3s} {'chF':>3s}")
for r in rows:
    print(f"{r['run']:11s} {r['model']:7s} {r['pretrained_mean']:6.3f} {r['finetuned_mean']:6.3f} {r['gain_mean']:6.3f} "
          f"{r['gain_median']:6.3f} {r['gain_min']:6.3f} {r['gain_max']:6.3f} {r['subjects_gain_positive']:3d} "
          f"{r['wilcoxon_p_gain']:7.4f} {r['subjects_above_binomial_pretrained']:3d} {r['subjects_above_binomial_finetuned']:3d}")

# Where the EEGNet - recurrent gap arises: gap after pretraining, gap after fine-tuning, and the difference of the
# two models' fine-tuning gains (= how much of the final gap fine-tuning adds). Exact two-sided Wilcoxon, 9 subjects.
per = {}
for r in subj_rows:
    per.setdefault((r["run"], r["model"]), []).append((r["pretrained"], r["finetuned"]))
gap_rows = []
for run, a_name, b_run, b_name in [("v1", "EEGNet", "v1", "RSNN"), ("v1", "EEGNet", "v1", "RNN"),
                                   ("v1.6", "EEGNet", "v1.6", "RNN"), ("v1.7", "EEGNet", "v1.7", "RSNN"),
                                   ("v1.7", "EEGNet", "v1.7", "RNN"), ("v1", "RSNN", "v1.0-fflif", "FF-LIF"),
                                   ("v1.7", "RSNN", "v1.7-fflif", "FF-LIF")]:
    a, b = np.array(per[(run, a_name)]), np.array(per[(b_run, b_name)])
    d_pre, d_ft = a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]
    d_gain = d_ft - d_pre
    gap_rows.append({
        "run": run, "comparison": f"{a_name} - {b_name}",
        "gap_pretrained": round(d_pre.mean(), 3), "first_higher_pretrained": int((d_pre > 0).sum()),
        "p_pretrained": round(float(wilcoxon(d_pre, method="exact").pvalue), 4),
        "gap_finetuned": round(d_ft.mean(), 3),
        "gain_difference": round(d_gain.mean(), 3), "first_gains_more": int((d_gain > 0).sum()),
        "p_gain_difference": round(float(wilcoxon(d_gain, method="exact").pvalue), 4),
        "share_of_final_gap_from_finetuning": round(float(d_gain.mean() / d_ft.mean()), 2),
    })
with open(HERE / "session2_by_phase_gap.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gap_rows[0]))
    w.writeheader()
    w.writerows(gap_rows)
print()
for r in gap_rows:
    print(r)
