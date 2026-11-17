"""Prints the numbers used in letter.tex from ../lex-review/latest_summary.json (run 20261113_124400), so every value in
the letter can be checked against its source:  python export_numbers.py"""
import json
from pathlib import Path

S = json.loads((Path(__file__).resolve().parent.parent / "lex-review" / "latest_summary.json").read_text())
print(f"chance threshold {S['chance_threshold']:.4f}")
for m in ["ann", "snn", "rsnn", "rnn", "rsnn_wide"]:
    s = S["summary"][m]
    print(f"{m:10s} folds {S['coverage'][m]['folds']:2d}  acc {s['accuracy']['mean']:.3f} +- {s['accuracy']['sd']:.3f}  "
          f"kappa {s['kappa']['mean']:.3f}  F1 {s['macro_f1']['mean']:.3f}  above {s['subjects_above_binomial']}/{s['n_subjects']}")
print("\nper-subject table for the figure (row,label,snn,rsnn,ann,mid,half)")
for i, t in enumerate([str(x) for x in range(1, 10)] + ["mean"], 1):
    v = {m: (S["summary"][m]["accuracy"]["mean"] if t == "mean" else S["per_subject"][m][t]) for m in ("snn", "rsnn", "ann")}
    lo, hi = min(v.values()), max(v.values())
    label = "Mean" if t == "mean" else f"Subject {t}"
    print(f"{i},{label},{v['snn']:.3f},{v['rsnn']:.3f},{v['ann']:.3f},{(lo + hi) / 2:.4f},{(hi - lo) / 2:.4f}")
print("\npaired tests")
for t in S["tests"]:
    p = f"{t['p_holm']:.3f}" if t["p_holm"] is not None else f"{t['p_raw']:.3f} (uncorrected)"
    print(f"{t['pair']:9s} {t['metric']:9s} diff {t['mean_difference']:+.3f}  first higher {t['first_higher']}/9  p {p}")

import statistics as st  # noqa: E402

print("\nfull summary per model (accuracy median, min, max; kappa and macro-F1 mean +- SD)")
for m in ["ann", "snn", "rsnn", "rnn"]:
    acc = list(S["per_subject"][m].values())
    s = S["summary"][m]
    print(f"{m:5s} acc {st.mean(acc):.3f} +- {st.stdev(acc):.3f}  median {st.median(acc):.3f}  range {min(acc):.3f}-{max(acc):.3f}  "
          f"kappa {s['kappa']['mean']:.3f} +- {s['kappa']['sd']:.3f}  F1 {s['macro_f1']['mean']:.3f} +- {s['macro_f1']['sd']:.3f}")
