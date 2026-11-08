"""Accuracy, Cohen's kappa, macro-F1, macro specificity and macro precision for every model of the full runs
(v1 to v1.4, v1.6-full, v1.7-full, and the control runs v1.1-rnn, v1.2-rnn, v1.0-fflif, v1.7-fflif).

Session 2 (test), 90 folds per model. Per fold the metrics come from the 4 x 4 confusion matrix saved in
metrics/<model>/s<S>_r<R>.json (rows = true class, columns = predicted class); kappa and macro-F1 are read from the
same file. As in the pipeline, a subject's value is the mean of its 10 repeats and the table value is the mean (and SD)
over the 9 subjects. "Above chance" is read from the run's comparison/chance_tests.csv (accuracy). Models that a
version copied from v1 are reported once, under the run that trained them.
The test set is balanced (72 trials per class), so macro sensitivity (mean recall) equals accuracy.

Usage (from the repository root): python experiments/summary/compute_metrics.py  -> prints a table and writes
experiments/summary/metrics_by_version.csv
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # experiments/
RUNS = {
    "v1": "v1/results/20261029_141407",
    "v1.1": "v1.1/results/20261031_164232",
    "v1.2": "v1.2/results/20261101_191212",
    "v1.3": "v1.3/results/20261102_163908",
    "v1.4": "v1.4/results/20261103_175004",
    "v1.6": "v1.6-full/results/20261107_085030",
    "v1.7": "v1.7-full/results/20261106_144835",
    "v1.1-rnn": "v1.1-rnn/results/20261106_144857",
    "v1.2-rnn": "v1.2-rnn/results/20261107_093450",
    "v1.0-fflif": "v1.0-fflif/results/20261107_160943",
    "v1.7-fflif": "v1.7-fflif/results/20261107_160945",
}
# (version, model) pairs that the version trained; the rest were copied from v1
TRAINED = [("v1", "ann"), ("v1", "snn"), ("v1", "rsnn"), ("v1", "rnn"), ("v1.1", "rsnn"), ("v1.2", "rsnn"),
           ("v1.3", "rsnn"), ("v1.3", "rnn"), ("v1.4", "rsnn"), ("v1.4", "rnn"),
           ("v1.6", "ann"), ("v1.6", "snn"), ("v1.6", "rnn"),                      # v1.6-full: no RSNN
           ("v1.7", "ann"), ("v1.7", "snn"), ("v1.7", "rsnn"), ("v1.7", "rnn"),
           ("v1.1-rnn", "rnn"), ("v1.2-rnn", "rnn"),                               # RNN with the v1.1 / v1.2 change
           ("v1.0-fflif", "fflif"), ("v1.7-fflif", "fflif")]                        # recurrence-free control
METRICS = ("accuracy", "kappa", "macro_f1", "specificity", "precision")


def fold_metrics(m: dict) -> dict:
    c = np.array(m["confusion"], dtype=float)
    tp = np.diag(c)
    pred = c.sum(0)
    true = c.sum(1)
    tn = c.sum() - true - pred + tp
    spec = tn / (tn + (pred - tp))
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)  # a class never predicted counts as 0
    return {"accuracy": tp.sum() / c.sum(), "kappa": m["kappa"], "macro_f1": m["macro_f1"],
            "specificity": spec.mean(), "precision": prec.mean()}


def main() -> None:
    rows = []
    for version, model in TRAINED:
        run = ROOT / RUNS[version]
        per_subject = {k: [] for k in METRICS}
        folds = [json.loads(f.read_text()) for f in (run / "metrics" / model).glob("*.json")]
        assert len(folds) == 90, (version, model, len(folds))
        for s in range(1, 10):
            fm = [fold_metrics(f) for f in folds if f["subject"] == s]
            for k in METRICS:
                per_subject[k].append(np.mean([x[k] for x in fm]))
        with (run / "comparison" / "chance_tests.csv").open() as fh:
            above = sum(r["above_chance"] == "True" for r in csv.DictReader(fh)
                        if r["model"] == model and r["metric"] == "accuracy")
        row = {"version": version, "model": model, "above_chance_accuracy": above}
        for k in METRICS:
            v = np.array(per_subject[k])
            row[k], row[k + "_sd"] = round(float(v.mean()), 3), round(float(v.std(ddof=1)), 3)
        rows.append(row)
    out = ROOT / "summary/metrics_by_version.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['version']:5} {r['model']:5} " + "  ".join(f"{k} {r[k]:.3f}±{r[k + '_sd']:.3f}" for k in METRICS)
              + f"  above {r['above_chance_accuracy']}/9")


if __name__ == "__main__":
    main()
