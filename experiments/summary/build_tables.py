"""Tables 31, 32 and 33 from the saved run files in experiments/. No number is typed by hand.

Table 31: one row per version (v1 to v1.7, plus the main branch 8549d78), all four models.
  Left block : session 2 (test), full run, 90 folds, mean over 9 subjects (subject = mean of 10 repeats). Preliminary.
  Right block: session 1, target's 20 % fine-tuning validation split, 9 development folds (repeat 0), saved fine-tuned
               checkpoints scored by score_saved.py (session1_scores/<run>.json). The v1.6 RSNN has no full run: its
               cell is the v1.6 screen value (v1.6/results/summary.json).
  A model that a version did not retrain is shown with the run it comes from, e.g. "0.569 (v1)".
Table 32: accuracy, kappa, macro-F1, macro specificity, macro precision and subjects above chance (binomial and
  permutation, accuracy) for every model that a full run trained, session 2.
Table 33: recurrence-free control (FF-LIF = the RSNN without its recurrent matrix, widened to the same parameter
  budget) against the RSNN of the same version, v1 and v1.7: both sessions, median paired difference over the 9
  subjects, subjects where the RSNN is better, exact two-sided Wilcoxon signed-rank p (one test per version, not
  corrected).

Usage (from the repository root):
    python experiments/summary/build_tables.py   -> prints the tables, writes table31.csv, table32.csv, table33.csv
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent      # experiments/summary/
EXP = HERE.parent                           # experiments/

# run name -> run folder, relative to experiments/
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

MODELS = ("ann", "snn", "rsnn", "rnn")
LABEL = {"ann": "EEGNet (ANN)", "snn": "Converted SNN", "rsnn": "RSNN", "rnn": "RNN control",
         "fflif": "FF-LIF (no recurrence)"}
# Table 33: version -> (run of its RSNN, run of its FF-LIF)
FFLIF = {"v1": ("v1", "v1.0-fflif"), "v1.7": ("v1.7", "v1.7-fflif")}


def base(run: str) -> str:
    """Version a run belongs to: 'v1.1-rnn' -> 'v1.1', 'v1.0-fflif' -> 'v1'."""
    return run.replace("-rnn", "").replace("-fflif", "").replace("v1.0", "v1")
# version -> model -> run that trained it (None = not run). Versions reuse the models they did not change.
SOURCES = {
    "v1": {"ann": "v1", "snn": "v1", "rsnn": "v1", "rnn": "v1"},
    "v1.1": {"ann": "v1", "snn": "v1", "rsnn": "v1.1", "rnn": "v1.1-rnn"},
    "v1.2": {"ann": "v1", "snn": "v1", "rsnn": "v1.2", "rnn": "v1.2-rnn"},
    "v1.3": {"ann": "v1", "snn": "v1", "rsnn": "v1.3", "rnn": "v1.3"},
    "v1.4": {"ann": "v1", "snn": "v1", "rsnn": "v1.4", "rnn": "v1.4"},
    "v1.6": {"ann": "v1.6", "snn": "v1.6", "rsnn": None, "rnn": "v1.6"},
    "v1.7": {"ann": "v1.7", "snn": "v1.7", "rsnn": "v1.7", "rnn": "v1.7"},
    "main": {"ann": "v1", "snn": "v1", "rsnn": "v1.1", "rnn": "v1.1-rnn"},
}
CHANGES = {
    "v1": "Group code with two local changes (chance test, quick test mode). All 4 models",
    "v1.1": "Batch size 16 for the RSNN and the RNN (on v1)",
    "v1.2": "Readout on 4 time windows for the RSNN and the RNN, 14 neurons (on v1)",
    "v1.3": "Mu and beta frequency front end, 44 inputs, 12 neurons (on v1). RSNN, RNN",
    "v1.4": "Gradient clipping 1.0 and capped initial recurrent weights (on v1.3). RSNN, RNN",
    "v1.5": "No new model: measures v1.4 after pretraining and after fine-tuning; fine-tuning-only arm",
    "v1.6": "Per-subject input normalization (on v1.4). EEGNet, SNN, RNN; RSNN session-1 screen only",
    "v1.7": "Pooled pretraining split, all 8 other subjects 80 / 20 (on v1.4). All 4 models",
    "main": "Group repository main branch (8549d78): batch size 16 for the RSNN and the RNN (= v1.1). "
            "Its neuron-matched RSNN and RNN (2,560 neurons) are not run",
}
SCREEN_V16 = EXP / "v1.6" / "results" / "summary.json"


def run_dir(name: str) -> Path | None:
    """The run's folder, or None if it is not in experiments/."""
    p = EXP / RUNS[name]
    return p if p.exists() else None


def fold_metrics(m: dict) -> dict:
    """Per-fold metrics from the 4 x 4 confusion matrix (rows = true, columns = predicted); kappa and F1 as saved."""
    c = np.array(m["confusion"], dtype=float)
    tp, pred, true = np.diag(c), c.sum(0), c.sum(1)
    tn = c.sum() - true - pred + tp
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)  # a class never predicted counts as 0
    return {"accuracy": tp.sum() / c.sum(), "kappa": m["kappa"], "macro_f1": m["macro_f1"],
            "specificity": float(np.mean(tn / (tn + pred - tp))), "precision": float(prec.mean())}


METRICS = ("accuracy", "kappa", "macro_f1", "specificity", "precision")


def session2(name: str, model: str) -> dict:
    """Subject means (9) per metric and subjects above chance, or {'n_done': k} while the run is not complete."""
    rd = run_dir(name)
    if rd is None:
        return {"n_done": 0, "queued": True}
    files = sorted((rd / "metrics" / model).glob("*.json"))
    chance_csv = rd / "comparison" / "chance_tests.csv"
    if len(files) < 90 or not (rd / "COMPLETE").exists() or not chance_csv.exists():
        return {"n_done": len(files)}
    folds = [json.loads(f.read_text()) for f in files]
    out = {"per_subject": {}}
    for k in METRICS:
        out["per_subject"][k] = [float(np.mean([fold_metrics(f)[k] for f in folds if f["subject"] == s])) for s in range(1, 10)]
    with chance_csv.open() as fh:
        out["above"] = sum(r["above_chance"] == "True" for r in csv.DictReader(fh)
                           if r["model"] == model and r["metric"] == "accuracy")
    return out


def session1(name: str, model: str) -> float | None:
    p = HERE / "session1_scores" / f"{name}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text()).get(model)
    return rec["mean"] if rec and rec["n_subjects"] == 9 else None


def fmt(x: float) -> str:
    return f"{x:.3f}".replace("-", "−")


def cell_s2(version: str, model: str, cache: dict) -> tuple[str, float | None]:
    src = SOURCES[version][model]
    if src is None:
        return "not run", None
    r = cache.setdefault((src, model), session2(src, model))
    if "n_done" in r:
        return ("queued" if r.get("queued") else f"running ({r['n_done']} of 90)"), None
    v = float(np.mean(r["per_subject"]["accuracy"]))
    return fmt(v) + ("" if src == version else f" ({base(src)})"), v


def cell_s1(version: str, model: str) -> str:
    src = SOURCES[version][model]
    if src is None:  # v1.6 RSNN: only the screen exists
        if version == "v1.6" and model == "rsnn" and SCREEN_V16.exists():
            return fmt(json.loads(SCREEN_V16.read_text())["models"]["rsnn"]["ft_ft_val"]["mean_new"]) + " (screen)"
        return "not run"
    v = session1(src, model)
    if v is None:
        return "queued" if run_dir(src) is None else "running"
    return fmt(v) + ("" if src == version else f" ({base(src)})")


def build() -> tuple[str, str, list, list]:
    cache: dict = {}
    rows31, html31 = [], []
    for version in ("v1", "v1.1", "v1.2", "v1.3", "v1.4", "v1.5", "v1.6", "v1.7", "main"):
        label = "main (8549d78)" if version == "main" else version
        if version == "v1.5":
            cells = ["—"] * 6 + ["as v1.4"] * 4
        else:
            s2 = {m: cell_s2(version, m, cache) for m in MODELS}
            diff, above = "—", "—"
            rs, sn = SOURCES[version]["rsnn"], SOURCES[version]["snn"]
            if s2["rsnn"][1] is not None and s2["snn"][1] is not None:
                a = np.array(cache[(rs, "rsnn")]["per_subject"]["accuracy"])
                b = np.array(cache[(sn, "snn")]["per_subject"]["accuracy"])
                diff = fmt(float(np.median(a - b)))
                above = f"{cache[(rs, 'rsnn')]['above']} of 9"
            elif s2["rsnn"][0].startswith("running") or s2["snn"][0].startswith("running"):
                diff = above = "running"
            cells = [s2[m][0] for m in MODELS] + [diff, above] + [cell_s1(version, m) for m in MODELS]
        rows31.append({"version": label, **dict(zip(
            ["s2_ann", "s2_snn", "s2_rsnn", "s2_rnn", "s2_rsnn_minus_snn_median", "s2_rsnn_above_chance",
             "s1_ann", "s1_snn", "s1_rsnn", "s1_rnn"], cells))})
        tds = "".join(f'<td class="num">{c}</td>' for c in cells)
        html31.append(f"    <tr><td><b>{label}</b></td><td>{CHANGES[version]}</td>{tds}</tr>")

    rows32, html32 = [], []
    trained = [("ann", "v1"), ("ann", "v1.6"), ("ann", "v1.7"), ("snn", "v1"), ("snn", "v1.6"), ("snn", "v1.7"),
               ("rsnn", "v1"), ("rsnn", "v1.1"), ("rsnn", "v1.2"), ("rsnn", "v1.3"), ("rsnn", "v1.4"), ("rsnn", "v1.7"),
               ("rnn", "v1"), ("rnn", "v1.1-rnn"), ("rnn", "v1.2-rnn"), ("rnn", "v1.3"), ("rnn", "v1.4"), ("rnn", "v1.6"),
               ("rnn", "v1.7"), ("fflif", "v1.0-fflif"), ("fflif", "v1.7-fflif")]
    for model, src in trained:
        r = cache.setdefault((src, model), session2(src, model))
        version = base(src)
        if "n_done" in r:
            vals = ["queued" if r.get("queued") else f"running ({r['n_done']} of 90)"] + ["—"] * 5
            rows32.append({"model": model, "version": version, "status": vals[0]})
        else:
            means = {k: float(np.mean(r["per_subject"][k])) for k in METRICS}
            sds = {k: float(np.std(r["per_subject"][k], ddof=1)) for k in METRICS}
            vals = [fmt(means[k]) for k in METRICS] + [f"{r['above']} of 9"]
            rows32.append({"model": model, "version": version, "above_chance_accuracy": r["above"],
                           **{k: round(means[k], 3) for k in METRICS}, **{k + "_sd": round(sds[k], 3) for k in METRICS}})
        tds = "".join(f'<td class="num">{v}</td>' for v in vals)
        html32.append(f"    <tr><td>{LABEL[model]}</td><td>{version}</td>{tds}</tr>")

    rows33, html33 = [], []
    for version, (rs, ff) in FFLIF.items():
        r, f = cache.setdefault((rs, "rsnn"), session2(rs, "rsnn")), cache.setdefault((ff, "fflif"), session2(ff, "fflif"))
        s1 = [cell_s1_run(rs, "rsnn"), cell_s1_run(ff, "fflif")]
        if "n_done" in f:
            status = "queued" if f.get("queued") else f"running ({f['n_done']} of 90)"
            cells = [fmt(np.mean(r["per_subject"]["accuracy"])), status, "—", "—", "—", "—"] + s1
            rows33.append({"version": version, "status": status})
        else:
            a, b = np.array(r["per_subject"]["accuracy"]), np.array(f["per_subject"]["accuracy"])
            p = float(stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue)
            cells = [fmt(a.mean()), fmt(b.mean()), fmt(float(np.median(a - b))), f"{int((a > b).sum())} of 9",
                     f"{p:.3f}", f"{f['above']} of 9"] + s1
            rows33.append({"version": version, "s2_rsnn": round(float(a.mean()), 4), "s2_fflif": round(float(b.mean()), 4),
                           "s2_rsnn_minus_fflif_median": round(float(np.median(a - b)), 4),
                           "rsnn_better_subjects": int((a > b).sum()), "wilcoxon_p_exact": round(p, 4),
                           "fflif_above_chance": f["above"]})
        rows33[-1].update({"s1_rsnn": s1[0], "s1_fflif": s1[1]})
        tds = "".join(f'<td class="num">{c}</td>' for c in cells)
        html33.append(f"    <tr><td><b>{version}</b></td>{tds}</tr>")
    return "\n".join(html31), "\n".join(html32), "\n".join(html33), rows31, rows32, rows33


def cell_s1_run(run: str, model: str) -> str:
    """Session-1 cell of one run (Table 33)."""
    v = session1(run, model)
    if v is None:
        return "queued" if run_dir(run) is None else "running"
    return fmt(v)


def main() -> None:
    t31, t32, t33, rows31, rows32, rows33 = build()
    for path, rows in ((HERE / "table31.csv", rows31), (HERE / "table32.csv", rows32), (HERE / "table33.csv", rows33)):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    for r in rows31:
        print(" | ".join(str(v) for v in r.values()))
    print()
    for r in rows32:
        print(" | ".join(str(v) for k, v in r.items() if not k.endswith("_sd")))
    print()
    for r in rows33:
        print(" | ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
