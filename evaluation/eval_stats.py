"""Evaluation and statistics for BMEN 600 Group 11 (RSNN vs SNN vs EEGNet, BCI IV 2a).

Implements the Midterm Research Plan, Section 4 (metrics, chance, comparisons):
  * per-fold accuracy (primary), Cohen's kappa, macro-F1, 4x4 confusion matrix
  * subject = unit of analysis: 10 repeats averaged -> 9 paired values per model/metric
  * chance, PER SUBJECT: the subject-level mean of each
    metric is compared with a label-permutation null (10,000 shuffles). One shuffle
    of the 288 session-2 labels is shared by all 10 repeats, because the repeats
    predict the same trials. Accuracy must also exceed the binomial threshold for
    n = 288 (Combrisson & Jerbi 2015). Kappa and macro-F1 have no binomial
    threshold, so the permutation test alone decides for them.
  * exact two-sided Wilcoxon signed-rank, Holm across the 3 comparisons (per metric)
  * median paired difference with a percentile bootstrap 95% CI over subjects
  * spread: per-subject mean +/- SD over repeats, and mean +/- SD across subjects

Input is a run folder written by the team pipeline (rssns_for_mi_EEG/src/benchmark.py),
read by `load_run`. Every statistic is a pure function of labels and predictions.
"""
from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import binom, wilcoxon

MODELS = ("ANN", "SNN", "RSNN")
# d = first - second. Two-sided tests, so the sign only affects how the CI reads.
# RSNN - SNN answers the research question; ANN - SNN is the conversion loss.
DEFAULT_PAIRS = (("RSNN", "SNN"), ("RSNN", "ANN"), ("ANN", "SNN"))
METRICS = ("accuracy", "kappa", "macro_f1")

DEFAULT_CONFIG = {
    "labels": [0, 1, 2, 3],
    "alpha": 0.05,
    "n_permutations": 10_000,
    "n_bootstrap": 10_000,
    "seed": 0,
    "pairs": [list(p) for p in DEFAULT_PAIRS],
    "chance_level": "subject",  # chance is tested on subject-level means; recorded for the log
}


# --------------------------------------------------------------------------- #
# 1. Per-fold metrics
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred, labels=(0, 1, 2, 3)) -> np.ndarray:
    """Counts; rows = true class, columns = predicted class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    for arr, name in ((y_true, "y_true"), (y_pred, "y_pred")):
        bad = set(np.unique(arr).tolist()) - set(labels)
        if bad:
            raise ValueError(f"{name} contains labels not in {labels}: {sorted(bad)}")
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    np.add.at(cm, ([idx[v] for v in y_true.tolist()], [idx[v] for v in y_pred.tolist()]), 1)
    return cm


def metrics_batch(cm) -> np.ndarray:
    """Accuracy, kappa and macro-F1 from one or many confusion matrices.

    cm has shape (..., C, C); the result has shape (..., 3) in METRICS order.
    The batched form lets the permutation test score thousands of shuffles at once.
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum(axis=(-2, -1))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    true_n, pred_n = cm.sum(axis=-1), cm.sum(axis=-2)
    acc = tp.sum(axis=-1) / n
    # Cohen's kappa: (p_o - p_e) / (1 - p_e), p_e from the row and column marginals
    p_e = (true_n * pred_n).sum(axis=-1) / n**2
    kappa = np.divide(acc - p_e, 1 - p_e, out=np.full_like(acc, np.nan), where=(1 - p_e) > 1e-12)
    # macro-F1: per-class F1 = 2TP / (2TP + FP + FN) = 2TP / (true_n + pred_n), 0 if empty
    denom = true_n + pred_n
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return np.stack([acc, kappa, f1.mean(axis=-1)], axis=-1)


def metrics_from_cm(cm) -> dict:
    return dict(zip(METRICS, (float(v) for v in metrics_batch(cm))))


def fold_metrics(y_true, y_pred, labels=(0, 1, 2, 3)) -> dict:
    """Accuracy, kappa, macro-F1 and the raw confusion matrix for one fold.
    Balanced accuracy is not returned: the test set is balanced, so it equals accuracy."""
    cm = confusion(y_true, y_pred, labels)
    out = metrics_from_cm(cm)
    out["confusion"] = cm
    return out


def row_normalize(cm) -> np.ndarray:
    """Per-class recall view of a confusion matrix (for the figure)."""
    cm = np.asarray(cm, dtype=float)
    return cm / cm.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# 2. Chance level (per subject)
# --------------------------------------------------------------------------- #
def binomial_threshold(n_trials: int, n_classes: int = 4, alpha: float = 0.05) -> float:
    """Accuracy a classifier must EXCEED to be significant under pure guessing.

    k = binom.ppf(1 - alpha) is the largest count with P(X > k) <= alpha, so the
    rule is `accuracy > k / n` (strict). For n = 288, p = 0.25, alpha = 0.05 this
    is 84/288 = 0.2917, i.e. at least 85 correct trials. For a subject mean over
    repeats this threshold is conservative: averaging repeats can only shrink the
    spread of a guessing classifier's score.
    """
    k = binom.ppf(1 - alpha, n_trials, 1.0 / n_classes)
    return float(k / n_trials)


def subject_chance(y_true, y_pred_repeats, n_classes: int = 4, alpha: float = 0.05,
                   n_perm: int = 10_000, seed: int = 0, chunk: int = 500) -> dict:
    """Chance test of one subject x model, for all three metrics.

    y_true          : (n_trials,) session-2 labels, identical for every repeat
    y_pred_repeats  : (n_repeats, n_trials) predictions; a single fold is (1, n_trials)

    Statistic: the metric averaged over repeats. Null: shuffle the true labels ONCE
    per permutation and score every repeat against that same shuffle, then average.
    Shuffling each repeat separately would average away noise that the real repeats
    share (they predict the same trials), giving a null that is too narrow.
    p = (#null >= observed + 1) / (n_perm + 1), one-sided.
    """
    y_true = np.asarray(y_true)
    P = np.atleast_2d(np.asarray(y_pred_repeats))
    n_rep, n = P.shape
    eye = np.eye(n_classes)
    pred_1h = eye[P]                                             # (R, n, C)
    observed = metrics_batch(np.stack([confusion(y_true, p, range(n_classes)) for p in P])).mean(axis=0)

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(METRICS)))
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        shuffled = rng.permuted(np.tile(y_true, (m, 1)), axis=1)    # (m, n): one shuffle per row
        true_1h = eye[shuffled]                                      # (m, n, C)
        # cm[i, r, t, c] = trials with shuffled label t that repeat r predicted as c
        cm = np.einsum("int,rnc->irtc", true_1h, pred_1h, optimize=True)
        null[done:done + m] = metrics_batch(cm).mean(axis=1)
        done += m

    binom_thr = binomial_threshold(n, n_classes, alpha)
    out = {"n_repeats": n_rep, "n_trials": n}
    for k, metric in enumerate(METRICS):
        p = (1 + int(np.sum(null[:, k] >= observed[k] - 1e-12))) / (n_perm + 1)
        res = {"observed": float(observed[k]), "p_perm": p,
               "perm_threshold": float(np.quantile(null[:, k], 1 - alpha)),
               "null_mean": float(null[:, k].mean()), "null_sd": float(null[:, k].std(ddof=1))}
        if metric == "accuracy":
            res["binomial_threshold"] = binom_thr
            res["above_chance"] = bool(observed[k] > binom_thr and p < alpha)
        else:
            res["binomial_threshold"] = None
            res["above_chance"] = bool(p < alpha)
        out[metric] = res
    return out


# --------------------------------------------------------------------------- #
# 3-5. Subject aggregation and paired comparisons
# --------------------------------------------------------------------------- #
def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(pvals)):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_exact(x, y) -> dict:
    """Exact two-sided Wilcoxon signed-rank on paired subject values.

    SciPy's default zero_method='wilcox' drops zero differences, so the effective
    n can drop below 9; it is reported. Ties among |d| make the 'exact' p
    approximate (SciPy still returns it), so ties are flagged too. Differences are
    rounded to 10 decimals so float noise does not create fake non-zeros or non-ties.
    """
    d = np.round(np.asarray(x, float) - np.asarray(y, float), 10)
    n_zero = int(np.sum(d == 0))
    nz = d[d != 0]
    has_ties = bool(len(np.unique(np.abs(nz))) < len(nz))
    if len(nz) == 0:
        return {"W": float("nan"), "p": 1.0, "n_eff": 0, "n_zero": n_zero, "has_ties": False}
    res = wilcoxon(d, alternative="two-sided", method="exact")
    return {"W": float(res.statistic), "p": float(res.pvalue), "n_eff": int(len(nz)),
            "n_zero": n_zero, "has_ties": has_ties}


def bootstrap_median_ci(d, n_boot: int = 10_000, seed: int = 0, ci: float = 95.0) -> dict:
    """Percentile bootstrap CI of the median paired difference over subjects.
    With n = 9 this is coarse; report the observed median alongside it."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(d, size=(n_boot, d.size), replace=True), axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"median": float(np.median(d)), "ci_low": float(lo), "ci_high": float(hi)}


def mean_sd(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan")}


def compare_models(subject_means: dict, pairs=DEFAULT_PAIRS, n_boot=10_000, seed=0) -> dict:
    """subject_means[model][metric] = array of per-subject values (same subject order).
    Returns, per metric, each pair's W, raw p, Holm p, median diff and bootstrap CI."""
    out = {}
    for metric in METRICS:
        rows = []
        for k, (a, b) in enumerate(pairs):
            x = np.asarray(subject_means[a][metric]); y = np.asarray(subject_means[b][metric])
            w = wilcoxon_exact(x, y)
            bs = bootstrap_median_ci(x - y, n_boot, seed + k)
            rows.append({"pair": f"{a}-{b}", **w, **bs})
        adj = holm([r["p"] for r in rows])
        for r, pa in zip(rows, adj):
            r["p_holm"] = float(pa)
        out[metric] = rows
    return out


# --------------------------------------------------------------------------- #
# Input: a run folder of the team pipeline
# --------------------------------------------------------------------------- #
_STEM = re.compile(r"^s(\d+)_r(\d+)$")


def load_run(run_dir) -> list[dict]:
    """Read every fold of a benchmark run folder (rssns_for_mi_EEG/results/benchmark/<run id>).

    Layout written by benchmark.py:
        predictions/<model>/s<S>_r<R>.npz   y_true, y_pred, scores (session-2 test trials)
        metrics/<model>/s<S>_r<R>.json      model, subject, repeat, seed, metrics
        folds.json                          every fold's seed and subject draw
    Model folder names (ann, snn, rsnn) are returned upper-case. The fold seed is
    read from the metrics file and checked against folds.json when it exists.
    """
    run_dir = Path(run_dir)
    fold_seeds = {}
    if (run_dir / "folds.json").exists():
        for f in json.loads((run_dir / "folds.json").read_text())["folds"]:
            fold_seeds[(f["target"], f["repeat"])] = f["seed"]
    recs = []
    for model_dir in sorted(p for p in (run_dir / "predictions").iterdir() if p.is_dir()):
        for f in sorted(model_dir.glob("*.npz")):
            match = _STEM.match(f.stem)
            if not match:
                continue
            subject, repeat = int(match[1]), int(match[2])
            meta = json.loads((run_dir / "metrics" / model_dir.name / f"{f.stem}.json").read_text())
            if (meta["subject"], meta["repeat"]) != (subject, repeat):
                raise ValueError(f"{f}: file name and metrics file disagree on subject/repeat")
            if fold_seeds and fold_seeds.get((subject, repeat)) != meta["seed"]:
                raise ValueError(f"{f}: seed {meta['seed']} does not match folds.json")
            z = np.load(f, allow_pickle=False)
            recs.append({"model": model_dir.name.upper(), "subject": subject, "repeat": repeat,
                         "seed": int(meta["seed"]), "y_true": z["y_true"], "y_pred": z["y_pred"]})
    return recs


def check_folds(records) -> None:
    """Checks the subject-level analysis relies on:
    * all models cover the same (subject, repeat) folds with the same fold seed;
    * all models and all repeats of a subject share one y_true (the same session-2 trials)."""
    keys = {}
    for r in records:
        keys.setdefault(r["model"], {})[(r["subject"], r["repeat"])] = r
    models = sorted(keys)
    ref = keys[models[0]]
    for m in models[1:]:
        if set(keys[m]) != set(ref):
            raise ValueError(f"{m} and {models[0]} do not have the same folds")
        for k in ref:
            if not np.array_equal(keys[m][k]["y_true"], ref[k]["y_true"]):
                raise ValueError(f"y_true differs between {m} and {models[0]} at {k}")
            if keys[m][k]["seed"] != ref[k]["seed"]:
                raise ValueError(f"seed differs between {m} and {models[0]} at {k}")
    first = {}
    for (s, rep), r in sorted(ref.items()):
        if s in first and not np.array_equal(r["y_true"], first[s]):
            raise ValueError(f"subject {s}: repeat {rep} has a different y_true from repeat 0")
        first.setdefault(s, r["y_true"])


def test_set_report(records, labels=(0, 1, 2, 3)) -> dict:
    """Size and class counts of each subject's test set (pipeline check: 288 trials, 72 per class)."""
    out = {}
    for r in records:
        if r["subject"] not in out:
            y = np.asarray(r["y_true"])
            out[r["subject"]] = {"n_trials": int(y.size),
                                 "per_class": [int(np.sum(y == c)) for c in labels]}
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate(records, config=None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    labels, alpha = cfg["labels"], cfg["alpha"]
    n_perm, seed = cfg["n_permutations"], cfg["seed"]
    pairs = [tuple(p) for p in cfg["pairs"]]
    check_folds(records)

    models = sorted({r["model"] for r in records}, key=lambda m: (MODELS.index(m) if m in MODELS else len(MODELS), m))
    subjects = sorted({r["subject"] for r in records})
    folds, per_subject, subject_means = [], {}, {}

    for mi, m in enumerate(models):
        per_subject[m] = {}
        subject_means[m] = {k: [] for k in METRICS}
        cms = []
        for s in subjects:
            recs = sorted((r for r in records if r["model"] == m and r["subject"] == s),
                          key=lambda r: r["repeat"])
            vals = {k: [] for k in METRICS}
            for r in recs:
                fm = fold_metrics(r["y_true"], r["y_pred"], labels)
                folds.append({"model": m, "subject": s, "repeat": r["repeat"],
                              "fold_seed": r["seed"], **{k: fm[k] for k in METRICS}})
                cms.append(row_normalize(fm["confusion"]))
                for k in METRICS:
                    vals[k].append(fm[k])
            # permutation seed derived from (seed, model, subject), so reruns match
            chance = subject_chance(recs[0]["y_true"], np.stack([r["y_pred"] for r in recs]),
                                    len(labels), alpha, n_perm, seed=seed + 1000 * s + 100 * mi)
            per_subject[m][s] = {**{k: mean_sd(vals[k]) for k in METRICS},
                                 "n_repeats": len(recs), "chance": chance}
            for k in METRICS:
                subject_means[m][k].append(float(np.mean(vals[k])))
        per_subject[m]["confusion_row_normalized_mean"] = np.mean(cms, axis=0).tolist()

    across = {m: {k: mean_sd(subject_means[m][k]) for k in METRICS} for m in models}
    chance_summary = {
        m: {k: sum(per_subject[m][s]["chance"][k]["above_chance"] for s in subjects) for k in METRICS}
        | {"n_subjects": len(subjects)}
        for m in models
    }
    comparisons = compare_models(subject_means, pairs, cfg["n_bootstrap"], seed)

    return {
        "config": cfg,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "fold_seeds": sorted({(r["subject"], r["repeat"], r["seed"]) for r in records}),
        "subjects": subjects,
        "test_sets": test_set_report(records, labels),
        "folds": folds,
        "per_subject": per_subject,
        "subject_means": subject_means,
        "across_subjects": across,
        "chance_summary": chance_summary,
        "comparisons": comparisons,
    }


def save_results(results: dict, path) -> None:
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(results, indent=2, default=default))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a benchmark run folder.")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    if a.seed is not None:
        cfg["seed"] = a.seed
    res = evaluate(load_run(a.run_dir), cfg)
    save_results(res, a.out)
    for metric, rows in res["comparisons"].items():
        for r in rows:
            print(f"{metric:9s} {r['pair']:10s} median={r['median']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] W={r['W']} "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f} n_eff={r['n_eff']}")
