"""Evaluation and statistics for BMEN 600 Group 11 (RSNN vs SNN vs EEGNet, BCI IV 2a).

Implements the Midterm Research Plan, Section 4 (metrics, chance, comparisons):
  * per-fold accuracy (primary), Cohen's kappa, macro-F1, 4x4 confusion matrix
  * chance: binomial threshold from the real number of test trials (Combrisson &
    Jerbi 2015, plan ref [10]) AND a 10,000-permutation label test; "above chance"
    only if both pass
  * subject = unit of analysis: 10 repeats averaged -> 9 paired values per model/metric
  * exact two-sided Wilcoxon signed-rank, Holm across the 3 comparisons (per metric)
  * median paired difference with a percentile bootstrap 95% CI over subjects
  * spread: per-subject mean +/- SD over repeats, and mean +/- SD across subjects

Every statistic is a pure function of predictions and labels. Training code only
has to save one record per fold (see `load_predictions` for the file format).

Metrics are written out in NumPy (a few lines each, so they can be explained in
class); test_eval_stats.py checks them against scikit-learn when it is installed.
"""
from __future__ import annotations

import json
import platform
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
    # OPEN TEAM DECISION: level of the chance test ("fold" or "subject").
    # Both are computed; this only marks which one the report treats as primary.
    "chance_level": "fold",
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


def metrics_from_cm(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    tp = np.diag(cm)
    acc = tp.sum() / n
    # Cohen's kappa: (p_o - p_e) / (1 - p_e), p_e from the row and column marginals
    p_e = (cm.sum(axis=1) * cm.sum(axis=0)).sum() / n**2
    kappa = (acc - p_e) / (1 - p_e) if p_e < 1 else float("nan")
    # macro-F1: per-class F1 = 2TP / (2TP + FP + FN), 0 if the class never appears
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"accuracy": float(acc), "kappa": float(kappa), "macro_f1": float(f1.mean())}


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
# 2. Chance level
# --------------------------------------------------------------------------- #
def binomial_threshold(n_trials: int, n_classes: int = 4, alpha: float = 0.05) -> float:
    """Accuracy a classifier must EXCEED to be significant under pure guessing.

    k = binom.ppf(1 - alpha) is the largest count with P(X > k) <= alpha, so the
    rule is `accuracy > k / n` (strict). For n = 288, p = 0.25, alpha = 0.05 this
    is 84/288 = 0.2917, i.e. at least 85 correct trials.
    """
    k = binom.ppf(1 - alpha, n_trials, 1.0 / n_classes)
    return float(k / n_trials)


def permutation_pvalue(y_true, y_pred, n_perm: int = 10_000, seed: int = 0,
                       chunk: int = 1000) -> float:
    """Shuffle the true labels against fixed predictions; one-sided p for accuracy.
    p = (#null >= observed + 1) / (n_perm + 1)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    observed = np.mean(y_true == y_pred)
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perms = rng.permuted(np.tile(y_true, (m, 1)), axis=1)
        null = (perms == y_pred).mean(axis=1)
        hits += int(np.sum(null >= observed - 1e-12))
        done += m
    return (hits + 1) / (n_perm + 1)


def subject_permutation_pvalue(y_true, y_pred_repeats, n_perm: int = 10_000,
                               seed: int = 0, chunk: int = 1000) -> float:
    """Subject-level version: the 10 repeats predict the SAME 288 test trials, so one
    permutation of those 288 labels is applied to every repeat and the statistic is
    the mean accuracy over repeats. (Do not treat 10 x 288 as 2880 independent trials.)"""
    y_true = np.asarray(y_true)
    P = np.asarray(y_pred_repeats)                      # (n_repeats, n_trials)
    observed = np.mean(P == y_true[None, :])
    labels = np.unique(np.concatenate([y_true, P.ravel()]))
    # frac[t, c] = fraction of repeats that predicted class c on trial t
    frac = np.stack([(P == c).mean(axis=0) for c in labels], axis=1)
    lab_idx = np.searchsorted(labels, y_true)
    rng = np.random.default_rng(seed)
    t = np.arange(y_true.size)
    hits, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        perm_idx = rng.permuted(np.tile(lab_idx, (m, 1)), axis=1)
        null = frac[t[None, :], perm_idx].mean(axis=1)
        hits += int(np.sum(null >= observed - 1e-12))
        done += m
    return (hits + 1) / (n_perm + 1)


def chance_assessment(y_true, y_pred, n_classes=4, alpha=0.05, n_perm=10_000, seed=0) -> dict:
    """Both chance checks for one fold. Above chance only if both pass."""
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    thr = binomial_threshold(len(y_true), n_classes, alpha)
    p = permutation_pvalue(y_true, y_pred, n_perm, seed)
    return {"accuracy": acc, "threshold": thr, "p_perm": p,
            "above_chance": bool(acc > thr and p < alpha)}


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
# Input: predictions saved by the training code
# --------------------------------------------------------------------------- #
def load_predictions(folder) -> list[dict]:
    """Read every *.npz in `folder`. One file per (model, subject, repeat) with keys:
        model   : "ANN" | "SNN" | "RSNN"
        subject : int 1..9 (A01..A09)
        repeat  : int 0..9
        seed    : int, the fold seed (the random draw of 5 train / 3 val subjects);
                  must be the same for all three models on a given fold
        y_true  : (288,) int labels of the target's session-2 trials, fixed order
        y_pred  : (288,) int predicted labels, same order
    (PROPOSED format; to be confirmed with the training pipeline.)"""
    recs = []
    for f in sorted(Path(folder).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        recs.append({"model": str(z["model"]), "subject": int(z["subject"]),
                     "repeat": int(z["repeat"]), "seed": int(z["seed"]),
                     "y_true": z["y_true"], "y_pred": z["y_pred"]})
    return recs


def check_folds(records) -> None:
    """All models must cover the same (subject, repeat) folds with identical y_true."""
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


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def evaluate(records, config=None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    labels, alpha = cfg["labels"], cfg["alpha"]
    n_perm, seed = cfg["n_permutations"], cfg["seed"]
    pairs = [tuple(p) for p in cfg["pairs"]]
    check_folds(records)

    models = sorted({r["model"] for r in records}, key=lambda m: (m not in MODELS, m))
    subjects = sorted({r["subject"] for r in records})
    folds, per_subject, subject_means = [], {}, {}

    for m in models:
        per_subject[m] = {}
        subject_means[m] = {k: [] for k in METRICS}
        cms = []
        for s in subjects:
            recs = sorted((r for r in records if r["model"] == m and r["subject"] == s),
                          key=lambda r: r["repeat"])
            vals = {k: [] for k in METRICS}
            for r in recs:
                fm = fold_metrics(r["y_true"], r["y_pred"], labels)
                # seed for the permutation stream is derived from the fold, so reruns match
                ch = chance_assessment(r["y_true"], r["y_pred"], len(labels), alpha, n_perm,
                                       seed=seed + 1000 * s + r["repeat"])
                folds.append({"model": m, "subject": s, "repeat": r["repeat"],
                              "train_seed": r["seed"],
                              **{k: fm[k] for k in METRICS},
                              "threshold": ch["threshold"], "p_perm": ch["p_perm"],
                              "above_chance": ch["above_chance"]})
                cms.append(row_normalize(fm["confusion"]))
                for k in METRICS:
                    vals[k].append(fm[k])
            y_true = recs[0]["y_true"]
            P = np.stack([r["y_pred"] for r in recs])
            p_subj = subject_permutation_pvalue(y_true, P, n_perm, seed=seed + 1000 * s)
            thr = binomial_threshold(len(y_true), len(labels), alpha)
            mean_acc = float(np.mean(vals["accuracy"]))
            per_subject[m][s] = {
                **{k: mean_sd(vals[k]) for k in METRICS},
                "n_repeats": len(recs),
                "chance_subject_level": {"accuracy": mean_acc, "threshold": thr, "p_perm": p_subj,
                                         "above_chance": bool(mean_acc > thr and p_subj < alpha)},
            }
            for k in METRICS:
                subject_means[m][k].append(float(np.mean(vals[k])))
        per_subject[m]["confusion_row_normalized_mean"] = np.mean(cms, axis=0).tolist()

    across = {m: {k: mean_sd(subject_means[m][k]) for k in METRICS} for m in models}
    chance_summary = {
        m: {"folds_above_chance": sum(f["above_chance"] for f in folds if f["model"] == m),
            "n_folds": sum(1 for f in folds if f["model"] == m),
            "subjects_above_chance": sum(per_subject[m][s]["chance_subject_level"]["above_chance"]
                                         for s in subjects),
            "n_subjects": len(subjects)}
        for m in models
    }
    comparisons = compare_models(subject_means, pairs, cfg["n_bootstrap"], seed)

    return {
        "config": cfg,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
        "train_seeds": sorted({(r["subject"], r["repeat"], r["seed"]) for r in records}),
        "subjects": subjects,
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
    ap = argparse.ArgumentParser(description="Evaluate saved predictions.")
    ap.add_argument("pred_dir")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--config", help="JSON file overriding DEFAULT_CONFIG")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    if a.seed is not None:
        cfg["seed"] = a.seed
    res = evaluate(load_predictions(a.pred_dir), cfg)
    save_results(res, a.out)
    for metric, rows in res["comparisons"].items():
        for r in rows:
            print(f"{metric:9s} {r['pair']:10s} median={r['median']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] W={r['W']} "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f} n_eff={r['n_eff']}")
