"""Comparative metrics between the three models (research plan, Section 4).

Implements 'Outcome and comparisons' and 'Avoiding misleading results and showing uncertainty':
  * The subject is the independent case: the 10 repeats are averaged to ONE value per subject, model and metric,
    giving nine paired observations.
  * Chance: every subject x model x metric result is compared with a significance threshold and confirmed by a
    label-permutation test; only results passing both are called above chance.
      - accuracy: binomial threshold from the actual number of test trials (Combrisson & Jerbi 2015) AND permutation.
      - kappa, macro-F1: no analytic threshold exists, so the permutation test alone decides (its threshold is the
        (1 - alpha) quantile of the permutation null).
    The permutation null matches the subject-level mean: in each of the 10 repeats the session-2 labels are shuffled
    against that repeat's predictions, the metric is computed, and the 10 values are averaged (independent shuffles).
  * Differences between models: exact two-sided Wilcoxon signed-rank test on the nine subject means, Holm-corrected
    across the five comparisons of each metric, plus the median paired difference with a bootstrap CI over subjects.
  * Summaries: per-subject mean +/- SD across repeats, and mean +/- SD across subjects.
All three metrics are tested; accuracy is the primary one.

Output (<run_dir>/comparison/): subject_summary.csv, model_summary.csv, chance_tests.csv, paired_differences.csv,
paired_tests.csv, config.json. Figures are made by comparison_plots.py.
"""

import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from benchmark_config import COMPARISON_CFG, N_CLASSES, ComparisonConfig

MODEL_ORDER: tuple[str, ...] = ("ann", "snn", "rsnn", "rnn")  # new models are appended: keeps existing RNG seeds
METRIC_NAMES: tuple[str, ...] = ("accuracy", "kappa", "macro_f1")
PRIMARY_METRIC: str = "accuracy"
# (model a, model b, label): the paired difference is always a - b
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("rsnn", "snn", "RSNN - SNN (research question)"),
    ("rsnn", "ann", "RSNN - ANN (vs baseline)"),
    ("ann", "snn", "ANN - SNN (lost in conversion)"),
    # RNN control (appended so the seeds of the original three comparisons are unchanged)
    ("rnn", "rsnn", "RNN - RSNN (effect of spiking)"),
    ("rnn", "ann", "RNN - ANN (vs baseline)"),
)


# ---------------------------------------------------------------- metrics from confusion matrices (vectorized)


def _metrics_from_confusion(conf: NDArray[np.float64]) -> NDArray[np.float64]:
    """conf: (..., C, C), rows = true, columns = predicted. Returns (..., 3) = accuracy, kappa, macro-F1.

    Same definitions as sklearn's accuracy_score, cohen_kappa_score and f1_score(average='macro', zero_division=0).
    """
    n = conf.sum(axis=(-2, -1))
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    true_n, pred_n = conf.sum(axis=-1), conf.sum(axis=-2)
    acc = tp.sum(axis=-1) / n
    p_e = (true_n * pred_n).sum(axis=-1) / n**2
    kappa = np.divide(acc - p_e, 1 - p_e, out=np.zeros_like(acc), where=(1 - p_e) > 1e-12)
    denom = true_n + pred_n
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return np.stack([acc, kappa, f1.mean(axis=-1)], axis=-1)


def _permuted_metrics(
    y_true: NDArray[np.int64], y_pred: NDArray[np.int64], n_perm: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Metrics of n_perm label shuffles of one fold: (n_perm, 3). Predictions stay fixed, true labels are permuted."""
    pred_onehot = np.eye(N_CLASSES, dtype=np.float32)[y_pred]  # (n, C)
    shuffled = rng.permuted(np.tile(y_true, (n_perm, 1)), axis=1)  # (n_perm, n)
    conf = np.empty((n_perm, N_CLASSES, N_CLASSES))
    for t in range(N_CLASSES):
        conf[:, t, :] = (shuffled == t).astype(np.float32) @ pred_onehot
    return _metrics_from_confusion(conf)


# ---------------------------------------------------------------- statistics helpers


def holm_correction(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values; NaNs are ignored (and stay NaN)."""
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running, m = 0.0, len(order)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def _wilcoxon(diff: NDArray[np.float64]) -> tuple[float, float]:
    """Exact two-sided Wilcoxon signed-rank test on paired differences. Returns (statistic, p); NaN if untestable."""
    if len(diff) < 2 or np.all(diff == 0):
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = stats.wilcoxon(diff, alternative="two-sided", method="exact")
        except (TypeError, ValueError):  # zeros/ties make the exact method unavailable: fall back to scipy's default
            res = stats.wilcoxon(diff, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def _bootstrap_median_ci(diff: NDArray[np.float64], cfg: ComparisonConfig, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile bootstrap CI of the median paired difference, resampling SUBJECTS with replacement."""
    if len(diff) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(diff), size=(cfg.n_bootstrap, len(diff)))
    medians = np.median(diff[idx], axis=1)
    lo, hi = np.quantile(medians, [(1 - cfg.ci_level) / 2, 1 - (1 - cfg.ci_level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------- tables


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per subject x model x metric: mean and SD across the repeats (the mean is the independent observation)."""
    long = df.melt(id_vars=["model", "subject", "repeat"], value_vars=list(METRIC_NAMES), var_name="metric")
    out = long.groupby(["model", "subject", "metric"])["value"].agg(mean="mean", sd_repeats="std", n_repeats="count")
    return out.reset_index()


def model_summary(subjects: pd.DataFrame) -> pd.DataFrame:
    """Per model x metric: mean +/- SD across the subject-level means (one value per subject)."""
    out = subjects.groupby(["model", "metric"])["mean"].agg(
        mean_across_subjects="mean", sd_across_subjects="std", median_across_subjects="median",
        min_subject="min", max_subject="max", n_subjects="count",
    )
    return out.reset_index()


def chance_tests(df: pd.DataFrame, run_dir: Path, cfg: ComparisonConfig) -> pd.DataFrame:
    """Binomial threshold (accuracy) and permutation test of every subject x model x metric result."""
    observed = df.groupby(["model", "subject"])[list(METRIC_NAMES)].mean()
    rows = []
    for (model, subject), group in df.groupby(["model", "subject"]):
        rng = np.random.default_rng([cfg.seed, int(subject), MODEL_ORDER.index(model)])
        null = np.zeros((cfg.n_permutations, len(METRIC_NAMES)))
        for repeat in sorted(group["repeat"]):
            pred = np.load(run_dir / "predictions" / model / f"s{subject}_r{repeat}.npz")
            null += _permuted_metrics(pred["y_true"], pred["y_pred"], cfg.n_permutations, rng)
        null /= len(group)  # null distribution of the subject-level mean over repeats
        n_test = int(group["n_test"].iloc[0])  # the repeats share the same session-2 trials
        binomial_threshold = stats.binom.isf(cfg.alpha, n_test, 1 / N_CLASSES) / n_test
        for k, metric in enumerate(METRIC_NAMES):
            value = float(observed.loc[(model, subject), metric])
            p_perm = (1 + int(np.sum(null[:, k] >= value - 1e-12))) / (cfg.n_permutations + 1)
            passes_perm = bool(p_perm < cfg.alpha)
            passes_binomial = bool(value > binomial_threshold) if metric == "accuracy" else None
            rows.append(
                {
                    "model": model, "subject": subject, "metric": metric, "observed": value, "n_test": n_test,
                    "binomial_threshold": binomial_threshold if metric == "accuracy" else np.nan,
                    "perm_threshold": float(np.quantile(null[:, k], 1 - cfg.alpha)),
                    "perm_null_mean": float(null[:, k].mean()),
                    "perm_p": p_perm, "passes_binomial": passes_binomial, "passes_perm": passes_perm,
                    "above_chance": passes_perm and (passes_binomial is None or passes_binomial),
                }
            )
    return pd.DataFrame(rows)


def paired_tests(df: pd.DataFrame, cfg: ComparisonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired model comparisons on subject-level means: (test table, per-subject differences)."""
    means = df.groupby(["model", "subject"])[list(METRIC_NAMES)].mean()
    test_rows, diff_rows = [], []
    for metric in METRIC_NAMES:
        wide = means[metric].unstack("model")
        family = []
        for a, b, label in COMPARISONS:
            if a not in wide.columns or b not in wide.columns:
                continue
            pair = wide[[a, b]].dropna()
            diff = (pair[a] - pair[b]).to_numpy()
            rng = np.random.default_rng([cfg.seed, METRIC_NAMES.index(metric), len(family)])
            statistic, p = _wilcoxon(diff)
            lo, hi = _bootstrap_median_ci(diff, cfg, rng)
            family.append(
                {
                    "metric": metric, "comparison": f"{a}-{b}", "label": label, "n_subjects": len(diff),
                    "mean_a": float(pair[a].mean()) if len(diff) else np.nan,
                    "mean_b": float(pair[b].mean()) if len(diff) else np.nan,
                    "median_diff": float(np.median(diff)) if len(diff) else np.nan,
                    "ci_low": lo, "ci_high": hi, "n_a_better": int(np.sum(diff > 0)), "n_b_better": int(np.sum(diff < 0)),
                    "wilcoxon_statistic": statistic, "p_raw": p,
                }
            )
            for subject, va, vb in zip(pair.index, pair[a], pair[b]):
                diff_rows.append(
                    {"metric": metric, "comparison": f"{a}-{b}", "subject": subject, "a": va, "b": vb, "diff": va - vb}
                )
        p_holm = holm_correction(np.array([r["p_raw"] for r in family], dtype=float))  # family = 5 comparisons per metric
        for r, ph in zip(family, p_holm):
            r["p_holm"] = float(ph)
            r["significant"] = bool(ph < cfg.alpha)
        test_rows.extend(family)
    return pd.DataFrame(test_rows), pd.DataFrame(diff_rows)


# ---------------------------------------------------------------- driver


def compare_models(run_dir: Path, cfg: ComparisonConfig = COMPARISON_CFG) -> None:
    """Write the comparison tables to <run_dir>/comparison/. Needs results.csv and predictions/ from the benchmark."""
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        print("No results.csv found; run collect_results() first.")
        return
    df = pd.read_csv(results_path)
    n_subjects = df["subject"].nunique()
    if n_subjects < 9:
        print(f"Only {n_subjects} subject(s) in the results (plan: 9); paired tests need at least 2 and have little power below 6.")
    out_dir = run_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = subject_summary(df)
    summary = model_summary(subjects)
    chance = chance_tests(df, run_dir, cfg)
    tests, diffs = paired_tests(df, cfg)

    subjects.to_csv(out_dir / "subject_summary.csv", index=False)
    summary.to_csv(out_dir / "model_summary.csv", index=False)
    chance.to_csv(out_dir / "chance_tests.csv", index=False)
    diffs.to_csv(out_dir / "paired_differences.csv", index=False)
    tests.to_csv(out_dir / "paired_tests.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({"comparison": asdict(cfg), "primary_metric": PRIMARY_METRIC}, indent=1))

    pd.set_option("display.width", 200)
    print("\nMean +/- SD across subjects:")
    print(summary[["model", "metric", "mean_across_subjects", "sd_across_subjects", "n_subjects"]].round(3).to_string(index=False))
    print("\nPaired comparisons (difference = first - second, Holm across the 5 comparisons of each metric):")
    cols = ["metric", "comparison", "n_subjects", "median_diff", "ci_low", "ci_high", "p_raw", "p_holm", "significant"]
    print(tests[cols].round(4).to_string(index=False))
    share = chance.groupby(["model", "metric"])["above_chance"].sum().unstack()
    print(f"\nSubjects above chance (out of {n_subjects}):\n{share.to_string()}")
    print(f"Saved comparison tables to {out_dir}")
