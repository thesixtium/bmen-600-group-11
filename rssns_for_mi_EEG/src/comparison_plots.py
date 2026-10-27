"""Comparison figures, in the style of benchmark_plots.py (Agg backend, 'How to read / Takeaways' caption under each
figure, 150 dpi). Reads the tables written by comparison.compare_models() and saves to <run_dir>/comparison/figures/.

The subject is the independent unit: every mark is a subject-level mean (over the 10 repeats), never a single fold.
Accuracy is the primary metric (research plan, Section 4); kappa and macro-F1 are shown alongside.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from benchmark_plots import CHANCE_ACCURACY, METRICS, MODEL_COLORS, MODEL_LABELS
from comparison import COMPARISONS, METRIC_NAMES, MODEL_ORDER, PRIMARY_METRIC

METRIC_TITLES: dict[str, str] = {m: (t + " (primary)" if m == PRIMARY_METRIC else t) for m, t, _ in METRICS}
METRIC_YLIM: dict[str, tuple[float, float]] = {m: lim for m, _, lim in METRICS}

NOTES: dict[str, tuple[str, str]] = {
    "accuracy_slopes.png": (
        "x = model, y = accuracy on session 2. Each gray line is one target subject (mean over its 10 repeats). A filled "
        "marker passed BOTH chance checks (binomial threshold for the actual number of test trials, dotted line, and the "
        "10,000-permutation label test); a hollow marker did not. The diamond with bars is the mean +/- SD across "
        "subjects. The dashed line is nominal chance (25%).",
        "Parallel lines mean a model change helps or hurts every subject alike; crossing lines mean the effect depends "
        "on the subject. Hollow markers are results that cannot be told apart from chance. Compare ANN to SNN for the "
        "accuracy lost in conversion and SNN to RSNN for the research question.",
    ),
    "paired_differences.png": (
        "One column per comparison (difference = first minus second model), one row per metric. Each dot is one subject's "
        "paired difference of subject means (nine per panel). The solid line is the median, the shaded band its bootstrap "
        "95% confidence interval over subjects, and the dashed line marks zero (no difference). Titles give the "
        "Holm-corrected p-value of the exact two-sided Wilcoxon signed-rank test (5 comparisons per metric). Rows have "
        "their own y-axis.",
        "A band that excludes zero together with p(Holm) < 0.05 supports a real difference between the models. Dots mostly "
        "on one side of zero but a band touching zero mean a consistent but small or uncertain effect. With nine subjects "
        "the smallest possible exact two-sided p is 2/512 (about 0.004), and the Holm correction raises it further, so "
        "only consistent effects can reach significance. RSNN vs SNN differs in more than recurrence; RNN vs RSNN isolates the spiking "
        "neuron model (same topology and size), and RNN vs ANN isolates the recurrent architecture.",
    ),
    "chance_tests.png": (
        "One panel per metric; rows = models, columns = target subjects, cell = subject-level mean over the 10 repeats. "
        "A star marks a result that passed the chance checks (accuracy: binomial threshold and permutation test; "
        "kappa and macro F1: permutation test, since they have no analytic threshold).",
        "Cells without a star are not distinguishable from chance for that subject and model. Chance is 0.25 for accuracy, "
        "0 for kappa. A model that is above chance for few subjects has not shown it can decode motor imagery reliably "
        "across sessions, however good its mean looks.",
    ),
}


def _save(fig: Figure, out_dir: Path, name: str) -> None:
    read, takeaway = NOTES[name]
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {read}", width) + "\n\n" + textwrap.fill(f"Takeaways: {takeaway}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _accuracy_slopes(subjects: pd.DataFrame, chance: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    acc = subjects[subjects["metric"] == PRIMARY_METRIC].pivot(index="subject", columns="model", values="mean")[models]
    passed = chance[chance["metric"] == PRIMARY_METRIC].pivot(index="subject", columns="model", values="above_chance")[models]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(models))
    for s in acc.index:
        ax.plot(x, acc.loc[s].to_numpy(), color="gray", alpha=0.45, lw=1)
        ax.text(x[-1] + 0.05, acc.loc[s].to_numpy()[-1], f"S{s}", fontsize=7, va="center", color="gray")
    for i, model in enumerate(models):
        ok = passed[model].to_numpy(dtype=bool)
        vals = acc[model].to_numpy()
        ax.scatter(np.full(ok.sum(), i), vals[ok], s=45, color=MODEL_COLORS[model], edgecolor="k", zorder=3)
        ax.scatter(np.full((~ok).sum(), i), vals[~ok], s=45, facecolor="white", edgecolor=MODEL_COLORS[model], zorder=3)
        ax.errorbar(i + 0.18, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, fmt="D", ms=9, color=MODEL_COLORS[model],
                    mec="k", capsize=4, zorder=4, label=f"{MODEL_LABELS[model]} mean +/- SD")
    ax.axhline(CHANCE_ACCURACY, ls="--", c="gray", label="nominal chance (25%)")
    thr = chance[chance["metric"] == PRIMARY_METRIC]["binomial_threshold"].dropna()
    if len(thr):
        ax.axhline(thr.iloc[0], ls=":", c="k", label=f"binomial threshold ({thr.iloc[0]:.1%})")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in models])
    ax.set(ylim=(0, 1), xlim=(-0.4, len(models) - 0.4), ylabel="Accuracy (session 2)", title="Accuracy per subject across models")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    _save(fig, out_dir, "accuracy_slopes.png")


def _paired_differences(diffs: pd.DataFrame, tests: pd.DataFrame, out_dir: Path) -> None:
    comparisons = [c for c in COMPARISONS if f"{c[0]}-{c[1]}" in set(tests["comparison"])]
    fig, axes = plt.subplots(len(METRIC_NAMES), len(comparisons), figsize=(4.6 * len(comparisons), 3.3 * len(METRIC_NAMES)),
                             squeeze=False, sharey="row")
    rng = np.random.default_rng(0)  # jitter only
    for r, metric in enumerate(METRIC_NAMES):
        for c, (a, b, label) in enumerate(comparisons):
            ax = axes[r, c]
            key = f"{a}-{b}"
            d = diffs[(diffs["metric"] == metric) & (diffs["comparison"] == key)]["diff"].to_numpy()
            row = tests[(tests["metric"] == metric) & (tests["comparison"] == key)].iloc[0]
            ax.axhline(0, ls="--", c="gray")
            ax.scatter(rng.uniform(-0.12, 0.12, len(d)), d, s=35, color=MODEL_COLORS[a], edgecolor="k", zorder=3)
            if not np.isnan(row["ci_low"]):
                ax.axhspan(row["ci_low"], row["ci_high"], color="k", alpha=0.12)
            ax.hlines(row["median_diff"], -0.3, 0.3, color="k", lw=2.5, zorder=4)
            p_text = "n/a" if np.isnan(row["p_holm"]) else f"{row['p_holm']:.3f}"
            ax.set_title(f"{label}\nmedian {row['median_diff']:+.3f}, p(Holm) = {p_text}" + (" *" if row["significant"] else ""), fontsize=9)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            if c == 0:
                ax.set_ylabel(f"Difference in {METRIC_TITLES[metric]}")
    fig.suptitle("Paired differences between models (dots: subjects; line: median; band: bootstrap 95% CI)")
    fig.tight_layout()
    _save(fig, out_dir, "paired_differences.png")


def _chance_tests(chance: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(5.2 * len(METRIC_NAMES), 3.0), squeeze=False)
    for ax, metric in zip(axes[0], METRIC_NAMES):
        sub = chance[chance["metric"] == metric]
        value = sub.pivot(index="model", columns="subject", values="observed").loc[models]
        ok = sub.pivot(index="model", columns="subject", values="above_chance").loc[models]
        lo, hi = METRIC_YLIM[metric]
        im = ax.imshow(value.to_numpy(), cmap="Blues", vmin=lo, vmax=hi, aspect="auto")
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                v = value.to_numpy()[i, j]
                ax.text(j, i, f"{v:.2f}" + ("*" if bool(ok.to_numpy()[i, j]) else ""), ha="center", va="center", fontsize=8,
                        color="w" if (v - lo) / (hi - lo) > 0.6 else "k")
        ax.set_xticks(range(value.shape[1]), [f"S{s}" for s in value.columns])
        ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
        ax.set_title(METRIC_TITLES[metric])
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "chance_tests.png")


def plot_comparison(run_dir: Path) -> None:
    """Save comparison figures to <run_dir>/comparison/figures/."""
    in_dir = run_dir / "comparison"
    needed = ("subject_summary.csv", "chance_tests.csv", "paired_differences.csv", "paired_tests.csv")
    if not all((in_dir / f).exists() for f in needed):
        print("No comparison tables found; run compare_models() first.")
        return
    subjects, chance = pd.read_csv(in_dir / "subject_summary.csv"), pd.read_csv(in_dir / "chance_tests.csv")
    diffs, tests = pd.read_csv(in_dir / "paired_differences.csv"), pd.read_csv(in_dir / "paired_tests.csv")
    models = [m for m in MODEL_ORDER if m in set(subjects["model"])]
    out_dir = in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _accuracy_slopes(subjects, chance, models, out_dir)
    if len(tests):
        _paired_differences(diffs, tests, out_dir)
    _chance_tests(chance, models, out_dir)
    print(f"Saved comparison figures to {out_dir}")
