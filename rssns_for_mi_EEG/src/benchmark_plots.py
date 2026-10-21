"""Benchmark figures, in the style of data_statistics.py (Agg backend, 'How to read / Takeaways' caption under each
figure, 150 dpi). Reads results.csv and confusion_matrices.csv written by benchmark.collect_results().

Descriptive plots only: no statistical tests (those belong to the later 'Comparative metrics' task). The subject, not
the fold, is the independent unit, so subject-level means are drawn as the main marks and individual repeats are shown
as faint dots for spread.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from data_statistics import CLASSES

MODEL_ORDER: tuple[str, ...] = ("ann", "snn", "rsnn")
MODEL_LABELS: dict[str, str] = {"ann": "EEGNet ANN", "snn": "Converted SNN", "rsnn": "RSNN"}
MODEL_COLORS: dict[str, str] = {"ann": "tab:blue", "snn": "tab:orange", "rsnn": "tab:green"}
CHANCE_ACCURACY: float = 0.25
METRICS: tuple[tuple[str, str, tuple[float, float]], ...] = (
    ("accuracy", "Accuracy", (0.0, 1.0)),
    ("kappa", "Cohen's kappa", (-0.2, 1.0)),
    ("macro_f1", "Macro F1", (0.0, 1.0)),
)

NOTES: dict[str, tuple[str, str]] = {
    "accuracy_per_subject.png": (
        "x = target subject, y = accuracy on that subject's session 2 (288 trials). Faint dots are the 10 random "
        "pretraining draws (repeats); the large marker is their mean. The gray dashed line is nominal chance (25%). "
        "All models share the y-axis.",
        "Compare the large markers of the models within each subject. The spread of the faint dots is the variability "
        "caused by which subjects were used for pretraining and by training randomness. Dots near the dashed line mean "
        "the model barely beats nominal chance (the formal chance threshold is a later analysis).",
    ),
    "metric_distributions.png": (
        "One panel per metric. Each box summarizes the 9 subject-level means (mean over the 10 repeats of each "
        "subject); faint dots are the individual folds (9 subjects x 10 repeats). Same scale for all models.",
        "The boxes are the quantity the research plan compares (nine paired observations). The faint dots only show "
        "fold-to-fold spread: the 90 folds are not independent, so they are not 90 samples. Kappa is 0 at chance, "
        "accuracy is 0.25 at chance.",
    ),
    "confusion_matrices.png": (
        "Rows = true class, columns = predicted class, summed over all subjects and repeats and normalized by row "
        "(each row sums to 1, so the diagonal is per-class recall). Every test session has 72 trials per class.",
        "A bright diagonal means the class is recognized. Bright off-diagonal cells show which classes are confused "
        "(for example the two hand classes with each other). Differences between panels show whether a model is "
        "strong only on some classes, which macro F1 is meant to expose.",
    ),
}


def _save(fig: Figure, out_dir: Path, name: str) -> None:
    read, takeaway = NOTES[name]
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {read}", width) + "\n\n" + textwrap.fill(f"Takeaways: {takeaway}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _accuracy_per_subject(df: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    subjects = sorted(df["subject"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    rng = np.random.default_rng(0)  # jitter only
    width = 0.8 / len(models)
    for m, model in enumerate(models):
        offset = (m - (len(models) - 1) / 2) * width
        for i, s in enumerate(subjects):
            vals = df[(df["model"] == model) & (df["subject"] == s)]["accuracy"].to_numpy()
            if len(vals) == 0:
                continue
            ax.scatter(i + offset + rng.uniform(-0.06, 0.06, len(vals)), vals, s=10, alpha=0.35, color=MODEL_COLORS[model])
            ax.scatter(i + offset, vals.mean(), s=70, marker="D", color=MODEL_COLORS[model], edgecolor="k",
                       label=MODEL_LABELS[model] if i == 0 else None)
    ax.axhline(CHANCE_ACCURACY, ls="--", c="gray", label="nominal chance (25%)")
    ax.set_xticks(range(len(subjects)), [f"S{s}" for s in subjects])
    ax.set(ylim=(0, 1), ylabel="Accuracy (session 2)", xlabel="Target subject", title="Accuracy per target subject")
    ax.legend(ncol=4, fontsize=8)
    _save(fig, out_dir, "accuracy_per_subject.png")


def _metric_distributions(df: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.5), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (col, title, ylim) in zip(axes[0], METRICS):
        subject_means = [df[df["model"] == m].groupby("subject")[col].mean().to_numpy() for m in models]
        boxes = ax.boxplot(subject_means, positions=range(len(models)), widths=0.5, patch_artist=True, showfliers=False)
        for box, model in zip(boxes["boxes"], models):
            box.set_facecolor(MODEL_COLORS[model])
            box.set_alpha(0.6)
        for i, model in enumerate(models):
            folds = df[df["model"] == model][col].to_numpy()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(folds)), folds, s=6, alpha=0.2, color="k")
        if col == "accuracy":
            ax.axhline(CHANCE_ACCURACY, ls="--", c="gray")
        if col == "kappa":
            ax.axhline(0.0, ls="--", c="gray")
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=15)
        ax.set(ylim=ylim, title=title)
    fig.suptitle("Distribution across subjects (boxes: subject means; dots: individual folds)")
    fig.tight_layout()
    _save(fig, out_dir, "metric_distributions.png")


def _confusion_matrices(conf: pd.DataFrame, models: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.8), squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = conf[conf["model"] == model]
        matrix = sub.groupby(["true_class", "pred_class"])["count"].sum().unstack(fill_value=0).to_numpy().astype(float)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w" if matrix[i, j] > 0.5 else "k")
        ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30)
        ax.set_yticks(range(len(CLASSES)), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=MODEL_LABELS[model])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Row-normalized fraction")
    _save(fig, out_dir, "confusion_matrices.png")


def plot_benchmarks(run_dir: Path) -> None:
    """Save benchmark figures to <run_dir>/figures/."""
    results_path, conf_path = run_dir / "results.csv", run_dir / "confusion_matrices.csv"
    if not results_path.exists() or not conf_path.exists():
        print("No results.csv / confusion_matrices.csv found; run collect_results() first.")
        return
    df, conf = pd.read_csv(results_path), pd.read_csv(conf_path)
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _accuracy_per_subject(df, models, out_dir)
    _metric_distributions(df, models, out_dir)
    _confusion_matrices(conf, models, out_dir)
    print(f"Saved benchmark figures to {out_dir}")
