"""Training-loss figures: one figure per trained model, in the style of benchmark_plots.py (Agg backend, 'How to read /
Takeaways' caption under each figure, 150 dpi). Reads histories/<model>/s<S>_r<R>.json written by benchmark.py and saves
to <run_dir>/figures/training_loss_<model>.png.

Each figure has two panels (pretraining, fine-tuning) with training and validation loss against the epoch. The subject,
not the fold, is the independent unit: every repeat curve is first averaged within its target subject, and the plotted
line and band are the mean and SD over the nine subject-level curves. The converted SNN is not trained, so it has no
figure. All figures share the same y-axes so models can be compared by eye.
"""

import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from benchmark_config import N_CLASSES
from benchmark_plots import MODEL_LABELS, MODEL_ORDER

CHANCE_LOSS: float = float(np.log(N_CLASSES))  # cross-entropy of a uniform guess over the classes (ln 4 = 1.386)
PHASES: tuple[tuple[str, str], ...] = (
    ("pretrain", "Pretraining"),
    ("finetune", "Fine-tuning"),
)
TRAIN_COLOR, VAL_COLOR = "tab:blue", "tab:orange"
_STEM = re.compile(r"s(\d+)_r(\d+)")

READ = (
    "x = epoch, y = cross-entropy loss; left panel = cross-subject pretraining (validation: subjects not in the "
    "pretraining set), right panel = fine-tuning on the target subject's session 1 (validation: a random 20% of it). "
    "Each repeat is averaged within its target subject; the lines are the mean over the 9 subjects and the bands are "
    "+/- 1 SD across subjects. Blue = training loss, a running average over the epoch with dropout active; orange = "
    "validation loss, computed without dropout, so the training loss can sit above the validation loss. The gray dashed "
    "line is the loss of a uniform guess (ln 4 = 1.386). The dotted vertical line is the median epoch of the kept "
    "(lowest-validation-loss) checkpoint over all folds; the panel title also gives the share of folds whose kept epoch "
    "is the last one. All training-loss figures share the same y-axes."
)
TAKEAWAY = (
    "Validation loss close to ln 4 means nothing that generalizes to the validation data has been learned. Validation loss "
    "rising while training loss keeps falling is overfitting; the kept checkpoint is the minimum before the rise. A large "
    "share of folds whose kept epoch is the last epoch means the schedule ended while validation loss was still "
    "improving, so more optimizer steps could still help. Compare the pretraining panels to see whether cross-subject "
    "pretraining transfers at all for a model, and the fine-tuning panels to see how far a model gets on the target subject."
)


def _save(fig: Figure, out_dir: Path, name: str) -> None:
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {READ}", width) + "\n\n" + textwrap.fill(f"Takeaways: {TAKEAWAY}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load(run_dir: Path, model: str) -> dict[int, list[dict]]:
    """{target subject: [history of each repeat]}; a history is {"pretrain": [epoch dicts], "finetune": [...]}."""
    out: dict[int, list[dict]] = {}
    for path in sorted((run_dir / "histories" / model).glob("s*_r*.json")):
        match = _STEM.fullmatch(path.stem)
        if match:
            out.setdefault(int(match.group(1)), []).append(json.loads(path.read_text()))
    return out


def _phase_curves(histories: dict[int, list[dict]], phase: str) -> dict:
    """Subject-level curves (n_subjects, n_epochs) of train and validation loss, plus the kept epoch of every fold."""
    n_epochs = min(len(h[phase]) for reps in histories.values() for h in reps)
    train, val, best, at_end = [], [], [], []
    for subject in sorted(histories):
        reps = histories[subject]
        train.append(np.mean([[e["train_loss"] for e in h[phase][:n_epochs]] for h in reps], axis=0))
        val.append(np.mean([[e["val_loss"] for e in h[phase][:n_epochs]] for h in reps], axis=0))
        best.extend(h[phase][0]["best_epoch"] for h in reps)
        at_end.extend(h[phase][0]["best_epoch"] == len(h[phase]) for h in reps)
    return {"train": np.array(train), "val": np.array(val), "best": np.array(best), "at_end": np.array(at_end), "n_epochs": n_epochs}


def _mean_sd(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return mean, sd


def _draw_band(ax, curves: np.ndarray, color: str, label: str) -> None:
    mean, sd = _mean_sd(curves)
    x = np.arange(1, len(mean) + 1)
    ax.plot(x, mean, color=color, lw=1.6, label=label)
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)


def plot_training(run_dir: Path) -> None:
    """Save one training-loss figure per trained model to <run_dir>/figures/."""
    models = [m for m in MODEL_ORDER if any((run_dir / "histories" / m).glob("s*_r*.json"))]
    if not models:
        print("No training histories found; nothing to plot.")
        return
    curves = {(m, ph): _phase_curves(_load(run_dir, m), ph) for m in models for ph, _ in PHASES}

    # shared y-limits per phase, so the figures of different models are comparable
    ylim = {}
    for ph, _ in PHASES:
        lows, highs = [CHANCE_LOSS], [CHANCE_LOSS]
        for m in models:
            for key in ("train", "val"):
                mean, sd = _mean_sd(curves[(m, ph)][key])
                lows.append(float((mean - sd).min()))
                highs.append(float((mean + sd).max()))
        ylim[ph] = (min(lows) - 0.03, max(highs) + 0.03)

    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        fig, axes = plt.subplots(1, len(PHASES), figsize=(12, 4.5), squeeze=False)
        for ax, (ph, ph_title) in zip(axes[0], PHASES):
            c = curves[(m, ph)]
            _draw_band(ax, c["train"], TRAIN_COLOR, "training loss")
            _draw_band(ax, c["val"], VAL_COLOR, "validation loss")
            ax.axhline(CHANCE_LOSS, ls="--", c="gray", label="chance (ln 4)")
            ax.axvline(float(np.median(c["best"])), ls=":", c="k", label="median kept epoch")
            ax.set_xlim(1, c["n_epochs"])
            ax.set_ylim(*ylim[ph])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title(
                f"{ph_title}\nkept epoch: median {np.median(c['best']):.0f}; {100 * c['at_end'].mean():.0f}% of "
                f"{len(c['best'])} folds at the last epoch",
                fontsize=10,
            )
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Training loss: {MODEL_LABELS[m]}")
        fig.tight_layout()
        _save(fig, out_dir, f"training_loss_{m}.png")
    print(f"Saved training-loss figures to {out_dir}")
