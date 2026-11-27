"""Training-process figures, in the style of benchmark_plots.py (Agg backend, 'How to read / Takeaways' caption under each
figure, 150 dpi). Reads histories/<model>/s<S>_r<R>.json written by benchmark.py (and, for diverged folds, the worker logs in
logs/) and saves to <run_dir>/figures/:

  training_loss_<model>.png     one per trained model: training and validation loss vs epoch, pretraining and fine-tuning
  training_curves_overlay.png   all models on one set of axes (log epoch axis), so early-stopping behaviour can be compared
  training_kept_epochs.png      the kept (lowest-validation-loss) epoch of every fold, per model and phase
  training_stability.png        gradient norms, loss spikes, and folds that diverged to NaN/Inf (training stopped)

Robust statistics: one rare loss spike (a single fold diverging for one epoch) can be millions of times larger than a normal
loss and would dominate a mean +/- SD. The loss figures therefore use, per epoch, the median over the repeats of a subject, then
the median and inter-quartile range (IQR) over the nine subjects; y-axes are set from these robust curves and capped, and spikes
are counted and shown separately in training_stability.png. The subject, not the fold, is the unit for the loss curves.
The converted SNN is not trained, so it has no training curves. Every figure writes a CSV with the same stem.
"""

import ast
import json
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from benchmark_config import GRAD_CLIP_NORM, N_CLASSES
from benchmark_plots import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER

CHANCE_LOSS: float = float(np.log(N_CLASSES))  # cross-entropy of a uniform guess over the classes (ln 4 = 1.386)
SPIKE_LOSS: float = 10 * CHANCE_LOSS  # an epoch whose training or validation loss exceeds this counts as a loss spike
PHASES: tuple[tuple[str, str], ...] = (
    ("pretrain", "Pretraining"),
    ("finetune", "Fine-tuning"),
)
TRAIN_COLOR, VAL_COLOR = "tab:blue", "tab:orange"
_STEM = re.compile(r"s(\d+)_r(\d+)")
# NonFiniteTrainingError message written by training.fit() into the worker logs (one line)
_NONFINITE = re.compile(r"Non-finite value in \[(\w+) s(\d+)_r(\d+)\] (pretrain|finetune) at epoch (\d+)/(\d+): "
                        r"train_loss=([^,]+), val_loss=([^,]+), max pre-clip grad norm=(\S+) \(")
_LAST_FINITE = re.compile(r"Last finite epochs: (\[.*?\])\. Note")

NOTES: dict[str, tuple[str, str]] = {
    "training_loss": (
        "x = epoch, y = cross-entropy loss; left panel = cross-subject pretraining (validation: subjects not in the "
        "pretraining set), right panel = fine-tuning on the target subject's session 1 (validation: a random 20% of it). "
        "For every epoch the 10 repeats of a target subject are reduced to their median, then the line is the median over the "
        "9 subjects and the band their inter-quartile range (robust to single-fold loss spikes; spikes are counted in the "
        "panel title and shown in training_stability.png). Blue = training loss (running average over the epoch, dropout "
        "on), orange = validation loss (no dropout). Gray dashed = loss of a uniform guess (ln 4 = 1.386). Dotted vertical "
        "line = median kept (lowest-validation-loss) epoch over all folds. All figures share the y-axes of each phase.",
        "Validation loss near ln 4 means nothing that generalizes has been learned. Validation loss rising while training loss "
        "keeps falling is overfitting; the kept checkpoint is the minimum before the rise. Many folds kept at the last epoch "
        "means validation loss was still improving when the schedule ended.",
    ),
    "training_curves_overlay.png": (
        "Median curves of all trained models on one set of axes: rows = training / validation loss, columns = pretraining / "
        "fine-tuning. Each line is the median over the 9 subjects of the per-subject median over repeats. The epoch axis is "
        "logarithmic so the first epochs are visible. A triangle on a validation curve marks that model's median kept epoch. "
        "Gray dashed = ln 4 (uniform guess). Y-axes are set from the curves and capped at 10 x ln 4.",
        "Compare where each validation curve reaches its minimum (triangle) and what happens after it: a validation curve that "
        "rises while the training curve of the same model keeps falling is overfitting. A model whose validation minimum comes "
        "within the first few epochs learns nothing that generalizes after that point, so most of its epochs are spent on "
        "weights that are never kept.",
    ),
    "training_kept_epochs.png": (
        "Each dot is one fold (target subject x repeat); y = the epoch whose weights were kept (lowest validation loss), "
        "log scale; boxes summarize the folds of a model; the label gives the median. Left = pretraining, right = fine-tuning. "
        "The dashed line is the last epoch of the schedule.",
        "Kept epochs near 1 mean validation loss was lowest almost immediately and never improved afterwards. Kept epochs at "
        "the dashed line mean the schedule ended while validation loss was still falling. A wide spread means the best stopping "
        "point depends strongly on the fold.",
    ),
    "training_stability.png": (
        "Left: largest pre-clipping gradient norm of each pretraining epoch (log-log); solid line = median over folds, dotted = "
        "90th percentile; the gray dashed line is the clipping threshold (gradients above it were scaled down). Crosses are the "
        "last finite epochs of folds that diverged (from the NonFiniteTrainingError messages in logs/). Middle: share of folds "
        "with at least one loss spike (training or validation loss > 10 x ln 4) in each phase, and share of attempted folds that "
        "diverged to NaN/Inf and were stopped (the label gives the counts). Right: validation loss of the last finite epochs of "
        "every diverged fold (log y); the cross at the top marks the epoch where the loss became NaN/Inf.",
        "A median gradient norm far above the threshold means nearly every step was clipped. Loss spikes that recover leave a "
        "usable kept checkpoint (the kept epoch is the validation minimum); a fold that diverged produced no checkpoint and no "
        "result. Divergence counts refer only to folds that were attempted: folds that never started (because their worker "
        "stopped at an earlier diverged fold) are not counted.",
    ),
}


def _save(fig: Figure, out_dir: Path, name: str, note_key: str | None = None) -> None:
    read, takeaway = NOTES[note_key or name]
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {read}", width) + "\n\n" + textwrap.fill(f"Takeaways: {takeaway}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- loading


def _load_folds(run_dir: Path, model: str) -> list[tuple[int, int, dict]]:
    """[(subject, repeat, history)]; a history is {"pretrain": [epoch dicts], "finetune": [...]}."""
    out = []
    for path in sorted((run_dir / "histories" / model).glob("s*_r*.json")):
        match = _STEM.fullmatch(path.stem)
        if match:
            out.append((int(match.group(1)), int(match.group(2)), json.loads(path.read_text())))
    return out


def _group(folds: list[tuple[int, int, dict]]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for subject, _, h in folds:
        out.setdefault(subject, []).append(h)
    return out


def _load(run_dir: Path, model: str) -> dict[int, list[dict]]:
    """{target subject: [history of each repeat]} (used by csv_export)."""
    return _group(_load_folds(run_dir, model))


def _divergences(run_dir: Path) -> list[dict]:
    """Folds stopped by NonFiniteTrainingError, parsed from logs/*_shard*.txt (one record per model / fold / phase)."""
    records: dict[tuple, dict] = {}
    for path in sorted((run_dir / "logs").glob("*_shard*.txt")):
        for line in path.read_text(errors="replace").splitlines():
            match = _NONFINITE.search(line)
            if not match:
                continue
            last = []
            last_match = _LAST_FINITE.search(line)
            if last_match:
                try:
                    last = ast.literal_eval(last_match.group(1))
                except (ValueError, SyntaxError):
                    last = []
            model, s, r, phase = match.group(1), int(match.group(2)), int(match.group(3)), match.group(4)
            records[(model, s, r, phase)] = {"model": model, "subject": s, "repeat": r, "phase": phase,
                                             "epoch": int(match.group(5)), "n_epochs": int(match.group(6)), "last": last}
    return list(records.values())


# ---------------------------------------------------------------- curve helpers


def _phase_curves(histories: dict[int, list[dict]], phase: str, agg: str = "mean") -> dict:
    """Subject-level curves (n_subjects, n_epochs) of train and validation loss, plus the kept epoch of every fold.
    agg = how the repeats of one subject are combined per epoch ("mean" or "median")."""
    reduce = np.median if agg == "median" else np.mean
    n_epochs = min(len(h[phase]) for reps in histories.values() for h in reps)
    train, val, best, at_end = [], [], [], []
    for subject in sorted(histories):
        reps = histories[subject]
        train.append(reduce([[e["train_loss"] for e in h[phase][:n_epochs]] for h in reps], axis=0))
        val.append(reduce([[e["val_loss"] for e in h[phase][:n_epochs]] for h in reps], axis=0))
        best.extend(h[phase][0]["best_epoch"] for h in reps)
        at_end.extend(h[phase][0]["best_epoch"] == len(h[phase]) for h in reps)
    return {"train": np.array(train), "val": np.array(val), "best": np.array(best), "at_end": np.array(at_end), "n_epochs": n_epochs}


def _mean_sd(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = curves.mean(axis=0)
    sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return mean, sd


def _median_iqr(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q1, med, q3 = np.percentile(curves, [25, 50, 75], axis=0)
    return med, q1, q3


def _spike_share(folds: list[tuple[int, int, dict]], phase: str) -> float:
    """Share of folds with at least one epoch whose training or validation loss exceeds SPIKE_LOSS (non-finite counts too)."""
    if not folds:
        return float("nan")
    spiked = [any(not (e["train_loss"] <= SPIKE_LOSS and e["val_loss"] <= SPIKE_LOSS) for e in h[phase]) for _, _, h in folds]
    return float(np.mean(spiked))


def _draw_band(ax, curves: np.ndarray, color: str, label: str) -> None:
    med, q1, q3 = _median_iqr(curves)
    x = np.arange(1, len(med) + 1)
    ax.plot(x, med, color=color, lw=1.6, label=label)
    ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- figures


def _per_model_figures(curves: dict, folds: dict, models: list[str], out_dir: Path) -> None:
    ylim = {}
    for ph, _ in PHASES:
        lows, highs = [CHANCE_LOSS], [CHANCE_LOSS]
        for m in models:
            for key in ("train", "val"):
                _, q1, q3 = _median_iqr(curves[(m, ph)][key])
                lows.append(float(np.nanpercentile(q1, 2)))
                highs.append(float(np.nanpercentile(q3, 98)))
        ylim[ph] = (max(0.0, min(lows) - 0.05), min(max(highs), SPIKE_LOSS) + 0.05)

    for m in models:
        fig, axes = plt.subplots(1, len(PHASES), figsize=(12, 4.5), squeeze=False)
        for ax, (ph, ph_title) in zip(axes[0], PHASES):
            c = curves[(m, ph)]
            _draw_band(ax, c["train"], TRAIN_COLOR, "training loss (median, IQR)")
            _draw_band(ax, c["val"], VAL_COLOR, "validation loss (median, IQR)")
            ax.axhline(CHANCE_LOSS, ls="--", c="gray", label="chance (ln 4)")
            ax.axvline(float(np.median(c["best"])), ls=":", c="k", label="median kept epoch")
            ax.set_xlim(1, c["n_epochs"])
            ax.set_ylim(*ylim[ph])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cross-entropy loss")
            ax.set_title(
                f"{ph_title}\nkept epoch: median {np.median(c['best']):.0f}; {100 * c['at_end'].mean():.0f}% of "
                f"{len(c['best'])} folds at the last epoch; {100 * _spike_share(folds[m], ph):.0f}% with a loss spike",
                fontsize=9.5,
            )
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Training loss: {MODEL_LABELS[m]}")
        fig.tight_layout()
        _save(fig, out_dir, f"training_loss_{m}.png", note_key="training_loss")


def _overlay_figure(curves: dict, models: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, len(PHASES), figsize=(13, 8), squeeze=False)
    rows = []
    for c, (ph, ph_title) in enumerate(PHASES):
        for r, (key, key_label) in enumerate((("train", "Training loss"), ("val", "Validation loss"))):
            ax = axes[r, c]
            values = []
            for m in models:
                med, q1, q3 = _median_iqr(curves[(m, ph)][key])
                x = np.arange(1, len(med) + 1)
                ax.plot(x, med, color=MODEL_COLORS[m], lw=1.6, label=MODEL_LABELS[m])
                values.append(med)
                if key == "val":
                    kept = int(min(max(round(float(np.median(curves[(m, ph)]["best"]))), 1), len(med)))
                    ax.plot(kept, med[kept - 1], marker="v", ms=10, color=MODEL_COLORS[m], mec="k", zorder=4)
                rows.append(pd.DataFrame({"phase": ph, "curve": key, "model": m, "epoch": x, "median": med, "q1": q1, "q3": q3}))
            allv = np.concatenate(values)
            lo = max(0.0, float(np.nanpercentile(allv, 1)) - 0.05)
            hi = min(float(np.nanpercentile(allv, 99)) + 0.05, SPIKE_LOSS)
            ax.axhline(CHANCE_LOSS, ls="--", c="gray", lw=1, label="chance (ln 4)")
            ax.set_xscale("log")
            ax.set_ylim(lo, max(hi, CHANCE_LOSS + 0.1))
            ax.set(xlabel="Epoch (log scale)", ylabel=key_label, title=f"{ph_title}: {key_label.lower()}")
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Training curves of all trained models (median over subjects; triangles: median kept epoch)")
    fig.tight_layout()
    pd.concat(rows).to_csv(out_dir / "training_curves_overlay.csv", index=False)
    _save(fig, out_dir, "training_curves_overlay.png")


def _kept_epochs_figure(folds: dict, models: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(PHASES), figsize=(13, 4.8), squeeze=False)
    rng = np.random.default_rng(0)  # jitter only
    rows = []
    for ax, (ph, ph_title) in zip(axes[0], PHASES):
        n_max = 1
        for i, m in enumerate(models):
            best = np.array([h[ph][0]["best_epoch"] for _, _, h in folds[m]], dtype=float)
            n_max = max(n_max, max(len(h[ph]) for _, _, h in folds[m]))
            ax.boxplot([best], positions=[i], widths=0.5, showfliers=False, medianprops={"color": "k"})
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(best)), best, s=12, alpha=0.5, color=MODEL_COLORS[m], zorder=3)
            ax.text(i, n_max * 1.3, f"median {np.median(best):.0f}\n(n={len(best)})", ha="center", va="bottom", fontsize=8)
            rows.extend({"phase": ph, "model": m, "subject": s, "repeat": r, "kept_epoch": int(h[ph][0]["best_epoch"]),
                         "n_epochs": len(h[ph])} for s, r, h in folds[m])
        ax.axhline(n_max, ls="--", c="gray", label=f"last epoch ({n_max})")
        ax.set_yscale("log")
        ax.set_ylim(0.8, n_max * 3)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=20, ha="right")
        ax.set(ylabel="Kept epoch (log scale)", title=ph_title)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Epoch of the kept (lowest-validation-loss) checkpoint, every fold")
    fig.tight_layout()
    pd.DataFrame(rows).to_csv(out_dir / "training_kept_epochs.csv", index=False)
    _save(fig, out_dir, "training_kept_epochs.png")


def _stability_figure(run_dir: Path, folds: dict, models: list[str], out_dir: Path) -> None:
    div = _divergences(run_dir)
    div_models = {d["model"] for d in div}
    all_models = [m for m in MODEL_ORDER if m in models or m in div_models]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1.4, 1.1, 1.1]})
    grad_rows, share_rows, div_rows = [], [], []

    # A: gradient norms during pretraining
    ax = axes[0]
    for m in models:
        g = [[e.get("grad_norm_max", np.nan) for e in h["pretrain"]] for _, _, h in folds[m]]
        n = min(len(x) for x in g)
        G = np.array([x[:n] for x in g], dtype=float)
        if np.all(np.isnan(G)):
            continue  # histories written before gradient norms were logged
        x = np.arange(1, n + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med, p90 = np.nanmedian(G, axis=0), np.nanpercentile(G, 90, axis=0)
        ax.plot(x, med, color=MODEL_COLORS[m], lw=1.5, label=f"{MODEL_LABELS[m]} (median)")
        ax.plot(x, p90, color=MODEL_COLORS[m], lw=0.8, ls=":")
        grad_rows.append(pd.DataFrame({"model": m, "epoch": x, "grad_norm_median": med, "grad_norm_p90": p90}))
    labelled: set[str] = set()
    for d in div:
        if d["phase"] != "pretrain" or not d["last"]:
            continue
        ep = [e["epoch"] for e in d["last"]]
        gn = [e.get("grad_norm_max", np.nan) for e in d["last"]]
        label = None if d["model"] in labelled else f"{MODEL_LABELS.get(d['model'], d['model'])}: last finite epochs before divergence"
        labelled.add(d["model"])
        ax.scatter(ep, gn, marker="x", s=30, color=MODEL_COLORS.get(d["model"], "k"), label=label)
    if GRAD_CLIP_NORM is not None:
        ax.axhline(GRAD_CLIP_NORM, ls="--", c="gray", label=f"clipping threshold ({GRAD_CLIP_NORM:g})")
    ax.set(xscale="log", yscale="log", xlabel="Pretraining epoch (log scale)", ylabel="Max pre-clip gradient norm (log scale)",
           title="Gradient norm during pretraining")
    ax.legend(fontsize=6.5, loc="upper left")

    # B: loss spikes and divergence per model
    ax = axes[1]
    width = 0.27
    for i, m in enumerate(all_models):
        done = folds.get(m, [])
        failed = {(d["subject"], d["repeat"]) for d in div if d["model"] == m}
        attempted = len(done) + len(failed)
        shares = [_spike_share(done, "pretrain") if done else np.nan, _spike_share(done, "finetune") if done else np.nan,
                  len(failed) / attempted if attempted else np.nan]
        for k, (share, label, hatch) in enumerate(zip(shares, ("loss spike, pretraining", "loss spike, fine-tuning",
                                                                 "diverged to NaN/Inf (stopped)"), ("", "..", "xx"))):
            ax.bar(i + (k - 1) * width, 0 if np.isnan(share) else share, width, color=MODEL_COLORS[m], edgecolor="k",
                   hatch=hatch, alpha=0.85, label=label if i == 0 else None)
        ax.text(i + width, (shares[2] if not np.isnan(shares[2]) else 0) + 0.02, f"{len(failed)}/{attempted}",
                ha="center", va="bottom", fontsize=7)
        share_rows.append({"model": m, "folds_completed": len(done), "folds_diverged": len(failed), "folds_attempted": attempted,
                           "share_spike_pretrain": shares[0], "share_spike_finetune": shares[1], "share_diverged": shares[2]})
    ax.set_xticks(range(len(all_models)), [MODEL_LABELS[m] for m in all_models], rotation=20, ha="right")
    ax.set(ylim=(0, 1.15), ylabel="Share of folds", title=f"Loss spikes (> 10 x ln 4 = {SPIKE_LOSS:.1f}) and divergence")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="k", hatch=h) for h in ("", "..", "xx")]
    ax.legend(legend_handles, ["loss spike, pretraining", "loss spike, fine-tuning", "diverged to NaN/Inf (stopped)"], fontsize=7)

    # C: what the diverged folds looked like just before NaN
    ax = axes[2]
    if div:
        top = 1.0
        for d in div:
            vals = [e["val_loss"] for e in d["last"]]
            top = max([top, *vals])
        for d in div:
            ep = [e["epoch"] for e in d["last"]]
            val = [e["val_loss"] for e in d["last"]]
            color = MODEL_COLORS.get(d["model"], "k")
            ax.plot(ep + [d["epoch"]], val + [top * 3], color=color, lw=0.9, alpha=0.7, marker="o", ms=3)
            ax.scatter([d["epoch"]], [top * 3], marker="x", s=50, color=color, zorder=4)
            div_rows.append({"model": d["model"], "subject": d["subject"], "repeat": d["repeat"], "phase": d["phase"],
                             "nonfinite_epoch": d["epoch"], "n_epochs": d["n_epochs"],
                             **{f"last{k}_{f}": e.get(f) for k, e in enumerate(d["last"]) for f in ("epoch", "train_loss", "val_loss",
                                                                                                  "grad_norm_max", "clip_fraction")}})
        ax.axhline(CHANCE_LOSS, ls="--", c="gray", lw=1, label="chance (ln 4)")
        ax.set(yscale="log", xlabel="Epoch", ylabel="Validation loss (log scale)",
               title=f"Diverged folds: last finite epochs (n = {len(div)})")
        ax.text(0.98, 0.02, "x at top = epoch where the loss became NaN/Inf", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=8)
        ax.legend(fontsize=7, loc="lower left")
    else:
        ax.text(0.5, 0.5, "No diverged folds found in logs/*_shard*.txt\n(or the logs were not copied with the run)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle("Training stability")
    fig.tight_layout()
    if grad_rows:
        pd.concat(grad_rows).to_csv(out_dir / "training_stability_grad_norm.csv", index=False)
    pd.DataFrame(share_rows).to_csv(out_dir / "training_stability.csv", index=False)
    pd.DataFrame(div_rows).to_csv(out_dir / "training_stability_diverged_folds.csv", index=False)
    _save(fig, out_dir, "training_stability.png")


def plot_training(run_dir: Path) -> None:
    """Save the training-process figures (and their CSVs) to <run_dir>/figures/."""
    models = [m for m in MODEL_ORDER if any((run_dir / "histories" / m).glob("s*_r*.json"))]
    if not models:
        print("No training histories found; nothing to plot.")
        return
    folds = {m: _load_folds(run_dir, m) for m in models}
    curves = {(m, ph): _phase_curves(_group(folds[m]), ph, agg="median") for m in models for ph, _ in PHASES}
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _per_model_figures(curves, folds, models, out_dir)
    _overlay_figure(curves, models, out_dir)
    _kept_epochs_figure(folds, models, out_dir)
    _stability_figure(run_dir, folds, models, out_dir)
    print(f"Saved training-process figures to {out_dir}")