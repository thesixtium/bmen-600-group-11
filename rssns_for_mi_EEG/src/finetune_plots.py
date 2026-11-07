"""How much does fine-tuning help each model? Pre- vs post-fine-tuning performance, per subject and aggregated.

"Pre" = the saved PRETRAINED checkpoint (cross-subject pretraining only, the epoch with the lowest pretraining validation
loss) evaluated on the target subject's session 2, i.e. zero-shot: no data of the target subject was used. "Post" = the
fine-tuned model, exactly the result already in results.csv. The converted SNN is not trained, so it is left out.
Pre metrics are computed from checkpoints/<model>/s<S>_r<R>.pt after the benchmark and cached in pretrained_metrics.csv;
nothing is retrained and no existing result changes.

Output (<run_dir>/finetune/): pretrained_metrics.csv, finetune_folds.csv (every fold), and for each figure a PNG with a CSV of
the same stem: finetune_pre_post_per_subject, finetune_gain_per_subject, finetune_pre_post_aggregate, finetune_gain_aggregate.
The subject is the independent unit: marks are subject means over the 10 repeats, aggregates are mean +/- SD over subjects.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from benchmark import EEGNET_CFG, RNN_CFG, RSNN_CFG, _recurrent_hidden, _stem, compute_metrics
from benchmark_config import N_CHANNELS, N_CLASSES
from benchmark_plots import CHANCE_ACCURACY, METRICS, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from comparison import _wilcoxon, holm_correction
from csv_export import METRIC_COLS, save_fig, write_csv
from folds import Fold, build_fold_data
from models import EEGNet, RecurrentELUNet, RecurrentLIFNet
from training import predict_logits

TRAINED: tuple[str, ...] = tuple(m for m in MODEL_ORDER if m != "snn")
METRIC_LABEL: dict[str, str] = {c: t for c, t, _ in METRICS}
REF_LINE: dict[str, float] = {"accuracy": CHANCE_ACCURACY, "kappa": 0.0}  # nominal chance

NOTES: dict[str, tuple[str, str]] = {
    "finetune_pre_post_per_subject.png": (
        "One column per trained model, one row per metric. x = target subject. Hollow marker = pretrained model evaluated on the "
        "subject's session 2 with NO target data (zero-shot); filled marker = after fine-tuning on the subject's session 1. The gray "
        "segment is the change. Each marker is the mean over the 10 repeats. The dashed line is nominal chance (accuracy 0.25, "
        "kappa 0).",
        "Long upward segments mean fine-tuning helps that subject a lot; a segment near zero means the pretrained model already "
        "did as well as it can, or fine-tuning added nothing. A hollow marker near the dashed line means cross-subject "
        "pretraining alone did not transfer to that subject. A downward segment means fine-tuning hurt.",
    ),
    "finetune_gain_per_subject.png": (
        "Bars = gain from fine-tuning (post minus pre) for each subject, mean over the 10 repeats, one bar per model; error bars "
        "are the SD over the 10 repeats. One panel per metric. The dashed line is zero (no gain).",
        "Compare bar heights within a subject to see which model profits most from fine-tuning, and across subjects to see "
        "whether the same subjects always gain little. Error bars that reach zero mean the gain is not stable over the "
        "random pretraining draws.",
    ),
    "finetune_pre_post_aggregate.png": (
        "One panel per metric. Pale bar = mean over the 9 subjects before fine-tuning, solid bar = after; error bars are the SD "
        "over subjects. Gray lines join each subject's pre and post value (subject means over repeats). The dashed line is "
        "nominal chance.",
        "The distance between the pale and the solid bar is the average benefit of fine-tuning for that model. Gray lines that "
        "all rise mean the benefit is consistent across subjects. Compare the pale bars to see which model transfers best "
        "without any target data, and the solid bars for the final performance.",
    ),
    "finetune_gain_aggregate.png": (
        "Bar = mean gain (post minus pre) over the 9 subjects, error bar = SD over subjects, dots = subjects. The label gives the "
        "number of subjects that improved and the Holm-corrected p-value of the exact two-sided Wilcoxon signed-rank test on the "
        "nine paired subject means (Holm over the trained models, separately for each metric).",
        "A large positive bar means fine-tuning matters a lot for that model. With nine subjects the smallest exact two-sided p "
        "is about 0.004, and Holm over five models raises it, so only a gain with the same sign in every subject can pass 0.05. "
        "The comparison is between models' gains, not a test that one model's gain is larger than another's.",
    ),
}


# ---------------------------------------------------------------- pre-fine-tuning evaluation from the checkpoints


def _build(model: str) -> torch.nn.Module:
    if model == "ann":
        return EEGNet(EEGNET_CFG)
    n_hidden = _recurrent_hidden(model.endswith("_wide"))
    if model.startswith("rsnn"):
        return RecurrentLIFNet(N_CHANNELS, n_hidden, N_CLASSES, RSNN_CFG)
    return RecurrentELUNet(N_CHANNELS, n_hidden, N_CLASSES, RNN_CFG)


def compute_pretrained_metrics(processed: dict, folds: list[Fold], run_dir: Path, device: torch.device) -> pd.DataFrame:
    """Test metrics of the pretrained (pre fine-tuning) checkpoint of every fold and trained model; cached on disk."""
    path = run_dir / "finetune" / "pretrained_metrics.csv"
    cols = ["model", "subject", "repeat", "pre_accuracy", "pre_kappa", "pre_macro_f1"]
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    done = {(r["model"], int(r["subject"]), int(r["repeat"])) for r in rows}
    nets: dict[str, torch.nn.Module] = {}
    for fold in folds:
        todo = [m for m in TRAINED if (m, fold.target, fold.repeat) not in done
                and (run_dir / "checkpoints" / m / f"{_stem(fold)}.pt").exists()]
        if not todo:
            continue
        test = build_fold_data(processed, fold).test
        for m in todo:
            if m not in nets:
                nets[m] = _build(m).to(device)
            ckpt_path = run_dir / "checkpoints" / m / f"{_stem(fold)}.pt"
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)["pretrained"]
            except Exception:  # the file is our own checkpoint (tensors only); fall back to the unrestricted loader
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["pretrained"]
            nets[m].load_state_dict(state)
            met = compute_metrics(test.y, predict_logits(nets[m], test.X, device).argmax(1))
            rows.append({"model": m, "subject": fold.target, "repeat": fold.repeat, "pre_accuracy": met["accuracy"],
                         "pre_kappa": met["kappa"], "pre_macro_f1": met["macro_f1"]})
    df = pd.DataFrame(rows, columns=cols)
    write_csv(df, path)
    return df


# ---------------------------------------------------------------- tables


def _fold_table(res: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "subject", "repeat"]
    f = res.merge(pre, on=keys, how="inner")
    out = f[keys].copy()
    for c in METRIC_COLS:
        out[f"pre_{c}"] = f[f"pre_{c}"]
        out[f"post_{c}"] = f[c]
        out[f"gain_{c}"] = f[c] - f[f"pre_{c}"]
    return out.sort_values(keys).reset_index(drop=True)


def _subject_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, s), g in folds.groupby(["model", "subject"]):
        for c in METRIC_COLS:
            rows.append({"model": m, "subject": s, "metric": c, "pre_mean": g[f"pre_{c}"].mean(), "post_mean": g[f"post_{c}"].mean(),
                         "gain_mean": g[f"gain_{c}"].mean(), "gain_sd_repeats": g[f"gain_{c}"].std(), "n_repeats": len(g)})
    return pd.DataFrame(rows)


def _aggregate_table(per: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for c in METRIC_COLS:
        fam = []
        for m in models:
            d = per[(per["model"] == m) & (per["metric"] == c)].sort_values("subject")
            gain = (d["post_mean"] - d["pre_mean"]).to_numpy()
            stat, p = _wilcoxon(gain)
            fam.append({"model": m, "metric": c, "pre_mean": d["pre_mean"].mean(), "pre_sd": d["pre_mean"].std(),
                        "post_mean": d["post_mean"].mean(), "post_sd": d["post_mean"].std(), "gain_mean": gain.mean(),
                        "gain_sd": gain.std(ddof=1), "gain_median": float(np.median(gain)), "n_subjects": len(gain),
                        "n_subjects_improved": int(np.sum(gain > 0)), "wilcoxon_statistic": stat, "p_raw": p})
        p_holm = holm_correction(np.array([r["p_raw"] for r in fam], dtype=float))
        for r, ph in zip(fam, p_holm):
            r["p_holm"] = float(ph)
        rows.extend(fam)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def _pre_post_per_subject(per: pd.DataFrame, models: list[str], out: Path) -> None:
    subjects = sorted(per["subject"].unique())
    x = np.arange(len(subjects))
    fig, axes = plt.subplots(len(METRICS), len(models), figsize=(3.4 * len(models), 2.9 * len(METRICS)), squeeze=False, sharey="row")
    for r, (col, title, ylim) in enumerate(METRICS):
        for c, m in enumerate(models):
            ax = axes[r, c]
            d = per[(per["model"] == m) & (per["metric"] == col)].set_index("subject").loc[subjects]
            ax.vlines(x, d["pre_mean"], d["post_mean"], color="gray", lw=1)
            ax.scatter(x, d["pre_mean"], s=38, facecolor="white", edgecolor=MODEL_COLORS[m], zorder=3, label="pre (zero-shot)")
            ax.scatter(x, d["post_mean"], s=38, color=MODEL_COLORS[m], edgecolor="k", zorder=3, label="post (fine-tuned)")
            if col in REF_LINE:
                ax.axhline(REF_LINE[col], ls="--", c="gray", lw=1)
            ax.set_ylim(*ylim)
            ax.set_xticks(x, [f"S{s}" for s in subjects], fontsize=7)
            if r == 0:
                ax.set_title(MODEL_LABELS[m], fontsize=10)
            if c == 0:
                ax.set_ylabel(title)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Pre- vs post-fine-tuning, per target subject (mean over 10 repeats)")
    fig.tight_layout()
    save_fig(fig, out, "finetune_pre_post_per_subject.png", *NOTES["finetune_pre_post_per_subject.png"])


def _gain_per_subject(per: pd.DataFrame, models: list[str], out: Path) -> None:
    subjects = sorted(per["subject"].unique())
    x = np.arange(len(subjects))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 3.3 * len(METRICS)), squeeze=False)
    for r, (col, title, _) in enumerate(METRICS):
        ax = axes[r, 0]
        for i, m in enumerate(models):
            d = per[(per["model"] == m) & (per["metric"] == col)].set_index("subject").loc[subjects]
            ax.bar(x + (i - (len(models) - 1) / 2) * width, d["gain_mean"], width, yerr=d["gain_sd_repeats"], capsize=2,
                   color=MODEL_COLORS[m], edgecolor="k", lw=0.5, label=MODEL_LABELS[m])
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_xticks(x, [f"S{s}" for s in subjects])
        ax.set_ylabel(f"Gain in {title}\n(post - pre)")
    axes[0, 0].legend(ncol=len(models), fontsize=7)
    axes[-1, 0].set_xlabel("Target subject")
    fig.suptitle("Gain from fine-tuning per target subject (bars: mean over repeats; error bars: SD over repeats)")
    fig.tight_layout()
    save_fig(fig, out, "finetune_gain_per_subject.png", *NOTES["finetune_gain_per_subject.png"])


def _pre_post_aggregate(per: pd.DataFrame, agg: pd.DataFrame, models: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.6 * len(METRICS), 5), squeeze=False)
    for ax, (col, title, ylim) in zip(axes[0], METRICS):
        pre = per[per["metric"] == col].pivot(index="subject", columns="model", values="pre_mean")
        post = per[per["metric"] == col].pivot(index="subject", columns="model", values="post_mean")
        a = agg[agg["metric"] == col].set_index("model")
        for i, m in enumerate(models):
            ax.bar(i - 0.2, a.loc[m, "pre_mean"], 0.38, yerr=a.loc[m, "pre_sd"], capsize=3, color=MODEL_COLORS[m], alpha=0.35,
                   edgecolor="k", label="pre (zero-shot)" if i == 0 else None)
            ax.bar(i + 0.2, a.loc[m, "post_mean"], 0.38, yerr=a.loc[m, "post_sd"], capsize=3, color=MODEL_COLORS[m], edgecolor="k",
                   label="post (fine-tuned)" if i == 0 else None)
            for s in pre.index:
                ax.plot([i - 0.2, i + 0.2], [pre.loc[s, m], post.loc[s, m]], color="gray", lw=0.7, alpha=0.7, marker="o", ms=2.5)
        if col in REF_LINE:
            ax.axhline(REF_LINE[col], ls="--", c="gray", lw=1)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        ax.set(ylim=ylim, title=title)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pre- vs post-fine-tuning, aggregated (bars: mean +/- SD over 9 subjects; lines: subjects)")
    fig.tight_layout()
    save_fig(fig, out, "finetune_pre_post_aggregate.png", *NOTES["finetune_pre_post_aggregate.png"])


def _gain_aggregate(per: pd.DataFrame, agg: pd.DataFrame, models: list[str], out: Path) -> None:
    rng = np.random.default_rng(0)  # jitter only
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.6 * len(METRICS), 5), squeeze=False)
    for ax, (col, title, _) in zip(axes[0], METRICS):
        a = agg[agg["metric"] == col].set_index("model")
        for i, m in enumerate(models):
            ax.bar(i, a.loc[m, "gain_mean"], 0.6, yerr=a.loc[m, "gain_sd"], capsize=4, color=MODEL_COLORS[m], alpha=0.7, edgecolor="k")
            g = per[(per["model"] == m) & (per["metric"] == col)]["gain_mean"].to_numpy()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), g, s=18, color="k", zorder=3)
            ph = a.loc[m, "p_holm"]
            label = f"{int(a.loc[m, 'n_subjects_improved'])}/{int(a.loc[m, 'n_subjects'])} up\np(Holm)={'n/a' if np.isnan(ph) else f'{ph:.3f}'}"
            top = max(a.loc[m, "gain_mean"] + a.loc[m, "gain_sd"], g.max())
            ax.text(i, top, label, ha="center", va="bottom", fontsize=7)
        ax.axhline(0, ls="--", c="gray", lw=1)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        ax.set(title=f"Gain in {title}", ylabel="post - pre")
        ax.margins(y=0.2)
    fig.suptitle("Gain from fine-tuning, aggregated (bars: mean +/- SD over 9 subjects; dots: subjects)")
    fig.tight_layout()
    save_fig(fig, out, "finetune_gain_aggregate.png", *NOTES["finetune_gain_aggregate.png"])


# ---------------------------------------------------------------- entry point


def run_finetune_analysis(processed: dict, folds: list[Fold], run_dir: Path, device: torch.device) -> None:
    """Evaluate the pretrained checkpoints, then write the tables, CSVs and figures to <run_dir>/finetune/."""
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        print("[finetune] no results.csv; run collect_results() first.")
        return
    out = run_dir / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    pre = compute_pretrained_metrics(processed, folds, run_dir, device)
    folds_tbl = _fold_table(pd.read_csv(results_path), pre)
    if folds_tbl.empty:
        print("[finetune] no folds with pre- and post-fine-tuning metrics; nothing to plot.")
        return
    models = [m for m in TRAINED if m in set(folds_tbl["model"])]
    per, agg = _subject_table(folds_tbl), _aggregate_table(_subject_table(folds_tbl), models)
    write_csv(folds_tbl, out / "finetune_folds.csv")
    for stem, table in (("finetune_pre_post_per_subject", per), ("finetune_gain_per_subject", per),
                        ("finetune_pre_post_aggregate", agg), ("finetune_gain_aggregate", agg)):
        write_csv(table, out / f"{stem}.csv")
    _pre_post_per_subject(per, models, out)
    _gain_per_subject(per, models, out)
    _pre_post_aggregate(per, agg, models, out)
    _gain_aggregate(per, agg, models, out)
    print(f"[finetune] saved pre/post fine-tuning tables and figures to {out}")
