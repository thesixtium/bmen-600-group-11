"""Are some subjects simply worse at motor-imagery BCI, and are their signals worse? Uses only the existing six models and
the existing preprocessed data; no model is added.

Part A, decoding (needs results.csv): do the SAME subjects rank low for EVERY model?
  subject x model accuracy heatmap, per-subject rank for every model, model-to-model Spearman rank correlation of the subject
  accuracies, Kendall's W (agreement of the six rankings), and a subject / model / residual variance decomposition.
Part B, raw signal quality per subject and session (model independent), from the 4-40 Hz epochs:
  mu (8-13 Hz) and beta (13-30 Hz) log band power; class separability of band power (eta^2, the share of band-power variance
  explained by the four classes) at the sensorimotor channels C3/Cz/C4, and the same as an SNR eta^2/(1-eta^2); trial-to-trial
  variability (coefficient of variation of the trial RMS); artifact rate; and the session 1 -> 2 shift.
Part C: Spearman correlation of subject accuracy with each signal measure.

Output (<run_dir>/subject_analysis/): every figure PNG has a CSV with the same stem (see the NOTES keys), plus
subject_signal_vs_accuracy_correlations.csv. With nine subjects, correlations are exploratory: a rho must be very large to be
significant, and no multiple-comparison correction is applied.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from benchmark_plots import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER
from csv_export import save_fig, write_csv
from data_statistics import BANDS, KEY_CHANNELS, _eeg_channels, _log_band_power
from download_data import RawData
from preprocess_data import ProcessedData

ARTIFACT_UV: float = 100.0  # a trial is flagged if any channel exceeds +/-100 uV after the 4-40 Hz band-pass
SESSIONS: tuple[tuple[str, int], ...] = (("0train", 1), ("1test", 2))
SESSION_COLORS: dict[int, str] = {1: "tab:blue", 2: "tab:orange"}

NOTES: dict[str, tuple[str, str]] = {
    "subject_difficulty_heatmap.png": (
        "Rows = target subjects sorted from best to worst mean accuracy over the six models, columns = models, cell = accuracy "
        "on session 2 (mean over the 10 repeats). The last column is the mean over models. A star means the result passed the "
        "chance checks of the comparison step.",
        "If the same rows are dark across all columns, those subjects are hard for every model, which points to the subject (the "
        "signal, or how consistently they imagine movements) rather than the model. Rows that are good for some models and bad "
        "for others would be a model effect.",
    ),
    "subject_rank_consistency.png": (
        "Left: rank of each subject (1 = best) for each model, one line per subject. Middle: Spearman rank correlation between the "
        "subject accuracies of two models. Right: share of the variation of the 9 x 6 subject-by-model accuracy table that is "
        "explained by the subject, by the model, and by the rest (their interaction). The title gives Kendall's W (1 = all models "
        "rank the subjects identically) and its chi-square p-value.",
        "Flat, parallel lines, correlations near 1 and a large 'subject' share mean subject difficulty is a stable property that "
        "every model sees. A large 'model' share would mean models differ more than subjects do. Correlations involving the weakest "
        "models are noisier, because their accuracy is close to chance for everyone.",
    ),
    "subject_signal_quality.png": (
        "One panel per signal measure, x = subject; blue = session 1 (training / fine-tuning data), orange = session 2 (test). "
        "eta^2 and SNR: how strongly mu+beta band power at C3/Cz/C4 differs between the four classes (SNR = eta^2 / (1 - eta^2)); "
        "log10 power: mean over trials and channels; trial RMS CV: spread of the trial RMS amplitude relative to its mean; "
        "artifact rate: share of trials with any channel beyond +/-100 uV.",
        "Subjects with low class separability have weak band-power differences between the imagined movements in the raw "
        "signal, so any model has little to learn from. High trial-RMS variability or artifact rates point to noisy recordings. "
        "A gap between blue and orange is a session change. eta^2 of pure noise is about 0.01 with 288 trials.",
    ),
    "subject_session_shift.png": (
        "Per subject: standardized shift of log band power from session 1 to 2 (mean over channels of |difference| / pooled SD, "
        "mu and beta), log-Euclidean distance between the two sessions' mean spatial covariance matrices (trace-normalized), and "
        "the ratio of the median trial RMS (session 2 / session 1).",
        "A large shift means the data the model was fine-tuned on look different from the test data, which hurts every model "
        "that learned session-1 patterns. A ratio far from 1 is an amplitude change between sessions.",
    ),
    "subject_signal_vs_accuracy.png": (
        "Each panel: x = a signal measure of one subject, y = that subject's accuracy averaged over the six models; points are "
        "labelled with the subject number; the title gives the Spearman correlation (rho, p) over the nine subjects.",
        "A strong correlation (for example accuracy rising with session-1 class separability) supports the idea that the "
        "subjects who decode poorly have worse signals. With nine points a correlation is only a hint: the shown p-values are "
        "uncorrected and the subjects are not a random sample.",
    ),
}


# ---------------------------------------------------------------- Part B: signal quality


def _eta2(bp: np.ndarray, y: np.ndarray) -> np.ndarray:
    """bp: (n_trials, n_channels, n_bands). Share of the variance of band power explained by the class: (channels, bands)."""
    grand = bp.mean(0)
    total = ((bp - grand) ** 2).sum(0)
    between = sum((y == c).sum() * (bp[y == c].mean(0) - grand) ** 2 for c in np.unique(y))
    return between / total


def _mean_cov(X: np.ndarray) -> np.ndarray:
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(-1, keepdims=True)
    C = Xc @ Xc.transpose(0, 2, 1) / X.shape[-1]
    C = C / np.trace(C, axis1=1, axis2=2)[:, None, None]
    return C.mean(0)


def _logm(C: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(C)
    return (v * np.log(np.clip(w, 1e-12, None))) @ v.T


def signal_quality(processed: ProcessedData, channels: list[str]) -> pd.DataFrame:
    """One row per subject and session."""
    motor = [channels.index(c) for c in KEY_CHANNELS]
    rows = []
    for s in sorted(processed):
        for sess_name, sess in SESSIONS:
            d = processed[s][sess_name]
            X, y = (d.X * 1e6).astype(np.float32), d.y  # microvolts
            bp = _log_band_power(X)
            eta = _eta2(bp, y)
            rms = np.sqrt((X.astype(np.float64) ** 2).mean(axis=(1, 2)))
            row = {"subject": s, "session": sess, "n_trials": len(y), "rms_median_uV": float(np.median(rms)),
                   "rms_cv": float(rms.std(ddof=1) / rms.mean()),
                   "artifact_rate_abs100uV": float(np.mean(np.abs(X).max(axis=(1, 2)) > ARTIFACT_UV)),
                   "artifact_rate_rms3x": float(np.mean(rms > 3 * np.median(rms)))}
            for b, band in enumerate(BANDS):
                row[f"log10_power_{band}"] = float(bp[..., b].mean())
                row[f"eta2_{band}_allch"] = float(eta[:, b].mean())
                row[f"eta2_{band}_motor"] = float(eta[motor, b].mean())
            row["eta2_motor"] = float(eta[motor].mean())
            row["snr_motor"] = row["eta2_motor"] / (1 - row["eta2_motor"])
            rows.append(row)
    return pd.DataFrame(rows)


def session_shift(processed: ProcessedData) -> pd.DataFrame:
    rows = []
    for s in sorted(processed):
        X1, X2 = (processed[s]["0train"].X * 1e6).astype(np.float32), (processed[s]["1test"].X * 1e6).astype(np.float32)
        bp1, bp2 = _log_band_power(X1), _log_band_power(X2)
        d = (bp2.mean(0) - bp1.mean(0)) / np.sqrt((bp1.var(0, ddof=1) + bp2.var(0, ddof=1)) / 2)  # (channels, bands)
        row = {"subject": s}
        for b, band in enumerate(BANDS):
            row[f"abs_std_shift_{band}"] = float(np.abs(d[:, b]).mean())
        row["abs_std_shift_mean"] = float(np.mean([row[f"abs_std_shift_{band}"] for band in BANDS]))
        row["cov_logeuclid_distance"] = float(np.linalg.norm(_logm(_mean_cov(X1)) - _logm(_mean_cov(X2)), "fro"))
        rms1 = np.sqrt((X1.astype(np.float64) ** 2).mean(axis=(1, 2)))
        rms2 = np.sqrt((X2.astype(np.float64) ** 2).mean(axis=(1, 2)))
        row["rms_ratio_s2_over_s1"] = float(np.median(rms2) / np.median(rms1))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_signal_quality(sig: pd.DataFrame, out: Path) -> None:
    panels = (("eta2_motor", "Class separability eta$^2$ (mu+beta, C3/Cz/C4)"), ("snr_motor", "Class SNR = eta$^2$/(1-eta$^2$)"),
              ("log10_power_mu", "Mean log10 power, mu (uV$^2$)"), ("log10_power_beta", "Mean log10 power, beta (uV$^2$)"),
              ("rms_cv", "Trial RMS coefficient of variation"), ("artifact_rate_abs100uV", "Artifact rate (|x| > 100 uV)"))
    subjects = sorted(sig["subject"].unique())
    x = np.arange(len(subjects))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title) in zip(axes.ravel(), panels):
        wide = sig.pivot(index="subject", columns="session", values=col).loc[subjects]
        ax.vlines(x, wide[1], wide[2], color="gray", lw=1)
        for sess in (1, 2):
            ax.scatter(x, wide[sess], s=40, color=SESSION_COLORS[sess], edgecolor="k", zorder=3, label=f"session {sess}")
        ax.set_xticks(x, [f"S{s}" for s in subjects])
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out, "subject_signal_quality.png", *NOTES["subject_signal_quality.png"])
    write_csv(sig, out / "subject_signal_quality.csv")


def _plot_session_shift(shift: pd.DataFrame, out: Path) -> None:
    subjects = shift["subject"].tolist()
    x = np.arange(len(subjects))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(x - 0.2, shift["abs_std_shift_mu"], 0.4, color="tab:purple", label="mu")
    axes[0].bar(x + 0.2, shift["abs_std_shift_beta"], 0.4, color="tab:cyan", label="beta")
    axes[0].set(title="Standardized band-power shift\n(mean |diff| / pooled SD over channels)")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, shift["cov_logeuclid_distance"], 0.6, color="tab:gray")
    axes[1].set(title="Log-Euclidean distance between\nsession mean covariances")
    axes[2].bar(x, shift["rms_ratio_s2_over_s1"], 0.6, color="tab:olive")
    axes[2].axhline(1, ls="--", c="gray")
    axes[2].set(title="Median trial RMS, session 2 / session 1")
    for ax in axes:
        ax.set_xticks(x, [f"S{s}" for s in subjects])
    fig.tight_layout()
    save_fig(fig, out, "subject_session_shift.png", *NOTES["subject_session_shift.png"])
    write_csv(shift, out / "subject_session_shift.csv")


# ---------------------------------------------------------------- Part A: decoding difficulty


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = stats.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


def _plot_heatmap(acc: pd.DataFrame, passed: pd.DataFrame | None, chance_note: pd.Series | None, out: Path) -> None:
    order = acc.mean(axis=1).sort_values(ascending=False).index
    tbl = acc.loc[order].copy()
    tbl["mean over models"] = tbl.mean(axis=1)
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * tbl.shape[1], 1.5 + 0.55 * len(tbl)))
    im = ax.imshow(tbl.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(tbl.index):
        for j, col in enumerate(tbl.columns):
            star = "*" if (passed is not None and col in passed.columns and bool(passed.loc[s, col])) else ""
            v = tbl.iloc[i, j]
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=9, color="w" if v > 0.6 else "k")
    ax.set_xticks(range(tbl.shape[1]), [MODEL_LABELS.get(c, c) for c in tbl.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(tbl)), [f"S{s}" for s in tbl.index])
    ax.set_title("Accuracy per subject and model (subjects sorted by mean accuracy)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    save_fig(fig, out, "subject_difficulty_heatmap.png", *NOTES["subject_difficulty_heatmap.png"])


def _plot_rank_consistency(ranks: pd.DataFrame, rho: pd.DataFrame, decomp: pd.DataFrame, w: dict, out: Path) -> None:
    models = list(ranks.columns)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.2, 1, 0.7]})
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    for k, s in enumerate(ranks.index):
        ax.plot(range(len(models)), ranks.loc[s], marker="o", color=cmap(k % 10), label=f"S{s}")
    ax.invert_yaxis()
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=30, ha="right")
    ax.set(ylabel="Rank (1 = best subject)", title="Subject rank per model")
    ax.legend(fontsize=7, ncol=3)
    ax = axes[1]
    im = ax.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{rho.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models], fontsize=8)
    ax.set_title("Spearman rho between models\n(subject accuracies)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    ax.bar(decomp["source"], decomp["fraction_of_total_variance"], color=["tab:green", "tab:blue", "tab:gray"])
    ax.set(ylim=(0, 1), ylabel="Fraction of variance", title="Variance of the\nsubject x model table")
    ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Kendall's W = {w['kendall_w']:.2f} (chi2 p = {w['p_value']:.4f}; {w['n_subjects']} subjects, {w['n_models']} models)")
    fig.tight_layout()
    save_fig(fig, out, "subject_rank_consistency.png", *NOTES["subject_rank_consistency.png"])


def _decoding_part(res: pd.DataFrame, run_dir: Path, out: Path) -> pd.Series | None:
    """Writes the decoding-difficulty tables/figures; returns the subject mean accuracy over models (index = subject)."""
    models = [m for m in MODEL_ORDER if m in set(res["model"])]
    acc = res.groupby(["subject", "model"])["accuracy"].mean().unstack("model")[models].dropna()
    n, m = acc.shape
    if n < 3 or m < 2:
        print("[subjects] too few subjects/models for the consistency analysis.")
        return None

    passed = None
    chance_path = run_dir / "comparison" / "chance_tests.csv"
    if chance_path.exists():
        ch = pd.read_csv(chance_path)
        passed = ch[ch["metric"] == "accuracy"].pivot(index="subject", columns="model", values="above_chance")

    ranks = acc.rank(ascending=False, method="average")
    tbl = acc.copy()
    tbl["mean_all_models"] = acc.mean(axis=1)
    tbl["sd_across_models"] = acc.std(axis=1, ddof=1)
    tbl["rank_overall_1_best"] = tbl["mean_all_models"].rank(ascending=False)
    if passed is not None:
        tbl["n_models_above_chance"] = passed.loc[acc.index, models].sum(axis=1).astype(int)
    for mod in models:
        tbl[f"rank_{mod}"] = ranks[mod]
    write_csv(tbl.reset_index().sort_values("rank_overall_1_best"), out / "subject_difficulty_heatmap.csv")
    _plot_heatmap(acc, passed, None, out)

    rho_tbl = pd.DataFrame(index=models, columns=models, dtype=float)
    rho_rows = []
    for a in models:
        for b in models:
            r, p = (1.0, 0.0) if a == b else _spearman(acc[a].to_numpy(), acc[b].to_numpy())
            rho_tbl.loc[a, b] = r
            rho_rows.append({"model_a": a, "model_b": b, "spearman_rho": r, "p_value": p})
    R = ranks.sum(axis=1)
    S = float(((R - R.mean()) ** 2).sum())
    kw = 12 * S / (m**2 * (n**3 - n))  # no tie correction (ties are very unlikely with continuous accuracies)
    w = {"kendall_w": kw, "chi2": m * (n - 1) * kw, "df": n - 1, "p_value": float(stats.chi2.sf(m * (n - 1) * kw, n - 1)),
         "n_subjects": n, "n_models": m}
    grand = acc.to_numpy().mean()
    ss_tot = float(((acc.to_numpy() - grand) ** 2).sum())
    ss_sub = float(m * ((acc.mean(axis=1) - grand) ** 2).sum())
    ss_mod = float(n * ((acc.mean(axis=0) - grand) ** 2).sum())
    decomp = pd.DataFrame({"source": ["subject", "model", "subject x model (residual)"], "sum_squares": [ss_sub, ss_mod, ss_tot - ss_sub - ss_mod]})
    decomp["fraction_of_total_variance"] = decomp["sum_squares"] / ss_tot

    write_csv(ranks.reset_index().melt(id_vars="subject", var_name="model", value_name="rank_1_best"), out / "subject_rank_consistency.csv")
    write_csv(pd.DataFrame(rho_rows), out / "subject_rank_correlation.csv")
    write_csv(pd.DataFrame([w]), out / "subject_rank_consistency_summary.csv")
    write_csv(decomp, out / "subject_variance_decomposition.csv")
    _plot_rank_consistency(ranks, rho_tbl, decomp, w, out)
    return acc


# ---------------------------------------------------------------- Part C: signal vs accuracy


def _plot_signal_vs_accuracy(ana: pd.DataFrame, out: Path) -> pd.DataFrame:
    panels = (("eta2_motor_s1", "Class separability eta$^2$, session 1"), ("eta2_motor_s2", "Class separability eta$^2$, session 2"),
              ("artifact_rate_abs100uV_s2", "Artifact rate, session 2"), ("rms_cv_s2", "Trial RMS CV, session 2"),
              ("abs_std_shift_mean", "Band-power shift, session 1 to 2"), ("cov_logeuclid_distance", "Covariance distance, session 1 to 2"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (col, title) in zip(axes.ravel(), panels):
        ax.scatter(ana[col], ana["mean_accuracy"], s=50, color="tab:blue", edgecolor="k", zorder=3)
        for _, r in ana.iterrows():
            ax.annotate(f"S{int(r['subject'])}", (r[col], r["mean_accuracy"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
        rho, p = _spearman(ana[col].to_numpy(), ana["mean_accuracy"].to_numpy())
        ax.set(xlabel=title, ylabel="Mean accuracy over models", title=f"rho = {rho:.2f}, p = {p:.3f}")
    fig.tight_layout()
    save_fig(fig, out, "subject_signal_vs_accuracy.png", *NOTES["subject_signal_vs_accuracy.png"])
    write_csv(ana, out / "subject_signal_vs_accuracy.csv")
    return ana


def _correlations(ana: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    skip = {"subject", "mean_accuracy", "n_trials_s1", "n_trials_s2"}
    metrics = [c for c in ana.columns if c not in skip and ana[c].dtype.kind == "f"]
    targets = {"mean_all_models": ana.set_index("subject")["mean_accuracy"], **{m: acc[m] for m in acc.columns}}
    rows = []
    for name, y in targets.items():
        y = y.loc[ana["subject"]].to_numpy()
        for c in metrics:
            rho, p = _spearman(ana[c].to_numpy(), y)
            rows.append({"accuracy_of": name, "signal_measure": c, "spearman_rho": rho, "p_value_uncorrected": p, "n_subjects": len(y)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- entry point


def analyze_subjects(data: RawData, processed: ProcessedData, run_dir: Path) -> None:
    """Write the subject-comparison figures and CSVs to <run_dir>/subject_analysis/."""
    out = run_dir / "subject_analysis"
    out.mkdir(parents=True, exist_ok=True)
    channels = _eeg_channels(data[min(data)]["0train"]["0"])
    sig, shift = signal_quality(processed, channels), session_shift(processed)
    _plot_signal_quality(sig, out)
    _plot_session_shift(shift, out)

    results_path = run_dir / "results.csv"
    if not results_path.exists():
        print("[subjects] no results.csv; only the signal-quality part was written.")
        return
    acc = _decoding_part(pd.read_csv(results_path), run_dir, out)
    if acc is None:
        return
    wide = sig.pivot(index="subject", columns="session")
    wide.columns = [f"{name}_s{sess}" for name, sess in wide.columns]
    ana = wide.reset_index().merge(shift, on="subject")
    ana.insert(1, "mean_accuracy", ana["subject"].map(acc.mean(axis=1)))
    ana = ana[ana["subject"].isin(acc.index)].reset_index(drop=True)
    _plot_signal_vs_accuracy(ana, out)
    write_csv(_correlations(ana, acc), out / "subject_signal_vs_accuracy_correlations.csv")
    print(f"[subjects] saved subject-comparison figures and CSVs to {out}")
