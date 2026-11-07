"""CSV files with the exact numbers behind every benchmark / training / comparison figure.

Convention: the CSV has the SAME FILE STEM as its PNG and sits next to it
  <run_dir>/figures/<stem>.csv                (benchmark_plots.py, training_plots.py)
  <run_dir>/comparison/figures/<stem>.csv     (comparison_plots.py)
Figures with several kinds of marks get extra files <stem>_<what>.csv (for example per-repeat dots).
The figures of data_statistics.py are handled by data_statistics_csv.py, the new figures write their own CSVs.

Also holds two helpers shared by the new modules: save_fig() (same 'How to read / Takeaways' caption style as the
existing plots) and write_csv(). Nothing here changes an existing figure or table.
"""

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from benchmark_plots import MODEL_ORDER
from data_statistics import CLASSES
from training_plots import PHASES, _load, _mean_sd, _phase_curves

METRIC_COLS: tuple[str, ...] = ("accuracy", "kappa", "macro_f1")


# ---------------------------------------------------------------- helpers shared with the new modules


def save_fig(fig: Figure, out_dir: Path, name: str, read: str, takeaway: str) -> None:
    """Caption under the figure in the style of the existing plots, 150 dpi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    width = int(fig.get_figwidth() * 12)
    text = textwrap.fill(f"How to read: {read}", width) + "\n\n" + textwrap.fill(f"Takeaways: {takeaway}", width)
    fig.text(0.02, -0.02, text, ha="left", va="top", fontsize=9)
    fig.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def box_stats(x) -> dict:
    """Numbers drawn by a matplotlib box plot (whiskers at the most extreme point within 1.5 IQR), plus mean and SD."""
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return {
        "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else np.nan, "min": x.min(), "q1": q1,
        "median": med, "q3": q3, "max": x.max(), "whisker_low": inside.min(), "whisker_high": inside.max(),
        "n_outliers": int(len(x) - len(inside)),
    }


def _by_model_order(df: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.assign(_o=df["model"].map(order)).sort_values(["_o", *extra]).drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- benchmark_plots.py figures


def _export_benchmark(run_dir: Path) -> None:
    results, conf_path = run_dir / "results.csv", run_dir / "confusion_matrices.csv"
    if not results.exists() or not conf_path.exists():
        print("[csv] no results.csv / confusion_matrices.csv; skipping benchmark CSVs.")
        return
    res, conf = pd.read_csv(results), pd.read_csv(conf_path)
    models = [m for m in MODEL_ORDER if m in set(res["model"])]
    out = run_dir / "figures"

    # accuracy_per_subject.png: large marker = mean over the repeats, faint dots = the repeats
    per_subject = res.groupby(["model", "subject"])["accuracy"].agg(
        mean_accuracy="mean", sd_repeats="std", min_repeat="min", max_repeat="max", n_repeats="count").reset_index()
    write_csv(_by_model_order(per_subject, ["subject"]), out / "accuracy_per_subject.csv")
    write_csv(_by_model_order(res[["model", "subject", "repeat", "accuracy"]], ["subject", "repeat"]),
              out / "accuracy_per_subject_repeats.csv")

    # metric_distributions.png: boxes over the subject-level means, faint dots = folds
    means = res.groupby(["model", "subject"])[list(METRIC_COLS)].mean().reset_index()
    stats_rows = [{"model": m, "metric": c, **box_stats(means[means["model"] == m][c])} for m in models for c in METRIC_COLS]
    write_csv(pd.DataFrame(stats_rows), out / "metric_distributions.csv")
    write_csv(_by_model_order(means.melt(id_vars=["model", "subject"], var_name="metric", value_name="subject_mean"),
                              ["metric", "subject"]), out / "metric_distributions_subject_means.csv")
    write_csv(_by_model_order(res[["model", "subject", "repeat", *METRIC_COLS]], ["subject", "repeat"]),
              out / "metric_distributions_folds.csv")

    # confusion_matrices.png: counts summed over subjects and repeats, normalized by row
    rows = []
    for m in models:
        mat = conf[conf["model"] == m].groupby(["true_class", "pred_class"])["count"].sum().unstack(fill_value=0)
        frac = mat.div(mat.sum(axis=1), axis=0)
        for t in mat.index:
            for p in mat.columns:
                rows.append({"model": m, "true_class": t, "pred_class": p, "true_label": CLASSES[t], "pred_label": CLASSES[p],
                             "count": int(mat.loc[t, p]), "row_fraction": float(frac.loc[t, p])})
    write_csv(pd.DataFrame(rows), out / "confusion_matrices.csv")
    print(f"[csv] benchmark figure CSVs written to {out}")


# ---------------------------------------------------------------- training_plots.py figures


def _export_training(run_dir: Path) -> None:
    models = [m for m in MODEL_ORDER if any((run_dir / "histories" / m).glob("s*_r*.json"))]
    out = run_dir / "figures"
    for m in models:
        hist = _load(run_dir, m)
        subjects = sorted(hist)
        curve_rows, subject_rows, kept_rows = [], [], []
        for ph, ph_title in PHASES:
            c = _phase_curves(hist, ph)
            epochs = np.arange(1, c["n_epochs"] + 1)
            tm, ts = _mean_sd(c["train"])
            vm, vs = _mean_sd(c["val"])
            curve_rows.append(pd.DataFrame({
                "phase": ph, "epoch": epochs, "train_loss_mean": tm, "train_loss_sd": ts, "val_loss_mean": vm,
                "val_loss_sd": vs, "n_subjects": len(subjects)}))
            for i, s in enumerate(subjects):
                subject_rows.append(pd.DataFrame({"phase": ph, "subject": s, "epoch": epochs,
                                                  "train_loss": c["train"][i], "val_loss": c["val"][i]}))
            kept_rows.append({"phase": ph, "n_folds": len(c["best"]), "median_kept_epoch": float(np.median(c["best"])),
                              "pct_folds_kept_epoch_is_last": 100 * float(c["at_end"].mean()), "n_epochs_shown": c["n_epochs"]})
        write_csv(pd.concat(curve_rows), out / f"training_loss_{m}.csv")
        write_csv(pd.concat(subject_rows), out / f"training_loss_{m}_by_subject.csv")
        write_csv(pd.DataFrame(kept_rows), out / f"training_loss_{m}_kept_epochs.csv")
    if models:
        print(f"[csv] training-loss CSVs written to {out}")


# ---------------------------------------------------------------- comparison_plots.py figures


def _export_comparison(run_dir: Path) -> None:
    cdir = run_dir / "comparison"
    needed = ("subject_summary.csv", "chance_tests.csv", "paired_differences.csv", "paired_tests.csv")
    if not all((cdir / f).exists() for f in needed):
        print("[csv] no comparison tables; skipping comparison CSVs.")
        return
    subjects, chance = pd.read_csv(cdir / "subject_summary.csv"), pd.read_csv(cdir / "chance_tests.csv")
    diffs, tests = pd.read_csv(cdir / "paired_differences.csv"), pd.read_csv(cdir / "paired_tests.csv")
    out = cdir / "figures"

    # accuracy_slopes.png
    acc = subjects[subjects["metric"] == "accuracy"].rename(columns={"mean": "accuracy_mean"})
    ch = chance[chance["metric"] == "accuracy"][["model", "subject", "above_chance", "binomial_threshold", "perm_threshold", "perm_p"]]
    slopes = _by_model_order(acc[["model", "subject", "accuracy_mean", "sd_repeats", "n_repeats"]].merge(ch, on=["model", "subject"]), ["subject"])
    write_csv(slopes, out / "accuracy_slopes.csv")
    summ = slopes.groupby("model")["accuracy_mean"].agg(mean_across_subjects="mean", sd_across_subjects="std", n_subjects="count").reset_index()
    write_csv(_by_model_order(summ, []), out / "accuracy_slopes_summary.csv")

    # paired_differences.png (primary) and paired_differences_secondary.png: one row per subject and comparison
    cols = ["family", "metric", "comparison", "label", "median_diff", "ci_low", "ci_high", "p_raw", "p_holm", "significant"]
    merged = diffs.merge(tests[cols], on=["family", "metric", "comparison"], how="left")
    write_csv(merged[merged["family"] == "primary"], out / "paired_differences.csv")
    write_csv(merged[merged["family"] == "secondary"], out / "paired_differences_secondary.csv")

    # chance_tests.png
    write_csv(_by_model_order(chance[["metric", "model", "subject", "observed", "above_chance", "perm_p", "perm_threshold", "binomial_threshold"]],
                              ["metric", "subject"]), out / "chance_tests.csv")
    print(f"[csv] comparison figure CSVs written to {out}")


def export_figure_csvs(run_dir: Path) -> None:
    """Write the numbers behind every benchmark, training-loss and comparison figure next to the PNGs."""
    _export_benchmark(run_dir)
    _export_training(run_dir)
    _export_comparison(run_dir)
