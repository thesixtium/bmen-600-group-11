"""Analysis entry point: figures, statistics and CSVs from a SAVED run. Never trains, never runs a benchmark.

Usage (from the project folder, e.g. rssns_for_mi_EEG/):
  python src/make_plots.py                         # newest non-quick run, every stage that has its inputs
  python src/make_plots.py 20261113_140000         # a given run id (folder name under results/benchmark/)
  python src/make_plots.py <id> --stages benchmark comparison csv
  python src/make_plots.py <id> --stages data_statistics subjects --load-data
  python src/make_plots.py <id> --stages finetune --finetune-inference
  python src/make_plots.py --list-runs

Stages (run in this order): data_statistics, benchmark, training, comparison, csv, finetune, subjects.
Outputs go where the existing modules put them: <run_dir>/figures/, <run_dir>/comparison/, <run_dir>/finetune/,
<run_dir>/subject_analysis/, and results.csv / predictions.csv / confusion_matrices.csv in <run_dir>. data_statistics does not
depend on the run and writes to results/data_statistics/ (its existing location).

Partial runs: no COMPLETE marker is needed. The fold coverage of every model is printed first; tables and figures are built
from the folds that are finished (nothing is filled in). A stage without its inputs is skipped with the reason; a stage that
raises is reported as FAILED and the other stages still run (exit code 1 if any stage failed).

Data and inference are opt-in, never implicit:
  --load-data            loads the CACHED BNCI2014_001 recordings and preprocesses them in memory. Needed by data_statistics
                         and by the signal-quality part of subjects. Refuses to start if the data folder is empty (no download);
                         note that MOABB may still fetch individual files that are missing from a partial cache.
  --finetune-inference   evaluates the pretrained checkpoints that are not yet in <run_dir>/finetune/pretrained_metrics.csv
                         (one inference pass per fold and trained model; implies --load-data). Without it, finetune uses only
                         the cached pre-fine-tuning metrics.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from benchmark_config import RESULTS_DIR

STAGES: tuple[str, ...] = ("data_statistics", "benchmark", "training", "comparison", "csv", "finetune", "subjects")
RESULT_STAGES = {"benchmark", "comparison", "csv", "finetune", "subjects"}  # read results.csv (collected first)


class Skip(Exception):
    """A stage lacks its inputs; the message is the reason."""


# ---------------------------------------------------------------- run selection and coverage


def _runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir() and (p / "config.json").exists())


def _resolve_run(run_id: str | None) -> Path:
    if run_id:
        run_dir = RESULTS_DIR / run_id
    else:
        full = [p for p in _runs() if not p.name.startswith("quick_")]
        if not full:
            sys.exit(f"No run found under {RESULTS_DIR}. Give a run id (see --list-runs).")
        run_dir = full[-1]
        print(f"[make_plots] no run id given: using the newest run, {run_dir.name}")
    if not (run_dir / "config.json").exists() or not (run_dir / "folds.json").exists():
        sys.exit(f"{run_dir} is not a run directory (config.json / folds.json missing). See --list-runs.")
    return run_dir


def _coverage(run_dir: Path) -> tuple[dict[str, int], int]:
    """Finished folds per model (metrics files) and the run's total number of folds. Exits if a metrics file belongs to
    another run, so results of different runs are never collected together."""
    from benchmark import MODELS

    n_folds = len(json.loads((run_dir / "folds.json").read_text())["folds"])
    done, foreign = {}, []
    for m in MODELS:
        files = sorted((run_dir / "metrics" / m).glob("*.json"))
        done[m] = len(files)
        for f in files:
            rid = json.loads(f.read_text()).get("run_id")
            if rid is not None and rid != run_dir.name:
                foreign.append(f"{m}/{f.name} (run_id {rid})")
    if foreign:
        sys.exit(f"[make_plots] {len(foreign)} metrics file(s) in {run_dir.name} belong to another run, e.g. {foreign[:3]}. "
                 "Remove them; results of different runs are not mixed.")
    print(f"[make_plots] run {run_dir.name}: finished folds per model (of {n_folds}):")
    for m, n in done.items():
        state = "complete" if n == n_folds else ("PARTIAL" if n else "no results")
        print(f"    {m:<10s} {n:>3d}  {state}")
    if any(0 < n < n_folds for n in done.values()):
        print("[make_plots] partial models are analysed on their finished folds only: subject means may average fewer "
              "repeats and subjects without results drop out of paired tests. Treat these numbers as provisional.")
    if (run_dir / "COMPLETE").exists():
        print("[make_plots] COMPLETE marker present: every fold of every model is trained.")
    return done, n_folds


# ---------------------------------------------------------------- opt-in data loading


class _Data:
    def __init__(self, allowed: bool) -> None:
        self.allowed, self.raw, self.processed = allowed, None, None

    def get(self, purpose: str):
        """(raw, processed); raises Skip if --load-data was not given."""
        if not self.allowed:
            raise Skip(f"{purpose} needs the EEG recordings; rerun with --load-data (uses the cached dataset, preprocesses in memory)")
        if self.processed is None:
            from download_data import DATA_DIR, download_data
            from preprocess_data import preprocess_data

            if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
                raise Skip(f"no cached dataset in {DATA_DIR}; make_plots.py does not download data (run the training once)")
            print("[make_plots] loading the cached dataset and preprocessing it in memory ...")
            self.raw = download_data(redo=False)
            self.processed = preprocess_data(self.raw)
        return self.raw, self.processed


def _run_folds(run_dir: Path):
    """The run's own fold definitions (folds.json), not regenerated."""
    from folds import Fold

    raw = json.loads((run_dir / "folds.json").read_text())["folds"]
    return [Fold(**{k: tuple(v) if isinstance(v, list) else v for k, v in f.items()}) for f in raw]


# ---------------------------------------------------------------- stages


def _need_results(run_dir: Path) -> None:
    if not (run_dir / "results.csv").exists():
        raise Skip("no finished folds of any model (no metrics files, so no results.csv)")


def stage_data_statistics(run_dir: Path, data: _Data, args) -> None:
    raw, processed = data.get("data_statistics")
    from data_statistics import data_statistics
    from data_statistics_csv import export_data_statistics_csvs

    data_statistics(raw, processed)
    export_data_statistics_csvs(raw, processed)


def stage_benchmark(run_dir: Path, data: _Data, args) -> None:
    _need_results(run_dir)
    from benchmark_plots import plot_benchmarks

    plot_benchmarks(run_dir)


def stage_training(run_dir: Path, data: _Data, args) -> None:
    if not any((run_dir / "histories").glob("*/s*_r*.json")):
        raise Skip("no training histories (no trained fold of ann / rsnn / rnn / rsnn_wide / rnn_wide)")
    from training_plots import plot_training

    plot_training(run_dir)


def stage_comparison(run_dir: Path, data: _Data, args) -> None:
    _need_results(run_dir)
    from comparison import compare_models
    from comparison_plots import plot_comparison

    compare_models(run_dir)
    plot_comparison(run_dir)


def stage_csv(run_dir: Path, data: _Data, args) -> None:
    from csv_export import export_figure_csvs

    if not (run_dir / "comparison" / "paired_tests.csv").exists():
        print("[make_plots] csv: no comparison tables yet; the comparison CSVs are skipped (run the comparison stage).")
    export_figure_csvs(run_dir)  # each part checks its own inputs


def stage_finetune(run_dir: Path, data: _Data, args) -> None:
    _need_results(run_dir)
    if not any((run_dir / "checkpoints").glob("*/s*_r*.pt")):
        raise Skip("no checkpoints of trained models")
    from finetune_plots import run_finetune_analysis

    if args.finetune_inference:
        _, processed = data.get("finetune --finetune-inference")
        from benchmark import select_device

        run_finetune_analysis(processed, _run_folds(run_dir), run_dir, select_device())
    else:
        if not (run_dir / "finetune" / "pretrained_metrics.csv").exists():
            raise Skip("no cached pre-fine-tuning metrics (finetune/pretrained_metrics.csv); computing them needs one "
                       "inference pass per fold: rerun with --finetune-inference")
        run_finetune_analysis(None, None, run_dir, None)  # cached metrics only, no inference


def stage_subjects(run_dir: Path, data: _Data, args) -> None:
    _need_results(run_dir)
    from subject_analysis import analyze_subjects

    raw = processed = None
    if args.load_data:
        raw, processed = data.get("subjects (signal quality)")
    else:
        print("[make_plots] subjects: signal-quality parts (B, C) need --load-data; only the decoding part (A) is made.")
    analyze_subjects(raw, processed, run_dir)


STAGE_FUNCS = {
    "data_statistics": stage_data_statistics, "benchmark": stage_benchmark, "training": stage_training,
    "comparison": stage_comparison, "csv": stage_csv, "finetune": stage_finetune, "subjects": stage_subjects,
}


# ---------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analysis of a saved benchmark run (no training).")
    parser.add_argument("run_id", nargs="?", help="folder name under results/benchmark/ (default: newest non-quick run)")
    parser.add_argument("--stages", nargs="+", choices=STAGES, metavar="STAGE",
                        help=f"stages to run (default: all). Choices: {', '.join(STAGES)}")
    parser.add_argument("--load-data", action="store_true", help="allow loading + preprocessing the cached EEG dataset")
    parser.add_argument("--finetune-inference", action="store_true",
                        help="evaluate uncached pretrained checkpoints for the finetune stage (implies --load-data)")
    parser.add_argument("--list-runs", action="store_true", help="list run directories and exit")
    args = parser.parse_args(argv)
    if args.finetune_inference:
        args.load_data = True

    if args.list_runs:
        for p in _runs():
            print(p.name + ("  COMPLETE" if (p / "COMPLETE").exists() else ""))
        return 0

    run_dir = _resolve_run(args.run_id)
    stages = [s for s in STAGES if s in set(args.stages or STAGES)]
    done, _ = _coverage(run_dir)

    if any(s in RESULT_STAGES for s in stages):
        if any(done.values()):
            from benchmark import collect_results

            collect_results(run_dir)  # rebuilt from the per-fold files every time, so it is never stale
        elif (run_dir / "results.csv").exists():
            print("[make_plots] WARNING: results.csv exists but no metrics files do; it is ignored.")
            (run_dir / "results.csv").rename(run_dir / "results.stale.csv")

    data = _Data(args.load_data)
    status: dict[str, str] = {}
    for s in stages:
        print(f"\n[make_plots] ===== stage: {s} =====")
        try:
            STAGE_FUNCS[s](run_dir, data, args)
            status[s] = "done"
        except Skip as reason:
            status[s] = f"skipped: {reason}"
        except Exception as exc:  # one broken stage must not stop the others
            traceback.print_exc()
            status[s] = f"FAILED: {type(exc).__name__}: {exc}"
        print(f"[make_plots] {s}: {status[s]}")

    print(f"\n[make_plots] summary for run {run_dir.name}:")
    for s, st in status.items():
        print(f"    {s:<16s} {st}")
    return 1 if any(st.startswith("FAILED") for st in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
