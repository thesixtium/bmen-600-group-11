import json
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", message="Montage name 'standard_1005'")

from benchmark import (
    MODELS,
    collect_results,
    pending_folds,
    prepare_run_dir,
    run_ann_benchmark,
    run_rnn_benchmark,
    run_rsnn_benchmark,
    run_snn_benchmark,
    select_device,
)
from benchmark_plots import plot_benchmarks
from comparison import compare_models
from comparison_plots import plot_comparison
from csv_export import export_figure_csvs
from data_statistics import data_statistics
from data_statistics_csv import export_data_statistics_csvs
from download_data import download_data
from finetune_plots import run_finetune_analysis
from folds import filter_folds, generate_folds
from preprocess_data import preprocess_data
from subject_analysis import analyze_subjects
from timing import PipelineTimer
from training import EVAL_BATCH_SIZE, PRETRAIN_PATIENCE
from training_plots import plot_training


def _env_int_list(name: str) -> list[int] | None:
    """Optional comma-separated integers, e.g. BENCH_TARGETS=1,2,3 to select folds (smoke tests)."""
    value = os.environ.get(name)
    return [int(v) for v in value.split(",")] if value else None


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else default


def _acquire_lock(path: Path, stale_s: float = 7200.0) -> bool:
    """True for exactly one process: whoever creates the file first. A lock older than stale_s (a crashed or killed
    analysis) is taken over."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        try:
            if time.time() - path.stat().st_mtime > stale_s:
                path.unlink(missing_ok=True)
                return _acquire_lock(path, stale_s)
        except OSError:
            pass
        return False


# Parallel workers: BENCH_SHARD (0..BENCH_N_SHARDS-1) selects the folds i with i % BENCH_N_SHARDS == BENCH_SHARD of the fixed
# fold list, so every worker (and every chained job of a worker) computes the same split without coordination. Unset =
# one worker runs all folds, exactly as before. BENCH_FINALIZE=1 skips training and only (re)runs the analysis.
SHARD = _env_int("BENCH_SHARD", None)
N_SHARDS = _env_int("BENCH_N_SHARDS", 1) or 1
FINALIZE = os.environ.get("BENCH_FINALIZE") == "1"
IS_MAIN_SHARD = SHARD in (None, 0)

timer = PipelineTimer()


def _extra(name: str, fn) -> None:
    """Additional analyses never block the run: an error is printed and the core results stay valid."""
    with timer.stage(name):
        try:
            fn()
        except Exception:
            print(f"[extras] {name} FAILED (the benchmark results and core analysis are unaffected; "
                  f"rerun with BENCH_FINALIZE=1 BENCH_RUN_ID=<id> after fixing):")
            traceback.print_exc()


try:
    with timer.stage("download_data"):
        data = download_data(redo=False)

    with timer.stage("preprocess_data"):
        processed = preprocess_data(data)

    if IS_MAIN_SHARD:  # descriptive plots do not depend on the shard: only one worker makes them
        with timer.stage("data_statistics"):
            data_statistics(data, processed)
        _extra("data_statistics_csv", lambda: export_data_statistics_csvs(data, processed))

    # Folds shared by all models. The full set is always saved; the shard and BENCH_TARGETS / BENCH_REPEATS only select
    # which folds this process runs. Set BENCH_RUN_ID to resume or to share one run directory between jobs.
    all_folds = generate_folds(processed)
    run_dir = prepare_run_dir(all_folds)
    scope = filter_folds(all_folds, targets=_env_int_list("BENCH_TARGETS"), repeats=_env_int_list("BENCH_REPEATS"))
    mine = [f for i, f in enumerate(all_folds) if i % N_SHARDS == (SHARD or 0)]
    folds = filter_folds(mine, targets=_env_int_list("BENCH_TARGETS"), repeats=_env_int_list("BENCH_REPEATS"))
    device = select_device()
    print(f"[shard] shard {SHARD if SHARD is not None else 'all'} of {N_SHARDS}: {len(folds)} folds "
          f"(whole run: {len(scope)}); finalize={FINALIZE}")
    if IS_MAIN_SHARD:
        (run_dir / "speed_settings.json").write_text(json.dumps(
            {"BENCH_PATIENCE (pretrain early stopping, None = off)": PRETRAIN_PATIENCE, "BENCH_EVAL_BATCH": EVAL_BATCH_SIZE,
             "n_shards": N_SHARDS}, indent=1))

    if not FINALIZE:
        timer.bind(run_dir, device, MODELS, {m: len(pending_folds(run_dir, m, folds)) for m in MODELS})

        # Make and train EEGNet ANN and benchmark
        with timer.stage("benchmark: ann"):
            run_ann_benchmark(processed, folds, run_dir, device, timer)

        # Convert EEGNet ANN to SNN and benchmark
        with timer.stage("benchmark: snn"):
            run_snn_benchmark(processed, folds, run_dir, device, timer)

        # Make and train RSNN model and benchmark
        with timer.stage("benchmark: rsnn"):
            run_rsnn_benchmark(processed, folds, run_dir, device, timer)

        # Make and train the non-spiking RNN control (same topology and size as the RSNN, ELU instead of LIF)
        with timer.stage("benchmark: rnn"):
            run_rnn_benchmark(processed, folds, run_dir, device, timer)

        # Neuron-matched pair: RSNN and RNN with as many hidden neurons as EEGNet / the converted SNN (2560)
        with timer.stage("benchmark: rsnn_wide"):
            run_rsnn_benchmark(processed, folds, run_dir, device, timer, wide=True)
        with timer.stage("benchmark: rnn_wide"):
            run_rnn_benchmark(processed, folds, run_dir, device, timer, wide=True)

    # Chained jobs / parallel workers: a worker that ran out of time ends cleanly with folds still pending. The analysis only
    # makes sense on the finished set, so it is skipped until every selected fold of every model has metrics, in ALL shards
    # (this is checked on the whole run, not only on this worker's folds). The first worker that sees the run finished takes
    # the lock, runs the analysis and writes the COMPLETE marker; any later worker or job with the same BENCH_RUN_ID exits.
    remaining = {m: len(pending_folds(run_dir, m, scope)) for m in MODELS}
    lock = run_dir / "ANALYSIS.lock"
    if any(remaining.values()):
        print(f"[chain] INCOMPLETE: folds still pending per model (whole run): {remaining}. Analysis skipped; "
              f"rerun/resubmit with BENCH_RUN_ID={run_dir.name} to continue.")
    elif (run_dir / "COMPLETE").exists() and not FINALIZE:
        print("[chain] already COMPLETE.")
    elif FINALIZE or _acquire_lock(lock):
        try:
            with timer.stage("collect_results"):
                collect_results(run_dir)
            with timer.stage("plot_benchmarks"):
                plot_benchmarks(run_dir)
            with timer.stage("plot_training"):
                plot_training(run_dir)

            # Comparative metrics between models (chance tests, paired Wilcoxon + Holm, bootstrap CIs, figures)
            with timer.stage("compare_models"):
                compare_models(run_dir)
            with timer.stage("plot_comparison"):
                plot_comparison(run_dir)

            # New: numbers behind every figure, pre/post fine-tuning, subject comparison
            _extra("export_figure_csvs", lambda: export_figure_csvs(run_dir))
            _extra("finetune_analysis", lambda: run_finetune_analysis(processed, all_folds, run_dir, device))
            _extra("subject_analysis", lambda: analyze_subjects(data, processed, run_dir))

            (run_dir / "COMPLETE").write_text(datetime.now().isoformat() + "\n")
            print(f"[chain] COMPLETE: all folds done and analysis written to {run_dir}")
        finally:
            if not FINALIZE:
                lock.unlink(missing_ok=True)  # released; COMPLETE (if written) stops everyone else
    else:
        print("[chain] all folds are done; another worker is running the analysis.")
finally:
    timer.finish()  # also runs after a crash / Ctrl-C, so partial timings are still saved
