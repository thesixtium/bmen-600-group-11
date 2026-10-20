import os
import warnings

warnings.filterwarnings("ignore", message="Montage name 'standard_1005'")

from benchmark import (
    MODELS,
    collect_results,
    pending_folds,
    prepare_run_dir,
    run_ann_benchmark,
    run_rsnn_benchmark,
    run_snn_benchmark,
    select_device,
)
from benchmark_plots import plot_benchmarks
from comparison import compare_models
from comparison_plots import plot_comparison
from data_statistics import data_statistics
from download_data import download_data
from folds import filter_folds, generate_folds
from preprocess_data import preprocess_data
from timing import PipelineTimer


def _env_int_list(name: str) -> list[int] | None:
    """Optional comma-separated integers, e.g. BENCH_TARGETS=1,2,3 to shard folds over several jobs."""
    value = os.environ.get(name)
    return [int(v) for v in value.split(",")] if value else None


timer = PipelineTimer()

try:
    with timer.stage("download_data"):
        data = download_data(redo=False)

    with timer.stage("preprocess_data"):
        processed = preprocess_data(data)

    with timer.stage("data_statistics"):
        data_statistics(data, processed)

    # Folds shared by all three models. The full set is always saved; BENCH_TARGETS / BENCH_REPEATS only select
    # which folds this process runs. Set BENCH_RUN_ID to resume or to share one run directory between jobs.
    all_folds = generate_folds(processed)
    run_dir = prepare_run_dir(all_folds)
    folds = filter_folds(all_folds, targets=_env_int_list("BENCH_TARGETS"), repeats=_env_int_list("BENCH_REPEATS"))
    device = select_device()
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

    with timer.stage("collect_results"):
        collect_results(run_dir)
    with timer.stage("plot_benchmarks"):
        plot_benchmarks(run_dir)

    # Comparative metrics between models (chance tests, paired Wilcoxon + Holm, bootstrap CIs, figures)
    with timer.stage("compare_models"):
        compare_models(run_dir)
    with timer.stage("plot_comparison"):
        plot_comparison(run_dir)
finally:
    timer.finish()  # also runs after a crash / Ctrl-C, so partial timings are still saved
