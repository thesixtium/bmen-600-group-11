"""Training entry point: runs the benchmark folds of the selected models and saves their per-fold outputs.

Training only. No figures, no data-statistics plots and no comparison/analysis are made here; those stages read the saved
run directory afterwards (analysis workflow, make_plots.py). A training job therefore never depends on plotting.

Interface (environment variables, as used by run_benchmark.slurm):
  BENCH_MODELS=ann,snn            models to run, comma-separated, any of benchmark.MODELS
                                  (ann, snn, rsnn, rnn, rsnn_wide, rnn_wide). Default: all, in that order.
  BENCH_RUN_ID=<id>               run directory results/benchmark/<id>. Set it to resume, or to let several jobs / model
                                  chains contribute to ONE run. Default: a new timestamp id (prefixed "quick_" in QUICK mode).
  BENCH_SHARD / BENCH_N_SHARDS    this process runs folds i with i % N_SHARDS == SHARD of the fixed fold list (default: all).
  BENCH_TARGETS=1,2 BENCH_REPEATS=0   restrict the folds (smoke tests).
  BENCH_QUICK=1                   bug-testing mode (benchmark_config.py).
  BENCH_PATIENCE, BENCH_EVAL_BATCH (training.py), BENCH_DEADLINE_EPOCH (timing.py): unchanged.
  BENCH_FINALIZE=1                no longer used for analysis here: main.py then trains nothing and exits.

Each fold of each model is skipped when its metrics file exists, so rerunning one model never touches the other models and an
interrupted job resumes where it stopped. The SNN only converts ANN checkpoints of the SAME run whose ANN fold is finished; the
other SNN folds stay pending until the ANN has produced them. When every fold of every model of the run is done, COMPLETE is
written to the run directory (training finished; the analysis is run separately).
"""

import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="Montage name 'standard_1005'")

from benchmark import (
    MODELS,
    _atomic_write,
    pending_folds,
    prepare_run_dir,
    run_ann_benchmark,
    run_rnn_benchmark,
    run_rsnn_benchmark,
    run_snn_benchmark,
    select_device,
)
from download_data import download_data
from folds import filter_folds, generate_folds
from preprocess_data import preprocess_data
from timing import PipelineTimer
from training import EVAL_BATCH_SIZE, PRETRAIN_PATIENCE


def _env_int_list(name: str) -> list[int] | None:
    """Optional comma-separated integers, e.g. BENCH_TARGETS=1,2,3 to select folds (smoke tests)."""
    value = os.environ.get(name)
    return [int(v) for v in value.split(",")] if value else None


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_models() -> tuple[str, ...]:
    """BENCH_MODELS (comma-separated) -> selected models in the fixed MODELS order; unset = all models."""
    value = os.environ.get("BENCH_MODELS")
    if not value:
        return MODELS
    names = {v.strip() for v in value.split(",") if v.strip()}
    unknown = sorted(names - set(MODELS))
    if unknown or not names:
        raise SystemExit(f"BENCH_MODELS={value!r}: unknown model(s) {unknown}; choose from {', '.join(MODELS)}")
    return tuple(m for m in MODELS if m in names)


SHARD = _env_int("BENCH_SHARD", None)
N_SHARDS = _env_int("BENCH_N_SHARDS", 1) or 1
FINALIZE = os.environ.get("BENCH_FINALIZE") == "1"
IS_MAIN_SHARD = SHARD in (None, 0)
SELECTED = _env_models()

timer = PipelineTimer()


def _run_model(model: str, processed, folds, run_dir, device) -> None:
    """Dispatch to the benchmark runners of benchmark.py (no model logic here)."""
    if model == "ann":
        run_ann_benchmark(processed, folds, run_dir, device, timer)
    elif model == "snn":
        # The SNN needs each fold's fine-tuned ANN checkpoint of THIS run (run_snn_benchmark verifies the tag). A fold whose
        # ANN is not finished yet (no ANN metrics file, which is written after the checkpoint) is left pending, not failed,
        # so an SNN chain can run alongside or after the ANN chain.
        ann_waiting = set(pending_folds(run_dir, "ann", folds))
        blocked = [f for f in pending_folds(run_dir, "snn", folds) if f in ann_waiting]
        if blocked:
            print(f"[snn] {len(blocked)} fold(s) wait for their ANN fold of run {run_dir.name}; left pending.")
        run_snn_benchmark(processed, [f for f in folds if f not in ann_waiting], run_dir, device, timer)
    elif model in ("rsnn", "rsnn_wide"):
        run_rsnn_benchmark(processed, folds, run_dir, device, timer, wide=model == "rsnn_wide")
    elif model in ("rnn", "rnn_wide"):
        run_rnn_benchmark(processed, folds, run_dir, device, timer, wide=model == "rnn_wide")
    else:  # unreachable: _env_models() only returns names from MODELS
        raise ValueError(model)


try:
    if FINALIZE:
        print("[main] BENCH_FINALIZE=1: main.py only trains and the analysis is run separately; nothing to do.")
    else:
        with timer.stage("download_data"):
            data = download_data(redo=False)  # only files not already cached are downloaded

        with timer.stage("preprocess_data"):
            processed = preprocess_data(data)

        # Folds shared by all models. The full set is always saved; the shard and BENCH_TARGETS / BENCH_REPEATS only select
        # which folds this process runs.
        all_folds = generate_folds(processed)
        run_dir = prepare_run_dir(all_folds)  # refuses a run directory made with other settings
        scope = filter_folds(all_folds, targets=_env_int_list("BENCH_TARGETS"), repeats=_env_int_list("BENCH_REPEATS"))
        mine = [f for i, f in enumerate(all_folds) if i % N_SHARDS == (SHARD or 0)]
        folds = filter_folds(mine, targets=_env_int_list("BENCH_TARGETS"), repeats=_env_int_list("BENCH_REPEATS"))
        device = select_device()
        print(f"[main] run {run_dir.name} | models: {', '.join(SELECTED)}")
        print(f"[shard] shard {SHARD if SHARD is not None else 'all'} of {N_SHARDS}: {len(folds)} folds (whole run: {len(scope)})")
        if IS_MAIN_SHARD:
            settings = {"BENCH_PATIENCE (pretrain early stopping, None = off)": PRETRAIN_PATIENCE,
                        "BENCH_EVAL_BATCH": EVAL_BATCH_SIZE, "n_shards": N_SHARDS}
            _atomic_write(run_dir / "speed_settings.json", lambda fh: fh.write(json.dumps(settings, indent=1).encode()))

        timer.bind(run_dir, device, SELECTED, {m: len(pending_folds(run_dir, m, folds)) for m in SELECTED})
        for model in SELECTED:
            with timer.stage(f"benchmark: {model}"):
                _run_model(model, processed, folds, run_dir, device)

        # Progress of the selected models (selected folds) and of the whole run (all models, all folds).
        print(f"[chain] pending folds of the selected models (selected folds): "
              f"{ {m: len(pending_folds(run_dir, m, scope)) for m in SELECTED} }")
        run_left = {m: len(pending_folds(run_dir, m, all_folds)) for m in MODELS}
        if any(run_left.values()):
            print(f"[chain] run {run_dir.name} INCOMPLETE, pending folds per model: {run_left}. "
                  f"Resume with BENCH_RUN_ID={run_dir.name}.")
        else:
            if not (run_dir / "COMPLETE").exists():
                _atomic_write(run_dir / "COMPLETE", lambda fh: fh.write((datetime.now().isoformat() + "\n").encode()))
            print(f"[chain] COMPLETE: every fold of every model is trained in {run_dir}. Run the analysis separately.")
finally:
    timer.finish()  # also runs after a crash / Ctrl-C, so partial timings are still saved