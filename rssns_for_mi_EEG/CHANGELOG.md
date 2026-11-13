# Task 1 CHANGELOG: numerical stability and configuration
(Static edits only; nothing was run or tested.)

## benchmark_config.py
- New module constant `GRAD_CLIP_NORM = 1.0` (global L2 norm; `None` disables).
- `TrainConfig.grad_clip_norm` defaults to it. `ANN_TRAIN`, `RSNN_TRAIN`, `RNN_TRAIN` inherit it, so ann, rsnn, rnn,
  rsnn_wide and rnn_wide share one threshold (wide models reuse the same train configs). Recorded in config.json.

## training.py
- `fit(..., grad_clip_norm)`: `clip_grad_norm_` between `backward()` and `optimizer.step()`.
- `pretrain_then_finetune` passes `cfg.grad_clip_norm` to both pretraining and fine-tuning (the only training path).
- New `NonFiniteTrainingError`. Once per epoch (no per-step host sync) train loss, validation loss and max pre-clip
  gradient norm are checked; on NaN/Inf it raises with the tag (model, subject, repeat, phase), epoch, losses, grad norm,
  lr, dropout, batch size, seed and the last 3 finite epochs. Nothing is replaced or skipped.
- History entries gain `grad_norm_max` and `clip_fraction` (existing keys unchanged; plotting code unaffected).
- `predict_logits` raises on non-finite logits instead of letting argmax mask them.

## ann_to_snn.py
- Conversion raises if any calibration scale (lambda) is non-finite. The SNN itself is not trained, so no clipping applies.

## Unchanged
Architectures, preprocessing, folds, epochs, lr, Adam, early stopping, loss, conversion method, evaluation.
Note: clipping can slightly change ANN/RSNN/RNN trajectories versus the earlier run (permitted).

# Task 2A CHANGELOG: benchmark execution logic
(Static edits only; nothing was run or tested.)

`benchmark.py` already existed in the supplied project (with the ANN, SNN, RSNN, RNN, RSNN-wide and RNN-wide runners), so it
was modified, not recreated. Models, training, folds, metrics and the conversion are unchanged.

## benchmark.py
- New `_atomic_write()`: temp file (per process) + `os.replace`. Used for metrics, predictions, histories, checkpoints and
  config.json, so an interrupted job cannot leave a half-written file that counts as "fold done" or feeds the SNN.
  Predictions are still written before metrics; the metrics file remains the done marker.
- New `_save_checkpoint()` replaces the three duplicated `torch.save` blocks (ann, rsnn/rsnn_wide, rnn/rnn_wide). The checkpoint
  keeps its `pretrained` / `finetuned` keys (finetune_plots.py unaffected) and gains a `fold` entry
  (model, run_id, subject, repeat, seed).
- `run_snn_benchmark`: after loading the fold's ANN checkpoint it checks that `fold` matches (model "ann", subject, repeat,
  seed); a mismatch or a missing tag raises a RuntimeError naming the file. A missing checkpoint already raised
  FileNotFoundError.
- Each metrics record gains `run_id` (extra column in results.csv; existing readers select columns by name).

## Unchanged (Task 1 intact)
training.py, benchmark_config.py, ann_to_snn.py, folds.py (fold generation, seeds, splits, save_folds), models.py.

## Notes
- ANN checkpoints made by the previous code have no `fold` tag, so the SNN will refuse them; rerun those ANN folds (Task 1
  already changed trajectories via clipping, so a fresh run is expected anyway).
- `prepare_run_dir` still rewrites config.json / folds.json on resume; with an unchanged config the content is identical.

# Task 2B CHANGELOG: main.py, run management, output organization
(Static edits only; nothing was run or tested.)

## src/main.py (training-only entry point)
- Removed every plotting / analysis stage: data_statistics (+ CSV), collect_results, plot_benchmarks, plot_training,
  compare_models, plot_comparison, export_figure_csvs, finetune analysis, subject analysis, and the ANALYSIS.lock logic.
  These modules are untouched; they are to be driven by the separate analysis workflow (make_plots.py, Task 3).
- New `BENCH_MODELS` (comma-separated subset of `benchmark.MODELS`; default all). Unknown names stop the process. Models run
  in the fixed MODELS order and dispatch to the existing `run_*_benchmark` functions (`wide=` for the wide variants).
- SNN: folds whose ANN fold of the same run is not finished are left pending (printed), not crashed on, so an SNN-only job
  can run alongside/after the ANN chain. The checkpoint verification itself stays in `run_snn_benchmark`.
- `COMPLETE` now means "every fold of every model of the run is trained" and is checked against ALL folds (previously the
  BENCH_TARGETS/REPEATS scope could mark a partial run complete). Written atomically; the analysis is not run.
- `BENCH_FINALIZE=1` no longer runs the analysis: main.py prints a message and trains nothing.
- `speed_settings.json` written with `_atomic_write`. Sharding, BENCH_TARGETS/REPEATS, deadline handling, timer: unchanged.
- Download (cache-only, `redo=False`) and preprocessing are kept: the project has no saved preprocessed data and the folds
  are built from it.

## src/benchmark.py (minimal adjustments)
- `prepare_run_dir`: refuses to reuse a run directory whose existing config.json or folds.json differs from what the current
  code/settings would write (`_refuse_mixing`), so old/new experiments or QUICK/full runs cannot be mixed. The check happens
  before any folder is created or file written.
- folds.json now written atomically (via `save_folds` to a per-process temp file, then `os.replace`); content unchanged.
- Default run id gets a `quick_` prefix in QUICK mode when no BENCH_RUN_ID is given (matches run_benchmark.slurm).
- `run_snn_benchmark`: the ANN checkpoint's `run_id` tag must equal the current run, so a checkpoint from another run is
  never used.

## Unchanged
Models, preprocessing, folds/splits/seeds, epochs, lr, optimizer, early stopping, loss, conversion, metrics, evaluation, all
Task 1 safeguards, all other Task 2A behaviour, timing.py, training.py, all plotting/analysis modules.

# Task 2C CHANGELOG: SLURM execution and submission chains
(Static edits only; nothing was run, submitted or tested. Append to CHANGELOG.md.)

## ARC settings (taken unchanged from the supplied run_benchmark.slurm)
Partition gpu-v100, 1 node, 1 task, 8 CPUs, 64G, 1 GPU, 24:00:00, mail settings, PROJECT_DIR, PY_MODULE python/3.12.5,
per-job /tmp venv with the same pip pins, GPU check, 40 min setup watchdog, optional MPS, 30 min deadline margin
(BENCH_DEADLINE_EPOCH). No account line existed and none was added.

## run_benchmark.slurm
- One shared SLURM file for all chains (resource lines unchanged). New `CHAIN_NAME` (default `all`); `BENCH_MODELS` reaches
  src/main.py through the environment (Task 2B interface).
- Log names carry the chain: `logs/<chain>_job<g>_<jobid>.txt`, `logs/<chain>_shard<k>.txt`,
  `logs/pip_freeze_<chain>_job<g>.txt`, so two chains on one run id never append to the same log. Header prints chain/models.
- `BENCH_FINALIZE` handling removed (main.py no longer runs analysis). COMPLETE (written by main.py when every fold of every
  model is trained) still ends a job at once.

## submit_chain.sh
- Exports `BENCH_MODELS` and `CHAIN_NAME` (via `--export=ALL`; BENCH_MODELS contains commas) and sets
  `--job-name=bench_<chain>`, so the slurm .out files and `scancel --name` are per chain. `quick_` run-id prefix when
  BENCH_QUICK=1. Without BENCH_MODELS it still runs all models. Dependencies stay within one GPU-chain of one model chain.

## New: submit_chain_ann_snn.sh, submit_chain_rsnn.sh, submit_chain_rnn.sh, submit_chain_rsnn_wide.sh, submit_chain_rnn_wide.sh
One-line wrappers that set BENCH_MODELS / CHAIN_NAME and call submit_chain.sh. ANN and SNN share one chain; within each worker
main.py runs ann before snn on the same folds, and the SNN only uses ANN checkpoints tagged with the same run id.

## Unchanged
All Python files (Tasks 1, 2A, 2B), resources, time limit, margin, scientific settings.