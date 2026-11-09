#!/bin/bash
#SBATCH --job-name=rsnn_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aleksander.berezowsk@ucalgary.ca

# -----------------------------------------------------------------------------
# Benchmark of EEGNet ANN / converted SNN / RSNN / RNN (+ neuron-matched pair) on BNCI2014_001
# (9 subjects x 10 seeded repeats = 90 folds shared by all models).
#
# PARALLEL LAYOUT. The 90 folds are split into N_SHARDS = BENCH_N_JOBS x WORKERS_PER_JOB shards (fold i belongs to shard
# i % N_SHARDS). One SLURM job owns ONE GPU and runs WORKERS_PER_JOB worker processes on it (shard = BENCH_JOB_INDEX *
# WORKERS_PER_JOB + worker). Several processes on one GPU help because the small recurrent models are launch-bound (the GPU
# is almost idle). ARC allows 4 GPUs per user, so submit 4 chains with:
#
#     cd ~/bmen-600-group-11/rssns_for_mi_EEG
#     ./submit_chain.sh 4            # 4 GPUs x 4 chained 24 h jobs, WORKERS_PER_JOB=3 -> 12 shards
#
# Do not call sbatch on this file directly for the full run (BENCH_JOB_INDEX / BENCH_N_JOBS would be unset: one job, one
# shard set, the other GPUs unused). Per-worker logs: results/benchmark/<run id>/logs/shard<k>.txt
# Setup log of every job:                           results/benchmark/<run id>/logs/job<g>_<jobid>.txt
#
# Optional overrides (environment passed through by submit_chain.sh):
#   WORKERS_PER_JOB=4          workers per GPU (default 3; 8 CPUs / worker count threads each)
#   BENCH_PATIENCE=100         pretraining early stopping (changes the protocol! see how_to_run / the answer notes)
#   BENCH_EVAL_BATCH=288       validation batch size (no effect on results, only speed)
#   USE_MPS=1                  start CUDA MPS so the workers really share the GPU concurrently (optional, if available)
#   BENCH_QUICK=1 BENCH_TARGETS=1 BENCH_REPEATS=0 WORKERS_PER_JOB=1 sbatch run_benchmark.slurm   fast smoke test (one
#                              worker, one fold; the analysis then runs on that single fold, some plots may be degenerate)
#   BENCH_FINALIZE=1 BENCH_RUN_ID=<id>   skip training, only rerun the analysis of a finished run
#   PY_MODULE=python/3.11.x    force a python module
# -----------------------------------------------------------------------------

# ----------------------------- settings --------------------------------------
JOB_START=$(date +%s)
TIME_LIMIT_S=86400                       # keep in sync with --time=24:00:00 above
DEADLINE_MARGIN_MIN="${DEADLINE_MARGIN_MIN:-30}"   # stop starting new folds this long before the limit
PROJECT_DIR="${PROJECT_DIR:-/home/aleksander.berezowsk/bmen-600-group-11/rssns_for_mi_EEG}"
# python/3.12.5 is what the auto-detection found on every earlier run. The auto-detection ('module avail') once hung for the
# whole 24 h of a job on node fg6, so it is skipped by default. Set PY_MODULE="" to use the old auto-detection.
PY_MODULE="${PY_MODULE-python/3.12.5}"
BENCH_JOB_INDEX="${BENCH_JOB_INDEX:-0}"
BENCH_N_JOBS="${BENCH_N_JOBS:-1}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-3}"
N_SHARDS=$(( BENCH_N_JOBS * WORKERS_PER_JOB ))
THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / WORKERS_PER_JOB ))
[ "$THREADS" -lt 1 ] && THREADS=1

# ----------------------------- sanity check ----------------------------------
if [ ! -f "$PROJECT_DIR/src/main.py" ]; then
    echo "ERROR: $PROJECT_DIR/src/main.py not found. Fix PROJECT_DIR at the top of this script."
    exit 1
fi

# ----------------------------- run id + log file -----------------------------
PREFIX=""
[ "${BENCH_QUICK:-0}" = "1" ] && PREFIX="quick_"
export BENCH_RUN_ID="${BENCH_RUN_ID:-${PREFIX}$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/results/benchmark/$BENCH_RUN_ID"
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/job${BENCH_JOB_INDEX}_${SLURM_JOB_ID}.txt"

# From here on, stdout and stderr go to the terminal/slurm .out AND to this job's setup log.
exec > >(tee -a "$LOG") 2>&1

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Project   : $PROJECT_DIR"
echo "Run ID    : $BENCH_RUN_ID"
echo "Log file  : $LOG"
echo "Git commit: $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Layout    : job index $BENCH_JOB_INDEX of $BENCH_N_JOBS, $WORKERS_PER_JOB workers per GPU, $N_SHARDS shards, $THREADS threads/worker"
echo "BENCH_QUICK=${BENCH_QUICK:-0} BENCH_TARGETS=${BENCH_TARGETS:-all} BENCH_REPEATS=${BENCH_REPEATS:-all} BENCH_PATIENCE=${BENCH_PATIENCE:-off} BENCH_FINALIZE=${BENCH_FINALIZE:-0}"
echo "-----------------------------------------------------------------------"

# ----------------------------- already finished? -----------------------------
if [ -f "$RUN_DIR/COMPLETE" ] && [ "${BENCH_FINALIZE:-0}" != "1" ]; then
    echo "Run $BENCH_RUN_ID is already COMPLETE ($(cat "$RUN_DIR/COMPLETE")). Nothing to do."
    exit 0
fi
# main.py stops starting new folds once the next one would not finish before this deadline, so the job ends cleanly
# (no fold is lost to the hard time limit) and the next chained job resumes.
export BENCH_DEADLINE_EPOCH=$(( JOB_START + TIME_LIMIT_S - DEADLINE_MARGIN_MIN * 60 ))
echo "Deadline  : $(date -d @"$BENCH_DEADLINE_EPOCH") (no new folds after this; margin ${DEADLINE_MARGIN_MIN} min)"

# ----------------------------- setup watchdog --------------------------------
# Setup (python module, venv, pip, GPU check) normally takes ~3 minutes. If it has not finished after 40 minutes something
# is hung (a stuck module command cost one job its whole 24 h): cancel this job so the next chained job starts instead.
SETUP_OK="/tmp/setup_ok_${SLURM_JOB_ID}"
( sleep 2400; if [ ! -f "$SETUP_OK" ]; then echo "WATCHDOG: setup not finished after 40 min, cancelling job $SLURM_JOB_ID"; scancel "$SLURM_JOB_ID"; fi ) &

# ----------------------------- python selection ------------------------------
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "$f" ] && source "$f" && break
    done
fi

python_ok() {   # true if the current python3 is 3.10-3.12
    python3 -c 'import sys; v=sys.version_info[:2]; sys.exit(0 if (3,10)<=v<=(3,12) else 1)' 2>/dev/null
}

if [ -z "$PY_MODULE" ] && ! python_ok; then
    echo "python3 is $(python3 --version 2>&1), not 3.10-3.12: searching for a module..."
    PY_MODULE="$(module -t avail python 2>&1 \
        | grep -E '^python/3\.(10|11|12)(\.|$)' \
        | sed 's/(.*//' | sort -V | tail -1)"
    if [ -z "$PY_MODULE" ]; then
        echo "ERROR: no python 3.10-3.12 module found. Output of 'module avail python':"
        module avail python 2>&1
        echo "Set PY_MODULE explicitly, or create a conda env with python 3.11."
        exit 1
    fi
    echo "Auto-detected module: $PY_MODULE"
fi

if [ -n "$PY_MODULE" ]; then
    module load "$PY_MODULE" || { echo "ERROR: could not load module $PY_MODULE"; exit 1; }
fi

python3 - <<'EOF' || exit 1
import sys
v = sys.version_info[:2]
print("System python:", sys.version.split()[0])
if not ((3, 10) <= v <= (3, 12)):
    print("ERROR: need Python 3.10-3.12 (the code uses 'int | None' syntax; torch 2.4.0 has no 3.13 wheels).")
    print("Run `module avail python` on ARC and set PY_MODULE at the top of this script.")
    sys.exit(1)
EOF

# ----------------------------- fresh environment -----------------------------
VENV_DIR="/tmp/venv_${SLURM_JOB_ID}"
MPS_DIR=""
cleanup() {
    rm -rf "$VENV_DIR" "$SETUP_OK"
    if [ -n "$MPS_DIR" ]; then echo quit | nvidia-cuda-mps-control 2>/dev/null; rm -rf "$MPS_DIR"; fi
}
python3 -m venv "$VENV_DIR" || { echo "ERROR: venv creation failed"; exit 1; }
source "$VENV_DIR/bin/activate"
trap cleanup EXIT

python -m pip install --upgrade pip wheel "setuptools==69.5.1" || exit 1
pip install torch==2.4.0 || exit 1
pip install snntorch moabb mne numpy scipy pandas scikit-learn matplotlib || exit 1
pip freeze > "$RUN_DIR/logs/pip_freeze_job${BENCH_JOB_INDEX}.txt"

echo "-----------------------------------------------------------------------"
nvidia-smi
python - <<'EOF' || exit 1
import sys, torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: no GPU visible to torch; refusing to run the benchmark on CPU.")
    sys.exit(1)
print("GPU:", torch.cuda.get_device_name(0))
EOF
echo "-----------------------------------------------------------------------"

# ----------------------------- runtime settings ------------------------------
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export PYTHONUNBUFFERED=1      # log lines appear immediately instead of at the end

# Optional: CUDA MPS lets the worker processes run kernels on the GPU at the same time (otherwise they are time-sliced).
if [ "${USE_MPS:-0}" = "1" ] && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    MPS_DIR="/tmp/mps_${SLURM_JOB_ID}"
    mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"
    nvidia-cuda-mps-control -d && echo "CUDA MPS daemon started" || echo "WARNING: could not start MPS, continuing without"
fi

# ----------------------------- run -------------------------------------------
touch "$SETUP_OK"   # tells the watchdog that setup finished
cd "$PROJECT_DIR" || exit 1
PIDS=()
for w in $(seq 0 $(( WORKERS_PER_JOB - 1 ))); do
    SHARD=$(( BENCH_JOB_INDEX * WORKERS_PER_JOB + w ))
    echo "Starting worker $w -> shard $SHARD of $N_SHARDS (log: $RUN_DIR/logs/shard${SHARD}.txt)"
    BENCH_SHARD=$SHARD BENCH_N_SHARDS=$N_SHARDS python -u src/main.py >> "$RUN_DIR/logs/shard${SHARD}.txt" 2>&1 &
    PIDS+=($!)
    sleep 20   # stagger the start (data loading) so the workers do not read/download the dataset at the same moment
done
STATUS=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || STATUS=$?
done

echo "-----------------------------------------------------------------------"
echo "Finished  : $(date)"
echo "Exit code : $STATUS"
sleep 3   # let tee flush the last lines to the log file
exit $STATUS
