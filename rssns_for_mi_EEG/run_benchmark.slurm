#!/bin/bash
#SBATCH --job-name=rsnn_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aleksander.berezowsk@ucalgary.ca

# -----------------------------------------------------------------------------
# Benchmark of EEGNet ANN / converted SNN / RSNN on BNCI2014_001
# (9 subjects x 10 seeded repeats = 90 folds shared by all three models).
#
# Submit from the project folder:
#     cd ~/bmen-600-group-11/rssns_for_mi_EEG
#     sbatch run_benchmark.slurm
#
# Everything printed (setup, pip, Python stdout + stderr) is ALSO written to
#     results/benchmark/<run id>/run_log.txt
# so you can:   cat results/benchmark/<run id>/run_log.txt
# (the same text also lands in slurm_rsnn_benchmark_<jobid>.out)
#
# Optional overrides (sbatch passes your shell environment through):
#   BENCH_RUN_ID=20261020_120000 sbatch run_benchmark.slurm   resume a run (finished folds are skipped)
#   BENCH_QUICK=1 BENCH_TARGETS=1 BENCH_REPEATS=0 sbatch run_benchmark.slurm   fast smoke test
#   PY_MODULE=python/3.11.x sbatch run_benchmark.slurm        python module to load (see `module avail python`)
# -----------------------------------------------------------------------------

# ----------------------------- settings --------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/home/aleksander.berezowsk/bmen-600-group-11/rssns_for_mi_EEG}"
PY_MODULE="${PY_MODULE:-}"      # leave empty if python3 on the node is already 3.10-3.12

# ----------------------------- sanity check ----------------------------------
if [ ! -f "$PROJECT_DIR/src/main.py" ]; then
    echo "ERROR: $PROJECT_DIR/src/main.py not found. Fix PROJECT_DIR at the top of this script."
    exit 1
fi

# ----------------------------- run id + log file -----------------------------
# Same run id for the results folder and the log, so the log sits next to the results.
PREFIX=""
[ "${BENCH_QUICK:-0}" = "1" ] && PREFIX="quick_"
export BENCH_RUN_ID="${BENCH_RUN_ID:-${PREFIX}$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/results/benchmark/$BENCH_RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run_log.txt"

# From here on, stdout and stderr go to the terminal/slurm .out AND to the log file.
exec > >(tee -a "$LOG") 2>&1

echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Started   : $(date)"
echo "Project   : $PROJECT_DIR"
echo "Run ID    : $BENCH_RUN_ID"
echo "Log file  : $LOG"
echo "Git commit: $(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "BENCH_QUICK=${BENCH_QUICK:-0} BENCH_TARGETS=${BENCH_TARGETS:-all} BENCH_REPEATS=${BENCH_REPEATS:-all}"
echo "-----------------------------------------------------------------------"

# ----------------------------- fresh environment -----------------------------
# ARC is treated as a clean slate every job: a new venv on the node-local disk
# (/tmp is wiped when the job ends), everything pip-installed from scratch.
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

VENV_DIR="/tmp/venv_${SLURM_JOB_ID}"
python3 -m venv "$VENV_DIR" || { echo "ERROR: venv creation failed"; exit 1; }
source "$VENV_DIR/bin/activate"
trap 'rm -rf "$VENV_DIR"' EXIT

python -m pip install --upgrade pip wheel "setuptools==69.5.1" || exit 1
# torch 2.4.0 from PyPI bundles its own CUDA runtime, so no `module load cuda` is needed.
pip install torch==2.4.0 || exit 1
pip install snntorch moabb mne numpy scipy pandas scikit-learn matplotlib || exit 1
pip freeze > "$RUN_DIR/pip_freeze.txt"

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
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1      # log lines appear immediately instead of at the end

# ----------------------------- run -------------------------------------------
cd "$PROJECT_DIR" || exit 1
python -u src/main.py
STATUS=$?

echo "-----------------------------------------------------------------------"
echo "Finished  : $(date)"
echo "Exit code : $STATUS"
sleep 3   # let tee flush the last lines to the log file
exit $STATUS
