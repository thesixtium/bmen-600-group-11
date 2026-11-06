#!/usr/bin/env bash
# Run one benchmark run id as parallel shards on this laptop, one process per group of target subjects.
# Each shard runs every model for its own subjects only, so no fold is trained twice; finished folds are skipped,
# so the same command also resumes. Sleep is blocked while a shard runs.
#   usage (from experiments/v1/code/):  ./launch_shards.sh <run id> [shards, default "1,2,3 4,5,6 7,8,9"]
set -euo pipefail
RUN_ID="${1:?usage: $0 <run id> [\"1,2,3 4,5,6 7,8,9\"]}"
SHARDS="${2:-1,2,3 4,5,6 7,8,9}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$HERE/rssns_for_mi_EEG"
PY="$HERE/../../.venv/bin/python"
RUN_DIR="$PROJECT/results/benchmark/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$PROJECT"
for targets in $SHARDS; do
    log="$RUN_DIR/run_log_t${targets//,/}.txt"
    setsid nohup systemd-inhibit --what=sleep:idle --who=bmen600 --why="BMEN600 benchmark shard $targets" \
        env BENCH_RUN_ID="$RUN_ID" BENCH_TARGETS="$targets" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
        "$PY" -u src/main.py >> "$log" 2>&1 < /dev/null &
    echo "started shard $targets -> $log"
    sleep 45  # let each shard write folds.json / config.json before the next one starts
done
