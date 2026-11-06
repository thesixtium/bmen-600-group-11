#!/usr/bin/env bash
# Run one benchmark run id as parallel shards on this laptop, one process per group of target subjects.
# Each shard runs every model for its own subjects only, so no fold is trained twice; finished folds are skipped,
# so the same command also resumes. Sleep is blocked while a shard runs.
#   usage (from experiments/v1.1/code/):  ./launch_shards.sh <run id> [shards, default "1,2,3 4,5,6 7,8,9"]
set -euo pipefail
RUN_ID="${1:?usage: $0 <run id> [\"1,2,3 4,5,6 7,8,9\"]}"
SHARDS="${2:-1,2,3 4,5,6 7,8,9}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$HERE/rssns_for_mi_EEG"
PY="$HERE/../../.venv/bin/python"
RUN_DIR="$PROJECT/results/benchmark/$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$PROJECT"
# The sleep lock sometimes times out ("Failed to inhibit: Connection timed out") and the shard then never starts.
# Test it first; if it fails, fall back to an idle-only lock (screen idle does not suspend the laptop, but a
# sleep request from the power settings could). Other running shards may still hold a sleep lock.
if timeout 60 systemd-inhibit --what=sleep:idle --who=bmen600 --why="lock test" true 2>/dev/null; then
    WHAT="sleep:idle"
else
    WHAT="idle"
    echo "warning: sleep lock unavailable, using an idle-only lock"
fi
for targets in $SHARDS; do
    log="$RUN_DIR/run_log_t${targets//,/}.txt"
    setsid nohup systemd-inhibit --what="$WHAT" --who=bmen600 --why="BMEN600 benchmark shard $targets" \
        env BENCH_RUN_ID="$RUN_ID" BENCH_TARGETS="$targets" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
        "$PY" -u src/main.py >> "$log" 2>&1 < /dev/null &
    pid=$!
    sleep 45  # let each shard write folds.json / config.json before the next one starts
    # the pid is the systemd-inhibit wrapper; it exits at once if the lock or Python fails to start
    if kill -0 "$pid" 2>/dev/null; then
        echo "started shard $targets (lock: $WHAT) -> $log"
    else
        echo "shard $targets is not running after 45 s: see $log"; exit 1
    fi
done
