#!/usr/bin/env bash
# One benchmark shard under a sleep lock, started by launch_shards.sh in its own tmux window.
#   usage: shard.sh BENCH_RUN_ID=<run id> BENCH_TARGETS=<targets> <log file>
# The arguments are written as KEY=value on purpose: the finisher, stop_v1.4.sh and run_v1.4.sh find shards with
# pgrep -f "BENCH_RUN_ID=<id> ", so this wrapper is found even while it waits for a lock.
#
# Why the retries: systemd-inhibit sometimes fails with "Failed to inhibit: Connection timed out" (logind busy, seen
# right after a boot on 2026-11-03). It then exits 1 without starting the shard. This script tries the sleep lock
# LOCK_TRIES times, then an idle-only lock LOCK_TRIES times, waiting LOCK_WAIT seconds between tries.
# A lock failure is recognised by its message being the first thing the attempt wrote to the log: main.py always
# prints its own lines first, so a failure of the training itself is never mistaken for a lock failure and retried.
set -uo pipefail
RUN_ID="${1#BENCH_RUN_ID=}"
TARGETS="${2#BENCH_TARGETS=}"
LOG="${3:?usage: $0 BENCH_RUN_ID=<run id> BENCH_TARGETS=<targets> <log file>}"
[ -n "$RUN_ID" ] && [ -n "$TARGETS" ] || { echo "usage: $0 BENCH_RUN_ID=<run id> BENCH_TARGETS=<targets> <log file>"; exit 2; }
LOCK_TRIES="${LOCK_TRIES:-5}"
LOCK_WAIT="${LOCK_WAIT:-30}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(cd "$HERE/../.." && pwd)/.venv/bin/python"
cd "$HERE/rssns_for_mi_EEG"
touch "$LOG"

attempt() {  # attempt <lock type>: run the shard once under that lock; returns systemd-inhibit's exit code
    # env execs python, so python is the direct child of systemd-inhibit (stop_v1.4.sh signals it that way).
    systemd-inhibit --what="$1" --who=bmen600 --why="BMEN600 benchmark shard $TARGETS" \
        env BENCH_RUN_ID="$RUN_ID" BENCH_TARGETS="$TARGETS" \
            ${BENCH_QUICK:+BENCH_QUICK=$BENCH_QUICK BENCH_REPEATS=0} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
            "$PY" -u src/main.py 2>&1 | tee -a "$LOG"
    return "${PIPESTATUS[0]}"
}

rc=1
for what in sleep:idle idle; do
    for try in $(seq "$LOCK_TRIES"); do
        offset=$(stat -c %s "$LOG")
        attempt "$what"; rc=$?
        # Lock failure: non-zero exit and the attempt's first log line is systemd-inhibit's message.
        if [ "$rc" -ne 0 ] && tail -c +"$((offset + 1))" "$LOG" | head -1 | grep -q '^Failed to inhibit'; then
            echo "lock $what failed (try $try of $LOCK_TRIES); next try in ${LOCK_WAIT} s" | tee -a "$LOG"
            sleep "$LOCK_WAIT"
            continue
        fi
        echo "shard $TARGETS ended (exit $rc, lock $what)"
        exit "$rc"
    done
    [ "$what" = sleep:idle ] && echo "sleep lock unavailable after $LOCK_TRIES tries; trying an idle-only lock" | tee -a "$LOG"
done
echo "shard $TARGETS ended (exit $rc): no lock after $((2 * LOCK_TRIES)) tries, the shard did not start" | tee -a "$LOG"
exit "$rc"
