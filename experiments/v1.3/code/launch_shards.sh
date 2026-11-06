#!/usr/bin/env bash
# Run one benchmark run id as parallel shards on this laptop, one process per group of target subjects.
# Each shard runs every model for its own subjects only, so no fold is trained twice; finished folds are skipped,
# so the same command also resumes. Sleep is blocked while a shard runs (if the lock is available).
#
# The shards run as windows of the tmux session "bmen600-<run id>" on the tmux socket "bmen600". The tmux server is
# started in its own systemd user scope (systemd-run --user --scope), not inside the terminal, and lingering is on
# for this account (loginctl enable-linger), so the shards keep running after a logout or a switch to another account.
#   watch:  tmux -L bmen600 attach -t bmen600-<run id>     (detach: Ctrl+b then d; never Ctrl+c inside a shard window)
#   list:   tmux -L bmen600 ls
#   usage (from experiments/v1.3/code/):  ./launch_shards.sh <run id> [shards, default "1,2,3 4,5,6 7,8,9"]
#   smoke test in tmux: BENCH_QUICK=1 ./launch_shards.sh quick_<id> "1"   (quick mode, repeat 0 only)
set -euo pipefail
RUN_ID="${1:?usage: $0 <run id> [\"1,2,3 4,5,6 7,8,9\"]}"
SHARDS="${2:-1,2,3 4,5,6 7,8,9}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$HERE/rssns_for_mi_EEG"
PY="$(cd "$HERE/../.." && pwd)/.venv/bin/python"
RUN_DIR="$PROJECT/results/benchmark/$RUN_ID"
TMUX_SOCKET="bmen600"
SESSION="bmen600-$RUN_ID"
mkdir -p "$RUN_DIR"

[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ] \
    || echo "warning: lingering is off; a logout will stop the shards (fix: loginctl enable-linger $USER)"

# The sleep lock sometimes times out ("Failed to inhibit: Connection timed out") and the shard then never starts.
# Test it first; if it fails, fall back to an idle-only lock (screen idle does not suspend the laptop, but a
# sleep request from the power settings could). Other running shards may still hold a sleep lock.
if timeout 60 systemd-inhibit --what=sleep:idle --who=bmen600 --why="lock test" true 2>/dev/null; then
    WHAT="sleep:idle"
else
    WHAT="idle"
    echo "warning: sleep lock unavailable, using an idle-only lock"
fi

# One tmux session per run. Its first window only explains the layout; the server lives in its own systemd scope.
if ! tmux -L "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null; then
    systemd-run --user --scope --quiet --unit="bmen600-tmux-$RUN_ID-$(date +%s)" \
        tmux -L "$TMUX_SOCKET" new-session -d -s "$SESSION" -n info -c "$HERE" \
        "echo 'run $RUN_ID: one window per shard + the finisher (Ctrl+b then a window number; Ctrl+b then d to detach)'; exec bash"
fi

for targets in $SHARDS; do
    log="$RUN_DIR/run_log_t${targets//,/}.txt"
    window="s${targets//,/}"
    # "BENCH_RUN_ID=<id> " stays on the systemd-inhibit command line: the finisher and stop script find shards by it.
    # Output goes to the log file and to the window (tee); after the shard ends the window stays open with a shell.
    tmux -L "$TMUX_SOCKET" new-window -d -t "$SESSION" -n "$window" -c "$PROJECT" \
        "bash -c 'systemd-inhibit --what=$WHAT --who=bmen600 --why=\"BMEN600 benchmark shard $targets\" \
            env BENCH_RUN_ID=$RUN_ID BENCH_TARGETS=$targets ${BENCH_QUICK:+BENCH_QUICK=$BENCH_QUICK BENCH_REPEATS=0} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 \
            $PY -u src/main.py 2>&1 | tee -a $log; echo \"shard $targets ended (exit \${PIPESTATUS[0]})\"'; exec bash"
    sleep 45  # let each shard write folds.json / config.json before the next one starts
    if pgrep -f "BENCH_RUN_ID=$RUN_ID BENCH_TARGETS=$targets " > /dev/null; then
        echo "started shard $targets (lock: $WHAT) -> tmux window $SESSION:$window, log $log"
    elif tmux -L "$TMUX_SOCKET" capture-pane -p -t "$SESSION:$window" | grep -q "shard $targets ended (exit 0)"; then
        echo "shard $targets already finished (exit 0; all its folds were done, or a quick run)"
    else
        echo "shard $targets is not running after 45 s: see $log and tmux window $SESSION:$window"; exit 1
    fi
done
echo "watch: tmux -L $TMUX_SOCKET attach -t $SESSION"
