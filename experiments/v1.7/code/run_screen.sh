#!/usr/bin/env bash
# Run one screen: every line of jobs/<name>.txt is one screen.py call (model, fold, variant, arm), 3 at a time on the
# GPU. Writes results/<name>/*.json, one log per job in results/<name>/logs/, and results/<name>/DONE at the end.
# Finished jobs (result file present) are skipped, so the same command resumes.
#   usage (from this folder):  ./run_screen.sh v1.5        results go to experiments/<name>/results/
set -uo pipefail
NAME="${1:?usage: $0 <screen name, e.g. v1.5>}"
cd "$(dirname "$0")"
PY="${PYTHON:-python}"   # the Python with torch 2.4.0 (set PYTHON=... to choose another)
mkdir -p "../../$NAME/results/logs"
rm -f "../../$NAME/results/DONE"
[ "$NAME" = v1.5 ] && { "$PY" -u eval_saved.py || echo "FAILED eval_saved.py"; }   # part A: saved checkpoints
grep -v '^\s*\(#\|$\)' "jobs/$NAME.txt" | xargs -P 3 -I{} bash -c '
    py=$1; name=$2; job=$3
    set -- $job
    variant=$2; arm=$4; model=$6; fold=$8
    stem="${variant}_${arm}_${model}_s${fold%:*}_r${fold#*:}"
    [ -e "../../$name/results/$stem.json" ] && { echo "skip $stem (done)"; exit 0; }
    if "$py" -u screen.py $job --out "$name" > "../../$name/results/logs/$stem.txt" 2>&1; then echo "ok   $stem"
    else echo "FAILED $stem (see ../../$name/results/logs/$stem.txt)"; fi' _ "$PY" "$NAME" {}
n_jobs=$(grep -cv '^\s*\(#\|$\)' "jobs/$NAME.txt"); n_done=$(ls "../../$NAME/results"/v1.*.json 2>/dev/null | wc -l)
echo "$(date -Iseconds) screen $NAME: $n_done of $n_jobs jobs have results" | tee "../../$NAME/results/DONE"
