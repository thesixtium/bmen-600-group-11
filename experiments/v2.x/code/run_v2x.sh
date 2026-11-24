#!/usr/bin/env bash
# Runs the v2.x screens. Every line of jobs.txt is "<script> <arm> <target>" (repeat 0); 3 jobs at a time on one GPU.
# A job whose result file exists is skipped, so the same command resumes. Then the reservoir screen, fold by fold.
# Needs: the data cache of experiments/v1.5/code (python screen_common.py there), and for the frozen_* arms the
# EEGNet checkpoints of the v1 run in experiments/v1/results/checkpoints/ann/.
#   usage (from this folder):  PY=python ./run_v2x.sh
set -uo pipefail
cd "$(dirname "$0")"
export PY="${PY:-python}"
mkdir -p ../logs ../results
xargs -P 3 -L 1 bash -c '
    [ -e "../results/${1}_s${2}_r0.json" ] && { echo "skip $1 s$2"; exit 0; }
    if "$PY" -B -u "$0" --arm "$1" --fold "$2:0" > "../logs/${1}_s${2}.txt" 2>&1; then echo "ok   $1 s$2"; else echo "FAILED $1 s$2"; fi' < jobs.txt
for t in 1 2 3 4 5 6 7 8 9; do "$PY" -B -u screen_reservoir_v2x.py --fold "$t:0" > "../logs/reservoir_s$t.txt" 2>&1 && echo "ok   reservoir s$t" || echo "FAILED reservoir s$t"; done
"$PY" -B analyze_v2x.py
