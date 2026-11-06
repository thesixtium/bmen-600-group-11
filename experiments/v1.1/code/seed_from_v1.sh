#!/usr/bin/env bash
# Prepare a v1.1 run folder so that only the RSNN trains.
# The pipeline counts a fold as done when metrics/<model>/s<S>_r<R>.json exists. This script copies the per-fold
# metrics, predictions and histories of ann, snn and rnn from the v1 run, so main.py skips those three models and
# trains the RSNN on all 90 folds. The fold seeds are the same as in v1, so the copied folds are the same folds.
# It also links the v1 data download, so nothing is downloaded again.
#   usage (from experiments/v1.1/code/):  ./seed_from_v1.sh <new run id> [v1 run id, default 20261029_141407]
set -euo pipefail
RUN_ID="${1:?usage: $0 <new run id> [v1 run id]}"
V1_ID="${2:-20261029_141407}"
HERE="$(cd "$(dirname "$0")" && pwd)"
V1_PROJECT="$HERE/../local-train-v1/rssns_for_mi_EEG"
V1_RUN="$V1_PROJECT/results/benchmark/$V1_ID"
RUN_DIR="$HERE/rssns_for_mi_EEG/results/benchmark/$RUN_ID"

[ -d "$V1_RUN/metrics/rsnn" ] || { echo "v1 run not found: $V1_RUN"; exit 1; }
[ -e "$RUN_DIR" ] && { echo "run folder already exists: $RUN_DIR (pick a new run id)"; exit 1; }

# data: reuse the v1 download (MOABB cache)
if [ ! -e "$HERE/rssns_for_mi_EEG/data" ]; then
    ln -s "$V1_PROJECT/data" "$HERE/rssns_for_mi_EEG/data"
    echo "linked data -> $V1_PROJECT/data"
fi

for model in ann snn rnn; do
    for sub in metrics predictions histories; do
        # the SNN has no histories (it is converted, not trained): v1 has an empty histories/snn folder, so skip empty folders
        [ -n "$(ls -A "$V1_RUN/$sub/$model" 2>/dev/null)" ] || continue
        mkdir -p "$RUN_DIR/$sub/$model"
        cp -p "$V1_RUN/$sub/$model/"* "$RUN_DIR/$sub/$model/"
    done
    n=$(ls "$RUN_DIR/metrics/$model" | wc -l)
    echo "$model: $n folds copied from v1"
    [ "$n" -eq 90 ] || { echo "expected 90 $model folds in the v1 run, found $n: stop"; exit 1; }
done
mkdir -p "$RUN_DIR/metrics/rsnn"
echo "rsnn: $(ls "$RUN_DIR/metrics/rsnn" | wc -l) folds done (expected 0)"
# provenance: which files of this run were trained in v1 rather than in this run
{
    echo "ann, snn and rnn metrics, predictions and histories copied from v1 run $V1_ID on $(date -Iseconds)"
    echo "source: $V1_RUN"
    echo "only the rsnn is trained in this run"
} > "$RUN_DIR/SEEDED_FROM_V1.txt"
echo "Seeded $RUN_DIR from v1 run $V1_ID. Next: ./launch_shards.sh $RUN_ID"
