#!/bin/bash
# ANN -> SNN chain. Every worker trains its ANN folds first, then converts them to SNNs (src/main.py runs BENCH_MODELS in
# the fixed order ann, snn). The SNN only uses ANN checkpoints of the same run id (tag checked in benchmark.py); SNN folds
# whose ANN is not finished yet stay pending and are done by a later job of this chain, so queue enough jobs.
# Usage: [BENCH_RUN_ID=<id>] ./submit_chain_ann_snn.sh [jobs per GPU, default 4]
BENCH_MODELS=ann,snn CHAIN_NAME=ann_snn exec "$(dirname "$0")/submit_chain.sh" "$@"
