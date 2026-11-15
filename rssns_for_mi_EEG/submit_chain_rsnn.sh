#!/bin/bash
# RSNN chain (parameter-matched recurrent LIF network). Independent of all other chains.
# Usage: [BENCH_RUN_ID=<id>] ./submit_chain_rsnn.sh [jobs per GPU, default 4]
BENCH_MODELS=rsnn CHAIN_NAME=rsnn exec "$(dirname "$0")/submit_chain.sh" "$@"
