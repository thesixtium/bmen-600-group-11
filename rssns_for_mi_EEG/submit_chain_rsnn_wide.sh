#!/bin/bash
# RSNN-wide chain (neuron-matched recurrent LIF network, 2560 hidden neurons). Independent of all other chains.
# Usage: [BENCH_RUN_ID=<id>] ./submit_chain_rsnn_wide.sh [jobs per GPU, default 4]
BENCH_MODELS=rsnn_wide CHAIN_NAME=rsnn_wide exec "$(dirname "$0")/submit_chain.sh" "$@"
