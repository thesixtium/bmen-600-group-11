#!/bin/bash
# RNN-wide chain (neuron-matched recurrent ELU network, 2560 hidden neurons). Independent of all other chains.
# Usage: [BENCH_RUN_ID=<id>] ./submit_chain_rnn_wide.sh [jobs per GPU, default 4]
BENCH_MODELS=rnn_wide CHAIN_NAME=rnn_wide exec "$(dirname "$0")/submit_chain.sh" "$@"
