#!/bin/bash
# RNN chain (parameter-matched recurrent ELU network, non-spiking control). Independent of all other chains.
# Usage: [BENCH_RUN_ID=<id>] ./submit_chain_rnn.sh [jobs per GPU, default 4]
BENCH_MODELS=rnn CHAIN_NAME=rnn exec "$(dirname "$0")/submit_chain.sh" "$@"
