#!/bin/bash
# Queue one model chain: N_GPUS parallel GPU chains (default 4 = your GPU limit) of N chained 24 h jobs (default 4), all working
# on ONE run id. Each job owns one GPU and runs WORKERS_PER_JOB workers on it (see run_benchmark.slurm), so the 90 folds are
# split over N_GPUS x WORKERS_PER_JOB shards (default 4 x 3 = 12).
#
# Normally called through a model script, which sets BENCH_MODELS and CHAIN_NAME:
#   ./submit_chain_ann_snn.sh 4   ./submit_chain_rsnn.sh 4   ./submit_chain_rnn.sh 4
#   ./submit_chain_rsnn_wide.sh 4   ./submit_chain_rnn_wide.sh 4
# Called directly (no BENCH_MODELS), every job runs all six models, as before.
#
#   cd ~/bmen-600-group-11/rssns_for_mi_EEG
#   ./submit_chain_ann_snn.sh 4                                       # new run: prints its run id
#   BENCH_RUN_ID=<that id> ./submit_chain_rsnn.sh 4                   # another model chain, SAME experiment
#   BENCH_RUN_ID=<id> ./submit_chain_rnn.sh 2                         # resubmit / continue one chain only
#   BENCH_PATIENCE=100 ./submit_chain_rsnn.sh 3                       # optional pretraining early stopping (changes the protocol)
#   WORKERS_PER_JOB=4 ./submit_chain_rnn.sh 4                         # 4 workers per GPU
#   BENCH_RUN_ID=<id> AFTER_G1=<jobid> ./submit_chain_rsnn.sh 6       # GPU-chain 1 starts after that (running) job
#
# Safety: the run id is fixed here and exported to every job; finished folds are skipped on resume (a fold counts as done
# when its metrics file exists), so continuing with a different N_GPUS / WORKERS_PER_JOB is also correct. Each model chain
# only touches its own models' folders and its own <chain>_* log files, so chains are independent; src/main.py refuses a run
# id whose config.json was made with other settings. Dependencies ("afterany") exist only between the jobs of ONE GPU-chain
# of ONE model chain, so a timeout, node failure or crash in one model chain never blocks another model chain.
# Cancel one model chain:  scancel -u $USER --name=bench_<chain>    (e.g. bench_rsnn)

N="${1:-4}"
N_GPUS="${N_GPUS:-4}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-3}"
export BENCH_MODELS="${BENCH_MODELS:-}"          # empty = src/main.py runs all models
export CHAIN_NAME="${CHAIN_NAME:-all}"
cd "$(dirname "$0")" || exit 1
[ -f run_benchmark.slurm ] || { echo "run_benchmark.slurm not found next to this script"; exit 1; }

PREFIX=""
[ "${BENCH_QUICK:-0}" = "1" ] && PREFIX="quick_"
export BENCH_RUN_ID="${BENCH_RUN_ID:-${PREFIX}$(date +%Y%m%d_%H%M%S)}"
echo "Chain : $CHAIN_NAME (BENCH_MODELS=${BENCH_MODELS:-all})"
echo "Run ID: $BENCH_RUN_ID   ($N_GPUS GPUs x $N chained jobs, $WORKERS_PER_JOB workers per GPU, $((N_GPUS * WORKERS_PER_JOB)) shards)"

for g in $(seq 0 $(( N_GPUS - 1 ))); do
    after_var="AFTER_G$g"          # optional: AFTER_G1=<jobid> makes the chain of GPU 1 start after that (running) job
    prev="${!after_var:-}"
    for i in $(seq 1 "$N"); do
        # BENCH_MODELS / CHAIN_NAME / BENCH_RUN_ID are exported above and reach the job through ALL (BENCH_MODELS contains
        # commas, so it cannot be listed in --export itself).
        opts="--parsable --job-name=bench_$CHAIN_NAME --export=ALL,BENCH_JOB_INDEX=$g,BENCH_N_JOBS=$N_GPUS,WORKERS_PER_JOB=$WORKERS_PER_JOB"
        if [ -z "$prev" ]; then
            jid=$(sbatch $opts run_benchmark.slurm) || exit 1
        else
            jid=$(sbatch $opts --dependency=afterany:"$prev" run_benchmark.slurm) || exit 1
        fi
        echo "  $CHAIN_NAME gpu-chain $g job $i/$N: $jid${prev:+ (starts after $prev)}"
        prev="$jid"
    done
done
echo "Follow one worker: tail -f results/benchmark/$BENCH_RUN_ID/logs/${CHAIN_NAME}_shard0.txt"
echo "Other model chains of this experiment / resume later:  BENCH_RUN_ID=$BENCH_RUN_ID ./submit_chain_<model>.sh 2"
