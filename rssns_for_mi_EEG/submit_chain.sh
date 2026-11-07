#!/bin/bash
# Queue the parallel benchmark: N_GPUS chains (default 4 = your GPU limit) of N chained jobs (default 4), all working on ONE run
# id. Each job owns one GPU and runs WORKERS_PER_JOB workers on it (see run_benchmark.slurm), so the 90 folds are split over
# N_GPUS x WORKERS_PER_JOB shards (default 4 x 3 = 12).
#
#   cd ~/bmen-600-group-11/rssns_for_mi_EEG
#   scancel -u $USER --name=rsnn_benchmark                 # first cancel the old, slow chain
#   ./submit_chain.sh 4                                    # new run: 4 GPUs x 4 jobs
#   BENCH_PATIENCE=100 ./submit_chain.sh 3                 # optional pretraining early stopping (changes the protocol)
#   WORKERS_PER_JOB=4 ./submit_chain.sh 4                  # 4 workers per GPU
#   BENCH_RUN_ID=<existing id> ./submit_chain.sh 2         # continue an existing run with 2 more jobs per GPU
#
# Safety: the run id is fixed here and exported to every job; finished folds are skipped on resume (a fold counts as done
# when its metrics file exists), so continuing with a different N_GPUS / WORKERS_PER_JOB is also correct: it only
# redistributes the folds that are still pending. When all folds of all models are done, the first worker to see it runs the
# analysis and writes results/benchmark/<id>/COMPLETE; later jobs exit at once.
# Dependency is "afterany" so a timeout or node failure of one job does not break its chain.
# If a job crashes on a real bug, the later ones will crash too: cancel with  scancel -u $USER --name=rsnn_benchmark

N="${1:-4}"
N_GPUS="${N_GPUS:-4}"
WORKERS_PER_JOB="${WORKERS_PER_JOB:-3}"
cd "$(dirname "$0")" || exit 1
[ -f run_benchmark.slurm ] || { echo "run_benchmark.slurm not found next to this script"; exit 1; }

export BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
echo "Run ID: $BENCH_RUN_ID   ($N_GPUS GPUs x $N chained jobs, $WORKERS_PER_JOB workers per GPU, $((N_GPUS * WORKERS_PER_JOB)) shards)"

for g in $(seq 0 $(( N_GPUS - 1 ))); do
    prev=""
    for i in $(seq 1 "$N"); do
        opts="--parsable --export=ALL,BENCH_JOB_INDEX=$g,BENCH_N_JOBS=$N_GPUS,WORKERS_PER_JOB=$WORKERS_PER_JOB"
        if [ -z "$prev" ]; then
            jid=$(sbatch $opts run_benchmark.slurm) || exit 1
        else
            jid=$(sbatch $opts --dependency=afterany:"$prev" run_benchmark.slurm) || exit 1
        fi
        echo "  gpu-chain $g job $i/$N: $jid${prev:+ (starts after $prev)}"
        prev="$jid"
    done
done
echo "Follow one worker: tail -f results/benchmark/$BENCH_RUN_ID/logs/shard0.txt"
echo "Resume by hand later:  BENCH_RUN_ID=$BENCH_RUN_ID ./submit_chain.sh 2"
