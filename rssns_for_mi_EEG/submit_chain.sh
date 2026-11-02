#!/bin/bash
# Queue N chained benchmark jobs that all work on ONE run id, so a run that needs several days is a single command.
#
#   cd ~/bmen-600-group-11/rssns_for_mi_EEG
#   ./submit_chain.sh 6                                   # new run, 6 jobs (default 4)
#   BENCH_RUN_ID=<existing id> ./submit_chain.sh 3        # continue an existing run with 3 more jobs
#
# Safety: the run id is fixed here and exported to every job, so they all resume the same folder (finished folds are
# skipped). Once a job finishes everything it writes results/benchmark/<id>/COMPLETE; later jobs see it and exit
# immediately, before installing anything. Queue more jobs than needed; surplus ones are harmless.
# Dependency is "afterany" so a timeout or node failure of one job does not break the chain.
# If a job crashes on a real bug, the later ones will crash too: cancel them with  scancel -u $USER --name=rsnn_benchmark

N="${1:-4}"
cd "$(dirname "$0")" || exit 1
[ -f run_benchmark.slurm ] || { echo "run_benchmark.slurm not found next to this script"; exit 1; }

export BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
echo "Run ID: $BENCH_RUN_ID   (jobs: $N)"

prev=""
for i in $(seq 1 "$N"); do
    if [ -z "$prev" ]; then
        jid=$(sbatch --parsable run_benchmark.slurm) || exit 1
    else
        jid=$(sbatch --parsable --dependency=afterany:"$prev" run_benchmark.slurm) || exit 1
    fi
    echo "  job $i/$N: $jid${prev:+ (starts after $prev)}"
    prev="$jid"
done
echo "Follow: tail -f results/benchmark/$BENCH_RUN_ID/run_log.txt"
echo "Resume by hand later:  BENCH_RUN_ID=$BENCH_RUN_ID ./submit_chain.sh 2"
