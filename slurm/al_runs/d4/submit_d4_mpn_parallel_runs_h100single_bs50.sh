#!/bin/bash

#SBATCH -J d4_mpn_par_bs50
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_mpn_par_bs50_%j.txt
#SBATCH -e logs/d4_mpn_par_bs50_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH -A r00939

# D4 MPN (ParallelMolPALExplorer, greedy, frac=0.004) at the paper's own
# training batch size, 50 -- the control for the batch-size-1024 run
# (submit_d4_mpn_parallel_runs_h100single_bs1024.sh, 69.2% vs. LT-All's
# 76.9%, report sect. 16). If this lands near LT-All, batch size 1024 caused
# the gap; if near 69%, MPN is genuinely weaker on D4.
#
# --mpn-persistent-workers keeps the training data-loader workers alive
# between epochs: ~1.8x faster training with bit-identical predictions
# (probe job 11931298, report sect. 16). It changes speed only, so this is
# still the paper's configuration.
#
# Own coord-dir/run-dir (bs50-suffixed). Same launch pattern as the
# bs1024 run: workers are a separate array per round, submitted when (or
# before -- they wait) round R's checkpoint exists. Use --time=24:00:00 and
# exclude the slow nodes seen before:
#   1. mkdir -p /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/mpn_bs50_frac0.004
#   2. sbatch slurm/al_runs/d4/submit_d4_mpn_parallel_runs_h100single_bs50.sh
#   3. per round, with the round number in place of R (1, then 2, ... 5):
#        sbatch --array=0-7 --time=24:00:00 --exclude=g47,g53 \
#            slurm/al_runs/d4/submit_d4_mpn_parallel_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/mpn_bs50_frac0.004 8 R
# The orchestrator will likely hit its 48 h limit at least once: resubmit it
# only after the current round's 8 .done files exist (it resumes from the
# last finished round). If it dies after the last round's shards are done,
# finish that round on CPU instead of retraining:
#   TAG=bs50 sbatch slurm/al_runs/d4/submit_d4_mpn_finalize_round.sh 5

set -euo pipefail

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"
FRAC=0.004
NUM_SHARDS=8
COORD_DIR="$D4_ROOT/al_coord/mpn_bs50_frac${FRAC}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fi
set -a
[ -f config.env ] && source config.env
set +a
: "${CONDA_SH:=/N/slate/mengjing/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=py310}"
source "$CONDA_SH"
conda activate "$CONDA_ENV"
mkdir -p logs "$COORD_DIR"

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=116225927
count=$(python3 -c "print(round($N * $FRAC))")
echo "[orchestrator] dataset=D4_dedup mode=molpal model=mpn acq=greedy frac=$FRAC (init=batch=$count) mpn-batch-size=50 persistent-workers shards=$NUM_SHARDS coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset D4_dedup \
    --mode molpal --model mpn --conf-method mve --acq greedy \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards "$NUM_SHARDS" --coord-dir "$COORD_DIR" \
    --ncpu 8 --retrain-from-scratch --mpn-persistent-workers \
    --run-dir "runs/d4_mpn_bs50_greedy_frac${FRAC}" \
    --resume
