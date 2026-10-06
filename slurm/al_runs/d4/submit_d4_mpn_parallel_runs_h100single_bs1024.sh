#!/bin/bash

#SBATCH -J d4_mpn_par_bs1024
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_mpn_par_bs1024_%j.txt
#SBATCH -e logs/d4_mpn_par_bs1024_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=300G
#SBATCH -A r00939

# D4 equivalent of AmpC's submit_ampc_mpn_parallel_runs_h100single.sh
# (ParallelMolPALExplorer, paper's own MPN surrogate, greedy, frac=0.004),
# with ONE deliberate deviation: --mpn-batch-size 1024 instead of the
# paper's default 50.
#
# WHY: AmpC's own MPN run (this same pipeline, batch_size=50 default)
# measured round 5's training at 59,863.4s (~16.6h) for 1,969,785
# molecules -- ~1.97M individual 50-row training steps, ~30ms/step. A
# batch of 50 badly underutilizes an H100 (kernel-launch/featurization
# overhead dominates over the actual, very cheap matmuls for a model this
# small -- 355,502 total params, measured directly). batch_size=1024
# should cut per-round training wall-clock substantially by amortizing
# that overhead over ~20x more molecules per step.
#
# THIS IS NOT FREE: batch_size interacts with warmup_epochs and the LR
# schedule (see MPNN.__init__, molpal/models/mpnmodels.py), so this run is
# NOT a controlled ablation of batch size alone -- it may converge
# differently (faster, slower, or to a different recall) than the
# paper's own hyperparameters, not just run faster. Treat its recall
# numbers as a separate data point, not a drop-in comparison against
# AmpC's or D4's batch_size=50 runs without flagging this caveat.
#
# Own, separate coord-dir/run-dir -- bs1024-suffixed, so this never
# collides with a (possible, future) batch_size=50 D4 MPN run's state.
#
# LAUNCH (same two-step order as AmpC's MPN script -- workers only once
# the round's checkpoint exists):
#   1. mkdir -p /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/mpn_bs1024_frac0.004
#   2. sbatch this script.
#   3. When round R's checkpoint appears (coord-dir round_R_ready.marker),
#      submit that round's workers:
#        sbatch --array=0-7 slurm/al_runs/d4/submit_d4_mpn_parallel_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/mpn_bs1024_frac0.004 8 R

set -euo pipefail

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

FRAC=0.004
NUM_SHARDS=8
COORD_DIR="$D4_ROOT/al_coord/mpn_bs1024_frac${FRAC}"

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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=116225927
count=$(python3 -c "print(round($N * $FRAC))")

echo "[orchestrator] dataset=D4_dedup mode=molpal model=mpn acq=greedy frac=$FRAC (init=batch=$count) mpn-batch-size=1024 shards=$NUM_SHARDS coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset D4_dedup \
    --mode molpal --model mpn --conf-method mve --acq greedy \
    --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards "$NUM_SHARDS" --coord-dir "$COORD_DIR" \
    --ncpu 8 --retrain-from-scratch --mpn-batch-size 1024 \
    --run-dir "runs/d4_mpn_bs1024_greedy_frac${FRAC}" \
    --resume
