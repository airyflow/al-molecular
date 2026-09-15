#!/bin/bash

#SBATCH -J ampc_dedup_ltall_ucb
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_dedup_ltall_ucb_%j.txt
#SBATCH -e logs/ampc_dedup_ltall_ucb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=280G
#SBATCH -A r00939

# Same as submit_ampc_dedup_ltall_runs_h100single.sh (same dataset, same
# frac=0.004, same --mem=280G reasoning -- see that script's own comments
# for why), but --acq ucb instead of --acq greedy: molpal/acquirer/metrics.py's
# ucb(Y_mean, Y_var, beta=2) = Y_mean + 2*sqrt(Y_var), using LTAllSurrogate's
# own M=5 ensemble-member disagreement as the uncertainty term (already
# fully wired -- ParallelMVEExplorer.run()/_checkpoint() both branch on
# self.acq_name in ("ucb", "lcb", "thompson", "ts", "ei", "pi") to call
# self.acq_fn(mu, var) instead of self.acq_fn(mu), no code changes needed).
#
# Own, separate coord-dir/run-dir from the greedy run (al_coord/dedup_ltall_ucb_frac<FRAC>,
# not al_coord/dedup_ltall_frac<FRAC>) -- different acquisition function
# means a different sequence of acquired molecules from round 2 onward
# (round 1's random init draw is identical, same seed), so no state is
# shared or reusable between the two.
#
# TWO-STEP LAUNCH, in order:
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall_ucb_frac0.004
#        sbatch --array=0-7 slurm/al_runs/ampc/submit_ampc_dedup_ltall_ucb_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/ampc_99.5M/al_coord/dedup_ltall_ucb_frac0.004 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/ampc/submit_ampc_dedup_ltall_ucb_runs_h100single.sh

set -euo pipefail

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

FRAC=0.004
COORD_DIR="$AMPC_ROOT/al_coord/dedup_ltall_ucb_frac${FRAC}"

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

N=98489350
count=$(python3 -c "print(round($N * $FRAC))")

echo "[orchestrator] dataset=AmpC_dedup surrogate=ltall acq=ucb frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol \
    --acq ucb --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/ampc_dedup_ltall_ucb_frac${FRAC}" \
    --resume
