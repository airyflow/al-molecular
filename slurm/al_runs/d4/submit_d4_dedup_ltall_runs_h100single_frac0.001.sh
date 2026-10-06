#!/bin/bash

#SBATCH -J d4_dedup_ltall_f001
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_dedup_ltall_f001_%j.txt
#SBATCH -e logs/d4_dedup_ltall_f001_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=280G
#SBATCH -A r00939

# Same as submit_d4_dedup_ltall_runs_h100single.sh (frac=0.004), fixed at
# FRAC=0.001 instead, mirroring AmpC's own batch-fraction sweep (0.1%,
# 0.2%, 0.4% -- see report sect 11). A separate script file per frac
# (rather than edit-FRAC-and-resubmit, which is AmpC's own convention) so
# this can run concurrently with the already-in-flight frac=0.004 job
# without either one needing to be paused.
#
# Own, separate coord-dir/run-dir, keyed by FRAC (same reasoning as the
# frac=0.004 script).
#
# TWO-STEP LAUNCH, in order:
#   1. Submit the matching worker pool FIRST (it waits patiently, no
#      timeout, for round 1's ready marker):
#        mkdir -p /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/dedup_ltall_frac0.001
#        sbatch --array=0-7 slurm/al_runs/d4/submit_d4_dedup_ltall_predict_workers_h100single.sh \
#            /N/project/SingleCell_Image/mengjing/d4_138M/al_coord/dedup_ltall_frac0.001 8
#   2. Once those 8 tasks are RUNNING (check `squeue`), submit this script:
#        sbatch slurm/al_runs/d4/submit_d4_dedup_ltall_runs_h100single_frac0.001.sh

set -euo pipefail

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

FRAC=0.001
COORD_DIR="$D4_ROOT/al_coord/dedup_ltall_frac${FRAC}"

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

echo "[orchestrator] dataset=D4_dedup surrogate=ltall acq=greedy frac=$FRAC (init=batch=$count) coord-dir=$COORD_DIR"

srun --cpu-bind=none python3 -u run_experiment.py --dataset D4_dedup --mode mve --surrogate ltall \
    --backbones grover3400 mhgged molformer smited unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 50000 \
    --parallel-predict --num-shards 8 --coord-dir "$COORD_DIR" \
    --run-dir "runs/d4_dedup_ltall_greedy_frac${FRAC}" \
    --resume
