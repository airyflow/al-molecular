#!/bin/bash

#SBATCH -J ampc_mpn_probe
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mpn_probe_%j.txt
#SBATCH -e logs/ampc_mpn_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH -A r00939

# Timing probe for the paper's MolPAL MPN surrogate (--mode molpal --model
# mpn --conf-method mve) on the deduplicated AmpC pool. Purpose: decide
# whether a full 98.5M-molecule MPN run is realistic BEFORE building
# anything for it. MolPALExplorer is a single-process design (whole pool
# predicted sequentially each round, SMILES->graph featurized on the fly)
# and has only ever been run on the 2.1M Enamine pool, never above that.
#
# Two-point fit: the same one-round run at two truncated pool sizes
# (--pool-limit keeps the FIRST N library rows, then drops any without an
# oracle score). One round's elapsed time = train on init-size molecules +
# predict on the remaining pool + selection, so
#   slope of elapsed vs. pool size  ~= per-molecule MPN prediction cost
#   intercept                        ~= fixed training cost at this init-size
# Extrapolate the slope to ~98.5M to estimate one full-pool round.
# Recall printed here is meaningless (tiny truncated pool); only the
# elapsed times matter.
#
# init/batch = 4,000 (~0.4% of the smaller truncated pool). The training
# set at a real 0.4% run grows to ~394k-2.4M molecules, so the intercept
# here understates real per-round training cost -- read it as a floor.
#
# Memory: loading the full oracle (~95M scores) dominates, ~24GB, plus the
# library read; 128G leaves generous headroom for the probe itself.
#
# Usage: sbatch slurm/al_runs/ampc/submit_ampc_mpn_timing_probe_h100single.sh

set -euo pipefail

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
mkdir -p logs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

for N in 250000 1000000; do
    echo "===== [probe] pool-limit=$N ====="
    srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup \
        --mode molpal --model mpn --conf-method mve --acq greedy \
        --pool-limit "$N" --init-size 4000 --batch-size 4000 \
        --n-rounds 1 --topk 1000 --ncpu 8 \
        --run-dir "runs/ampc_mpn_probe_${N}"
done
