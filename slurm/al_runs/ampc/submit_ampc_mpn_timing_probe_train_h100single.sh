#!/bin/bash

#SBATCH -J ampc_mpn_probe_train
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mpn_probe_train_%j.txt
#SBATCH -e logs/ampc_mpn_probe_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH -A r00939

# Second MPN timing probe (see submit_ampc_mpn_timing_probe_h100single.sh,
# job 10544431): that one measured prediction at ~1.7 ms/molecule (~47h per
# full 98.5M pass on one GPU) but trained on only 4,000 molecules, so it
# could not say how MPN TRAINING time grows with training-set size -- the
# unknown that decides whether a full MolPAL MPN run is feasible at all
# (a real 0.4% run trains on ~394k-2.4M molecules, 50 epochs each round).
#
# Same pool (first 500,000 library rows, scored only) at two training-set
# sizes. Prediction cost is the same in both (~500k at 1.7 ms, ~14 min), so
#   elapsed(200k) - elapsed(40k)  ~= training-time growth over 160k more
# training molecules. Compare against the 4,000-molecule intercept of
# ~2,900s from the first probe.
#
# Usage: sbatch slurm/al_runs/ampc/submit_ampc_mpn_timing_probe_train_h100single.sh

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

for INIT in 40000 200000; do
    echo "===== [probe] init-size=$INIT pool-limit=500000 ====="
    srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC_dedup \
        --mode molpal --model mpn --conf-method mve --acq greedy \
        --pool-limit 500000 --init-size "$INIT" --batch-size "$INIT" \
        --n-rounds 1 --topk 1000 --ncpu 8 \
        --run-dir "runs/ampc_mpn_probe_train_${INIT}"
done
