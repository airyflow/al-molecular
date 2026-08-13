#!/bin/bash

#SBATCH -J molpal_al
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o logs/al_run_%A_%a.txt
#SBATCH -e logs/al_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# --mem=0: confirmed on Quartz's hopper partition that --exclusive alone does
# NOT override a per-CPU default memory cap tied to --cpus-per-task -- a job
# with a real ~24-25GB footprint OOM'd on a 503GB node because of this.
# --mem=0 is SLURM's explicit "grant all memory on the node" directive,
# needed independent of --exclusive. Apply the same fix here defensively
# even though BigRed200's actual per-CPU default hasn't been separately
# confirmed to have caused a failure.

# BigRed200 variant of submit_al_runs.sh (which targets Quartz's hopper/H100
# partition) -- use this when hopper is queue-blocked. --gpus-per-node=1 has
# no type string here: confirmed via `sinfo -p gpu -o "%N %G"` on BigRed200
# that its gpu partition's GRES is untyped (gpu:4, no a100/h100 tag), unlike
# Quartz's hopper (gpu:h100:4) -- requesting "a100:1" the way hopper needed
# "h100:1" would fail here the same way "h100:1" failed on BigRed200 earlier.
# GPUs are A100-PCIE-40GB (confirmed via nvidia-smi -L on a BigRed200 node).
#
# 40GB/node is tighter than Quartz's H100-80GB, but should now be workable:
# the earlier fusion-mode OOM here was NOT actually a hardware-headroom
# problem, it was run_experiment.py passing the entire ~2.09M-molecule
# remaining pool into get_means_and_vars() in one call, which multiplied
# memory use several times over (fancy-indexing copies + concatenate) on
# top of the ~24-25GB baseline of holding all 3 backbones' embeddings
# resident. That's fixed now (_chunked_get_means_and_vars(), 50K-molecule
# chunks) -- baseline ~24-25GB + bounded chunk overhead should fit in 40GB
# with real but not huge headroom. Worth watching the first fusion task's
# memory closely regardless, since this hasn't been verified at full scale
# on an actual 40GB node yet, only reasoned through and smoke-tested small.
#
# --exclusive: confirmed directly on BigRed200 (job 7897718) that without
# it, SLURM co-schedules multiple --gpus-per-node=1 array tasks onto the
# same physical node (up to 4 GPUs/node here), causing both the earlier
# fusion OOM and MPN's ray.init() port-collision failures.
#
# Runs all 18 configs (3 methods x 2 acquisitions x 3 batch-size fractions)
# as one SLURM array job, matching run_all_configs.sh's exact enumeration
# order (frac outer, acq middle, method inner) so array index <-> config is
# deterministic:
#   index 0-2   = frac 0.004, acq greedy : mpn, molformer_ft, fusion
#   index 3-5   = frac 0.004, acq ucb    : mpn, molformer_ft, fusion
#   index 6-8   = frac 0.002, acq greedy : mpn, molformer_ft, fusion
#   index 9-11  = frac 0.002, acq ucb    : mpn, molformer_ft, fusion
#   index 12-14 = frac 0.001, acq greedy : mpn, molformer_ft, fusion
#   index 15-17 = frac 0.001, acq ucb    : mpn, molformer_ft, fusion
#
# Usage:
#   sbatch --array=0-17 submit_al_runs_bigred.sh
#
# As of BigRed200 job 7897718, indices 1,4,6,7,10,13,15,16 already completed
# successfully -- their runs/*/history.json is already on disk. To resubmit
# only what's still missing:
#   sbatch --array=0,2,3,5,8,9,11,12,14,17 submit_al_runs_bigred.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-17 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=2104319

FRACS=(0.004 0.004 0.004 0.004 0.004 0.004 0.002 0.002 0.002 0.002 0.002 0.002 0.001 0.001 0.001 0.001 0.001 0.001)
ACQS=(greedy greedy greedy ucb ucb ucb greedy greedy greedy ucb ucb ucb greedy greedy greedy ucb ucb ucb)
METHODS=(mpn molformer_ft fusion mpn molformer_ft fusion mpn molformer_ft fusion mpn molformer_ft fusion mpn molformer_ft fusion mpn molformer_ft fusion)

frac="${FRACS[$TASK_ID]}"
acq="${ACQS[$TASK_ID]}"
method="${METHODS[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

echo "[task $TASK_ID] method=$method acq=$acq frac=$frac (init=batch=$count)"

case "$method" in
  mpn)
    srun --cpu-bind=none python run_experiment.py --mode molpal --model mpn --conf-method mve \
        --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
        --run-dir "runs/mpn_${acq}_frac${frac}"
    ;;
  molformer_ft)
    srun --cpu-bind=none python run_experiment.py --mode mve --surrogate ft_molformer_single --backbones molformer \
        --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
        --run-dir "runs/molformer_ft_${acq}_frac${frac}"
    ;;
  fusion)
    srun --cpu-bind=none python run_experiment.py --mode mve --surrogate ensemble --backbones grover molformer unimol \
        --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
        --run-dir "runs/fusion_${acq}_frac${frac}"
    ;;
esac
