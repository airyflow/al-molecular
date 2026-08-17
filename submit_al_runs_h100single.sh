#!/bin/bash

#SBATCH -J molpal_al
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/al_run_%A_%a.txt
#SBATCH -e logs/al_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# Quartz H100 variant of submit_al_runs.sh -- use this instead of the
# "hopper"-targeting original, which queues indefinitely. `sinfo` on the
# actual Quartz login node (2026-08-13) shows no "hopper" partition at
# all anymore -- only h100-debug, h100-multi, and h100-single (19 idle
# nodes at time of writing, g[61-79]). IU RT confirmed h100-single is the
# current single-GPU H100 partition. submit_al_runs.sh's own comments
# document "hopper" being verified as Quartz's only H100 partition when
# written; it's since been split into these three by job GPU-count,
# which is almost certainly why jobs submitted with -p hopper -q hopper
# now queue forever (a stale partition name, not ordinary contention).
#
# -q hopper dropped (no confirmed equivalent QOS name for h100-single in
# the current sinfo output) -- default partition QOS applies instead.
# --gpus-per-node h100:1 kept typed (matching the original hopper
# convention, and h100-single's nodes are explicitly H100-only by name);
# if this specific GRES type string is rejected, retry with the untyped
# --gpus-per-node=1 form submit_al_runs_bigred.sh already uses
# successfully for BigRed200's differently-configured gpu partition.
#
# --exclusive/--mem=0 kept for the same reasons submit_al_runs.sh
# documents (a per-CPU SLURM memory cap isn't overridden by --exclusive
# alone; node-sharing causes Ray port-collision failures for MPN
# independent of GPU usage) -- both failure modes are about SLURM/Ray
# behavior, not specific to which H100 partition is targeted, so they
# still apply here.
#
# Same 18-config array-index scheme as submit_al_runs.sh (frac outer, acq
# middle, method inner):
#   index 0-2   = frac 0.004, acq greedy : mpn, molformer_ft, fusion
#   index 3-5   = frac 0.004, acq ucb    : mpn, molformer_ft, fusion
#   index 6-8   = frac 0.002, acq greedy : mpn, molformer_ft, fusion
#   index 9-11  = frac 0.002, acq ucb    : mpn, molformer_ft, fusion
#   index 12-14 = frac 0.001, acq greedy : mpn, molformer_ft, fusion
#   index 15-17 = frac 0.001, acq ucb    : mpn, molformer_ft, fusion
#
# Usage:
#   sbatch --array=0-17 submit_al_runs_h100single.sh

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
