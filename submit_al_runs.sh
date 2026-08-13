#!/bin/bash

#SBATCH -J molpal_al
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p hopper
#SBATCH -q hopper
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

# --mem=0: confirmed via `sinfo -N -p hopper -o "%N %m %c"` that hopper nodes
# have 503GB RAM / 96 CPUs each -- yet job 9872917 task 0 OOM'd (SLURM-level
# oom_kill, not a Python/CUDA traceback) with a real memory footprint of only
# ~24-25GB (all 3 backbone embeddings resident). --exclusive grants exclusive
# NODE access but does not by itself override a per-CPU default memory cap
# tied to --cpus-per-task; with only 8 CPUs requested, the job's enforced
# cgroup memory limit was almost certainly a small fraction of the node's
# actual 503GB. --mem=0 is SLURM's explicit "grant all memory on the node"
# directive, independent of --exclusive.

# Confirmed via `sinfo -o "%P %G"` on the actual Quartz login node: hopper
# is the only partition with H100s here (gpu:h100:4) -- Quartz's plain
# "gpu" partition has V100s (gpu:v100:4), not H100s. -q hopper is required
# alongside -p hopper per IU RT's docs. (Two earlier submission attempts
# failed because they were run from a BigRed200 login node instead, which
# has neither a hopper partition nor any H100 hardware at all.)

# Targets Quartz's H100-80GB "hopper" partition (-p hopper -q hopper, per IU
# RT's GPU docs), not BigRed200's regular "gpu" partition -- BigRed200's GPU
# nodes only have 40GB RAM/node, not enough headroom for fusion mode (which
# loads all 3 backbone embeddings at once, ~25GB+ just for the raw arrays)
# once you add per-task overhead; H100 nodes have far more system RAM.
#
# --exclusive: confirmed directly on BigRed200 (job 7897718) that without
# it, SLURM co-schedules multiple --gpus-per-node=1 array tasks onto the
# same physical node, causing two distinct failure modes that both trace
# back to node-sharing -- fusion mode OOM'd, and MPN's ray.init() timed out
# starting a local Ray cluster (multiple tasks binding Ray's default ports
# on the same node simultaneously). Kept here defensively even on Quartz's
# larger-memory H100 nodes, since the Ray port collision isn't a
# memory-size problem and would recur under node-sharing regardless.
# Trade-off: fewer tasks run concurrently if hopper nodes have multiple
# GPUs each (each task now reserves the whole node) -- check
# `sinfo -p hopper -o "%n %G"` if throughput matters more than isolation.

# --time=24:00:00 is a rough extrapolation from ONE observed data point (colo27's
# mpn_greedy_frac0.004 run: round 1 took ~2h across ~2.1M-molecule prediction +
# growing-labeled-set training) -- not a verified bound. If a task hits the wall
# and gets killed, it resumes from scratch (this script does not checkpoint
# mid-round), so raise --time rather than assume 24h is enough for every config.
#
# Runs all 18 configs (3 methods x 2 acquisitions x 3 batch-size fractions) as
# one SLURM array job, matching run_all_configs.sh's exact enumeration order
# (frac outer, acq middle, method inner) so array index <-> config is
# deterministic:
#   index 0-2   = frac 0.004, acq greedy : mpn, molformer_ft, fusion
#   index 3-5   = frac 0.004, acq ucb    : mpn, molformer_ft, fusion
#   index 6-8   = frac 0.002, acq greedy : mpn, molformer_ft, fusion
#   index 9-11  = frac 0.002, acq ucb    : mpn, molformer_ft, fusion
#   index 12-14 = frac 0.001, acq greedy : mpn, molformer_ft, fusion
#   index 15-17 = frac 0.001, acq ucb    : mpn, molformer_ft, fusion
#
# Usage:
#   sbatch --array=0-17 submit_al_runs.sh
#
# As of BigRed200 job 7897718 (before this script moved to Quartz's hopper
# partition), indices 1,4,6,7,10,13,15,16 already completed successfully --
# their runs/*/history.json is already on disk and won't be touched by
# rerunning those indices (run_experiment.py doesn't skip existing run-dirs,
# so a resubmission of an already-done index overwrites it rather than
# erroring, but is wasted work). To resubmit only what's still missing:
#   sbatch --array=0,2,3,5,8,9,11,12,14,17 submit_al_runs.sh

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
