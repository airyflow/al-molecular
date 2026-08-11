#!/bin/bash

#SBATCH -J molpal_ftfusion
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p hopper
#SBATCH -q hopper
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ftfusion_run_%A_%a.txt
#SBATCH -e logs/ftfusion_run_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# FTFusionSurrogate: GROVER + MoLFormer fine-tuned JOINTLY each round from
# round 3 on (frozen for rounds 1-2, matching n_ft_delay=2 default), UniMol
# kept FROZEN throughout (contributes its existing pre-extracted embeddings
# unchanged every round). See surrogates.py's FTFusionSurrogate docstring
# for why: UniMol's embeddings come from a ragged, pre-generated conformer
# cache (10 ETKDG conformers + MMFF optimize/molecule, ~400ms/molecule) that
# can't be memory-mapped like GROVER/MoLFormer's flat arrays, and regenerating
# it on the fly for the full 2.1M pool would take ~2.4 days even with 4
# workers -- exceeding this job's own 24h budget for a single extraction
# sweep. Confirmed via local test: BackboneFinetuner's GROVER path used to
# eagerly cache the full-pool graph set (51GB on disk, 300GB+ once
# deserialized) regardless of --pool-limit; it's since been rewritten to
# build graphs on demand from raw SMILES per batch (measured 3ms/molecule,
# cheap since GROVER needs no conformers) instead of caching the whole pool.
# The npz pool embeddings (grover/molformer/unimol_embeddings.npz, used by
# every surrogate type via EmbeddingFeaturizer) are also now memory-mapped
# from standalone .npy files instead of loaded fully into RAM.
#
# --exclusive/--mem=0: same proven necessity as every other script in this
# repo -- --exclusive alone does not override a per-CPU SLURM memory cap,
# and node-sharing causes Ray-port-collision-class failures independent of
# GPU usage. See submit_al_runs.sh's comments for the original diagnosis.
#
# Greedy acquisition only, 3 batch-size fractions -- starting narrower than
# the other methods' greedy+UCB sweep given how expensive this one already
# looks; add UCB indices later if the cost/benefit looks worth it.
#
# Array index <-> config:
#   index 0 = frac 0.004
#   index 1 = frac 0.002
#   index 2 = frac 0.001
#
# Usage:
#   sbatch --array=0-2 submit_ft_fusion_runs.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-2 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/molpal-fusion-hts
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

N=2104319

FRACS=(0.004 0.002 0.001)
frac="${FRACS[$TASK_ID]}"
count=$(python3 -c "print(round($N * $frac))")

echo "[task $TASK_ID] surrogate=ft_fusion acq=greedy frac=$frac (init=batch=$count)"

srun --cpu-bind=none python run_experiment.py --mode mve --surrogate ft_fusion \
    --backbones grover molformer unimol \
    --acq greedy --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
    --run-dir "runs/ftfusion_greedy_frac${frac}"
