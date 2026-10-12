#!/bin/bash

#SBATCH -J aa_build_sim
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p general
#SBATCH -o logs/aa_build_sim_%A_%a.txt
#SBATCH -e logs/aa_build_sim_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=200G
#SBATCH -A r00939

# Build cached simulator tasks (adaptive_acq/sim_env.py build): random
# 200k-molecule subsamples, PCA-128, one .npz each. One array task per
# source dataset. Training sources (D4, AmpC) are deliberately different
# targets from the evaluation pool (ENHITS) -- see sim_env.py's leakage note.
# ENHITS tasks (index 2) are built only for an in-distribution upper-bound
# comparison, never for picking the controller you report.
#
#   sbatch --array=0-2 adaptive_acq/slurm/submit_build_sim_tasks.sh
#
# Most of the RAM goes to load_oracle() (116M-row D4 / 96M-row AmpC dicts);
# embeddings are memmapped and only the 200k sampled rows are read.

set -euo pipefail
source adaptive_acq/slurm/_preamble.sh

DATASETS=(D4_dedup AmpC_dedup_pca ENHITS)
ds="${DATASETS[${SLURM_ARRAY_TASK_ID:?submit with --array}]}"
OUT="${SIM_TASK_DIR:-adaptive_acq/sim_tasks}/$ds"

python -m adaptive_acq.sim_env build --dataset "$ds" \
    --backbones molformer mhgged smited unimol2 \
    --n-tasks 4 --n-sub 200000 --pca-dim 128 --seed 0 --out-dir "$OUT"
