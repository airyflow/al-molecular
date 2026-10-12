#!/bin/bash

#SBATCH -J pca_stage
#SBATCH -p general
#SBATCH -o logs/pca_stage_%A_%a.txt
#SBATCH -e logs/pca_stage_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -A r00939

# One stage of the row-range parallel PCA (pca_reduce_embeddings.py --stage
# gather|fit|transform). Normally submitted by submit_pca_parallel.sh, which
# chains the three stages with --dependency=afterok; see that script.
#
# Args: <stage> <backbone> <dim> <embeddings-path> <out-path> <num-tasks>

set -euo pipefail
STAGE="$1"; BACKBONE="$2"; DIM="$3"; SRC="$4"; OUT="$5"; NUM_TASKS="$6"

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
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python3 -u embed/stitch/pca_reduce_embeddings.py --stage "$STAGE" \
    --backbone "$BACKBONE" --dim "$DIM" --n-components 0.95 \
    --embeddings-path "$SRC" --out-path "$OUT" \
    --n-sample 1000000 --seed 42 --resume --num-tasks "$NUM_TASKS"
