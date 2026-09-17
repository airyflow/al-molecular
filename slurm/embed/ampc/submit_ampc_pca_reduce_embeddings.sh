#!/bin/bash

#SBATCH -J ampc_pca_reduce
#SBATCH -p general
#SBATCH -o logs/ampc_pca_reduce_%A_%a.txt
#SBATCH -e logs/ampc_pca_reduce_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH -A r00939

# Runs embed/stitch/pca_reduce_embeddings.py (see that file's own docstring
# for the two-pass fit/transform design) against the 5 remaining AmpC
# backbones NOT yet PCA-reduced (grover3400/mhgged/molformer/smited/unimol2
# -- unimol itself already done as a pilot: 512-d -> 114-d, 95.01% variance
# retained, verified against a full-data-fit ground truth and a
# simulated-resume test on synthetic data before being trusted on the real
# 202GB file). Pure CPU/IO work, no GPU needed -- same reasoning as
# submit_ampc_stitch_embeddings_striped.sh.
#
# ARRAY job, one task per backbone (--array=0-4) -- unlike the AL predict
# workers (predict_pool_shard_worker.py), which all read the SAME shared
# embedding files concurrently and genuinely contend for the same I/O
# path, each backbone here reads a completely INDEPENDENT source file, so
# there's no shared-file contention to avoid by serializing. Running all 5
# concurrently cuts wall-clock from the sum of all 5 (sequential, worst
# case >30h given grover3400 alone is ~1.34TB) down to roughly whichever
# ONE backbone is slowest.
#
# --cpus-per-task=16/--mem=64G sized for the worst case (grover3400,
# dim=3400) even though the other 4 backbones need much less: full-SVD fit
# cost scales roughly with dim^2, not dim, so grover3400's fit step alone
# could plausibly run ~44x longer than the unimol pilot's 651.8s (d=512)
# just from that scaling, on top of the larger dim's proportionally larger
# I/O volume -- a single array job can't request different resources per
# task, so every task gets the same generous allocation; the 4 smaller
# backbones simply won't need all of it.
#
# Safe to resubmit (same --array=0-4) if any task hits its --time cutoff:
# pca_reduce_embeddings.py's own --resume skips a backbone already fully
# written and resumes a partially-written one from its last completed row
# (binary-searched, same technique as dedup_embeddings.py) -- no progress
# lost. Also safe/idempotent to resubmit only the specific failed index
# (e.g. --array=4 alone) rather than all 5.
#
# --n-components 0.95 (a variance-retention threshold, not a fixed width)
# for every backbone -- see pca_reduce_embeddings.py's note on why this
# requires svd_solver="full" internally (a real bug hit and fixed during
# the pilot: "randomized" doesn't support a float n_components at all).
#
# Output stripe count: dedup_pca/ is a brand-new directory, created with
# the filesystem's default (unstriped, single-OST) layout unless set
# explicitly -- same issue submit_ampc_stitch_embeddings_striped.sh's own
# comment describes finding for embeddings_striped/. Idempotent to run
# from every array task (mkdir -p + setstripe on an already-configured
# directory is a no-op).
#
# Usage: sbatch --array=0-4 slurm/embed/ampc/submit_ampc_pca_reduce_embeddings.sh
#
# Big Red 200 notes (this cluster, not Quartz -- see
# al-eval-framework/scripts/submit_conformer_gen.sh, which runs there):
#   - srun on this cluster's Cray MPI does not reliably inherit
#     --cpus-per-task from the sbatch allocation for CPU-binding purposes
#     without SRUN_CPUS_PER_TASK explicitly exported -- srun otherwise
#     aborts at launch with "CPU binding outside of job step allocation"
#     (observed directly on that repo's job 7766000).
#   - Confirmed /N/project/SingleCell_Image and /N/slate/mengjing/miniconda3
#     are both reachable here (al-eval-framework sources the exact same
#     conda.sh path and reads/writes the exact same /N/project/SingleCell_Image
#     tree from this cluster), so no path changes vs. the Quartz-side
#     al-molecular scripts were needed beyond the srun/CPU-binding fix.

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-4 (SLURM_ARRAY_TASK_ID is unset)}"

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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"
DEDUP_DIR="$AMPC_ROOT/dedup"
OUT_DIR="$AMPC_ROOT/dedup_pca"

mkdir -p "$OUT_DIR"
lfs setstripe -c -1 "$OUT_DIR" 2>/dev/null || true
echo "[stripe] $OUT_DIR default layout: $(lfs getstripe -c "$OUT_DIR")-way (new files created inside inherit this)"

# index 0..4 -> backbone -- ORDER here is cosmetic (all 5 run concurrently
# as separate array tasks, not sequentially), kept smallest-to-largest by
# convention only.
BACKBONES=(mhgged molformer smited unimol2 grover3400)
DIMS=(1024 768 768 1536 3400)
BACKBONE="${BACKBONES[$TASK_ID]}"
DIM="${DIMS[$TASK_ID]}"

echo "[task $TASK_ID] backbone=$BACKBONE dim=$DIM"
srun --cpu-bind=none python3 -u embed/stitch/pca_reduce_embeddings.py --backbone "$BACKBONE" --dim "$DIM" --n-components 0.95 \
    --embeddings-path "$DEDUP_DIR/${BACKBONE}_embeddings.npy" \
    --out-path "$OUT_DIR/${BACKBONE}_embeddings.npy" \
    --n-sample 1000000 --seed 42 --resume
echo "[task $TASK_ID] $BACKBONE done"
