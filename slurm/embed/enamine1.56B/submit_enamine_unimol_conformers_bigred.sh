#!/bin/bash

#SBATCH -J er_u1_conf
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=FAIL
#SBATCH -p general
#SBATCH -o logs/er_u1_conf_%A_%a.txt
#SBATCH -e logs/er_u1_conf_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Stage 1 of Uni-Mol v1 for the Enamine REAL lead-like pool (1,531,875,935
# deduplicated molecules): RDKit ETKDG conformer generation, CPU-only, on
# BigRed200's `general` partition. Stage 2 (compute_unimol_embeddings_chunk.py)
# needs a GPU and is a separate job.
#
# Sizing is measured, not extrapolated: D4's own 82-chunk v1 run (job
# 10592862) did 1,417,528 molecules/chunk in a median 6,733 s with these same
# 16 workers -- 4.75 ms/molecule -- and wrote 347 GB for 116M molecules.
# Scaled to this pool:
#   1000 chunks x 1,531,876 molecules = ~2,021 chunk-hours, ~4.6 TB output
#   per chunk ~7,280 s median, ~11,800 s at D4's worst-case chunk
# so --time=12:00:00 holds with margin. Wall time is set by how many array
# tasks run at once: ~5 h at 400 concurrent, ~10 h at 200, ~20 h at 100.
#
# NUM_CHUNKS is a positional arg and MUST match stage 2's --num-chunks
# exactly; chunk boundaries are recomputed there and asserted.
#
# Run this AFTER submit_enamine_build_pool.sh. It reads the deduplicated,
# shuffled pool, so (a) no conformer is generated for the 27,977,307
# duplicate rows, and (b) chunk i lines up with pool rows [start_i, end_i)
# with no permutation bookkeeping. Generating from the raw surA_smiles.txt
# instead would waste ~36 chunk-hours and leave every later stage to remap
# through permutation.npy. Chaining the two jobs is the simple way:
#   POOL=$(sbatch --parsable slurm/embed/enamine1.56B/submit_enamine_build_pool.sh)
#   sbatch --dependency=afterok:$POOL --array=0-999%400 \
#       slurm/embed/enamine1.56B/submit_enamine_unimol_conformers_bigred.sh 1000
#
# Resumable: generate_unimol_conformers_chunk.py skips molecules already
# present in a chunk's lmdb, so re-running a timed-out task continues it.
# To re-run only the chunks that never finished, diff the written .lmdb
# files against the full 0-999 range (the same approach used for MolFormer's
# 26 timed-out chunks, report sect. 16) rather than assuming contiguity.

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-(N-1) $0 <NUM_CHUNKS>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-(NUM_CHUNKS-1)}"

ER_ROOT="${ER_ROOT:-/N/project/SingleCell_Image/mengjing/enamine_real_1.56B}"
SMILES="$ER_ROOT/dedup/enamine_surA_smiles.txt"
TOTAL=1531875935

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# Stripe the conformer folder before the first chunk is written: ~4.6 TB
# landing single-stripe would have to be migrated later (0.8-18.7 h/file
# when that happened to the dedup embeddings, report sect. 16).
mkdir -p "$ER_ROOT/_unimol_conformers"
lfs setstripe -c -1 "$ER_ROOT/_unimol_conformers" 2>/dev/null || true

[ -f "$SMILES" ] || { echo "[conf] ERROR: $SMILES not found -- run submit_enamine_build_pool.sh first"; exit 1; }

echo "[conf] chunk $TASK_ID/$NUM_CHUNKS  smiles=$SMILES  total=$TOTAL"

srun --cpu-bind=none python generate_unimol_conformers_chunk.py \
    --smiles-file "$SMILES" --total-count "$TOTAL" \
    --out-dir "$ER_ROOT/_unimol_conformers" \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" --n-conformer 1 \
    --num-workers 16
