#!/bin/bash

#SBATCH -J d4_mhgged
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/d4_mhgged_%A_%a.txt
#SBATCH -e logs/d4_mhgged_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Sharded MHG-GED (dim 1024) embedding extraction for the D4 pool --
# same compute script, same GPU partition as
# slurm/embed/ampc/submit_ampc_mhgged_extract.sh, repointed at D4's own
# smiles file/root/total-count. PretrainedModelWrapper.encode() rate
# measured on AmpC (idle h100-single, 2026-09-01): ~4.97 ms/mol -- not
# re-measured for D4, but D4's molecules are a similar general-purpose
# ZINC15/Enamine-style small-molecule library, so this is a reasonable
# starting assumption (unlike grover3400/unimol/unimol2, this backbone is
# cheap enough that a 2-3x miss on the estimate is not a big deal).
#
# This cluster's MaxArraySize is 1000, so ONE array wave covers indices
# 0..999. Use NUM_CHUNKS=1000 and a single submission:
#
#   sbatch --array=0-999%64 slurm/embed/d4/submit_d4_mhgged_extract.sh 1000
#
#   NUM_CHUNKS=1000 -> ~116.2k mol/chunk -> ~9.6 min/chunk at AmpC's
#   measured rate, ~2.9h wall at %64 -- re-check the first completed
#   shard's actual log line before trusting this.
#
# OFFSET (2nd positional arg, default 0): real chunk id = ARRAY_TASK_ID +
# OFFSET. Only needed if NUM_CHUNKS > 1000 (a second wave for the high half).
#
# total-count 116241184 == `wc -l d4_smiles.txt` (verified directly,
# 2026-09-17) -- NOTE this is D4_ROOT/d4_smiles.txt's count, built from
# d4_enamine_style.csv (already scored-molecules-only, 116,241,184 rows),
# NOT the raw d4.csv's 138,312,677 rows (which includes ~22M rows with an
# empty dockscore field -- confirmed directly by inspection, e.g.
# ZINC000000000021/ZINC000000000030 have blank dockscore in d4.csv).
# d4_enamine_style.csv appears to already be the scored-only subset,
# mirroring how AmpC's own oracle drops unscored rows -- see
# run_experiment.py's "[oracle] dropped N/M rows with a non-numeric/
# missing score" pattern. Must match exactly for _chunk_bounds() to line
# up with the stitch step.
#
# IMPORTANT: if you change NUM_CHUNKS after a partial run, wipe
# $D4_ROOT/_mhgged_chunks first -- the compute script skips any existing
# chunk file by name without checking its row count, so leftover chunks
# from a different NUM_CHUNKS would be silently wrong.
#
# After ALL chunks exist (pass the SAME NUM_CHUNKS):
#   sbatch slurm/embed/d4/submit_d4_stitch_embeddings.sh mhgged 1000

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/d4/submit_d4_mhgged_extract.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do for this task"
    exit 0
fi

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

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"

srun --cpu-bind=none python embed/compute/compute_mhgged_embeddings_chunk.py \
    --smiles-file "$D4_ROOT/d4_smiles.txt" --total-count 116241184 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$D4_ROOT/_mhgged_chunks"
