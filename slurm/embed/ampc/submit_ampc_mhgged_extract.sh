#!/bin/bash

#SBATCH -J ampc_mhgged
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_mhgged_%A_%a.txt
#SBATCH -e logs/ampc_mhgged_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Sharded MHG-GED (ibm-research materials.mhg-ged, dim 1024) embedding
# extraction for the AmpC 99.5M pool. Mirrors slurm/embed/enamine/submit_mhgged_extraction.sh
# (the EnamineHTS 2.1M version) -- same script, same GPU partition, just
# repointed at ampc_smiles.txt and a much bigger --num-chunks.
#
# PretrainedModelWrapper.encode() runs one molecule at a time through
# torch_geometric.from_smiles() + a per-molecule GNN forward -- no batched
# inference. The EnamineHTS-era comment guessed ~267 ms/mol; the actual
# rate on an idle h100-single node is ~5 ms/mol (measured 2026-09-01:
# 49,730-mol chunk in 247s = 4.97 ms/mol), i.e. ~137 GPU-hours for the
# whole 99.5M pool. That is cheap enough that NUM_CHUNKS just controls how
# many parallel tasks you want -- no need to over-shard.
#
# This cluster's MaxArraySize is 1000, so ONE array wave covers indices
# 0..999. Use NUM_CHUNKS=1000 and a single submission:
#
#   sbatch --array=0-999%64 slurm/embed/ampc/submit_ampc_mhgged_extract.sh 1000
#
#   NUM_CHUNKS=1000 -> ~99.5k mol/chunk -> ~8 min/chunk, ~2.5 h wall at %64
#
# (%64 = at most 64 of this array's tasks running at once -- be a good
# h100-single citizen.) Each task writes its own independent chunk file and
# self-skips if it already exists, so just re-fire the same line to mop up
# timeouts/stragglers.
#
# OFFSET (3rd positional arg, default 0): real chunk id = ARRAY_TASK_ID +
# OFFSET. Only needed if you deliberately pick NUM_CHUNKS > 1000 and have to
# cover the high half in a second wave (e.g. `... 2000 1000`).
#
# total-count 99459561 == `wc -l ampc_smiles.txt` (verified), identical to
# every other AmpC extraction script so _chunk_bounds() lines up with the
# stitch step.
#
# IMPORTANT: if you change NUM_CHUNKS after a partial run, wipe
# $AMPC_ROOT/_mhgged_chunks first -- the compute script skips any existing
# chunk file by name without checking its row count, so leftover chunks
# from a different NUM_CHUNKS would be silently wrong.
#
# After ALL chunks exist (pass the SAME NUM_CHUNKS):
#   sbatch slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh mhgged 1000

set -euo pipefail

NUM_CHUNKS="${1:?Usage: sbatch --array=0-999 slurm/embed/ampc/submit_ampc_mhgged_extract.sh <NUM_CHUNKS> [OFFSET]}"
OFFSET="${2:-0}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?submit with --array=0-999} + OFFSET ))

if [ "$TASK_ID" -ge "$NUM_CHUNKS" ]; then
    echo "[skip] chunk-id $TASK_ID >= NUM_CHUNKS $NUM_CHUNKS -- nothing to do for this task"
    exit 0
fi

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="/N/project/SingleCell_Image/mengjing/ampc_99.5M"

srun --cpu-bind=none python embed/compute/compute_mhgged_embeddings_chunk.py \
    --smiles-file "$AMPC_ROOT/ampc_smiles.txt" --total-count 99459561 \
    --chunk-id "$TASK_ID" --num-chunks "$NUM_CHUNKS" \
    --chunks-dir "$AMPC_ROOT/_mhgged_chunks"
