#!/bin/bash

#SBATCH -J ampc_unimol_embed_h100
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/ampc_unimol_embed_h100_%A_%a.txt
#SBATCH -e logs/ampc_unimol_embed_h100_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# GPU (h100-single) replacement for submit_ampc_unimol_embed.sh's CPU-only
# Stage 2 -- moved off the two local A100s (measured too slow: steady-
# state throughput observed at 1.2-2.3 ms/mol, i.e. ~30-55 min/chunk once
# past the first, cache-warm chunk, putting a full local 70-chunk/2-GPU
# run at ~17-32 hours) and off BigRed200's contended general/CPU
# partition, onto Quartz's h100-single (2-day/job cap, `sinfo`-confirmed
# up-to-date name for the single-GPU H100 partition -- see
# submit_al_runs_h100single.sh for the "hopper" partition's removal).
#
# No array concurrency cap -- SLURM only starts as many of the 70 tasks
# as there are free h100-single nodes, so a self-imposed limit here would
# only slow things down, not protect anything. `sinfo` at submission time
# showed h100-single mostly busy (only a handful of idle nodes), so most
# tasks will queue and backfill as nodes free up; that's expected, not a
# failure.
#
# Uses BOTH fixes applied to compute_unimol_embeddings_chunk.py this
# session: (1) conformer-duplication fix (each LMDB shard stores 2
# conformers/molecule -- 1 real + 1 always-appended 2D fallback -- only
# the first is now kept, previously every chunk silently shipped 2x rows)
# and (2) a fixed torch.manual_seed() before model construction (the
# checkpoint's strict=False load leaves hidden_layer, the final
# embedding projection, at PyTorch's random init -- unseeded, every
# separate chunk process got its own incomparable random projection).
#
# --num-chunks MUST stay 70 -- must match Stage 1's shard count exactly.
# Each array task writes its own independent chunk file (skip-if-exists
# built into the script itself -- already-completed chunks from the
# earlier local run, e.g. 0/1/35/36, are detected and skipped
# automatically, so this array can safely be submitted to finish the
# remaining chunks without redoing completed work).
#
# --time=36:00:00 sized well above the slowest local chunk observed so
# far (3292.9s ~ 55min); large margin given per-chunk variance already
# seen (0.12-2.32 ms/mol range) and unknown Quartz H100 vs local A100
# relative speed.
#
# After ALL 70 chunks exist, run stitch_embedding_chunks.py once (see
# submit_ampc_stitch_embeddings.sh) to produce the final
# unimol_embeddings.npy that EmbeddingFeaturizer.load() expects, then
# verify (row count, no NaN/Inf, no unstitched rows) before trusting it
# for the AmpC fusion/learned_fusion AL runs.
#
# Usage:
#   sbatch --array=0-69 submit_ampc_unimol_embed_h100single.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-69 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python3 compute_unimol_embeddings_chunk.py \
    --smiles-file /N/project/SingleCell_Image/mengjing/ampc_99.5M/ampc_smiles.txt \
    --total-count 99459561 \
    --shards-dir /N/project/SingleCell_Image/mengjing/ampc_99.5M/_unimol_conformers/_shards \
    --chunk-id "$TASK_ID" \
    --num-chunks 70 \
    --chunks-dir /N/project/SingleCell_Image/mengjing/ampc_99.5M/_unimol_chunks \
    --num-workers 4
