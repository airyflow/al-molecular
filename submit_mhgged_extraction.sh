#!/bin/bash

#SBATCH -J mhgged_gen
#SBATCH -p h100-single
#SBATCH --gpus-per-node=1
#SBATCH -o logs/mhgged_gen_%A_%a.txt
#SBATCH -e logs/mhgged_gen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# NOTE: -p gpu / --gpus-per-node=1 mirror submit_grover_extraction.sh's guess
# at the GPU partition name -- verify against `sinfo` on the actual cluster
# and adjust if wrong. This session (colo27, no SLURM client available) could
# not verify it directly.
#
# Sharded MHG-GED (IBM materials.mhg-ged) embedding extraction for the
# EnamineHTS 2.1M pool. Needed because PretrainedModelWrapper.encode()
# (ibm_materials/models/mhg_model/load.py) loops one molecule at a time via
# torch_geometric's from_smiles() -- no real batched GPU inference, despite
# --batch-size. Measured directly (2026-08-18, this session's own 2xA100,
# 100-molecule sample): 267.86 ms/mol -> ~159.3hr for the full 2.1M pool as
# one process. --num-chunks 50 (~42,830 molecules/shard) targets ~3.2hr/shard
# from that same measurement -- an estimate from 100 molecules, not the
# per-shard scale, so check the first completed shard's actual log line
# before trusting it (same caveat as submit_ampc_molformer_extract.sh).
# --mem 32G matches GROVER's chunk allocation as a starting point: MHG-GED
# hits the identical torch_geometric pyg-lib/torch-sparse GLIBC_2.29 fallback
# (confirmed in this session's own smoke test output) that made GROVER's
# RDKit/graph path memory-heavy, but this has not been directly measured for
# MHG-GED's chunk size -- bump if a shard OOMs.
#
# embedding_dim = 1024, confirmed via smoke test (not published on IBM's
# model card).
#
# --total-count 2141500 (not the 2,141,515 the README's prose quotes for the
# ORIGINAL EnamineHTS.csv.gz library) -- that's the real line count of
# enamine_smiles_canonical.txt (`wc -l`, confirmed 2026-08-18), which is what
# every extraction script here actually reads from. Matches
# submit_grover_extraction.sh's own documented usage example (2141500).
# The first real run of this script used 2141515 by mistake; chunks 0-48
# still ended up correct (their boundaries never reach real EOF), only
# chunk 49 undershot (42,781 real rows vs. 42,796 expected from the wrong
# total) and only the stitch step's validation caught it -- recovered via a
# one-off cumulative-offset restitch rather than a costly re-extraction,
# since all 50 chunk files' data was already complete and correctly ordered.
#
# Each task writes its own independent chunk file -- no shared state, no
# coordination, no race condition possible between the 50 parallel tasks.
# After all 50 finish, run stitch_embedding_chunks.py once (added "smited"/
# "mhgged" to its --backbone choices) to produce the final shared .npy that
# EmbeddingFeaturizer.load() expects:
#   python stitch_embedding_chunks.py --backbone mhgged --dim 1024 \
#       --chunks-dir results/embed/EnamineHTS/_mhgged_chunks \
#       --num-chunks 50 --total-count 2141500 \
#       --embeddings-path results/embed/EnamineHTS/mhgged_embeddings.npy
#
# Usage (array job, one task per chunk):
#   sbatch --array=0-49 submit_mhgged_extraction.sh

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-49 (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

srun --cpu-bind=none python compute_mhgged_embeddings_chunk.py \
    --smiles-file results/embed/EnamineHTS/enamine_smiles_canonical.txt --total-count 2141500 \
    --chunk-id "$TASK_ID" --num-chunks 50 \
    --chunks-dir results/embed/EnamineHTS/_mhgged_chunks
