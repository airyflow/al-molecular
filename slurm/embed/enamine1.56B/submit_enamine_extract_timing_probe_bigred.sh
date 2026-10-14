#!/bin/bash

#SBATCH -J er_probe_bigred
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=FAIL
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o logs/er_probe_bigred_%j.txt
#SBATCH -e logs/er_probe_bigred_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -A r00939

# THROWAWAY diagnostic: how fast is BigRed200's GPU partition for embedding
# extraction, compared with the Quartz H100 numbers everything is currently
# sized from?
#
# Every per-molecule rate in the enamine1.56B scripts was measured on Quartz
# h100-single:  SMI-TED 0.35, Uni-Mol v1 1.76, MHG-GED 2.12 ms/mol. BigRed200's
# `gpu` partition is A100-PCIE-40GB (untyped gpu:4 GRES; confirmed in this repo
# via nvidia-smi -L, see slurm/al_runs/enamine/submit_al_runs_bigred.sh), which
# is slower than an H100 for this inference and has half the memory. Extraction
# is 1,814 GPU-h at H100 rates; a 2-3x slowdown would make it 3,600-5,400, so
# the factor is worth one 4 h probe before committing the array jobs.
#
# Method: run ONE small chunk per backbone and read the "(N ms/mol)" the
# compute scripts already print. --num-chunks 100000 makes each chunk
# 1,531,875,935/100000 = ~15,319 molecules: a few seconds to a minute of real
# work, enough to dominate model-load overhead without burning the allocation.
# Chunk 1 is used rather than chunk 0 so a cold first-chunk read of the SMILES
# file is not what gets timed.
#
# Output goes to a throwaway --chunks-dir, NOT the real one: these chunk files
# are computed with a different --num-chunks, so leaving them where the real
# run writes would silently corrupt it (the compute scripts skip an existing
# chunk file by name without checking its row count).
#
# Read the three "ms/mol" numbers from the log and divide by the H100 values
# above to get the slowdown factor. If it is near 1, redirect the real jobs to
# BigRed200 with command-line overrides, which take precedence over the
# scripts' own #SBATCH lines, no edits needed:
#   sbatch -p gpu --gpus-per-node=1 --array=0-999%64 \
#       slurm/embed/enamine1.56B/submit_enamine_smited_extract.sh 1000
# If it is 2-3x, decide per backbone: BigRed200 may still win on wall time if
# it grants far more concurrent GPU tasks than Quartz's h100-single does.
#
# Usage:
#   sbatch slurm/embed/enamine1.56B/submit_enamine_extract_timing_probe_bigred.sh

set -euo pipefail

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

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

ER_ROOT="${ER_ROOT:-/N/project/SingleCell_Image/mengjing/enamine_real_1.56B}"
SMILES="$ER_ROOT/dedup/enamine_surA_smiles.txt"
TOTAL=1531875935
PROBE_CHUNKS=100000
CHUNK_ID=1
PROBE_DIR="$ER_ROOT/_probe_chunks_$SLURM_JOB_ID"

echo "[probe] node=$(hostname)"
nvidia-smi -L || true
echo "[probe] ~$((TOTAL / PROBE_CHUNKS)) molecules per probe chunk"
echo "[probe] Quartz H100 reference: SMI-TED 0.35, Uni-Mol v1 1.76, MHG-GED 2.12 ms/mol"
echo

mkdir -p "$PROBE_DIR"

if [ -f "$SMILES" ]; then
    for spec in "smited:compute_smited_embeddings_chunk.py" "mhgged:compute_mhgged_embeddings_chunk.py"; do
        tag="${spec%%:*}"; script="${spec##*:}"
        echo "=== $tag"
        python -u "embed/compute/$script" \
            --smiles-file "$SMILES" --total-count "$TOTAL" \
            --chunk-id "$CHUNK_ID" --num-chunks "$PROBE_CHUNKS" \
            --chunks-dir "$PROBE_DIR/$tag" || echo "[probe] $tag FAILED"
        echo
    done
else
    echo "[probe] SKIP smited/mhgged: $SMILES not found (run submit_enamine_build_pool.sh first)"
fi

# Uni-Mol v1 reads the raw source-order file and the existing conformer shards,
# so its probe must use the raw total -- and its chunking is pinned to stage 1's
# 1000 shards, so a probe cannot re-chunk. Chunk 0 of the real run is timed
# instead; it writes a real chunk file, into the throwaway dir.
CONF_DIR="$ER_ROOT/_conformer_output/_unimol_cache_7d48f755fa/_chunks"
if [ -d "$CONF_DIR" ]; then
    echo "=== unimol v1 (full 1,559,854-molecule shard -- the real chunk size)"
    python -u embed/compute/compute_unimol_embeddings_chunk.py \
        --smiles-file "$ER_ROOT/surA_smiles.txt" --total-count 1559853242 \
        --conformer-chunks-dir "$CONF_DIR" \
        --chunk-id 0 --num-chunks 1000 \
        --chunks-dir "$PROBE_DIR/unimol" || echo "[probe] unimol FAILED"
else
    echo "[probe] SKIP unimol: conformer shards not found at $CONF_DIR"
fi

echo
echo "[probe] done. Throwaway output in $PROBE_DIR -- delete it:"
echo "  rm -rf $PROBE_DIR"
