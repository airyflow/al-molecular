#!/bin/bash

#SBATCH -J ampc_u2_conf_probe
#SBATCH -p general
#SBATCH -o logs/ampc_u2_conf_probe_%j.txt
#SBATCH -e logs/ampc_u2_conf_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# THROWAWAY diagnostic for Stage 1 (generate_unimol2_conformers_chunk.py,
# CPU-only conformer generation) -- v2, larger samples.
#
# v1 (job 8176422, 500/2000/500-warm samples) came back NON-MONOTONIC
# (500-warm: 383.4s, 2000: 336.9s -- the bigger sample finished faster),
# proving fixed overhead + this shared partition's run-to-run node
# contention completely swamped the true per-molecule signal at that
# scale (RDKit conformer generation is genuinely cheap per molecule --
# ~19ms/mol extrapolated from Uni-Mol v1's own real precedent -- so even
# 2000 molecules is only ~38s of real variable-cost work, buried under
# several hundred seconds of noise). Scaling up 10-25x so true per-
# molecule cost has a chance to dominate over that noise floor.
#
# Design: run SAMPLE_A once first (absorbs cold-start disk-cache-miss
# overhead for the smiles file / shared libraries -- DISCARD this number),
# then SAMPLE_A again (warm -- KEEP) and SAMPLE_B (also warm, since the
# first run already touched the shared OS-level caches on this node --
# KEEP). Only the two "KEEP" numbers go into the two-point fit; comparing
# them tells you immediately if this run was also too noisy (non-
# monotonic again -> still dominated by node contention, not molecule
# count -- consider requesting a less-contended node or accepting wider
# error bars on the final job sizing).
#
# After this runs, read the two "KEEP" numbers from the log and compute:
#   variable_s_per_mol = (elapsed_B - elapsed_A) / (SAMPLE_B - SAMPLE_A)
#   fixed_s            = elapsed_A - SAMPLE_A * variable_s_per_mol
# then extrapolate to a real ~99,460-molecule chunk and the full
# 1000-chunk job:
#   per_chunk_s     = fixed_s + 99460 * variable_s_per_mol
#   total_cpu_hours = 1000 * per_chunk_s / 3600
#
# Usage:
#   sbatch slurm/embed/ampc/submit_ampc_unimol2_conformers_timing_probe_bigred.sh

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

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

AMPC_ROOT="${AMPC_ROOT:-/N/project/SingleCell_Image/mengjing/ampc_99.5M}"
PROBE_DIR="$AMPC_ROOT/_unimol2_conformers_timing_probe"
rm -rf "$PROBE_DIR"
mkdir -p "$PROBE_DIR"

SAMPLE_A=20000
SAMPLE_B=50000

head -n "$SAMPLE_A" "$AMPC_ROOT/ampc_smiles.txt" > "$PROBE_DIR/sample_a.txt"
head -n "$SAMPLE_B" "$AMPC_ROOT/ampc_smiles.txt" > "$PROBE_DIR/sample_b.txt"

echo "=== cold pass: $SAMPLE_A molecules (DISCARD this number -- pays first-touch disk-cache overhead) ==="
python generate_unimol2_conformers_chunk.py \
    --smiles-file "$PROBE_DIR/sample_a.txt" --total-count "$SAMPLE_A" \
    --out-dir "$PROBE_DIR/out_a_cold" --chunk-id 0 --num-chunks 1 --num-workers 16

echo
echo "=== KEEP 1: $SAMPLE_A molecules (warm) ==="
python generate_unimol2_conformers_chunk.py \
    --smiles-file "$PROBE_DIR/sample_a.txt" --total-count "$SAMPLE_A" \
    --out-dir "$PROBE_DIR/out_a_warm" --chunk-id 0 --num-chunks 1 --num-workers 16

echo
echo "=== KEEP 2: $SAMPLE_B molecules (warm) ==="
python generate_unimol2_conformers_chunk.py \
    --smiles-file "$PROBE_DIR/sample_b.txt" --total-count "$SAMPLE_B" \
    --out-dir "$PROBE_DIR/out_b" --chunk-id 0 --num-chunks 1 --num-workers 16

echo
echo "[probe done] -- use the two KEEP lines' elapsed times for the two-point fit (see header comment)"
