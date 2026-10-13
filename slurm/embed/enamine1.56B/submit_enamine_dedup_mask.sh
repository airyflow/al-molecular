#!/bin/bash

#SBATCH -J er_dedup_mask
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p general
#SBATCH -o logs/er_dedup_mask_%j.txt
#SBATCH -e logs/er_dedup_mask_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# Dedup keep-mask for the Enamine REAL lead-like library (1.56B rows), the
# first step before any embedding extraction -- duplicates would otherwise
# be embedded, stored and scored more than once.
#
# Uses the partitioned builder (embed/stitch/build_dedup_mask_partitioned.py),
# which hashes each SMILES to 128 bits and sorts within 256 partitions
# instead of holding every SMILES in one dict. Validated against the old
# single-dict builder on AmpC and D4: identical masks, 1.6 GB peak
# (report sect. 16, job 11926364).
#
# Scale, extrapolated from the validated runs (AmpC 99.5M in 3 min 03 s,
# D4 116M in 3 min 35 s, both --stage all): 1.56B rows is 13.4x D4, so
# expect roughly 45-60 min, with the scan dominating. --time=12:00:00 is
# deliberately generous: the scan is a single-process Python line loop and
# this is the first run at this scale. Memory stays ~2 GB (hashing streams;
# only the merge holds the 1.56 GB bool mask), so 32G is already ample.
#
# Disk in --work-dir: 24 bytes/row x 1.56B = ~37 GB of partition files,
# deleted by hand after the mask is written.
#
# Usage:
#   sbatch slurm/embed/enamine1.56B/submit_enamine_dedup_mask.sh
#
# If the scan proves too slow in one process, the design doc's fallback is
# to split it by byte range at line boundaries (docs/scaling_to_1.3B_design.md,
# Fix 2); the reduce stage is already array-friendly (--stage reduce --part N).

set -euo pipefail

ER_ROOT="/N/project/SingleCell_Image/mengjing/enamine_real_1.56B"
SMILES="$ER_ROOT/surA_smiles.txt"
WORK="$ER_ROOT/dedup_work"
OUT="$ER_ROOT/enamine_dedup_keep_mask.npy"

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
mkdir -p logs "$WORK"

echo "[dedup] smiles=$SMILES"
echo "[dedup] work=$WORK  out=$OUT"
df -h "$ER_ROOT" | tail -1

python3 -u embed/stitch/build_dedup_mask_partitioned.py \
    --smiles-file "$SMILES" \
    --work-dir "$WORK" \
    --out-path "$OUT" \
    --stage all

echo "[dedup] done; partition files in $WORK can now be deleted"
