#!/bin/bash

#SBATCH -J d4_scores_dedup
#SBATCH -p general
#SBATCH -o logs/d4_scores_dedup_%j.txt
#SBATCH -e logs/d4_scores_dedup_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# The D4 scores were never actually missing -- d4_smiles.txt (in
# $D4_ROOT) was extracted from d4_enamine_style.csv's "smiles" column only,
# dropping its "score" column, and the source d4_enamine_style.csv itself
# was left at its original location
# (/N/project/SingleCell_Image/Yang/AI Drug/LargeData/D4/), never copied
# into $D4_ROOT alongside d4_smiles.txt. Verified directly: that file's
# header is "smiles,score" (identical format to ampc_scores.csv.gz),
# 116,241,184 data rows, and row i's smiles column matches d4_smiles.txt's
# line i exactly (spot-checked row 500,000 vs. line 499,999, accounting
# for the CSV's header row).
#
# Step 1: gzip that CSV as-is into $D4_ROOT/d4_scores.csv.gz -- no
# reformatting needed, its header/column order already match
# ampc_scores.csv.gz's convention and it's already row-aligned with
# d4_smiles.txt.
# Step 2: apply the dedup keep-mask (built earlier via
# submit_d4_build_dedup_mask.sh, embed/stitch/dedup_smiles_and_scores.py)
# to BOTH d4_smiles.txt and the new d4_scores.csv.gz, writing deduplicated
# copies into $D4_ROOT/dedup/.
#
# Usage: sbatch slurm/embed/d4/submit_d4_build_scores_and_dedup.sh

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

D4_ROOT="/N/project/SingleCell_Image/mengjing/d4_138M"
RAW_SCORES_CSV="/N/project/SingleCell_Image/Yang/AI Drug/LargeData/D4/d4_enamine_style.csv"

echo "[step 1/2] gzip raw scores -> $D4_ROOT/d4_scores.csv.gz"
t0=$(date +%s)
gzip -c "$RAW_SCORES_CSV" > "$D4_ROOT/d4_scores.csv.gz"
echo "[step 1/2] done in $(( $(date +%s) - t0 ))s"

echo "[step 2/2] applying dedup keep-mask"
python embed/stitch/dedup_smiles_and_scores.py \
    --keep-mask "$D4_ROOT/d4_dedup_keep_mask.npy" \
    --smiles-file "$D4_ROOT/d4_smiles.txt" \
    --scores-file "$D4_ROOT/d4_scores.csv.gz" \
    --out-dir "$D4_ROOT/dedup"
