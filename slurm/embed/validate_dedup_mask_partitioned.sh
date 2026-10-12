#!/bin/bash

#SBATCH -J dedup_part_check
#SBATCH -p general
#SBATCH -o logs/dedup_part_check_%j.txt
#SBATCH -e logs/dedup_part_check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH -A r00939

# Validates embed/stitch/build_dedup_mask_partitioned.py (1.3B design, Fix 2)
# against the keep-masks already in use: rebuilds the AmpC (99,459,561 rows)
# and D4 (116,241,184 rows) masks from the raw SMILES files and checks they
# are identical to the existing ones. Writes new masks and partition files
# under $WORK only; never touches the existing masks.
#
# --mem=16G: the partitioned build never holds more than one batch of lines,
# one partition, and the final boolean mask (~116 MB) -- the point of Fix 2.
#
# Usage: sbatch slurm/embed/validate_dedup_mask_partitioned.sh

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

P=/N/project/SingleCell_Image/mengjing
WORK="$P/dedup_partitioned_check"
mkdir -p "$WORK"

check() {  # name smiles_file existing_mask
    echo "=== $1"
    /usr/bin/time -v python3 -u embed/stitch/build_dedup_mask_partitioned.py --smiles-file "$2" \
        --work-dir "$WORK/$1_parts" --out-path "$WORK/$1_keep_mask.npy" --stage all 2>&1 \
        | grep -E "^\[scan\] done|^\[done\]|Maximum resident|Elapsed \(wall"
    python3 -c "
import numpy as np
a=np.load('$WORK/$1_keep_mask.npy'); b=np.load('$3')
print('$1: identical to existing mask:', a.shape==b.shape and bool(np.array_equal(a,b)), a.sum(), b.sum())"
    rm -rf "$WORK/$1_parts"
}
check ampc "$P/ampc_99.5M/ampc_smiles.txt" "$P/ampc_99.5M/dedup/keep_mask.npy"
check d4   "$P/d4_138M/d4_smiles.txt"      "$P/d4_138M/d4_dedup_keep_mask.npy"
