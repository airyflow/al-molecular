#!/bin/bash

#SBATCH -J restripe_dedup
#SBATCH -p general
#SBATCH -o logs/restripe_dedup_%A_%a.txt
#SBATCH -e logs/restripe_dedup_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH -A r00939

# Restripes the deduplicated embedding files across all Lustre storage
# servers in place (lfs migrate -c -1). Found 2026-10-12: every
# ampc_99.5M/dedup/*_embeddings.npy and d4_138M/dedup/*_embeddings.npy has
# stripe count 1 (the whole file on ONE server), because
# dedup_embeddings.py's output folder was never striped -- unlike the
# embeddings_striped/ files they were made from (28 stripes). Every later
# reader (PCA, LT-All rounds, prediction workers) was capped at one server's
# bandwidth.
#
# lfs migrate rewrites each file with the new layout; the contents are
# unchanged. Do NOT run while any job reads or writes these files. One file
# per array task (12 files, 4.8 TB total):
#   sbatch --array=0-11 slurm/embed/restripe_dedup_embeddings.sh

set -euo pipefail
P=/N/project/SingleCell_Image/mengjing
FILES=(
  "$P/ampc_99.5M/dedup/grover3400_embeddings.npy"
  "$P/ampc_99.5M/dedup/unimol2_embeddings.npy"
  "$P/ampc_99.5M/dedup/mhgged_embeddings.npy"
  "$P/ampc_99.5M/dedup/molformer_embeddings.npy"
  "$P/ampc_99.5M/dedup/smited_embeddings.npy"
  "$P/ampc_99.5M/dedup/unimol_embeddings.npy"
  "$P/d4_138M/dedup/grover3400_embeddings.npy"
  "$P/d4_138M/dedup/unimol2_embeddings.npy"
  "$P/d4_138M/dedup/mhgged_embeddings.npy"
  "$P/d4_138M/dedup/molformer_embeddings.npy"
  "$P/d4_138M/dedup/smited_embeddings.npy"
  "$P/d4_138M/dedup/unimol_embeddings.npy"
)
F="${FILES[$SLURM_ARRAY_TASK_ID]}"
echo "[restripe] $F: stripe count $(lfs getstripe -c "$F") before; size $(du -h --apparent-size "$F" | cut -f1)"
start=$(date +%s)
lfs migrate -c -1 "$F"
echo "[restripe] $F: stripe count $(lfs getstripe -c "$F") after, $(( $(date +%s) - start ))s"
# New files in these folders default to all servers too.
lfs setstripe -c -1 "$(dirname "$F")"
