#!/bin/bash

#SBATCH -J er_build_pool
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p general
#SBATCH -o logs/er_build_pool_%j.txt
#SBATCH -e logs/er_build_pool_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH -A r00939

# Builds the Enamine REAL lead-like (SurA) pool in this project's format:
# deduplicated, shuffled, row-aligned SMILES + scores. Run after
# submit_enamine_dedup_mask.sh, before any embedding extraction.
#
# The shuffle is not optional here. The downloaded file is sorted by score,
# best first -- verified over all 1,559,853,242 rows, zero violations -- so
# row position encodes the label. Anything correlated with position (a
# contiguous extraction batch, a "first N rows" subset, a shard boundary)
# would leak it. AmpC's and D4's own score files are in random order, which
# is why no earlier pool needed this step.
#
# Scale: 1,559,853,242 source rows -> 1,531,875,935 kept (27,977,307
# duplicates, 1.79%). Two passes over the data, both single-process:
#   scan   decompress 37.9 GB .gz, hash and bucket each kept row  (~115 GB read)
#   merge  sort each of 512 buckets in memory, write the outputs
# A plain zcat+parse pass over this file measured ~50 min, so budget several
# hours per stage; --time=24:00:00 is deliberate slack on a first run.
# Memory is one bucket (~300 MB of lines) plus the 6.1 GB uint32 permutation.
#
# Disk: --work-dir holds ~155 GB of bucket files, deletable afterwards.
# Outputs are ~65 GB (smiles) + ~15 GB (scores.csv.gz) + ~25 GB (ids).
#
# Usage:
#   sbatch slurm/embed/enamine1.56B/submit_enamine_build_pool.sh          # both stages
#   STAGE=scan  sbatch slurm/embed/enamine1.56B/submit_enamine_build_pool.sh
#   STAGE=merge sbatch slurm/embed/enamine1.56B/submit_enamine_build_pool.sh
# (splitting the stages lets a timed-out merge restart without redoing the scan)

set -euo pipefail

ER_ROOT="/N/project/SingleCell_Image/mengjing/enamine_real_1.56B"
SCORES="$ER_ROOT/hasten_docking_scores/Enamine_REAL_lead-like_SurA_glide_HTVS_docking_scores.csv.gz"
MASK="$ER_ROOT/enamine_dedup_keep_mask.npy"
WORK="$ER_ROOT/shuffle_work"
OUT="$ER_ROOT/dedup"
PREFIX="enamine_surA"
SEED=42

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
mkdir -p logs "$WORK" "$OUT"

# Stripe the output folder before anything is written there, so the large
# pool files land multi-server (report sect. 16: every dedup/*.npy was written
# single-stripe and had to be migrated afterwards, 0.8-18.7 h per file).
lfs setstripe -c -1 "$OUT" 2>/dev/null || true

echo "[pool] scores=$SCORES"
echo "[pool] mask=$MASK"
echo "[pool] work=$WORK  out=$OUT  prefix=$PREFIX  seed=$SEED  stage=${STAGE:-all}"
df -h "$ER_ROOT" | tail -1

python3 -u embed/stitch/build_pool_from_scores.py \
    --scores-file "$SCORES" \
    --keep-mask "$MASK" \
    --work-dir "$WORK" \
    --out-dir "$OUT" \
    --prefix "$PREFIX" \
    --num-buckets 512 \
    --seed "$SEED" \
    --stage "${STAGE:-all}"

echo "[pool] done"
ls -la "$OUT"
