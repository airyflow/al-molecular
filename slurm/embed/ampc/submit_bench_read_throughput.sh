#!/bin/bash

#SBATCH -J bench_read
#SBATCH -p general
#SBATCH -o logs/bench_read_%A_%a.txt
#SBATCH -e logs/bench_read_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=24G
#SBATCH -A r00939

# Read-only bandwidth probe on the AmpC GROVER3400 embeddings (1.34 TB,
# the 28-stripe copy in embeddings_striped/; the dedup/ copy had stripe count 1),
# same read pattern as pca_reduce_embeddings.py. Each task reads up to
# 20 GB of its own row range. Writes nothing but logs.
#
# Compare per-task MB/s between:
#   sbatch --array=0 slurm/embed/ampc/submit_bench_read_throughput.sh 1
#   sbatch --array=0-7 slurm/embed/ampc/submit_bench_read_throughput.sh 8
# Optional 2nd arg "buffered" uses plain seek+read instead of memmap.

set -euo pipefail
NUM_TASKS="${1:?Usage: sbatch --array=0-(N-1) slurm/embed/ampc/submit_bench_read_throughput.sh <N> [buffered]}"
MODE="${2:-}"

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

FLAG=""
[ "$MODE" = "buffered" ] && FLAG="--buffered"
python3 -u embed/stitch/bench_read_throughput.py \
    --path /N/project/SingleCell_Image/mengjing/ampc_99.5M/embeddings_striped/grover3400_embeddings.npy \
    --num-tasks "$NUM_TASKS" --max-gb 20 $FLAG
