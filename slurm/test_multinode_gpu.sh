#!/bin/bash

#SBATCH -J multinode_gpu_test
#SBATCH -p h100-multi
#SBATCH --gpus-per-node h100:1
#SBATCH -o logs/multinode_gpu_test_%j.txt
#SBATCH -e logs/multinode_gpu_test_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --mem=16G
#SBATCH -A r00939

# One-shot test: does this cluster actually grant >1 GPU-bearing node to a
# SINGLE job at once, or does --nodes=2 only get fulfilled for CPU-only
# partitions (the thing left unconfirmed before investing in any real
# multi-node DDP training work)? Each of the 2 tasks (one per node, via
# srun --ntasks=2 --ntasks-per-node=1) prints its own hostname and visible
# GPU count -- two DIFFERENT hostnames, each reporting >=1 GPU, is the only
# way this test actually passes. A single-hostname result, or a hang,
# means multi-node GPU scheduling isn't actually available here.
#
# Usage: sbatch slurm/test_multinode_gpu.sh

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

echo "[job] SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES SLURM_NODELIST=$SLURM_NODELIST"

srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -c '
    echo "[task $SLURM_PROCID] host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    python3 -c "import torch; print(f\"[task $SLURM_PROCID] torch sees {torch.cuda.device_count()} GPU(s): \" + \", \".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))"
'

echo "[job] done"
