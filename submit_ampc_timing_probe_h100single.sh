#!/bin/bash

#SBATCH -J ampc_timing_probe
#SBATCH --mail-user=mengjing@iu.edu
#SBATCH --mail-type=ALL
#SBATCH -p h100-single
#SBATCH --gpus-per-node h100:1
#SBATCH -o /N/slate/mengjing/repos/al-molecular/logs/ampc_timing_probe_%j.txt
#SBATCH -e /N/slate/mengjing/repos/al-molecular/logs/ampc_timing_probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --mem=96G
#SBATCH -A r00939

# THROWAWAY diagnostic, not part of the real fusion probe -- answers "how
# long does EnsembleFusionSurrogate training actually take per labeled
# molecule?" with a real measurement instead of a guess.
#
# v2: first attempt (job 9956935, --surrogate-epochs unset -> default 50)
# did NOT finish within its 2h budget even at only 5,000 labeled molecules
# (~2,930 total steps -- ~20x fewer than round 1 of the real run) -- oracle
# +embeddings+init all completed, so the multi-hour stall was somewhere
# inside training itself, far beyond what per-step Python/CUDA dispatch
# overhead alone should cost. --surrogate-epochs 2 (new flag, threaded
# through run_experiment.py -> EmbeddingMVEModel.train() ->
# EnsembleFusionSurrogate.fit(), which now also prints real per-backbone
# wall time) cuts the step count to ~4% of that unfinished attempt, and
# --time=0:30:00 is deliberately tight: if THIS also fails to finish,
# that's strong evidence the real bottleneck is the load_oracle()/
# load_embeddings() stage (Lustre contention from many concurrent h100-
# single jobs hitting the same huge files), not training step count.
#
# --init-size/--batch-size 5000, --n-rounds 1: trains on a KNOWN, small
# labeled-set size so elapsed time / (5000/256 steps/epoch * 2 epochs * 3
# backbones) gives a real per-step rate, precisely extrapolatable to the
# real run's round sizes (99,460 / 198,920 / 298,380 / 397,840 / 497,300)
# at whatever --surrogate-epochs value is eventually chosen for real.
#
# --pool-limit 200000: keeps prediction cheap enough to run sequentially,
# in-process (no --parallel-predict, no worker pool needed for a
# throwaway 1-round timing probe) -- prediction isn't what we're
# measuring here, training is.
#
# --topk kept small (50) since it only affects a recall metric printout,
# not timing.
#
# Usage:
#   sbatch /N/slate/mengjing/repos/al-molecular/submit_ampc_timing_probe_h100single.sh

set -euo pipefail

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate py310

cd /N/slate/mengjing/repos/al-molecular
mkdir -p logs

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "[timing probe] started $(date)"

srun --cpu-bind=none python3 -u run_experiment.py --dataset AmpC --mode mve --surrogate ensemble \
    --backbones grover molformer unimol \
    --acq greedy --init-size 5000 --batch-size 5000 --n-rounds 1 --topk 50 \
    --pool-limit 200000 --surrogate-epochs 2 \
    --run-dir "runs/ampc_timing_probe_v2"

echo "[timing probe] finished $(date)"
