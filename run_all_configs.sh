#!/usr/bin/env bash
# Launches all 18 configs: 3 methods x 2 acquisitions x 3 batch-size fractions.
# Run directories are named {method}_{acq}_frac{frac} to match plot_figures.py.
#
# Pool size used for fraction->count conversion: 2,104,319 (the scored subset
# of EnamineHTS.csv.gz, i.e. len(load_oracle()) -- verify this matches your
# actual run's "[oracle] ... molecules" printout before trusting the numbers
# below; if it drifts, recompute N and the INIT/BATCH arrays.
set -euo pipefail
cd "$(dirname "$0")"

N=2104319
FRACS=(0.004 0.002 0.001)
ACQS=(greedy ucb)

frac_to_count() {
    python3 -c "print(round($N * $1))"
}

for frac in "${FRACS[@]}"; do
    count=$(frac_to_count "$frac")
    for acq in "${ACQS[@]}"; do
        echo "=== mpn  acq=$acq  frac=$frac  (init=batch=$count) ==="
        python run_experiment.py --mode molpal --model mpn --conf-method mve \
            --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
            --run-dir "runs/mpn_${acq}_frac${frac}"

        echo "=== molformer_ft  acq=$acq  frac=$frac  (init=batch=$count) ==="
        python run_experiment.py --mode mve --surrogate ft_molformer_single --backbones molformer \
            --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
            --run-dir "runs/molformer_ft_${acq}_frac${frac}"

        echo "=== fusion  acq=$acq  frac=$frac  (init=batch=$count) ==="
        python run_experiment.py --mode mve --surrogate ensemble --backbones grover molformer unimol \
            --acq "$acq" --init-size "$count" --batch-size "$count" --n-rounds 5 --topk 1000 \
            --run-dir "runs/fusion_${acq}_frac${frac}"
    done
done

echo "All 18 runs complete. Generating figures..."
python plot_figures.py
