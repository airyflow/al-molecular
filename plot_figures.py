#!/usr/bin/env python3
"""
Produce the two comparison figures: one for greedy acquisition, one for UCB.
Each figure has 3 panels (one per batch-size fraction: 0.4%, 0.2%, 0.1% of
the scored pool) and 3 traces per panel (MPN, MoLFormer-finetune, our
fusion model) -- mirrors the layout of the original MolPAL paper's Figure 4
(panels = batch size, traces = model), see notebooks/hts-figures.ipynb in
coleygroup/molpal for the reference implementation this follows.

Metric: fraction of the true top-1000 docking scores found vs. number of
molecules explored (same k=1000 as the original paper -- NOT top-1%, which
would be a different, larger k at this pool size).

Reads runs/<run_dir>/history.json written by run_experiment.py. Expects
run directories named by run_all_configs.sh's convention:
  {method}_{acq}_frac{frac}/history.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, ticker

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

METHODS = [
    ("mpn", "MolPAL + MPN"),
    ("molformer_ft", "MolPAL + MoLFormer (fine-tuned)"),
    ("fusion", "Our fusion model"),
]
FRACTIONS = [0.004, 0.002, 0.001]
COLORS = {"mpn": "#1b9e77", "molformer_ft": "#d95f02", "fusion": "#7570b3"}


def load_history(method: str, acq: str, frac: float):
    run_dir = RUNS_DIR / f"{method}_{acq}_frac{frac}"
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print(f"[skip] missing {hist_path}")
        return None
    return json.loads(hist_path.read_text())


def plot_acquisition_figure(acq: str, topk: int = 1000):
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(5 * len(FRACTIONS), 5), sharey=True)

    for ax, frac in zip(axes, FRACTIONS):
        for method, label in METHODS:
            history = load_history(method, acq, frac)
            if history is None:
                continue
            x = [r["n_labeled"] for r in history]
            y = [r["topk_recall"] for r in history]
            ax.plot(x, y, marker="o", color=COLORS[method], label=label, mec="k", mew=1)

        ax.set_title(f"batch = {frac:.1%} of pool")
        ax.set_xlabel("Molecules Explored")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(ticker.EngFormatter(sep=""))
        ax.grid(True, which="major", axis="both", ls="--", alpha=0.5)

    axes[0].set_ylabel(f"Fraction of top-{topk} Scores Found")
    axes[-1].legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle(f"Enamine HTS (2.1M) — {acq.upper()} acquisition", y=1.02)
    fig.tight_layout()

    out_path = FIG_DIR / f"enamine_hts_{acq}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    out_png = FIG_DIR / f"enamine_hts_{acq}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    print(f"[saved] {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=1000)
    args = p.parse_args()

    plot_acquisition_figure("greedy", topk=args.topk)
    plot_acquisition_figure("ucb", topk=args.topk)


if __name__ == "__main__":
    main()
