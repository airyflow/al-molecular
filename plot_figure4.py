#!/usr/bin/env python3
"""
Reproduce the MolPAL paper's actual Figure 4 (Graff, Shakhnovich & Coley,
Chem. Sci. 2021, 12, 7866-7881, page 7876): RF/NN/MPN surrogate models under
GREEDY acquisition only, 3 panels (one per batch-size fraction: 0.4%, 0.2%,
0.1% of the scored EnamineHTS pool), same top-1000-recall-vs-molecules-
explored metric as our other figures. Unlike plot_figures.py's two new
comparison figures (our 3 methods x greedy/UCB), this is a literal
reproduction of the paper's own experimental grid -- greedy only, since
Figure 4 itself never plots UCB (see README.md's provenance note).

Reads runs/<run_dir>/history.json, expecting run_all_configs-style naming:
  {model}_greedy_frac{frac}/history.json  for model in {rf, nn, mpn}
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

MODELS = [
    ("rf", "RF"),
    ("nn", "NN"),
    ("mpn", "MPN"),
]
FRACTIONS = [0.004, 0.002, 0.001]
COLORS = {"rf": "#e7298a", "nn": "#66a61e", "mpn": "#1b9e77"}


def load_history(model: str, frac: float):
    run_dir = RUNS_DIR / f"{model}_greedy_frac{frac}"
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print(f"[skip] missing {hist_path}")
        return None
    return json.loads(hist_path.read_text())


def plot_figure4(topk: int = 1000):
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(5 * len(FRACTIONS), 5), sharey=True)

    for ax, frac in zip(axes, FRACTIONS):
        for model, label in MODELS:
            history = load_history(model, frac)
            if history is None:
                continue
            x = [r["n_labeled"] for r in history]
            y = [r["topk_recall"] for r in history]
            ax.plot(x, y, marker="o", color=COLORS[model], label=label, mec="k", mew=1)

        ax.set_title(f"batch = {frac:.1%} of pool")
        ax.set_xlabel("Molecules Explored")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(ticker.EngFormatter(sep=""))
        ax.grid(True, which="major", axis="both", ls="--", alpha=0.5)

    axes[0].set_ylabel(f"Fraction of top-{topk} Scores Found")
    axes[-1].legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("Enamine HTS (2.1M) — Figure 4 reproduction (greedy acquisition)", y=1.02)
    fig.tight_layout()

    out_path = FIG_DIR / "figure4_reproduction.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    out_png = FIG_DIR / "figure4_reproduction.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    print(f"[saved] {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=1000)
    args = p.parse_args()
    plot_figure4(topk=args.topk)


if __name__ == "__main__":
    main()
