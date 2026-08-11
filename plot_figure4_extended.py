#!/usr/bin/env python3
"""
Extends the paper's own Figure 4 (RF/NN/MPN, greedy acquisition) with our
four follow-up methods (MoLFormer fine-tuned in-loop, the frozen fusion
model, ft_fusion -- GROVER+MoLFormer jointly fine-tuned, UniMol frozen --
and learned_fusion -- 3 frozen backbones combined by a RidgeCV meta-learner
instead of Borda rank-sum) plotted on the same 3 panels -- all 7 methods,
one figure, directly comparable, rather than split across plot_figure4.py
and plot_figures.py.

Greedy acquisition only: RF/NN/ft_fusion/learned_fusion were never run
under UCB in this pipeline (RF/NN match the paper's own Figure 4, which is
greedy-only; ft_fusion was scoped to greedy-only given its cost; learned_fusion's
predict() returns zeros for sigma -- see surrogates.py -- so UCB would be
mathematically identical to greedy anyway), so a UCB version of this
combined figure isn't possible without running RF/NN/ft_fusion under UCB
first.

Reads runs/<run_dir>/history.json, expecting run_all_configs-style naming:
  {model}_greedy_frac{frac}/history.json
  for model in {rf, nn, mpn, molformer_ft, fusion, ftfusion, learned}
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

# Line style encodes provenance: dashed = the paper's own Figure 4 models,
# solid = the follow-up methods being compared against them. Color is unique
# per method regardless of group.
MODELS = [
    ("rf", "RF (paper)", "#e7298a", "--"),
    ("nn", "NN (paper)", "#66a61e", "--"),
    ("mpn", "MPN (paper)", "#1b9e77", "--"),
    ("molformer_ft", "MolPAL + MoLFormer (ours)", "#d95f02", "-"),
    ("fusion", "Our fusion model, frozen (ours)", "#7570b3", "-"),
    ("ftfusion", "ft_fusion: GROVER+MoLFormer fine-tuned (ours)", "#e6ab02", "-"),
    ("learned", "learned_fusion: RidgeCV meta-learner (ours)", "#666666", "-"),
]
FRACTIONS = [0.004, 0.002, 0.001]


def load_history(model: str, frac: float):
    run_dir = RUNS_DIR / f"{model}_greedy_frac{frac}"
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        print(f"[skip] missing {hist_path}")
        return None
    return json.loads(hist_path.read_text())


def plot_figure4_extended(topk: int = 1000):
    fig, axes = plt.subplots(1, len(FRACTIONS), figsize=(5.5 * len(FRACTIONS), 5.5), sharey=True)

    for ax, frac in zip(axes, FRACTIONS):
        for model, label, color, ls in MODELS:
            history = load_history(model, frac)
            if history is None:
                continue
            x = [r["n_labeled"] for r in history]
            y = [r["topk_recall"] for r in history]
            ax.plot(x, y, marker="o", color=color, ls=ls, label=label, mec="k", mew=1, lw=2)

        ax.set_title(f"batch = {frac:.1%} of pool")
        ax.set_xlabel("Molecules Explored")
        ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(ticker.EngFormatter(sep=""))
        ax.grid(True, which="major", axis="both", ls="--", alpha=0.5)

    axes[0].set_ylabel(f"Fraction of top-{topk} Scores Found")
    axes[-1].legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.suptitle(
        "Enamine HTS (2.1M) — Figure 4 (RF/NN/MPN) extended with MoLFormer-finetune, frozen fusion, ft_fusion, and learned_fusion, greedy acquisition",
        y=1.03,
    )
    fig.tight_layout()

    out_path = FIG_DIR / "figure4_extended.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    out_png = FIG_DIR / "figure4_extended.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    print(f"[saved] {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topk", type=int, default=1000)
    args = p.parse_args()
    plot_figure4_extended(topk=args.topk)


if __name__ == "__main__":
    main()
