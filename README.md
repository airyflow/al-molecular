# al-molecular

**Environment: `conda activate py310`** (not `al-eval`, which is missing
`torch-scatter`, `chemprop`/`pytorch_lightning`, and `ray[train]` -- it was
built for `al-eval-framework`'s UniMol-only pipeline, never GROVER or MPN).
`py310` is FusionAL's own environment and is the one this repo's GROVER,
MoLFormer, and MPN paths were actually fixed against this session.

Reproduces the experimental setup of MolPAL's Enamine HTS Collection result
(Graff, Shakhnovich & Coley, *Chem. Sci.* 2021, 12, 7866-7881, Figure 4,
page 7876; arXiv:2012.07127; code/data: github.com/coleygroup/molpal), and
extends it to compare three methods under both greedy and UCB acquisition
-- two comparisons the original paper's Figure 4 does not itself show (Fig.
4 is greedy-only; UCB numbers for this dataset exist only as final-epoch
values in the paper's Supporting Information Tables S3-S5, not as full
learning curves).

## The experiment

- **Pool**: `molpal/libraries/EnamineHTS.csv.gz`, 2,141,515 SMILES.
- **Oracle**: `data/EnamineHTS_scores.csv.gz`, 2,104,319 AutoDock Vina scores
  against thymidylate kinase (PDB 4UNN), lower = better. Sourced directly
  from `coleygroup/molpal`'s own data release -- **not redocked here**.
- **Metric**: fraction of the true top-1000 scores found (fixed k=1000, as
  used in the paper -- not top-1%, which would be a much larger k=~21,000
  at this pool size) vs. number of molecules explored.
- **Protocol**: init-size == batch-size, swept over {0.4%, 0.2%, 0.1%} of
  the scored pool (~8417 / 4209 / 2104 molecules), 5 exploration rounds --
  matching `examples/config/EnamineHTS_online.ini` (`max-iters = 5`,
  `top-k = 0.0005`) in the original repo.
- **Fingerprint** (MPN input): Atom-pair, 2048 bits, radius 2 (`pair`, not
  Morgan) -- matches the paper's stated encoder for all fingerprint models.

## Three methods compared

| Method | Flags | Notes |
|---|---|---|
| [1] MolPAL + MPN | `--mode molpal --model mpn --conf-method mve` | Chemprop D-MPNN with MVE uncertainty, as in the original paper |
| [2] MolPAL + MoLFormer (fine-tuned each round) | `--mode mve --surrogate ft_molformer_single --backbones molformer` | Frozen for the first 2 rounds, then fine-tuned on the accumulated labeled set + full-pool re-embedding each round thereafter (`SingleBackboneFinetuneScheduleSurrogate` in `surrogates.py`) |
| [3] Our fusion model | `--mode mve --surrogate ensemble --backbones grover molformer unimol` | `EnsembleFusionSurrogate`: three frozen backbones (GROVER, MoLFormer, UniMol) combined via inter-backbone rank-disagreement UCB |

**Why `ensemble`, not `bigfusion`, for method [3]**: `BigFusionSurrogate`
combines backbones via hard Borda-count ranking and always returns
`sigma=0`. Since `ucb(mu, var) = mu + beta*sqrt(var)`, zero variance makes
UCB acquisition identical to greedy acquisition for that surrogate
specifically -- the greedy and UCB figures would show an *identical* fusion
trace, which would undermine the comparison the two figures are meant to
make. `EnsembleFusionSurrogate` keeps the same three-backbone Borda-style
combination but uses rank disagreement across backbones as real epistemic
uncertainty (`UCB_score = -mean_rank + beta*std_rank`), so its greedy and
UCB curves can actually differ.

## Known bug fixed here (not yet fixed upstream in FusionAL)

`molpal/models/__init__.py`'s `mve()` factory derived the single-backbone
fine-tune target via `surrogate_type.replace("ft_", "")`, which turns
`"ft_molformer_single"` into `"molformer_single"` -- never a valid key in
`emb_dict`. Fixed here with an explicit mapping. Worth porting the same fix
back to `FusionAL/molpal/models/__init__.py` if that repo's `ft_molformer_single`
path is ever used there.

## Configuration

Environment-specific values (conda install location, SLURM account, and
data that lives outside the repo -- `AMPC_ROOT`, `ENHITS_LARGE_EMBED_DIR`)
are read from `config.env` at the repo root, not hardcoded. Copy the
template and edit it once per environment:
```bash
cp config.env.example config.env
# edit config.env: replace the user_name placeholders with your real paths
```
Every `slurm/*.sh` script sources this automatically; `run_experiment.py`
reads it directly too, so it works the same way whether you run through
SLURM or invoke it standalone. Leaving `config.env` unedited or absent
falls back to this repo's original values -- nothing is required to keep
working exactly as before. `#SBATCH` partition/log-path directives are
*not* covered by this file (see the comment in `config.env.example` for
why) -- edit those lines directly, or override at submit time with
`sbatch -p <partition> -o <path> script.sh`.

To confirm `config.env` is actually being picked up, without running
anything heavy:
```bash
python3 -c "
import run_experiment as exp
print('AMPC_ROOT:', exp.AMPC_ROOT)
print('RUNS_DIR:', exp.RUNS_DIR)
print('ENHITS embed_dir:', exp.DATASETS['ENHITS']['embed_dir'])
"
```
The printed paths should match whatever you put in `config.env`, not the
committed defaults.

## Setup

```bash
conda activate py310   # same environment as FusionAL

# 1. Extract frozen backbone embeddings for the full pool (needed for
#    method [2]'s starting point and method [3]'s three backbones)
python embed/extract_embeddings.py --backbone all
# -> results/embed/EnamineHTS/{grover,unimol,molformer}_embeddings.npz
#
# GROVER specifically may need the sharded BigRed path instead (see below)
# -- a single-process full-pool run on this host measured ~33min just for
# RDKit graph preprocessing and grew past 260GB RSS with the forward pass
# still not started (torch_geometric falls back to slow pure-Python graph
# ops here; pyg_lib/torch_sparse's accelerated kernels fail to load due to
# a GLIBC_2.29 mismatch). UniMol needs no extraction step at all: a
# pre-existing, verified-aligned conformer cache already covers the full
# pool (muben/data/files/EnamineHTS/processed/unimol-unimol/train.pt,
# 21.6GB) -- embed/extract_embeddings.py --backbone unimol will pick it up
# automatically. MoLFormer needs no sharding either (~8 min for the full
# pool, measured directly).
#
# Sharded GROVER path, if the single-process run is too slow/memory-hungry:
#   sbatch --array=0-49 slurm/embed/enamine/submit_grover_extraction.sh 50 2141500
#   python embed/stitch/concat_grover_chunks.py --chunks-dir results/embed/EnamineHTS/_grover_chunks \
#       --num-chunks 50 --total-count 2141500 --out-path results/embed/EnamineHTS/grover_embeddings.npz

# 2. Smoke-test on a small subsample before committing GPU time to the
#    full 2.1M-molecule sweep (see "Smoke test" below)

# 3. Run everything (3 methods x 2 acquisitions x 3 batch-size fractions
#    = 18 full AL runs against the 2.1M pool) and generate both figures
./slurm/al_runs/enamine/run_all_configs.sh
```

Each run writes to `runs/<method>_<acq>_frac<fraction>/`:
`history.json` (per-round metrics), `all_explored_final.csv`, and
per-round checkpoints. `plot_figures.py` reads `history.json` from all 18
directories and writes `figures/enamine_hts_greedy.{pdf,png}` and
`figures/enamine_hts_ucb.{pdf,png}`, each a 3-panel (one per batch
fraction) x 3-trace (one per method) figure matching Figure 4's layout.

## Smoke test (do this before the real run)

```bash
python embed/extract_embeddings.py --backbone all --limit 3000
python run_experiment.py --mode molpal --model mpn --acq greedy \
    --init-size 100 --batch-size 100 --n-rounds 2 --topk 20 --pool-limit 3000 \
    --run-dir runs/_smoke_mpn
python run_experiment.py --mode mve --surrogate ft_molformer_single --backbones molformer \
    --acq ucb --init-size 100 --batch-size 100 --n-rounds 2 --topk 20 --pool-limit 3000 \
    --run-dir runs/_smoke_molformer_ft
python run_experiment.py --mode mve --surrogate ensemble --backbones grover molformer unimol \
    --acq ucb --init-size 100 --batch-size 100 --n-rounds 2 --topk 20 --pool-limit 3000 \
    --run-dir runs/_smoke_fusion
```

To exercise the newer ENHITS/LT-All path end-to-end (config resolution +
dataset loading + the fused 5-backbone surrogate), on a tiny slice so it
finishes in under a minute regardless of GPU availability:
```bash
python run_experiment.py --dataset ENHITS --mode mve --surrogate ltall --acq greedy \
    --backbones molformer grover mhgged smited unimol2 \
    --init-size 50 --batch-size 50 --n-rounds 2 --topk 20 --pool-limit 500 \
    --run-dir runs/_smoke_config_test
```
A clean 2-round history table printed at the end means both `config.env`
and the LT-All pipeline are working together, not just the path
resolution in isolation.

## Repository layout

```
al-molecular/
├── data/EnamineHTS_scores.csv.gz        # real 4UNN docking scores (coleygroup/molpal)
├── molpal/libraries/EnamineHTS.csv.gz   # 2.1M-molecule pool
├── molpal/                              # vendored MolPAL package (models, acquirer, featurizer, ...)
├── surrogates.py                        # ALSU surrogate classes, incl. EnsembleFusionSurrogate
├── backbone_finetuner.py                # online backbone fine-tuning (GROVER/UniMol/MoLFormer)
├── embed/extract_embeddings.py          # one-time frozen-embedding extraction for the pool
├── run_experiment.py                    # single-config AL driver (top-1000 metric, EnamineHTS-only)
├── plot_figures.py                      # produces the two comparison figures
├── slurm/                               # all SLURM submit scripts, grouped by stage then dataset
│   ├── embed/{enamine,ampc}/            # embedding extraction + stitch jobs
│   └── al_runs/{enamine,ampc}/          # active-learning driver submissions (incl. run_all_configs.sh)
├── models -> FusionAL/models            # symlink: shared pretrained backbone checkpoints
├── muben  -> FusionAL/muben             # symlink: shared MUBen backbone library
├── results/embed/EnamineHTS/            # extracted frozen embeddings (.npz)
├── runs/                                # per-config AL run outputs
└── figures/                             # final figures
```
