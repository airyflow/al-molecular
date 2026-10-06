#!/usr/bin/env python3
"""
run_experiment.py -- active learning driver for the MolPAL Enamine HTS
(2.1M-compound, thymidylate kinase / PDB 4UNN) reproduction + comparison.

Adapted from FusionAL/run_al.py, narrowed to a single dataset (EnamineHTS)
and the exact evaluation protocol used in the original MolPAL paper's
Figure 4 / notebooks/hts-figures.ipynb (coleygroup/molpal):
  - metric: fraction of the TRUE top-k (k=1000, not top-1%) docking scores
    found, vs. number of molecules explored (labeled)
  - 5 exploration rounds
  - init-size == batch-size, swept over {0.4%, 0.2%, 0.1%} of the scored pool

Three methods are compared here (see README.md for the full rationale):
  [1] MolPAL + MPN            --mode molpal --model mpn --conf-method mve
  [2] MolPAL + MoLFormer,      --mode mve --surrogate ft_molformer_single
      fine-tuned each AL round      --backbones molformer
  [3] "our fusion model"       --mode mve --surrogate ensemble
      (EnsembleFusionSurrogate)     --backbones grover molformer unimol
      -- NOT bigfusion: bigfusion's Borda combination returns sigma=0,
      which makes UCB acquisition collapse to greedy for it specifically.
      EnsembleFusionSurrogate uses inter-backbone rank disagreement as a
      real (non-degenerate) uncertainty signal instead.

Each of these is run under both --acq greedy and --acq ucb, at all three
batch-size fractions -- orchestrated by slurm/al_runs/enamine/run_all_configs.sh, not this file
directly (this file runs exactly one config per invocation).
"""

import argparse
import json
import mmap
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from smiles_chunking import _chunk_bounds

ROOT = Path(__file__).resolve().parent


def _load_config_env(path: Path) -> None:
    """Populate os.environ from a plain KEY=value file, without overriding
    anything already set (so real env vars -- e.g. exported by a
    slurm/*.sh script that already sourced this same file -- win over it).
    No-op if the file doesn't exist, so this repo behaves exactly as
    before for anyone who hasn't created config.env -- see
    config.env.example for the variables this reads."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if value.strip():
            os.environ.setdefault(key.strip(), value.strip())


_load_config_env(ROOT / "config.env")

EMBED_DIR = ROOT / "results" / "embed"
DATA_DIR = ROOT / "data"
LIBRARY_DIR = ROOT / "molpal" / "libraries"
RUNS_DIR = Path(os.environ.get("RUNS_DIR") or (ROOT / "runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

AMPC_ROOT = Path(os.environ.get("AMPC_ROOT", "/N/project/SingleCell_Image/mengjing/ampc_99.5M"))
D4_ROOT = Path(os.environ.get("D4_ROOT", "/N/project/SingleCell_Image/mengjing/d4_138M"))

# Registry so a second, much larger dataset (AmpC, 99.5M molecules, Figure 5)
# can share this same driver without EnamineHTS's paths (all under this
# Slate-based repo) being hardcoded throughout. AmpC's data/embeddings live
# on project storage instead (~1.9TB of embeddings alone -- far past what's
# reasonable for the 800GB personal Slate quota). "library" may be a plain
# .txt (one SMILES/line, AmpC) or a .csv.gz with a smiles column
# (EnamineHTS) -- load_library_smiles() dispatches on suffix.
DATASETS = {
    "EnamineHTS": {
        "library": LIBRARY_DIR / "EnamineHTS.csv.gz",
        "oracle": DATA_DIR / "EnamineHTS_scores.csv.gz",
        "embed_dir": EMBED_DIR / "EnamineHTS",
    },
    "ENHITS": {
        # Enamine HTS re-embedded by Yang's sharded pipeline
        # (EnamineHITS_large_embedding_shards): the 2,104,318 *scored*
        # molecules only, row-aligned to data/EnamineHTS_scores.csv.gz by
        # construction. 5 backbones stitched from per-shard .npy via
        # stitch_enhits_large_shards.py -- dims here differ from the
        # results/embed/EnamineHTS set (grover=3400, unimol2=768, ...).
        # Bare (N, D) .npy per backbone + a plain smiles txt, so
        # load_embeddings() recovers row-order SMILES from smiles_source
        # (= "library" below), same fallback path AmpC uses.
        "library": DATA_DIR / "EnamineHTS_scores.csv.gz",
        "oracle": DATA_DIR / "EnamineHTS_scores.csv.gz",
        "embed_dir": Path(os.environ.get(
            "ENHITS_LARGE_EMBED_DIR", "/N/project/SingleCell_Image/mengjing/enhits_large/embed"
        )),
    },
    "ENHITS_shuffled": {
        # Same 2,104,318 molecules/scores as ENHITS, but embeddings +
        # SMILES rows are permuted by a fixed random shuffle
        # (embed/stitch/build_shuffle_permutation.py +
        # embed/stitch/shuffle_embeddings.py) -- built to test whether
        # ENHITS's paper-matching result (report Section 10) depends on
        # its embeddings having been extracted directly from
        # EnamineHTS_scores.csv.gz's own row order, which is sorted by
        # docking score (verified directly, 2026-09-28) -- a plausible
        # leakage path if any backbone's extraction uses batch-level
        # statistics over contiguous file chunks. "oracle" is untouched
        # (SMILES-keyed, so row order never matters for it); "library"
        # MUST be the shuffled SMILES file, not the original -- it's used
        # as EmbeddingFeaturizer's smiles_source fallback for row
        # alignment, and the embeddings here are in shuffled order.
        "library": Path(os.environ.get(
            "ENHITS_SHUFFLED_EMBED_DIR", "/N/project/SingleCell_Image/mengjing/enhits_large/embed_shuffled"
        )) / "enhits_large_smiles.txt",
        "oracle": DATA_DIR / "EnamineHTS_scores.csv.gz",
        "embed_dir": Path(os.environ.get(
            "ENHITS_SHUFFLED_EMBED_DIR", "/N/project/SingleCell_Image/mengjing/enhits_large/embed_shuffled"
        )),
    },
    "AmpC": {
        "library": AMPC_ROOT / "ampc_smiles.txt",
        "oracle": AMPC_ROOT / "ampc_scores.csv.gz",
        # Lustre-striped (28-way, vs. the original's stripe count 1) copy of
        # the same data -- re-stitched from the same per-chunk files via
        # slurm/embed/ampc/submit_ampc_stitch_embeddings_striped.sh, spot-verified bit-identical
        # to the original embeddings/ (shape, sample rows, NaN counts all
        # matched, 2026-08-15). Switched to fix scattered fancy-indexed reads
        # (AL training-set fetches, shard 7's per-chunk correction reads)
        # funneling through a single OST -- confirmed via nvidia-smi showing
        # 0% GPU utilization during a 2+ hour stall that this doesn't fix the
        # cause of, just spreads the same access pattern across 28 OSTs
        # instead of 1. The original embeddings/ is left on disk, untouched.
        "embed_dir": AMPC_ROOT / "embeddings_striped",
    },
    "AmpC_dedup": {
        # Deduplicated copy of AmpC's pool: dropped 970,211/99,459,561 rows
        # whose SMILES also occur elsewhere in the pool (last-occurrence
        # kept, matching load_oracle()/smi2idx's own last-write-wins
        # semantics -- see embed/stitch/build_dedup_mask.py). Built after
        # discovering ~957,702 of those duplicates (98.7% of all of them)
        # are concentrated in shard 7 alone (7.7% of its rows), which is
        # what was actually driving shard 7's chronic worker hangs, not a
        # node/hardware issue -- see molpal/models/mvemodels.py's
        # EmbeddingMVEModel._get_X() for the (separately fixed) indexing
        # bug that made those duplicate-heavy chunks slow. Every one of the
        # 98,489,350 kept rows verified bit-identical to its corresponding
        # row in the original embeddings_striped/ set (2026-09-11).
        "library": AMPC_ROOT / "dedup" / "ampc_smiles.txt",
        "oracle": AMPC_ROOT / "dedup" / "ampc_scores.csv.gz",
        "embed_dir": AMPC_ROOT / "dedup",
    },
    "AmpC_dedup_pca": {
        # Same pool/library/oracle as AmpC_dedup (identical molecules, same
        # row order, same docking scores) -- only embed_dir differs. Each
        # backbone here is a per-backbone PCA reduction of AmpC_dedup's own
        # embeddings (embed/stitch/pca_reduce_embeddings.py: fit on a
        # 1M-row random sample, 95% variance retained, transform the full
        # pool), NOT a re-extraction -- so this is directly comparable to
        # AmpC_dedup's own LT-All results at the same frac/acquisition,
        # isolating the effect of the dimensionality reduction itself.
        # Per-backbone reduced widths (95% variance): mhgged 1024->88,
        # molformer 768->271, smited 768->148, unimol 512->114; unimol2 and
        # grover3400 pending as of 2026-09-17 (see logs/ampc_pca_reduce_*).
        "library": AMPC_ROOT / "dedup" / "ampc_smiles.txt",
        "oracle": AMPC_ROOT / "dedup" / "ampc_scores.csv.gz",
        "embed_dir": AMPC_ROOT / "dedup_pca",
    },
    "D4_dedup": {
        # Deduplicated D4 pool, AmpC_dedup's sibling target. Raw d4.csv
        # (138,312,677 rows, from Yang's Figure-5-source Figshare release,
        # /N/project/SingleCell_Image/Yang/AI Drug/LargeData/D4/) already
        # had ~22M rows with an empty dockscore field; d4_enamine_style.csv
        # is that same release pre-filtered to the 116,241,184 scored-only
        # rows (verified: header "smiles,score", row i matches d4_smiles.txt
        # line i exactly). Deduping (embed/stitch/build_dedup_mask.py, same
        # last-occurrence-kept rule as AmpC's own mask) dropped 15,257/
        # 116,241,184 duplicate rows -- a far smaller fraction than AmpC's
        # 970,211/99,459,561, so no equivalent of AmpC's shard-7 concentration
        # issue is expected here.
        # embed_dir is populated per-backbone by embed/stitch/dedup_embeddings.py
        # once that backbone's raw extraction + stitch
        # (slurm/embed/d4/submit_d4_*_extract.sh, submit_d4_stitch_embeddings.sh)
        # finishes -- not yet populated as of 2026-09-22.
        "library": D4_ROOT / "dedup" / "d4_smiles.txt",
        "oracle": D4_ROOT / "dedup" / "d4_scores.csv.gz",
        "embed_dir": D4_ROOT / "dedup",
    },
}

DATASET = "EnamineHTS"  # overridden by --dataset in main()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_embeddings(backbones: list) -> tuple:
    from molpal.featurizer import EmbeddingFeaturizer
    ef = EmbeddingFeaturizer(
        embed_dir=str(DATASETS[DATASET]["embed_dir"]),
        backbones=backbones,
        # AmpC's sharded extraction pipeline (al-eval-framework) writes bare
        # (N, D) .npy embeddings with no per-backbone smiles side-channel --
        # this fallback lets EmbeddingFeaturizer recover row-order smiles
        # from the same canonical library file the extraction pipeline used.
        # A no-op for EnamineHTS, whose .npz files already carry their own
        # smiles array and never reach this fallback path.
        smiles_source=str(DATASETS[DATASET]["library"]),
    )
    return ef.load()


def load_oracle() -> dict:
    """{smiles: score} oracle, lower = better (minimize) for both datasets.
    EnamineHTS: thymidylate-kinase (4UNN) AutoDock Vina scores, sourced from
    coleygroup/molpal's data/EnamineHTS_scores.csv.gz -- not redocked here.
    AmpC: AmpC beta-lactamase (12LS) DOCK3.7 scores, sourced from Balius et
    al.'s Figshare release (AmpC_screen_table.csv.gz) -- not redocked here.
    """
    gz = DATASETS[DATASET]["oracle"]
    assert gz.exists(), f"Oracle not found: {gz}"
    df = pd.read_csv(gz)
    df.columns = df.columns.str.strip().str.lower()
    smi_col = next(c for c in df.columns if "smiles" in c)
    score_col = next(c for c in df.columns if "score" in c)

    # AmpC's score column carries a literal "no_score" sentinel for molecules
    # DOCK3.7 failed/skipped to dock (~3.25M/99.5M, ~3.3%) -- pandas reads the
    # whole column as object dtype because of it, so even the real numeric
    # entries come through as Python str, not float. Coerce to numeric and
    # drop anything that doesn't parse (silently leaving these in as strings
    # would make them look like real oracle entries downstream, not just
    # crash the range print here).
    n_before = len(df)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    n_unscored = df[score_col].isna().sum()
    if n_unscored:
        df = df.dropna(subset=[score_col])
        print(f"[oracle] dropped {n_unscored:,}/{n_before:,} rows with a non-numeric/missing {score_col!r} "
              f"(e.g. AmpC's \"no_score\" sentinel for un-docked molecules)")

    oracle = dict(zip(df[smi_col], df[score_col]))
    print(f"[oracle] {gz.name}: {len(oracle):,} molecules  range [{df[score_col].min():.2f}, {df[score_col].max():.2f}]")
    return oracle


def load_library_smiles(limit: int = None) -> list:
    lib = DATASETS[DATASET]["library"]
    if lib.suffix == ".txt":
        with open(lib) as f:
            smis = [line.rstrip("\n") for line in (f.readlines()[:limit] if limit else f)]
        return smis
    df = pd.read_csv(lib)
    df.columns = df.columns.str.strip().str.lower()
    smi_col = next(c for c in df.columns if "smiles" in c)
    smis = df[smi_col].dropna().tolist()
    return smis[:limit] if limit is not None else smis


class DictObjective:
    def __init__(self, oracle: dict, minimize: bool = True):
        self.oracle = oracle
        self.c = -1.0 if minimize else 1.0

    def __call__(self, smis):
        return {s: self.c * self.oracle[s] if s in self.oracle else None for s in smis}

    @property
    def path(self):
        return None


POOL_PREDICT_CHUNK = 50_000


def _chunked_get_means_and_vars(model, pool_smi: list) -> tuple:
    """Call model.get_means_and_vars() in bounded-size chunks over the pool,
    instead of passing the whole remaining pool (millions of molecules for
    the fusion/molformer-finetune surrogates) in one call.

    EmbeddingMVEModel._get_X() does idxs = [smi2idx[s] for s in xs]; parts =
    [emb[idxs] for emb in emb_dict.values()]; np.concatenate(parts, axis=1)
    -- each of those builds a new near-full-size array when xs is most of
    the pool. For fusion mode (3 backbones concatenated) that's ~24GB of
    original arrays plus another ~24GB of fancy-indexed copies plus another
    ~24GB for the concatenated result, all before surrogates.py's own
    DataLoader-based batching (which does work correctly) ever gets
    involved -- confirmed as the actual cause of the fusion-mode OOMs
    (separate from the earlier node-sharing OOM). Chunking here bounds each
    _get_X() call to POOL_PREDICT_CHUNK molecules regardless of surrogate.
    """
    mu_chunks, var_chunks = [], []
    for start in range(0, len(pool_smi), POOL_PREDICT_CHUNK):
        chunk = pool_smi[start : start + POOL_PREDICT_CHUNK]
        mu, var = model.get_means_and_vars(chunk)
        mu_chunks.append(mu)
        var_chunks.append(var)
    return np.concatenate(mu_chunks), np.concatenate(var_chunks)


# Self-featurizing model types (get_means/get_means_and_vars take raw SMILES
# and featurize internally) -- matches molpal/models/base.py's Model.apply(),
# the original framework's own dispatch convention that MolPALExplorer's
# simplified reimplementation had dropped. Everything else (rf, nn, gp, lgbm)
# expects pre-featurized (N, D) arrays.
SELF_FEATURIZING_TYPES = {"mpn", "transformer", "molclr"}


def _chunked_predict_molpal(model, featurizer, pool_smi: list, needs_var: bool):
    """Predict over the pool in bounded chunks, featurizing each chunk only
    (not the whole pool at once) for models that need pre-featurized input.

    RFModel/NNModelTorch's get_means()/get_means_and_vars() expect an (N, D)
    array, not raw SMILES -- featurizing the full ~2.09M-molecule remaining
    pool in one call before chunking would recreate the same class of memory
    blowup already fixed for MPN/fusion, just one step earlier (feature
    extraction instead of dataset construction). Self-featurizing models
    (mpn/transformer/molclr) skip the featurizer entirely here since their
    own get_means_and_vars() already chunks internally (see MPNN.predict()).
    """
    if model.type_ in SELF_FEATURIZING_TYPES:
        if needs_var:
            return _chunked_get_means_and_vars(model, pool_smi)
        return model.get_means(pool_smi), None

    from molpal.featurizer import feature_matrix

    mu_chunks, var_chunks = [], []
    for start in range(0, len(pool_smi), POOL_PREDICT_CHUNK):
        chunk_smi = pool_smi[start : start + POOL_PREDICT_CHUNK]
        X_chunk = np.array(feature_matrix(chunk_smi, featurizer))
        if needs_var:
            mu, var = model.get_means_and_vars(X_chunk)
            var_chunks.append(var)
        else:
            mu = model.get_means(X_chunk)
        mu_chunks.append(mu)

    mu = np.concatenate(mu_chunks)
    var = np.concatenate(var_chunks) if needs_var else None
    return mu, var


# ==============================================================================
# SHARED: true top-k lookup (fixed k, matching the paper's metric -- NOT top-1%)
# ==============================================================================

def true_top_k_set(oracle: dict, k: int) -> set:
    return {s for s, _ in sorted(oracle.items(), key=lambda x: x[1])[:k]}


def find_resume_checkpoint(run_dir: Path) -> tuple:
    """Finds the highest-numbered run_dir/iter_N with both state.json and
    scores.pkl (a round is only fully checkpointed once both exist --
    _checkpoint() writes them together). Returns (round_num, labeled_scores)
    or (0, None) if no complete checkpoint exists (fresh-init fallback)."""
    if not run_dir.exists():
        return 0, None
    candidates = []
    for d in run_dir.glob("iter_*"):
        if (d / "state.json").exists() and (d / "scores.pkl").exists():
            try:
                candidates.append(int(d.name.removeprefix("iter_")))
            except ValueError:
                continue
    if not candidates:
        return 0, None
    best_round = max(candidates)
    with open(run_dir / f"iter_{best_round}" / "scores.pkl", "rb") as f:
        labeled_scores = pickle.load(f)
    return best_round, labeled_scores


# ==============================================================================
# MVE ACTIVE LEARNING LOOP (embedding-based surrogates)
# ==============================================================================

class MVEExplorer:
    def __init__(
        self, emb_dict, pool_smiles, oracle, surrogate_type, backbone, acq,
        init_size=8417, batch_size=8417, n_rounds=5, topk=1000,
        run_dir=None, seed=42, usable_mask=None,
        surrogate_epochs=None, surrogate_batch=None,
        resume_scores=None, resume_round=0,
    ):
        from molpal.models import mve as build_mve
        from molpal.acquirer.metrics import get_metric

        self.pool_smiles = np.array(pool_smiles)
        self.oracle = oracle
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.topk = topk
        self.run_dir = run_dir or RUNS_DIR / "mve_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sign = -1.0
        # None means "use the surrogate class's own default" (currently
        # epochs=50, batch=256) -- only overridden by --surrogate-epochs/
        # --surrogate-batch. Preserves exact historical behavior (including
        # published EnamineHTS results) when not explicitly set.
        self.surrogate_epochs = surrogate_epochs
        self.surrogate_batch = surrogate_batch
        # None means "pool_smiles is already pre-filtered to oracle-scored
        # molecules only" (the historical behavior every non-parallel run,
        # including all published EnamineHTS results, relies on -- changing
        # this default would silently shift which indices rng.choice draws).
        # Only ParallelMVEExplorer's AmpC-scale path sets a real mask, to
        # avoid pre-filtering (see main()'s comment on why).
        self.usable_mask = usable_mask

        self.model = build_mve(
            surrogate_type=surrogate_type, backbone=backbone, emb_dict=emb_dict,
            pool_smiles=pool_smiles, dataset_name=DATASET,
        )

        self.acq_fn = get_metric(acq)
        self.acq_name = acq
        self.true_top_k = true_top_k_set(oracle, topk)
        self._start_round = resume_round
        self._resumed_history = []

        if resume_scores is not None:
            # Recovering from a crashed run (e.g. the orchestrator OOM-killed
            # partway into a later round -- see slurm/al_runs/ampc/submit_ampc_fusion_runs_h100single.sh)
            # using the iter_N/scores.pkl checkpoint _checkpoint() already
            # writes every round. Reconstruct labeled_idx via the pool's
            # smi2idx (already built by build_mve above). AmpC has some
            # within-pool duplicate SMILES (confirmed earlier: usable_mask's
            # population exceeds the oracle's unique-molecule count) --
            # smi2idx's last-write-wins means a labeled duplicate could
            # reconstruct to a DIFFERENT position than the one originally
            # selected. Harmless for training/prediction (same molecule, same
            # embedding either way); the only edge case is that duplicate
            # position could theoretically be re-offered in a later round --
            # negligible given how rare duplicates are relative to pool size.
            self.labeled_scores = dict(resume_scores)
            self.labeled_idx = {self.model.smi2idx[s] for s in self.labeled_scores if s in self.model.smi2idx}
            for r in range(1, resume_round + 1):
                state_path = self.run_dir / f"iter_{r}" / "state.json"
                if state_path.exists():
                    self._resumed_history.append(json.loads(state_path.read_text()))
            print(f"[resume] loaded {len(self.labeled_idx):,} labeled molecules from "
                  f"{self.run_dir}/iter_{resume_round}  best={self._best():.3f} kcal/mol  "
                  f"resuming at round {resume_round + 1}/{n_rounds}")
        else:
            rng = np.random.default_rng(seed)
            candidate_idx = np.where(self.usable_mask)[0] if self.usable_mask is not None else len(self.pool_smiles)
            init_idx = rng.choice(candidate_idx, init_size, replace=False)
            self.labeled_idx = set(init_idx.tolist())
            self.labeled_scores = {self.pool_smiles[i]: oracle[self.pool_smiles[i]] for i in init_idx}
            print(f"[init] {init_size} random molecules  best={self._best():.3f} kcal/mol")

    def _best(self) -> float:
        return min(self.labeled_scores.values())

    def _recall(self) -> float:
        return sum(1 for s in self.labeled_scores if s in self.true_top_k) / self.topk

    def run(self) -> list:
        history = list(self._resumed_history)
        n = len(self.pool_smiles)

        for rnd in range(self._start_round, self.n_rounds):
            t0 = time.perf_counter()

            idx = list(self.labeled_idx)
            xs = [self.pool_smiles[i] for i in idx]
            ys = self._sign * np.array([self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32)

            self.model.train(xs, ys, epochs=self.surrogate_epochs, batch=self.surrogate_batch)

            mask = np.ones(n, bool)
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            pool_smi = [self.pool_smiles[i] for i in pool_idx]

            mu, var = _chunked_get_means_and_vars(self.model, pool_smi)

            if self.acq_name in ("ucb", "lcb", "thompson", "ts", "ei", "pi"):
                scores = self.acq_fn(mu, var)
            else:
                scores = self.acq_fn(mu)

            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            recall = self._recall()
            elapsed = time.perf_counter() - t0
            print(f"  Round {rnd+1:02d}/{self.n_rounds}  labeled={len(self.labeled_idx):,}  "
                  f"best={self._best():.3f} kcal/mol  top-{self.topk} recall={recall:.1%}  ({elapsed:.1f}s)")

            record = dict(round=rnd + 1, n_labeled=len(self.labeled_idx),
                          best_score=float(self._best()), topk_recall=float(recall), elapsed=round(elapsed, 2))
            history.append(record)
            self._checkpoint(rnd + 1, record)

        self._save_final(history)
        return history

    def _checkpoint(self, rnd, record):
        d = self.run_dir / f"iter_{rnd}"
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps(record, indent=2))
        with open(d / "scores.pkl", "wb") as f:
            pickle.dump(dict(self.labeled_scores), f)

    def _save_final(self, history):
        pd.DataFrame(sorted(self.labeled_scores.items(), key=lambda x: x[1]), columns=["smiles", "score"]) \
            .to_csv(self.run_dir / "all_explored_final.csv", index=False)
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\n[done] results -> {self.run_dir}")


class ParallelMVEExplorer(MVEExplorer):
    """Same AL loop as MVEExplorer, but the per-round pool-prediction step is
    delegated to a persistent pool of single-GPU worker processes
    (predict_pool_shard_worker.py) coordinated via marker files under
    `coord_dir`, instead of predicting in-process.

    Rationale: at AmpC scale (99.5M molecules) pool prediction, not surrogate
    training, is the AL loop's bottleneck -- the surrogate is a small MLP
    trained on the labeled set only (thousands of molecules, cheap on one
    GPU), so training stays in-process here, unchanged from MVEExplorer.
    Splitting *prediction* across N independent single-GPU SLURM jobs is
    the actual win. See predict_pool_shard_worker.py's module docstring for
    the full protocol.

    Workers must already be running (submitted ONCE for the whole run, not
    resubmitted per round -- avoids paying SLURM queue-wait latency
    n_rounds times over) and waiting on round 1 before .run() is called --
    this class only ever writes ready markers and waits for done markers,
    it never launches or manages worker processes itself.
    """

    def __init__(self, *args, num_shards: int, coord_dir, poll_interval: float = 5.0,
                 exclude_shard_ids: frozenset = frozenset(), **kwargs):
        # Permanently drop a shard from future consideration -- e.g. one
        # whose embedding data lives in a part of a single-OST, unstriped
        # file (see EmbeddingMVEModel._get_X()'s docstring) that keeps
        # producing hung scattered reads even after the mostly-contiguous
        # fetch optimization (measured repeatedly on AmpC's shard 7, 2026-08).
        # Molecules in the excluded range are masked out of future
        # acquisition; any that happen to already be labeled (from before
        # exclusion, e.g. via --resume) stay correctly labeled and
        # trained-on -- only FUTURE candidacy is affected, not past labels.
        #
        # MUST happen BEFORE super().__init__() runs, not after: the parent
        # MVEExplorer.__init__ performs the fresh-init random rng.choice()
        # draw itself (when not resuming), consulting self.usable_mask AT
        # THAT POINT. Patching usable_mask afterward is too late for that
        # draw -- confirmed by a local test (verify_exclude_shard.py) that
        # failed with 2/10 labeled molecules landing inside the "excluded"
        # shard 1 range when the mask was applied post-hoc. Intercepting
        # kwargs["usable_mask"] here, before delegating to super(), fixes
        # both the fresh-init draw and every later round's acquisition in
        # one place. (For an actual --resume run this ordering is moot --
        # resume reconstructs labeled_idx directly from checkpoint data and
        # never calls rng.choice -- but fixing it here makes the flag
        # correct for fresh runs too, not just the one scenario we needed.)
        exclude_shard_ids = frozenset(exclude_shard_ids)
        if exclude_shard_ids:
            assert "pool_smiles" in kwargs, (
                "exclude_shard_ids requires pool_smiles to be passed as a keyword "
                "argument (as main() does) so the exclusion mask can be computed "
                "before delegating to MVEExplorer.__init__"
            )
            n_total = len(kwargs["pool_smiles"])
            excl_mask = np.zeros(n_total, dtype=bool)
            for sid in exclude_shard_ids:
                s, e = _chunk_bounds(n_total, sid, num_shards)
                excl_mask[s:e] = True
            base_mask = kwargs.get("usable_mask")
            base_mask = base_mask if base_mask is not None else np.ones(n_total, dtype=bool)
            kwargs["usable_mask"] = base_mask & ~excl_mask
            print(f"[exclude-shards] dropping shard(s) {sorted(exclude_shard_ids)} "
                  f"({excl_mask.sum():,} molecules) from candidate selection "
                  f"(applied before initial random draw)")
        super().__init__(*args, **kwargs)
        self.num_shards = num_shards
        self.coord_dir = Path(coord_dir)
        self.coord_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.exclude_shard_ids = exclude_shard_ids
        # Cache of already-fetched labeled-row embeddings (global pool index
        # -> concatenated-across-backbones row). self.model.emb_dict is the
        # FULL, unfiltered pool array (see main()'s --parallel-predict
        # comment -- kept memmapped/untouched specifically to avoid the
        # ~1.1TB eager-materialization OOM), so fancy-indexing into it is a
        # scattered random read against a potentially Lustre-backed file.
        # Measured: ~87 minutes just to fetch round 1's initial 99,460-row
        # labeled set this way. Without caching, EVERY round re-fetches the
        # WHOLE (growing) labeled set from scratch via EmbeddingMVEModel's
        # own _get_X() -- this cache makes each round only pay that cost for
        # the molecules labeled THIS round (a constant ~batch_size rows),
        # not the cumulative total.
        # A single incrementally-merged (idx, X) array pair -- NOT a dict of
        # per-row arrays. A dict version of this cache OOM-killed two real
        # runs at --mem=160G (2026-09-14): every round rebuilt a FRESH
        # stacked copy of the entire cumulative cache (in _get_X_cached AND,
        # separately, in _checkpoint()) while the dict itself ALSO held the
        # same data as many small arrays -- 2-3 full copies of a
        # monotonically-growing, never-shrinking dataset, all alive at once.
        # Here there is exactly ONE steady-state array (self._emb_cache_X);
        # each round's merge transiently allocates one new array sized for
        # the round's larger total before the old one is freed, but nothing
        # is ever kept 2x permanently. self._emb_cache_keys (plain ints, not
        # embedding data -- trivial memory) exists purely for the O(1)
        # "already cached?" membership test _get_X_cached needs per row.
        self._emb_cache_idx: np.ndarray = np.empty(0, dtype=np.int64)
        self._emb_cache_X: np.ndarray | None = None
        self._emb_cache_keys: set = set()
        # On a --resume restart, reload whatever embedding rows the PRIOR
        # process already fetched (persisted by _checkpoint() below) instead
        # of re-fetching the entire labeled set from the slow remote memmap.
        # Without this, a resumed process starts with a cold cache and must
        # fetch the whole (genuinely-scattered, AL-acquired) labeled set in
        # one go -- confirmed to stall for 2+ hours with 0% GPU utilization
        # on a real resume attempt (2026-08-15, job 9972734), since the
        # original process only ever built this cache up incrementally,
        # ~batch_size rows at a time, and never needed to do this in one shot.
        if self._start_round > 0:
            # Check the opportunistic "latest" snapshot FIRST (written after
            # every _get_X_cached fetch, not just at round-end -- see that
            # method) -- it can be strictly newer than the per-round
            # checkpoint, e.g. if a round's fetch completed but then TRAINING
            # itself crashed before the round could finish and checkpoint
            # normally (confirmed: a CUDA "illegal instruction" crash inside
            # _train_cached did exactly this, 2026-08-16, job 9979810 -- the
            # 298,269-row fetch it took ~5.2 HOURS to build would otherwise
            # have been silently discarded and re-fetched from scratch on
            # this very resume). Falls back to the per-round checkpoint for
            # a resume from an OLDER, already-checkpointed round instead.
            latest_path = self.run_dir / "emb_cache_latest.npz"
            cache_path = latest_path if latest_path.exists() else (
                self.run_dir / f"iter_{self._start_round}" / "emb_cache.npz"
            )
            if cache_path.exists():
                t0 = time.perf_counter()
                with np.load(cache_path) as data:
                    self._emb_cache_idx = data["idx"]
                    self._emb_cache_X = data["X"]
                self._emb_cache_keys = set(self._emb_cache_idx.tolist())
                print(f"[resume] loaded {len(self._emb_cache_idx):,} cached embedding rows "
                      f"from {cache_path} ({time.perf_counter() - t0:.1f}s)")
            else:
                print(f"[resume] no emb_cache.npz found at {cache_path} -- labeled-set "
                      f"embeddings will be re-fetched from the remote memmap (slow/scattered)")

    def _get_X_cached(self, idx_list: list) -> np.ndarray:
        """Like EmbeddingMVEModel._get_X(), but only fetches (fancy-indexes
        into self.model.emb_dict) rows not already cached. idx_list entries
        are global pool indices, which line up 1:1 with emb_dict's rows
        because self.model's smi2idx was built by enumerate()-ing the same
        pool_smiles array MVEExplorer holds (see EmbeddingMVEModel.__init__)
        -- no smi2idx round-trip needed here."""
        new_idx = sorted(i for i in idx_list if i not in self._emb_cache_keys)
        if new_idx:
            # Sequential streaming pass per backbone, NOT scattered
            # fancy-indexing (`emb[new_idx]`). new_idx is a uniformly
            # random subset of a 99M+-row pool (random init draw, or
            # AL-acquired top-k -- neither is chunk-local), so at any
            # useful block size almost every block contains at least one
            # wanted row: there is no "skip most of the file" shortcut
            # available here, the wanted rows are spread across virtually
            # the whole array regardless of how it's sliced. The win is
            # purely access pattern: reading the array in large sequential
            # blocks (a handful of big contiguous reads/backbone) instead
            # of one point-read per wanted row is dramatically faster on
            # Lustre, where scattered small random reads are latency-bound
            # (measured: ~270ms/row aggregate across 5 backbones this way,
            # e.g. ~87 minutes for round 1's 99,460-row init batch) while
            # large sequential reads run at near-full striped throughput.
            # Correctness: every wanted row still gets read exactly once,
            # just via a streaming two-pointer walk (both new_idx and the
            # block boundaries are monotonically increasing) rather than a
            # random-access gather.
            READ_BLOCK = 200_000  # rows/block; ~2.7GB for grover3400 (widest backbone) -- bounds peak memory
            n_new = len(new_idx)
            t0 = time.perf_counter()
            rows_by_idx: dict = {i: [] for i in new_idx}
            page_size = mmap.PAGESIZE
            for name, emb in self.model.emb_dict.items():
                n_total, dim = emb.shape
                # A reused, pre-allocated buffer -- NOT np.array(emb[a:b]) fresh
                # each iteration -- so this loop's own anonymous-memory footprint
                # is fixed at one block's size, never accumulating.
                buf = np.empty((min(READ_BLOCK, n_total), dim), dtype=emb.dtype)
                is_memmap = isinstance(emb, np.memmap)
                row_bytes = dim * emb.dtype.itemsize
                ptr = 0
                blocks_read = 0
                for block_start in range(0, n_total, READ_BLOCK):
                    if ptr >= n_new:
                        break  # every wanted row already found -- rest of the file is unneeded
                    block_end = min(block_start + READ_BLOCK, n_total)
                    if new_idx[ptr] >= block_end:
                        continue  # no wanted row in this block -- skip without reading it
                    n_rows = block_end - block_start
                    # buf[:n_rows] = ... forces a REAL copy from the memmap (a
                    # bare np.asarray(emb[a:b]) does NOT: it returns a view still
                    # backed by the memmap -- OWNDATA=False -- silently deferring
                    # the actual disk read to whatever touches it later. That was
                    # a real bug here: every page-fault ended up happening one row
                    # at a time inside the np.concatenate() loop after this one --
                    # the exact scattered-read cost this function exists to avoid,
                    # just moved to an unmeasured line, and masked by page-cache
                    # warmth on the first two real runs (py-spy caught a 2+ hour
                    # stall landed exactly on that concatenate line, 2026-09-13).
                    buf[:n_rows] = emb[block_start:block_end]
                    blocks_read += 1
                    # Release this block's pages from the process's resident set
                    # now that the data we need is copied out. Reading through a
                    # memmap (even into a reused buffer -- the copy above doesn't
                    # help here) still leaves the touched FILE-BACKED pages mapped
                    # and counted in RSS: Linux only reclaims clean mmap'd pages
                    # under real memory pressure, not proactively, so a full
                    # sequential scan of a huge file (unavoidable at this ~0.1%
                    # density -- virtually every block has a wanted row, so
                    # nothing gets skipped) drove RSS to the FULL file size before
                    # this call was added -- OOM-killed a real run at --mem=160G
                    # (2026-09-13, ~1.35TB grover3400 file). madvise(DONTNEED)
                    # tells the kernel these pages are safe to drop immediately;
                    # confirmed directly this bounds peak RSS to ~one block
                    # (~5.3GB measured) instead of the whole file (~40.8GB) on an
                    # isolated test of the same pattern.
                    if is_memmap:
                        byte_start = emb.offset + block_start * row_bytes
                        byte_end = emb.offset + block_end * row_bytes
                        aligned_start = (byte_start // page_size) * page_size
                        emb._mmap.madvise(mmap.MADV_DONTNEED, aligned_start, byte_end - aligned_start)
                    local_offsets, wanted_global = [], []
                    while ptr < n_new and new_idx[ptr] < block_end:
                        i = new_idx[ptr]
                        local_offsets.append(i - block_start)
                        wanted_global.append(i)
                        ptr += 1
                    if local_offsets:
                        # ONE fancy-index call, not a per-row basic-index append:
                        # basic indexing (buf[k]) returns a VIEW whose .base keeps
                        # the WHOLE buf alive for as long as that one row survives
                        # in rows_by_idx -- harmless with a single reused buffer
                        # (it's meant to stay alive), but fancy-indexing also
                        # produces its own small, independent array, which is what
                        # we actually want retained long-term in rows_by_idx.
                        extracted = buf[:n_rows][local_offsets]
                        for k, i in enumerate(wanted_global):
                            rows_by_idx[i].append(extracted[k])
                print(f"  [emb-cache] streamed {name}: {ptr:,}/{n_new:,} rows found "
                      f"via {blocks_read} sequential block reads "
                      f"({time.perf_counter() - t0:.1f}s elapsed)")
            new_idx_arr = np.array(new_idx, dtype=np.int64)
            new_X = np.stack([np.concatenate(rows_by_idx[i]) for i in new_idx])
            del rows_by_idx  # each row's small arrays are now duplicated into new_X; drop the originals

            # Merge old (already-cached) + new rows into ONE fresh array, in
            # sorted-by-global-index order -- NOT a per-row dict rebuild.
            # idx_list is always exactly sorted(self.labeled_idx) (the one
            # caller, _train_cached, guarantees this), and old cached indices
            # are always a subset of it, so merged_idx below always equals
            # sorted(idx_list); asserted rather than assumed so a future
            # caller that breaks this invariant fails loudly instead of
            # silently misaligning rows.
            merged_idx = np.array(sorted(idx_list), dtype=np.int64)
            assert merged_idx.tolist() == sorted(self._emb_cache_keys | set(new_idx)), (
                "idx_list doesn't match the union of already-cached + newly-fetched "
                "indices -- the merge below assumes every element of idx_list is "
                "either already cached or was just fetched into new_idx."
            )
            dim_total = new_X.shape[1] if self._emb_cache_X is None else self._emb_cache_X.shape[1]
            merged_X = np.empty((len(merged_idx), dim_total), dtype=np.float32)
            if self._emb_cache_X is not None and len(self._emb_cache_idx):
                old_positions = np.searchsorted(merged_idx, self._emb_cache_idx)
                merged_X[old_positions] = self._emb_cache_X
            new_positions = np.searchsorted(merged_idx, new_idx_arr)
            merged_X[new_positions] = new_X
            del new_X  # copied into merged_X above; the old self._emb_cache_X is dropped by reassignment below

            self._emb_cache_idx = merged_idx
            self._emb_cache_X = merged_X
            self._emb_cache_keys = set(merged_idx.tolist())

            self._snapshot_emb_cache()
            return merged_X
        # idx_list already fully covered by the cache (e.g. a redundant
        # repeat call) -- self._emb_cache_X is already in idx_list's exact
        # order per the invariant above, so this is a no-op reuse, not a copy.
        return self._emb_cache_X

    def _snapshot_emb_cache(self) -> None:
        """Opportunistically persists the FULL current cache to a single
        fixed path (overwritten each call, not per-round) right after any
        real fetch -- so an expensive fetch survives a crash in whatever
        comes AFTER it (e.g. training), not just a crash between rounds.
        See __init__'s resume-loading comment for why this exists.

        Reads self._emb_cache_idx/_emb_cache_X directly -- no rebuild, since
        those ARE the canonical single copy of the cache now (see
        _get_X_cached, which merges into them incrementally rather than
        keeping a separate dict this would otherwise have to be stacked
        from)."""
        t0 = time.perf_counter()
        idx_arr, X_arr = self._emb_cache_idx, self._emb_cache_X
        tmp_path = self.run_dir / "emb_cache_latest.npz.tmp"
        final_path = self.run_dir / "emb_cache_latest.npz"
        # np.savez() silently appends ".npz" to any path that doesn't already
        # end with it (tmp_path ends in ".tmp", not ".npz") -- passing an
        # open file OBJECT instead of a path string avoids that auto-append,
        # since numpy only does it for str/Path inputs (see
        # write_chunk_file()'s docstring in shared_embedding_store.py for
        # the same footgun hit earlier this session).
        with open(tmp_path, "wb") as f:
            np.savez(f, idx=idx_arr, X=X_arr)
        tmp_path.replace(final_path)  # atomic on POSIX -- never a truncated/partial file
        print(f"  [emb-cache] snapshotted {len(idx_arr):,} rows to {final_path} "
              f"({time.perf_counter() - t0:.1f}s)")

    def _train_cached(self, idx_list: list, ys: np.ndarray) -> None:
        """Equivalent to self.model.train(xs, ys, epochs=..., batch=...),
        but builds X via the cache above instead of EmbeddingMVEModel's own
        _get_X() (which would re-fetch the whole labeled set from the
        unfiltered memmap every round). Safe to bypass EmbeddingMVEModel.train()
        directly like this because the ensemble/learned surrogates used here
        never set needs_smiles=True (that's only the scheduled fine-tuning
        surrogates, not used by --parallel-predict)."""
        X = self._get_X_cached(idx_list)
        fit_kwargs = {}
        if self.surrogate_epochs is not None:
            fit_kwargs["epochs"] = self.surrogate_epochs
        if self.surrogate_batch is not None:
            fit_kwargs["batch"] = self.surrogate_batch
        assert not getattr(self.model.surrogate, "needs_smiles", False), (
            "ParallelMVEExplorer's embedding cache bypasses EmbeddingMVEModel.train()'s "
            "smiles-based _get_X() -- not valid for surrogates that need raw SMILES "
            "(scheduled fine-tuning types), only frozen-embedding ones like ensemble/learned."
        )
        self.model.surrogate.fit(X, ys, **fit_kwargs)
        self.model.embeddings_refreshed = getattr(self.model.surrogate, "embeddings_refreshed", False)

    def _wait_for_shard(self, done_path: Path, r: int, shard_id: int) -> None:
        waited = 0.0
        while not done_path.exists():
            time.sleep(self.poll_interval)
            waited += self.poll_interval
            if waited % 60 < self.poll_interval:  # roughly once/minute
                print(f"  [round {r}] still waiting on shard {shard_id} ({waited:.0f}s so far)")

    def run(self) -> list:
        history = list(self._resumed_history)
        n = len(self.pool_smiles)

        for rnd in range(self._start_round, self.n_rounds):
            t0 = time.perf_counter()
            r = rnd + 1

            idx = sorted(self.labeled_idx)  # sorted: better locality for the (possibly-new) fetch below
            ys = self._sign * np.array([self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32)
            self._train_cached(idx, ys)

            # Hand the freshly-trained surrogate to the workers. torch.save
            # on the whole surrogate object (not just a state_dict) since
            # fusion surrogates (Ensemble/Learned) also carry non-torch
            # state (e.g. LearnedFusionSurrogate's sklearn RidgeCV) that a
            # bare state_dict wouldn't capture.
            ckpt_path = self.coord_dir / f"round_{r}_surrogate.pt"
            torch.save(self.model.surrogate, ckpt_path)
            (self.coord_dir / f"round_{r}_ready.marker").touch()

            # Each worker predicts its ENTIRE fixed shard (including
            # already-labeled molecules) every round -- simpler and more
            # robust than keeping N workers' notion of "still unlabeled" in
            # sync with this process's growing labeled set. Mask afterward.
            mu_parts, var_parts = [], []
            for shard_id in range(self.num_shards):
                if shard_id in self.exclude_shard_ids:
                    # Never waited on, never predicted -- fill with a
                    # placeholder just to keep mu_full/var_full's length
                    # matching the full pool (usable_mask guarantees this
                    # range is never selected regardless of placeholder value).
                    s, e = _chunk_bounds(n, shard_id, self.num_shards)
                    mu_parts.append(np.zeros(e - s, dtype=np.float32))
                    var_parts.append(np.zeros(e - s, dtype=np.float32))
                    continue
                done_path = self.coord_dir / f"round_{r}_shard_{shard_id}.done"
                self._wait_for_shard(done_path, r, shard_id)
                mu_parts.append(np.load(self.coord_dir / f"round_{r}_shard_{shard_id}_mu.npy"))
                var_parts.append(np.load(self.coord_dir / f"round_{r}_shard_{shard_id}_var.npy"))
            mu_full = np.concatenate(mu_parts)
            var_full = np.concatenate(var_parts)
            assert len(mu_full) == n, f"gathered {len(mu_full):,} predictions but pool has {n:,} molecules"

            mask = np.ones(n, bool) if self.usable_mask is None else self.usable_mask.copy()
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            mu, var = mu_full[pool_idx], var_full[pool_idx]

            if self.acq_name in ("ucb", "lcb", "thompson", "ts", "ei", "pi"):
                scores = self.acq_fn(mu, var)
            else:
                scores = self.acq_fn(mu)

            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            recall = self._recall()
            elapsed = time.perf_counter() - t0
            print(f"  Round {rnd+1:02d}/{self.n_rounds}  labeled={len(self.labeled_idx):,}  "
                  f"best={self._best():.3f} kcal/mol  top-{self.topk} recall={recall:.1%}  ({elapsed:.1f}s)")

            record = dict(round=rnd + 1, n_labeled=len(self.labeled_idx),
                          best_score=float(self._best()), topk_recall=float(recall), elapsed=round(elapsed, 2))
            history.append(record)
            self._checkpoint(rnd + 1, record)

        # Lets workers exit their poll loops instead of waiting forever on
        # a round n_rounds+1 that will never come.
        (self.coord_dir / "STOP.marker").touch()
        self._save_final(history)
        return history

    def _checkpoint(self, rnd, record):
        super()._checkpoint(rnd, record)
        # The embedding cache itself is NOT re-persisted here on top of
        # state.json/scores.pkl -- emb_cache_latest.npz (written by
        # _snapshot_emb_cache(), atomically, after every _get_X_cached fetch,
        # not just at round-end) already fully supersedes any per-round copy
        # as the resume source (see __init__'s resume-loading comment/order).
        # A per-round iter_N/emb_cache.npz used to be written here too, but
        # it was pure redundant disk bloat -- an unbounded, ever-growing
        # duplicate of the SAME cache on every single round, never cleaned
        # up (2026-09-16: this run's iter_1..4 alone had grown to ~120GB of
        # exact duplicates before deletion, and combined with a separate
        # completed run's own uncapped leftovers, blew the account's 800GB
        # quota mid-round -- OSError: Disk quota exceeded killed both the
        # UCB and greedy orchestrators). No functionality is lost: resume
        # never actually fell back to it in practice, since emb_cache_latest.npz
        # is written strictly more often and atomically.


# ==============================================================================
# MOLPAL ACTIVE LEARNING LOOP (fingerprint-based models, e.g. MPN)
# ==============================================================================

class MolPALExplorer:
    def __init__(
        self, pool_smiles, oracle, model_type, acq, conf_method="mve",
        fingerprint="pair", radius=2, length=2048,
        init_size=8417, batch_size=8417, n_rounds=5, topk=1000,
        run_dir=None, seed=42, ncpu=1,
    ):
        from molpal.models import model as build_model
        from molpal.featurizer import Featurizer
        from molpal.acquirer.metrics import get_metric

        self.pool_smiles = np.array(pool_smiles)
        self.oracle = oracle
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.topk = topk
        self.run_dir = run_dir or RUNS_DIR / "molpal_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sign = -1.0

        self.featurizer = Featurizer(fingerprint=fingerprint, radius=radius, length=length)
        self.model = build_model(model=model_type, conf_method=conf_method, input_size=length,
                                  test_batch_size=4096, ncpu=ncpu)

        self.acq_fn = get_metric(acq)
        self.acq_name = acq
        self.needs_var = acq in ("ucb", "lcb", "thompson", "ts", "ei", "pi")
        self.true_top_k = true_top_k_set(oracle, topk)

        rng = np.random.default_rng(seed)
        init_idx = rng.choice(len(self.pool_smiles), init_size, replace=False)
        self.labeled_idx = set(init_idx.tolist())
        self.labeled_scores = {self.pool_smiles[i]: oracle[self.pool_smiles[i]] for i in init_idx}
        print(f"[init] {init_size} random molecules  best={self._best():.3f} kcal/mol")

    def _best(self) -> float:
        return min(self.labeled_scores.values())

    def _recall(self) -> float:
        return sum(1 for s in self.labeled_scores if s in self.true_top_k) / self.topk

    def run(self) -> list:
        history = []
        n = len(self.pool_smiles)

        for rnd in range(self.n_rounds):
            t0 = time.perf_counter()

            idx = list(self.labeled_idx)
            xs = [self.pool_smiles[i] for i in idx]
            ys = self._sign * np.array([self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32)

            self.model.train(xs, ys, featurizer=self.featurizer)

            mask = np.ones(n, bool)
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            pool_smi = [self.pool_smiles[i] for i in pool_idx]

            mu, var = _chunked_predict_molpal(self.model, self.featurizer, pool_smi, self.needs_var)
            scores = self.acq_fn(mu, var) if self.needs_var else self.acq_fn(mu)

            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            recall = self._recall()
            elapsed = time.perf_counter() - t0
            print(f"  Round {rnd+1:02d}/{self.n_rounds}  labeled={len(self.labeled_idx):,}  "
                  f"best={self._best():.3f} kcal/mol  top-{self.topk} recall={recall:.1%}  ({elapsed:.1f}s)")

            record = dict(round=rnd + 1, n_labeled=len(self.labeled_idx),
                          best_score=float(self._best()), topk_recall=float(recall), elapsed=round(elapsed, 2))
            history.append(record)
            self._checkpoint(rnd + 1, record)

        self._save_final(history)
        return history

    def _checkpoint(self, rnd, record):
        d = self.run_dir / f"iter_{rnd}"
        d.mkdir(exist_ok=True)
        (d / "state.json").write_text(json.dumps(record, indent=2))
        with open(d / "scores.pkl", "wb") as f:
            pickle.dump(dict(self.labeled_scores), f)

    def _save_final(self, history):
        pd.DataFrame(sorted(self.labeled_scores.items(), key=lambda x: x[1]), columns=["smiles", "score"]) \
            .to_csv(self.run_dir / "all_explored_final.csv", index=False)
        (self.run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\n[done] results -> {self.run_dir}")


def build_mpn_model(ncpu: int = 1, length: int = 2048, batch_size: Optional[int] = None):
    """The MolPAL MPN (mean-variance-estimation head) exactly as
    MolPALExplorer builds it -- shared by the parallel orchestrator and its
    prediction workers so both sides construct an identical architecture.

    batch_size=None preserves MPNN's own default (50, the paper's value);
    prediction workers never need to override it (test_batch_size is set
    independently and batch_size never enters inference), so only the
    orchestrator's call site ever passes a non-None value."""
    from molpal.models import model as build_model
    kwargs = dict(model="mpn", conf_method="mve", input_size=length,
                  test_batch_size=4096, ncpu=ncpu)
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    return build_model(**kwargs)


class ParallelMolPALExplorer(MolPALExplorer):
    """MolPALExplorer's loop (MolPAL's own MPN surrogate, retrained each
    round on all labeled data) with the per-round pool-prediction step
    delegated to a pool of single-GPU workers (predict_pool_shard_worker_mpn.py)
    over the same marker-file coord-dir protocol ParallelMVEExplorer uses.

    Why this exists: MolPALExplorer predicts the whole pool in one process
    every round. Measured at ~1.7 ms/molecule, that is ~47h per pass over
    AmpC's 98.5M molecules on one GPU -- the only way to a full run is
    sharding that step. Training stays in this process (single GPU).

    Differences from MolPALExplorer worth knowing:
      * pool_smiles stays a plain list of the FULL library, with a
        boolean usable_mask for oracle-scored molecules (np.array of 98M
        unicode strings would itself be tens of GB); workers shard the
        same unfiltered library, so gathered predictions line up.
      * Initial labeled set is drawn exactly as MVEExplorer does (same
        seed, same usable_mask) so round 1 matches the LT-All runs.
      * retrain_from_scratch=True (default) reinitializes the MPN every
        round, matching the paper's "fully retrained from scratch with all
        acquired data at the beginning of each iteration". MolPALExplorer
        continues training the previous weights instead (its default).
      * --resume supported: labeled set restored from iter_N/scores.pkl;
        exact only with retrain_from_scratch=True, since model weights
        are not part of the resume state.
    """

    def __init__(
        self, pool_smiles, oracle, acq, usable_mask, num_shards, coord_dir,
        init_size=8417, batch_size=8417, n_rounds=5, topk=1000, run_dir=None,
        seed=42, ncpu=1, poll_interval=5.0, retrain_from_scratch=True,
        resume_scores=None, resume_round=0, length=2048, mpn_batch_size=None,
    ):
        # Deliberately does NOT call MolPALExplorer.__init__ (it np.array()s
        # the pool and draws the init set over unusable molecules too);
        # sets every attribute the inherited _best/_recall/_checkpoint/
        # _save_final use.
        from molpal.acquirer.metrics import get_metric

        self.pool_smiles = pool_smiles
        self.oracle = oracle
        self.usable_mask = usable_mask
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.topk = topk
        self.run_dir = Path(run_dir) if run_dir else RUNS_DIR / "molpal_parallel_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._sign = -1.0
        self.num_shards = num_shards
        self.coord_dir = Path(coord_dir)
        self.coord_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.retrain_from_scratch = retrain_from_scratch

        self.model = build_mpn_model(ncpu=ncpu, length=length, batch_size=mpn_batch_size)
        self.acq_fn = get_metric(acq)
        self.acq_name = acq
        self.needs_var = acq in ("ucb", "lcb", "thompson", "ts", "ei", "pi")
        self.true_top_k = true_top_k_set(oracle, topk)
        self._start_round = resume_round
        self._resumed_history = []

        if resume_scores is not None:
            if not retrain_from_scratch:
                print("[resume] WARNING: retrain_from_scratch=False -- model weights are not "
                      "checkpointed, so this resume restarts training from fresh weights")
            self.labeled_scores = dict(resume_scores)
            wanted = set(self.labeled_scores)
            smi2idx = {}
            for i, s in enumerate(pool_smiles):
                if s in wanted:
                    smi2idx[s] = i  # last-write-wins, same as MVEExplorer's smi2idx
            self.labeled_idx = {smi2idx[s] for s in wanted if s in smi2idx}
            for r in range(1, resume_round + 1):
                state_path = self.run_dir / f"iter_{r}" / "state.json"
                if state_path.exists():
                    self._resumed_history.append(json.loads(state_path.read_text()))
            print(f"[resume] loaded {len(self.labeled_idx):,} labeled molecules from "
                  f"{self.run_dir}/iter_{resume_round}  best={self._best():.3f} kcal/mol  "
                  f"resuming at round {resume_round + 1}/{n_rounds}")
        else:
            rng = np.random.default_rng(seed)
            candidate_idx = np.where(usable_mask)[0]
            init_idx = rng.choice(candidate_idx, init_size, replace=False)
            self.labeled_idx = set(init_idx.tolist())
            self.labeled_scores = {pool_smiles[i]: oracle[pool_smiles[i]] for i in init_idx}
            print(f"[init] {init_size} random molecules  best={self._best():.3f} kcal/mol")

    def _wait_for_shard(self, done_path: Path, r: int, shard_id: int) -> None:
        waited = 0.0
        while not done_path.exists():
            time.sleep(self.poll_interval)
            waited += self.poll_interval
            if waited % 60 < self.poll_interval:
                print(f"  [round {r}] still waiting on shard {shard_id} ({waited:.0f}s so far)", flush=True)

    def run(self) -> list:
        history = list(self._resumed_history)
        n = len(self.pool_smiles)

        # A previous (finished or interrupted-after-finish) run leaves
        # STOP.marker behind, and workers exit the moment they see it --
        # a resumed run must clear it or its workers would leave immediately.
        stale_stop = self.coord_dir / "STOP.marker"
        if stale_stop.exists():
            stale_stop.unlink()

        for rnd in range(self._start_round, self.n_rounds):
            t0 = time.perf_counter()
            r = rnd + 1

            idx = sorted(self.labeled_idx)
            xs = [self.pool_smiles[i] for i in idx]
            ys = self._sign * np.array([self.labeled_scores[self.pool_smiles[i]] for i in idx], dtype=np.float32)
            t_train = time.perf_counter()
            self.model.train(xs, ys, retrain=self.retrain_from_scratch)
            print(f"  [MPN] trained on {len(idx):,} molecules in {time.perf_counter() - t_train:.1f}s", flush=True)

            # Checkpoint dir (model.pt + state.json incl. target scaler) is
            # fully written BEFORE the ready marker is touched.
            self.model.save(self.coord_dir / f"round_{r}_mpn")
            (self.coord_dir / f"round_{r}_ready.marker").touch()

            mu_parts, var_parts = [], []
            for shard_id in range(self.num_shards):
                done_path = self.coord_dir / f"round_{r}_shard_{shard_id}.done"
                self._wait_for_shard(done_path, r, shard_id)
                mu_parts.append(np.load(self.coord_dir / f"round_{r}_shard_{shard_id}_mu.npy"))
                var_parts.append(np.load(self.coord_dir / f"round_{r}_shard_{shard_id}_var.npy"))
            mu_full = np.concatenate(mu_parts)
            var_full = np.concatenate(var_parts)
            assert len(mu_full) == n, f"gathered {len(mu_full):,} predictions but pool has {n:,} molecules"

            mask = self.usable_mask.copy()
            for i in self.labeled_idx:
                mask[i] = False
            pool_idx = np.where(mask)[0]
            mu, var = mu_full[pool_idx], var_full[pool_idx]

            scores = self.acq_fn(mu, var) if self.needs_var else self.acq_fn(mu)
            top_local = np.argsort(scores)[::-1][: self.batch_size]
            selected = pool_idx[top_local]

            for i in selected:
                smi = self.pool_smiles[i]
                self.labeled_scores[smi] = self.oracle[smi]
                self.labeled_idx.add(int(i))

            recall = self._recall()
            elapsed = time.perf_counter() - t0
            print(f"  Round {r:02d}/{self.n_rounds}  labeled={len(self.labeled_idx):,}  "
                  f"best={self._best():.3f} kcal/mol  top-{self.topk} recall={recall:.1%}  ({elapsed:.1f}s)")

            record = dict(round=r, n_labeled=len(self.labeled_idx),
                          best_score=float(self._best()), topk_recall=float(recall), elapsed=round(elapsed, 2))
            history.append(record)
            self._checkpoint(r, record)

        (self.coord_dir / "STOP.marker").touch()
        self._save_final(history)
        return history


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MolPAL Enamine HTS (2.1M, thymidylate kinase/4UNN) method comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--dataset", default="EnamineHTS", choices=list(DATASETS.keys()),
                    help="EnamineHTS (2.1M, Figure 4, top-1000) or AmpC (99.5M, Figure 5, top-50000)")
    p.add_argument("--mode", default="molpal", choices=["mve", "molpal"])
    p.add_argument("--acq", default="greedy", choices=["greedy", "ucb"])
    p.add_argument("--init-size", type=int, default=8417, help="Absolute count (init-frac * n_scored_pool)")
    p.add_argument("--batch-size", type=int, default=8417, help="Absolute count (batch-frac * n_scored_pool)")
    p.add_argument("--n-rounds", type=int, default=5)
    p.add_argument("--topk", type=int, default=None,
                    help="Fixed top-k for the recall metric. Default: 1000 for EnamineHTS (Figure 4), "
                         "50000 for AmpC (Figure 5) -- set from --dataset if not given explicitly.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ncpu", type=int, default=1,
                    help="Worker processes for MPN's DataLoaders (train + predict). Defaulted to 1 "
                         "everywhere upstream; match to your --cpus-per-task for real benefit.")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--pool-limit", type=int, default=None, help="Truncate the pool to the first N molecules (smoke testing only)")
    p.add_argument("--resume", action="store_true",
                    help="Resume from the latest runs/<run-dir>/iter_N/scores.pkl checkpoint instead of "
                         "starting a fresh random init -- for recovering from a crash (e.g. the orchestrator "
                         "OOM-killed partway into a later round) without redoing already-completed rounds. "
                         "--run-dir must point at the crashed run's directory. No-op (falls back to fresh "
                         "init) if no iter_N checkpoint exists there yet.")

    mve_grp = p.add_argument_group("MVE surrogate options (--mode mve)")
    mve_grp.add_argument(
        "--surrogate", default="ft_molformer_single",
        choices=["single", "ft_molformer_single", "ensemble", "ft_fusion", "learned", "ltall"],
        help="ft_molformer_single = method [2] (MoLFormer, fine-tuned each round); "
             "ensemble = method [3] (\"our fusion model\", EnsembleFusionSurrogate); "
             "ft_fusion = grover+molformer fine-tuned jointly each round, unimol frozen "
             "(FTFusionSurrogate; unimol excluded because its conformer cache can't be "
             "made memory-safe the same way -- see surrogates.py docstring); "
             "learned = 3 frozen per-backbone models combined by a RidgeCV meta-learner "
             "fit on held-out backbone predictions each round (LearnedFusionSurrogate) "
             "-- no fine-tuning, cheap, comparable cost to ensemble; "
             "ltall = reproduction of the LT-All paper's fusion architecture "
             "(LTAllSurrogate): learned per-source-weighted concatenation of ALL "
             "backbones into one shared MLP (1024/512 hidden, LayerNorm), trained "
             "jointly (not per-backbone) with a fresh reinit every round -- see "
             "surrogates.py's LTAllSurrogate docstring for the exact paper mapping",
    )
    mve_grp.add_argument("--backbone", default="molformer", choices=["molformer", "grover", "grover3400", "unimol", "unimol2", "smited", "mhgged"])
    mve_grp.add_argument("--backbones", nargs="+", default=["molformer"], choices=["molformer", "grover", "grover3400", "unimol", "unimol2", "smited", "mhgged"])
    mve_grp.add_argument("--parallel-predict", action="store_true",
                          help="Delegate per-round pool prediction to a persistent pool of single-GPU "
                               "workers (predict_pool_shard_worker.py) instead of predicting in-process. "
                               "Workers must already be running against the same --coord-dir before this "
                               "starts (see slurm/al_runs/ampc/submit_ampc_al_predict_workers_h100single.sh).")
    mve_grp.add_argument("--num-shards", type=int, default=None, help="Required with --parallel-predict")
    mve_grp.add_argument("--coord-dir", default=None,
                          help="Marker-file coordination directory shared with the workers. "
                               "Required with --parallel-predict.")
    mve_grp.add_argument("--poll-interval", type=float, default=5.0,
                          help="Seconds between checks for worker completion markers (--parallel-predict only)")
    mve_grp.add_argument("--exclude-shard-ids", default=None,
                          help="Comma-separated shard IDs to permanently drop from future candidate selection "
                               "(--parallel-predict only) -- e.g. a shard whose embedding data keeps producing "
                               "hung scattered reads. Already-labeled molecules in that range stay correctly "
                               "labeled; only future acquisition is affected. Example: --exclude-shard-ids 7")
    mve_grp.add_argument("--surrogate-epochs", type=int, default=None,
                          help="Override the surrogate's training epochs (default: each surrogate class's "
                               "own default, currently 50 -- tuned for EnamineHTS-scale labeled sets, likely "
                               "too many at AmpC scale where labeled sets are ~12x larger per round).")
    mve_grp.add_argument("--surrogate-batch", type=int, default=None,
                          help="Override the surrogate's training batch size (default: each surrogate "
                               "class's own default, currently 256).")

    mp_grp = p.add_argument_group("MolPAL model options (--mode molpal)")
    mp_grp.add_argument("--model", default="mpn", choices=["mpn", "rf", "nn"],
                         help="method [1] is MPN; rf/nn added for the paper's own Figure 4 (RF/NN/MPN, greedy-only)")
    mp_grp.add_argument("--conf-method", default="mve", choices=["none", "dropout", "mve"])
    mp_grp.add_argument("--fingerprint", default="pair", choices=["morgan", "pair", "rdkit", "maccs"],
                         help="paper uses Atom-pair (pair), not Morgan, for RF/NN/MPN inputs")
    mp_grp.add_argument("--radius", type=int, default=2)
    mp_grp.add_argument("--length", type=int, default=2048)
    mp_grp.add_argument("--retrain-from-scratch", action=argparse.BooleanOptionalAction, default=True,
                         help="(--mode molpal --parallel-predict only) reinitialize the MPN every round, "
                              "as in the paper ('fully retrained from scratch with all acquired data at the "
                              "beginning of each iteration'). --no-retrain-from-scratch continues training "
                              "the previous round's weights, as the single-process MolPALExplorer does.")
    mp_grp.add_argument("--mpn-batch-size", type=int, default=None,
                         help="Override MPNN's training batch_size (default: 50, chemprop/paper default). "
                              "A deliberate deviation from the paper's own hyperparameter, NOT just a speed "
                              "knob -- batch_size interacts with warmup_epochs/the LR schedule, so this can "
                              "change convergence behavior and recall, not only wall-clock. Only affects "
                              "ParallelMolPALExplorer's training (the orchestrator); prediction workers are "
                              "unaffected (test_batch_size is set independently and batch_size never enters "
                              "inference).")

    return p.parse_args()


def main():
    args = parse_args()
    global DATASET
    DATASET = args.dataset
    if args.topk is None:
        args.topk = 50_000 if DATASET == "AmpC" else 1000
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}\n  {DATASET}  mode={args.mode}  acq={args.acq}\n  device={DEVICE}  seed={args.seed}\n{'='*60}\n")

    oracle = load_oracle()

    if args.mode == "mve":
        backbones = [args.backbone] if args.surrogate == "single" else args.backbones
        emb_dict, pool_smiles = load_embeddings(backbones)

        if args.pool_limit is not None:
            pool_smiles = pool_smiles[: args.pool_limit]
            emb_dict = {bb: emb[: args.pool_limit] for bb, emb in emb_dict.items()}

        usable_set = set(oracle.keys())

        suffix = f"mve_{args.surrogate}_{'_'.join(backbones)}_{args.acq}_init{args.init_size}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        resume_round, resume_scores = (0, None)
        if args.resume:
            resume_round, resume_scores = find_resume_checkpoint(run_dir)
            if resume_scores is None:
                print(f"[resume] --resume given but no iter_N checkpoint found under {run_dir} -- starting fresh")

        if args.parallel_predict:
            if args.num_shards is None or args.coord_dir is None:
                raise SystemExit("--parallel-predict requires both --num-shards and --coord-dir")
            # Do NOT pre-filter emb_dict/pool_smiles here the way the
            # non-parallel branch below does -- emb[np.array(keep_idx)] on a
            # huge, non-contiguous index set forces a full in-RAM copy
            # (measured: ~1.1TB for AmpC's 3 backbones concatenated across
            # ~96M rows, well past any single node's memory -- this is what
            # OOM-killed the first --parallel-predict attempt). Keep
            # emb_dict/pool_smiles as the full, untouched (memmapped) arrays
            # instead, and carry "which positions are oracle-scored" as a
            # cheap boolean mask (bits, not floats) that ParallelMVEExplorer
            # applies when choosing the initial labeled set and each round's
            # acquisition candidates. This also keeps pool_smiles's length
            # equal to what predict_pool_shard_worker.py's workers
            # independently shard (the full, unfiltered pool) -- required
            # for the round-end gather/concatenate to line up at all.
            usable_mask = np.fromiter((s in usable_set for s in pool_smiles), dtype=bool, count=len(pool_smiles))
            print(f"[pool] {usable_mask.sum():,}/{len(pool_smiles):,} molecules have oracle scores "
                  f"(embeddings kept unfiltered/memmapped -- see --parallel-predict comment)")
            exclude_shard_ids = frozenset(
                int(x) for x in args.exclude_shard_ids.split(",") if x.strip()
            ) if args.exclude_shard_ids else frozenset()
            explorer = ParallelMVEExplorer(
                emb_dict=emb_dict, pool_smiles=pool_smiles, oracle=oracle,
                surrogate_type=args.surrogate, backbone=args.backbone, acq=args.acq,
                init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
                topk=args.topk, run_dir=run_dir, seed=args.seed, usable_mask=usable_mask,
                num_shards=args.num_shards, coord_dir=args.coord_dir, poll_interval=args.poll_interval,
                exclude_shard_ids=exclude_shard_ids,
                surrogate_epochs=args.surrogate_epochs, surrogate_batch=args.surrogate_batch,
                resume_scores=resume_scores, resume_round=resume_round,
            )
        else:
            keep_idx = [i for i, s in enumerate(pool_smiles) if s in usable_set]
            emb_dict = {bb: emb[np.array(keep_idx)] for bb, emb in emb_dict.items()}
            pool_smiles = [pool_smiles[i] for i in keep_idx]
            print(f"[pool] {len(pool_smiles):,} molecules with embeddings + oracle scores")
            explorer = MVEExplorer(
                emb_dict=emb_dict, pool_smiles=pool_smiles, oracle=oracle,
                surrogate_type=args.surrogate, backbone=args.backbone, acq=args.acq,
                resume_scores=resume_scores, resume_round=resume_round,
                init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
                topk=args.topk, run_dir=run_dir, seed=args.seed,
                surrogate_epochs=args.surrogate_epochs, surrogate_batch=args.surrogate_batch,
            )

    elif args.parallel_predict:
        if args.model != "mpn":
            raise SystemExit("--parallel-predict with --mode molpal is only implemented for --model mpn")
        if args.num_shards is None or args.coord_dir is None:
            raise SystemExit("--parallel-predict requires both --num-shards and --coord-dir")
        # Full, unfiltered library + a usable mask -- same reasoning as the
        # --mode mve --parallel-predict branch above: workers shard the
        # unfiltered library, so the gathered predictions must cover it.
        pool_smiles = load_library_smiles(limit=args.pool_limit)
        usable_mask = np.fromiter((s in oracle for s in pool_smiles), dtype=bool, count=len(pool_smiles))
        print(f"[pool] {usable_mask.sum():,}/{len(pool_smiles):,} molecules have oracle scores")

        suffix = f"molpal_{args.model}_parallel_{args.acq}_init{args.init_size}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        resume_round, resume_scores = (0, None)
        if args.resume:
            resume_round, resume_scores = find_resume_checkpoint(run_dir)
            if resume_scores is None:
                print(f"[resume] --resume given but no iter_N checkpoint found under {run_dir} -- starting fresh")

        explorer = ParallelMolPALExplorer(
            pool_smiles=pool_smiles, oracle=oracle, acq=args.acq, usable_mask=usable_mask,
            num_shards=args.num_shards, coord_dir=args.coord_dir,
            init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
            topk=args.topk, run_dir=run_dir, seed=args.seed, ncpu=args.ncpu,
            poll_interval=args.poll_interval, retrain_from_scratch=args.retrain_from_scratch,
            resume_scores=resume_scores, resume_round=resume_round, length=args.length,
            mpn_batch_size=args.mpn_batch_size,
        )

    else:
        pool_smiles = load_library_smiles(limit=args.pool_limit)
        pool_smiles = [s for s in pool_smiles if s in oracle]
        print(f"[pool] {len(pool_smiles):,} molecules with oracle scores")

        suffix = f"molpal_{args.model}_{args.acq}_init{args.init_size}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        explorer = MolPALExplorer(
            pool_smiles=pool_smiles, oracle=oracle, model_type=args.model, acq=args.acq,
            conf_method=args.conf_method, fingerprint=args.fingerprint, radius=args.radius, length=args.length,
            init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
            topk=args.topk, run_dir=run_dir, seed=args.seed, ncpu=args.ncpu,
        )

    history = explorer.run()

    print(f"\n{'Round':>6} {'Labeled':>10} {'Best (kcal/mol)':>16} {'Top-k recall':>14}")
    print("-" * 52)
    for r in history:
        print(f"{r['round']:>6} {r['n_labeled']:>10,} {r['best_score']:>16.3f} {r['topk_recall']:>13.1%}")


if __name__ == "__main__":
    main()
