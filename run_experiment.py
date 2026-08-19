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
batch-size fractions -- orchestrated by run_all_configs.sh, not this file
directly (this file runs exactly one config per invocation).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generate_unimol_conformers_chunk import _chunk_bounds

ROOT = Path(__file__).resolve().parent

EMBED_DIR = ROOT / "results" / "embed"
DATA_DIR = ROOT / "data"
LIBRARY_DIR = ROOT / "molpal" / "libraries"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

AMPC_ROOT = Path("/N/project/SingleCell_Image/mengjing/ampc_99.5M")

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
    "AmpC": {
        "library": AMPC_ROOT / "ampc_smiles.txt",
        "oracle": AMPC_ROOT / "ampc_scores.csv.gz",
        # Lustre-striped (28-way, vs. the original's stripe count 1) copy of
        # the same data -- re-stitched from the same per-chunk files via
        # submit_ampc_stitch_embeddings_striped.sh, spot-verified bit-identical
        # to the original embeddings/ (shape, sample rows, NaN counts all
        # matched, 2026-08-15). Switched to fix scattered fancy-indexed reads
        # (AL training-set fetches, shard 7's per-chunk correction reads)
        # funneling through a single OST -- confirmed via nvidia-smi showing
        # 0% GPU utilization during a 2+ hour stall that this doesn't fix the
        # cause of, just spreads the same access pattern across 28 OSTs
        # instead of 1. The original embeddings/ is left on disk, untouched.
        "embed_dir": AMPC_ROOT / "embeddings_striped",
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
            # partway into a later round -- see submit_ampc_fusion_runs_h100single.sh)
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
        self._emb_cache: dict = {}
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
                    idx_arr, X_arr = data["idx"], data["X"]
                self._emb_cache = {int(i): row for i, row in zip(idx_arr, X_arr)}
                print(f"[resume] loaded {len(self._emb_cache):,} cached embedding rows "
                      f"from {cache_path} ({time.perf_counter() - t0:.1f}s)")
            else:
                print(f"[resume] no emb_cache.npz found at {cache_path} -- labeled-set "
                      f"embeddings will be re-fetched from the remote memmap (slow/scattered)")

    def _get_X_cached(self, idx_list: list) -> np.ndarray:
        """Like EmbeddingMVEModel._get_X(), but only fetches (fancy-indexes
        into self.model.emb_dict) rows not already in self._emb_cache.
        idx_list entries are global pool indices, which line up 1:1 with
        emb_dict's rows because self.model's smi2idx was built by
        enumerate()-ing the same pool_smiles array MVEExplorer holds (see
        EmbeddingMVEModel.__init__) -- no smi2idx round-trip needed here."""
        new_idx = sorted(i for i in idx_list if i not in self._emb_cache)
        if new_idx:
            # Bounded, chunked fetch -- NOT for correctness (fancy-indexing
            # the whole list at once would give the same result), but for
            # visibility and blast-radius. A single fancy-index call over
            # the WHOLE (cold-cache, genuinely-scattered, AL-acquired)
            # labeled set gave zero progress signal -- on a --resume restart
            # this sat with no log output and no checkpoint write for 2+
            # hours (2026-08-15, job 9972734), indistinguishable from a hang
            # vs. just slow, on the same single-OST/unstriped embedding
            # files that have hung on shard 7 all session. Chunking doesn't
            # remove the scattered-read risk itself, but bounds it: a stall
            # now shows which range it's stuck on instead of one opaque
            # multi-hour silence.
            CHUNK = 20_000
            n_new = len(new_idx)
            t0 = time.perf_counter()
            for start in range(0, n_new, CHUNK):
                chunk_idx = new_idx[start : start + CHUNK]
                parts = [emb[chunk_idx] for emb in self.model.emb_dict.values()]
                chunk_rows = np.concatenate(parts, axis=1)
                for row, i in zip(chunk_rows, chunk_idx):
                    self._emb_cache[i] = row
                done = min(start + CHUNK, n_new)
                print(f"  [emb-cache] fetched {done:,}/{n_new:,} new rows "
                      f"({time.perf_counter() - t0:.1f}s elapsed)")
            self._snapshot_emb_cache()
        return np.stack([self._emb_cache[i] for i in idx_list])

    def _snapshot_emb_cache(self) -> None:
        """Opportunistically persists the FULL current cache to a single
        fixed path (overwritten each call, not per-round) right after any
        real fetch -- so an expensive fetch survives a crash in whatever
        comes AFTER it (e.g. training), not just a crash between rounds.
        See __init__'s resume-loading comment for why this exists."""
        t0 = time.perf_counter()
        idx_arr = np.array(sorted(self._emb_cache.keys()), dtype=np.int64)
        X_arr = np.stack([self._emb_cache[i] for i in idx_arr])
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
        # Persist the embedding cache too -- see __init__'s resume-loading
        # comment. A local sequential write of a few GB is fast/reliable,
        # unlike the scattered remote read it exists to let a future resume
        # skip. Written every round (not just incrementally) so resume only
        # ever needs the single latest iter_N/emb_cache.npz, no merging.
        if self._emb_cache:
            d = self.run_dir / f"iter_{rnd}"
            t0 = time.perf_counter()
            idx_arr = np.array(sorted(self._emb_cache.keys()), dtype=np.int64)
            X_arr = np.stack([self._emb_cache[i] for i in idx_arr])
            np.savez(d / "emb_cache.npz", idx=idx_arr, X=X_arr)
            print(f"  [checkpoint] saved {len(idx_arr):,} cached embedding rows "
                  f"({time.perf_counter() - t0:.1f}s)")


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
    mve_grp.add_argument("--backbone", default="molformer", choices=["molformer", "grover", "unimol", "unimol2", "smited", "mhgged"])
    mve_grp.add_argument("--backbones", nargs="+", default=["molformer"], choices=["molformer", "grover", "unimol", "unimol2", "smited", "mhgged"])
    mve_grp.add_argument("--parallel-predict", action="store_true",
                          help="Delegate per-round pool prediction to a persistent pool of single-GPU "
                               "workers (predict_pool_shard_worker.py) instead of predicting in-process. "
                               "Workers must already be running against the same --coord-dir before this "
                               "starts (see submit_ampc_al_predict_workers_h100single.sh).")
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
