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
        "embed_dir": AMPC_ROOT / "embeddings",
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


# ==============================================================================
# MVE ACTIVE LEARNING LOOP (embedding-based surrogates)
# ==============================================================================

class MVEExplorer:
    def __init__(
        self, emb_dict, pool_smiles, oracle, surrogate_type, backbone, acq,
        init_size=8417, batch_size=8417, n_rounds=5, topk=1000,
        run_dir=None, seed=42,
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

        self.model = build_mve(
            surrogate_type=surrogate_type, backbone=backbone, emb_dict=emb_dict,
            pool_smiles=pool_smiles, dataset_name=DATASET,
        )

        self.acq_fn = get_metric(acq)
        self.acq_name = acq
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

            self.model.train(xs, ys)

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

    mve_grp = p.add_argument_group("MVE surrogate options (--mode mve)")
    mve_grp.add_argument(
        "--surrogate", default="ft_molformer_single",
        choices=["single", "ft_molformer_single", "ensemble", "ft_fusion", "learned"],
        help="ft_molformer_single = method [2] (MoLFormer, fine-tuned each round); "
             "ensemble = method [3] (\"our fusion model\", EnsembleFusionSurrogate); "
             "ft_fusion = grover+molformer fine-tuned jointly each round, unimol frozen "
             "(FTFusionSurrogate; unimol excluded because its conformer cache can't be "
             "made memory-safe the same way -- see surrogates.py docstring); "
             "learned = 3 frozen per-backbone models combined by a RidgeCV meta-learner "
             "fit on held-out backbone predictions each round (LearnedFusionSurrogate) "
             "-- no fine-tuning, cheap, comparable cost to ensemble",
    )
    mve_grp.add_argument("--backbone", default="molformer", choices=["molformer", "grover", "unimol"])
    mve_grp.add_argument("--backbones", nargs="+", default=["molformer"], choices=["molformer", "grover", "unimol"])

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
        keep_idx = [i for i, s in enumerate(pool_smiles) if s in usable_set]
        emb_dict = {bb: emb[np.array(keep_idx)] for bb, emb in emb_dict.items()}
        pool_smiles = [pool_smiles[i] for i in keep_idx]
        print(f"[pool] {len(pool_smiles):,} molecules with embeddings + oracle scores")

        suffix = f"mve_{args.surrogate}_{'_'.join(backbones)}_{args.acq}_init{args.init_size}"
        run_dir = Path(args.run_dir) if args.run_dir else RUNS_DIR / suffix

        explorer = MVEExplorer(
            emb_dict=emb_dict, pool_smiles=pool_smiles, oracle=oracle,
            surrogate_type=args.surrogate, backbone=args.backbone, acq=args.acq,
            init_size=args.init_size, batch_size=args.batch_size, n_rounds=args.n_rounds,
            topk=args.topk, run_dir=run_dir, seed=args.seed,
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
