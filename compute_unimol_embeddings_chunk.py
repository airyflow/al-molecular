#!/usr/bin/env python3
"""Stage 2 of the split Uni-Mol pipeline, chunked: compute embeddings
directly from one conformer shard produced by generate_unimol_conformers_chunk.py.

Ported from al-eval-framework's src/representations/compute_unimol_embeddings_chunk.py,
adapted for molpal-fusion-hts: uses a self-contained config class (not
extract_embeddings.py's MubenRuntimeConfig, which hardcodes
DATASET = "EnamineHTS" at module level and would silently point
unimol_feature_dir at the wrong dataset for AmpC), and writes into a shared
memmap (see shared_embedding_store.py) instead of a per-chunk .npz.

No conformer-shard merge is needed because embedding computation only
needs, for each molecule, its SMILES plus its conformer (atoms/coordinates)
in matching order -- both are already available per-chunk: read_smiles_chunk()
gives the same SMILES slice generate_unimol_conformers_chunk.py used for
this chunk-id, and that chunk's shard (chunk_{id:05d}.lmdb) holds exactly
those molecules' conformers in the same order (global LMDB keys sort
correctly within a contiguous chunk range).

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks, so no coordination/race condition is
possible during this parallel compute phase. Once every chunk finishes,
run stitch_embedding_chunks.py once (a separate, sequential job) to copy
all chunk files into the final shared (N, D) .npy that
EmbeddingFeaturizer.load() (molpal/featurizer.py) expects.

Usage
-----
python compute_unimol_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --shards-dir /path/to/_unimol_conformers/_shards \\
    --chunk-id 0 --num-chunks 70 \\
    --chunks-dir /path/to/_unimol_embed_chunks
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent
MODEL_ZOO = ROOT / "models"

_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))

from generate_unimol_conformers_chunk import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


class _UniMolConfig:
    """Minimal attribute bag for DatasetUniMol/CollatorUniMol/UniMol model
    construction -- same field values already validated in
    backbone_finetuner.py's _MubenConfig (used throughout this repo's AL
    pipeline), kept self-contained here rather than imported to avoid this
    script depending on backbone_finetuner.py's AL-specific machinery."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = str(checkpoint_path)
        self.disable_checkpoint_loading = False
        self.feature_type = "unimol"
        self.task_type = "regression"
        self.uncertainty_method = "none"
        self.bbp_prior_sigma = 0.5
        self.n_lbs = 1
        self.n_tasks = 1
        self.dropout = 0.0
        self.max_atoms = 64
        self.max_seq_len = 80
        self.only_polar_hydrogens = False
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.encoder_embed_dim = 512
        self.encoder_layers = 15
        self.encoder_attention_heads = 64
        self.encoder_ffn_embed_dim = 2048
        self.activation_fn = "gelu"
        self.pooler_stride = 1
        self.pooler_dropout = 0.0
        self.emb_dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.delta_pair_repr_norm_loss = -1
        self.masked_coord_loss = 0.0
        self.masked_dist_loss = 0.0
        self.masked_type_loss = 0.0
        self.pooler_activation_fn = "Tanh"


def _verify_alignment(chunk_smiles: list, atoms: list, sample_size: int = 20, seed: int = 0) -> None:
    """A count match (len(atoms) == len(chunk_smiles)) does not prove
    molecule i's conformer actually came from chunk_smiles[i] -- if
    generation and this script were ever run with a different
    --num-chunks/--total-count, _chunk_bounds() would silently compute
    different global-index boundaries, potentially preserving the count
    while pairing every molecule with the wrong conformer. Re-derive each
    sampled molecule's expected all-hydrogen atom count from its own
    SMILES and compare against what the shard actually stored at that
    position -- a real content check, not just a count."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(chunk_smiles), size=min(sample_size, len(chunk_smiles)), replace=False)

    mismatches = []
    for i in idxs:
        smi = chunk_smiles[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        expected_n_atoms = AllChem.AddHs(mol).GetNumAtoms()
        actual_n_atoms = len(atoms[i])
        if expected_n_atoms != actual_n_atoms:
            mismatches.append((i, smi, expected_n_atoms, actual_n_atoms))

    if mismatches:
        detail = "; ".join(f"index {i}: {smi!r} expected {exp} atoms, shard has {act}" for i, smi, exp, act in mismatches[:5])
        raise RuntimeError(
            f"Conformer/SMILES alignment check FAILED for {len(mismatches)}/{len(idxs)} sampled molecules "
            f"({detail}) -- do not trust these embeddings. Almost certainly a --num-chunks/--total-count "
            f"mismatch between the generate_unimol_conformers_chunk.py run that produced this shard and "
            f"this invocation."
        )


def load_chunk_dataset(chunk_smiles: list, shard_path: Path, config):
    """Builds a DatasetUniMol directly from one conformer shard, bypassing
    DatasetUniMol.prepare()'s directory/partition-based file lookup (which
    expects one merged {partition}.lmdb, not many per-chunk shards)."""
    from muben.dataset import DatasetUniMol
    from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
    from muben.dataset.dataset_unimol.process import ProcessingPipeline
    from muben.utils.io import load_lmdb

    dictionary = DictionaryUniMol.load()
    dictionary.add_symbol("[MASK]", is_special=True)

    dataset = DatasetUniMol()
    dataset._partition = "test"
    dataset.processing_pipeline = ProcessingPipeline(
        dictionary=dictionary, max_atoms=config.max_atoms, max_seq_len=config.max_seq_len,
        remove_hydrogen_flag=config.remove_hydrogen, remove_polar_hydrogen_flag=config.remove_polar_hydrogen,
    )
    # "training" routes through process_training() -> conformer_sampling(), which
    # asserts exactly 11 stored conformers (the original UniMol paper's
    # training-time augmentation scheme). Stage 1 (generate_unimol_conformers_chunk.py)
    # deliberately stores only n_conformer=1 per molecule to save disk, so this
    # must use "inference" instead -- process_inference() loops over however many
    # conformers are actually present rather than asserting a fixed count.
    dataset.set_processor_variant("inference")

    n = len(chunk_smiles)
    dataset._smiles = chunk_smiles
    dataset._lbs = np.zeros((n, 1), dtype=np.float32)
    dataset._masks = np.ones((n, 1), dtype=np.float32)
    dataset._ori_ids = None
    dataset._atoms, dataset._coordinates = load_lmdb(str(shard_path), ["atoms", "coordinates"])
    assert len(dataset._atoms) == n, (
        f"Shard {shard_path} has {len(dataset._atoms)} records but the chunk has {n} SMILES -- "
        f"mismatched chunk boundaries (wrong --num-chunks/--total-count?) or an incomplete shard."
    )
    _verify_alignment(chunk_smiles, dataset._atoms)

    return dataset, dictionary


def compute_embeddings_for_chunk(
    chunk_smiles: list, shard_path: Path, checkpoint_path: Path,
    batch_size: int = 256, num_workers: int = 4,
) -> np.ndarray:
    from muben.dataset.dataset_unimol import CollatorUniMol
    from muben.model.unimol.unimol import UniMol

    config = _UniMolConfig(checkpoint_path=checkpoint_path)
    dataset, unimol_dict = load_chunk_dataset(chunk_smiles, shard_path, config)

    collator = CollatorUniMol(config, unimol_dict)
    pad_idx = unimol_dict.pad()
    collator._atom_pad_idx = pad_idx
    collator.pad_idx = pad_idx
    collator.atom_pad_idx = pad_idx

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=num_workers, pin_memory=True,
    )

    model = UniMol(config=config, dictionary=unimol_dict).to(DEVICE)

    def _get_embeddings(self, batch):
        src_tokens, src_distance, src_edge_type = batch.atoms, batch.distances, batch.edge_types
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(src_tokens)
        n_node = src_distance.size(-1)
        gbf_feat = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feat)
        attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        encoder_rep, _, _, _, _ = self.encoder(x, padding_mask=padding_mask, attn_mask=attn_bias)
        return self.hidden_layer(encoder_rep[:, 0, :])

    model.get_embeddings = types.MethodType(_get_embeddings, model)
    model.eval()

    embeddings = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else dict(device_type="cpu", enabled=False)

    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            with torch.autocast(**autocast_kwargs):
                feat = model.get_embeddings(batch)
            embeddings.append(feat.float().cpu().numpy())

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--shards-dir", required=True, help="Directory containing chunk_XXXXX.lmdb shards from generate_unimol_conformers_chunk.py")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent unimol_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "unimol", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    shard_path = Path(args.shards_dir) / f"chunk_{args.chunk_id:05d}.lmdb"
    if not shard_path.exists():
        raise SystemExit(f"Shard not found: {shard_path}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt"

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(
        chunk_smiles, shard_path, checkpoint_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "unimol", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
