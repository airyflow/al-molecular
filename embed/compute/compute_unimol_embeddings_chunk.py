#!/usr/bin/env python3
"""Stage 2 of the split Uni-Mol pipeline, chunked: compute embeddings
directly from one conformer chunk file produced by generate_unimol_conformers_chunk.py.

Ported from al-eval-framework's src/representations/compute_unimol_embeddings_chunk.py,
adapted for molpal-fusion-hts: uses a self-contained config class, and
writes into a shared memmap (see shared_embedding_store.py) instead of a
per-chunk .npz.

Uses the standalone `unimol1` package (unimol1/, repo root) -- a
muben-free port of muben's own Uni-Mol v1 model/dataset code, verified
bit-exact against it (see unimol1/model.py's docstring and
unimol1/tests/). This script no longer imports anything from muben.

No conformer-chunk merge is needed because embedding computation only
needs, for each molecule, its SMILES plus its conformer (atoms/coordinates)
in matching order -- both are already available per-chunk: read_smiles_chunk()
gives the same SMILES slice generate_unimol_conformers_chunk.py used for
this chunk-id, and that chunk's file (chunk_{id:05d}.lmdb) holds exactly
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
    --conformer-chunks-dir /path/to/_unimol_conformers/_chunks \\
    --chunk-id 0 --num-chunks 70 \\
    --chunks-dir /path/to/_unimol_embed_chunks
"""
from __future__ import annotations

import argparse
import os
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (this file lives in embed/compute/)
MODEL_ZOO = ROOT / "models"

# smiles_chunking.py / shared_embedding_store.py / unimol1/ all live at the
# repo root, not next to this file -- put ROOT on sys.path so these bare
# imports still resolve regardless of where this script itself was invoked from.
sys.path.insert(0, str(ROOT))
from smiles_chunking import _chunk_bounds, count_lines, read_smiles_chunk
from shared_embedding_store import chunk_file_path, write_chunk_file
from unimol1 import UniMolConfig, build_model_from_checkpoint, load_production_dictionary
from unimol1.data import CollatorUniMol, ConformerDataset, ProcessingPipeline

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def _load_conformer_chunk(chunk_path: Path) -> tuple[list, list]:
    """Reads the LMDB chunk file Stage 1 wrote -- one pickled
    {"atoms": [...], "coordinates": [...]} record per molecule, keyed by
    zero-padded global index. LMDB's cursor iterates keys in
    byte-lexicographic order, which (thanks to the zero-padding) is also
    correct numeric order, matching chunk_smiles' positional order.
    Replaces muben's `utils.io.load_lmdb` -- this repo's own Stage 1
    writes a fixed 2-key record shape, so a generic multi-key reader isn't
    needed."""
    env = lmdb.open(str(chunk_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=256)
    atoms_list, coordinates_list = [], []
    with env.begin() as txn:
        for _, value in txn.cursor():
            record = pickle.loads(value)
            atoms_list.append(record["atoms"])
            coordinates_list.append(record["coordinates"])
    env.close()
    return atoms_list, coordinates_list


def _verify_alignment(chunk_smiles: list, atoms: list, sample_size: int = 20, seed: int = 0) -> None:
    """A count match (len(atoms) == len(chunk_smiles)) does not prove
    molecule i's conformer actually came from chunk_smiles[i] -- if
    generation and this script were ever run with a different
    --num-chunks/--total-count, _chunk_bounds() would silently compute
    different global-index boundaries, potentially preserving the count
    while pairing every molecule with the wrong conformer. Re-derive each
    sampled molecule's expected all-hydrogen atom count from its own
    SMILES and compare against what the chunk file actually stored at that
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
        detail = "; ".join(f"index {i}: {smi!r} expected {exp} atoms, chunk file has {act}" for i, smi, exp, act in mismatches[:5])
        raise RuntimeError(
            f"Conformer/SMILES alignment check FAILED for {len(mismatches)}/{len(idxs)} sampled molecules "
            f"({detail}) -- do not trust these embeddings. Almost certainly a --num-chunks/--total-count "
            f"mismatch between the generate_unimol_conformers_chunk.py run that produced this chunk file and "
            f"this invocation."
        )


def compute_embeddings_for_chunk(
    chunk_smiles: list, chunk_path: Path, checkpoint_path: Path,
    batch_size: int = 256, num_workers: int = 4,
) -> np.ndarray:
    config = UniMolConfig(checkpoint_path=str(checkpoint_path))
    dictionary = load_production_dictionary()

    atoms_list, coordinates_list = _load_conformer_chunk(chunk_path)
    assert len(atoms_list) == len(chunk_smiles), (
        f"Chunk file {chunk_path} has {len(atoms_list)} records but the chunk has {len(chunk_smiles)} SMILES -- "
        f"mismatched chunk boundaries (wrong --num-chunks/--total-count?) or an incomplete chunk file."
    )
    _verify_alignment(chunk_smiles, atoms_list)

    # generate_unimol_conformers_chunk.py's smiles_to_coords(n_conformer=1) ALWAYS
    # appends a 2D-fallback conformer in ADDITION to the 1 requested 3D conformer,
    # so each chunk file's record actually stores 2 conformers, not 1 -- keep only
    # the first (real, or 2D-fallback-on-failure) conformer per molecule, index 0
    # is always the primary one Stage 1 intended (verified directly in the muben-
    # based version of this script: omitting this fix silently doubled every
    # embedding row).
    coordinates_list = [c[0] for c in coordinates_list]

    pipeline = ProcessingPipeline(
        dictionary=dictionary, max_atoms=config.max_atoms, max_seq_len=config.max_seq_len,
        remove_hydrogen_flag=config.remove_hydrogen, remove_polar_hydrogen_flag=config.remove_polar_hydrogen,
    )
    dataset = ConformerDataset(atoms_list, coordinates_list, pipeline)
    collator = CollatorUniMol(atom_pad_idx=dictionary.pad())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=num_workers, pin_memory=True,
    )

    # build_model_from_checkpoint() calls torch.manual_seed(config.construction_seed)
    # immediately before constructing the model -- see unimol1/checkpoint.py and
    # unimol1/model.py's docstrings for why every chunk task needs this same seed:
    # hidden_layer (the actual embedding projection) isn't covered by the
    # checkpoint, so without a fixed seed each chunk's projection would be a
    # different, mutually-incomparable random space.
    model = build_model_from_checkpoint(config=config, dictionary=dictionary).to(DEVICE)

    embeddings = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else dict(device_type="cpu", enabled=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            with torch.autocast(**autocast_kwargs):
                emb = model(batch)
            embeddings.append(emb.float().cpu().numpy())

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--conformer-chunks-dir", required=True, help="Directory containing chunk_XXXXX.lmdb files from generate_unimol_conformers_chunk.py")
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

    chunk_path = Path(args.conformer_chunks_dir) / f"chunk_{args.chunk_id:05d}.lmdb"
    if not chunk_path.exists():
        raise SystemExit(f"Conformer chunk file not found: {chunk_path}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt"

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(
        chunk_smiles, chunk_path, checkpoint_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "unimol", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
