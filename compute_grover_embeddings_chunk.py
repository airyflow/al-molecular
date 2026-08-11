#!/usr/bin/env python3
"""GROVER embedding extraction, chunked, single-stage.

Unlike Uni-Mol, GROVER does NOT get a separate CPU preprocessing stage that
persists its intermediate representation to disk. The two are
architecturally different problems: Uni-Mol's conformer generation is
expensive (~400ms/molecule measured), so paying disk to avoid regenerating
it is a clear win. GROVER's graph construction from a raw SMILES string is
a 2D operation (RDKit mol parsing + feature extraction, no conformers)
measured at ~3ms/molecule -- cheap enough that persisting it isn't clearly
worth the disk: a pickled MolGraphAttrs object is comparable in size to
chemprop's own graph cache (measured elsewhere at 153.4 KB/molecule, a
different codebase's graph representation but a structurally similar
object) -- ~12-15TB for the full 99.5M-molecule AmpC pool if persisted.
So this script builds each molecule's graph AND runs it through GROVER in
the same process, per chunk, discarding the graph immediately after -- the
same on-the-fly pattern already proven correct in backbone_finetuner.py's
FTFusionSurrogate path, just sharded across many parallel CPU tasks here
instead of used inside one active-learning round.

CPU-primary by design (matches "hard to get GPU" on BigRed200's scarce GPU
partition vs. its much larger CPU allocation), but DEVICE falls back to
CUDA automatically if available.

Output: writes this chunk's embeddings to its own INDEPENDENT .npy file
under --chunks-dir (see shared_embedding_store.py's write_chunk_file()) --
no shared state between chunk tasks, so no coordination/race condition is
possible during this parallel compute phase. Once every chunk finishes,
run stitch_embedding_chunks.py once (a separate, sequential job) to copy
all chunk files into the final shared (N, D) .npy that
EmbeddingFeaturizer.load() (molpal/featurizer.py) expects.

Usage
-----
python compute_grover_embeddings_chunk.py \\
    --smiles-file /path/to/ampc_smiles.txt --total-count 99459561 \\
    --chunk-id 0 --num-chunks 150 \\
    --chunks-dir /path/to/_grover_chunks
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
from pathlib import Path

import numpy as np
import torch

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


class _GroverConfig:
    """Minimal attribute bag for GROVER model/collator construction -- same
    field values already validated in backbone_finetuner.py's _MubenConfig
    (used throughout this repo's AL pipeline), kept self-contained here."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = str(checkpoint_path)
        self.disable_checkpoint_loading = False
        self.hidden_size = 128
        self.dropout = 0.0
        self.bias = False
        self.num_mt_block = 1
        self.num_attn_head = 4
        self.embedding_output_type = "both"
        self.ffn_num_layers = 2
        self.ffn_hidden_size = 128
        self.activation = "ReLU"
        self.uncertainty_method = "none"
        self.task_type = "regression"
        self.bbp_prior_sigma = 0.5
        self.n_lbs = 1
        self.n_tasks = 1


def compute_embeddings_for_chunk(
    chunk_smiles: list, checkpoint_path: Path, batch_size: int = 128,
) -> np.ndarray:
    from muben.dataset.dataset_grover.dataset import DatasetGrover
    from muben.dataset.dataset_grover import CollatorGrover
    from muben.model import GROVER

    config = _GroverConfig(checkpoint_path=checkpoint_path)
    collator = CollatorGrover(config)
    model = GROVER(config).to(DEVICE)
    model.eval()

    def _instance(smi: str) -> dict:
        return {
            "molecule_graphs": DatasetGrover.get_mol_attr(smi),
            "lbs": np.zeros(1, dtype=np.float32),
            "masks": np.ones(1, dtype=np.float32),
        }

    amp_dtype = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if DEVICE.type == "cuda" else dict(device_type="cpu", enabled=False)

    parts = []
    with torch.no_grad():
        for i in range(0, len(chunk_smiles), batch_size):
            smi_batch = chunk_smiles[i : i + batch_size]
            items = [_instance(s) for s in smi_batch]
            batch = collator(items)
            batch.to(DEVICE)

            components = batch.molecule_graphs.components
            _, _, _, _, _, a_scope, _, _ = components
            with torch.autocast(**autocast_kwargs):
                out = model.grover(components)
                mol_from_bond = model.readout(out["atom_from_bond"], a_scope)
                mol_from_atom = model.readout(out["atom_from_atom"], a_scope)
                combined = torch.cat([mol_from_bond, mol_from_atom], dim=1)

            parts.append(combined.float().cpu().numpy())

    return np.vstack(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunks-dir", required=True, help="Directory to write this chunk's independent grover_embeddings_chunk_NNNNN.npy into")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)

    out_path = chunk_file_path(args.chunks_dir, "grover", args.chunk_id)
    if out_path.exists():
        print(f"[skip] {out_path} exists -- chunk {args.chunk_id} already written")
        return

    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else MODEL_ZOO / "grover" / "grover_base.pt"

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))  device={DEVICE}")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(chunk_smiles, checkpoint_path, batch_size=args.batch_size)
    elapsed = time.perf_counter() - t0

    write_chunk_file(args.chunks_dir, "grover", args.chunk_id, matrix)
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s "
          f"({elapsed/max(1,matrix.shape[0])*1000:.2f} ms/mol) -> {out_path}")


if __name__ == "__main__":
    main()
