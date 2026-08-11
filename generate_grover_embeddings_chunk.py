#!/usr/bin/env python3
"""
Sharded GROVER embedding extraction for the EnamineHTS pool.

Why this exists: a single-process run of extract_embeddings.py --backbone
grover against the full 2,141,500-molecule pool measured ~33 min just for
RDKit graph preprocessing (vs. ~14 min extrapolated from a 3000-molecule
test) and grew to 260+ GB RSS with no clear completion time for the
subsequent DataLoader/forward pass -- torch_geometric silently falls back
to slow, memory-hungry pure-Python graph ops here because pyg_lib /
torch_sparse's accelerated kernels can't load on this host (GLIBC_2.29 too
old for the precompiled wheels). Chunking bounds each process's working set
to one chunk instead of the full pool, and lets chunks run in parallel
across BigRed200 CPU nodes instead of serially on one workstation.

Each chunk gets its own muben dataset_name (EnamineHTS_grover_chunk_NNNNN)
-- NOT the shared "EnamineHTS" name extract_embeddings.py uses -- because
muben's Dataset.prepare() caches to a fixed path keyed only by
(data_dir, dataset_name), with no chunk-awareness. Concurrent SLURM array
tasks sharing one dataset_name would race on the same cache file; a unique
name per chunk avoids that entirely.

Usage
-----
python generate_grover_embeddings_chunk.py \\
    --smiles-file molpal/libraries/EnamineHTS.csv.gz --total-count 2141500 \\
    --chunk-id 0 --num-chunks 50 --out-dir results/embed/EnamineHTS/_grover_chunks

Output: {out_dir}/grover_embeddings_chunk_{chunk_id:05d}.npz
        ('embeddings' (n,1600) float32, 'smiles' (n,) str)
Concatenate all chunks with concat_grover_chunks.py once done.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent
MODEL_ZOO = ROOT / "models"

_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def _chunk_bounds(total: int, chunk_id: int, num_chunks: int):
    """Deterministic, contiguous [start, end) split of [0, total) into
    num_chunks pieces -- must be called with identical total/num_chunks
    across every invocation touching this pool, or boundaries silently
    shift between runs."""
    base, rem = divmod(total, num_chunks)
    start = chunk_id * base + min(chunk_id, rem)
    end = start + base + (1 if chunk_id < rem else 0)
    return start, end


def count_lines(path: str) -> int:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    smi_col = next(c for c in df.columns if "smiles" in c)
    return len(df[smi_col].dropna())


def read_smiles_chunk(path: str, start: int, end: int) -> list:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    smi_col = next(c for c in df.columns if "smiles" in c)
    smiles = df[smi_col].dropna().tolist()
    return smiles[start:end]


def _make_molpal_reader(smiles_list: list):
    def molpal_read_csv(self, data_dir: str, partition: str):
        n = len(smiles_list)
        self._smiles = smiles_list
        self._lbs = np.zeros((n, 1), dtype=np.float32)
        self._masks = np.ones((n, 1), dtype=np.float32)
        self._ori_ids = None
        return self

    return molpal_read_csv


class MubenRuntimeConfig:
    def __init__(self, dataset_name, checkpoint_path=""):
        self.data_dir = str(ROOT / "muben" / "data" / "files" / dataset_name)
        self.model_name = "grover"
        self.feature_type = "none"
        self.checkpoint_path = str(checkpoint_path)
        self.unimol_feature_dir = str(ROOT / "results" / "embed" / dataset_name)
        self.num_preprocess_workers = 4
        self.ignore_preprocessed_dataset = False
        self.disable_dataset_saving = False
        self.disable_checkpoint_loading = False

        self.hidden_size = 128
        self.dropout = 0.1
        self.bias = False
        self.num_mt_block = 1
        self.num_attn_head = 4
        self.embedding_output_type = "both"

        self.uncertainty_method = "none"
        self.task_type = "regression"
        self.bbp_prior_sigma = 0.5
        self.n_lbs = 1
        self.n_tasks = 1
        self.activation = "ReLU"
        self.ffn_num_layers = 2
        self.ffn_hidden_size = 128


def extract_grover_chunk(chunk_smiles: list, dataset_name: str) -> np.ndarray:
    from muben.dataset import DatasetGrover
    import muben.dataset.dataset as _ds_module
    from muben.dataset.dataset_grover import CollatorGrover
    from muben.model import GROVER

    _ds_module.Dataset.read_csv = _make_molpal_reader(chunk_smiles)

    config = MubenRuntimeConfig(dataset_name=dataset_name)
    dataset = DatasetGrover()
    dataset.prepare(config=config, partition="train")

    collator = CollatorGrover(config)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True,
    )

    ckpt_path = MODEL_ZOO / "grover" / "grover_base.pt"
    model_cfg = MubenRuntimeConfig(dataset_name=dataset_name, checkpoint_path=ckpt_path)
    model = GROVER(model_cfg).to(DEVICE)
    model.eval()

    embeddings = []
    amp_dtype = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            components = batch.molecule_graphs.components
            _, _, _, _, _, a_scope, _, _ = components

            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == "cuda")):
                output = model.grover(components)
                mol_from_bond = model.readout(output["atom_from_bond"], a_scope)
                mol_from_atom = model.readout(output["atom_from_atom"], a_scope)
                combined = torch.cat([mol_from_bond, mol_from_atom], dim=1)

            embeddings.append(combined.float().cpu().numpy())

    return np.vstack(embeddings)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles-file", required=True)
    p.add_argument("--total-count", type=int, default=None)
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--num-chunks", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)
    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grover_embeddings_chunk_{args.chunk_id:05d}.npz"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))")

    dataset_name = f"EnamineHTS_grover_chunk_{args.chunk_id:05d}"
    t0 = time.perf_counter()
    matrix = extract_grover_chunk(chunk_smiles, dataset_name)
    elapsed = time.perf_counter() - t0

    np.savez(out_path, embeddings=matrix, smiles=np.array(chunk_smiles))
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
