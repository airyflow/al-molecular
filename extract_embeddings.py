#!/usr/bin/env python3
"""
Embedding extraction using MUBen backbones, for the EnamineHTS pool
(2,141,515 SMILES, molpal/libraries/EnamineHTS.csv.gz).

Adapted from FusionAL/extract_embeddings.py -- same monkey-patch approach
(inject the molpal library's SMILES directly into MUBen's Dataset object),
same three backbones (GROVER, UniMol, MoLFormer). Needed here for the
"our fusion model" (EnsembleFusionSurrogate: frozen embeddings from all
three backbones, combined via rank-disagreement UCB) and as the starting
point for the MoLFormer-finetune baseline (which fine-tunes from these
frozen embeddings during AL rounds).

Output: results/embed/EnamineHTS/{backbone}_embeddings.npz
        ('embeddings' (N,D) float32, 'smiles' (N,) str -- row alignment
        is self-documenting, verified by EmbeddingFeaturizer.load() on read).
"""

import datetime
import os
import sys
import time
import types
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).parent
MOLPAL_LIB = ROOT / "molpal" / "libraries"
MODEL_ZOO = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "embed"

_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")

DATASET = "EnamineHTS"
OUT_DIR = OUTPUT_DIR / DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SMILES SOURCE -- read directly from molpal library
# ==============================================================================

def load_molpal_smiles(dataset: str, limit: int = None) -> list:
    lib_path = MOLPAL_LIB / f"{dataset}.csv.gz"
    assert lib_path.exists(), f"Library not found: {lib_path}"

    df = pd.read_csv(lib_path)
    df.columns = df.columns.str.strip().str.lower()
    smiles_col = next(c for c in df.columns if "smiles" in c)
    smiles = df[smiles_col].dropna().tolist()
    if limit is not None:
        smiles = smiles[:limit]

    print(f"[library] {lib_path.name}: {len(smiles):,} SMILES loaded")
    return smiles


# ==============================================================================
# MONKEY-PATCH -- replace MUBen's read_csv with a molpal-aware version
# ==============================================================================

def _make_molpal_reader(smiles_list: list):
    def molpal_read_csv(self, data_dir: str, partition: str):
        n = len(smiles_list)
        self._smiles = smiles_list
        self._lbs = np.zeros((n, 1), dtype=np.float32)
        self._masks = np.ones((n, 1), dtype=np.float32)
        self._ori_ids = None
        print(f"[dataset] Injected {n:,} SMILES from molpal library (partition='{partition}' ignored)")
        return self

    return molpal_read_csv


def patch_muben_dataset(smiles_list: list):
    import muben.dataset.dataset as _ds_module
    _ds_module.Dataset.read_csv = _make_molpal_reader(smiles_list)


# ==============================================================================
# SHARED CONFIG
# ==============================================================================

class MubenRuntimeConfig:
    def __init__(self, model_name, feature_type="none", checkpoint_path=""):
        self.data_dir = str(ROOT / "muben" / "data" / "files" / DATASET)
        self.model_name = model_name
        self.feature_type = feature_type
        self.checkpoint_path = str(checkpoint_path)
        self.unimol_feature_dir = str(OUT_DIR)
        self.num_preprocess_workers = 4

        self.ignore_preprocessed_dataset = False
        self.disable_dataset_saving = False
        self.disable_checkpoint_loading = False

        # GROVER
        self.hidden_size = 128
        self.dropout = 0.1
        self.bias = False
        self.num_mt_block = 1
        self.num_attn_head = 4
        self.embedding_output_type = "both"

        # Uni-Mol
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

        # MoLFormer
        self.pretrained_model_name_or_path = "ibm-research/MoLFormer-XL-both-10pct"
        self.tokenizer_trust_remote_code = True

        # Task
        self.uncertainty_method = "none"
        self.task_type = "regression"
        self.bbp_prior_sigma = 0.5
        self.n_lbs = 1
        self.n_tasks = 1
        self.activation = "ReLU"
        self.ffn_num_layers = 2
        self.ffn_hidden_size = 128


# ==============================================================================
# SAVE
# ==============================================================================

def save_embeddings(name: str, matrix: np.ndarray, smiles: list) -> dict:
    if matrix.shape[0] != len(smiles):
        raise RuntimeError(
            f"{name}: embeddings matrix has {matrix.shape[0]:,} rows but smiles list has "
            f"{len(smiles):,} entries -- refusing to save a misaligned .npz. This happens when "
            f"muben's own dataset cache (keyed only by dataset name, not molecule count) silently "
            f"ignores --limit and returns a differently-sized dataset than the SMILES list passed in."
        )
    out_npz = OUT_DIR / f"{name}_embeddings.npz"
    np.savez(out_npz, embeddings=matrix, smiles=np.array(smiles))

    print(f"[saved] {out_npz.name}  shape={matrix.shape}")
    return {
        "n_molecules": matrix.shape[0],
        "embedding_dim": matrix.shape[1],
        "output_path": str(out_npz),
    }


# ==============================================================================
# EXTRACTORS
# ==============================================================================

def extract_grover(smiles: list):
    print("\n>>> GROVER 2D Graph Representations...")

    from muben.dataset import DatasetGrover
    from muben.dataset.dataset_grover import CollatorGrover
    from muben.model import GROVER

    config = MubenRuntimeConfig(model_name="grover")
    dataset = DatasetGrover()
    dataset.prepare(config=config, partition="train")

    collator = CollatorGrover(config)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    ckpt_path = MODEL_ZOO / "grover" / "grover_base.pt"
    assert ckpt_path.exists(), f"Missing: {ckpt_path}"

    model_cfg = MubenRuntimeConfig(model_name="grover", checkpoint_path=ckpt_path)
    model = GROVER(model_cfg).to(DEVICE)
    model.eval()

    embeddings = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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

    matrix = np.vstack(embeddings)
    return save_embeddings("grover", matrix, smiles)


def extract_unimol(smiles: list):
    print("\n>>> Uni-Mol 3D Conformational Representations...")

    from muben.dataset import DatasetUniMol
    from muben.dataset.dataset_unimol import CollatorUniMol
    from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
    from muben.model.unimol.unimol import UniMol

    unimol_ckpt = MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt"
    config = MubenRuntimeConfig(model_name="unimol", feature_type="unimol", checkpoint_path=unimol_ckpt)

    dataset = DatasetUniMol()
    dataset.prepare(config=config, partition="train")

    unimol_dict = DictionaryUniMol.load()
    unimol_dict.add_symbol("[MASK]", is_special=True)

    collator = CollatorUniMol(config, unimol_dict)
    pad_idx = unimol_dict.pad()
    collator._atom_pad_idx = pad_idx
    collator.pad_idx = pad_idx
    collator.atom_pad_idx = pad_idx

    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
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
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == "cuda")):
                feat = model.get_embeddings(batch)
            embeddings.append(feat.float().cpu().numpy())

    matrix = np.vstack(embeddings)
    return save_embeddings("unimol", matrix, smiles)


def extract_molformer(smiles: list):
    print("\n>>> MoLFormer 1D Chemical Language Representations...")

    from transformers import AutoModel, AutoTokenizer

    molformer_path = str(MODEL_ZOO / "molformer")
    tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(molformer_path, trust_remote_code=True).to(DEVICE)
    model.eval()

    embeddings = []
    batch_size = 256
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for i in range(0, len(smiles), batch_size):
        batch_smi = smiles[i : i + batch_size]
        inputs = tokenizer(batch_smi, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=(DEVICE.type == "cuda")):
                outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output.float()
            else:
                emb = outputs.last_hidden_state.mean(dim=1).float()
        embeddings.append(emb.cpu().numpy())

    matrix = np.vstack(embeddings)
    return save_embeddings("molformer", matrix, smiles)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--backbone", default="all", choices=["all", "grover", "unimol", "molformer"])
    _p.add_argument("--limit", type=int, default=None, help="Only extract the first N pool molecules (smoke testing)")
    _cli = _p.parse_args()

    smiles = load_molpal_smiles(DATASET, limit=_cli.limit)

    import muben.dataset.dataset
    patch_muben_dataset(smiles)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _all_extractors = [
        ("grover", extract_grover),
        ("unimol", extract_unimol),
        ("molformer", extract_molformer),
    ]
    extractors = _all_extractors if _cli.backbone == "all" else [
        (bb, fn) for bb, fn in _all_extractors if bb == _cli.backbone
    ]

    records = []
    total_t0 = time.perf_counter()

    for backbone, fn in extractors:
        t0 = time.perf_counter()
        meta = fn(smiles)
        elapsed = time.perf_counter() - t0
        records.append({**meta, "backbone": backbone, "elapsed_s": round(elapsed, 1)})
        print(f"  [{backbone}] done in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n[done] Total extraction time: {total_elapsed:.1f}s")

    log_path = OUT_DIR.parent / "extraction_log.txt"
    with open(log_path, "a") as f:
        f.write(f"\n=== {run_ts}  dataset={DATASET}  device={device_str}  limit={_cli.limit} ===\n")
        for r in records:
            f.write(f"  {r['backbone']:<12}  {r['n_molecules']:>9,} mol  {r['embedding_dim']:>5}d  {r['elapsed_s']:>8.1f}s  ->  {r['output_path']}\n")

    print(f"[log]  {log_path}")
    print(f"[done] Embeddings in {OUT_DIR}")
