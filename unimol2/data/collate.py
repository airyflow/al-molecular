"""Per-molecule feature assembly + batch collation for Uni-Mol2.

Ports the remaining pipeline steps from `tasks/unimol_finetune.py::
UniMolFinetuneTask.load_dataset` (the finetune/eval task, chosen as the
reference over the pretraining task since it has no masking/noise
augmentation -- deterministic given a conformer, matching what frozen
embedding extraction needs) and `unimol2_dataset.py::
Unimol2FinetuneFeatureDataset.collater`:

  1. Conformer generation (conformer.py) -- WITH hydrogens.
  2. Remove hydrogens (`RemoveHydrogenDataset`, `remove_hydrogen=True` is
     the literal argument passed in the finetune task).
  3. Crop to <=256 atoms if exceeded (`CroppingDataset`, max_atoms=256 --
     effectively a no-op for drug-like molecules; implemented as a safety
     net, not because it's expected to fire).
  4. Center coordinates (`NormalizeDataset`, `normalize_coord=True`).
  5. Tokenize (tokenize.py) + compute graph features (graph_features.py).
  6. Pad and batch (this module, `pad_1d`/`pad_1d_feat`/`pad_2d`/
     `pad_2d_feat`/`pad_attn_bias`, ported verbatim from
     unimol2_dataset.py).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .conformer import generate_conformers
from .graph_features import compute_graph_features, mol_to_graph
from .tokenize import atoms_to_tokens

_MAX_ATOMS = 256  # CroppingDataset's max_atoms upstream default for the finetune task


def _remove_hydrogen(atoms: List[str], coords: np.ndarray) -> tuple[List[str], np.ndarray]:
    mask = np.array([a != "H" for a in atoms])
    return [a for a, m in zip(atoms, mask) if m], coords[mask]


def _crop_if_needed(atoms: List[str], coords: np.ndarray, seed: int) -> tuple[List[str], np.ndarray]:
    if len(atoms) <= _MAX_ATOMS:
        return atoms, coords
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(atoms), _MAX_ATOMS, replace=False)
    idx.sort()
    return [atoms[i] for i in idx], coords[idx]


def build_molecule_features(smiles: str, atoms_with_h: List[str], coords_with_h: np.ndarray, seed: int = 0) -> Dict[str, torch.Tensor]:
    """SMILES + a generated (with-H) conformer -> the full per-molecule
    feature dict the model's collater expects (before batch padding)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    atoms, coords = _remove_hydrogen(atoms_with_h, coords_with_h)
    atoms, coords = _crop_if_needed(atoms, coords, seed)
    coords = coords - coords.mean(axis=0)  # NormalizeDataset: centering only, no scaling

    src_token = atoms_to_tokens(atoms)

    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol, addCoords=True)
    mol = AllChem.RemoveAllHs(mol)
    node_attr, edge_index, edge_attr = mol_to_graph(mol)
    assert node_attr.shape[0] == len(atoms), (
        f"{smiles}: RDKit heavy-atom count ({node_attr.shape[0]}) doesn't match the "
        f"conformer's H-removed atom count ({len(atoms)}) -- atom ordering mismatch."
    )

    feat = compute_graph_features(node_attr, edge_index, edge_attr)
    feat["src_token"] = torch.from_numpy(src_token).long()
    feat["src_pos"] = torch.from_numpy(coords.astype(np.float32))
    return feat


def _pad_1d(samples: List[torch.Tensor], pad_len: int, pad_value=0) -> torch.Tensor:
    t = torch.full((len(samples), pad_len), pad_value, dtype=samples[0].dtype)
    for i, s in enumerate(samples):
        t[i, : s.shape[0]] = s
    return t


def _pad_1d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0) -> torch.Tensor:
    feat_size = samples[0].shape[-1]
    t = torch.full((len(samples), pad_len, feat_size), pad_value, dtype=samples[0].dtype)
    for i, s in enumerate(samples):
        t[i, : s.shape[0]] = s
    return t


def _pad_2d(samples: List[torch.Tensor], pad_len: int, pad_value=0) -> torch.Tensor:
    t = torch.full((len(samples), pad_len, pad_len), pad_value, dtype=samples[0].dtype)
    for i, s in enumerate(samples):
        t[i, : s.shape[0], : s.shape[1]] = s
    return t


def _pad_2d_feat(samples: List[torch.Tensor], pad_len: int, pad_value=0) -> torch.Tensor:
    feat_size = samples[0].shape[-1]
    t = torch.full((len(samples), pad_len, pad_len, feat_size), pad_value, dtype=samples[0].dtype)
    for i, s in enumerate(samples):
        t[i, : s.shape[0], : s.shape[1]] = s
    return t


def _pad_attn_bias(samples: List[torch.Tensor], pad_len: int) -> torch.Tensor:
    pad_len = pad_len + 1
    t = torch.full((len(samples), pad_len, pad_len), float("-inf"), dtype=samples[0].dtype)
    for i, s in enumerate(samples):
        t[i, : s.shape[0], : s.shape[1]] = s
        t[i, s.shape[0]:, : s.shape[1]] = 0
    return t


_PAD_FNS = {
    "src_token": _pad_1d, "src_pos": _pad_1d_feat, "atom_feat": _pad_1d_feat,
    "atom_mask": _pad_1d, "edge_feat": _pad_2d_feat, "shortest_path": _pad_2d,
    "degree": _pad_1d, "pair_type": _pad_2d_feat, "attn_bias": _pad_attn_bias,
}


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Ported verbatim from unimol2_dataset.py::Unimol2FinetuneFeatureDataset.collater."""
    max_node_num = max(item["atom_mask"].shape[0] for item in items)
    max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
    batched: Dict[str, torch.Tensor] = {}
    for key, pad_fn in _PAD_FNS.items():
        batched[key] = pad_fn([item[key] for item in items], max_node_num)
    return batched


def prepare_batch(smiles_list: List[str], num_workers: int = 4, timeout_s: int = 30) -> Dict[str, torch.Tensor]:
    """SMILES strings -> a model-ready batched_data dict."""
    conformers = generate_conformers(smiles_list, n_conformer=1, num_workers=num_workers, timeout_s=timeout_s)
    items = [
        build_molecule_features(smi, atoms, coords, seed=i)
        for i, (smi, (atoms, coords)) in enumerate(zip(smiles_list, conformers))
    ]
    return collate_batch(items)
