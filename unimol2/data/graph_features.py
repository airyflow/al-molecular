"""RDKit-derived per-atom/per-bond/per-pair feature computation for Uni-Mol2.

Ported from TWO upstream files that needed to be told apart carefully (this
was the main open research question this port had to resolve):

  - `unimol2/unimol2/data/molecule_dataset.py`: `allowable_features`,
    `atom_to_feature_vector`, `bond_to_feature_vector`, `get_graph` -- the
    raw RDKit atom/bond feature extraction (OGB-style, unmodified).
  - `unimol2/unimol2/data/unimol2_dataset.py`: its OWN LOCAL
    `get_graph_features(edge_attr, edge_index, node_attr, drop_feat)` and
    `floyd_warshall` -- NOT `molecule_dataset.py`'s same-named
    `get_graph_features`/`floyd_warshall` functions, which are a different,
    unused, older variant (confirmed dead: `unimol2/data/__init__.py`
    exports `MoleculeFeatureDataset` from molecule_dataset.py but never its
    top-level `get_graph_features`/`smi2_graph_features` functions; the
    task that actually builds the eval/finetune pipeline,
    `tasks/unimol_finetune.py::UniMolFinetuneTask.load_dataset`, wires
    `Unimol2FinetuneFeatureDataset`, which imports and calls
    unimol2_dataset.py's own local `get_graph_features`, not
    molecule_dataset.py's).

`drop_feat` is always False here -- a training-time-only feature-dropout
regularization (`unimol2_dataset.py`'s `MoleculeFeatureDataset(...,
drop_feat_prob=...)` mechanism), confirmed to default to 0.0 for the
finetune/eval task specifically (`tasks/unimol_finetune.py`:
`parser.add_argument("--drop-feat-prob", default=0.0, ...)`)  -- so frozen
embedding extraction should never exercise the drop_feat=True branch.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# From molecule_dataset.py -- verbatim, matches
# https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
_ALLOWABLE_FEATURES = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL", "CHI_SQUAREPLANAR", "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def _safe_index(values: list, e) -> int:
    try:
        return values.index(e)
    except ValueError:
        return len(values) - 1


def _atom_to_feature_vector(atom) -> List[int]:
    return [
        _safe_index(_ALLOWABLE_FEATURES["possible_atomic_num_list"], atom.GetAtomicNum()),
        _ALLOWABLE_FEATURES["possible_chirality_list"].index(str(atom.GetChiralTag())),
        _safe_index(_ALLOWABLE_FEATURES["possible_degree_list"], atom.GetTotalDegree()),
        _safe_index(_ALLOWABLE_FEATURES["possible_formal_charge_list"], atom.GetFormalCharge()),
        _safe_index(_ALLOWABLE_FEATURES["possible_numH_list"], atom.GetTotalNumHs()),
        _safe_index(_ALLOWABLE_FEATURES["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
        _safe_index(_ALLOWABLE_FEATURES["possible_hybridization_list"], str(atom.GetHybridization())),
        _ALLOWABLE_FEATURES["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        _ALLOWABLE_FEATURES["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]


def _bond_to_feature_vector(bond) -> List[int]:
    return [
        _safe_index(_ALLOWABLE_FEATURES["possible_bond_type_list"], str(bond.GetBondType())),
        _ALLOWABLE_FEATURES["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        _ALLOWABLE_FEATURES["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]


def mol_to_graph(mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ported from molecule_dataset.py::get_graph. `mol` must already have
    hydrogens removed (upstream calls this on `AllChem.RemoveAllHs(mol)`)."""
    atom_features_list = [_atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features_list, dtype=np.int64)

    if len(mol.GetBonds()) > 0:
        edges_list, edge_features_list = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feat = _bond_to_feature_vector(bond)
            edges_list += [(i, j), (j, i)]
            edge_features_list += [feat, feat]
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 3), dtype=np.int64)
    return x, edge_index, edge_attr


def _floyd_warshall(adj: np.ndarray) -> np.ndarray:
    """Ported from unimol2_dataset.py::floyd_warshall -- note the distance
    cap is 509 here, NOT molecule_dataset.py's 510 (the two files' otherwise
    near-identical floyd_warshall functions differ in this one constant;
    since unimol2_dataset.py's version is the one the real pipeline uses,
    509 is the correct cap for this port)."""
    M = adj.copy()
    n = M.shape[0]
    M[M == 0] = 509
    np.fill_diagonal(M, 0)
    for k in range(n):
        M = np.minimum(M, M[:, k:k + 1] + M[k:k + 1, :])
    M[M >= 509] = 509
    return M


def _convert_to_single_emb(x: np.ndarray, sizes: List[int]) -> np.ndarray:
    """Ported verbatim from unimol2_dataset.py::convert_to_single_emb."""
    assert x.shape[-1] == len(sizes)
    offset = 1
    for i, size in enumerate(sizes):
        assert (x[..., i] < size).all(), f"feature dim {i} has a value >= {size}"
        x[..., i] = x[..., i] + offset
        offset += size
    return x


def compute_graph_features(node_attr: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> Dict[str, torch.Tensor]:
    """Ported from unimol2_dataset.py::get_graph_features with drop_feat
    always False (see module docstring). Operates on N real atoms only (no
    virtual-node slot) EXCEPT attn_bias, which upstream already sizes
    (N+1, N+1) at this stage -- the model's AtomFeature/EdgeFeature modules
    prepend the virtual-node row/column for everything else."""
    atom_feat_sizes = [16] * 8
    edge_feat_sizes = [16, 16, 16]
    N = node_attr.shape[0]

    # NOTE: node_attr[:, 1:] -- column 0 (the raw atomic-num index) is
    # deliberately excluded here, verified directly against unimol2_dataset.py.
    atom_feat = _convert_to_single_emb(node_attr[:, 1:].copy(), atom_feat_sizes)

    adj = np.zeros((N, N), dtype=np.int64)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    degree = adj.sum(axis=-1)

    edge_feat = np.zeros((N, N, edge_attr.shape[-1]), dtype=np.int64)
    edge_feat[edge_index[0, :], edge_index[1, :]] = _convert_to_single_emb(edge_attr.copy(), edge_feat_sizes) + 1

    shortest_path_result = _floyd_warshall(adj)

    # drop_feat=False branch (verbatim offsets from unimol2_dataset.py):
    atom_feat = atom_feat + 2
    edge_feat = edge_feat + 2
    degree = degree + 2
    shortest_path_result = shortest_path_result + 1

    feat: Dict[str, torch.Tensor] = {}
    feat["atom_feat"] = torch.from_numpy(atom_feat).long()
    feat["atom_mask"] = torch.ones(N, dtype=torch.long)
    feat["edge_feat"] = torch.from_numpy(edge_feat).long()
    feat["shortest_path"] = torch.from_numpy(shortest_path_result).long()
    feat["degree"] = torch.from_numpy(degree).long().view(-1)

    # pair_type is derived from atom_feat[..., 0] -- i.e. the (already
    # offset-shifted) FIRST of the 8 non-atomic-number features (chirality),
    # NOT atom identity/atomic number. Verified directly, not assumed --
    # this is exactly the kind of detail that would silently produce a
    # plausible-but-wrong embedding if guessed instead of read from source.
    atoms = feat["atom_feat"][..., 0]
    pair_type = torch.cat(
        [atoms.view(-1, 1, 1).expand(-1, N, -1), atoms.view(1, -1, 1).expand(N, -1, -1)], dim=-1,
    )
    feat["pair_type"] = torch.from_numpy(
        _convert_to_single_emb(pair_type.numpy().copy(), [128, 128])
    ).long()

    feat["attn_bias"] = torch.zeros((N + 1, N + 1), dtype=torch.float32)
    return feat
