"""Per-molecule feature building + batch collation for Uni-Mol (v1).

Ports `Batch`/`unpack_instances` (generic, from muben's
`muben/muben/dataset/dataset.py`) and `CollatorUniMol` (from
`muben/muben/dataset/dataset_unimol/collate.py`) -- simplified to the 4
fields `UniMolModel.forward()` actually reads (`atoms`, `coordinates`,
`distances`, `edge_types`). Production's `DatasetUniMol` also carries
`lbs`/`masks` (dummy zero/one arrays for extraction -- see
`compute_unimol_embeddings_chunk.py`'s `dataset._lbs = np.zeros(...)`),
needed only by muben's supervised-training loop, never read by the model's
forward pass -- dropped here rather than faked.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .process import ProcessingPipeline


class Batch:
    """Ported from muben's `dataset.Batch` -- a plain dynamic attribute bag
    with device-transfer support for its tensor members."""

    def __init__(self, **kwargs):
        self._tensor_members = {}
        for k, v in kwargs.items():
            setattr(self, k, v)
            if isinstance(v, torch.Tensor):
                self._tensor_members[k] = v

    def to(self, device):
        for k, v in self._tensor_members.items():
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        return len(next(iter(self._tensor_members.values())))


def unpack_instances(instance_list: List[dict], attr_names: List[str] = None):
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    return [[inst[name] for inst in instance_list] for name in attr_names]


def build_instance(atoms: List[str], coordinates: np.ndarray, pipeline: ProcessingPipeline) -> Dict[str, torch.Tensor]:
    """One molecule's (atoms, single conformer) -> the feature dict
    CollatorUniMol expects. `coordinates` is one (with-hydrogen) conformer,
    shape (n_atoms, 3) -- matching production's "keep only conformer index
    0" convention (see generate_unimol_conformers_chunk.py /
    compute_unimol_embeddings_chunk.py). `process_inference` is called with
    a single-element conformer list, matching production's own call
    pattern exactly (it always passes whatever conformer list the LMDB
    record stored, here always length 1)."""
    atoms_t, coords_t, dist_t, edge_t = pipeline.process_inference(atoms, [coordinates])
    return {"atoms": atoms_t, "coordinates": coords_t, "distances": dist_t, "edge_types": edge_t}


class CollatorUniMol:
    """Ported from muben's `CollatorUniMol` -- pads and concatenates a list
    of per-molecule instance dicts (see `build_instance`) into one `Batch`.
    `atom_pad_idx` must be `dictionary.pad()` (0 for the real Uni-Mol
    dictionary)."""

    def __init__(self, atom_pad_idx: int = 0):
        self._atom_pad_idx = atom_pad_idx

    def __call__(self, instances: List[dict]) -> Batch:
        atoms, coordinates, distances, edge_types = unpack_instances(instances, ["atoms", "coordinates", "distances", "edge_types"])

        max_length = max(tk.shape[1] for tk in atoms)

        atoms_batch = torch.cat([
            torch.cat([a, a.new(a.shape[0], max_length - a.shape[1]).fill_(self._atom_pad_idx)], dim=1)
            for a in atoms
        ], dim=0)
        coordinates_batch = torch.cat([
            torch.cat([c, c.new(c.shape[0], max_length - c.shape[1], c.shape[2]).fill_(0)], dim=1)
            for c in coordinates
        ], dim=0)
        distances_batch = torch.cat([
            F.pad(d, (0, max_length - d.shape[1], 0, max_length - d.shape[1]), value=0)
            for d in distances
        ], dim=0)
        edge_types_batch = torch.cat([
            F.pad(e, (0, max_length - e.shape[1], 0, max_length - e.shape[1]), value=0)
            for e in edge_types
        ], dim=0)

        return Batch(atoms=atoms_batch, coordinates=coordinates_batch, distances=distances_batch, edge_types=edge_types_batch)
