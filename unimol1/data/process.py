"""Standalone port of muben's `muben/muben/dataset/dataset_unimol/process.py`
(itself ported from dptech-corp/Uni-Mol) -- pure feature preprocessing, no
muben dependency, no RNG-construction-order concerns (this only runs at
forward-pass/data-loading time, not during model construction).

Only `process_inference` is ported, not `process_training` /
`conformer_sampling` -- production's own extraction code
(`compute_unimol_embeddings_chunk.py::load_chunk_dataset`) explicitly
routes through `"inference"`, never `"training"`, because Stage 1
(conformer generation) stores exactly one conformer per molecule for
frozen extraction, not the 11-conformer augmentation set
`conformer_sampling` expects (see that script's own comment on this).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from scipy.spatial import distance_matrix

from ..dictionary import DictionaryUniMol

Matrix = "torch.Tensor | np.ndarray"


class ProcessingPipeline:
    def __init__(
        self,
        dictionary: DictionaryUniMol,
        coordinate_padding: Optional[float] = 0.0,
        max_atoms: Optional[int] = 256,
        max_seq_len: Optional[int] = 512,
        remove_hydrogen_flag: Optional[bool] = False,
        remove_polar_hydrogen_flag: Optional[bool] = False,
    ):
        self._dictionary = dictionary
        self._coordinate_padding = coordinate_padding
        self._max_atoms = max_atoms
        self._max_seq_len = max_seq_len
        self._remove_hydrogen_flag = remove_hydrogen_flag
        self._remove_polar_hydrogen_flag = remove_polar_hydrogen_flag

    def process_instance(self, atoms: np.ndarray, coordinates: np.ndarray):
        atoms, coordinates = check_atom_types(atoms, coordinates)
        atoms, coordinates = remove_hydrogen(atoms, coordinates, self._remove_hydrogen_flag, self._remove_polar_hydrogen_flag)
        atoms, coordinates = cropping(atoms, coordinates, self._max_atoms)

        atoms = tokenize_atoms(atoms, self._dictionary, self._max_seq_len)
        atoms = prepend_and_append(atoms, self._dictionary.bos(), self._dictionary.eos())

        edge_types = get_edge_type(atoms, len(self._dictionary))

        coordinates = normalize_coordinates(coordinates)
        coordinates = torch.from_numpy(coordinates)
        coordinates = prepend_and_append(coordinates, self._coordinate_padding, self._coordinate_padding)

        distances = get_distance(coordinates)

        return atoms, coordinates, distances, edge_types

    def process_inference(self, atoms: List[str], coordinates: List[np.ndarray]):
        atoms = np.array(atoms)
        atoms_, coordinates_, distances_, edge_types_ = [], [], [], []

        for coord in coordinates:
            a, c, d, e = self.process_instance(atoms, coord)
            atoms_.append(a)
            coordinates_.append(c)
            distances_.append(d)
            edge_types_.append(e)

        return torch.stack(atoms_), torch.stack(coordinates_), torch.stack(distances_), torch.stack(edge_types_)


def check_atom_types(atoms, coordinates):
    if len(atoms) != len(coordinates):
        min_len = min(len(atoms), len(coordinates))
        atoms = atoms[:min_len]
        coordinates = coordinates[:min_len]
    return atoms, coordinates


def remove_hydrogen(atoms, coordinates, remove_hydrogen_flag=False, remove_polar_hydrogen_flag=False):
    if remove_hydrogen_flag:
        mask_hydrogen = atoms != "H"
        atoms = atoms[mask_hydrogen]
        coordinates = coordinates[mask_hydrogen]
    if not remove_hydrogen_flag and remove_polar_hydrogen_flag:
        end_idx = 0
        for i, atom in enumerate(atoms[::-1]):
            if atom != "H":
                break
            end_idx = i + 1
        if end_idx != 0:
            atoms = atoms[:-end_idx]
            coordinates = coordinates[:-end_idx]
    return atoms, coordinates


def cropping(atoms, coordinates, max_atoms=256):
    if max_atoms and len(atoms) > max_atoms:
        index = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = np.array(atoms)[index]
        coordinates = coordinates[index]
    return atoms, coordinates


def normalize_coordinates(coordinates):
    coordinates = coordinates - coordinates.mean(axis=0)
    return coordinates.astype(np.float32)


def tokenize_atoms(atoms: np.ndarray, dictionary: DictionaryUniMol, max_seq_len=512):
    assert max_seq_len > len(atoms) > 0
    return torch.from_numpy(dictionary.vec_index(atoms)).long()


def prepend_and_append(item: torch.Tensor, prepend_value, append_value):
    item = torch.cat([torch.full_like(item[0], prepend_value).unsqueeze(0), item], dim=0)
    item = torch.cat([item, torch.full_like(item[0], append_value).unsqueeze(0)], dim=0)
    return item


def get_edge_type(atoms, num_types: int):
    return atoms.view(-1, 1) * num_types + atoms.view(1, -1)


def get_distance(coordinates: torch.Tensor):
    pos = coordinates.view(-1, 3).numpy()
    dist = distance_matrix(pos, pos).astype(np.float32)
    return torch.from_numpy(dist)
