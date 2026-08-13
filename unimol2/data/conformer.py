"""3D conformer generation for Uni-Mol2 embedding extraction.

`smiles_to_2d_coords`/`smiles_to_3d_coords`/`smiles_to_coords` below are
ported verbatim from muben's `muben/muben/utils/chem.py` (RDKit ETKDG
embedding + MMFF optimization, falling back to 2D coordinates on failure --
already exercised successfully across all 99,459,561 AmpC molecules for the
original Uni-Mol backbone). Pulled in directly rather than imported from
muben so this package has zero muben dependency -- unlike the model/feature
code (which is a genuine, Uni-Mol2-specific port), conformer generation is a
generic RDKit utility with no Uni-Mol2-specific logic, so there's no reason
for this package to reach into muben for it.

The timeout-protected multiprocessing pattern (below, in
`generate_conformers`) matches
`generate_unimol_conformers_chunk.py::generate_conformers_for_chunk` (one
bad molecule can otherwise hang RDKit's embedding forever, blocking an
entire chunk). This module returns `(atoms, coordinates)` pairs in memory
rather than writing to LMDB -- Uni-Mol2's extraction is single-stage (per
compute_unimol2_embeddings_chunk.py), unlike the original Uni-Mol's separate
persisted-conformer-shard stage.

n_conformer=1 (a single 3D conformer per molecule, not the 10-conformer
test-time-augmentation ensemble upstream's eval task uses) -- the same
deliberate compute/disk tradeoff already made for the original Uni-Mol
backbone in this repo, for the same reason: TTA-style multi-conformer
averaging is a variance-reduction technique layered on top of a
well-defined single-conformer forward pass, not a correctness requirement
(verified directly: upstream's own train-split path, ConformerSampleDataset,
also uses exactly one randomly-sampled conformer per item).
"""
from __future__ import annotations

import logging
import warnings
from functools import partial
from multiprocessing import get_context
from multiprocessing import TimeoutError as MPTimeoutError
from typing import List, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")

logger = logging.getLogger(__name__)


def smiles_to_2d_coords(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    assert len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {smiles}"
    return coordinates


def smiles_to_3d_coords(smiles: str, n_conformer: int) -> List[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(n_conformer):
        coordinates = list()
        try:
            # will random generate conformer with seed equal to -1. else fixed random seed.
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)  # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smiles)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)  # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except Exception as e:
                    logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
                    coordinates = smiles_to_2d_coords(smiles)
        except Exception as e:
            logger.warning(f"Failed to generate 3D, replace with 2D: {e}")
            coordinates = smiles_to_2d_coords(smiles)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {smiles}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smiles_to_coords(smiles: str, n_conformer: int = 10) -> Tuple[List[str], List[np.ndarray]]:
    mol = Chem.MolFromSmiles(smiles)
    if len(mol.GetAtoms()) > 400:
        coordinates = [smiles_to_2d_coords(smiles)] * (n_conformer + 1)
        logger.warning(f"atom num > 400, use 2D coords {smiles}")
    else:
        coordinates = smiles_to_3d_coords(smiles, n_conformer)
        coordinates.append(smiles_to_2d_coords(smiles).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    return atoms, coordinates


def generate_conformers(
    smiles_list: List[str], n_conformer: int = 1, num_workers: int = 4, timeout_s: int = 30,
) -> List[Tuple[List[str], np.ndarray]]:
    """Returns one (atoms_with_h, coordinates) pair per input SMILES, in the
    same order. `atoms_with_h` includes hydrogens (H-removal happens later,
    in graph_features.py, matching upstream's RemoveHydrogenDataset which
    operates AFTER conformer embedding -- H atoms matter for accurate 3D
    geometry during embedding, even though they're dropped before the model
    sees them)."""
    s2c = partial(smiles_to_coords, n_conformer=n_conformer)
    results: List[Tuple[List[str], np.ndarray]] = [None] * len(smiles_list)

    with get_context("fork").Pool(num_workers) as pool:
        pending = [(i, smi, pool.apply_async(s2c, (smi,))) for i, smi in enumerate(smiles_list)]
        for i, smi, async_result in pending:
            try:
                atoms, coordinates = async_result.get(timeout=timeout_s)
            except MPTimeoutError:
                print(f"[timeout] {smi!r} -- falling back to 2D coordinates")
                mol = Chem.MolFromSmiles(smi)
                coordinates = [smiles_to_2d_coords(smi)] * (n_conformer + 1)
                mol = AllChem.AddHs(mol)
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            # coordinates is a list of n_conformer 3D conformers (or their
            # 2D-fallback substitutes) plus one final appended 2D fallback --
            # take the first real conformer slot.
            results[i] = (list(atoms), np.asarray(coordinates[0], dtype=np.float32))

    return results
