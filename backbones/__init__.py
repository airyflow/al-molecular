"""Unified backbone API. Usage: `backbones.load("molformer")` (or
"grover"/"unimol"/"unimol2") -> a ready Backbone instance."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Backbone, ConformerStagedBackbone, ConformerStore, MoleculeBatch, MoleculeFeatures

_REGISTRY = {}


def _lazy_registry():
    """Imports are deferred per-backbone so e.g. loading GROVER doesn't
    require UniMol2's torch/RDKit-heavy import chain (and vice versa)."""
    if not _REGISTRY:
        from .molformer import MoLFormerBackbone
        from .grover import GroverBackbone
        from .unimol import UniMolBackbone
        from .unimol2 import UniMol2Backbone

        _REGISTRY.update(
            molformer=MoLFormerBackbone,
            grover=GroverBackbone,
            unimol=UniMolBackbone,
            unimol2=UniMol2Backbone,
        )
    return _REGISTRY


def load(name: str, checkpoint_path: Optional[Path] = None, device: Optional[str] = None) -> Backbone:
    registry = _lazy_registry()
    if name not in registry:
        raise ValueError(f"Unknown backbone {name!r} -- choices are {sorted(registry)}")
    return registry[name].load(checkpoint_path=checkpoint_path, device=device)


__all__ = ["load", "Backbone", "ConformerStagedBackbone", "ConformerStore", "MoleculeBatch", "MoleculeFeatures"]
