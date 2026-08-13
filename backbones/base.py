"""Unified interface for the 4 embedding backbones (GROVER, MoLFormer,
UniMol, UniMol2), used identically by frozen extraction
(compute_*_embeddings_chunk.py) and — once Phase 2 lands — live
fine-tuning (backbone_finetuner.py).

This collapses what were 2-3 near-duplicate copies of "forward pass minus
supervised head, then pool" per backbone (extraction script,
backbone_finetuner.py's _emb_*, and for UniMol a third copy inside the
model class) down to exactly one implementation per backbone, called from
every consumer.

Design notes (see the approved plan for the full reasoning):
- An ABC, not a Protocol -- instantiation (checkpoint loading, config
  resolution) is the interesting/shared part, and an ABC lets the base
  class own the device-selection/TF32 boilerplate as one concrete method
  every subclass inherits, rather than a helper function each one calls
  separately.
- featurize() returns Optional -- None means "this molecule failed to
  featurize" (bad SMILES, RDKit parse error, timeout); callers filter
  these out before collating rather than crashing a batch.
- forward_and_pool() doesn't know or care whether it's called under
  torch.no_grad()/model.eval() (extraction) or model.train() with
  gradients on (fine-tuning) -- that's entirely the caller's
  responsibility, same as nn.Module.forward() itself.
- Conformer staging (UniMol persists to LMDB; UniMol2 generates inline
  per-batch) is NOT part of the base contract -- it's an optional
  capability via the ConformerStagedBackbone mixin, so UniMol2 isn't
  forced to fake a no-op store to satisfy a shared method every backbone
  would otherwise have to implement.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Protocol

import torch


@dataclass
class MoleculeFeatures:
    """One molecule's backbone-specific preprocessed representation.
    Opaque outside the Backbone that produced it -- only the same
    instance's collate()/forward_and_pool() know how to interpret
    `payload` (a MolGraphAttrs, a plain SMILES string, an (atoms,
    coordinates) tuple, ...)."""

    payload: Any
    smiles: str


@dataclass
class MoleculeBatch:
    """A collated batch, device-agnostic until .to(device)."""

    payload: Any
    n: int

    def to(self, device: torch.device) -> "MoleculeBatch":
        p = self.payload
        if hasattr(p, "to") and callable(p.to):
            return MoleculeBatch(payload=p.to(device), n=self.n)
        if isinstance(p, dict):
            return MoleculeBatch(payload={k: (v.to(device) if torch.is_tensor(v) else v) for k, v in p.items()}, n=self.n)
        raise NotImplementedError(
            f"MoleculeBatch.to(): payload of type {type(p)} has no .to() and isn't a dict of tensors -- "
            f"add a case here or give the payload its own .to()."
        )


class ConformerStore(Protocol):
    """Looks up a pre-generated (atoms, coordinates) conformer by index.
    Two implementations (backbones/unimol.py): LmdbShardConformerStore
    (extraction -- wraps a chunk's LMDB shard, positional index within the
    chunk) and, in Phase 2, PoolCacheConformerStore (fine-tuning -- wraps
    the whole-pool conformer cache, index into the full pool)."""

    def __getitem__(self, index: int) -> tuple: ...  # -> (atoms: list[str], coordinates: np.ndarray)


class Backbone(abc.ABC):
    """One embedding model: SMILES -> preprocessed features -> pooled
    embedding vector. Subclasses: GroverBackbone, MoLFormerBackbone,
    UniMolBackbone, UniMol2Backbone (backbones/<name>.py)."""

    name: str
    embedding_dim: int  # resolved AFTER construction (probed, not assumed from config)
    default_batch_size: int  # each backbone declares its own -- never one global knob
    default_device: str  # "cuda" | "cpu" -- what this backbone's compute cost actually calls for

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def resolve_device(preferred: Optional[str] = None) -> torch.device:
        """Shared device-selection logic, identical across all 4 backbones
        today (device-selection block duplicated verbatim in every
        compute_*_embeddings_chunk.py) -- now a single concrete method
        every subclass inherits instead of a copy-pasted block."""
        if preferred is not None:
            device = torch.device(preferred)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return device

    @abc.abstractmethod
    def featurize(self, smi: str) -> Optional[MoleculeFeatures]:
        """Pure, stateless per-molecule preprocessing step."""
        ...

    @abc.abstractmethod
    def collate(self, features: List[MoleculeFeatures]) -> MoleculeBatch:
        """Batch N MoleculeFeatures into one padded/stacked MoleculeBatch."""
        ...

    @abc.abstractmethod
    def forward_and_pool(self, batch: MoleculeBatch) -> torch.Tensor:
        """The single canonical embedding computation: forward pass +
        pooling into one (B, embedding_dim) vector per molecule. Caller
        controls train()/eval()/no_grad() -- this method doesn't set any
        of that itself."""
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, checkpoint_path: Optional[Path] = None, device: Optional[str] = None) -> "Backbone":
        """Resolves config (including any checkpoint-baked-in overrides),
        builds the model + collator, moves to device, and returns a ready
        instance with .embedding_dim already resolved. Does NOT call
        .eval()/.train() -- extraction callers call .eval() themselves
        right after load(); fine-tuning callers (Phase 2) call .train()."""
        ...


class ConformerStagedBackbone(Backbone):
    """Mixin for backbones whose featurize() needs an externally-staged
    conformer store attached first (today: UniMol only -- UniMol2 does
    NOT implement this, its conformer generation is cheap enough to run
    inline inside featurize() itself, matching its existing design)."""

    @abc.abstractmethod
    def attach_conformer_store(self, store: ConformerStore) -> None:
        """Must be called once before featurize() will succeed. Raises if
        featurize() is called before this."""
        ...
