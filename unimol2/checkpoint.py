"""Loads the public Uni-Mol2 checkpoint.pt into a standalone UniMol2Model.

The public release's checkpoint (verified by direct inspection) has exactly
one top-level key, `'model'` -- no `'args'`, unlike the fairseq/Uni-Core
`{"args": Namespace, "model": OrderedDict, ...}` shape `infer.py` upstream
expects when reading a checkpoint saved by its own trainer. So the 1.1B
config is hardcoded in config.py from the verified `unimol2_1100M`
architecture registration, not read off the file.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .config import UniMol2Config
from .model import UniMol2Model

# Prefixes present in the checkpoint for submodules this port deliberately
# does not instantiate (movement_pred_head: see model.py's docstring for why
# it's provably unused for embedding extraction; lm_head/classification_heads:
# pretraining/finetune-only heads, never touched by a frozen forward pass).
_EXPECTED_UNUSED_PREFIXES = ("movement_pred_head.", "lm_head.", "classification_heads.")


def load_raw_state_dict(checkpoint_path: str | Path) -> dict:
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "model" not in state:
        raise ValueError(f"{checkpoint_path}: expected a top-level 'model' key, got {list(state.keys())}")
    return state["model"]


def build_model_from_checkpoint(config: UniMol2Config | None = None) -> UniMol2Model:
    config = config or UniMol2Config()
    model = UniMol2Model(config)

    raw_state = load_raw_state_dict(config.checkpoint_path)
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(raw_state.keys())

    missing = model_keys - checkpoint_keys
    if missing:
        raise RuntimeError(
            f"{len(missing)} keys required by the model are missing from the checkpoint "
            f"(e.g. {sorted(missing)[:5]}) -- config/architecture mismatch."
        )

    unexpected = checkpoint_keys - model_keys
    unaccounted = [k for k in unexpected if not k.startswith(_EXPECTED_UNUSED_PREFIXES)]
    if unaccounted:
        raise RuntimeError(
            f"{len(unaccounted)} checkpoint keys are neither loaded into the model nor "
            f"one of the deliberately-unported {_EXPECTED_UNUSED_PREFIXES} submodules "
            f"(e.g. {sorted(unaccounted)[:5]}) -- something in the checkpoint isn't accounted for."
        )

    state_to_load = {k: v for k, v in raw_state.items() if k in model_keys}
    result = model.load_state_dict(state_to_load, strict=True)
    assert not result.missing_keys and not result.unexpected_keys

    model.eval()
    return model
