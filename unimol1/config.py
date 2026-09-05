"""Config for the standalone Uni-Mol (v1) port.

Every field's value is copied directly from `_UniMolConfig` in
`embed/compute/compute_unimol_embeddings_chunk.py` (the config muben's
`UniMol` is actually constructed with in this repo's production extraction
path today) -- not re-derived or guessed. Fields muben's `UniMol.__init__`
never reads (`feature_type`, `pooler_stride`) are dropped; everything that
influences either the architecture or the model-construction RNG sequence
(see model.py's docstring) is kept.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ZOO = ROOT / "models"


@dataclass
class UniMolConfig:
    # Architecture -- matches muben's _UniMolConfig / the real mol_pre_all_h_220816.pt checkpoint.
    encoder_embed_dim: int = 512
    encoder_layers: int = 15
    encoder_attention_heads: int = 64
    encoder_ffn_embed_dim: int = 2048
    activation_fn: str = "gelu"
    pooler_activation_fn: str = "Tanh"

    # Dropout -- eval-mode override, same reasoning as unimol2/config.py:
    # nn.Dropout is a no-op when module.training=False, so the exact value
    # only matters for matching the RNG sequence during construction (see
    # model.py), not for correctness of a frozen forward pass.
    dropout: float = 0.0
    emb_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    pooler_dropout: float = 0.0

    # Gates muben's UniMol.__init__ uses to decide whether to build
    # pair2coord_proj/dist_head (both False in production -- neither is
    # built, matching this port, which doesn't implement them at all).
    masked_coord_loss: float = 0.0
    masked_dist_loss: float = 0.0
    # Gates TransformerEncoderWithPair's no_final_head_layer_norm (< 0 here
    # -> True -> final_head_layer_norm is NOT built, matching production).
    delta_pair_repr_norm_loss: float = -1.0

    max_atoms: int = 64
    max_seq_len: int = 80
    remove_hydrogen: bool = True
    remove_polar_hydrogen: bool = False

    # torch.manual_seed() value called immediately before constructing
    # UniMolModel -- see checkpoint.py. Must never change once any chunk's
    # embeddings have been computed with it (see
    # compute_unimol_embeddings_chunk.py's own comment on this same
    # constant): every separate chunk-extraction process constructs its own
    # UniMolModel, and hidden_layer's weights (the actual embedding
    # projection) are NOT covered by the checkpoint -- this seed is the only
    # thing making that projection identical, and thus comparable, across
    # all chunks/processes.
    construction_seed: int = 20260813

    checkpoint_path: str = str(MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt")
