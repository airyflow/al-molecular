"""Standalone port of unimol2/unimol2/models/unimol2.py::UniMol2Model
(deepmodeling/Uni-Mol). Only the frozen-embedding-extraction path is ported
-- this is NOT a general-purpose reimplementation of the pretraining model.

Deliberately NOT ported, both verified safe to drop by tracing the exact
upstream control flow (not guessed):

  - `MovementPredictionHead` and the coordinate-refinement (`pos`) update it
    drives. Tracing upstream's `forward()`: `one_block()` is called exactly
    ONCE (no outer iterative-refinement loop), and its updated `pos` return
    value is only ever read inside `if not features_only: if
    self.args.masked_coord_loss > 0: ...` / `masked_dist_loss > 0: ...`
    branches -- both dead whenever `mode == "infer"` (upstream returns
    `(x, pair)` before reaching those branches at all in that mode) AND
    dead regardless since `masked_coord_loss`/`masked_dist_loss` default to
    -1.0 and are never overridden by any released architecture size. So
    `movement_pred_head` has zero effect on `x`/`pair`/the pooled embedding
    -- it only ever produces a `pos` value nothing downstream reads, for
    every actual code path this port needs.
  - `MaskLMHead` (only built when `args.masked_token_loss > 0`, never true
    for any released architecture) and `ClassificationHead` (only used for
    supervised finetune heads registered via `register_classification_head`,
    never called by this port -- pooling is reproduced directly, see below).

Pooling: upstream's `ClassificationHead.forward` does `x = features[:, 0,
:]  # take <s> token (equiv. to [CLS])` -- this port reproduces that single
line directly in `forward()` rather than porting the whole head class,
since nothing else in `ClassificationHead` (a dense+activation+dropout+
out_proj classifier trunk) is relevant to raw embedding extraction.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import UniMol2Config
from .modules.atom_edge_features import AtomFeature, EdgeFeature
from .modules.gaussian_kernel import SE3InvariantKernel
from .modules.transformer_encoder_with_pair import TransformerEncoderWithPair


class UniMol2Model(nn.Module):
    def __init__(self, config: UniMol2Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.token_num, config.encoder_embed_dim, config.padding_idx)

        self.atom_feature = AtomFeature(
            num_atom=config.num_atom, num_degree=config.num_degree, hidden_dim=config.encoder_embed_dim,
        )
        self.edge_feature = EdgeFeature(
            pair_dim=config.pair_embed_dim, num_edge=config.num_edge, num_spatial=config.num_spatial,
        )
        self.encoder = TransformerEncoderWithPair(
            num_encoder_layers=config.encoder_layers, embedding_dim=config.encoder_embed_dim,
            pair_dim=config.pair_embed_dim, pair_hidden_dim=config.pair_hidden_dim,
            ffn_embedding_dim=config.encoder_ffn_embed_dim, num_attention_heads=config.encoder_attention_heads,
            dropout=config.dropout, attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout, activation_fn=config.activation_fn,
        )
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=config.pair_embed_dim, num_pair=config.num_pair, num_kernel=config.gaussian_kernel_k,
            std_width=config.gaussian_std_width, start=config.gaussian_mean_start, stop=config.gaussian_mean_stop,
        )

    @torch.no_grad()
    def forward(self, batched_data: dict) -> torch.Tensor:
        """batched_data keys (see unimol2/data/collate.py for how these are
        built): src_token [B,N], atom_feat [B,N,8], atom_mask [B,N],
        edge_feat [B,N,N,4], shortest_path [B,N,N], degree [B,N],
        pair_type [B,N,N,2], src_pos [B,N,3], attn_bias [B,N+1,N+1].

        Returns the pooled per-molecule embedding, shape [B, encoder_embed_dim].
        """
        cfg = self.config
        src_token = batched_data["src_token"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]
        pos = batched_data["src_pos"]

        n_mol = atom_mask.shape[0]
        token_feat = self.embed_tokens(src_token)
        x = self.atom_feature(batched_data, token_feat)

        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, cfg.encoder_attention_heads, 1, 1)
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, cfg.pair_embed_dim)
        attn_bias = self.edge_feature(batched_data, attn_bias)

        atom_mask_cls = torch.cat(
            [torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype), atom_mask], dim=1,
        )
        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.se3_invariant_kernel(dist, pair_type)
        new_attn_bias = attn_bias.clone()
        new_attn_bias[:, 1:, 1:, :] = new_attn_bias[:, 1:, 1:, :] + attn_bias_3d

        x, pair = self.encoder(
            x, new_attn_bias, atom_mask=atom_mask_cls, pair_mask=pair_mask, attn_mask=attn_mask,
        )

        return x[:, 0, :]  # pooled "virtual node" (== [CLS]) embedding
