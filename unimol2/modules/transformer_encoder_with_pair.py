"""Ported from unimol2/unimol2/models/layers.py (Transition, Attention,
OuterProduct, TriangleMultiplication, TransformerEncoderLayer) and
unimol2/unimol2/models/transformer_encoder_with_pair.py::TransformerEncoderWithPair
(deepmodeling/Uni-Mol). This is the core two-track (atom + pair)
Evoformer-style encoder: every layer updates BOTH the atom representation
`x` and the pair representation `pair` (self-attention biased by `pair`,
then an outer-product update of `pair` from the new `x`, then triangle
multiplication on `pair`) -- not a one-shot 3D bias applied once, which is
what makes this architecture materially more complex than the original
Uni-Mol's TransformerEncoderWithPair (same class name, different model).

Two upstream mechanisms deliberately dropped, both verified safe for this
port's frozen/no_grad-only use case:
  - OuterProduct's gradient-checkpointing branch (`torch.utils.checkpoint`)
    is skipped -- it only ever activates when `torch.is_grad_enabled()`,
    which is never true for frozen embedding extraction (always run under
    `torch.no_grad()`).
  - `DropPath` (stochastic depth) is not ported -- upstream only uses it
    when `droppath_prob > 0`, and 1.1B's config (`unimol2_1100M`, and every
    other registered Uni-Mol2 size) never overrides `droppath_prob` away
    from `base_architecture`'s default of 0.0, so upstream's own
    `if droppath_prob > 0: DropPath else: Dropout` always resolves to
    `Dropout` for every released checkpoint. `Dropout` is also a no-op
    whenever `module.training=False` (always true here), so both would
    behave identically regardless.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import permute_final_dims, softmax_dropout

# Matches TransformerEncoderLayer's own upstream constructor default. Never
# overridden by UniMol2Model's TransformerEncoderWithPair(...) call (verified
# directly -- pair_dropout is conspicuously absent from that call site), and
# irrelevant at inference regardless (dropout is a no-op in eval mode).
_PAIR_DROPOUT = 0.25


class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p > 0 and self.training:
            return F.dropout(x, p=self.p, training=True)
        return x


class Transition(nn.Module):
    def __init__(self, d_in: int, n: int, dropout: float = 0.0):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, n * d_in)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(n * d_in, d_in)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)
        return x


class Attention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, pair_dim: int,
                 head_dim: int, num_heads: int, gating: bool = False, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        total_dim = head_dim * num_heads
        self.gating = gating
        self.linear_q = nn.Linear(q_dim, total_dim, bias=False)
        self.linear_k = nn.Linear(k_dim, total_dim, bias=False)
        self.linear_v = nn.Linear(v_dim, total_dim, bias=False)
        self.linear_o = nn.Linear(total_dim, q_dim)
        self.linear_g = nn.Linear(q_dim, total_dim) if gating else None
        self.norm = head_dim ** -0.5
        self.dropout = dropout
        self.linear_bias = nn.Linear(pair_dim, num_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                pair: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        g = self.linear_g(q) if self.linear_g is not None else None

        q = self.linear_q(q) * self.norm
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        bias = self.linear_bias(pair).permute(0, 3, 1, 2).contiguous()
        attn = softmax_dropout(attn, self.dropout, self.training, mask=mask, bias=bias)
        o = torch.matmul(attn, v)

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        return self.linear_o(o)


class OuterProduct(nn.Module):
    """Updates the pair representation from an outer product of the (newly
    self-attended) atom representation with itself."""

    def __init__(self, d_atom: int, d_pair: int, d_hid: int = 32):
        super().__init__()
        self.d_hid = d_hid
        self.linear_in = nn.Linear(d_atom, d_hid * 2)
        self.linear_out = nn.Linear(d_hid ** 2, d_pair)

    def _opm(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        bsz, n, d = a.shape
        a = a.view(bsz, n, 1, d, 1)
        b = b.view(bsz, 1, n, 1, d)
        outer = (a * b).view(bsz, n, n, d * d)
        return self.linear_out(outer)

    def forward(self, m: torch.Tensor, op_mask: torch.Tensor, op_norm: torch.Tensor) -> torch.Tensor:
        ab = self.linear_in(m) * op_mask
        a, b = ab.chunk(2, dim=-1)
        z = self._opm(a, b)
        return z * op_norm


class TriangleMultiplication(nn.Module):
    def __init__(self, d_pair: int, d_hid: int):
        super().__init__()
        self.linear_ab_p = nn.Linear(d_pair, d_hid * 2)
        self.linear_ab_g = nn.Linear(d_pair, d_hid * 2)
        self.linear_g = nn.Linear(d_pair, d_pair)
        self.linear_z = nn.Linear(d_hid, d_pair)
        self.layer_norm_out = nn.LayerNorm(d_hid)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        mask = mask * (mask.shape[-2] ** -0.5)

        g = self.linear_g(z)
        ab = self.linear_ab_p(z)
        ab = ab * mask
        ab = ab * torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)

        a1 = permute_final_dims(a, [2, 0, 1])
        b1 = b.transpose(-1, -3)
        x = torch.matmul(a1, b1)
        b2 = permute_final_dims(b, [2, 0, 1])
        a2 = a.transpose(-1, -3)
        x = x + torch.matmul(a2, b2)

        x = permute_final_dims(x, [1, 2, 0])
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        return g * x


class TransformerEncoderLayer(nn.Module):
    """One block of the two-track encoder: self-attention + FFN on the atom
    track, then outer-product + triangle-multiplication + FFN on the pair
    track, each residually connected."""

    def __init__(self, embedding_dim: int, pair_dim: int, pair_hidden_dim: int,
                 ffn_embedding_dim: int, num_attention_heads: int, dropout: float,
                 attention_dropout: float, activation_dropout: float, activation_fn: str):
        super().__init__()
        self.dropout_module = Dropout(dropout)

        head_dim = embedding_dim // num_attention_heads
        self.self_attn = Attention(
            embedding_dim, embedding_dim, embedding_dim, pair_dim=pair_dim,
            head_dim=head_dim, num_heads=num_attention_heads, gating=False,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)

        self.ffn = Transition(embedding_dim, ffn_embedding_dim // embedding_dim, dropout=activation_dropout)
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.x_layer_norm_opm = nn.LayerNorm(embedding_dim)

        self.opm = OuterProduct(embedding_dim, pair_dim, d_hid=pair_hidden_dim)

        self.pair_layer_norm_ffn = nn.LayerNorm(pair_dim)
        self.pair_ffn = Transition(pair_dim, 1, dropout=activation_dropout)

        self.pair_dropout = _PAIR_DROPOUT
        self.pair_layer_norm_trimul = nn.LayerNorm(pair_dim)
        self.pair_tri_mul = TriangleMultiplication(pair_dim, pair_hidden_dim)

    def _shared_dropout(self, x: torch.Tensor, shared_dim: int, dropout: float) -> torch.Tensor:
        shape = list(x.shape)
        shape[shared_dim] = 1
        with torch.no_grad():
            mask = x.new_ones(shape)
        return F.dropout(mask, p=dropout, training=self.training) * x

    def forward(self, x: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor,
                self_attn_mask: Optional[torch.Tensor], op_mask: torch.Tensor, op_norm: torch.Tensor):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x, x, pair=pair, mask=self_attn_mask)
        x = self.dropout_module(x)
        x = residual + x

        x = x + self.dropout_module(self.ffn(self.final_layer_norm(x)))

        pair = pair + self.dropout_module(self.opm(self.x_layer_norm_opm(x), op_mask, op_norm))

        pair = pair + self._shared_dropout(
            self.pair_tri_mul(self.pair_layer_norm_trimul(pair), pair_mask), -3, self.pair_dropout,
        )

        pair = pair + self.dropout_module(self.pair_ffn(self.pair_layer_norm_ffn(pair)))

        return x, pair


class TransformerEncoderWithPair(nn.Module):
    def __init__(self, num_encoder_layers: int, embedding_dim: int, pair_dim: int,
                 pair_hidden_dim: int, ffn_embedding_dim: int, num_attention_heads: int,
                 dropout: float, attention_dropout: float, activation_dropout: float,
                 activation_fn: str):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.pair_layer_norm = nn.LayerNorm(pair_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embedding_dim=embedding_dim, pair_dim=pair_dim, pair_hidden_dim=pair_hidden_dim,
                ffn_embedding_dim=ffn_embedding_dim, num_attention_heads=num_attention_heads,
                dropout=dropout, attention_dropout=attention_dropout,
                activation_dropout=activation_dropout, activation_fn=activation_fn,
            )
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x: torch.Tensor, pair: torch.Tensor, atom_mask: torch.Tensor,
                pair_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = self.layer_norm(x)
        pair = self.pair_layer_norm(pair)

        op_mask = atom_mask.unsqueeze(-1)
        op_mask = op_mask * (op_mask.size(-2) ** -0.5)
        eps = 1e-3
        op_norm = 1.0 / (eps + torch.einsum("...bc,...dc->...bdc", op_mask, op_mask))

        for layer in self.layers:
            x, pair = layer(x, pair, pair_mask=pair_mask, self_attn_mask=attn_mask, op_mask=op_mask, op_norm=op_norm)
        return x, pair
