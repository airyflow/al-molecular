"""Ported from unimol2/unimol2/models/layers.py::SE3InvariantKernel,
GaussianKernel, NonLinear (deepmodeling/Uni-Mol). Verbatim forward-pass
logic; `@torch.jit.script` dropped (pure perf optimization, not needed for
correctness, and avoids any TorchScript version-compat risk); custom
`Linear`/`Embedding` init-scheme classes replaced with plain
`nn.Linear`/`nn.Embedding` for the same reason as atom_edge_features.py."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianKernel(nn.Module):
    """Expands pairwise 3D distances into a K-dim Gaussian radial basis,
    scaled/shifted per atom-pair-type via learned `mul`/`bias` embeddings."""

    def __init__(self, K: int = 128, num_pair: int = 512, std_width: float = 1.0,
                 start: float = 0.0, stop: float = 9.0):
        super().__init__()
        self.K = K
        mean = torch.linspace(start, stop, K)
        self.std = std_width * (mean[1] - mean[0])
        self.register_buffer("mean", mean)
        self.mul = nn.Embedding(num_pair, 1, padding_idx=0)
        self.bias = nn.Embedding(num_pair, 1, padding_idx=0)

    def forward(self, x: torch.Tensor, atom_pair: torch.Tensor) -> torch.Tensor:
        mul = self.mul(atom_pair).abs().sum(dim=-2)
        bias = self.bias(atom_pair).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.mean.float().view(-1)
        return _gaussian(x.float(), mean, self.std)


class NonLinear(nn.Module):
    def __init__(self, input_dim: int, output_size: int, hidden: int | None = None):
        super().__init__()
        hidden = input_dim if hidden is None else hidden
        self.layer1 = nn.Linear(input_dim, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class SE3InvariantKernel(nn.Module):
    """3D-geometry-aware pair-bias: pairwise atom distances -> Gaussian
    kernel expansion -> projected into the pair-track's embedding dim."""

    def __init__(self, pair_dim: int, num_pair: int, num_kernel: int,
                 std_width: float = 1.0, start: float = 0.0, stop: float = 9.0):
        super().__init__()
        self.num_kernel = num_kernel
        self.gaussian = GaussianKernel(num_kernel, num_pair, std_width=std_width, start=start, stop=stop)
        self.out_proj = NonLinear(self.num_kernel, pair_dim)

    def forward(self, dist: torch.Tensor, node_type_edge: torch.Tensor) -> torch.Tensor:
        edge_feature = self.gaussian(dist, node_type_edge.long())
        return self.out_proj(edge_feature)
