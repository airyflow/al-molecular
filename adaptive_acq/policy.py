"""Small MLP policy pi(beta_arm | s_t) for Stage 2 (GRPO).

Deliberately tiny: the input is the 10-dim campaign summary from
state.featurize(), never molecule embeddings. That is the design choice that
separates this from GLARE (which scores molecules directly with a learned
policy): here the policy only modulates how much the existing surrogate's
uncertainty counts, so it is cheap to train, surrogate-agnostic, and can be
moved between datasets/backbone ensembles without retraining.
"""

import numpy as np
import torch
import torch.nn as nn

from .state import FEATURE_DIM, FEATURE_NAMES


class BetaPolicy(nn.Module):
    def __init__(self, n_arms: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_arms),
        )
        # Start near-uniform so early GRPO groups actually explore all arms.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=-1)

    @torch.no_grad()
    def probs(self, ctx: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(ctx, dtype=torch.float32).unsqueeze(0)
        p = self.forward(x).exp().squeeze(0).numpy().astype(np.float64)
        return p / p.sum()  # float32 softmax can miss 1.0 by more than rng.choice tolerates

    def save(self, path, arms, extra=None):
        torch.save({"state_dict": self.state_dict(), "arms": list(arms),
                    "hidden": self.net[0].out_features, "feature_names": list(FEATURE_NAMES),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location="cpu")
        if tuple(ck["feature_names"]) != FEATURE_NAMES:
            raise ValueError("policy was trained with a different feature set")
        pol = cls(len(ck["arms"]), hidden=ck["hidden"])
        pol.load_state_dict(ck["state_dict"])
        pol.eval()
        return pol, tuple(ck["arms"])
