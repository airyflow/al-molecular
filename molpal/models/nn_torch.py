"""PyTorch reimplementation of MolPAL's NN model (originally Keras/TensorFlow
+ tensorflow_addons in nnmodels.py -- neither installed in this environment,
and tensorflow_addons is deprecated/archived upstream). Architecture matches
the paper's stated spec exactly: two 100-unit ReLU hidden layers, Adam
lr=0.01, MC-Dropout p=0.2 with T=10 stochastic forward passes at inference
for uncertainty. Unlike the Keras version, this expects pre-featurized
(N, D) float arrays for both train() and predict(), matching RFModel's
convention -- molpal/models/base.py's Model.apply() already documents this
distinction (type_ in {"mpn", "transformer", "molclr"} get raw identifiers,
everything else gets features).
"""
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from molpal.models.base import Model
from molpal.featurizer import feature_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NNTorch(nn.Module):
    def __init__(self, input_size: int, hidden: int = 100, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NNModelTorch(Model):
    """Feed-forward NN with MC-Dropout uncertainty, matching MolPAL paper's
    stated NN spec (2x100 ReLU, Adam lr=0.01, dropout p=0.2, T=10 passes).

    Parameters
    ----------
    input_size : int
        dimension of the (pre-computed) feature vectors passed to train/predict
    test_batch_size : int, default=4096
    dropout : float, default=0.2
    mc_passes : int, default=10
        number of stochastic forward passes (dropout active) for uncertainty
    epochs : int, default=50
    lr : float, default=0.01
    model_seed : Optional[int]
    """

    def __init__(
        self,
        input_size: int,
        test_batch_size: Optional[int] = 4096,
        dropout: float = 0.2,
        mc_passes: int = 10,
        epochs: int = 50,
        lr: float = 0.01,
        model_seed: Optional[int] = None,
        **kwargs,
    ):
        test_batch_size = test_batch_size or 4096
        self.input_size = input_size
        self.dropout = dropout
        self.mc_passes = mc_passes
        self.epochs = epochs
        self.lr = lr
        if model_seed is not None:
            torch.manual_seed(model_seed)

        self.model = _NNTorch(input_size, dropout=dropout).to(DEVICE)
        self.mean = 0.0
        self.std = 1.0

        super().__init__(test_batch_size, **kwargs)

    @property
    def provides(self):
        return {"means", "vars"}

    @property
    def type_(self):
        return "nn"

    def train(
        self,
        xs: Iterable,
        ys: Sequence[Optional[float]],
        *,
        featurizer,
        retrain: bool = False,
    ) -> bool:
        if retrain:
            self.model = _NNTorch(self.input_size, dropout=self.dropout).to(DEVICE)

        X = np.array(feature_matrix(xs, featurizer), dtype=np.float32)
        Y = np.array(list(ys), dtype=np.float32)
        self.mean = float(Y.mean())
        self.std = float(Y.std()) + 1e-8
        Y_norm = (Y - self.mean) / self.std

        Xt = torch.tensor(X, dtype=torch.float32)
        Yt = torch.tensor(Y_norm, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, Yt),
            batch_size=min(self.test_batch_size, len(Xt)), shuffle=True,
        )

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.model(xb), yb)
                loss.backward()
                opt.step()

        return True

    def _mc_predict(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        X = np.stack(xs, axis=0).astype(np.float32)
        Xt = torch.tensor(X, dtype=torch.float32)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt), batch_size=self.test_batch_size, shuffle=False,
        )

        # MC-Dropout: keep dropout active at inference, run mc_passes stochastic
        # forward passes per batch to estimate predictive mean/variance.
        self.model.train()
        means_batches, vars_batches = [], []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                passes = torch.stack([self.model(xb) for _ in range(self.mc_passes)], dim=0)
                means_batches.append(passes.mean(dim=0).cpu().numpy())
                vars_batches.append(passes.var(dim=0).cpu().numpy())

        mu = np.concatenate(means_batches) * self.std + self.mean
        var = np.concatenate(vars_batches) * (self.std ** 2)
        return mu, var

    def get_means(self, xs: Sequence) -> np.ndarray:
        mu, _ = self._mc_predict(xs)
        return mu

    def get_means_and_vars(self, xs: Sequence) -> Tuple[np.ndarray, np.ndarray]:
        return self._mc_predict(xs)

    def save(self, path) -> str:
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "mean": self.mean, "std": self.std},
            path / "model.pt",
        )
        return str(path / "model.pt")

    def load(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.mean, self.std = ckpt["mean"], ckpt["std"]
