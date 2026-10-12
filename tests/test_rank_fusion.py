"""Regression tests for the rank-based fusion surrogates (EnsembleFusion,
BigFusion): chunked/sharded prediction must equal whole-pool prediction,
greedy ordering must match the original Borda formula, and the ensemble's
UCB must actually differ from greedy.

    python -m pytest tests/test_rank_fusion.py -q     (from the repo root)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_experiment as exp  # noqa: E402
from molpal.acquirer.metrics import ucb as molpal_ucb  # noqa: E402
from surrogates import BigFusionSurrogate, EnsembleFusionSurrogate  # noqa: E402

DIMS = {"grover": 6, "molformer": 5, "unimol": 4}


class _FakeModel:
    """Just the two attributes _chunked_get_means_and_vars uses."""

    def __init__(self, surrogate, X):
        self.surrogate, self._X = surrogate, X

    def _get_X(self, xs):
        return self._X[np.asarray(xs)]


def _fitted(cls, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, sum(DIMS.values()))).astype(np.float32)
    y = (X @ rng.normal(size=X.shape[1]) + 0.3 * rng.normal(size=n)).astype(np.float32)
    s = cls(dims=DIMS)
    s.fit(X[:300], y[:300], epochs=2, batch=64)
    return s, X


def _old_borda(comps, weights=None):
    """The pre-fix predict() body in effect: -sum_k w_k * rank_k, ranks via
    argsort(mu)[::-1] (identical to average ranks when there are no ties)."""
    borda = np.zeros(len(comps))
    for j in range(comps.shape[1]):
        r = np.empty(len(comps))
        r[np.argsort(comps[:, j])[::-1]] = np.arange(1, len(comps) + 1)
        borda += (weights[j] if weights else 1.0) * r
    return -borda


def _continuous_comps(n=3000, seed=1):
    """Tie-free per-backbone predictions (real GPU predictions are bf16 and
    full of ties, which would make formula checks depend on tie-breaking)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    return np.stack([z + 0.5 * rng.normal(size=n) for _ in range(3)], axis=1)


def _chunked(surrogate, X, chunk, monkeypatch):
    monkeypatch.setattr(exp, "POOL_PREDICT_CHUNK", chunk)
    return exp._chunked_get_means_and_vars(_FakeModel(surrogate, X), list(range(len(X))))


def test_ensemble_chunked_equals_whole(monkeypatch):
    s, X = _fitted(EnsembleFusionSurrogate)
    mu_w, sig_w = s.predict(X)
    mu_c, var_c = _chunked(s, X, 700, monkeypatch)  # 700 doesn't divide 3000: ragged last chunk too
    np.testing.assert_array_equal(mu_c, mu_w)
    np.testing.assert_allclose(var_c, sig_w ** 2)


def test_bigfusion_chunked_equals_whole(monkeypatch):
    s, X = _fitted(BigFusionSurrogate)
    mu_w, _ = s.predict(X)
    mu_c, _ = _chunked(s, X, 700, monkeypatch)
    np.testing.assert_array_equal(mu_c, mu_w)


def test_greedy_order_matches_original_formula_without_ties():
    comps = _continuous_comps()
    s, _ = _fitted(EnsembleFusionSurrogate)
    mu, _ = s.combine(comps)
    assert np.array_equal(np.argsort(-mu, kind="stable"), np.argsort(-_old_borda(comps), kind="stable"))
    b, _ = _fitted(BigFusionSurrogate)
    w = [b._w[k] for k in b._KEYS]
    np.testing.assert_allclose(b.combine(comps)[0], _old_borda(comps, w), rtol=1e-6)


def test_ties_are_not_broken_by_pool_position():
    s, _ = _fitted(EnsembleFusionSurrogate)
    comps = np.zeros((10, 3))  # all tied
    mu, sigma = s.combine(comps)
    assert np.all(mu == mu[0]) and np.all(sigma == 0)


def test_ensemble_ucb_differs_from_greedy_and_matches_docstring():
    s, _ = _fitted(EnsembleFusionSurrogate)
    comps = _continuous_comps()
    mu, sigma = s.combine(comps)
    assert sigma.max() > 0
    score = molpal_ucb(mu, sigma ** 2)  # what --acq ucb computes
    rn = np.stack([np.argsort(np.argsort(-comps[:, j])) + 1 for j in range(3)], axis=1) / len(comps)
    np.testing.assert_allclose(score, -rn.mean(1) + s._beta * rn.std(1), atol=1e-5)
    B = 300
    assert set(np.argsort(-score)[:B]) != set(np.argsort(-mu)[:B])


def test_per_chunk_ranking_would_have_been_wrong():
    """The bug itself, on a pool sorted best-first like ENHITS: ranked
    globally, the top 300 all come from the best chunk; ranked per chunk,
    the WORST chunk contributes about as many as the best one (its own
    extremes get within-chunk ranks as good as chunk 0's)."""
    s, _ = _fitted(EnsembleFusionSurrogate)
    comps = _continuous_comps()
    comps = comps[np.argsort(-comps.mean(1))]  # sorted best-first, like ENHITS
    per_chunk = np.concatenate([s.combine(comps[i:i + 500])[0] for i in range(0, len(comps), 500)])
    glob = s.combine(comps)[0]
    assert np.bincount(np.argsort(-glob)[:300] // 500, minlength=6)[0] == 300
    assert np.bincount(np.argsort(-per_chunk)[:300] // 500, minlength=6)[5] > 50
