"""CPU-only smoke tests on a synthetic pool -- no embeddings, oracle or GPU.

    python -m pytest adaptive_acq/tests -q        (from the repo root)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from adaptive_acq.acquisition import select_batch, ucb  # noqa: E402
from adaptive_acq.controllers import (BatchSplitThompson, FixedBeta, LinUCB, PolicyController,  # noqa: E402
                                      Schedule)
from adaptive_acq.policy import BetaPolicy  # noqa: E402
from adaptive_acq.sim_env import SimTask, episode_return, run_episode  # noqa: E402
from adaptive_acq.state import FEATURE_DIM  # noqa: E402


@pytest.fixture(scope="module")
def task():
    rng = np.random.default_rng(0)
    n, d = 20_000, 16
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = -(X @ w + 0.5 * np.sin(3 * X[:, 0]) + 0.3 * rng.normal(size=n))  # lower = better
    return SimTask(X, y, name="synthetic", source="synthetic")


def test_select_batch_no_duplicates_and_counts():
    rng = np.random.default_rng(1)
    mu, sigma = rng.normal(size=5000), rng.random(5000)
    plan = [(0, 0.0, 30), (1, 1.0, 50), (2, 4.0, 20)]
    sel, owner = select_batch(mu, sigma, plan)
    assert len(sel) == 100 and len(np.unique(sel)) == 100
    assert [int((owner == a).sum()) for a in range(3)] == [30, 50, 20]


def test_beta_zero_is_greedy():
    rng = np.random.default_rng(2)
    mu, sigma = rng.normal(size=1000), rng.random(1000)
    sel, _ = select_batch(mu, sigma, [(0, 0.0, 10)])
    assert set(sel) == set(np.argsort(-mu)[:10])
    assert np.allclose(ucb(mu, sigma, 0.0), mu)


def test_raw_beta_matches_molpal_ucb():
    from molpal.acquirer.metrics import ucb as molpal_ucb
    rng = np.random.default_rng(3)
    mu, var = rng.normal(size=100), rng.random(100)
    assert np.allclose(ucb(mu, np.sqrt(var), 2.0, "raw"), molpal_ucb(mu, var, 2))


@pytest.mark.parametrize("make", [
    lambda: FixedBeta(0.0),
    lambda: FixedBeta(1.0),
    lambda: Schedule("linear_decay"),
    lambda: Schedule("alternating"),
    lambda: BatchSplitThompson(seed=0),
    lambda: LinUCB(),
    lambda: PolicyController(BetaPolicy(6), mode="sample"),
])
def test_episode_runs(task, make):
    h = run_episode(task, make(), seed=0, batch_frac=0.005, topk_frac=0.005)
    assert len(h) == 5
    rec = [r["topk_recall"] for r in h]
    assert all(0 <= r <= 1 for r in rec) and rec == sorted(rec)
    assert all(len(r["context"]) == FEATURE_DIM for r in h)
    assert h[-1]["n_labeled"] == 6 * 100
    assert episode_return(h) > 0.05  # far above random (~0.03) on this easy pool


def test_mlp_surrogate_episode(task):
    small = SimTask(task.X[:5000], task.y[:5000], "small", "synthetic")
    h = run_episode(small, FixedBeta(1.0), seed=0, surrogate="mlp", batch_frac=0.01, n_rounds=2)
    assert len(h) == 2


def test_same_seed_same_init(task):
    a = run_episode(task, FixedBeta(0.0), seed=7, batch_frac=0.005, n_rounds=1)
    b = run_episode(task, FixedBeta(4.0), seed=7, batch_frac=0.005, n_rounds=1)
    assert a[0]["context"][6] == b[0]["context"][6]  # identical start -> identical first-round state


def test_linucb_counterfactual_and_roundtrip(task, tmp_path):
    ctrl = LinUCB()
    h = run_episode(task, ctrl, seed=0, batch_frac=0.005, counterfactual=True)
    assert "counterfactual_reward" in h[0]
    ctrl.save(tmp_path / "p.json")
    args = argparse.Namespace(controller="linucb", arms=list(ctrl.arms), linucb_alpha=0.5,
                              linucb_frozen=True, controller_path=str(tmp_path / "p.json"), seed=0)
    from adaptive_acq.controllers import build_controller
    c2 = build_controller(args)
    assert np.allclose(c2.A, ctrl.A)


def test_thompson_state_roundtrip(task):
    c = BatchSplitThompson(seed=0)
    run_episode(task, c, seed=0, batch_frac=0.005)
    c2 = BatchSplitThompson(seed=0)
    c2.load_state_dict(c.state_dict())
    assert np.allclose(c2.a, c.a) and np.allclose(c2.b, c.b)


def test_grpo_update_moves_policy():
    import copy
    import torch
    from adaptive_acq.train_grpo import grpo_update
    torch.manual_seed(0)
    pol = BetaPolicy(3)
    ref = copy.deepcopy(pol)
    x = np.zeros(FEATURE_DIM)
    x[0] = 1.0
    batch = [(x, 2, float(np.log(1 / 3)), 1.0), (x, 0, float(np.log(1 / 3)), -1.0)]
    opt = torch.optim.Adam(pol.parameters(), lr=0.05)
    grpo_update(pol, ref, batch, epochs=5, lr_opt=opt, clip=0.2, kl_coef=0.0, ent_coef=0.0)
    p = pol.probs(x)
    assert p[2] > p[1] > p[0]


def test_diverse_lambda_zero_is_topk_and_positive_reduces_redundancy():
    from adaptive_acq.acquisition import batch_redundancy, select_diverse, top_k
    rng = np.random.default_rng(5)
    # 50 tight clusters of 40 near-duplicates; cluster quality varies
    centers = rng.normal(size=(50, 16))
    X = np.repeat(centers, 40, axis=0) + 0.05 * rng.normal(size=(2000, 16))
    score = np.repeat(rng.normal(size=50), 40) + 0.01 * rng.normal(size=2000)
    feats = lambda loc: X[loc]
    assert set(select_diverse(score, feats, 100, 0.0)) == set(top_k(score, 100))
    g = top_k(score, 100)
    clusters, red = [len(np.unique(g // 40))], [batch_redundancy(X[g])]
    for lam in (1.0, 2.0, 5.0):
        d = select_diverse(score, feats, 100, lam, device="cpu")
        assert len(np.unique(d)) == 100
        clusters.append(len(np.unique(d // 40)))
        red.append(batch_redundancy(X[d]))
    # larger lambda -> more clusters covered, less redundant; lambda=5 covers >3x greedy's clusters
    assert clusters == sorted(clusters) and red == sorted(red, reverse=True)
    assert clusters[-1] > 3 * clusters[0]


def test_episode_with_diversity(task):
    h = run_episode(task, FixedBeta(0.0), seed=0, batch_frac=0.005, topk_frac=0.005, diversity_lambda=0.5)
    assert len(h) == 5 and h[-1]["n_labeled"] == 600
    assert all("batch_redundancy" in r for r in h)


def test_pulse_schedule():
    c = Schedule("pulse", beta_hi=0.7, beta_lo=0.0, pulse_round=3)
    assert [c.beta_at(r, 5) for r in range(5)] == [0.0, 0.0, 0.7, 0.0, 0.0]
    try:
        Schedule("pulse")
        raise AssertionError("pulse without pulse_round should raise")
    except ValueError:
        pass


def test_ucb_rejects_non_finite():
    """-inf in mu used to make every score nan and randomize selection."""
    from adaptive_acq.acquisition import ucb
    mu = np.array([3.0, 2.0, -np.inf]); sigma = np.ones(3)
    for scale in ("normalized", "raw"):
        try:
            ucb(mu, sigma, 0.0, scale)
            raise AssertionError(f"ucb({scale}) should reject non-finite mu")
        except ValueError:
            pass


def test_prefilter_restricts_scoring_and_is_noop_when_off(task):
    off = run_episode(task, FixedBeta(0.0), seed=0, batch_frac=0.01, topk_frac=0.01, prefilter_mult=0.0)
    assert all(abs(r["scored_frac"] - (1 - 0.01 * (r["round"]))) < 0.02 for r in off)  # ~full pool each round

    on = run_episode(task, FixedBeta(0.0), seed=0, batch_frac=0.01, topk_frac=0.01, prefilter_mult=2.0)
    assert on[0]["scored_frac"] == off[0]["scored_frac"]      # round 1 still scores everything
    assert on[-1]["scored_frac"] < 0.5 * off[-1]["scored_frac"]  # later rounds score far less
    assert sum(r["scored_frac"] for r in on) < sum(r["scored_frac"] for r in off)
    assert on[-1]["n_labeled"] == off[-1]["n_labeled"]        # same budget spent


def test_prefilter_refresh_rescans_pool_and_rebuilds(task):
    static = run_episode(task, FixedBeta(0.0), seed=0, batch_frac=0.01, topk_frac=0.01, prefilter_mult=2.0)
    refr = run_episode(task, FixedBeta(0.0), seed=0, batch_frac=0.01, topk_frac=0.01,
                       prefilter_mult=2.0, prefilter_refresh=3)
    # round 3 re-scores (nearly) the whole pool; the static arm does not
    assert refr[2]["scored_frac"] > 5 * static[2]["scored_frac"]
    assert refr[2]["scored_frac"] > 0.8
    # rounds 4-5 are back on a shortlist, so still far cheaper than full scoring
    assert refr[3]["scored_frac"] < 0.2 and refr[4]["scored_frac"] < 0.2
    assert sum(r["scored_frac"] for r in refr) < 3.0   # vs ~5 for full scoring
    assert refr[-1]["n_labeled"] == static[-1]["n_labeled"]


def test_budget_is_fixed_not_recomputed(task):
    """A refresh must not inflate the shortlist by counting labeled molecules twice."""
    from adaptive_acq.campaign import Campaign
    import numpy as _np
    c = Campaign(n_pool=1000, fit_predict=lambda *a: (_np.zeros(1), _np.zeros(1)),
                 score_fn=lambda idx: _np.zeros(len(idx)), init_idx=_np.arange(10),
                 batch_size=10, n_rounds=5, k_obs=5)
    b0 = c.budget()
    c.labeled_idx = _np.arange(40)          # pretend several rounds have happened
    assert c.budget() == b0 == 10 + 10 * 5
