"""Campaign-state features (the controller's context s_t) and the observable
per-round reward.

Everything here uses only information a real screening campaign has: the
labeled set's oracle scores, the surrogate's current predictions, and how the
previous round's predictions compared with what docking actually returned.
The true top-k set is never used here -- it's for evaluation only (recall),
so a controller trained on rewards from this module could run unchanged on a
prospective screen where no ground truth exists.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .acquisition import EPS, top_k, ucb

FEATURE_NAMES = (
    "bias",
    "progress",            # round / n_rounds
    "is_first",            # 1 on the first acquisition (no feedback yet)
    "last_reward",         # previous round's hit fraction (see HIT_FRAC)
    "last_spearman",       # rank corr(predicted mu, observed) on last batch
    "last_sigma_err_corr", # rank corr(sigma, |error|) on last batch: is sigma informative?
    "headroom",            # mean sigma / std(mu) over the greedy top-B
    "ucb_greedy_jaccard",  # overlap of greedy top-B and beta_max top-B
    "improvement",         # change in mean of best-K labeled, in labeled-std units
    "last_beta",           # previous beta / beta_max
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class LastBatch:
    """What the previous round predicted for the molecules it acquired, and
    what docking returned. Stored so the next round can compute feedback."""
    mu: np.ndarray       # higher = better
    sigma: np.ndarray
    y: np.ndarray        # oracle scores, lower = better
    reward: float
    beta: float          # mean beta over the batch's arms (count-weighted)
    best_k_mean_before: float


def _rank(x):
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(a, b) -> float:
    if len(a) < 3:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa < EPS or sb < EPS:
        return 0.0
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


# A "hit" is a molecule estimated to be in the library's top HIT_FRAC,
# with the cutoff estimated ONCE from the initial labeled set -- a uniform
# random sample of the library -- and held fixed for the whole campaign.
# Two moving-target alternatives both failed on real runs: the labeled set's
# running top-1000 saturates (~100% of any batch "hits" in round 1, since
# the 1000th best of ~2k random molecules is about the median), and the
# labeled set's own top 1% collapses to ~0 hits by round 3 as the set fills
# with good molecules. Neither lets a bandit tell beta values apart.
HIT_FRAC = 0.01


def hit_threshold(random_sample_y: np.ndarray, frac: float = HIT_FRAC) -> float:
    """Estimated top-`frac` cutoff of the library (lower = better), from a
    uniform random sample of it (the initial labeled set)."""
    k = int(max(1, np.ceil(frac * len(random_sample_y))))
    return float(np.partition(random_sample_y, k - 1)[k - 1])


def hit_mask(new_y: np.ndarray, threshold: float) -> np.ndarray:
    return new_y < threshold


def best_k_mean(labeled_y, k_obs) -> float:
    k = min(k_obs, len(labeled_y))
    return float(np.partition(labeled_y, k - 1)[:k].mean())


def featurize(
    mu: np.ndarray,
    sigma: np.ndarray,
    labeled_y: np.ndarray,
    rnd: int,
    n_rounds: int,
    batch_size: int,
    k_obs: int,
    beta_max: float,
    beta_scale: str,
    last: Optional[LastBatch],
) -> np.ndarray:
    greedy_top = top_k(mu, batch_size)
    explore_top = top_k(ucb(mu, sigma, beta_max, beta_scale), batch_size)
    jacc = len(np.intersect1d(greedy_top, explore_top)) / len(np.union1d(greedy_top, explore_top))
    headroom = float(sigma[greedy_top].mean() / (mu.std() + EPS))

    if last is None:
        is_first, last_reward, rho, rho_sig, improvement, last_beta = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        is_first = 0.0
        last_reward = last.reward
        rho = spearman(last.mu, -last.y)  # mu is higher=better, y lower=better
        rho_sig = spearman(last.sigma, np.abs(last.mu - (-last.y)))
        improvement = (last.best_k_mean_before - best_k_mean(labeled_y, k_obs)) / (labeled_y.std() + EPS)
        last_beta = last.beta / (beta_max + EPS)

    x = np.array([
        1.0, rnd / max(n_rounds, 1), is_first, last_reward, rho, rho_sig,
        headroom, jacc, improvement, last_beta,
    ], dtype=np.float64)
    # Bound the unbounded features so a surrogate with an unusual sigma scale
    # can't push a linear/MLP controller far outside its training range.
    x[6] = np.tanh(x[6])
    x[8] = np.tanh(x[8])
    return x
