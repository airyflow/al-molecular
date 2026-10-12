"""UCB scoring with a scale-free beta, and multi-arm batch selection.

Convention throughout adaptive_acq: mu is "higher = better" (the surrogate is
trained on -docking_score, exactly like run_experiment.MVEExplorer), sigma is
a standard deviation in the same units as mu (NOT a variance).
"""

import numpy as np

EPS = 1e-12

# Default arm set, in normalized-beta units (see effective_beta). 0 == greedy.
# Narrowed 2026-10-10 from (0, 0.25, 0.5, 1, 2, 4) after the first full ENHITS
# sweep (LT-All, 5 seeds): recall at round 5 fell monotonically with beta
# (0: 0.876, 0.5: 0.856, 1: 0.785, 2: 0.578), and molpal's raw beta=2
# corresponds to only ~0.25-0.33 here (0.873). The useful range is [0, 0.5].
DEFAULT_ARMS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
WIDE_ARMS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)  # the original grid, for reproducing thompson_frac*


def effective_beta(mu: np.ndarray, sigma: np.ndarray, beta: float, beta_scale: str) -> float:
    """Convert a controller's beta into the coefficient actually multiplied
    onto sigma.

    "raw": beta is used as-is -- ucb = mu + beta*sigma, identical to
        molpal.acquirer.metrics.ucb (whose default is beta=2). Use this to
        reproduce the existing fixed-UCB baselines.
    "normalized": beta is in units of std(mu)/mean(sigma), so beta=1 means
        "an average-sized uncertainty bonus is worth one standard deviation
        of predicted score". Rescaling mu and sigma together doesn't change
        a UCB ranking, but the RATIO std(mu)/mean(sigma) differs between
        surrogates (LT-All's 5-member std vs an MVE head) and targets --
        normalizing it out is what lets one controller transfer across
        surrogates/datasets with the same arm set.
    """
    if beta_scale == "raw":
        return float(beta)
    if beta_scale == "normalized":
        return float(beta) * float(mu.std()) / (float(sigma.mean()) + EPS)
    raise ValueError(f"unknown beta_scale {beta_scale!r}")


def ucb(mu, sigma, beta, beta_scale="normalized"):
    return mu + effective_beta(mu, sigma, beta, beta_scale) * sigma


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, sorted best-first (argpartition, so
    O(n) rather than a full argsort over a ~2M-candidate pool)."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part], kind="stable")]


def select_batch(mu, sigma, plan, beta_scale="normalized"):
    """Pick one batch shared among several (beta, count) arms.

    plan: list of (arm_index, beta, count). A single-arm plan is ordinary
    UCB acquisition. With several arms, quotas are filled round-robin, each
    arm taking its own best not-yet-taken candidate, so no molecule is
    acquired twice and every molecule is credited to exactly one arm
    (needed for the batch-split bandit's per-arm reward).

    Returns (selected local indices, arm index per selected molecule).
    """
    total = sum(c for _, _, c in plan)
    rankings, cursor, quota = {}, {}, {}
    for arm, beta, count in plan:
        if count <= 0:
            continue
        # count + total candidates always suffice: at most `total` of this
        # arm's top entries can already have been taken by other arms.
        rankings[arm] = top_k(ucb(mu, sigma, beta, beta_scale), count + total)
        cursor[arm], quota[arm] = 0, count

    taken = np.zeros(len(mu), dtype=bool)
    sel, owner = [], []
    arms, n_pass = list(quota), 0
    while any(q > 0 for q in quota.values()):
        # Rotate which arm picks first each pass: candidates shared by
        # several arms' rankings shouldn't always be credited to the same arm.
        k = n_pass % len(arms)
        n_pass += 1
        for arm in arms[k:] + arms[:k]:
            if quota[arm] == 0:
                continue
            r = rankings[arm]
            while cursor[arm] < len(r) and taken[r[cursor[arm]]]:
                cursor[arm] += 1
            if cursor[arm] == len(r):  # candidate pool exhausted
                quota[arm] = 0
                continue
            i = r[cursor[arm]]
            taken[i] = True
            sel.append(i)
            owner.append(arm)
            quota[arm] -= 1
    return np.asarray(sel, dtype=np.int64), np.asarray(owner, dtype=np.int64)


def select_diverse(score, feats_fn, batch_size, lam, shortlist_mult=10, device=None):
    """Greedy max-marginal-relevance batch: repeatedly take
        argmax_i  z(score_i) - lam * max_{j in batch} cos(x_i, x_j)
    over a shortlist of the top shortlist_mult * batch_size candidates by
    score (z = score standardized within the shortlist; x = centered,
    L2-normalized features). lam = 0 is exactly top_k(score, batch_size).

    Greedy batches of thousands from a 2M pool tend to be near-duplicates of
    one another (one predicted-good scaffold, many decorations); every
    duplicate spends a docking call that teaches the surrogate little and
    finds few new top-k molecules. This trades a little predicted score for
    coverage. A cheap stand-in for a k-DPP: one matmul per pick, so
    batch_size x shortlist x dim work (~1e10 flops for ENHITS' 2104 x 21k x 768).

    score: (n,) higher = better (e.g. ucb(mu, sigma, beta)); feats_fn(local
    indices) -> (m, d) features for those candidates. Returns local indices.
    """
    if lam <= 0:
        return top_k(score, batch_size)
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    short = top_k(score, min(len(score), shortlist_mult * batch_size))
    s = score[short]
    z = torch.as_tensor((s - s.mean()) / (s.std() + EPS), dtype=torch.float32, device=dev)
    F = torch.as_tensor(np.asarray(feats_fn(short), dtype=np.float32), device=dev)
    F = F - F.mean(0, keepdim=True)  # centering: raw embeddings are anisotropic (all cosines high)
    F = F / (F.norm(dim=1, keepdim=True) + EPS)
    maxsim = torch.zeros(len(short), device=dev)
    taken = torch.zeros(len(short), dtype=torch.bool, device=dev)
    picks = []
    for _ in range(min(batch_size, len(short))):
        obj = (z - lam * maxsim).masked_fill(taken, float("-inf"))
        j = int(torch.argmax(obj))
        picks.append(j)
        taken[j] = True
        maxsim = torch.maximum(maxsim, F @ F[j])
    return short[np.asarray(picks, dtype=np.int64)]


def batch_redundancy(feats: np.ndarray) -> float:
    """Mean over the batch of each molecule's highest cosine similarity to
    another batch member (centered features) -- a diagnostic for how
    near-duplicate a batch is; 1.0 = every molecule has a twin."""
    import torch

    F = torch.as_tensor(np.asarray(feats, dtype=np.float32))
    F = F - F.mean(0, keepdim=True)
    F = F / (F.norm(dim=1, keepdim=True) + EPS)
    S = F @ F.T
    S.fill_diagonal_(-1.0)
    return float(S.max(dim=1).values.mean())
