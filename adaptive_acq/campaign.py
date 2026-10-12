"""One AL campaign driven by a beta controller -- shared by the simulator and
the real 2.1M-molecule run so training and evaluation use identical logic.

The surrogate is abstracted as fit_predict(labeled_idx, y_hb, cand_idx) ->
(mu, sigma) over cand_idx (y_hb / mu: higher = better; sigma: std, not var).
The oracle is score_fn(idx) -> docking scores (lower = better).
"""

import time
from typing import Callable, Optional

import numpy as np

from .acquisition import batch_redundancy, select_batch, select_diverse, top_k, ucb
from .state import LastBatch, best_k_mean, featurize, hit_mask, hit_threshold


class Campaign:
    def __init__(
        self,
        n_pool: int,
        fit_predict: Callable,
        score_fn: Callable,
        init_idx: np.ndarray,
        batch_size: int,
        n_rounds: int,
        k_obs: int,
        beta_scale: str = "normalized",
        true_top_mask: Optional[np.ndarray] = None,
        candidate_mask: Optional[np.ndarray] = None,
        counterfactual: bool = False,
        start_round: int = 0,
        hit_thr: Optional[float] = None,
        diversity_lambda: float = 0.0,
        shortlist_mult: int = 10,
        feats_fn: Optional[Callable] = None,
    ):
        """
        k_obs: top-k used for the "improvement" context feature -- set to the
            evaluation top-k (1000 for EnamineHTS).
        hit_thr: fixed hit cutoff for the observable reward (see
            state.HIT_FRAC). None = estimate it from init_idx, which must then
            be the campaign's uniform random initial draw -- on resume, pass
            the value saved from the original start instead.
        true_top_mask: bool over the pool, True for the true top-k. Used ONLY
            to report recall; never shown to the controller.
        candidate_mask: molecules allowed to be acquired (e.g. oracle-scored);
            None = all.
        counterfactual: also score every arm's would-be batch with the oracle
            and pass those rewards to the controller (simulator only).
        diversity_lambda: > 0 selects each batch by greedy max-marginal
            relevance (acquisition.select_diverse) on the controller's UCB
            score instead of plain top-k; needs a single-arm plan and
            feats_fn(pool_idx) -> (m, d) features. 0 = original behavior.
        feats_fn: also used, when given, to log each batch's redundancy.
        """
        self.n_pool, self.fit_predict, self.score_fn = n_pool, fit_predict, score_fn
        self.batch_size, self.n_rounds, self.k_obs = batch_size, n_rounds, k_obs
        self.beta_scale, self.counterfactual = beta_scale, counterfactual
        self.true_top_mask = true_top_mask
        self.true_k = int(true_top_mask.sum()) if true_top_mask is not None else None
        self.candidate_mask = candidate_mask
        self.start_round = start_round
        self.diversity_lambda, self.shortlist_mult, self.feats_fn = diversity_lambda, shortlist_mult, feats_fn
        if diversity_lambda > 0 and feats_fn is None:
            raise ValueError("diversity_lambda > 0 needs feats_fn")

        init_idx = np.asarray(init_idx, dtype=np.int64)
        self.labeled = np.zeros(n_pool, dtype=bool)
        self.labeled[init_idx] = True
        self.labeled_idx = init_idx.copy()
        self.labeled_y = np.asarray(score_fn(init_idx), dtype=np.float64)
        self.hit_thr = hit_threshold(self.labeled_y) if hit_thr is None else float(hit_thr)
        self.last: Optional[LastBatch] = None

    def recall(self) -> Optional[float]:
        if self.true_top_mask is None:
            return None
        return float(self.true_top_mask[self.labeled_idx].sum() / self.true_k)

    def _candidates(self):
        avail = ~self.labeled
        if self.candidate_mask is not None:
            avail &= self.candidate_mask
        return np.flatnonzero(avail)

    def step(self, controller, rnd: int) -> dict:
        t0 = time.perf_counter()
        cand = self._candidates()
        mu, sigma = self.fit_predict(self.labeled_idx, -self.labeled_y, cand)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0)
        t_model = time.perf_counter() - t0

        beta_max = max(controller.arms)
        ctx = featurize(mu, sigma, self.labeled_y, rnd, self.n_rounds, self.batch_size,
                        self.k_obs, beta_max if beta_max > 0 else 1.0, self.beta_scale, self.last)
        plan = controller.plan(ctx, rnd, self.n_rounds, self.batch_size)
        if self.diversity_lambda > 0:
            if len(plan) != 1:
                raise ValueError("diversity selection supports single-arm plans only (not the batch-split bandit)")
            arm, beta, count = plan[0]
            sel = select_diverse(ucb(mu, sigma, beta, self.beta_scale), lambda loc: self.feats_fn(cand[loc]),
                                 count, self.diversity_lambda, self.shortlist_mult)
            owner = np.full(len(sel), arm, dtype=np.int64)
        else:
            sel, owner = select_batch(mu, sigma, plan, self.beta_scale)
        new_idx = cand[sel]
        new_y = np.asarray(self.score_fn(new_idx), dtype=np.float64)
        hits = hit_mask(new_y, self.hit_thr)

        cf = None
        if self.counterfactual:
            cf = {}
            for i, beta in enumerate(controller.arms):
                alt = cand[top_k(ucb(mu, sigma, beta, self.beta_scale), self.batch_size)]
                cf[i] = float(hit_mask(np.asarray(self.score_fn(alt)), self.hit_thr).mean())
        controller.update(ctx, plan, owner, hits, counterfactual=cf)

        counts = np.array([c for _, _, c in plan], dtype=np.float64)
        betas = np.array([b for _, b, _ in plan], dtype=np.float64)
        self.last = LastBatch(
            mu=mu[sel], sigma=sigma[sel], y=new_y, reward=float(hits.mean()),
            beta=float((counts * betas).sum() / max(counts.sum(), 1)),
            best_k_mean_before=best_k_mean(self.labeled_y, self.k_obs),
        )

        self.labeled[new_idx] = True
        self.labeled_idx = np.concatenate([self.labeled_idx, new_idx])
        self.labeled_y = np.concatenate([self.labeled_y, new_y])

        per_arm = {}
        for arm, beta, count in plan:
            m = owner == arm
            per_arm[str(arm)] = {"beta": beta, "n": int(m.sum()), "hits": int(hits[m].sum())}
        rec = dict(
            round=rnd + 1, n_labeled=int(len(self.labeled_idx)),
            best_score=float(self.labeled_y.min()), topk_recall=self.recall(),
            reward=float(hits.mean()), beta_mean=self.last.beta, plan=per_arm,
            context=[round(float(v), 5) for v in ctx],
            elapsed=round(time.perf_counter() - t0, 2), model_time=round(t_model, 2),
            diversity_lambda=self.diversity_lambda,
        )
        if self.feats_fn is not None:
            rec["batch_redundancy"] = round(batch_redundancy(self.feats_fn(new_idx)), 4)
        if cf is not None:
            rec["counterfactual_reward"] = {str(k): v for k, v in cf.items()}
        return rec

    def run(self, controller, on_round: Optional[Callable] = None) -> list:
        history = []
        for rnd in range(self.start_round, self.n_rounds):
            rec = self.step(controller, rnd)
            history.append(rec)
            if on_round is not None:
                on_round(rec, self, controller)
        return history
