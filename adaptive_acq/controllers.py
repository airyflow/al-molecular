"""beta controllers: given the campaign state, decide how this round's batch
is acquired.

A controller returns a *plan*: a list of (arm_index, beta, count) whose counts
sum to the batch size. Single-arm plans are ordinary UCB at one beta; the
batch-split bandit spreads one batch across several betas so it can learn
online within a single 5-round campaign (one arm pull per round would give
it only 5 observations in total).

  fixed        beta_t = beta                         (baseline, incl. greedy)
  schedule     beta_t = f(t), hand-designed           (Explainable-AL protocols)
  thompson     batch split across arms, Beta-Bernoulli Thompson sampling
               on the per-arm hit rate                (ALBL-style, online)
  linucb       contextual: arm = argmax_a theta_a.x + alpha*||x||_{A_a^-1}
               (CAAL-style; prior fit offline in the simulator, optionally
               keeps updating online)
  policy       MLP pi(a | s_t) trained with GRPO      (Stage 2)
"""

import json
from pathlib import Path

import numpy as np

from .acquisition import DEFAULT_ARMS
from .state import FEATURE_DIM, FEATURE_NAMES


def _single(arm, beta, batch_size):
    return [(arm, float(beta), int(batch_size))]


class Controller:
    name = "base"
    arms = DEFAULT_ARMS

    def plan(self, ctx, rnd, n_rounds, batch_size):
        raise NotImplementedError

    def update(self, ctx, plan, owner, hits, counterfactual=None):
        """owner: arm index per acquired molecule; hits: bool per acquired
        molecule. counterfactual: {arm: reward had the whole batch used that
        arm} -- only the simulator can provide this (it needs the oracle on
        molecules that were NOT acquired)."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict):
        pass

    def describe(self) -> dict:
        return {"controller": self.name, "arms": list(self.arms)}


class FixedBeta(Controller):
    name = "fixed"

    def __init__(self, beta: float):
        self.beta = float(beta)
        self.arms = (self.beta,)

    def plan(self, ctx, rnd, n_rounds, batch_size):
        return _single(0, self.beta, batch_size)

    def describe(self):
        return {"controller": self.name, "beta": self.beta}


class Schedule(Controller):
    """Hand-designed beta schedules over rounds -- the non-learned adaptive
    protocols (alternating, gradually changing, explore-then-exploit) that
    any learned controller has to beat to be worth its cost."""
    name = "schedule"
    KINDS = ("linear_decay", "linear_increase", "alternating", "explore_then_exploit", "pulse")

    def __init__(self, kind: str, beta_hi: float = 2.0, beta_lo: float = 0.0, pulse_round=None):
        """pulse: beta_hi at round `pulse_round` (1-indexed, as in history.json's
        "round") and beta_lo everywhere else -- the one-round deviation used to
        measure how much adapting beta could possibly be worth."""
        if kind not in self.KINDS:
            raise ValueError(f"schedule kind must be one of {self.KINDS}")
        if kind == "pulse" and pulse_round is None:
            raise ValueError("schedule pulse needs pulse_round")
        self.kind, self.beta_hi, self.beta_lo = kind, float(beta_hi), float(beta_lo)
        self.pulse_round = pulse_round
        self.arms = (self.beta_lo, self.beta_hi)

    def beta_at(self, rnd, n_rounds):
        frac = rnd / max(n_rounds - 1, 1)
        if self.kind == "linear_decay":
            return self.beta_hi + (self.beta_lo - self.beta_hi) * frac
        if self.kind == "linear_increase":
            return self.beta_lo + (self.beta_hi - self.beta_lo) * frac
        if self.kind == "alternating":
            return self.beta_hi if rnd % 2 == 0 else self.beta_lo
        if self.kind == "pulse":
            return self.beta_hi if rnd + 1 == self.pulse_round else self.beta_lo
        return self.beta_hi if rnd < n_rounds / 2 else self.beta_lo

    def plan(self, ctx, rnd, n_rounds, batch_size):
        return _single(0, self.beta_at(rnd, n_rounds), batch_size)

    def describe(self):
        return {"controller": self.name, "kind": self.kind, "beta_hi": self.beta_hi, "beta_lo": self.beta_lo,
                "pulse_round": self.pulse_round}


class BatchSplitThompson(Controller):
    """Each round's batch is divided among all arms in proportion to each
    arm's posterior probability of having the best hit rate (plus a floor,
    so every arm keeps getting some feedback).

    Known bias, inherited from any myopic reward: the hit rate only measures
    what an arm finds *this* round, not how much its picks improve the next
    surrogate. That pushes it toward small beta. GRPO (Stage 2) is the
    comparison that tests whether the delayed value of exploration matters.
    """
    name = "thompson"

    def __init__(self, arms=DEFAULT_ARMS, discount=0.7, min_frac=0.05, n_eff_cap=50.0,
                 n_samples=4000, seed=0):
        self.arms = tuple(float(a) for a in arms)
        self.discount, self.min_frac, self.n_eff_cap = discount, min_frac, n_eff_cap
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.a = np.ones(len(self.arms))
        self.b = np.ones(len(self.arms))

    def prob_best(self):
        draws = self.rng.beta(self.a[:, None], self.b[:, None], size=(len(self.arms), self.n_samples))
        wins = np.bincount(draws.argmax(axis=0), minlength=len(self.arms))
        return wins / wins.sum()

    def plan(self, ctx, rnd, n_rounds, batch_size):
        k = len(self.arms)
        floor = self.min_frac * batch_size
        share = floor + (batch_size - floor * k) * self.prob_best()
        counts = np.floor(share).astype(int)
        # largest-remainder rounding so counts sum to batch_size exactly
        for i in np.argsort(-(share - counts))[: batch_size - counts.sum()]:
            counts[i] += 1
        return [(i, self.arms[i], int(c)) for i, c in enumerate(counts)]

    def update(self, ctx, plan, owner, hits, counterfactual=None):
        self.a = 1.0 + self.discount * (self.a - 1.0)
        self.b = 1.0 + self.discount * (self.b - 1.0)
        for i in range(len(self.arms)):
            m = owner == i
            n = int(m.sum())
            if n == 0:
                continue
            h = float(hits[m].sum())
            # Molecules within one batch are far from independent draws, so a
            # 4000-molecule slice must not count as 4000 Bernoulli trials --
            # cap each round's evidence at n_eff_cap pseudo-observations.
            w = min(1.0, self.n_eff_cap / n)
            self.a[i] += w * h
            self.b[i] += w * (n - h)

    def state_dict(self):
        return {"a": self.a.tolist(), "b": self.b.tolist()}

    def load_state_dict(self, d):
        self.a, self.b = np.array(d["a"]), np.array(d["b"])

    def describe(self):
        return {"controller": self.name, "arms": list(self.arms), "discount": self.discount,
                "min_frac": self.min_frac, "n_eff_cap": self.n_eff_cap}


class LinUCB(Controller):
    """Disjoint LinUCB over the beta arms, context = state.featurize().

    Within one real campaign there are only n_rounds (=5) pulls, far too few
    to learn a 10-dim linear model per arm from scratch, so the intended use
    is: fit the prior offline in the simulator (train_linucb.py, with
    full-information counterfactual rewards), save it, then load it here for
    the real run -- optionally continuing to update online (online=True).
    """
    name = "linucb"

    def __init__(self, arms=DEFAULT_ARMS, alpha=0.5, ridge=1.0, online=True):
        self.arms = tuple(float(a) for a in arms)
        self.alpha, self.ridge, self.online = alpha, ridge, online
        self.A = np.stack([ridge * np.eye(FEATURE_DIM) for _ in self.arms])
        self.bvec = np.zeros((len(self.arms), FEATURE_DIM))

    def scores(self, x):
        out = np.empty(len(self.arms))
        for i in range(len(self.arms)):
            Ainv = np.linalg.inv(self.A[i])
            theta = Ainv @ self.bvec[i]
            out[i] = theta @ x + self.alpha * np.sqrt(x @ Ainv @ x)
        return out

    def plan(self, ctx, rnd, n_rounds, batch_size):
        i = int(np.argmax(self.scores(ctx)))
        return _single(i, self.arms[i], batch_size)

    def _add(self, i, x, r):
        self.A[i] += np.outer(x, x)
        self.bvec[i] += r * x

    def update(self, ctx, plan, owner, hits, counterfactual=None):
        if counterfactual is not None:
            for i, r in counterfactual.items():
                self._add(i, ctx, r)
        elif self.online:
            i = plan[0][0]
            self._add(i, ctx, float(hits.mean()) if len(hits) else 0.0)

    def state_dict(self):
        return {"arms": list(self.arms), "alpha": self.alpha, "ridge": self.ridge,
                "feature_names": list(FEATURE_NAMES), "A": self.A.tolist(), "b": self.bvec.tolist()}

    def load_state_dict(self, d):
        if tuple(d["feature_names"]) != FEATURE_NAMES:
            raise ValueError("LinUCB prior was fit with a different feature set")
        self.arms = tuple(d["arms"])
        self.A, self.bvec = np.array(d["A"]), np.array(d["b"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict()))

    def describe(self):
        return {"controller": self.name, "arms": list(self.arms), "alpha": self.alpha, "online": self.online}


class PolicyController(Controller):
    """Wraps a policy network (policy.BetaPolicy). mode="sample" draws actions
    and records (ctx, action, logp) for GRPO training; mode="greedy" takes the
    argmax and is what evaluation/deployment uses."""
    name = "policy"

    def __init__(self, policy, arms=DEFAULT_ARMS, mode="greedy", seed=0):
        self.policy, self.arms, self.mode = policy, tuple(float(a) for a in arms), mode
        self.rng = np.random.default_rng(seed)
        self.trajectory = []

    def plan(self, ctx, rnd, n_rounds, batch_size):
        probs = self.policy.probs(ctx)
        i = int(self.rng.choice(len(probs), p=probs)) if self.mode == "sample" else int(np.argmax(probs))
        self.trajectory.append((ctx.copy(), i, float(np.log(probs[i] + 1e-12))))
        return _single(i, self.arms[i], batch_size)

    def describe(self):
        return {"controller": self.name, "arms": list(self.arms), "mode": self.mode}


def build_controller(args) -> Controller:
    """Factory used by run_adaptive.py and the simulator CLIs."""
    arms = tuple(args.arms) if getattr(args, "arms", None) else DEFAULT_ARMS
    c = args.controller
    if c == "fixed":
        return FixedBeta(args.beta)
    if c == "schedule":
        return Schedule(args.schedule, beta_hi=args.beta_hi, beta_lo=args.beta_lo,
                        pulse_round=getattr(args, "pulse_round", None))
    if c == "thompson":
        return BatchSplitThompson(arms=arms, discount=args.ts_discount, min_frac=args.ts_min_frac,
                                  n_eff_cap=args.ts_n_eff_cap, seed=args.seed)
    if c == "linucb":
        ctrl = LinUCB(arms=arms, alpha=args.linucb_alpha, online=not args.linucb_frozen)
        if args.controller_path:
            ctrl.load_state_dict(json.loads(Path(args.controller_path).read_text()))
        else:
            print("[linucb] WARNING: no --controller-path prior given -- starting from an "
                  "uninformed model, which cannot learn much in a 5-round campaign")
        return ctrl
    if c == "policy":
        from .policy import BetaPolicy
        if not args.controller_path:
            raise SystemExit("--controller policy requires --controller-path <policy.pt>")
        policy, arms = BetaPolicy.load(args.controller_path)
        return PolicyController(policy, arms=arms, mode="greedy", seed=args.seed)
    raise ValueError(f"unknown controller {c!r}")
