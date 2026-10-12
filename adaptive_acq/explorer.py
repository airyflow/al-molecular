"""Real-pool AL run with a beta controller.

AdaptiveMVEExplorer reuses run_experiment.MVEExplorer for everything except
acquisition -- surrogate construction, the seeded initial draw, and
--resume from iter_N/scores.pkl are inherited unchanged, so a fixed-beta run
here starts from exactly the same init set as the existing runs/ baselines
at the same seed. Only run() is replaced, by the shared Campaign loop.
"""

import json
import pickle
from pathlib import Path

import numpy as np

import run_experiment as exp

from .campaign import Campaign
from .state import LastBatch, hit_threshold


class AdaptiveMVEExplorer(exp.MVEExplorer):
    def __init__(self, controller, beta_scale="normalized", diversity_lambda=0.0, shortlist_mult=10,
                 diversity_backbone=None, prefilter_mult=0.0, **kwargs):
        super().__init__(acq="greedy", **kwargs)  # acq_fn unused; Campaign does acquisition
        self.controller = controller
        self.beta_scale = beta_scale
        self.diversity_lambda, self.shortlist_mult = diversity_lambda, shortlist_mult
        self.prefilter_mult = prefilter_mult
        # Similarity for batch diversity (and the batch_redundancy log) uses
        # ONE backbone's embeddings, row-aligned with self.pool_smiles.
        emb = kwargs["emb_dict"]
        bb = diversity_backbone if diversity_backbone in emb else next(iter(emb))
        self._div_emb, self.diversity_backbone = emb[bb], bb
        # Reproduce MVEExplorer's seeded initial draw (same rng call) so the
        # reward's hit cutoff always comes from the uniform random init --
        # also on --resume, when self.labeled_idx is already enriched.
        rng = np.random.default_rng(kwargs.get("seed", 42))
        cand = np.where(self.usable_mask)[0] if self.usable_mask is not None else len(self.pool_smiles)
        self._init_idx = rng.choice(cand, kwargs.get("init_size", 8417), replace=False)

    # ── surrogate / oracle adapters for Campaign ──────────────────────────────
    def _fit_predict(self, labeled_idx, y_hb, cand_idx):
        xs = [self.pool_smiles[i] for i in labeled_idx]
        self.model.train(xs, y_hb.astype(np.float32), epochs=self.surrogate_epochs, batch=self.surrogate_batch)
        pool_smi = [self.pool_smiles[i] for i in cand_idx]
        mu, var = exp._chunked_get_means_and_vars(self.model, pool_smi)
        sigma = np.sqrt(np.maximum(var, 0.0))
        if not np.any(sigma > 0) and max(self.controller.arms) > 0:
            raise SystemExit(
                "surrogate returned sigma == 0 for every candidate (e.g. --surrogate ensemble or "
                "bigfusion, whose predict() returns zeros), so every beta gives the same ranking as "
                "greedy. Use a surrogate with a real uncertainty, e.g. --surrogate ltall.")
        return mu, sigma

    def _score(self, idx):
        return np.array([self.oracle[self.pool_smiles[i]] for i in idx], dtype=np.float64)

    # ── checkpointing ─────────────────────────────────────────────────────────
    def _on_round(self, rec, campaign, controller):
        b = len(campaign.last.y)
        for i, y in zip(campaign.labeled_idx[-b:], campaign.labeled_y[-b:]):
            self.labeled_scores[self.pool_smiles[i]] = float(y)
            self.labeled_idx.add(int(i))
        rnd = rec["round"]
        betas = ", ".join(f"b={v['beta']:.2f}:{v['n']}" for v in rec["plan"].values())
        print(f"  Round {rnd:02d}/{self.n_rounds}  labeled={rec['n_labeled']:,}  best={rec['best_score']:.3f}  "
              f"top-{self.topk} recall={rec['topk_recall']:.1%}  reward={rec['reward']:.3f}  [{betas}]  "
              f"({rec['elapsed']:.1f}s)", flush=True)
        self._checkpoint(rnd, rec)
        if campaign.shortlist is not None and not self._sl_path.exists():
            np.save(self._sl_path, np.flatnonzero(campaign.shortlist))
        d = self.run_dir / f"iter_{rnd}"
        (d / "controller.json").write_text(json.dumps(controller.state_dict()))
        last = campaign.last
        np.savez(d / "last_batch.npz", mu=last.mu, sigma=last.sigma, y=last.y,
                 meta=np.array([last.reward, last.beta, last.best_k_mean_before]))

    def _restore(self, campaign):
        d = self.run_dir / f"iter_{self._start_round}"
        if (d / "controller.json").exists():
            self.controller.load_state_dict(json.loads((d / "controller.json").read_text()))
        if (d / "last_batch.npz").exists():
            z = np.load(d / "last_batch.npz")
            r, b, bk = z["meta"].tolist()
            campaign.last = LastBatch(mu=z["mu"], sigma=z["sigma"], y=z["y"], reward=r, beta=b, best_k_mean_before=bk)

    def run(self) -> list:
        n = len(self.pool_smiles)
        true_top = np.fromiter((s in self.true_top_k for s in self.pool_smiles), dtype=bool, count=n)
        init_y = self._score(self._init_idx)
        if self._start_round == 0:
            assert set(self._init_idx.tolist()) == self.labeled_idx, "init draw mismatch"
        hit_thr = hit_threshold(init_y)
        campaign = Campaign(
            n_pool=n, fit_predict=self._fit_predict, score_fn=self._score,
            init_idx=np.array(sorted(self.labeled_idx), dtype=np.int64),
            batch_size=self.batch_size, n_rounds=self.n_rounds, k_obs=self.topk,
            beta_scale=self.beta_scale, true_top_mask=true_top,
            candidate_mask=self.usable_mask, start_round=self._start_round, hit_thr=hit_thr,
            diversity_lambda=self.diversity_lambda, shortlist_mult=self.shortlist_mult,
            prefilter_mult=self.prefilter_mult,
            feats_fn=lambda idx: np.asarray(self._div_emb[idx]),
        )
        (self.run_dir / "hit_threshold.json").write_text(json.dumps({"hit_threshold": hit_thr}))
        # The shortlist is fixed in round 1 and must survive --resume; without
        # it a resumed run would silently go back to scoring the whole pool.
        sl_path = self.run_dir / "shortlist.npy"
        if self.prefilter_mult > 0 and self._start_round > 0:
            if not sl_path.exists():
                raise SystemExit(f"--prefilter-mult given and resuming at round {self._start_round + 1}, "
                                 f"but {sl_path} is missing -- cannot reproduce the round-1 shortlist")
            mask = np.zeros(n, dtype=bool)
            mask[np.load(sl_path)] = True
            campaign.shortlist = mask
            print(f"[resume] restored shortlist of {int(mask.sum()):,} molecules")
        self._sl_path = sl_path
        if self._start_round > 0:
            self._restore(campaign)
        (self.run_dir / "controller_config.json").write_text(
            json.dumps({**self.controller.describe(), "beta_scale": self.beta_scale,
                        "diversity_lambda": self.diversity_lambda, "shortlist_mult": self.shortlist_mult,
                        "diversity_backbone": self.diversity_backbone}, indent=2))
        history = list(self._resumed_history) + campaign.run(self.controller, on_round=self._on_round)
        self._save_final(history)
        return history
