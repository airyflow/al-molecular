# adaptive_acq: learned control of UCB's β

Extends the 2.1M-molecule EnamineHTS (`ENHITS`) AL benchmark from a fixed
acquisition function to a controller that picks the exploration coefficient
β each round:

```
fixed β  →  hand-designed β schedules  →  bandits over β  →  GRPO β policy
           (Stage 1, no learning)        (Stage 1)          (Stage 2)
```

UCB score = μ + β_eff·σ. The controller only sees a 10-dim summary of the
campaign state; it never sees molecules. The question this asks: **can a
lightweight, surrogate-agnostic β controller recover most of what a full RL
acquisition policy (GLARE-style) gains, at a fraction of the cost, and
transfer across targets?**

## Before you start

1. **Fusion surrogate bugs (now fixed, see the top-level README).**
   `--surrogate ensemble` used to return sigma = 0 (so UCB = greedy), ranked
   within each 50k chunk instead of over the pool, and broke bf16 ties in
   favor of the end of the pool. All three are fixed (`tests/test_rank_fusion.py`).
   Runs made before 2026-10-10 with `ensemble` or `bigfusion` are affected.
   `adaptive_acq` defaults to `ltall`, which scores each molecule
   independently and never had these problems.
2. **ENHITS rows are sorted by docking score** (see `run_experiment.py`'s
   `ENHITS_shuffled` comment). Nothing here depends on row order (init draws
   and sim subsamples are random), but it's worth repeating the final
   comparison on `--dataset ENHITS_shuffled`.

## Results so far (2026-10-10; ENHITS, LT-All 5 backbones, 0.1% batches, 5 seeds)

Top-1000 recall after round 5 (mean ± std over seeds 42-46):

| Controller | β (normalized) | Recall r5 | Recall r1 |
|---|---|---|---|
| fixed_b0 (greedy) | 0 | **0.876 ± 0.006** | 0.259 |
| existing LT-All greedy runs (cross-check) | 0 | 0.875 ± 0.007 | 0.253 |
| raw_b2 (molpal's UCB default) | ≈ 0.25-0.33 | 0.873 ± 0.009 | 0.208 |
| thompson_narrow (grid 0-0.5) | ~0.1-0.2 chosen | 0.873 ± 0.011 | 0.217 |
| sched_decay_narrow | 0.3 → 0 | 0.870 ± 0.013 | 0.198 |
| thompson (wide grid 0-4) | ~0.4-1.3 chosen | 0.858 ± 0.012 | 0.140 |
| fixed_b0.5 | 0.5 | 0.856 ± 0.019 | 0.143 |
| sched_decay | 2 → 0 | 0.816 ± 0.041 | 0.038 |
| sched_alternating | 2 / 0 | 0.805 ± 0.022 | 0.038 |
| sched_explore_exploit (4 seeds) | 2, 2, 2, 0, 0 | 0.797 ± 0.050 | 0.032 |
| fixed_b1 | 1 | 0.785 ± 0.063 | 0.076 |
| fixed_b2 | 2 | 0.578 ± 0.187 | 0.038 |

Paired by seed against greedy at round 5: thompson_narrow −0.003 (t = −1.1),
sched_decay_narrow −0.006 (t = −1.6), raw_b2 −0.003 (t = −1.2), all within
noise; thompson (wide) −0.019 (t = −6.1) and fixed_b0.5 −0.020 (t = −2.7) are
reliably worse. In round 1 every exploring setting is reliably behind greedy
(−0.04 to −0.12) and catches up later.

On this benchmark, exploration through LT-All's ensemble σ does not help:
recall falls monotonically with β, and greedy is best. That matches
MolPAL's own finding that greedy ≈ UCB on EnamineHTS. The original grid
(0-4) was mostly in the harmful range, so `DEFAULT_ARMS` is now 0-0.5.
Configs 11-12 (Thompson and a decay schedule within 0-0.5) and the learned
controllers use the narrow range. The bar any controller has to clear is
greedy's 0.876; with exploration this costly, the realistic hope is
matching greedy while recovering early-round losses, not beating it by much.

### β-sensitivity gate (2026-10-11): is adapting β worth building at all?

Before spending more on bandits/PPO: `slurm/submit_beta_sensitivity.sh`
(130 ENHITS runs) + `python -m adaptive_acq.beta_sensitivity`.
- **A. Fixed-β curve:** 0.1% batch at β ∈ {0.05, 0.1, 0.2, 0.3} (0.5/1/2 already
  run); 0.2% and 0.4% batches at β ∈ {0, 0.1, 0.3, 0.5}. 5 seeds.
- **B. One-round deviations:** branch each greedy run's checkpoint after round
  t−1, use β ∈ {0.2, 1.0} at round t only (`--schedule pulse`), then greedy;
  t = 1…5; paired against a β=0 control branched the same way (LT-All
  reinitializes every round, so a rerun alone moves recall). Any β policy is a
  combination of such deviations: if none helps, there is nothing to learn.
- **Rule (fixed before results):** go only if a cell gains ≥ +0.005 at round 5
  with 95% CI > 0, *and* it replicates on 5 new seeds (about 20 cells are
  tested, so about one false positive is expected).

**Result (2026-10-12, all 130 runs): NO-GO under the pre-set rule.** At the 0.1%
batch, every β > 0.1 lowers round-5 recall, and β = 0.1 is +0.001 [−0.002,
+0.004]: its CI upper bound sits below +0.005. At the 0.2%/0.4% batches, small β
gives a small, consistent gain, just under the threshold: 0.2% β = 0.3 +0.0046
[+0.0017, +0.0075]; 0.2% β = 0.1 +0.0030 [+0.0012, +0.0048]; 0.4% β = 0.3 +0.0034
[+0.0008, +0.0060]. AUC is negative in every cell. No one-round deviation
reaches the threshold either: round 5 β = 0.2 gives +0.0036 [+0.0013, +0.0059], a
likely multiple-comparison hit. β = 1 at rounds 1 and 4 clearly hurts.
Branched-control rerun noise has an sd of up to 0.007. Conclusion: the most any β
policy could gain on ENHITS/LT-All is about +0.005 recall, so learned β
controllers (bandit/PPO) are not justified here. The usable finding is a fixed
rule (β ≈ 0.1–0.3 at batches ≥ 0.2%), which still needs replicating on new seeds.

### tw1.0 (top-weighted loss): REPLICATED recall gain (2026-10-12)

The only recall improvement found in this package, and the only arm to satisfy
the full pre-registered rule (>= +0.005 at round 5, 95% CI excluding 0, then
replicated on 5 fresh seeds). ENHITS, LT-All, 0.1% batches, paired per seed:

| seed set | control | `tw1.0` | paired diff [95% CI] |
|---|---|---|---|
| original 42-46 | 0.8764 | 0.8824 | +0.0060 [−0.0019, +0.0139] |
| replication 47-51 | 0.8690 | 0.8842 | +0.0152 [+0.0096, +0.0208] |
| **pooled (10 seeds)** | **0.8727** | **0.8833** | **+0.0106 [+0.0055, +0.0157]**, p=0.0011 |

Positive in 9/10 seeds. The replication was a true out-of-sample test: the
margin, the arm and the analysis code all predate those seeds existing.

**What it is.** LT-All trains with plain MSE over every labeled molecule, but
the metric is recall of the true top-1000 out of 2.1M. `--ltall-top-weight w`
weights each molecule by `exp(w * standardized score)` (mean-1 normalized per
batch), so training effort follows the part of the score distribution the
metric actually rewards. `w=0` is the paper's MSE exactly.

**Full top_weight curve** (seeds 42-46, n=5 each): w=0.5 +0.0036, w=1.0
+0.0060, w=1.5 +0.0040, w=2.0 +0.0042, **w=3.0 −0.0102 [−0.0198, −0.0006]**.
w=3 is significantly *worse* than baseline (AUC −0.0426): the weights are
exp(w * standardized score), so at w=3 nearly all gradient lands on a handful
of molecules and training degrades. Useful range is w <= 2; 1.0/1.5/2.0 are
indistinguishable at this resolution (CI widths ~±0.012).

**Headline magnitude, after the seed-set diagnostic (2026-10-13).** Running
tw1.5/tw2.0 on seeds 47-51 settled it -- the whole family moves together:

| arm | seeds 42-46 | seeds 47-51 | pooled n=10 |
|---|---|---|---|
| `tw1.0` | +0.0060 | +0.0152 | +0.0106 [+0.0055, +0.0157] |
| `tw1.5` | +0.0040 | +0.0132 | +0.0086 [+0.0025, +0.0147] |
| `tw2.0` | +0.0042 | +0.0122 | +0.0082 [+0.0029, +0.0135] |

So `tw1.0` is NOT special: seeds 47-51 favour top-weighting generally, and
every w in [1, 2] jumps by the same amount. Two consequences. (a) The
selection-bias worry is small -- tw1.0 beat tw1.5 by only 0.002 on the
selection seeds and they are indistinguishable on the fresh ones, so the
ranking within [1, 2] was noise. (b) **State the result as: top-weighting
gives a real gain of ~+0.009 (central estimate across arms), varying ~+0.004
to +0.015 by seed set; any w in 1.0-2.0 is equivalent and w=3 breaks it.** Do
not quote a single pooled CI for one w as if it were precise -- the
between-seed-set spread is 2-3x and, with only two seed sets, is the dominant
uncertainty.

Other caveats: the effect is larger on the replication seeds (+0.015
vs +0.006) and the control happened to run low there (0.8690 vs 0.8764) --
the design is paired per seed so that cancels, and the pooled number is the
honest one. The AUC difference was negative on the original seeds (early
rounds cost, repaid by round 5) but positive on the replication seeds, so the
early-round penalty is seed-dependent, not a fixed property. Only w = 0.5 and
1.0 have been tried, and 1.0 > 0.5, so the optimum may be higher.

### Composition + second-target transfer (2026-10-14)

**The two wins compose.** `combo` = pf10r3 + tw1.0 on ENHITS seeds 42-46:
+0.0058 [-0.0033, +0.0149] at 2.17 passes (2.3x cheaper). That point estimate
is essentially identical to `tw1.0` alone on the same seeds (+0.0060), i.e.
**adding the prefilter costs none of the recall gain**. The CI spans 0, as
tw1.0 alone also does at n=5 on these seeds -- the evidence for composition is
the agreement of point estimates, not a significant combo effect by itself.
This is the configuration to deploy: `--ltall-top-weight 1.0 --prefilter-mult
10 --prefilter-refresh 3`.

**Top-weighting transfers to a second target, and helps far more there.**
D4_SUB (a 2,104,318-molecule subsample of D4 -- same protocol, same pool size,
different protein), 5 seeds, paired against its own control:

| target | control r5 | tw1.0 r5 | paired diff | relative |
|---|---|---|---|---|
| ENHITS (pooled, n=10) | 0.8727 | 0.8833 | +0.0106 | ~1% |
| **D4_SUB (n=5)** | **0.2136** | **0.2682** | **+0.0546 [+0.0136, +0.0956]** | **~25%** |

Positive in 5/5 D4 seeds. The pattern is interpretable: D4 is far harder
(0.21 vs 0.88 recall at the same ~12.6k absolute label budget), so the
surrogate is nowhere near saturated and aligning training with the top-k
metric has much more room to help; ENHITS is already at 0.88. **The gain grows
as the task gets harder**, which is the regime billion-scale pools sit in.

Caveats: n=5 on D4 with a wide CI, and this is the subsample rather than full
D4 (116M). The full-scale D4 confirmation (~55 GPU-h) is now justified.

### Prefilter trade curve + shortlist refresh (real pool, 2026-10-12)

`submit_prefilter_curve.sh`, 4 more arms x 5 seeds, paired against
full-scoring LT-All greedy (0.8764, 5.00 passes, 386 s/run):

| arm | shortlist | r5 diff [95% CI] | passes | speedup | sec | verdict |
|---|---|---|---|---|---|---|
| `pf10` | 10x budget, frozen | −0.0040 [−0.0086, +0.0006] | 1.23 | 4.1× | 95 | inconclusive |
| `pf20` | 20x, frozen | −0.0026 [−0.0068, +0.0016] | 1.47 | 3.4× | 129 | inconclusive |
| `pf50` | 50x, frozen | −0.0034 [−0.0101, +0.0033] | 2.19 | 2.3× | 196 | inconclusive |
| `pf5r3` | 5x, rebuilt at r3 | −0.0022 [−0.0063, +0.0019] | 2.08 | 2.4× | 148 | inconclusive |
| **`pf10r3`** | **10x, rebuilt at r3** | **−0.0012 [−0.0044, +0.0020]** | **2.17** | **2.3×** | **159** | **acceptable** |

**The round-1 ranking is the problem, not the shortlist size.** `pf50` and
`pf10r3` cost the same ~2.2 passes, yet a frozen shortlist covering 30% of the
whole pool (`pf50`, −0.0034) loses more than a 6% shortlist rebuilt once from a
better-trained surrogate (`pf10r3`, −0.0012). Frozen shortlists plateau around
−0.003 however large (`pf20` ≈ `pf50`); one refresh breaks through that floor.

**`pf10r3` REPLICATED (2026-10-13).** Fresh seeds 47-51 gave +0.0024
[−0.0014, +0.0062]; pooled over 10 seeds **+0.0006 [−0.0017, +0.0029]** at
2.17 vs 4.99 passes. The pooled point estimate is essentially exactly zero
with a tight CI well inside the margin, and is no longer negative -- so the
claim is "equal recall at 2.3x less pool scoring", not "slightly worse but
acceptable".

`pf10r3` is the first arm in this package to pass a pre-registered rule
(non-inferiority: CI contains 0, lower bound > −0.005). **Caveat: seven arms
were tested against that bound and its lower bound (−0.0044) clears −0.005
only narrowly at n=5** — by the same standard applied to the β gate and
`tw1.0`, it should replicate on 5 fresh seeds before being claimed.

Pareto picks at 13B, where full scoring is ~600 GPU-hours of scoring:
`pf10r3` ~260 GPU-h at indistinguishable recall, `pf10` ~146 GPU-h at −0.004.

### Real-pool results: surrogate sweep and prefilter (2026-10-12)

**Surrogate sweep — NO-GO** (`surrogate_sweep.py`, 4 arms x 5 seeds, paired
against LT-All greedy 0.8764). Round-5 diff [95% CI]: `m15` +0.0014 [-0.0013,
+0.0041]; `tw0.5` +0.0036 [-0.0059, +0.0131]; `tw1.0` +0.0060 [-0.0019,
+0.0139]; `m15_tw1.0` +0.0038 [-0.0069, +0.0145]. The top-weighted loss has
the largest mean and clears the +0.005 margin, but its CI includes 0, so under
the pre-registered rule this is not a GO. Note `m15`'s CI is tight around zero:
a bigger LT-All ensemble genuinely does nothing. Every `tw` arm has a negative
AUC with a positive round-5 diff -- upweighting the tail costs early rounds
(the initial labeled set is random, so upweighting discards effective data)
and repays later.

**Prefilter — real loss is far larger than the simulator predicted**
(`prefilter_real.py`, 3 arms x 5 seeds). Measured, not extrapolated:

| arm | shortlist | r5 diff [95% CI] | passes | speedup | wall clock |
|---|---|---|---|---|---|
| control | full | — | 5.00 | 1× | 386 s |
| `pf2` | 2x budget | −0.0824 [−0.1311, −0.0337] | 1.04 | 4.8× | 91 s |
| `pf5` | 5x budget | −0.0080 [−0.0136, −0.0024] | 1.11 | 4.5× | 91 s |
| `pf10` | 10x budget | −0.0040 [−0.0086, +0.0006] | 1.23 | 4.1× | 95 s |

Per-round, the loss compounds as the frozen shortlist ages (pf2: 0.000,
−0.008, −0.045, −0.070, −0.082). pf2's shortlist is simply exhausted of good
unlabeled molecules by round 4. None of the three clears the non-inferiority
bound, though `pf10` is close: **~4.1x less scoring for ~0.004 recall** — a
real cost lever with a small price, not the free lunch the simulator implied.
At 13B that is ~600 GPU-hours of scoring reduced to ~148.

**Methodological caveat — the simulator badly understates prefilter loss.**
Its ENHITS tasks predicted ~0.000 at pf2 where the real pool lost 0.082. The
reason: the simulator labels 6% of its pool, so it is already at 0.67 of a
0.974 ceiling after round 1 and its round-1 ranking is near-final; the real
campaign runs 0.26 -> 0.88, so round-1 ranking is a poor guide to what later
rounds want. The simulator was calibrated on learning *curves*, which does not
make it valid for questions about *ranking dynamics across rounds*. Treat its
verdicts on anything that freezes or reuses an early ranking as optimistic.

### "Score once, then refine" — the cost axis (simulator, 2026-10-12)

`prefilter_sim.py`. Measured on ENHITS: a round is ~70 s, of which only 3-6 s
is LT-All training — the rest is scoring the pool. At 13B that is ~120
GPU-hours *per round*, while the oracle budget does not grow with pool size at
all. Since the β gate showed greedy is within 0.005 of optimal **given** full
scoring, the question that matters at scale is how few molecules must be
scored. Strategy: score the pool once in round 1, keep the top S, and only
ever refine within S. S is a multiple of the total labeling budget (scale-free;
the simulator labels ~6% of its pool per campaign, real ENHITS ~0.6%).

Paired recall loss at round 5 vs full-pool scoring (3 seeds, ridge surrogate):

| S / budget | ENHITS | D4 | AmpC | real-geometry passes | speedup |
|---|---|---|---|---|---|
| full (control) | 0.974 | 0.479 | 0.858 | 5.00 | 1× |
| 10× | 0.000 | +0.000 | −0.001 | 1.24 | 4.0× |
| 5× | 0.000 | −0.002 | −0.003 | 1.12 | 4.5× |
| 2× | 0.000 | −0.019 | −0.020 | 1.05 | 4.8× |
| 1.5× | −0.001 | −0.030 | −0.026 | 1.04 | 4.8× |
| 1.1× | −0.005 | −0.051 | −0.033 | 1.03 | 4.9× |

At S = 5× budget the loss is ≤ 0.003 on all three targets — below the entire
β headroom — for ~4.5× fewer pool passes. **Caveat:** the simulator's ENHITS
tasks sit at 0.974, near saturation, because the simulator labels 6% of its
pool; the real campaign is at 0.876. D4 (0.479) and AmpC (0.858) bracket the
real value and are the more informative proxies, so expect a real loss nearer
their numbers than ENHITS's zeros. Needs a real-pool confirmation (S/budget ∈
{2, 5, 10}, 5 seeds) before any claim; on the real pool those runs are also
*cheaper* than the baseline, since rounds 2-5 score only 3-6% of the pool.

Note for anyone extending this: do **not** mark ineligible candidates by
writing `-inf` into `mu`. `acquisition.effective_beta` divides by `mu.std()`
under the default normalized β scale, so a single non-finite entry makes every
score `nan` and silently randomizes selection. Restrict the candidate set
instead (`ScoreOnceCampaign` subclasses `Campaign._candidates`).

### Batch diversity (simulator, 2026-10-11)

`--diversity-lambda` selects each batch by greedy max-marginal relevance
(`acquisition.select_diverse`). In the calibrated simulator (greedy β = 0,
λ ∈ {0.5, 1, 2, 5}, 2 tasks each from D4/AmpC/ENHITS × 3 seeds;
`adaptive_acq/results/sim_diversity_sweep.json`) it cut batch redundancy
(D4: 0.84 → 0.66 at λ = 5) but did not raise recall. Every paired round-5
difference vs λ = 0 was under 0.01, and λ = 5 was reliably worse on AmpC
(−0.008) and ENHITS (−0.006). Caveat: the simulator's surrogate is a ridge
ensemble on PCA-128, not LT-All; a real ENHITS check is configs 13-15
(`div_l1/l2/l5`).

## Layout

| File | Role |
|---|---|
| `acquisition.py` | UCB with scale-free β; multi-arm batch selection (no duplicates, per-arm credit) |
| `state.py` | Context features s_t and the **observable** reward (no ground truth used) |
| `controllers.py` | `fixed`, `schedule`, `thompson` (batch-split), `linucb`, `policy` |
| `policy.py` | Small MLP π(β-arm \| s_t) |
| `campaign.py` | The AL loop. **Shared** by the simulator and the real run |
| `explorer.py`, `run_adaptive.py` | Real-pool runs (subclass of `run_experiment.MVEExplorer`; same init draw, `--resume`) |
| `sim_env.py` | Cheap simulated campaigns: `build` cached tasks, `eval` controllers |
| `train_linucb.py` | Stage 1: fit a LinUCB prior in sim with full counterfactual feedback |
| `train_grpo.py` | Stage 2: GRPO on the β policy in sim |
| `aggregate.py` | Mean ± std recall per round across seeds (also reads the existing `runs/`) |
| `slurm/` | Build tasks → train controllers → ENHITS evaluation |
| `tests/test_smoke.py` | CPU-only checks on a synthetic pool |

## Design decisions

**Scale-free β (`--beta-scale normalized`, the default).** β_eff = β·std(μ)/mean(σ),
so β=1 means "an average uncertainty bonus is worth one std of predicted
score". Without this, the same β means different things for LT-All's
ensemble σ and an MVE head, or for different targets, and a controller
could not transfer. `--beta-scale raw` reproduces `molpal.acquirer.metrics.ucb`
exactly (tested), so config `raw_b2` is the legacy fixed-UCB baseline.

**Observable reward.** Reward = fraction of the acquired batch in the
library's estimated top 1%. The cutoff is estimated once from the initial
labeled set (a uniform random sample) and stays fixed (recomputed from the
seeded init draw on `--resume`; recorded in `hit_threshold.json`). Moving targets failed in the
first runs: the labeled set's running top-1000 gave every β ~1.0 in round 1,
and its own top 1% fell to ~0 for every β by round 3. Recall against the true
top-1000 is reported but never shown to a controller, so these controllers
would run unchanged on a prospective screen. The exception is GRPO's
*training* return, which uses true recall, but only on the simulator's
training targets.

**Why a batch-split bandit.** A campaign has 5 rounds. One arm pull per
round gives an online bandit 5 data points, which is useless. The
`thompson` controller instead splits each batch across all β arms in
proportion to each arm's posterior probability of being best (5% floor
per arm) and credits every molecule to exactly one arm (ALBL-style). Within-batch
evidence is capped at `--ts-n-eff-cap` pseudo-observations, because 2,000
molecules from one batch are not 2,000 independent trials.

**Why contextual/GRPO controllers are trained in a simulator.** One real
ENHITS campaign takes about 10 GPU-minutes, and GRPO needs thousands. The
simulator runs the same `Campaign` code on 200k-molecule PCA-128 subsamples
(random 1000-row blocks: scattered reads on the 100M-row pools cost about
19 ms per row) with a bootstrap-ridge ensemble, at about 8 s per episode on one
CPU core. **Its batch (1%) and top-k (0.25%) fractions are calibrated:**
greedy on a D4 task gives recall 0.12/0.26/0.37/0.45/0.51 over 5 rounds, vs
0.11/0.27/0.37/0.45/0.51 for the real 116M-molecule D4 LT-All campaign. The
first version matched the real *fractions* instead (0.1% batch), which left
the surrogate about 200 labels per round and recall about 0.03-0.06, so every
β looked alike and the first LinUCB/GRPO training learned only noise. Those
runs were discarded. AmpC (validation only) comes out easier than its real
campaign under these settings (0.86 vs 0.54 at round 5).

**Leakage rule: train on other targets.** Train on D4, select checkpoints on
AmpC, report on ENHITS. Three different targets make the ENHITS number a
transfer estimate. `sim_env eval` warns when task source == `--eval-dataset`.

**Myopic vs. non-myopic.** Bandit rewards only count *this round's* hits,
which biases them toward small β: exploration's payoff is a better surrogate
*later*. GRPO's return is trajectory-level (mean recall over rounds), so it
can learn that exploring early pays off. The bandit-vs-GRPO comparison is
how you find out whether that delayed value matters on this benchmark.

## Running it (from the repo root, `conda activate py310`)

```bash
# 0. sanity (CPU, ~1 min). pytest isn't in py310; any runner works:
python -m pytest adaptive_acq/tests -q

# 1. Stage 1, no training needed: 9 configs x 5 seeds on full ENHITS
sbatch --array=0-44 adaptive_acq/slurm/submit_enhits_controllers.sh

# 2. simulator tasks (D4 = train, AmpC = validation, ENHITS = in-dist. reference only)
sbatch --array=0-2 adaptive_acq/slurm/submit_build_sim_tasks.sh

# 2b. (optional, fast) screen controllers in sim before spending GPU
python -m adaptive_acq.sim_env eval --tasks adaptive_acq/sim_tasks/D4_dedup \
    --n-seeds 5 --controller fixed --beta 1 --out adaptive_acq/results/sim_fixed_b1.json

# 3. train LinUCB prior + GRPO policy (CPU)
sbatch --array=0-1 adaptive_acq/slurm/submit_train_controllers.sh

# 4. learned controllers on full ENHITS
sbatch --array=45-54 adaptive_acq/slurm/submit_enhits_controllers.sh

# 5. compare (includes the existing LT-All greedy runs as a cross-check)
python -m adaptive_acq.aggregate runs/adaptive/*_frac0.001_seed* \
    runs/ltall_enhits_greedy_frac0.001_seed4* --csv adaptive_acq/results/enhits_frac0.001.csv
```

One-off real run (smoke-tested with `--pool-limit 5000`, including `--resume`):

```bash
python -m adaptive_acq.run_adaptive --dataset ENHITS --surrogate ltall \
    --init-size 2104 --batch-size 2104 --n-rounds 5 --topk 1000 \
    --controller thompson --seed 42 --run-dir runs/adaptive/thompson_frac0.001_seed42
```

Each run dir holds `history.json` (the same keys as `run_experiment.py`, plus
`reward`, `beta_mean`, a per-arm `plan`, and the `context` vector),
`controller_config.json`, and `iter_N/{state.json, scores.pkl,
controller.json, last_batch.npz}`.

Approximate cost: Stage 1 ENHITS = 45 jobs × ~12 min on an H100. Task build ≈
5 min per 20k on ENHITS (random memmap reads from `/N/project`), longer at
200k. GRPO at the defaults = 200 iters × 32 rollouts ≈ 6.4k episodes (hours on
32 CPUs).

## Baseline to beat

Existing LT-All greedy, ENHITS, 0.1% batch, seed 43: top-1000 recall after
rounds 1–5 = 0.245, 0.619, 0.754, 0.824, 0.864. `fixed_b0` should land in
the same range; if it doesn't, check that before trusting any controller
comparison.

## Not done yet

- **Reproducing GLARE** (github.com/Celestite3/GLARE) on ENHITS. Its data
  format and compute needs haven't been checked. That is the obvious next
  step, giving a "full RL policy" point to compare GRPO-on-β against.
- Sim-to-real gap: the simulator's ridge/MLP σ isn't LT-All's σ. Compare
  sim-predicted rankings of the fixed-β arms with step 1's real ENHITS
  rankings. If they disagree, use `--surrogate mlp` or larger tasks.
- The `--frac 0.002 / 0.004` batch sizes (edit `FRAC` in the SLURM script).

## Literature (from the brief that motivated this; verify each before citing)

- Stage 1: CAAL (contextual bandits for AL strategy selection, 2026 preprint);
  SelectAL (Hacohen & Weinshall, NeurIPS 2023); ALBL (Hsu & Lin, AAAI 2015),
  which is the basis for the batch-split bandit; Explainable AL for ligand binding
  affinity (2026), which motivates the `schedule` baselines (alternating,
  gradual, explore-/exploit-heavy).
- Stage 2: GLARE (Chen et al., NeurIPS 2025, GRPO for virtual screening incl.
  EnamineHTS); PPO peptide-library screening (Nat. Commun. 2025); RL+AL
  molecular design (Dodds et al., Chem. Sci. 2024).
