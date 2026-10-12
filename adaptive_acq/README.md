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
