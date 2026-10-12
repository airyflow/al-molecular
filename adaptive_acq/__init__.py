"""adaptive_acq -- learned control of the UCB exploration coefficient beta.

Stage 1: fixed beta -> hand-designed beta schedules -> bandits over beta
         (batch-split Thompson sampling, contextual LinUCB).
Stage 2: GRPO-trained beta policy (trajectory-level credit, non-myopic).

Every controller drives the same Campaign loop (campaign.py), used both by
the cheap simulator (sim_env.py, for training) and the real 2.1M-molecule
EnamineHTS run (explorer.py / run_adaptive.py, for evaluation), so what a
controller is trained on and what it is evaluated on can't drift apart.
See README.md.
"""
