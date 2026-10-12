"""Controller CLI flags shared by run_adaptive.py and the simulator scripts."""

from .acquisition import DEFAULT_ARMS
from .controllers import Schedule


def add_controller_args(p):
    g = p.add_argument_group("beta controller")
    g.add_argument("--controller", default="fixed", choices=["fixed", "schedule", "thompson", "linucb", "policy"])
    g.add_argument("--beta-scale", default="normalized", choices=["normalized", "raw"],
                   help="raw = ucb as in molpal.acquirer.metrics (mu + beta*sigma); normalized = beta in "
                        "units of std(mu)/mean(sigma), comparable across surrogates/datasets (see acquisition.py)")
    g.add_argument("--beta", type=float, default=1.0, help="--controller fixed (0 = greedy)")
    g.add_argument("--arms", type=float, nargs="+", default=list(DEFAULT_ARMS),
                   help="beta values available to thompson/linucb")
    g.add_argument("--schedule", default="linear_decay", choices=Schedule.KINDS)
    g.add_argument("--beta-hi", type=float, default=2.0)
    g.add_argument("--beta-lo", type=float, default=0.0)
    g.add_argument("--pulse-round", type=int, default=None,
                   help="--schedule pulse: the one round (1-indexed) that uses --beta-hi")
    g.add_argument("--ts-discount", type=float, default=0.7)
    g.add_argument("--ts-min-frac", type=float, default=0.05)
    g.add_argument("--ts-n-eff-cap", type=float, default=50.0)
    g.add_argument("--linucb-alpha", type=float, default=0.5)
    g.add_argument("--linucb-frozen", action="store_true", help="don't update the loaded LinUCB prior online")
    g.add_argument("--controller-path", default=None, help="LinUCB prior (.json) or GRPO policy (.pt)")
    d = p.add_argument_group("batch diversity")
    d.add_argument("--diversity-lambda", type=float, default=0.0,
                   help="> 0: pick each batch by max-marginal relevance (penalize similarity to molecules "
                        "already in the batch); 0 = plain top-k")
    d.add_argument("--shortlist-mult", type=int, default=10, help="MMR shortlist = this x batch size")
    return g
