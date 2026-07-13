"""Argument parser.

Mirrors the argument surface of kkapusniak/metric-flow-matching
mfm/train/parsers.py for the trajectory (scrna) experiments, with defaults taken
from the official single-cell configs (configs/single_cell/50dims/ot-mfm_*.yaml)
and parsers.py. Image/lidar/unet args are dropped.

Deviations from the official defaults: this port drives plain-torch iteration
loops with early stopping counted in VALIDATION CHECKS (not Lightning epochs), so
the epoch-based patience flags do not map 1:1. --patience (flow) = 50 val-checks
and --patience_geopath = 25 val-checks here, vs 5 epochs each in the official
parsers.py. metric_patience/metric_epochs keep epoch semantics (the RBF fit is
epoch-based) and match parsers.py. sigma follows parsers.py (0.1).
"""

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Metric Flow Matching on LARRY state-fate")

    # data
    p.add_argument("--data", default="larry_pca_mfm.npz",
                   help="npz with day2_pca/day4_pca/day6_pca (+ optional t_day4)")
    p.add_argument("--dim", type=int, default=50, help="PCA dimension used")
    p.add_argument("--t_exclude", type=int, default=1,
                   help="timestep index held out of training (1 = day4)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--split_ratio", type=float, default=0.9)
    p.add_argument("--whiten", action=argparse.BooleanOptionalAction, default=True,
                   help="standardize features (fit on all frames)")

    # metric (data-manifold)
    p.add_argument("--mfm", action=argparse.BooleanOptionalAction, default=True,
                   help="True = metric geodesic path; False = straight-line CondOT")
    p.add_argument("--velocity_metric", default="rbf", choices=["rbf", "land"])
    p.add_argument("--n_centers", type=int, default=150)
    p.add_argument("--kappa", type=float, default=1.5)
    p.add_argument("--rho", type=float, default=-2.75,
                   help="RBF: epsilon (neg -> data-driven); LAND: rho regularizer")
    p.add_argument("--gamma_current", type=float, default=0.125,
                   help="LAND kernel bandwidth (unused for rbf)")
    p.add_argument("--alpha_metric", type=float, default=1.0)
    p.add_argument("--metric_epochs", type=int, default=200)
    p.add_argument("--metric_patience", type=int, default=5)
    p.add_argument("--metric_lr", type=float, default=1e-2)

    # geopath net (Algorithm 1)
    p.add_argument("--time_geopath", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hidden_dims_geopath", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--activation_geopath", default="selu")
    p.add_argument("--geopath_optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--geopath_lr", type=float, default=1e-4)
    p.add_argument("--geopath_weight_decay", type=float, default=1e-5)
    p.add_argument("--geopath_iters", type=int, default=6000)
    p.add_argument("--patience_geopath", type=int, default=25)

    # flow net (Algorithm 2)
    p.add_argument("--hidden_dims_flow", type=int, nargs="+", default=[1024, 1024, 1024])
    p.add_argument("--activation_flow", default="selu")
    p.add_argument("--flow_optimizer", default="adamw", choices=["adam", "adamw"])
    p.add_argument("--flow_lr", type=float, default=1e-3)
    p.add_argument("--flow_weight_decay", type=float, default=1e-5)
    p.add_argument("--flow_iters", type=int, default=15000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.1, help="CFM path noise std")

    # ema / run
    p.add_argument("--ema_decay", type=float, default=0.999,
                   help="set <=0 to disable EMA")
    p.add_argument("--optimal_transport_method", default="exact",
                   choices=["exact", "sinkhorn", "None"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="mfm_predictions.npz")
    p.add_argument("--device", default=None, help="cuda/cpu; auto if unset")

    args = p.parse_args(argv)
    if args.ema_decay is not None and args.ema_decay <= 0:
        args.ema_decay = None
    return args
