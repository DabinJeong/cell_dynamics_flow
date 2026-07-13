"""Entry point: train MFM and straight-line FM, predict the held-out marginal.

Plain-torch analogue of kkapusniak/metric-flow-matching mfm/train/main.py.
Two-stage per model, exactly as the paper:
  ALGO 1 (MFM only): fit the data-manifold metric, then train the geodesic
                     interpolant (geopath net) to minimize metric geodesic energy.
  ALGO 2:            train the velocity field by (metric) flow matching, then
                     integrate to predict the held-out timepoint (day4) and day6.

Runs BOTH the MFM model (mfm=True, geodesic path) and the straight-line FM
baseline (mfm=False, CondOT path) under identical data/coupling/network budget,
so the only difference is the conditional path — the paper's core comparison.
Saves mfm_predictions.npz with fm_/mfm_ pred_day4 and pred_day6 in ORIGINAL
(un-whitened) PCA coordinates.
"""

import argparse
import copy
import time

import numpy as np
import torch

from mfm.utils import set_seed, wasserstein_distance
from mfm.train.parsers import parse_args
from mfm.dataloaders.trajectory_data import (
    TemporalData, OTPlanSampler, load_larry_frames,
)
from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.flow_matchers.ema import EMA
from mfm.flow_matchers.geopath_net_train import GeoPathTrainer
from mfm.flow_matchers.flow_net_train import FlowTrainer
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.geopath_networks.mlp import GeoPathMLP


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build_data(args):
    frames, times = load_larry_frames(args.data, max_dim=args.dim)
    skipped = [args.t_exclude] if args.t_exclude is not None else []
    data = TemporalData(
        frames, times, batch_size=args.batch_size, split_ratio=args.split_ratio,
        skipped_time_points=skipped, device=args.device, seed=args.seed,
        whiten=args.whiten,
    )
    return data, frames, times, skipped


def run_one(args, mfm_on, data):
    """Train one model (mfm_on True/False) and return endpoint predictions in
    the model's working (whitened) space, plus the trained flow trainer."""
    args = copy.deepcopy(args)
    args.mfm = mfm_on
    set_seed(args.seed)

    flow_net = VelocityNet(dim=args.dim, hidden_dims=args.hidden_dims_flow,
                           activation=args.activation_flow, batch_norm=False).to(args.device)
    geopath_net = GeoPathMLP(input_dim=args.dim, hidden_dims=args.hidden_dims_geopath,
                             time_geopath=args.time_geopath,
                             activation=args.activation_geopath,
                             batch_norm=False).to(args.device)
    if args.ema_decay is not None:
        flow_net = EMA(flow_net, decay=args.ema_decay)
        geopath_net = EMA(geopath_net, decay=args.ema_decay)

    ot_sampler = (OTPlanSampler(method=args.optimal_transport_method)
                  if args.optimal_transport_method != "None" else None)

    flow_matcher = MetricFlowMatcher(geopath_net=geopath_net, sigma=args.sigma,
                                     alpha=int(mfm_on))

    # ---- ALGO 1: geodesic interpolant (MFM only) ----
    if mfm_on:
        metric = DataManifoldMetric(args, skipped_time_points=data.skipped_time_points,
                                    num_timesteps=data.num_timesteps)
        if args.velocity_metric == "rbf":
            log("fitting RBF metric networks ...")
            metric.fit_rbf(data.interval_landmarks())
        log("=== MFM ALGO 1: training geodesic interpolant ===")
        GeoPathTrainer(flow_matcher, data, metric, ot_sampler, args).train(
            iters=args.geopath_iters, patience=args.patience_geopath)
        flow_matcher.geopath_net = geopath_net

    # ---- ALGO 2: (metric) flow matching ----
    tag = "MFM" if mfm_on else "straight-FM"
    log(f"=== {tag} ALGO 2: flow matching ===")
    ft = FlowTrainer(flow_matcher, flow_net, data, ot_sampler, args)
    ft.train(iters=args.flow_iters, patience=args.patience)

    # ---- predict held-out (day4) + day6 from day2 (integrate from t=0) ----
    x0 = data.all_frames[0]  # day2 (whitened) full frame
    t0 = data.times[0]
    t_mid = data.times[args.t_exclude] if args.t_exclude is not None else 0.5
    t_end = data.times[-1]
    pred_mid = ft.integrate(x0, t0, t_mid)
    pred_end = ft.integrate(x0, t0, t_end)
    return pred_mid, pred_end


def main(argv=None):
    args = parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} device={args.device} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log("gpu:", torch.cuda.get_device_name(0))

    data, frames, times, skipped = build_data(args)
    log(f"frames: day2={frames[0].shape} day4={frames[1].shape} day6={frames[2].shape} "
        f"times={times} skip={skipped} dim={args.dim} whiten={args.whiten}")

    preds = {}
    for mfm_on, prefix in [(False, "fm"), (True, "mfm")]:
        pm, pe = run_one(args, mfm_on, data)
        # un-whiten back to original PCA coordinates for downstream benchmarking
        preds[f"{prefix}_pred_day4"] = data.unwhiten(pm).astype(np.float32)
        preds[f"{prefix}_pred_day6"] = data.unwhiten(pe).astype(np.float32)

    preds["t_day4"] = np.float32(times[1])
    if args.whiten:
        preds["mu"] = data.mu.astype(np.float32)
        preds["sd"] = data.sd.astype(np.float32)

    # quick in-run W2 sanity vs real day4/day6 (original coords). EMD on the full
    # ~28k-vs-48k clouds is prohibitive, so subsample to <=4000 per cloud — this is
    # a plumbing sanity check only; the standalone benchmark does the real scoring.
    def _sub(a, n=4000, seed=0):
        a = np.asarray(a)
        if len(a) <= n:
            return a
        idx = np.random.default_rng(seed).choice(len(a), size=n, replace=False)
        return a[idx]

    real_day4 = frames[1]; real_day6 = frames[2]
    for prefix in ["fm", "mfm"]:
        w4 = wasserstein_distance(_sub(preds[f"{prefix}_pred_day4"]), _sub(real_day4), p=2)
        w6 = wasserstein_distance(_sub(preds[f"{prefix}_pred_day6"]), _sub(real_day6), p=2)
        log(f"[{prefix}] W2~(pred_day4, real_day4)={w4:.4f}  W2~(pred_day6, real_day6)={w6:.4f}  (4k subsample)")

    np.savez_compressed(args.out, **preds)
    log("saved", args.out, {k: getattr(v, "shape", v) for k, v in preds.items()})


if __name__ == "__main__":
    main()
