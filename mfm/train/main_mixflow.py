"""Entry point: train Mixture-Flow and predict the held-out marginal.

Builds fate channels (day6-GMM + mode-OT), trains the fate-conditioned velocity on
mode-block-coupled straight paths, then integrates each day2 cell along its sampled
fate channel to t_day4 and t=1. Saves mixflow_predictions.npz (predictions in original
PCA coordinates). Ported from train_mixflow.py into the package structure.
"""

import argparse
import time

import numpy as np
import torch

from mfm.dataloaders.channels import build_channels, sample_fate_channels
from mfm.networks.flow_networks.fate_cond_mlp import FateCondVelocity
from mfm.flow_matchers.models.mixflow import MixtureFlowTrainer, integrate


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Mixture-Flow on LARRY state-fate")
    ap.add_argument("--data", default="larry_pca_mfm.npz")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=12, help="number of fate modes (day6 GMM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mixflow_predictions.npz")
    ap.add_argument("--device", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"torch {torch.__version__} device={device} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log("gpu:", torch.cuda.get_device_name(0))

    z = np.load(args.data, allow_pickle=True)
    d2 = z["day2_pca"].astype(np.float32)
    d6 = z["day6_pca"].astype(np.float32)
    t_day4 = float(z["t_day4"])
    d = d2.shape[1]
    mu = np.concatenate([d2, d6]).mean(0)
    sd = np.concatenate([d2, d6]).std(0) + 1e-6
    d2n = ((d2 - mu) / sd).astype(np.float32)
    d6n = ((d6 - mu) / sd).astype(np.float32)
    log(f"day2={d2.shape} day6={d6.shape} dim={d} t_day4={t_day4} K={args.K}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lab0, mode6, Pi_cond, w0, w1 = build_channels(d2n, d6n, args.K, seed=args.seed)
    log(f"channels: K={args.K} day2 occ={np.count_nonzero(w0)} day6 occ={np.count_nonzero(w1)}")

    vnet = FateCondVelocity(d, args.K).to(device)
    trainer = MixtureFlowTrainer(vnet, d2n, d6n, lab0, mode6, Pi_cond, device)
    trainer.train(args.iters, args.bs, args.lr)

    j4 = sample_fate_channels(lab0, Pi_cond, seed=1)
    p4 = integrate(vnet, d2n, j4, device, t_end=t_day4) * sd + mu
    p6 = integrate(vnet, d2n, j4, device, t_end=1.0) * sd + mu

    preds = dict(
        mixflow_pred_day4=p4.astype(np.float32),
        mixflow_pred_day6=p6.astype(np.float32),
        channel_j=j4.astype(np.int16), mode_day2=lab0.astype(np.int16),
        Pi_cond=Pi_cond.astype(np.float32),
        w0=w0.astype(np.float32), w1=w1.astype(np.float32),
        mu=mu.astype(np.float32), sd=sd.astype(np.float32), t_day4=np.float32(t_day4),
    )
    np.savez_compressed(args.out, **preds)
    log("saved", args.out, {k: getattr(v, "shape", v) for k, v in preds.items()})


if __name__ == "__main__":
    main()
