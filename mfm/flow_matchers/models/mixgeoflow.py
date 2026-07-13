"""Mixture-Geodesic-Flow: fate-channel flow with a metric-geodesic conditional path.

Combines the two ingredients that each fixed one failure mode:
  * mode-block coupling + fate-channel conditioning (Mixture-Flow) -> recovers COMMITMENT
  * density-metric geodesic interpolant (MFM)                      -> bends the path off
                                                                       the inter-arm void

Controlled contrast to Mixture-Flow: identical net capacity/budget; ONLY the conditional
path target changes from the straight chord (x1-x0) to dphi/dt of the metric geodesic
between the mode-block-coupled endpoints. Two stages: (1) train the channel-aware
geodesic interpolant to minimize density-weighted path energy; (2) regress the
fate-conditioned velocity to the geodesic's dphi/dt. Ported from train_mixgeoflow.py.

NOTE (recorded negative result): on LARRY the geodesic did not lower energy but raised
it (~0.68 -> 0.74) and slightly dropped commitment, because the density metric treats
the dense undifferentiated blob as cheap and pulls paths toward it — structurally
opposed to the channel conditioning that pushes cells out to their arm. Kept for
reproducibility and ablation, not as the recommended configuration.
"""

import time

import numpy as np
import torch

from mfm.dataloaders.channels import ModeBlockSampler


def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class _IdxSampler:
    """Wrap ModeBlockSampler to return endpoint tensors + channel labels on device."""

    def __init__(self, d2n, d6n, lab0, mode6, Pi_cond, device, seed=0):
        self.d2n = d2n
        self.d6n = d6n
        self.device = device
        self.s = ModeBlockSampler(lab0, mode6, Pi_cond, n_day6=len(d6n), seed=seed)

    def sample(self, bs):
        i0, j, i1 = self.s.sample(bs)
        x0 = torch.tensor(self.d2n[i0], device=self.device)
        x1 = torch.tensor(self.d6n[i1], device=self.device)
        return x0, x1, torch.as_tensor(j, device=self.device)


def _dphi_dt(interp, x0, x1, t):
    with torch.enable_grad():
        t = t.detach().requires_grad_(True)
        phi = interp(x0, x1, t)
        grads = torch.zeros_like(phi)
        for kk in range(phi.shape[1]):
            grads[:, kk] = torch.autograd.grad(phi[:, kk].sum(), t, retain_graph=True)[0][:, 0]
    return grads


class MixtureGeodesicFlowTrainer:
    def __init__(self, interp, vnet, metric, d2n, d6n, lab0, mode6, Pi_cond, device):
        self.interp = interp
        self.vnet = vnet
        self.metric = metric
        self.device = device
        self.geo_sampler = _IdxSampler(d2n, d6n, lab0, mode6, Pi_cond, device, seed=0)
        self.fm_sampler = _IdxSampler(d2n, d6n, lab0, mode6, Pi_cond, device, seed=1)

    def train_geodesic(self, iters, bs, lr, log_every=1000):
        """Stage 1: minimize density-weighted geodesic energy on mode-block pairs."""
        opt = torch.optim.Adam(self.interp.parameters(), lr=lr)
        for it in range(iters):
            x0, x1, _ = self.geo_sampler.sample(bs)
            t = torch.rand(bs, 1, device=self.device, requires_grad=True)
            phi = self.interp(x0, x1, t)
            vel = torch.zeros_like(phi)
            for kk in range(phi.shape[1]):
                gk = torch.autograd.grad(phi[:, kk].sum(), t, retain_graph=True,
                                         create_graph=True)[0]
                vel[:, kk] = gk[:, 0]
            energy = (self.metric.g(phi) * (vel ** 2).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            energy.backward()
            opt.step()
            if it % log_every == 0:
                _log(f"  [mixgeo geo] it {it}: energy={energy.item():.4f}")
        return self.interp

    def train_flow(self, iters, bs, lr, log_every=1000):
        """Stage 2: regress fate-conditioned velocity to the geodesic dphi/dt."""
        opt = torch.optim.Adam(self.vnet.parameters(), lr=lr)
        for it in range(iters):
            x0, x1, j = self.fm_sampler.sample(bs)
            t = torch.rand(bs, device=self.device)
            tt = t[:, None]
            with torch.no_grad():
                xt = self.interp(x0, x1, tt)
                ut = _dphi_dt(self.interp, x0, x1, tt).detach()
            pred = self.vnet(xt, t, j)
            loss = ((pred - ut) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % log_every == 0:
                _log(f"  [mixgeo fm] it {it}: loss={loss.item():.4f}")
        return self.vnet
