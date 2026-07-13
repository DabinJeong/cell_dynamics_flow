"""Mixture-Flow: factorized generative flow for a held-out marginal.

    p_t(x) = sum_k  w_k(t) * p_k(x | t)
             \\____/   \\________/
          proportion   mode-internal generative flow

Two factors, both built from day2+day6 only (day4 held out):
 (1) Proportion w_k(t): K day6-GMM modes; a mode-level entropic-OT plan defines fate
     CHANNELS (source k -> target j) carrying the interpolated channel mass at t.
 (2) Mode-internal p_k(x|t): ONE velocity net conditioned on the target fate mode j
     (FateCondVelocity). Fate conditioning gives each channel its own arm-specific
     dynamics, so no inter-arm barycentric smear; integrating the field yields NOVEL
     on-arm coordinates (unlike a resampling/reweighting baseline).

Relative to straight-FM: identical net capacity/budget and a straight conditional path
x_t=(1-t)x0+t x1, u_t=x1-x0; the only additions are (a) the mode-OT channel coupling
and (b) fate-mode conditioning of the velocity. Ported from train_mixflow.py.
"""

import time

import numpy as np
import torch

from mfm.dataloaders.channels import ModeBlockSampler


def _log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


class MixtureFlowTrainer:
    def __init__(self, vnet, d2n, d6n, lab0, mode6, Pi_cond, device):
        self.vnet = vnet
        self.d2n = d2n
        self.d6n = d6n
        self.device = device
        self.sampler = ModeBlockSampler(lab0, mode6, Pi_cond, n_day6=len(d6n), seed=0)
        self.X0 = torch.tensor(d2n, device=device)
        self.X1 = torch.tensor(d6n, device=device)

    def train(self, iters, bs, lr, log_every=1000):
        opt = torch.optim.Adam(self.vnet.parameters(), lr=lr)
        for it in range(iters):
            i0, j, i1 = self.sampler.sample(bs)
            x0 = self.X0[i0]
            x1 = self.X1[i1]
            jt = torch.as_tensor(j, device=self.device)
            t = torch.rand(bs, device=self.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1  # straight chord
            ut = x1 - x0
            pred = self.vnet(xt, t, jt)
            loss = ((pred - ut) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if it % log_every == 0:
                _log(f"  [mixflow] it {it}: loss={loss.item():.4f}")
        return self.vnet


@torch.no_grad()
def integrate(vnet, x0, jvec, device, t_end, steps=100):
    """Midpoint (RK2) integration of the fate-conditioned field from t=0 to t_end."""
    x = torch.as_tensor(x0, dtype=torch.float32, device=device).clone()
    j = torch.as_tensor(jvec, device=device)
    dt = float(t_end) / steps
    for s in range(steps):
        t = torch.full((len(x),), s * dt, device=device)
        k1 = vnet(x, t, j)
        k2 = vnet(x + 0.5 * dt * k1, t + 0.5 * dt, j)
        x = x + dt * k2
    return x.cpu().numpy()
