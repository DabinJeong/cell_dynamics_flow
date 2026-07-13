"""Scalar conformal density metric g(x) = 1/(rho_hat(x) + rho0).

Used by Mixture-Geodesic-Flow. This is the isotropic special case of a data-manifold
metric M(x) = g(x)*I with g the inverse of a Gaussian-kernel density estimate on
day2+day6 landmarks (never day4). Low-density (off-manifold) x gets large g, so the
geodesic energy penalizes leaving the data support.

Distinct from geo_metrics/land.py (diagonal LAND) and geo_metrics/rbf.py (learned RBF
density network): this is a fixed, closed-form KDE conformal factor, matching the
DensityMetric in the Mixture-Flow reference script.
"""

import numpy as np
import torch


class DensityMetric:
    def __init__(self, landmarks, device, n_land=3000, sigma=None, rho0_quantile=0.10,
                 seed=0):
        idx = np.random.default_rng(seed).choice(
            len(landmarks), min(n_land, len(landmarks)), replace=False)
        self.L = torch.tensor(landmarks[idx], dtype=torch.float32, device=device)
        if sigma is None:
            d2 = torch.cdist(self.L[:500], self.L[:500]) ** 2
            sigma = float(torch.sqrt(torch.median(d2[d2 > 0])))
        self.sigma = sigma
        self.inv2s2 = 1.0 / (2 * sigma ** 2)
        with torch.no_grad():
            d2 = torch.cdist(self.L, self.L) ** 2
            rho = torch.exp(-d2 * self.inv2s2).mean(1)
        self.rho0 = float(torch.quantile(rho, rho0_quantile))

    def g(self, x):
        d2 = torch.cdist(x, self.L) ** 2
        rho = torch.exp(-d2 * self.inv2s2).mean(1)
        return 1.0 / (rho + self.rho0)
