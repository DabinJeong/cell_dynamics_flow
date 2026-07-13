"""DataManifoldMetric — selects LAND or RBF and gives metric-weighted velocity.

Mirrors kkapusniak/metric-flow-matching mfm/geo_metrics/metric_factory.py.
The RBF branch trains one RBFNetwork per (non-skipped) timestep interval on that
interval's landmark samples; the LAND branch is closed-form. `calculate_velocity`
returns sqrt( sum_d u_d^2 * M_dd(x) ), the metric norm of the tangent u whose
mean-square the geopath net minimizes (the geodesic energy).
"""

import torch

from mfm.geo_metrics.land import land_metric_tensor
from mfm.geo_metrics.rbf import RBFNetwork


class DataManifoldMetric:
    def __init__(self, args, skipped_time_points=None, num_timesteps=None):
        self.skipped_time_points = skipped_time_points or []
        self.num_timesteps = num_timesteps

        self.gamma = args.gamma_current
        self.rho = args.rho
        self.metric = args.velocity_metric
        self.n_centers = args.n_centers
        self.kappa = args.kappa
        self.metric_epochs = args.metric_epochs
        self.metric_patience = args.metric_patience
        self.lr = args.metric_lr
        self.alpha_metric = args.alpha_metric
        self.device = args.device

        self.rbf_networks = None

    def fit_rbf(self, interval_landmarks):
        """interval_landmarks: list of tensors, one per non-skipped interval,
        each the concatenation of that interval's endpoint metric-samples."""
        self.rbf_networks = []
        for i, samples in enumerate(interval_landmarks):
            print(f"[metric] learning RBF network for interval {i} "
                  f"(n={len(samples)}, K={self.n_centers})", flush=True)
            net = RBFNetwork(n_centers=self.n_centers, kappa=self.kappa, lr=self.lr)
            net.fit(samples, self.device, epochs=self.metric_epochs,
                    patience=self.metric_patience)
            self.rbf_networks.append(net)
        return self

    def calculate_metric(self, x_t, samples, current_interval):
        if self.metric == "land":
            M = land_metric_tensor(x_t, samples, self.gamma, self.rho) ** self.alpha_metric
        elif self.metric == "rbf":
            M = self.rbf_networks[current_interval].compute_metric(
                x_t, epsilon=self.rho, alpha=self.alpha_metric
            )
        else:
            raise ValueError(f"unknown metric {self.metric}")
        return M

    def calculate_velocity(self, x_t, u_t, samples, current_interval):
        if u_t.dim() > 2:
            u_t = u_t.reshape(u_t.shape[0], -1)
            x_t = x_t.reshape(x_t.shape[0], -1)
        M = self.calculate_metric(x_t, samples, current_interval)
        velocity = torch.sqrt(((u_t**2) * M).sum(dim=-1) + 1e-12)
        return velocity
