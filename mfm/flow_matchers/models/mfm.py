"""MetricFlowMatcher — the conditional path / conditional flow of MFM.

Mirrors kkapusniak/metric-flow-matching mfm/flow_matchers/models/mfm.py.
The only substantive difference from the official file is that we do NOT depend
on the `torchcfm` package: the two pieces we need from
`ConditionalFlowMatcher` (a Gaussian noise sampler and a small t-padding
helper) are inlined here so the package runs with a plain torch install.

Algorithm (Kapusniak et al. 2024, arXiv:2405.14780), unchanged:

  geodesic conditional path (alpha=MFM on):
      mu_t = (t_max - t)/(t_max - t_min) * x0
           + (t - t_min)/(t_max - t_min) * x1
           + gamma(t) * geopath_net(x0, x1, t)
      x_t  = mu_t + sigma * eps
  with the boundary envelope
      gamma(t)   = 1 - ((t-t_min)/(t_max-t_min))^2 - ((t_max-t)/(t_max-t_min))^2
      d_gamma(t) = 2*(-2t + t_max + t_min)/(t_max - t_min)^2
  which vanishes at t_min and t_max, so mu_{t_min}=x0, mu_{t_max}=x1 exactly.

  conditional target velocity:
      u_t = (x1 - x0)/(t_max - t_min)
          + d_gamma(t) * geopath_net(x0,x1,t)
          + gamma(t) * d/dt[geopath_net(x0,x1,t)]       (if time_geopath)

  alpha=0 recovers straight-line CondOT: mu_t linear, u_t = (x1-x0)/(t_max-t_min).
"""

import torch
from torch.func import jvp


def pad_t_like_x(t, x):
    """Reshape a length-B time vector to broadcast against x (B, ...).

    Inlined from torchcfm.conditional_flow_matching.pad_t_like_x.
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class MetricFlowMatcher:
    """Conditional flow matcher with a learned geodesic interpolant."""

    def __init__(self, geopath_net: torch.nn.Module = None, sigma: float = 0.0,
                 alpha: float = 1.0):
        self.sigma = sigma
        self.alpha = alpha
        self.geopath_net = geopath_net
        if self.alpha != 0:
            assert geopath_net is not None, "GeoPath model must be provided if alpha != 0"

    # ---- noise / sigma helpers (inlined from ConditionalFlowMatcher) ----
    def compute_sigma_t(self, t):
        return self.sigma

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    # ---- boundary envelope ----
    def gamma(self, t, t_min, t_max):
        return (
            1.0
            - ((t - t_min) / (t_max - t_min)) ** 2
            - ((t_max - t) / (t_max - t_min)) ** 2
        )

    def d_gamma(self, t, t_min, t_max):
        return 2 * (-2 * t + t_max + t_min) / (t_max - t_min) ** 2

    # ---- conditional path ----
    def compute_mu_t(self, x0, x1, t, t_min, t_max):
        with torch.enable_grad():
            t = pad_t_like_x(t, x0)
            if self.alpha == 0:
                return (t_max - t) / (t_max - t_min) * x0 + (t - t_min) / (
                    t_max - t_min
                ) * x1
            self.geopath_net_output = self.geopath_net(x0, x1, t)
            if self.geopath_net.time_geopath:
                self.doutput_dt = self.doutput_dt_fun(self.geopath_net, x0, x1, t)
        return (
            (t_max - t) / (t_max - t_min) * x0
            + (t - t_min) / (t_max - t_min) * x1
            + self.gamma(t, t_min, t_max) * self.geopath_net_output
        )

    def sample_xt(self, x0, x1, t, epsilon, t_min, t_max):
        mu_t = self.compute_mu_t(x0, x1, t, t_min, t_max)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def sample_location_and_conditional_flow(
        self, x0, x1, t_min, t_max,
        training_geopath_net=False, midpoint_only=False, t=None,
    ):
        self.training_geopath_net = training_geopath_net
        with torch.enable_grad():
            if t is None:
                t = torch.rand(x0.shape[0], requires_grad=True)
            t = t.type_as(x0)
            t = t * (t_max - t_min) + t_min
            if midpoint_only:
                t = (t_max + t_min) / 2 * torch.ones_like(t).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps, t_min, t_max)
        ut = self.compute_conditional_flow(x0, x1, t, xt, t_min, t_max)
        return t, xt, ut

    def compute_conditional_flow(self, x0, x1, t, xt, t_min, t_max):
        del xt
        t = pad_t_like_x(t, x0)
        if self.alpha == 0:
            return (x1 - x0) / (t_max - t_min)
        return (
            (x1 - x0) / (t_max - t_min)
            + self.d_gamma(t, t_min, t_max) * self.geopath_net_output
            + (
                self.gamma(t, t_min, t_max) * self.doutput_dt
                if self.geopath_net.time_geopath
                else 0
            )
        )

    @staticmethod
    def doutput_dt_fun(model, x0, x1, t_raw):
        def f(tt):
            t_padded = pad_t_like_x(tt, x0)
            return model(x0, x1, t_padded)

        _, dydt = jvp(f, (t_raw,), (torch.ones_like(t_raw),))
        return dydt.squeeze(-1)
