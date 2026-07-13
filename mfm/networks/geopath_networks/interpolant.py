"""Geodesic interpolant correction phi_t for Mixture-Geodesic-Flow.

phi_t = (1-t) x0 + t x1 + t(1-t) * NN(x0, x1, t)

The t(1-t) envelope forces phi_0 = x0 and phi_1 = x1 exactly, so only the interior of
the path is learned. This mirrors the endpoint-fixed correction of the metric-FM
GeoPathMLP (mfm/networks/geopath_networks/mlp.py) but uses the plain SiLU backbone of
the Mixture-Flow scripts and is trained on the mode-block-coupled endpoints (so the
geodesic is channel-aware).
"""

import torch
import torch.nn as nn

from mfm.networks.flow_networks.fate_cond_mlp import MLP


class InterpolantCorrection(nn.Module):
    def __init__(self, d, h=256, nl=3):
        super().__init__()
        self.mlp = MLP(2 * d + 1, d, h, nl)

    def forward(self, x0, x1, t):
        if t.dim() == 1:
            t = t[:, None]
        return (1 - t) * x0 + t * x1 + t * (1 - t) * self.mlp(torch.cat([x0, x1, t], dim=-1))
