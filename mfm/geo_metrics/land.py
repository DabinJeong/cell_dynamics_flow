"""LAND (Locally Adaptive Normal Distribution) diagonal metric.

Ported from kkapusniak/metric-flow-matching mfm/geo_metrics/land.py — code logic
identical; some inline comments reworded.

The metric tensor at x is diagonal with entries
    M_dd(x) = 1 / ( sum_n w_n(x) * (samples_n - x)^2  + rho )
where w_n(x) = exp(-||x - samples_n||^2 / (2 gamma^2)). Off-manifold / low-density
x sees small weights -> small denominator sum -> large M_dd -> expensive to
traverse, so geodesics bend toward the data.
"""

import torch


def weighting_function(x, samples, gamma):
    pairwise_sq_diff = (x[:, None, :] - samples[None, :, :]) ** 2
    pairwise_sq_dist = pairwise_sq_diff.sum(-1)
    weights = torch.exp(-pairwise_sq_dist / (2 * gamma**2))
    return weights


def land_metric_tensor(x, samples, gamma, rho):
    weights = weighting_function(x, samples, gamma)  # [B, N]
    differences = samples[None, :, :] - x[:, None, :]  # [B, N, D]
    squared_differences = differences**2  # [B, N, D]

    # weighted squared differences summed over landmarks, per dimension
    M_dd_diag = torch.einsum("bn,bnd->bd", weights, squared_differences) + rho

    # invert the diagonal metric tensor for each x_t
    M_dd_inv_diag = 1.0 / M_dd_diag  # [B, D]
    return M_dd_inv_diag


def weighting_function_dt(x, dx_dt, samples, gamma, weights):
    pairwise_sq_diff_dt = (x[:, None, :] - samples[None, :, :]) * dx_dt[:, None, :]
    return -pairwise_sq_diff_dt.sum(-1) * weights / (gamma**2)
