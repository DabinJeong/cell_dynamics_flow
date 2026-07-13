"""Fate-channel construction and mode-block coupling for Mixture-Flow.

Mixture-Flow factorizes the day2->day6 transport into K fate CHANNELS. A K-mode
Gaussian mixture is fit on day6 (the terminal marginal); every day2 and day6 cell
is assigned to its mode. A mode-level entropic-OT (Sinkhorn) plan Pi couples the
day2 mode-mass to the day6 mode-mass, and Pi_cond = P(target mode j | source mode
k) defines the channel a source cell commits to. Mode-block coupling then draws
training pairs (x0 in mode k, x1 a real day6 cell in mode j~Pi_cond[k]).

Nothing here peeks at the held-out day4 marginal: channels are built from day2 and
day6 only, and day4 proportions are the OT-interpolated channel mass at t.
"""

import numpy as np


def sinkhorn_plan(a, b, C, reg=0.05, iters=2000):
    """Entropic-OT transport plan between histograms a, b under cost C.

    Verbatim from the Mixture-Flow scripts: cost is max-normalized, kernel
    K=exp(-C/reg), standard Sinkhorn fixed-point.
    """
    Cn = C / (C.max() + 1e-9)
    K = np.exp(-Cn / reg)
    u = np.ones_like(a)
    for _ in range(iters):
        u = a / (K @ (b / (K.T @ u + 1e-30)) + 1e-30)
    v = b / (K.T @ u + 1e-30)
    return u[:, None] * K * v[None, :]


def build_channels(d2n, d6n, K, seed=0, gmm_subsample=20000):
    """Fit a K-mode day6 GMM, assign both marginals, build the mode-OT channels.

    Returns:
        lab0     : (N2,) day2 cell -> source mode k
        mode6    : list of K arrays, day6 cell indices per target mode j
        Pi_cond  : (K, K) row-normalized P(target j | source k)
        w0, w1   : (K,) day2 / day6 mode occupancy
    """
    from sklearn.mixture import GaussianMixture

    rng = np.random.default_rng(seed)
    sub = d6n[rng.choice(len(d6n), min(gmm_subsample, len(d6n)), replace=False)]
    gm = GaussianMixture(K, covariance_type="full", random_state=seed, max_iter=200).fit(sub)
    lab0 = gm.predict(d2n)
    lab1 = gm.predict(d6n)
    w0 = np.bincount(lab0, minlength=K) / len(lab0)
    w1 = np.bincount(lab1, minlength=K) / len(lab1)

    from scipy.spatial.distance import cdist
    Cm = cdist(gm.means_, gm.means_, "sqeuclidean")
    Pi = sinkhorn_plan(w0 + 1e-9, w1 + 1e-9, Cm)
    Pi /= Pi.sum()
    Pi_cond = Pi / Pi.sum(1, keepdims=True)  # P(j | k)
    mode6 = [np.where(lab1 == j)[0] for j in range(K)]
    return lab0, mode6, Pi_cond, w0, w1


class ModeBlockSampler:
    """Draw mode-block-coupled training pairs.

    A day2 cell (source mode k) picks a target mode j ~ Pi_cond[k] (via the
    precomputed CDF), then a real day6 cell uniformly from mode-block j. Returns
    index arrays so callers can build tensors on their own device.
    """

    def __init__(self, lab0, mode6, Pi_cond, n_day6, seed=0):
        self.lab0 = lab0
        self.mode6 = mode6
        self.cdf = np.cumsum(Pi_cond, 1)
        self.n_day6 = n_day6
        self.rng = np.random.default_rng(seed)
        self.K = Pi_cond.shape[1]

    def sample(self, bs):
        rng = self.rng
        i0 = rng.integers(0, len(self.lab0), bs)
        k = self.lab0[i0]
        r = rng.random(bs)
        j = np.clip((r[:, None] > self.cdf[k]).sum(1), 0, self.K - 1)
        i1 = np.empty(bs, dtype=np.int64)
        for m in range(bs):
            idx = self.mode6[j[m]]
            i1[m] = idx[rng.integers(len(idx))] if len(idx) > 0 else rng.integers(self.n_day6)
        return i0, j, i1


def sample_fate_channels(lab0, Pi_cond, seed=1):
    """Assign every day2 cell a fate channel j ~ Pi_cond[source mode] (inference)."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(Pi_cond, 1)
    r = rng.random(len(lab0))
    j = np.clip((r[:, None] > cdf[lab0]).sum(1), 0, Pi_cond.shape[1] - 1)
    return j.astype(np.int64)
