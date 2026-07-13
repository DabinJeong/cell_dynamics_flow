"""Temporal (trajectory) data for MFM.

Plain-torch analogue of kkapusniak/metric-flow-matching
mfm/dataloaders/trajectory_data.py. The official module is a
`pytorch_lightning.LightningDataModule` that yields, per step, a dict with a
`train_samples` list (one minibatch per timestep) and a `metric_samples` list
(landmark minibatch per timestep). We keep that exact structure — a list of
per-timestep frames, a train/val split, and landmark ("metric") pools — but as
plain tensors iterated by our own training loops.

Loader for the LARRY state-fate benchmark: reads day2/day4/day6 npz files
(each with an `X_pca` (N, D) array) and stacks them as timesteps [0, 1, 2] at
normalized times [0.0, 0.5, 1.0]. Holding out day4 (index 1) makes it the
intermediate marginal the model must recover — the same "skip a timepoint"
protocol as the official single-cell experiments (t_exclude).
"""

import numpy as np
import torch


class OTPlanSampler:
    """Minibatch OT coupling via POT (exact EMD).

    Mirrors torchcfm.optimal_transport.OTPlanSampler(method="exact").sample_plan:
    given equal-size batches x0, x1, returns (x0, x1[perm]) sampled from the
    optimal transport plan between the empirical batches.
    """

    def __init__(self, method="exact", reg=0.05):
        self.method = method
        self.reg = reg

    def get_map(self, x0, x1):
        import ot as pot
        a = pot.unif(x0.shape[0])
        b = pot.unif(x1.shape[0])
        M = torch.cdist(x0, x1) ** 2
        M = M / (M.max() + 1e-12)
        Mnp = M.detach().cpu().numpy()
        if self.method == "exact":
            p = pot.emd(a, b, Mnp)
        elif self.method == "sinkhorn":
            p = pot.sinkhorn(a, b, Mnp, reg=self.reg)
        else:
            raise ValueError(self.method)
        return p

    def sample_plan(self, x0, x1, replace=True):
        p = self.get_map(x0, x1)
        p = p.flatten()
        p = p / p.sum()
        n = x0.shape[0]
        choices = np.random.choice(p.shape[0], size=n, p=p, replace=replace)
        i, j = np.divmod(choices, x1.shape[0])
        return x0[i], x1[j]


class TemporalData:
    """Per-timestep frames with train/val split and metric landmark pools."""

    def __init__(self, frames, times, batch_size=128, split_ratio=0.9,
                 skipped_time_points=None, device="cpu", seed=0, whiten=False):
        """frames: list of (N_i, D) float arrays, one per timestep (ascending time).
        times: list of normalized times in [0,1] aligned to frames.
        skipped_time_points: indices held out of training (recovered at test)."""
        self.device = device
        self.batch_size = batch_size
        self.times = list(times)
        self.num_timesteps = len(frames)
        self.skipped_time_points = skipped_time_points or []
        rng = np.random.default_rng(seed)

        self.whiten = whiten
        if whiten:
            alld = np.concatenate(frames, axis=0)
            self.mu = alld.mean(0); self.sd = alld.std(0) + 1e-8
            frames = [(f - self.mu) / self.sd for f in frames]
        else:
            self.mu = None; self.sd = None

        self.train_frames, self.val_frames, self.all_frames = [], [], []
        for f in frames:
            f = np.asarray(f, dtype=np.float32)
            idx = rng.permutation(len(f))
            f = f[idx]
            split = int(len(f) * split_ratio)
            if len(f) - split < batch_size:
                split = len(f) - batch_size
            self.train_frames.append(torch.tensor(f[:split], device=device))
            self.val_frames.append(torch.tensor(f[split:], device=device))
            self.all_frames.append(torch.tensor(f, device=device))
        # metric landmarks = full (shuffled) frame; the official uses min-frame-size
        # minibatches drawn from the whole frame — we pass the whole frame.
        self.metric_frames = self.all_frames

    def kept_indices(self):
        return [i for i in range(self.num_timesteps) if i not in self.skipped_time_points]

    def interval_landmarks(self):
        """Concatenated endpoint landmarks for each consecutive kept interval."""
        kept = self.kept_indices()
        out = []
        for a, b in zip(kept[:-1], kept[1:]):
            out.append(torch.cat([self.metric_frames[a], self.metric_frames[b]], dim=0))
        return out

    def sample_batch(self, which="train"):
        """Return per-timestep minibatches as lists aligned to ALL timesteps
        (kept + skipped), so callers filter by skipped_time_points exactly like
        the official _compute_loss. Skipped frames still yield a batch (unused)."""
        frames = {"train": self.train_frames, "val": self.val_frames,
                  "metric": self.metric_frames}[which]
        out = []
        for f in frames:
            bs = min(self.batch_size, len(f))
            b = torch.randint(0, len(f), (bs,), device=self.device)
            out.append(f[b])
        return out

    def unwhiten(self, x):
        if not self.whiten:
            return x
        mu = torch.tensor(self.mu, device=x.device) if isinstance(x, torch.Tensor) else self.mu
        sd = torch.tensor(self.sd, device=x.device) if isinstance(x, torch.Tensor) else self.sd
        return x * sd + mu


def load_larry_frames(data_path, max_dim=50):
    """Load a bundled npz with keys day2_pca/day4_pca/day6_pca (+ t_* metadata) OR
    a dict of separate arrays. Returns (frames, times).

    frames = [day2, day4, day6]; times = [0.0, t_day4, 1.0]."""
    z = np.load(data_path, allow_pickle=True)
    keys = set(z.files)
    if {"day2_pca", "day4_pca", "day6_pca"}.issubset(keys):
        d2 = z["day2_pca"][:, :max_dim].astype(np.float32)
        d4 = z["day4_pca"][:, :max_dim].astype(np.float32)
        d6 = z["day6_pca"][:, :max_dim].astype(np.float32)
        t4 = float(z["t_day4"]) if "t_day4" in keys else 0.5
    elif {"X_pca_day2", "X_pca_day4", "X_pca_day6"}.issubset(keys):
        d2 = z["X_pca_day2"][:, :max_dim].astype(np.float32)
        d4 = z["X_pca_day4"][:, :max_dim].astype(np.float32)
        d6 = z["X_pca_day6"][:, :max_dim].astype(np.float32)
        t4 = 0.5
    else:
        raise KeyError(f"unrecognized npz keys: {sorted(keys)}")
    return [d2, d4, d6], [0.0, t4, 1.0]
