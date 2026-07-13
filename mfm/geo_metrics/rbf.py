"""RBF density network for the data-manifold metric.

Algorithm mirrors kkapusniak/metric-flow-matching mfm/geo_metrics/rbf.py.
The official class is a `pytorch_lightning.LightningModule`; here it is a plain
`torch.nn.Module` with a self-contained `fit()` so the package runs without
Lightning. The math is unchanged:

  h(x) = sum_k W_k * exp(-0.5 * lambda_k * ||x - C_k||^2)
  trained so h -> 1 on data (loss = mean((1 - h)^2)); W clamped >= 1e-4.
  centers C_k = KMeans centroids, lambda_k = 0.5 / (kappa * sigma_k)^2 with
  sigma_k the RMS spread of cluster k.
  metric   M(x) = 1 / (h(x) + epsilon)^alpha   (large where density h is low).
"""

import numpy as np
import torch
from sklearn.cluster import KMeans


class RBFNetwork(torch.nn.Module):
    def __init__(self, n_centers: int = 100, kappa: float = 1.0, lr: float = 1e-2):
        super().__init__()
        self.K = n_centers
        self.kappa = kappa
        self.lr = lr
        self.W = torch.nn.Parameter(torch.rand(self.K, 1))
        self.last_val_loss = torch.tensor(1.0)
        self._fitted = False

    def _init_centers(self, data_to_fit, device):
        """KMeans centers + per-cluster lambda; data_to_fit is a numpy array."""
        km = KMeans(n_clusters=self.K, n_init=10)
        km.fit(data_to_fit)
        clusters = km.cluster_centers_
        labels = km.labels_
        self.register_buffer("C", torch.tensor(clusters, dtype=torch.float32, device=device))
        sigmas = np.zeros((self.K, 1))
        for k in range(self.K):
            pts = data_to_fit[labels == k, :]
            if len(pts) == 0:
                sigmas[k, :] = 1.0
                continue
            variance = ((pts - clusters[k]) ** 2).mean(axis=0)
            sigmas[k, :] = np.sqrt(variance.mean())
        sigmas[sigmas == 0] = 1e-3
        self.register_buffer(
            "lamda", torch.tensor(0.5 / (self.kappa * sigmas) ** 2, dtype=torch.float32, device=device)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        dist2 = torch.cdist(x, self.C) ** 2
        phi_x = torch.exp(-0.5 * self.lamda[None, :, :] * dist2[:, :, None])
        h_x = (self.W.to(x.device) * phi_x).sum(dim=1)
        return h_x

    def fit(self, data, device, epochs=200, patience=25, batch_size=2048, seed=0):
        """Train h->1 on `data` (torch tensor or numpy), with early stopping on a held split."""
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)
        self._init_centers(data_np, device)
        self.to(device)

        rng = np.random.default_rng(seed)
        n = len(data_np)
        perm = rng.permutation(n)
        # hold out ~10% for validation but never more than half, so train stays non-empty
        n_val = int(np.clip(round(0.1 * n), 1, n // 2))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        batch_size = min(batch_size, len(tr_idx))
        X = torch.tensor(data_np, dtype=torch.float32, device=device)
        Xtr, Xval = X[tr_idx], X[val_idx]

        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        best = float("inf"); bad = 0; best_state = None
        for ep in range(epochs):
            self.train()
            b = torch.randint(0, len(Xtr), (min(batch_size, len(Xtr)),), device=device)
            h = self.forward(Xtr[b])
            loss = ((1 - h) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            with torch.no_grad():
                self.W.data.clamp_(min=1e-4)
                self.eval()
                vloss = ((1 - self.forward(Xval)) ** 2).mean()
            self.last_val_loss = vloss.detach()
            if vloss.item() < best - 1e-5:
                best = vloss.item(); bad = 0
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)
        self._fitted = True
        return self

    def compute_metric(self, x, alpha=1.0, epsilon=1e-2):
        if epsilon < 0:
            epsilon = (1 - self.last_val_loss.item()) / abs(epsilon)
        h_x = self.forward(x)
        M_x = 1 / (h_x + epsilon) ** alpha
        return M_x
