"""Fate-mode-conditioned velocity network for Mixture-Flow.

v(x, t, j): the velocity is conditioned on the target fate mode j through a learned
embedding, concatenated with (x, t). Because the field is fate-conditioned, each
channel learns its own arm-specific dynamics and does not average across arms — the
barycentric inter-arm smear of an unconditioned mixture-weighted flow is avoided.

Ported from the Mixture-Flow scripts (train_mixflow.py / train_mixgeoflow.py); the
plain-MLP backbone here is local to Mixture-Flow (SiLU, no batch-norm) rather than
the SimpleDenseNet used by the metric-FM networks, matching the reference scripts.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, din, dout, h=256, nl=3):
        super().__init__()
        L = [nn.Linear(din, h), nn.SiLU()]
        for _ in range(nl - 1):
            L += [nn.Linear(h, h), nn.SiLU()]
        L += [nn.Linear(h, dout)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


class FateCondVelocity(nn.Module):
    """v(x, t, j): velocity conditioned on target fate-mode j via a K-way embedding."""

    def __init__(self, d, K, emb=32, h=256, nl=4):
        super().__init__()
        self.emb = nn.Embedding(K, emb)
        self.mlp = MLP(d + 1 + emb, d, h, nl)

    def forward(self, x, t, j):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t[:, None]
        e = self.emb(j)
        return self.mlp(torch.cat([x, t, e], dim=-1))
