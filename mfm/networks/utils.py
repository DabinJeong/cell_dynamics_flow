"""torchdyn-compatible wrapper for the velocity model.

Ported from kkapusniak/metric-flow-matching mfm/networks/utils.py — forward()
identical; class docstring reworded.
"""

import torch


class flow_model_torch_wrapper(torch.nn.Module):
    """Wraps a v(t, x) model into the (t, x) signature torchdyn's NeuralODE expects."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(t, x)
