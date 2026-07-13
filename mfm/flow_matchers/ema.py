"""Exponential moving average of model parameters.

Mirrors the official Metric Flow Matching repo (kkapusniak/metric-flow-matching),
mfm/flow_matchers/ema.py — itself adapted from Meta's flow_matching. Logic is
identical; comments were lightly reworded and a `shaddow`->`shadow` typo fixed.
"""

import torch


class EMA(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        if hasattr(self.model, "time_geopath"):
            self.time_geopath = self.model.time_geopath

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.shadow_params = torch.nn.ParameterList(
            [
                torch.nn.Parameter(p.clone().detach(), requires_grad=False)
                for p in model.parameters()
                if p.requires_grad
            ]
        )
        self.backup_params = []

    def train(self, mode: bool = True):
        if self.training and mode == False:
            # Switching from train mode to eval mode: back up model params and
            # overwrite with shadow (ema) params.
            self.backup()
            self.copy_to_model()
        elif not self.training and mode == True:
            # Switching from eval to train mode: restore the backup params.
            self.restore_to_model()

        super().train(mode)

    def update_ema(self):
        self.num_updates += 1
        num_updates = self.num_updates.item()
        decay = min(self.decay, (1 + num_updates) / (10 + num_updates))
        with torch.no_grad():
            params = [p for p in self.model.parameters() if p.requires_grad]
            for shadow, param in zip(self.shadow_params, params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def copy_to_model(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        for shadow, param in zip(self.shadow_params, params):
            param.data.copy_(shadow.data)

    def backup(self):
        if len(self.backup_params) > 0:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self):
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)
