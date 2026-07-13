"""(Metric) Flow-matching training + trajectory prediction (MFM Algorithm 2).

Plain-torch analogue of kkapusniak/metric-flow-matching
mfm/flow_matchers/flow_net_train.py (FlowNetTrainBase + FlowNetTrainTrajectory).
The velocity net regresses the conditional target ut along the (geodesic or
straight) conditional path; prediction integrates the learned field with a
torchdyn NeuralODE (euler), exactly as the official test_step.

Loss per step (kept consecutive intervals i):
    (x0, x1) = OT-couple(frame_i, frame_{i+1})
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1, t_i, t_{i+1})
    vt = flow_net(t, xt)
    loss = MSE(vt, ut)
"""

import torch
from torchdyn.core import NeuralODE

from mfm.flow_matchers.ema import EMA
from mfm.networks.utils import flow_model_torch_wrapper


class FlowTrainer:
    def __init__(self, flow_matcher, flow_net, data, ot_sampler, args):
        self.flow_matcher = flow_matcher
        self.flow_net = flow_net
        self.data = data
        self.ot_sampler = ot_sampler
        self.args = args
        self.skipped = data.skipped_time_points
        self.device = args.device

        params = (self.flow_net.model if isinstance(self.flow_net, EMA)
                  else self.flow_net).parameters()
        if args.flow_optimizer == "adamw":
            self.opt = torch.optim.AdamW(params, lr=args.flow_lr,
                                         weight_decay=args.flow_weight_decay)
        else:
            self.opt = torch.optim.Adam(params, lr=args.flow_lr)

    def _kept_pairs(self):
        kept = self.data.kept_indices()
        times = self.data.times
        for k, (a, b) in enumerate(zip(kept[:-1], kept[1:])):
            yield k, a, b, times[a], times[b]

    def _compute_loss(self, which="train"):
        main = self.data.sample_batch(which)
        ts, xts, uts = [], [], []
        for k, a, b, t0, t1 in self._kept_pairs():
            x0, x1 = main[a], main[b]
            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t0, t1)
            ts.append(t); xts.append(xt); uts.append(ut)
        t = torch.cat(ts); xt = torch.cat(xts); ut = torch.cat(uts)
        vt = self.flow_net(t[:, None], xt)
        return ((vt - ut) ** 2).mean()

    def train(self, iters, log_every=500, val_every=500, patience=50):
        best = float("inf"); bad = 0; best_state = None
        self.flow_net.train()
        for it in range(iters):
            loss = self._compute_loss("train")
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            if isinstance(self.flow_net, EMA):
                self.flow_net.update_ema()
            if it % log_every == 0:
                print(f"[flow] it {it}: train_loss={loss.item():.4f}", flush=True)
            if it % val_every == 0 and it > 0:
                self.flow_net.eval()
                with torch.no_grad():
                    vloss = self._compute_loss("val").item()
                self.flow_net.train()
                if vloss < best - 1e-5:
                    best = vloss; bad = 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.flow_net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        print(f"[flow] early stop at it {it} (val={vloss:.4f})", flush=True)
                        break
        if best_state is not None:
            self.flow_net.load_state_dict(best_state)
        return self.flow_net

    @torch.no_grad()
    def integrate(self, x0, t_start, t_end, steps=101):
        """Integrate the learned field from t_start to t_end starting at x0.
        Returns the endpoint state (numpy). Uses torchdyn NeuralODE euler, like
        the official FlowNetTrainTrajectory.test_step."""
        self.flow_net.eval()
        node = NeuralODE(flow_model_torch_wrapper(self.flow_net),
                         solver="euler", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        t_span = torch.linspace(float(t_start), float(t_end), steps, device=self.device)
        traj = node.trajectory(x0, t_span=t_span)
        return traj[-1].detach().cpu().numpy()
