"""Geodesic-interpolant training (MFM Algorithm 1).

Plain-torch analogue of kkapusniak/metric-flow-matching
mfm/flow_matchers/geopath_net_train.py. The geopath network is trained to
minimize the mean squared metric-velocity of the conditional path — i.e. the
geodesic energy under the data-manifold metric. The loss is normalized by a
reference loss computed with alpha=0 (the straight-line path), exactly as in
the official `compute_initial_loss`.

Loss per step (over kept consecutive intervals i):
    (x0, x1) = OT-couple(frame_i, frame_{i+1})
    t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
                    x0, x1, t_i, t_{i+1}, training_geopath_net=True)
    vel = metric.calculate_velocity(xt, ut, landmarks_i, i)
    loss = mean(vel^2)  / first_loss
"""

import torch

from mfm.flow_matchers.ema import EMA


class GeoPathTrainer:
    def __init__(self, flow_matcher, data, metric, ot_sampler, args):
        self.flow_matcher = flow_matcher
        self.geopath_net = flow_matcher.geopath_net
        self.data = data
        self.metric = metric
        self.ot_sampler = ot_sampler
        self.args = args
        self.skipped = data.skipped_time_points
        self.device = args.device
        self.first_loss = None

        params = (self.geopath_net.model if isinstance(self.geopath_net, EMA)
                  else self.geopath_net).parameters()
        if args.geopath_optimizer == "adamw":
            self.opt = torch.optim.AdamW(params, lr=args.geopath_lr,
                                         weight_decay=args.geopath_weight_decay)
        else:
            self.opt = torch.optim.Adam(params, lr=args.geopath_lr)

    def _kept_pairs(self):
        """Yield (interval_index, i0, i1, t0, t1) over kept consecutive frames."""
        kept = self.data.kept_indices()
        times = self.data.times
        for k, (a, b) in enumerate(zip(kept[:-1], kept[1:])):
            yield k, a, b, times[a], times[b]

    def _compute_loss(self, which="train"):
        main = self.data.sample_batch(which)
        metric_b = self.data.sample_batch("metric")
        velocities = []
        for k, a, b, t0, t1 in self._kept_pairs():
            x0, x1 = main[a], main[b]
            if self.ot_sampler is not None:
                x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)
            _, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                x0, x1, t0, t1, training_geopath_net=True)
            samples = torch.cat([metric_b[a], metric_b[b]], dim=0)
            vel = self.metric.calculate_velocity(xt, ut, samples, k)
            velocities.append(vel)
        return torch.mean(torch.cat(velocities) ** 2)

    @torch.no_grad()
    def _reference_loss(self, n_batches=8):
        old = self.flow_matcher.alpha
        self.flow_matcher.alpha = 0
        tot = 0.0
        for _ in range(n_batches):
            with torch.enable_grad():
                tot += self._compute_loss("train").item()
        self.flow_matcher.alpha = old
        return tot / n_batches

    def train(self, iters, log_every=200, val_every=200, patience=25):
        self.first_loss = self._reference_loss()
        print(f"[geopath] reference (alpha=0) loss = {self.first_loss:.4f}", flush=True)
        best = float("inf"); bad = 0; best_state = None
        self.geopath_net.train()
        for it in range(iters):
            loss = self._compute_loss("train") / self.first_loss
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            if isinstance(self.geopath_net, EMA):
                self.geopath_net.update_ema()
            if it % log_every == 0:
                print(f"[geopath] it {it}: train_loss={loss.item():.4f}", flush=True)
            if it % val_every == 0 and it > 0:
                self.geopath_net.eval()
                with torch.enable_grad():
                    vloss = (self._compute_loss("val") / self.first_loss).item()
                self.geopath_net.train()
                if vloss < best - 1e-4:
                    best = vloss; bad = 0
                    best_state = {k: v.detach().clone()
                                  for k, v in self.geopath_net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        print(f"[geopath] early stop at it {it} (val={vloss:.4f})", flush=True)
                        break
        if best_state is not None:
            self.geopath_net.load_state_dict(best_state)
        return self.geopath_net
