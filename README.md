# Cell_dynamics_flow

Metric Flow Matching (MFM, [Kapusniak et al. 2024, arXiv:2405.14780](https://arxiv.org/abs/2405.14780))
vs straight-line Flow Matching (CondOT) on the LARRY state-fate hematopoiesis data
(day2 → day6, **day4 held out** as the intermediate marginal to recover).

The package mirrors the module layout of the official reference implementation
[kkapusniak/metric-flow-matching](https://github.com/kkapusniak/metric-flow-matching).
Algorithm code (EMA, the `MetricFlowMatcher` conditional path/flow, the LAND and
RBF data-manifold metrics, the OT coupling) follows the official files closely;
the pytorch_lightning + wandb orchestration is replaced with plain-torch training
loops so the code runs unattended on a batch scheduler without those heavy deps.

## Layout
```
mfm/
  flow_matchers/
    ema.py                 EMA of model params
    models/mfm.py          MetricFlowMatcher: geodesic/straight conditional path + target velocity
    geopath_net_train.py   ALGO 1: train geodesic interpolant (min metric geodesic energy)
    flow_net_train.py      ALGO 2: (metric) flow matching + NeuralODE prediction
  geo_metrics/
    land.py                LAND diagonal metric (closed form)
    rbf.py                 RBF density network metric (KMeans centers, h->1 on data)
    metric_factory.py      DataManifoldMetric: picks land/rbf, metric-weighted velocity
  networks/                VelocityNet, GeoPathMLP, SimpleDenseNet backbone
  dataloaders/
    trajectory_data.py     LARRY npz loader, per-timestep frames, OT sampler (POT)
  train/
    parsers.py             CLI args (defaults from official single_cell/50dims configs)
    main.py                two-stage train of MFM + straight-FM; predict day4/day6
prepare_data.py            build larry_pca_mfm.npz from day{2,4,6}.npz (X_pca)
run_umap.py                shared-UMAP embedding of all methods' day4 predictions
```

## Run
```bash
python prepare_data.py --src_dir <dir with day2/4/6.npz> --out larry_pca_mfm.npz
python -m mfm.train.main --data larry_pca_mfm.npz --out mfm_predictions.npz
```
`mfm_predictions.npz` holds `fm_pred_day4/day6` and `mfm_pred_day4/day6` in the
original 50-d PCA coordinates, ready for W2/energy/MMD scoring against real day4.
