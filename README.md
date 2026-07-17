# FateFlow: Flow matching for interpolating cell dynamics during development

## The scientific question

Development is a continuous, highly dynamic process, but we can only ever watch it in snapshots. Single-cell RNA sequencing is *destructive*: a cell is consumed the moment it is measured. What we get instead is a set of unpaired marginal distributions, one per sampled timepoint.

> **Given the cell states we *did* measure, can we recover the states at the
> timepoints we *did not* — the continuous dynamics hidden between snapshots?**

FateFlow frames this as an **interpolation problem in gene-expression space**: learn a
transport that carries the earlier population into the later one, and read off the
intermediate populations it passes through.

## Benchmark: recover a held-out timepoint

We test this on the LARRY state-fate hematopoiesis dataset
([Weinreb et al. 2020, *Science*](https://www.science.org/doi/abs/10.1126/science.aaw3381)),
which measures mouse hematopoietic progenitors at **day 2, day 4, and day 6**.

- **Observed:** day 2 (source) and day 6 (target)
- **Held out:** day 4 — the intermediate marginal we try to reconstruct

Because day 4 is never shown to the model, how closely the interpolated population
matches the real day-4 cells is a direct, quantitative test of whether the recovered
dynamics are biologically faithful.

![dataset]({{artifact:art_2f47a253-eff4-48ad-b607-ef6cece4dc0c}})

We use the HVG-filtered, PCA-embedded `invitro-hvg.h5ad` with cells grouped by collection time (day 2 / 4 / 6).

## What we propose: FateFlow

A natural framework is flow matching: learn how the day-2 population
flows toward day 6, then read off the intermediate state.

- The simplest choice connects cells along straight lines (optimal transport). This works well in many settings, though in a branched landscape the paths pass through the sparse region between fates.
- MFM refines this by bending paths toward denser regions, following the data manifold, keeping trajectories close to where real cells live. Its focus is on the *geometry of the path*.

These approaches shape *how cells travel*. Our observation is that the intermediate population is captured well by a complementary view: *which fates the cells occupy*.

We propose **FateFlow**, based on the observation that the intermediate cell population is close to a **reweighting of a fixed set of
fates** rather than movement into new territory. We build on this by separating two
questions:

1. **How the fate proportions shift**: we identify 12 fate groups (modes) from the
   end timepoint, then use optimal transport to interpolate how each group's share evolves
   from the start timepoint.
2. **How cells move within each fate**: we learn a flow that is conditioned on the
   target fate. Because each cell knows the fate it is heading toward, the branches
   stay distinct and trajectories remain on their own arm.

Once trained, we assign each cell its fate and integrate the flow to produce the
day-4 (intermediate) and day-6 (endpoint) populations.

## Repository layout

The package builds upon the [official MFM codebase](https://github.com/kkapusniak/metric-flow-matching) [[Kapuśniak et al. 2024, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf)].

<!-- TODO: confirm these match your current CLI before publishing -->
```
cdf/mfm/
├── flow_matchers/      # EMA, MetricFlowMatcher, geopath/flow-net training
├── geo_metrics/        # LAND, RBF, metric factory
├── networks/           # velocity / interpolant networks
├── dataloaders/        # trajectory_data
├── train/              # main entrypoint, argument parsers
└── utils/
```

## Installation

<!-- TODO: fill in with your actual environment manager -->
```bash
git clone git@github.com:DabinJeong/cell_dynamics_flow.git
cd cell_dynamics_flow
pip install -e .
```

## Usage

<!-- TODO: replace with the exact command your entrypoint exposes -->
```bash
# Train and generate the held-out day-4 interpolation
python -m cdf.mfm.train.main --config <config.yaml>

# Project all methods' predictions into a shared UMAP for comparison
python run_umap.py
```

## Baselines

Baseline implementations for reproducing the benchmarking results: TBU.

## Results

FateFlow's day-4 reconstruction is compared against the real held-out population and against straight-line flow matching and OT-based baselines using distributional metrics (2-Wasserstein, energy distance, MMD) in PCA space, plus qualitative overlays on the shared UMAP.

![benchmark]({{artifact:art_0494a3a8-cde9-4152-9b4f-ad7faf8b8600}})

![benchmark_viz]({{artifact:art_4ec8110b-b753-475a-a512-a79608c528a9}})
