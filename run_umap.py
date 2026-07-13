#!/usr/bin/env python
"""Fit one UMAP on ground-truth day2/4/6 PCA (50-d) and transform every method's
day4 prediction into the SAME embedding so all contours share one reducer.

Expects umap_fit_bundle.npz with keys:
  fit_day2, fit_day4, fit_day6  (ground-truth PCA per day)
  plus one key per method holding its day4 prediction PCA:
  OT-McCann, MOSCOT, CellOT, scIMF, EntangledSBM, CytoBridge, MFM, straight-FM
Outputs umap_coords.npz.
"""
import numpy as np
import time
import umap


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    z = np.load("umap_fit_bundle.npz", allow_pickle=True)
    fit = np.vstack([z["fit_day2"], z["fit_day4"], z["fit_day6"]]).astype(np.float32)
    log("fit set", fit.shape)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=2,
                        metric="euclidean", random_state=0)
    reducer.fit(fit)
    log("UMAP fit done")

    out = {}
    out["day2"] = reducer.transform(z["fit_day2"]).astype(np.float32)
    out["day4"] = reducer.transform(z["fit_day4"]).astype(np.float32)
    out["day6"] = reducer.transform(z["fit_day6"]).astype(np.float32)
    log("gt transformed")
    for m in ["OT-McCann", "MOSCOT", "CellOT", "scIMF", "EntangledSBM",
              "CytoBridge", "MFM", "straight-FM"]:
        if m not in z.files:
            log("skip (absent)", m)
            continue
        out[m] = reducer.transform(z[m]).astype(np.float32)
        log("transformed", m, out[m].shape)
    np.savez_compressed("umap_coords.npz", **out)
    log("saved umap_coords.npz", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
