#!/usr/bin/env python
"""Build larry_pca_mfm.npz from the LARRY state-fate day2/day4/day6 npz files.

Each source npz (day2.npz/day4.npz/day6.npz) carries X_pca (N, 50). We bundle the
three PCA matrices under day2_pca/day4_pca/day6_pca and record t_day4. day2=day 2,
day4=day 4, day6=day 6, so on a normalized [0,1] time axis day4 sits at
(4-2)/(6-2) = 0.5.

Usage:
    python prepare_data.py --src_dir <dir with day{2,4,6}.npz> --out larry_pca_mfm.npz
"""

import argparse
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True,
                    help="directory containing day2.npz, day4.npz, day6.npz")
    ap.add_argument("--out", default="larry_pca_mfm.npz")
    ap.add_argument("--dim", type=int, default=50)
    args = ap.parse_args()

    def load(day):
        z = np.load(os.path.join(args.src_dir, f"day{day}.npz"), allow_pickle=True)
        return z["X_pca"][:, :args.dim].astype(np.float32)

    d2, d4, d6 = load(2), load(4), load(6)
    t_day4 = (4 - 2) / (6 - 2)  # 0.5
    np.savez_compressed(args.out, day2_pca=d2, day4_pca=d4, day6_pca=d6,
                        t_day4=np.float32(t_day4))
    print(f"saved {args.out}: day2={d2.shape} day4={d4.shape} day6={d6.shape} "
          f"t_day4={t_day4}")


if __name__ == "__main__":
    main()
