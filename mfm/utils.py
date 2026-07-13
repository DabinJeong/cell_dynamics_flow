"""Shared utilities (seeding, evaluation metrics).

Trimmed analogue of kkapusniak/metric-flow-matching mfm/utils.py: keeps the
seeding helper and the Wasserstein distance used for evaluation; drops the
plotting helpers (arch/lidar/sphere/image) that the LARRY benchmark doesn't use.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def wasserstein_distance(x0, x1, p=2):
    """Exact empirical p-Wasserstein between two point clouds via POT EMD.

    Mirrors the metric used in the official evaluation (mfm.utils.wasserstein_distance).
    """
    import ot as pot
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    x1 = torch.as_tensor(x1, dtype=torch.float32)
    a = pot.unif(x0.shape[0])
    b = pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1, p=2)
    if p == 2:
        M = M ** 2
    Mnp = M.detach().cpu().numpy()
    cost = pot.emd2(a, b, Mnp, numItermax=1_000_000)
    return cost ** (1.0 / p) if p == 2 else cost
