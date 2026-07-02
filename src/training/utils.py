"""
Shared training utility functions.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Configure random seeds and deterministic backend settings.

    This covers Python, NumPy, PyTorch CPU/CUDA RNGs, and cuDNN algorithm
    selection. Full bitwise reproducibility can still depend on the active
    CUDA runtime and operator support.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)