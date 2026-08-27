"""Shared per-request seeds for API and WebUI generation."""

import random

import numpy as np


def normalize_seed(seed_value=-1):
    """Resolve random mode using OS entropy, independent of synthesis RNGs."""
    if seed_value is None or str(seed_value).strip() == "":
        seed_value = -1
    try:
        # Do not convert integers through float: large replay seeds lose bits.
        seed = int(seed_value)
        if isinstance(seed_value, float) and seed_value != seed:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Seed must be an integer or -1 for a random result.") from None
    if seed < 0:
        return random.SystemRandom().randrange(0, 2**63 - 1)
    return seed


def apply_seed(seed):
    """Reset synthesis RNGs immediately before inference, under the model lock."""
    # Keep API imports light; torch.manual_seed seeds CPU and accelerator RNGs.
    import torch

    random.seed(seed % (2**32))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed % (2**63 - 1))
