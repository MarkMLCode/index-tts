import random

import numpy as np
import pytest
import torch

from indextts.utils.seed import apply_seed, normalize_seed


@pytest.mark.parametrize("seed", [0, 42, 2**60 + 123])
def test_seed_resets_all_synthesis_rngs(seed):
    def draw():
        return random.random(), np.random.rand(4).tolist(), torch.rand(4).tolist()

    apply_seed(seed)
    expected = draw()
    draw()
    apply_seed(seed)
    assert draw() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_seed_resets_cuda_rng():
    apply_seed(42)
    expected = torch.rand(8, device="cuda")
    torch.rand(32, device="cuda")
    apply_seed(42)
    assert torch.equal(torch.rand(8, device="cuda"), expected)


@pytest.mark.parametrize("value", [2**60 + 123, str(2**60 + 123)])
def test_large_seed_is_not_rounded_through_float(value):
    assert normalize_seed(value) == 2**60 + 123


def test_integer_ui_float_is_accepted():
    assert normalize_seed(42.0) == 42


@pytest.mark.parametrize("value", [1.5, "invalid", float("inf"), float("nan")])
def test_invalid_seed_is_rejected(value):
    with pytest.raises(ValueError, match="Seed must be an integer"):
        normalize_seed(value)


@pytest.mark.parametrize("value", [-1, None, ""])
def test_random_mode_uses_system_entropy_not_synthesis_rngs(monkeypatch, value):
    seeds = iter([2**60 + 123, 2**60 + 456])

    class SystemEntropy:
        def randrange(self, start, stop):
            assert (start, stop) == (0, 2**63 - 1)
            return next(seeds)

    monkeypatch.setattr(random, "SystemRandom", SystemEntropy)
    apply_seed(42)
    assert normalize_seed(value) == 2**60 + 123
    apply_seed(42)
    assert normalize_seed(value) == 2**60 + 456
