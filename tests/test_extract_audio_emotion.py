from __future__ import annotations

import unittest

import numpy as np
import torch

from tools.extract_audio_emotion import (
    SLIDER_BIAS,
    _estimate_slider_vector,
)


class AudioEmotionProjectionTests(unittest.TestCase):
    def test_exact_prototype_recovers_its_axis(self) -> None:
        prototypes = torch.eye(8)
        embedding = prototypes[2].unsqueeze(0)

        effective, ui_values, fit_cosine, residual = _estimate_slider_vector(
            embedding, prototypes
        )

        expected = np.zeros(8)
        expected[2] = 0.8  # WebUI caps the total effective emotion strength.
        np.testing.assert_allclose(effective, expected, atol=1e-8)
        np.testing.assert_allclose(ui_values, expected / SLIDER_BIAS, atol=1e-8)
        self.assertAlmostEqual(fit_cosine, 1.0)
        self.assertAlmostEqual(residual, 0.0)

    def test_zero_embedding_returns_zero_vector(self) -> None:
        effective, ui_values, fit_cosine, residual = _estimate_slider_vector(
            torch.zeros(1, 8), torch.eye(8)
        )

        np.testing.assert_array_equal(effective, np.zeros(8))
        np.testing.assert_array_equal(ui_values, np.zeros(8))
        self.assertEqual(fit_cosine, 0.0)
        self.assertEqual(residual, 0.0)


if __name__ == "__main__":
    unittest.main()
