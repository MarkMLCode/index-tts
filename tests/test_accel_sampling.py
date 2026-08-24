from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from transformers import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from indextts.accel.accel_engine import AccelInferenceEngine
from indextts.accel.kv_manager import Seq


def sampling_engine() -> AccelInferenceEngine:
    engine = AccelInferenceEngine.__new__(AccelInferenceEngine)
    engine._gen_temperature = 0.8
    engine._gen_top_k = 3
    engine._gen_top_p = 0.75
    engine._gen_rep_penalty = 1.7
    engine._gen_do_sample = True
    return engine


class AccelSamplingParityTests(unittest.TestCase):
    def test_tts_positions_match_standard_inference_model(self) -> None:
        engine = sampling_engine()
        engine._tts_prompt_len = 48

        self.assertEqual(engine.TTS_START_POSITION, 0)
        self.assertEqual(engine._tts_decode_position(49), 2)
        self.assertEqual(engine._tts_decode_position(50), 3)

    def test_logits_processors_match_transformers_single_beam_pipeline(self) -> None:
        engine = sampling_engine()
        original = torch.tensor([[3.2, 1.1, -0.4, 2.7, 0.3, -1.2]])
        sequence = Seq([1, 3], block_size=8)

        actual = engine._apply_repetition_penalty(original.clone(), [sequence])
        actual = engine._warp_logits(actual)

        input_ids = torch.tensor([[1, 3]])
        expected = RepetitionPenaltyLogitsProcessor(1.7)(input_ids, original.clone())
        expected = TemperatureLogitsWarper(0.8)(input_ids, expected)
        expected = TopKLogitsWarper(3)(input_ids, expected)
        expected = TopPLogitsWarper(0.75)(input_ids, expected)

        torch.testing.assert_close(actual, expected)

    def test_greedy_mode_still_applies_repetition_penalty(self) -> None:
        engine = sampling_engine()
        engine._gen_do_sample = False
        engine._gen_rep_penalty = 2.0
        logits = torch.tensor([[10.0, 9.0, 1.0]])

        token = engine._sample_tokens(logits, [Seq([0], block_size=8)])

        self.assertEqual(token.item(), 1)

    def test_half_precision_head_projects_logits_in_float32(self) -> None:
        engine = sampling_engine()
        head = nn.Linear(3, 4, bias=True).half()
        engine.lm_head = head
        engine._lm_head_fp32 = None
        hidden = torch.tensor([[0.1234567, -1.234567, 2.345678]], dtype=torch.float32)

        actual = engine._compute_logits(hidden)
        expected = copy.deepcopy(head).float()(hidden)

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected)

    def test_model_reset_invalidates_cached_float32_head(self) -> None:
        engine = sampling_engine()

        class Manager:
            reset_called = False

            def reset(self) -> None:
                self.reset_called = True

        engine.current_sequences = []
        engine.kv_manager = Manager()
        engine._lm_head_fp32 = nn.Linear(2, 2)

        engine.reset_model_state()

        self.assertTrue(engine.kv_manager.reset_called)
        self.assertIsNone(engine._lm_head_fp32)


if __name__ == "__main__":
    unittest.main()
