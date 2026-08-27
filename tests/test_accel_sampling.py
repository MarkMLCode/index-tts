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
from indextts.accel.kv_manager import KVCacheManager, Seq


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

    def test_sampling_uses_transformers_multinomial_path(self) -> None:
        engine = sampling_engine()
        logits = torch.tensor([[3.2, 1.1, -0.4, 2.7, 0.3, -1.2]])
        sequence = Seq([1, 3], block_size=8)

        processed = engine._apply_repetition_penalty(
            logits.clone(), [sequence]
        )
        processed = engine._warp_logits(processed)
        probabilities = torch.softmax(processed, dim=-1, dtype=torch.float32)

        torch.manual_seed(1234)
        expected = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        torch.manual_seed(1234)
        actual = engine._sample_tokens(logits.clone(), [sequence])

        self.assertTrue(torch.equal(actual, expected))

    def test_deterministic_beam_candidates_rank_across_all_parents(self) -> None:
        engine = sampling_engine()
        engine._gen_do_sample = False
        engine._gen_rep_penalty = 1.0
        logits = torch.tensor([[4.0, 1.0, 0.0], [2.0, 3.0, -1.0]])
        sequences = [Seq([10], block_size=8), Seq([11], block_size=8)]
        beam_scores = torch.tensor([0.0, -0.25])

        candidates = engine._beam_candidates(
            logits, sequences, beam_scores, num_beams=2
        )

        self.assertEqual([(parent, token) for _, parent, token in candidates], [
            (0, 0),
            (1, 1),
            (1, 0),
            (0, 1),
        ])

    def test_sampled_beam_candidates_match_transformers_processor_order(self) -> None:
        engine = sampling_engine()
        logits = torch.tensor([
            [3.2, 1.1, -0.4, 2.7, 0.3, -1.2],
            [1.7, 2.4, 0.2, -0.5, 1.1, -1.0],
        ])
        sequences = [Seq([1, 3], block_size=8), Seq([0, 4], block_size=8)]
        beam_scores = torch.tensor([0.0, -0.35])

        input_ids = torch.tensor([sequence.token_ids for sequence in sequences])
        expected = torch.log_softmax(logits.float(), dim=-1)
        expected = RepetitionPenaltyLogitsProcessor(1.7)(input_ids, expected)
        expected = TemperatureLogitsWarper(0.8)(input_ids, expected)
        expected = TopKLogitsWarper(3, min_tokens_to_keep=2)(input_ids, expected)
        expected = TopPLogitsWarper(0.75, min_tokens_to_keep=2)(input_ids, expected)
        expected = (expected + beam_scores[:, None]).reshape(-1)

        torch.manual_seed(4321)
        indices = torch.multinomial(
            torch.softmax(expected, dim=0), num_samples=4, replacement=False
        )
        scores = expected[indices]
        order = torch.argsort(scores, descending=True)
        indices = indices[order]
        scores = scores[order]
        expected_candidates = [
            (score, index // logits.size(-1), index % logits.size(-1))
            for score, index in zip(scores.tolist(), indices.tolist())
        ]

        torch.manual_seed(4321)
        actual = engine._beam_candidates(
            logits, sequences, beam_scores, num_beams=2
        )

        self.assertEqual(
            [(parent, token) for _, parent, token in actual],
            [(parent, token) for _, parent, token in expected_candidates],
        )
        torch.testing.assert_close(
            torch.tensor([score for score, _, _ in actual]),
            torch.tensor([score for score, _, _ in expected_candidates]),
        )

    def test_beam_search_returns_best_completed_hypothesis(self) -> None:
        engine = sampling_engine()
        engine._gen_do_sample = False
        engine._gen_rep_penalty = 1.0
        engine.kv_manager = KVCacheManager(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            block_size=4,
            num_blocks=10,
            dtype=torch.float32,
        )
        prompt = Seq([1], block_size=4)
        engine.kv_manager.allocate(prompt)
        engine.current_sequences = [prompt]

        engine._prepare_decode = lambda sequences: (
            torch.zeros(len(sequences), dtype=torch.long),
            torch.zeros(len(sequences), dtype=torch.long),
        )
        engine._run_decode_with_graph = lambda *args, **kwargs: torch.zeros(
            len(engine.current_sequences), 1
        )

        def scripted_logits(_hidden: torch.Tensor) -> torch.Tensor:
            rows = []
            for sequence in engine.current_sequences:
                if sequence.last_token == 0:
                    rows.append([0.0, -10.0, -10.0, 10.0])
                else:
                    rows.append([-10.0, -10.0, 10.0, 0.0])
            return torch.tensor(rows)

        engine._compute_logits = scripted_logits
        output = engine._generate_beams(
            logits=torch.tensor([[10.0, 0.0, -1.0, -2.0]]),
            sequence=prompt,
            prompt_tokens=[1],
            max_new_tokens=2,
            num_beams=2,
            length_penalty=0.0,
            stop_tokens=[3],
            tts_mel_embedding=None,
            tts_text_pos_embedding=None,
            device=torch.device("cpu"),
        )

        self.assertEqual(output.tolist(), [[1, 0]])
        self.assertFalse(engine.current_sequences)
        self.assertTrue(
            all(block.ref_cnt == 0 for block in engine.kv_manager.blocks)
        )

    def test_single_winning_child_reuses_parent_cache(self) -> None:
        engine = sampling_engine()
        engine.kv_manager = KVCacheManager(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            block_size=4,
            num_blocks=4,
            dtype=torch.float32,
        )
        parent = Seq([1], block_size=4)
        engine.kv_manager.allocate(parent)
        original_block = parent.block_table[-1]

        children, scores = engine._branch_beam_sequences(
            [parent], [(0.0, 0, 2)]
        )

        self.assertIs(children[0], parent)
        self.assertEqual(children[0].block_table[-1], original_block)
        self.assertEqual(scores, [0.0])

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

    def test_resident_switch_preserves_cached_float32_head(self) -> None:
        engine = sampling_engine()
        engine.current_sequences = []
        reset_calls = []
        engine.kv_manager = type("Manager", (), {"reset": lambda self: reset_calls.append(True)})()
        head = nn.Linear(2, 2)
        engine._lm_head_fp32 = head
        engine.reset_model_state(weights_changed=False)
        self.assertEqual(reset_calls, [True])
        self.assertIs(engine._lm_head_fp32, head)


if __name__ == "__main__":
    unittest.main()
