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

    def test_head_uses_resident_precision_without_autocast(self) -> None:
        engine = sampling_engine()
        hidden = torch.tensor([[0.1234567, -1.234567, 2.345678]], dtype=torch.float32)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                head = nn.Linear(3, 4, bias=True).to(dtype)
                engine.lm_head = head
                actual = engine._compute_logits(hidden)
                expected = head(hidden.to(dtype))
                self.assertEqual(actual.dtype, dtype)
                self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(hasattr(engine, "_lm_head_fp32"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_autocast_head_matches_old_fp32_copy(self) -> None:
        engine = sampling_engine()
        for dtype in (torch.float16, torch.bfloat16):
            head = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, 8194)).cuda().to(dtype).eval()
            engine.lm_head = head
            old_head = copy.deepcopy(head).float()
            for input_dtype in (dtype, torch.float32):
                for batch in (1, 3):
                    with self.subTest(dtype=dtype, input_dtype=input_dtype, batch=batch):
                        hidden = torch.randn(batch, 1280, device="cuda", dtype=input_dtype)
                        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                            expected = old_head(hidden.float())
                            actual = engine._compute_logits(hidden)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertIs(engine.lm_head, head)
                        self.assertFalse(hasattr(engine, "_lm_head_fp32"))

    def test_half_logits_are_promoted_for_sampling_and_beam_scoring(self) -> None:
        engine = sampling_engine()
        original = engine._apply_repetition_penalty
        seen = []

        def checked(logits, sequences):
            seen.append(logits.dtype)
            return original(logits, sequences)

        engine._apply_repetition_penalty = checked
        logits = torch.tensor([[3.2, 1.1, -0.4, 2.7]], dtype=torch.bfloat16)
        sequences = [Seq([1, 3], block_size=8)]
        engine._sample_tokens(logits.clone(), sequences)
        engine._beam_candidates(logits, sequences, torch.zeros(1), num_beams=2)
        self.assertEqual(seen, [torch.float32, torch.float32])

    def test_model_reset_clears_kv_state_but_preserves_graphs_and_head(self) -> None:
        engine = sampling_engine()

        class Manager:
            reset_called = False

            def reset(self) -> None:
                self.reset_called = True

        engine.current_sequences = []
        engine.kv_manager = Manager()
        engine.lm_head = head = nn.Linear(2, 2)
        engine.graphs = graphs = {1: object()}
        engine.graph_captured = True

        engine.reset_model_state()

        self.assertTrue(engine.kv_manager.reset_called)
        self.assertIs(engine.lm_head, head)
        self.assertIs(engine.graphs, graphs)
        self.assertTrue(engine.graph_captured)

    def test_reset_rejects_active_generation(self) -> None:
        engine = sampling_engine()
        engine.current_sequences = [object()]
        with self.assertRaisesRegex(RuntimeError, "during generation"):
            engine.reset_model_state()


class StandardBeamSamplingTests(unittest.TestCase):
    def test_regular_distribution_preserves_seeded_multinomial_results(self):
        from indextts.gpt.transformers_generation_utils import _sample_beam_tokens

        scores = torch.tensor([[0.0, -0.5, -1.0, -1.5], [-1.5, -1.0, -0.5, 0.0]])
        torch.manual_seed(123)
        expected = torch.multinomial(torch.softmax(scores, dim=-1), num_samples=3)
        torch.manual_seed(123)
        actual = _sample_beam_tokens(scores, 3)
        self.assertTrue(torch.equal(actual, expected))

    def test_underflow_sampling_keeps_distinct_finite_candidates(self):
        from indextts.gpt.transformers_generation_utils import _sample_beam_tokens

        scores = torch.tensor([[0.0, -150.0, -200.0, -1e9, -torch.inf]])
        selected = _sample_beam_tokens(scores, 4)
        self.assertEqual(set(selected[0].tolist()), {0, 1, 2, 3})

    def test_invalid_scores_fail_before_multinomial(self):
        from unittest.mock import patch

        from indextts.gpt.transformers_generation_utils import _sample_beam_tokens

        for row in ([0.0, torch.nan], [0.0, torch.inf], [-torch.inf, -torch.inf]):
            with self.subTest(row=row), patch('torch.multinomial') as sampler:
                with self.assertRaisesRegex(RuntimeError, 'invalid probabilities'):
                    _sample_beam_tokens(torch.tensor([row]), 2)
                sampler.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'fewer finite candidates'):
            _sample_beam_tokens(torch.tensor([[0.0, -torch.inf]]), 2)

    def test_peaked_eos_distribution_does_not_leave_all_beams_masked(self):
        from transformers import GPT2Config, GPT2Model

        from indextts.gpt.model_v2 import GPT2InferenceModel, LearnedPositionEmbeddings

        config = GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=32, vocab_size=8194)
        model = GPT2InferenceModel(
            config, GPT2Model(config), LearnedPositionEmbeddings(32, 8),
            nn.Embedding(8194, 8), nn.LayerNorm(8), nn.Linear(8, 8194), kv_cache=True,
        ).eval()
        model.store_mel_emb(torch.zeros(1, 1, 8))

        def peaked_logits(module, inputs, output):
            # Finite logits, but the only non-EOS alternative underflows after
            # temperature scaling. Sampling six tokens used to select masked
            # candidates and poison the next step with all -inf beam scores.
            output.logits.fill_(-1000.0)
            output.logits[..., 20] = -30.0
            output.logits[..., 8193] = 0.0

        model.register_forward_hook(peaked_logits)
        torch.manual_seed(0)
        result = model.generate(
            torch.tensor([[1, 8192]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
            do_sample=True, num_beams=3, temperature=0.2, top_p=0.95, top_k=30,
            bos_token_id=8192, eos_token_id=8193, pad_token_id=8193,
            max_new_tokens=5,
        )
        self.assertEqual(result[0, -1].item(), 8193)


if __name__ == "__main__":
    unittest.main()
