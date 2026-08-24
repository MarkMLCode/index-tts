from __future__ import annotations

import unittest

import torch
from transformers import GPT2Config

from indextts.accel.gpt2_accel import GPT2AccelAttention


class AccelAttentionDtypeTests(unittest.TestCase):
    def setUp(self) -> None:
        config = GPT2Config(n_embd=16, n_head=2, n_layer=1)
        self.attention = GPT2AccelAttention(config)

    def test_bfloat16_cache_selects_bfloat16(self) -> None:
        self.attention.accel_attn.k_cache = torch.empty(1, dtype=torch.bfloat16)

        actual = self.attention._flash_attention_dtype(torch.bfloat16)

        self.assertEqual(actual, torch.bfloat16)

    def test_cache_dtype_wins_when_input_dtype_differs(self) -> None:
        self.attention.accel_attn.k_cache = torch.empty(1, dtype=torch.bfloat16)

        actual = self.attention._flash_attention_dtype(torch.float16)

        self.assertEqual(actual, torch.bfloat16)

    def test_float32_without_cache_falls_back_to_float16(self) -> None:
        self.attention.accel_attn.k_cache = torch.empty(0)

        actual = self.attention._flash_attention_dtype(torch.float32)

        self.assertEqual(actual, torch.float16)


if __name__ == "__main__":
    unittest.main()
