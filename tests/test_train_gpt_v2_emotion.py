from __future__ import annotations

import unittest

from torch import nn

from trainers.train_gpt_v2 import (
    CONDITIONING_INTERFACE_MODULES,
    freeze_conditioning_interface,
    target_emotion_path,
)


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in CONDITIONING_INTERFACE_MODULES:
            setattr(self, name, nn.Linear(2, 2))
        self.gpt = nn.Linear(2, 2)


class EmotionFineTuningTests(unittest.TestCase):
    def test_target_emotion_is_preferred_for_paired_training(self) -> None:
        record = {
            "prompt_emo_vec_path": "emo_vec/prompt.npy",
            "target_emo_vec_path": "emo_vec/target.npy",
        }
        self.assertEqual(target_emotion_path(record), "emo_vec/target.npy")

    def test_prompt_emotion_is_only_a_legacy_fallback(self) -> None:
        self.assertEqual(
            target_emotion_path({"prompt_emo_vec_path": "emo_vec/prompt.npy"}),
            "emo_vec/prompt.npy",
        )

    def test_conditioning_interface_is_frozen_without_freezing_gpt(self) -> None:
        model = DummyModel()
        frozen = freeze_conditioning_interface(model)
        self.assertEqual(frozen, list(CONDITIONING_INTERFACE_MODULES))
        for name in CONDITIONING_INTERFACE_MODULES:
            self.assertTrue(all(not parameter.requires_grad for parameter in getattr(model, name).parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.gpt.parameters()))


if __name__ == "__main__":
    unittest.main()
