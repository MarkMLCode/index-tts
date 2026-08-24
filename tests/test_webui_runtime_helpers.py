from __future__ import annotations

import ast
import json
import os
import random
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

import gradio as gr
import numpy as np
import torch


WEBUI_PATH = Path(__file__).resolve().parents[1] / "webui.py"
HELPERS = {
    "_portable_model_path",
    "_save_last_gpt_checkpoint",
    "_read_last_gpt_checkpoint",
    "_normalize_seed",
    "_apply_seed",
    "_format_stream_chunk",
    "gen_single",
}


def load_helpers(**overrides):
    tree = ast.parse(WEBUI_PATH.read_text(encoding="utf-8"))
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in HELPERS
    ]
    namespace = {
        "Path": Path,
        "json": json,
        "os": os,
        "random": random,
        "sys": sys,
        "threading": threading,
        "time": time,
        "gr": gr,
        "np": np,
    }
    namespace.update(overrides)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(WEBUI_PATH), "exec"),
        namespace,
    )
    return namespace


class FakeTTS:
    def __init__(self):
        self.gr_progress = None
        self.samples = []

    @staticmethod
    def normalize_emo_vec(values, apply_bias=True):
        del apply_bias
        return values

    def infer(self, **kwargs):
        self.samples.append(torch.rand(4))
        if kwargs.get("stream_return"):
            return iter(
                [
                    torch.tensor([[0.0, 32767.0, -32767.0]]),
                    torch.tensor([[0.0, 1000.0]]),
                ]
            )
        return kwargs["output_path"]


class WebUIRuntimeHelperTests(unittest.TestCase):
    def test_preset_helper_calls_match_callback_signatures(self):
        tree = ast.parse(WEBUI_PATH.read_text(encoding="utf-8"))
        targets = {"_build_preset_data", "_format_preset_preview", "on_preset_save"}
        arities = {
            node.name: len(node.args.args)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in targets
        }

        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id in targets:
                self.assertEqual(
                    len(call.args),
                    arities[call.func.id],
                    f"{call.func.id} call does not match its callback signature",
                )

    def test_seed_is_reproducible_and_random_mode_resolves(self):
        helpers = load_helpers()
        helpers["_apply_seed"](1234)
        first = torch.rand(8)
        helpers["_apply_seed"](1234)
        self.assertTrue(torch.equal(first, torch.rand(8)))
        self.assertGreaterEqual(helpers["_normalize_seed"](-1), 0)

    def test_stream_chunk_is_gradio_pcm(self):
        helpers = load_helpers()
        sampling_rate, audio = helpers["_format_stream_chunk"](
            torch.tensor([[0.0, 32767.0, -32767.0]])
        )
        self.assertEqual(sampling_rate, 22050)
        self.assertEqual(audio.dtype, np.int16)
        self.assertEqual(audio.shape, (3,))

    def test_last_model_setting_round_trips_as_portable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = root / ".webui_settings.json"
            helpers = load_helpers(
                current_dir=str(root),
                WEBUI_SETTINGS_PATH=settings,
            )
            checkpoint = root / "_trained" / "voice.pth"
            helpers["_save_last_gpt_checkpoint"](checkpoint)
            self.assertEqual(
                helpers["_read_last_gpt_checkpoint"](), str(checkpoint.resolve())
            )
            saved = json.loads(settings.read_text(encoding="utf-8"))
            self.assertEqual(saved["last_gpt_checkpoint"], "_trained/voice.pth")

    def test_generation_supports_normal_and_streaming_outputs(self):
        fake_tts = FakeTTS()
        helpers = load_helpers(
            tts=fake_tts,
            MODEL_LOCK=threading.RLock(),
            cmd_args=types.SimpleNamespace(verbose=False),
            IS_V25=True,
        )
        common = [
            0,
            "voice.wav",
            "Test text",
            "EN",
            None,
            0.65,
            *([0.0] * 8),
            "",
            False,
            120,
            1.0,
            42,
        ]
        advanced = [True, 0.8, 30, 0.8, 0.0, 3, 10.0, 1500]

        normal = list(helpers["gen_single"](*common, False, *advanced))
        self.assertEqual(len(normal), 2)  # initialize stream, then complete WAV
        self.assertIsNone(normal[0][0])
        first_sample = fake_tts.samples[-1]

        list(helpers["gen_single"](*common, False, *advanced))
        self.assertTrue(torch.equal(first_sample, fake_tts.samples[-1]))

        streamed = list(helpers["gen_single"](*common, True, *advanced))
        self.assertEqual(len(streamed), 4)  # clear, two chunks, final WAV
        self.assertEqual(streamed[1][0][0], 22050)


if __name__ == "__main__":
    unittest.main()
