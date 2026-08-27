"""Opt-in real-checkpoint regression: INDEXTTS_STARTUP_GPU_CHECKPOINT=/path/model.pth.

Requires auxiliary checkpoints, a CUDA GPU, and enough free VRAM for one engine.
Runs the original and optimized loaders sequentially; no audio files are written.
"""

import gc
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from indextts.utils.seed import apply_seed

@pytest.mark.gpu
@pytest.mark.skipif(not os.environ.get("INDEXTTS_STARTUP_GPU_CHECKPOINT"), reason="opt-in GPU/checkpoint test")
def test_real_checkpoint_loading_and_warmup_preserve_audio(monkeypatch):
    from indextts import infer_v2_5
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint
    from indextts.utils.warmup import warmup_engine

    checkpoint = str(Path(os.environ["INDEXTTS_STARTUP_GPU_CHECKPOINT"]).resolve())
    speaker = str(Path(os.environ.get("INDEXTTS_STARTUP_GPU_SPEAKER", "examples/voice_01.wav")).resolve())
    assert torch.cuda.is_available()
    original_post_init = UnifiedVoice.post_init_gpt2_config

    def legacy_loader(factory, path):
        model = factory()
        load_checkpoint(model, path)
        return model

    def legacy_post_init(self, **kwargs):
        kwargs["reuse_gpt_weights"] = False
        return original_post_init(self, **kwargs)

    samples = []
    for optimized in (False, True):
        with monkeypatch.context() as patch:
            if not optimized:
                patch.setattr(infer_v2_5, "load_inference_model", legacy_loader)
                patch.setattr(UnifiedVoice, "post_init_gpt2_config", legacy_post_init)
            started = time.perf_counter()
            engine = infer_v2_5.IndexTTS2(
                gpt_checkpoint_path=checkpoint, use_bf16=True, use_accel=True,
            )
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - started
            warmup_seconds = 0.0
            if optimized:
                started = time.perf_counter()
                warmup_engine(engine, [checkpoint], speaker)
                torch.cuda.synchronize()
                warmup_seconds = time.perf_counter() - started
                assert engine.gpt_path == checkpoint
                head = engine.gpt.accel_engine._lm_head_fp32
                assert head is not None
                engine.set_gpt_checkpoint(checkpoint)
                assert engine.gpt.accel_engine._lm_head_fp32 is head
                del head
            apply_seed(1234)
            result = engine.infer(
                spk_audio_prompt=speaker, text="Hello, world!", lang="EN",
                output_path=None, max_mel_tokens=256,
            )
            samples.append(result)
            print(f"STARTUP_BENCHMARK optimized={optimized} load={load_seconds:.3f}s warmup={warmup_seconds:.3f}s")
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    assert samples[0][0] == samples[1][0]
    assert np.array_equal(samples[0][1], samples[1][1]), "Audio differs from original loading path"


@pytest.mark.gpu
@pytest.mark.skipif(
    not os.environ.get("INDEXTTS_STARTUP_GPU_CHECKPOINT") or not os.environ.get("INDEXTTS_STARTUP_GPU_SECOND_CHECKPOINT"),
    reason="opt-in two-checkpoint GPU test",
)
def test_resident_warmup_and_switch_preserve_audio():
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.warmup import warmup_engine

    first = str(Path(os.environ["INDEXTTS_STARTUP_GPU_CHECKPOINT"]).resolve())
    second = str(Path(os.environ["INDEXTTS_STARTUP_GPU_SECOND_CHECKPOINT"]).resolve())
    speaker = str(Path("examples/voice_01.wav").resolve())
    engine = IndexTTS2(
        gpt_checkpoint_path=first, gpt_checkpoint_paths=[first, second],
        use_bf16=True, use_accel=True,
    )
    warmup_engine(engine, [first, second], speaker)
    assert engine.gpt_path == first
    first_accel = engine._gpt_models[first].accel_engine
    second_accel = engine._gpt_models[second].accel_engine
    assert first_accel.kv_manager is second_accel.kv_manager
    assert first_accel.graph_captured and second_accel.graph_captured
    heads = (first_accel._lm_head_fp32, second_accel._lm_head_fp32)
    assert all(head is not None for head in heads)
    outputs = []
    for path in (first, second, first):
        engine.set_gpt_checkpoint(path)
        apply_seed(1234)
        outputs.append(engine.infer(
            spk_audio_prompt=speaker, text="Hello, world!", lang="EN",
            output_path=None, max_mel_tokens=256,
        ))
    assert np.array_equal(outputs[0][1], outputs[2][1])
    assert first_accel._lm_head_fp32 is heads[0]
    assert second_accel._lm_head_fp32 is heads[1]
    del engine, first_accel, second_accel, heads
    gc.collect()
    torch.cuda.empty_cache()
