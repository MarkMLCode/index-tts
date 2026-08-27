import random

import numpy as np
import pytest
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from indextts.accel.gpt2_accel import GPT2AccelModel
from indextts.utils.checkpoint import load_checkpoint, load_inference_model
from indextts.utils.warmup import warmup_engine


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 4)
        self.register_buffer("offset", torch.ones(4))
        self.register_buffer("mask", torch.tril(torch.ones(4, 4)), persistent=False)
        self.positional = torch.arange(4).float()

    def forward(self, x):
        return self.layer(x) + self.offset + self.positional


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.float64])
@pytest.mark.parametrize("legacy", [False, True])
def test_empty_loader_matches_original_weights_and_outputs(tmp_path, dtype, legacy):
    path = tmp_path / "model.pth"
    state = TinyModel().to(dtype=dtype).state_dict()
    torch.save({"model": state}, path, _use_new_zipfile_serialization=not legacy)
    baseline = TinyModel()
    load_checkpoint(baseline, path)
    loaded = load_inference_model(TinyModel, path)
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
        assert loaded.state_dict()[name].dtype == tensor.dtype
    assert torch.equal(baseline.mask, loaded.mask)
    assert torch.equal(baseline.positional, loaded.positional)
    x = torch.randn(2, 3)
    assert torch.equal(baseline(x), loaded(x))
    assert not any(t.is_meta for t in (*loaded.parameters(), *loaded.buffers()))


def test_missing_weight_fallback_matches_original_initialization(tmp_path):
    state = TinyModel().state_dict()
    del state["layer.bias"]
    path = tmp_path / "partial.pth"
    torch.save(state, path)
    torch.manual_seed(22)
    baseline = TinyModel()
    baseline.load_state_dict(state, strict=False)
    torch.manual_seed(22)
    loaded = load_inference_model(TinyModel, path)
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])


def test_wrong_checkpoint_shape_is_not_silently_accepted(tmp_path):
    state = TinyModel().state_dict()
    state["layer.weight"] = torch.zeros(1, 1)
    path = tmp_path / "bad.pth"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_inference_model(TinyModel, path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_accel_attaches_identical_transformer_parameters_and_real_buffers(dtype):
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=16)
    source = GPT2Model(cfg).to(dtype=dtype).eval()
    del source.wte, source.wpe
    baseline = GPT2AccelModel(cfg)
    baseline.load_state_dict(source.state_dict(), strict=False)
    baseline = baseline.to(dtype=dtype).eval()
    loaded = GPT2AccelModel.from_standard_model(cfg, source, device="cpu", dtype=dtype)
    for name, tensor in source.state_dict().items():
        actual = loaded.state_dict()[name]
        assert torch.equal(baseline.state_dict()[name], actual)
        assert tensor.data_ptr() == actual.data_ptr()
    for name, tensor in baseline.named_buffers():
        assert torch.equal(tensor, dict(loaded.named_buffers())[name])
    assert not any(t.is_meta for t in (*loaded.parameters(), *loaded.buffers()))


def test_accel_rejects_missing_transformer_weights():
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=16, n_positions=32, vocab_size=16)
    source = GPT2Model(cfg)
    del source.h[0].mlp.c_proj.bias
    with pytest.raises(ValueError, match="Incomplete acceleration weights"):
        GPT2AccelModel.from_standard_model(cfg, source, device="cpu", dtype=torch.float32)


class WarmupEngine:
    device = "cpu"

    def __init__(self, failure=False):
        self.gpt_path = "main"
        self.cache_spk_cond = object()
        self.cache_spk_audio_prompt = "previous.wav"
        self.calls = []
        self.failure = failure

    def set_gpt_checkpoint(self, path, load_if_missing=False):
        self.gpt_path = path

    def infer(self, **kwargs):
        self.calls.append((self.gpt_path, kwargs))
        self.cache_spk_cond = object()
        self.cache_spk_audio_prompt = kwargs["spk_audio_prompt"]
        random.random()
        np.random.rand()
        torch.rand(3)
        if self.failure:
            raise ValueError("warmup failed")
        return 22050, np.zeros(3, dtype=np.int16)


@pytest.mark.parametrize("failure", [False, True])
def test_warmup_restores_active_model_and_reference_caches(failure):
    engine = WarmupEngine(failure)
    original = engine.cache_spk_cond
    if failure:
        with pytest.raises(ValueError, match="warmup failed"):
            warmup_engine(engine, ["other", "main"], "warmup.wav")
    else:
        warmup_engine(engine, ["other", "main"], "warmup.wav")
        assert [path for path, kwargs in engine.calls] == ["other", "main"]
    assert engine.gpt_path == "main"
    assert engine.cache_spk_cond is original
    assert engine.cache_spk_audio_prompt == "previous.wav"
    for _, kwargs in engine.calls:
        assert kwargs["output_path"] is None
        assert kwargs["num_beams"] == 3
