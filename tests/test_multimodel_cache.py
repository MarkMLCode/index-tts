from __future__ import annotations

import torch
from torch import nn

from indextts.accel.accel_engine import AccelInferenceEngine
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.infer_v2_5 import IndexTTS2


class FakeAccelEngine:
    def __init__(self):
        self.reset_count = 0

    def reset_model_state(self):
        self.reset_count += 1


class FakeGPT:
    def __init__(self, accelerated=False):
        self.accel_engine = FakeAccelEngine() if accelerated else None


def test_gpt_startup_timers_preserve_loaded_weights(tmp_path, monkeypatch, caplog):
    import importlib
    from types import SimpleNamespace

    inference = importlib.import_module("indextts.infer_v2_5")

    class TinyGPT(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.layer = nn.Linear(2, 2)
            self.accel_engine = None

        def post_init_gpt2_config(self, **kwargs):
            self.inference_options = kwargs

    source = TinyGPT()
    checkpoint = tmp_path / "voice.pth"
    torch.save({"model": source.state_dict()}, checkpoint)
    monkeypatch.setattr(inference, "UnifiedVoice", TinyGPT)
    runtime = object.__new__(IndexTTS2)
    runtime.cfg = SimpleNamespace(gpt={})
    runtime.device = "cpu"
    runtime.dtype = None
    runtime.use_accel = False
    runtime.use_deepspeed = False
    runtime.use_bf16 = False
    runtime._shared_accel_kv_manager = None
    runtime._gpt_models = {}
    caplog.set_level("INFO", logger="indextts.startup")

    model = runtime._build_gpt_model(str(checkpoint))

    assert not model.training
    assert model.inference_options["accel_kv_manager"] is None
    for name, tensor in source.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name])
    for stage in (
        "GPT module construction", "GPT checkpoint read/deserialization (CPU)",
        "GPT load_state_dict (CPU)", "GPT device transfer and precision setup (cpu)",
        "GPT inference/acceleration engine setup", "Compare/share bit-identical GPT modules",
        f"GPT model total: {checkpoint}",
    ):
        assert f"START {stage}" in caplog.text
        assert f"DONE {stage} | elapsed=" in caplog.text


def test_resident_cache_switches_by_reference_and_can_unload(tmp_path):
    base_path = tmp_path / "base.pth"
    voice_path = tmp_path / "voice.pth"
    base_path.touch()
    voice_path.touch()
    base_path = str(base_path.resolve())
    voice_path = str(voice_path.resolve())

    runtime = object.__new__(IndexTTS2)
    base = FakeGPT(accelerated=True)
    voice = FakeGPT(accelerated=True)
    runtime.gpt = base
    runtime.gpt_path = base_path
    runtime._gpt_models = {base_path: base}
    runtime.use_deepspeed = False
    runtime._build_gpt_model = lambda path: voice

    assert runtime.preload_gpt_checkpoint(voice_path) == voice_path
    assert runtime.loaded_gpt_checkpoints == (base_path, voice_path)
    assert runtime.set_gpt_checkpoint(voice_path) == voice_path
    assert runtime.gpt is voice
    assert voice.accel_engine.reset_count == 1

    assert runtime.unload_gpt_checkpoint(base_path) == base_path
    assert runtime.loaded_gpt_checkpoints == (voice_path,)


def test_accel_model_releases_only_the_redundant_transformer():
    model = object.__new__(UnifiedVoice)
    nn.Module.__init__(model)
    standard_transformer = nn.Linear(4, 4)
    inference_model = nn.Module()
    inference_model.transformer = standard_transformer
    model.gpt = standard_transformer
    model.inference_model = inference_model
    model.accel_engine = object()

    assert model.release_standard_transformer_for_accel() is True
    assert model.gpt is None
    assert model.inference_model.transformer is None


def test_shared_accel_release_does_not_change_checkpoint_specific_heads():
    model = object.__new__(UnifiedVoice)
    nn.Module.__init__(model)
    model.gpt = nn.Linear(4, 4)
    model.final_norm = nn.LayerNorm(4)
    model.mel_head = nn.Linear(4, 8)
    inference_model = nn.Module()
    inference_model.transformer = model.gpt
    model.inference_model = inference_model
    model.accel_engine = object()
    norm_weight = model.final_norm.weight
    head_weight = model.mel_head.weight

    model.release_standard_transformer_for_accel()

    assert model.final_norm.weight is norm_weight
    assert model.mel_head.weight is head_weight
    assert torch.equal(model.final_norm.weight, norm_weight)


def test_accel_engines_can_reuse_one_kv_workspace():
    class SharedKVManager:
        num_layers = 1
        num_heads = 1
        head_dim = 4
        block_size = 2
        num_blocks = 2
        dtype = torch.float32

        def __init__(self):
            self.wired_models = []

        def wire_kv_cache_to_model(self, model):
            self.wired_models.append(model)

    model = nn.Linear(4, 4)
    model.config = type("Config", (), {"hidden_size": 4})()
    shared = SharedKVManager()

    engine = AccelInferenceEngine(
        model=model,
        lm_head=nn.Identity(),
        num_layers=1,
        num_heads=1,
        head_dim=4,
        block_size=2,
        num_blocks=2,
        use_cuda_graph=False,
        kv_manager=shared,
    )

    assert engine.kv_manager is shared
    assert shared.wired_models == [model]


def test_only_bit_identical_conditioning_modules_are_shared():
    runtime = object.__new__(IndexTTS2)
    reference = nn.Module()
    reference.spk_emb_proj = nn.Linear(2, 2)
    reference.emo_layer = nn.Linear(2, 2)
    candidate = nn.Module()
    candidate.spk_emb_proj = nn.Linear(2, 2)
    candidate.emo_layer = nn.Linear(2, 2)
    candidate.spk_emb_proj.load_state_dict(reference.spk_emb_proj.state_dict())
    candidate.emo_layer.load_state_dict(reference.emo_layer.state_dict())
    candidate.emo_layer.weight.data.add_(1.0)
    runtime._gpt_models = {"base.pth": reference}

    shared = runtime._share_identical_gpt_modules(candidate)

    assert "spk_emb_proj" in shared
    assert "emo_layer" not in shared
    assert candidate.spk_emb_proj is reference.spk_emb_proj
    assert candidate.emo_layer is not reference.emo_layer


def test_single_resident_model_is_replaced_in_place(tmp_path):
    old_path = tmp_path / "old.pth"
    new_path = tmp_path / "new.pth"
    old_path.touch()
    old_path = str(old_path.resolve())
    source = nn.Module()
    source.layer = nn.Linear(2, 2)
    torch.save({"model": source.state_dict()}, new_path)

    runtime = object.__new__(IndexTTS2)
    resident = nn.Module()
    resident.layer = nn.Linear(2, 2)
    resident.accel_engine = None
    runtime.gpt = resident
    runtime.gpt_path = old_path
    runtime._gpt_models = {old_path: resident}
    runtime.use_deepspeed = False

    selected = runtime.replace_gpt_checkpoint(new_path)

    assert selected == str(new_path.resolve())
    assert runtime.gpt is resident
    assert runtime.loaded_gpt_checkpoints == (selected,)
    assert torch.equal(resident.layer.weight, source.layer.weight)
    assert torch.equal(resident.layer.bias, source.layer.bias)
