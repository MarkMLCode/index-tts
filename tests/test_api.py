from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

import api


class FakeEngine:
    def __init__(self):
        self.device = "cpu"
        self.calls = []
        self.gpt_path = "base.pth"
        self.loaded_gpt_checkpoints = ("base.pth",)
        self.preloaded = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return 22050, np.array([[1], [-2], [3]], dtype=np.int16)

    def preload_gpt_checkpoints(self, paths):
        self.preloaded.extend(paths)
        self.loaded_gpt_checkpoints = tuple(
            dict.fromkeys((*self.loaded_gpt_checkpoints, *paths))
        )

    def set_gpt_checkpoint(self, path, load_if_missing=False):
        assert load_if_missing is False
        self.gpt_path = path
        self.loaded_gpt_checkpoints = tuple(dict.fromkeys((*self.loaded_gpt_checkpoints, path)))
        return path

    def unload_gpt_checkpoint(self, path):
        self.loaded_gpt_checkpoints = tuple(
            loaded for loaded in self.loaded_gpt_checkpoints if loaded != path
        )
        return path


class RandomEngine(FakeEngine):
    fail_next = False

    def infer(self, **kwargs):
        super().infer(**kwargs)
        audio = np.array([
            random.randrange(32768),
            np.random.randint(32768),
            torch.randint(32768, ()).item(),
        ], dtype=np.int16)
        if self.fail_next:
            self.fail_next = False
            raise ValueError("generation failed")
        return 22050, audio


@pytest.fixture(autouse=True)
def clear_engine_state():
    with api.MODEL_LOCK:
        api.CURRENT_ENGINE = None
        api.CURRENT_MODEL_SPEC = None
        api.CURRENT_GPT_PATH = None
        api.CURRENT_WARMED_GPT_PATHS.clear()
    yield
    with api.MODEL_LOCK:
        api.CURRENT_ENGINE = None
        api.CURRENT_MODEL_SPEC = None
        api.CURRENT_GPT_PATH = None
        api.CURRENT_WARMED_GPT_PATHS.clear()


def test_generate_uses_the_single_loaded_engine():
    engine = FakeEngine()
    api.CURRENT_ENGINE = engine

    sampling_rate, audio = api.generate_audio(
        gpt_checkpoint="base.pth",
        speaker="speaker.wav",
        text="Hello",
        lang="EN",
        duration_factor=1.2,
        top_k=17,
    )

    assert sampling_rate == 22050
    assert audio.tolist() == [[1], [-2], [3]]
    assert engine.calls[0]["spk_audio_prompt"] == "speaker.wav"
    assert engine.calls[0]["lang"] == "EN"
    assert engine.calls[0]["duration_factor"] == 1.2
    assert engine.calls[0]["top_k"] == 17
    assert "gpt_checkpoint" not in engine.calls[0]
    assert api.CURRENT_GPT_PATH == str((api.Path.cwd() / "base.pth").resolve())


def test_generate_requires_a_loaded_model():
    with pytest.raises(RuntimeError, match="call /load_model first"):
        api.generate_audio(speaker="speaker.wav", text="Hello")


@pytest.mark.parametrize("interruption", ["generation", "warmup", "failed_warmup"])
def test_request_seed_is_independent_of_prior_random_consumption(interruption):
    from indextts.utils.warmup import warmup_engine

    engine = RandomEngine()
    api.CURRENT_ENGINE = engine
    kwargs = dict(gpt_checkpoint="base.pth", speaker="speaker.wav", text="Hello", seed=42)
    _, expected = api.generate_audio(**kwargs)
    if interruption == "generation":
        api.generate_audio(**{**kwargs, "seed": 999, "gpt_checkpoint": "other.pth"})
    else:
        with api.MODEL_LOCK:
            engine.fail_next = interruption == "failed_warmup"
            if engine.fail_next:
                with pytest.raises(ValueError, match="generation failed"):
                    warmup_engine(engine, ["other.pth", "base.pth"], "warmup.wav")
            else:
                warmup_engine(engine, ["other.pth", "base.pth"], "warmup.wav")
    _, actual = api.generate_audio(**kwargs)
    assert np.array_equal(actual, expected)
    assert "seed" not in engine.calls[-1]


def test_seed_is_applied_after_model_selection_under_the_lock(monkeypatch):
    events = []

    class TrackingLock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *args):
            self.held = False

    class TrackingEngine(FakeEngine):
        def set_gpt_checkpoint(self, path, load_if_missing=False):
            events.append("switch")
            return super().set_gpt_checkpoint(path, load_if_missing)

        def infer(self, **kwargs):
            assert api.MODEL_LOCK.held
            events.append("infer")
            return super().infer(**kwargs)

    def apply_seed(seed):
        assert seed == 42 and api.MODEL_LOCK.held
        events.append("seed")

    api.CURRENT_ENGINE = TrackingEngine()
    monkeypatch.setattr(api, "MODEL_LOCK", TrackingLock())
    monkeypatch.setattr(api, "apply_seed", apply_seed)
    api.generate_audio(gpt_checkpoint="base.pth", speaker="speaker.wav", text="Hello", seed=42)
    assert events == ["switch", "seed", "infer"]


@pytest.mark.parametrize("endpoint", ["/generate", "/tts"])
def test_api_reports_random_seed_and_honors_replay(monkeypatch, caplog, endpoint):
    caplog.set_level("INFO")
    api.CURRENT_ENGINE = RandomEngine()
    seeds = iter([2**60 + 123, 2**60 + 456])
    monkeypatch.setattr(random.SystemRandom, "randrange", lambda *args: next(seeds))

    def generate(client, seed=None):
        values = {} if seed is None else {"seed": seed}
        if endpoint == "/generate":
            return client.post(endpoint, json=dict(
                gpt_checkpoint="base.pth", speaker="speaker.wav", text="Hello", **values,
            ))
        return client.get(endpoint, params=dict(ref_audio_path="speaker.wav", text="Hello", **values))

    # No lifespan context: do not start real logging or load a TTS model.
    client = TestClient(api.APP)
    first = generate(client)
    assert first.status_code == 200
    assert first.headers["x-seed"] == str(2**60 + 123)
    assert "Generation seed: " + str(2**60 + 123) in caplog.text
    generate(client, seed=999)
    replay = generate(client, seed=int(first.headers["x-seed"]))
    assert replay.status_code == 200
    assert replay.headers["x-seed"] == first.headers["x-seed"]
    assert replay.content == first.content
    assert generate(client, seed=-1).headers["x-seed"] == str(2**60 + 456)


@pytest.mark.parametrize(
    "vector, message",
    [
        ([0.0] * 7, "8 floats"),
        ([-0.1] + [0.0] * 7, "cannot be negative"),
        ([0.2] * 8, "cannot exceed 1.5"),
        ([0.1875001] * 8, "cannot exceed 1.5"),
    ],
)
def test_emotion_vector_validation(vector, message):
    with pytest.raises(ValueError, match=message):
        api.validate_emo_vector(vector)


@pytest.mark.parametrize("vector", [[0.125] * 8, [0.1875] * 8])
def test_emotion_vector_accepts_totals_up_to_1_5(vector):
    assert api.validate_emo_vector(vector) == vector


@pytest.mark.parametrize("action,endpoint,model_request", [
    ("load_model", api.load_model_endpoint, api.LoadModelRequest(gpt_checkpoint_paths=["voice.pth"])),
    ("switch_model", api.switch_model_endpoint, api.GPTCheckpointRequest(gpt_checkpoint="voice.pth")),
    ("unload_model", api.unload_model_endpoint, api.GPTCheckpointRequest(gpt_checkpoint="voice.pth")),
])
def test_model_endpoints_preserve_responses_and_errors(monkeypatch, action, endpoint, model_request):
    monkeypatch.setattr(api, action, lambda **kwargs: "voice.pth")
    assert endpoint(model_request) == {"message": "success", "gpt_checkpoint": "voice.pth"}
    def fail(**kwargs):
        raise ValueError("checkpoint unavailable")
    monkeypatch.setattr(api, action, fail)
    with pytest.raises(api.HTTPException) as error:
        endpoint(model_request)
    assert error.value.status_code == 400
    assert error.value.detail == "checkpoint unavailable"


def test_pack_wav_returns_a_wav_container():
    data = np.array([[1], [-2], [3]], dtype=np.int16)
    assert api.pack_audio(data, 22050, "wav").startswith(b"RIFF")


def test_generate_schema_requires_per_request_model_selection():
    with pytest.raises(ValidationError, match="gpt_checkpoint"):
        api.GenerateRequest(
            speaker="speaker.wav",
            text="Hello",
        )
    request = api.GenerateRequest(
        speaker="speaker.wav",
        text="Hello",
        gpt_checkpoint="another-model.pth",
    )
    assert request.gpt_checkpoint == "another-model.pth"
    assert request.seed == -1


def test_generate_schema_rejects_fractional_seed():
    with pytest.raises(ValidationError, match="seed"):
        api.GenerateRequest(gpt_checkpoint="base.pth", speaker="speaker.wav", text="Hello", seed=1.5)


def test_switch_model_uses_the_resident_fast_path(tmp_path):
    checkpoint = tmp_path / "voice.pth"
    checkpoint.touch()
    engine = FakeEngine()
    api.CURRENT_ENGINE = engine

    selected = api.switch_model(str(checkpoint))

    assert selected == str(checkpoint.resolve())
    assert api.CURRENT_GPT_PATH == selected
    assert engine.preloaded == []
    assert api.model_status() == {
        "active_gpt_checkpoint": selected,
        "loaded_gpt_checkpoints": ["base.pth", selected],
    }


def test_unload_model_releases_an_inactive_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "voice.pth"
    checkpoint.touch()
    engine = FakeEngine()
    engine.loaded_gpt_checkpoints = ("base.pth", str(checkpoint.resolve()))
    api.CURRENT_ENGINE = engine
    monkeypatch.setattr(api.gc, "collect", lambda: 0)

    assert api.unload_model(str(checkpoint)) == str(checkpoint.resolve())
    assert engine.loaded_gpt_checkpoints == ("base.pth",)


def test_load_model_reuses_runtime_and_sets_the_exact_resident_set(tmp_path, caplog):
    caplog.set_level("INFO")
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "voice.pth"
    config.touch()
    checkpoint.touch()
    engine = FakeEngine()
    api.CURRENT_ENGINE = engine
    api.CURRENT_MODEL_SPEC = (
        str(config.resolve()),
        str(tmp_path.resolve()),
        None,
        False,
        None,
        False,
        False,
        False,
        False,
    )

    selected = api.load_model(
        config=str(config),
        model_dir=str(tmp_path),
        gpt_checkpoint_paths=[str(checkpoint)],
        main_gpt_checkpoint=str(checkpoint),
    )

    assert engine.preloaded == [str(checkpoint.resolve())]
    assert selected == str(checkpoint.resolve())
    assert api.CURRENT_ENGINE is engine
    assert engine.loaded_gpt_checkpoints == (str(checkpoint.resolve()),)
    assert "Update resident GPT set (reuse shared speech stack)" in caplog.text
    assert "Models ready: resident=1" in caplog.text
    assert "DONE load_model total (including validation and lock wait) | elapsed=" in caplog.text


def test_multiple_models_require_a_main_checkpoint(tmp_path):
    config = tmp_path / "config.yaml"
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    config.touch()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="main_gpt_checkpoint is required"):
        api.load_model(
            config=str(config),
            model_dir=str(tmp_path),
            gpt_checkpoint_paths=[str(first), str(second)],
        )


def test_main_checkpoint_must_be_in_resident_set(tmp_path):
    config = tmp_path / "config.yaml"
    first = tmp_path / "first.pth"
    other = tmp_path / "other.pth"
    config.touch()
    first.touch()
    other.touch()

    with pytest.raises(ValueError, match="must be included"):
        api.load_model(
            config=str(config),
            model_dir=str(tmp_path),
            gpt_checkpoint_paths=[str(first)],
            main_gpt_checkpoint=str(other),
        )


def test_new_runtime_logs_startup_stages_without_warmup(tmp_path, monkeypatch, caplog):
    import sys
    from types import SimpleNamespace

    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "voice.pth"
    config.touch()
    checkpoint.touch()
    engine = FakeEngine()
    engine.gpt_path = str(checkpoint.resolve())
    engine.loaded_gpt_checkpoints = (engine.gpt_path,)
    monkeypatch.setattr(api, "_release_engine_locked", lambda: None)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", SimpleNamespace(IndexTTS2=lambda **kwargs: engine))
    caplog.set_level("INFO")

    assert api.load_model(
        config=str(config), model_dir=str(tmp_path),
        gpt_checkpoint_paths=[str(checkpoint)],
        warmup=False,
    ) == engine.gpt_path

    assert engine.calls == []
    assert "DONE Release previous engine and accelerator cache | elapsed=" in caplog.text
    assert "DONE Import inference dependencies | elapsed=" in caplog.text
    assert "Generation warmup disabled" in caplog.text
    assert "DONE load_model total (including validation and lock wait) | elapsed=" in caplog.text


def test_api_lifespan_initializes_run_logging(monkeypatch, caplog):
    import asyncio

    calls = []
    monkeypatch.setattr(api, "configure_run_logging", lambda: calls.append("configured"))
    caplog.set_level("INFO")

    async def start_app():
        async with api.lifespan(api.APP):
            assert calls == ["configured"]

    asyncio.run(start_app())
    assert "elapsed_since_api_import=" in caplog.text


def test_warmup_only_runs_for_unwarmed_resident_models():
    engine = FakeEngine()
    engine.loaded_gpt_checkpoints = ("base.pth", "voice.pth")
    api._warmup_loaded_models_locked(engine, "speaker.wav")
    assert len(engine.calls) == 2
    assert engine.gpt_path == "base.pth"
    assert api.CURRENT_WARMED_GPT_PATHS == {"base.pth", "voice.pth"}
    api._warmup_loaded_models_locked(engine, "speaker.wav")
    assert len(engine.calls) == 2
    engine.loaded_gpt_checkpoints = ("base.pth", "new.pth")
    api._warmup_loaded_models_locked(engine, "speaker.wav")
    assert len(engine.calls) == 3
    assert api.CURRENT_WARMED_GPT_PATHS == {"base.pth", "new.pth"}


def test_disabled_warmup_does_not_mark_models_as_warmed():
    engine = FakeEngine()
    api._warmup_loaded_models_locked(engine, None)
    assert not engine.calls
    assert not api.CURRENT_WARMED_GPT_PATHS


def test_explicit_missing_warmup_audio_fails_before_changing_engine(tmp_path):
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "voice.pth"
    config.touch()
    checkpoint.touch()
    with pytest.raises(FileNotFoundError, match="warmup speaker audio"):
        api.load_model(
            config=str(config), model_dir=str(tmp_path),
            gpt_checkpoint_paths=[str(checkpoint)],
            warmup_speaker=str(tmp_path / "missing.wav"),
        )
    assert api.CURRENT_ENGINE is None
