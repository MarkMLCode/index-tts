from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import api


class FakeEngine:
    def __init__(self):
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
        self.loaded_gpt_checkpoints = ("base.pth", path)
        return path

    def unload_gpt_checkpoint(self, path):
        self.loaded_gpt_checkpoints = tuple(
            loaded for loaded in self.loaded_gpt_checkpoints if loaded != path
        )
        return path


@pytest.fixture(autouse=True)
def clear_engine_state():
    with api.MODEL_LOCK:
        api.CURRENT_ENGINE = None
        api.CURRENT_MODEL_SPEC = None
        api.CURRENT_GPT_PATH = None
    yield
    with api.MODEL_LOCK:
        api.CURRENT_ENGINE = None
        api.CURRENT_MODEL_SPEC = None
        api.CURRENT_GPT_PATH = None


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


@pytest.mark.parametrize(
    "vector, message",
    [
        ([0.0] * 7, "8 floats"),
        ([-0.1] + [0.0] * 7, "cannot be negative"),
        ([0.2] * 8, "cannot exceed 0.8"),
    ],
)
def test_emotion_vector_validation(vector, message):
    with pytest.raises(ValueError, match=message):
        api.validate_emo_vector(vector)


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


def test_load_model_reuses_runtime_and_sets_the_exact_resident_set(tmp_path):
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
