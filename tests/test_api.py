from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import api


class FakeEngine:
    def __init__(self):
        self.calls = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        return 22050, np.array([[1], [-2], [3]], dtype=np.int16)


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


def test_generate_schema_rejects_per_request_model_switching():
    with pytest.raises(ValidationError, match="gpt_checkpoint"):
        api.GenerateRequest(
            speaker="speaker.wav",
            text="Hello",
            gpt_checkpoint="another-model.pth",
        )
