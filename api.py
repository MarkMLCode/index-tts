#!/usr/bin/env python3
"""FastAPI server for persistent, single-model IndexTTS 2.5 inference."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import subprocess
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict


LOGGER = logging.getLogger(__name__)
APP = FastAPI(title="IndexTTS 2.5 API")
MODEL_LOCK = threading.RLock()
SUPPORTED_MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}

CURRENT_ENGINE: Optional[Any] = None
CURRENT_MODEL_SPEC: Optional[Tuple[Any, ...]] = None
CURRENT_GPT_PATH: Optional[str] = None


def _resolve_existing_file(path: str, description: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return str(resolved)


def _resolve_existing_dir(path: str, description: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return str(resolved)


def _release_engine_locked() -> None:
    """Release the current engine. MODEL_LOCK must be held by the caller."""
    global CURRENT_ENGINE, CURRENT_MODEL_SPEC, CURRENT_GPT_PATH

    CURRENT_ENGINE = None
    CURRENT_MODEL_SPEC = None
    CURRENT_GPT_PATH = None
    gc.collect()

    # Importing torch is intentionally delayed so importing api.py stays light.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except (ImportError, RuntimeError):
        LOGGER.debug("Unable to empty the accelerator cache", exc_info=True)


def get_engine() -> Any:
    if CURRENT_ENGINE is None:
        raise RuntimeError("no model is currently loaded; call /load_model first")
    return CURRENT_ENGINE


def load_model(
    *,
    config: str = "checkpoints/config.yaml",
    model_dir: str = "checkpoints",
    gpt_checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    use_bf16: bool = False,
    use_cuda_kernel: Optional[bool] = None,
    use_deepspeed: bool = False,
    use_accel: bool = False,
    use_torch_compile: bool = False,
    use_qwen_emo: bool = False,
) -> str:
    """Load exactly one IndexTTS 2.5 model and return its GPT path."""
    config_path = _resolve_existing_file(config, "config file")
    model_dir_path = _resolve_existing_dir(model_dir, "model directory")
    gpt_path = (
        _resolve_existing_file(gpt_checkpoint, "GPT checkpoint")
        if gpt_checkpoint is not None
        else None
    )
    model_spec = (
        config_path,
        model_dir_path,
        gpt_path,
        device,
        use_bf16,
        use_cuda_kernel,
        use_deepspeed,
        use_accel,
        use_torch_compile,
        use_qwen_emo,
    )

    with MODEL_LOCK:
        global CURRENT_ENGINE, CURRENT_MODEL_SPEC, CURRENT_GPT_PATH

        if CURRENT_ENGINE is not None and CURRENT_MODEL_SPEC == model_spec:
            return cast(str, CURRENT_GPT_PATH)

        # The old engine must be released first because two full model instances
        # generally do not fit in GPU memory at the same time.
        _release_engine_locked()

        from indextts.infer_v2_5 import IndexTTS2

        engine = IndexTTS2(
            cfg_path=config_path,
            model_dir=model_dir_path,
            gpt_checkpoint_path=gpt_path,
            device=device,
            use_bf16=use_bf16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
            use_accel=use_accel,
            use_torch_compile=use_torch_compile,
            use_qwen_emo=use_qwen_emo,
        )
        CURRENT_ENGINE = engine
        CURRENT_MODEL_SPEC = model_spec
        CURRENT_GPT_PATH = str(Path(engine.gpt_path).resolve())
        return CURRENT_GPT_PATH


def validate_emo_vector(emo_vector: Optional[List[float]]) -> Optional[List[float]]:
    if emo_vector is None:
        return None
    if len(emo_vector) != 8:
        raise ValueError(f"emotion vector must contain 8 floats, received {len(emo_vector)}")
    values = [float(value) for value in emo_vector]
    if not all(np.isfinite(value) for value in values):
        raise ValueError("emotion vector elements must be finite")
    if any(value < 0 for value in values):
        raise ValueError("emotion vector elements cannot be negative")
    if sum(values) > 0.8 + 1e-9:
        raise ValueError("emotion vector sum cannot exceed 0.8")
    return values


def build_generation_kwargs(
    *,
    do_sample: Optional[bool],
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: Optional[float],
    temperature: Optional[float],
    length_penalty: Optional[float],
    num_beams: Optional[int],
    max_mel_tokens: Optional[int],
) -> Dict[str, Any]:
    values = {
        "do_sample": do_sample,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "length_penalty": length_penalty,
        "num_beams": num_beams,
        "max_mel_tokens": max_mel_tokens,
    }
    return {key: value for key, value in values.items() if value is not None}


def generate_audio(
    *,
    speaker: str,
    text: str,
    lang: str = "ZH",
    emo_audio_prompt: Optional[str] = None,
    emo_alpha: float = 1.0,
    emo_vector: Optional[List[float]] = None,
    use_emo_text: bool = False,
    emo_text: Optional[str] = None,
    use_random: bool = False,
    interval_silence: int = 200,
    duration_factor: float = 1.0,
    verbose: bool = False,
    max_text_tokens: int = 120,
    text_normalization: bool = True,
    do_sample: Optional[bool] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
    length_penalty: Optional[float] = None,
    num_beams: Optional[int] = None,
    max_mel_tokens: Optional[int] = None,
) -> Tuple[int, np.ndarray]:
    if not text or not text.strip():
        raise ValueError("text is required")
    if not speaker or not speaker.strip():
        raise ValueError("speaker is required")
    if not lang or not lang.strip():
        raise ValueError("lang is required")
    if not 0.5 <= duration_factor <= 2.0:
        raise ValueError("duration_factor must be between 0.5 and 2.0")
    if not 0.0 <= emo_alpha <= 1.0:
        raise ValueError("emo_alpha must be between 0.0 and 1.0")

    validated_vector = validate_emo_vector(emo_vector)
    generation_kwargs = build_generation_kwargs(
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        length_penalty=length_penalty,
        num_beams=num_beams,
        max_mel_tokens=max_mel_tokens,
    )

    with MODEL_LOCK:
        result = get_engine().infer(
            spk_audio_prompt=speaker,
            text=text,
            output_path=None,
            lang=lang,
            emo_audio_prompt=emo_audio_prompt,
            emo_alpha=emo_alpha,
            emo_vector=validated_vector,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            interval_silence=interval_silence,
            duration_factor=duration_factor,
            verbose=verbose,
            max_text_tokens_per_segment=max_text_tokens,
            text_normalization=text_normalization,
            **generation_kwargs,
        )

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("inference did not return audio")
    sampling_rate, audio_data = result
    return int(sampling_rate), np.asarray(audio_data, dtype=np.int16)


def pack_audio(data: np.ndarray, rate: int, media_type: str) -> bytes:
    channels = data.shape[1] if data.ndim > 1 else 1
    buffer = BytesIO()

    if media_type == "raw":
        buffer.write(data.tobytes())
    elif media_type == "wav":
        sf.write(buffer, data, rate, format="wav")
    elif media_type == "ogg":
        sf.write(buffer, data, rate, format="ogg")
    elif media_type == "aac":
        try:
            process = subprocess.run(
                [
                    "ffmpeg",
                    "-f",
                    "s16le",
                    "-ar",
                    str(rate),
                    "-ac",
                    str(channels),
                    "-i",
                    "pipe:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-vn",
                    "-f",
                    "adts",
                    "pipe:1",
                ],
                input=data.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for AAC output but was not found") from exc
        if process.returncode != 0:
            error = process.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"ffmpeg encoding failed: {error}")
        buffer.write(process.stdout)
    else:
        raise ValueError(f"media_type '{media_type}' is not supported")

    return buffer.getvalue()


class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadModelRequest(APIRequest):
    gpt_checkpoint: Optional[str] = None
    config: str = "checkpoints/config.yaml"
    model_dir: str = "checkpoints"
    device: Optional[str] = None
    use_bf16: bool = False
    use_cuda_kernel: Optional[bool] = None
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False
    use_qwen_emo: bool = False


class GenerateRequest(APIRequest):
    speaker: str
    text: str
    lang: str = "ZH"
    emo_audio_prompt: Optional[str] = None
    emo_alpha: float = 1.0
    emo_vector: Optional[List[float]] = None
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    use_random: bool = False
    interval_silence: int = 200
    duration_factor: float = 1.0
    verbose: bool = False
    max_text_tokens: int = 120
    text_normalization: bool = True
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    length_penalty: Optional[float] = None
    num_beams: Optional[int] = None
    max_mel_tokens: Optional[int] = None
    media_type: str = "wav"


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return cast(Dict[str, Any], model.model_dump())
    return cast(Dict[str, Any], model.dict())


@APP.post("/load_model")
def load_model_endpoint(request: LoadModelRequest):
    try:
        gpt_path = load_model(**_model_dump(request))
    except Exception as exc:
        LOGGER.exception("Model loading failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "success", "gpt_checkpoint": gpt_path}


@APP.post("/generate")
def generate_endpoint(request: GenerateRequest):
    values = _model_dump(request)
    media_type = str(values.pop("media_type")).lower()
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail=f"media_type '{media_type}' is not supported")

    try:
        sampling_rate, audio_array = generate_audio(**values)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(-1, 1)
        audio_bytes = pack_audio(audio_array, sampling_rate, media_type)
    except Exception as exc:
        LOGGER.exception("Audio generation failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=audio_bytes, media_type=f"audio/{media_type}")


@APP.get("/tts")
def tts_get_endpoint(
    text: Optional[str] = None,
    text_lang: Optional[str] = None,
    ref_audio_path: Optional[str] = None,
    aux_ref_audio_paths: Optional[list] = None,
    prompt_lang: Optional[str] = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut0",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    del (
        aux_ref_audio_paths,
        prompt_lang,
        prompt_text,
        text_split_method,
        batch_size,
        batch_threshold,
        split_bucket,
        fragment_interval,
        seed,
        streaming_mode,
        parallel_infer,
        sample_steps,
        super_sampling,
    )
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if not ref_audio_path:
        raise HTTPException(status_code=400, detail="ref_audio_path is required")
    if speed_factor <= 0:
        raise HTTPException(status_code=400, detail="speed_factor must be greater than zero")

    duration_factor = max(0.5, min(2.0, 1.0 / speed_factor))
    request = GenerateRequest(
        speaker=ref_audio_path,
        text=text,
        lang=text_lang or "ZH",
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        duration_factor=duration_factor,
        media_type=media_type,
    )
    return generate_endpoint(request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS 2.5 FastAPI server")
    parser.add_argument("-a", "--bind-addr", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=9880)
    args = parser.parse_args()

    uvicorn_log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    uvicorn.run(
        app=APP,
        host=cast(str, args.bind_addr),
        port=args.port,
        workers=1,
        log_config=uvicorn_log_config,
    )
