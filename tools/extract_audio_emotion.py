"""Extract IndexTTS-2.5 emotion conditioning from a reference recording.

Emotion-reference inference uses a dense 1280-dimensional conditioning vector,
not the eight-value vector used by QwenEmotion and the WebUI sliders. This tool
extracts that exact dense vector and also projects it onto the eight emotion
prototype axes to provide an approximate, human-readable slider vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from scipy.optimize import nnls
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus


EMOTION_LABELS = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
SLIDER_BIAS = np.asarray(
    [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625],
    dtype=np.float64,
)


class AudioEmotionEncoder(nn.Module):
    """The subset of UnifiedVoice used by emotion-reference conditioning."""

    def __init__(self, model_dim: int, config: dict[str, Any]):
        super().__init__()
        self.conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=config["output_size"],
            linear_units=config["linear_units"],
            attention_heads=config["attention_heads"],
            num_blocks=config["num_blocks"],
            input_layer=config["input_layer"],
        )
        self.perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=config["output_size"],
            ff_mult=config["perceiver_mult"],
            heads=config["attention_heads"],
            num_latents=1,
        )
        self.emovec_layer = nn.Linear(1024, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.mask_pad = nn.ConstantPad1d((1, 0), True)

    def forward(
        self, semantic_features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        encoded, mask = self.conditioning_encoder(semantic_features, lengths)
        conditioned_mask = self.mask_pad(mask.squeeze(1))
        embedding = self.perceiver_encoder(encoded, conditioned_mask).squeeze(1)
        return self.emo_layer(self.emovec_layer(embedding))


def _load_checkpoint_payload(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint does not contain a state dictionary: {path}")
    return {
        key.removeprefix("module."): value
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, torch.Tensor)
    }


def _load_emotion_encoder(
    checkpoint: Path, model_dim: int, config: dict[str, Any]
) -> AudioEmotionEncoder:
    state = _load_checkpoint_payload(checkpoint)
    encoder = AudioEmotionEncoder(model_dim, config)
    mappings = {
        "conditioning_encoder.": "emo_conditioning_encoder.",
        "perceiver_encoder.": "emo_perceiver_encoder.",
        "emovec_layer.": "emovec_layer.",
        "emo_layer.": "emo_layer.",
    }
    selected = {}
    for destination_prefix, checkpoint_prefix in mappings.items():
        for key, value in state.items():
            if key.startswith(checkpoint_prefix):
                selected[destination_prefix + key[len(checkpoint_prefix) :]] = value
    missing, unexpected = encoder.load_state_dict(selected, strict=False)
    meaningful_missing = [key for key in missing if not key.startswith("mask_pad.")]
    if meaningful_missing or unexpected:
        raise ValueError(
            "Checkpoint is missing required emotion-conditioning weights: "
            f"missing={meaningful_missing[:5]}, unexpected={unexpected[:5]}"
        )
    return encoder


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_audio(path: Path, max_seconds: float) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
    if max_seconds > 0:
        audio = audio[:, : int(max_seconds * 16000)]
    if audio.numel() == 0:
        raise ValueError(f"Audio is empty: {path}")
    return audio.float()


def _selected_prototypes(
    style: torch.Tensor,
    emotion_groups: tuple[torch.Tensor, ...],
    speaker_groups: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, list[int], list[float]]:
    prototypes = []
    indices = []
    speaker_similarities = []
    for emotion_group, speaker_group in zip(emotion_groups, speaker_groups):
        similarities = F.cosine_similarity(
            style.float(), speaker_group.float(), dim=1
        )
        index = int(similarities.argmax().item())
        prototypes.append(emotion_group[index])
        indices.append(index)
        speaker_similarities.append(float(similarities[index].item()))
    return torch.stack(prototypes), indices, speaker_similarities


def _estimate_slider_vector(
    embedding: torch.Tensor, prototypes: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return approximate effective weights, UI values, fit cosine and residual."""
    target = embedding.detach().float().cpu().numpy().reshape(-1).astype(np.float64)
    basis = prototypes.detach().float().cpu().numpy().T.astype(np.float64)
    effective, residual = nnls(basis, target)
    total = float(effective.sum())
    if total > 0.8:
        effective *= 0.8 / total
    reconstructed = basis @ effective
    denominator = np.linalg.norm(target) * np.linalg.norm(reconstructed)
    fit_cosine = (
        float(np.dot(target, reconstructed) / denominator)
        if denominator > 0
        else 0.0
    )
    ui_values = np.divide(
        effective,
        SLIDER_BIAS,
        out=np.zeros_like(effective),
        where=SLIDER_BIAS != 0,
    )
    ui_values = np.clip(ui_values, 0.0, 1.0)
    return effective, ui_values, fit_cosine, float(residual)


def _rounded_mapping(values: np.ndarray | list[float]) -> dict[str, float]:
    return {
        label: round(float(value), 6)
        for label, value in zip(EMOTION_LABELS, values)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the exact IndexTTS-2.5 audio emotion embedding and an "
            "approximate eight-axis WebUI slider vector."
        )
    )
    parser.add_argument("audio", type=Path, help="Emotion-reference WAV/audio file")
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--max-seconds", type=float, default=15.0)
    parser.add_argument("--json", type=Path, default=None, help="Also save the report as JSON")
    parser.add_argument(
        "--embedding",
        type=Path,
        default=None,
        help="Save the exact 1280-D emotion-reference tensor as a .pt file",
    )
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Include all dense embedding values in the JSON/text report",
    )
    args = parser.parse_args()

    audio_path = args.audio.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else model_dir / "config.yaml"
    )
    if not audio_path.is_file():
        parser.error(f"Audio file does not exist: {audio_path}")
    if not config_path.is_file():
        parser.error(f"Config file does not exist: {config_path}")

    cfg = OmegaConf.load(config_path)
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else model_dir / str(cfg.gpt_checkpoint)
    )
    if not checkpoint.is_file():
        parser.error(f"GPT checkpoint does not exist: {checkpoint}")

    device = _resolve_device(args.device)
    print(f">> Loading audio emotion encoder on {device}...", flush=True)
    encoder = _load_emotion_encoder(
        checkpoint,
        int(cfg.gpt.model_dim),
        OmegaConf.to_container(cfg.gpt.emo_condition_module, resolve=True),
    ).to(device).eval()
    print(">> Audio emotion encoder loaded", flush=True)

    w2v_dir = model_dir / "hf_cache" / "w2v-bert-2.0"
    if not w2v_dir.is_dir():
        parser.error(f"Missing local Wav2Vec2-BERT model: {w2v_dir}")
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        w2v_dir, local_files_only=True
    )
    semantic_model = Wav2Vec2BertModel.from_pretrained(
        w2v_dir, local_files_only=True
    ).to(device).eval()
    stats = torch.load(model_dir / str(cfg.w2v_stat), map_location=device)
    semantic_mean = stats["mean"].to(device)
    semantic_std = torch.sqrt(stats["var"]).to(device)
    print(">> Wav2Vec2-BERT loaded", flush=True)

    campplus_path = model_dir / "hf_cache" / "campplus_cn_common.bin"
    if not campplus_path.is_file():
        parser.error(f"Missing local CAMPPlus model: {campplus_path}")
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
    campplus = campplus.to(device).eval()
    emotion_matrix = torch.load(
        model_dir / str(cfg.emo_matrix), map_location=device
    )
    speaker_matrix = torch.load(
        model_dir / str(cfg.spk_matrix), map_location=device
    )
    group_sizes = [int(value) for value in cfg.emo_num]
    emotion_groups = torch.split(emotion_matrix, group_sizes)
    speaker_groups = torch.split(speaker_matrix, group_sizes)
    print(">> CAMPPlus and emotion prototypes loaded", flush=True)

    audio = _load_audio(audio_path, args.max_seconds)
    feature_inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    input_features = feature_inputs["input_features"].to(device)
    attention_mask = feature_inputs["attention_mask"].to(device)
    print(">> Extracting emotion conditioning...", flush=True)

    with torch.inference_mode():
        semantic_output = semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        semantic_features = semantic_output.hidden_states[17]
        semantic_features = (semantic_features - semantic_mean) / semantic_std
        lengths = torch.tensor(
            [semantic_features.shape[1]], device=device, dtype=torch.long
        )
        emotion_embedding = encoder(semantic_features, lengths)

        fbank = torchaudio.compliance.kaldi.fbank(
            audio.to(device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        style = campplus(fbank.unsqueeze(0))
        prototypes, prototype_indices, speaker_similarities = _selected_prototypes(
            style, emotion_groups, speaker_groups
        )
        prototype_cosines = F.cosine_similarity(
            emotion_embedding.float(), prototypes.float(), dim=1
        ).detach().cpu().tolist()

    effective, ui_values, fit_cosine, residual = _estimate_slider_vector(
        emotion_embedding, prototypes
    )
    report: dict[str, Any] = {
        "audio": str(audio_path),
        "checkpoint": str(checkpoint),
        "exact_emotion_embedding": {
            "dimensions": int(emotion_embedding.shape[-1]),
            "l2_norm": round(float(emotion_embedding.float().norm().item()), 6),
        },
        "prototype_cosine_similarity": _rounded_mapping(prototype_cosines),
        "approximate_effective_vector": _rounded_mapping(effective),
        "approximate_webui_slider_vector": _rounded_mapping(ui_values),
        "projection_fit": {
            "cosine_similarity": round(fit_cosine, 6),
            "nnls_residual_before_webui_sum_cap": round(residual, 6),
        },
        "selected_prototype": {
            label: {
                "index": int(index),
                "speaker_cosine_similarity": round(float(similarity), 6),
            }
            for label, index, similarity in zip(
                EMOTION_LABELS, prototype_indices, speaker_similarities
            )
        },
        "note": (
            "Emotion-reference audio uses the exact dense embedding above. The "
            "eight-axis values are a prototype projection, not an exact inverse "
            "emotion classifier."
        ),
    }
    if args.include_embedding:
        report["exact_emotion_embedding"]["values"] = (
            emotion_embedding.detach().float().cpu().flatten().tolist()
        )
    if args.embedding is not None:
        embedding_path = args.embedding.expanduser().resolve()
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emotion_embedding.detach().float().cpu(), embedding_path)
        report["exact_emotion_embedding"]["saved_to"] = str(embedding_path)

    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.json is not None:
        json_path = args.json.expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(serialized + "\n", encoding="utf-8")
        print(f">> JSON report saved to: {json_path}")


if __name__ == "__main__":
    main()
