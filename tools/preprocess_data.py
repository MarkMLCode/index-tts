#!/usr/bin/env python3
"""Preprocess audio/text manifests for IndexTTS 2.5 GPT fine-tuning.

Input is JSONL with ``id``, ``text``, ``audio``, and preferably ``speaker`` and
``language`` fields. Output keeps the manifest contract used by the original
training fork, but cached features match the 2.5 inference path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from indextts.codec.models import EnhancedCodec
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.utils.front import TextNormalizer
from indextts.utils.ja_g2p import JapaneseG2PProcessor
from indextts.utils.nemo_tn import normalize_text as nemo_text_normalize
from indextts.utils.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer


SPEAKER_PATTERN = re.compile(r"^\s*(?:speaker|spk)\s*\d+\s*[:：]\s*", re.IGNORECASE)
PRONUNCIATION_PATTERN = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
LANGUAGE_ALIASES = {"cn": "zh", "zhen": "zh", "jp": "ja"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--gpt-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Deprecated compatibility option; 2.5 uses the tokenizer in --model-dir.",
    )
    parser.add_argument("--language", default="en", help="Fallback language.")
    parser.add_argument(
        "--audio-root",
        action="append",
        type=Path,
        default=[],
        help="Additional root used to resolve relative audio paths (repeatable).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="Reserved for CLI compatibility.")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=40.0,
        help="Reject longer utterances rather than training on truncated, misaligned audio.",
    )
    return parser.parse_args()


def canonical_language(value: str | None, fallback: str) -> str:
    lang = (value or fallback).strip().lower()
    lang = LANGUAGE_ALIASES.get(lang, lang)
    lang = TO_LANGUAGE_CODE.get(lang, lang)
    if lang not in LANGUAGES:
        raise ValueError(f"Unsupported IndexTTS 2.5 language: {value!r}")
    return lang


def clean_source_text(text: str) -> str:
    text = str(text or "").strip().replace("\u3000", " ").replace("\xa0", " ")
    return SPEAKER_PATTERN.sub("", text).strip()


def apply_pronunciation_annotations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        word, pronunciation = match.group(1), match.group(2).upper()
        if re.fullmatch(r"[\u3040-\u30ff]+", pronunciation):
            return f" {pronunciation} "
        token = "SPECIAL_TOKEN_2" if re.search(r"[\u4e00-\u9fff]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pronunciation}<|{token}|>"

    return PRONUNCIATION_PATTERN.sub(replace, text)


class TextProcessor:
    def __init__(self, model_dir: Path):
        self.tokenizer = get_tokenizer(multilingual=True, model_dir=str(model_dir))
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        glossary = model_dir / "glossary.yaml"
        if glossary.exists():
            self.normalizer.load_glossary_from_yaml(str(glossary))
        self.japanese = JapaneseG2PProcessor(g2p_ratio=0)

    def encode(self, raw_text: str, language: str) -> tuple[str, np.ndarray]:
        text = clean_source_text(raw_text)
        text = self.normalizer.clean_pattern.sub(
            lambda match: self.normalizer.char_rep_map[match.group()], text
        )
        if language in {"zh", "en"}:
            text = self.normalizer.normalize(text)
        elif language in {"ja", "es"}:
            text = nemo_text_normalize(text, language)
        if language in {"ja", "zh", "en"}:
            text = text.lower()
        elif language == "es":
            text = text.upper()
        text = apply_pronunciation_annotations(text)
        if language == "ja":
            text = self.japanese.process_ja_text(text)
        text = re.sub(r"<\|([^|]+)\|>", lambda match: f"<|{match.group(1).upper()}|>", text)
        ids = self.tokenizer.encode(f"<|{language}|> {text}", allowed_special="all")
        return text, np.asarray(ids, dtype=np.int32)


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.float(), sample_rate


def resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    return waveform if source_rate == target_rate else torchaudio.functional.resample(
        waveform, source_rate, target_rate
    )


def resolve_audio_path(value: str, roots: Iterable[Path]) -> Path | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for root in roots:
        candidate = (root / value).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {
        key.replace(".base_layer.", "."): value
        for key, value in state.items()
        if not key.startswith("inference_model.") and ".lora_" not in key and key != "gpt.wte.weight"
    }


class FeaturePipeline:
    def __init__(self, cfg: Any, model_dir: Path, gpt_checkpoint: Path, device: torch.device):
        self.device = device
        w2v_dir = model_dir / "hf_cache" / "w2v-bert-2.0"
        campplus_path = model_dir / "hf_cache" / "campplus_cn_common.bin"
        if not w2v_dir.is_dir() or not campplus_path.is_file():
            from indextts.utils.model_download import ensure_models_available

            paths = ensure_models_available(str(model_dir))
            w2v_dir = Path(paths["w2v_bert"])
            campplus_path = Path(paths["campplus"])

        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            str(w2v_dir), local_files_only=True
        )
        self.semantic_model = Wav2Vec2BertModel.from_pretrained(
            str(w2v_dir), local_files_only=True
        ).to(device).eval()
        stats = torch.load(model_dir / cfg.w2v_stat, map_location="cpu")
        self.semantic_mean = stats["mean"].to(device)
        self.semantic_std = torch.sqrt(stats["var"]).to(device)

        self.codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
        self.codec.load_checkpoint(str(model_dir / "codec.pth"))
        self.codec = self.codec.to(device).eval()

        self.gpt = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
        missing, unexpected = self.gpt.load_state_dict(load_state_dict(gpt_checkpoint), strict=False)
        real_missing = [key for key in missing if key != "gpt.wte.weight"]
        if real_missing or unexpected:
            raise RuntimeError(
                "Base GPT is incompatible with the 2.5 graph: "
                f"missing={real_missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.gpt = self.gpt.to(device).eval()

        self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
        self.campplus = self.campplus.to(device).eval()

    @torch.inference_mode()
    def extract(self, waveforms: Sequence[torch.Tensor], sample_rates: Sequence[int]) -> list[dict[str, np.ndarray]]:
        arrays = [
            resample(waveform, sample_rate, 16000).squeeze(0).cpu().numpy()
            for waveform, sample_rate in zip(waveforms, sample_rates)
        ]
        inputs = self.feature_extractor(arrays, sampling_rate=16000, padding=True, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        output = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        features = (output.hidden_states[17] - self.semantic_mean) / self.semantic_std
        if hasattr(self.semantic_model, "_get_feature_vector_attention_mask"):
            feature_mask = self.semantic_model._get_feature_vector_attention_mask(
                features.shape[1], attention_mask
            )
            feature_lengths = feature_mask.sum(dim=1).long()
        else:
            feature_lengths = torch.full(
                (features.shape[0],), features.shape[1], dtype=torch.long, device=self.device
            )

        emotion_vectors = self.gpt.get_emovec(features, feature_lengths)
        results: list[dict[str, np.ndarray]] = []
        for index, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):
            valid_features = features[index : index + 1, : int(feature_lengths[index].item())]
            codes, _ = self.codec.quantize(valid_features)
            if codes.ndim == 2:
                codes = codes[0]
            audio_16k = resample(waveform, sample_rate, 16000).to(self.device)
            fbank = torchaudio.compliance.kaldi.fbank(
                audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            fbank = fbank - fbank.mean(dim=0, keepdim=True)
            speaker = self.campplus(fbank.unsqueeze(0)).squeeze(0)
            results.append(
                {
                    "codes": codes.detach().cpu().numpy().astype(np.int32),
                    "condition": speaker.detach().cpu().numpy().astype(np.float32),
                    "emo_vec": emotion_vectors[index].detach().cpu().numpy().astype(np.float32),
                }
            )
        return results


def existing_ids(*paths: Path) -> set[str]:
    result: set[str] = set()
    for path in paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        result.add(str(json.loads(line)["id"]))
    return result


def is_validation(sample_id: str, ratio: float) -> bool:
    value = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 1_000_000
    return value < int(max(0.0, min(1.0, ratio)) * 1_000_000)


def save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def flush_batch(
    records: list[dict[str, Any]], pipeline: FeaturePipeline, text_processor: TextProcessor,
    output_dir: Path, train_file: Any, val_file: Any, fallback_language: str,
    max_audio_seconds: float, max_text_tokens: int, max_code_tokens: int,
    val_ratio: float, roots: Sequence[Path],
) -> tuple[int, int]:
    prepared: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        try:
            sample_id = str(record["id"])
            language = canonical_language(record.get("language"), fallback_language)
            text, text_ids = text_processor.encode(record.get("text", ""), language)
            if not text_ids.size or text_ids.size > max_text_tokens:
                raise ValueError(f"text token length {text_ids.size} is outside 1..{max_text_tokens}")
            audio_path = resolve_audio_path(str(record.get("audio", "")), roots)
            if audio_path is None:
                raise FileNotFoundError(f"audio not found: {record.get('audio')}")
            waveform, sample_rate = load_audio(audio_path)
            duration = waveform.shape[-1] / sample_rate
            if duration <= 0 or (max_audio_seconds > 0 and duration > max_audio_seconds):
                raise ValueError(f"audio duration {duration:.2f}s exceeds allowed range")
            prepared.append({
                "record": record, "id": sample_id, "language": language, "text": text,
                "text_ids": text_ids, "audio_path": audio_path, "waveform": waveform,
                "sample_rate": sample_rate, "duration": duration,
            })
        except Exception as exc:
            print(f"[Skip] {record.get('id', '<unknown>')}: {exc}")
            skipped += 1

    if not prepared:
        return 0, skipped
    try:
        features = pipeline.extract(
            [item["waveform"] for item in prepared], [item["sample_rate"] for item in prepared]
        )
    except Exception as exc:
        print(f"[Skip] feature batch failed: {exc}")
        return 0, skipped + len(prepared)

    written = 0
    for item, feature in zip(prepared, features):
        try:
            if feature["codes"].size > max_code_tokens:
                raise ValueError(f"semantic code length {feature['codes'].size} exceeds {max_code_tokens}")
            sample_id = item["id"]
            text_path = output_dir / "text_ids" / f"{sample_id}.npy"
            code_path = output_dir / "codes" / f"{sample_id}.npy"
            condition_path = output_dir / "condition" / f"{sample_id}.npy"
            emotion_path = output_dir / "emo_vec" / f"{sample_id}.npy"
            save_array(text_path, item["text_ids"])
            save_array(code_path, feature["codes"])
            save_array(condition_path, feature["condition"])
            save_array(emotion_path, feature["emo_vec"])
            source = item["record"]
            entry = {
                "id": sample_id, "audio_path": str(source.get("audio", item["audio_path"])),
                "text": item["text"], "speaker": source.get("speaker", ""),
                "language": item["language"], "duration": source.get("duration", item["duration"]),
                "text_ids_path": text_path.relative_to(output_dir).as_posix(),
                "text_len": int(item["text_ids"].size),
                "codes_path": code_path.relative_to(output_dir).as_posix(),
                "code_len": int(feature["codes"].size),
                "condition_path": condition_path.relative_to(output_dir).as_posix(),
                "condition_len": 1,
                "emo_vec_path": emotion_path.relative_to(output_dir).as_posix(),
                "feature_version": "2.5",
            }
            destination = val_file if is_validation(sample_id, val_ratio) else train_file
            destination.write(json.dumps(entry, ensure_ascii=False) + "\n")
            destination.flush()
            written += 1
        except Exception as exc:
            print(f"[Skip] {item['id']}: {exc}")
            skipped += 1
    return written, skipped


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.gpt_checkpoint.expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("The 2.5 config and GPT checkpoint must exist before preprocessing")
    cfg = OmegaConf.load(config_path)
    if str(cfg.get("version", "")) != "2.5":
        raise ValueError("This preprocessor targets IndexTTS 2.5; use a config with version: 2.5")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, val_path = output_dir / "train_manifest.jsonl", output_dir / "val_manifest.jsonl"
    already_done = existing_ids(train_path, val_path) if args.skip_existing else set()
    roots = [
        Path.cwd().resolve(), manifest.parent, manifest.parent.parent,
        *(path.expanduser().resolve() for path in args.audio_root),
    ]
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("id", "")) in already_done:
                continue
            records.append(record)
            if args.max_samples and len(records) >= args.max_samples:
                break
    if not records:
        print("No remaining samples. Nothing to do.")
        return

    text_processor = TextProcessor(model_dir)
    pipeline = FeaturePipeline(cfg, model_dir, checkpoint_path, device)
    written = skipped = 0
    with train_path.open("a", encoding="utf-8") as train_file, val_path.open("a", encoding="utf-8") as val_file:
        for start in tqdm(range(0, len(records), args.batch_size), desc="Preprocessing"):
            batch_written, batch_skipped = flush_batch(
                records[start : start + args.batch_size], pipeline, text_processor, output_dir,
                train_file, val_file, args.language, args.max_audio_seconds,
                int(cfg.gpt.max_text_tokens), int(cfg.gpt.max_mel_tokens) - 2,
                args.val_ratio, roots,
            )
            written += batch_written
            skipped += batch_skipped

    train_count, val_count = len(existing_ids(train_path)), len(existing_ids(val_path))
    stats = {
        "feature_version": "2.5", "written_this_run": written, "skipped_this_run": skipped,
        "train": train_count, "val": val_count, "model_dir": str(model_dir),
        "gpt_checkpoint": str(checkpoint_path),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"[Done] written={written} skipped={skipped} train={train_count} val={val_count}")


if __name__ == "__main__":
    main()
