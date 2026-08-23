#!/usr/bin/env python3
"""Fine-tune the IndexTTS 2.5 autoregressive GPT.

The paired manifests are produced by ``tools/generate_gpt_pairs.py``. Training
uses the same CAMPPlus speaker, emotion, language, text, and semantic-code
conditioning layout as ``indextts.infer_v2_5.IndexTTS2``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.tokenizer import LANGUAGE_DICT, TO_LANGUAGE_CODE, lang_to_token


LANGUAGE_ALIASES = {"cn": "zh", "zhen": "zh", "jp": "ja"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", dest="train_manifests", action="append", required=True)
    parser.add_argument("--val-manifest", dest="val_manifests", action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/config.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/gpt.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("trained_ckpts"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=0.2)
    parser.add_argument("--mel-loss-weight", type=float, default=0.8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto",
        help="AMP precision. Auto prefers bfloat16 on supported NVIDIA GPUs.",
    )
    parser.add_argument("--resume", default="", help="Checkpoint path or 'auto'.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def canonical_language(value: str | None) -> str:
    lang = (value or "common").strip().lower()
    lang = LANGUAGE_ALIASES.get(lang, lang)
    lang = TO_LANGUAGE_CODE.get(lang, lang)
    return lang if lang in LANGUAGE_DICT else "common"


@dataclass(frozen=True)
class ManifestSpec:
    path: Path
    language: str | None = None


def parse_manifest_specs(values: Sequence[str]) -> list[ManifestSpec]:
    result: list[ManifestSpec] = []
    for raw in values:
        value, language = raw.strip(), None
        if "::" in value:
            value, language = value.rsplit("::", 1)
        result.append(ManifestSpec(Path(value).expanduser().resolve(), language))
    return result


@dataclass(frozen=True)
class Sample:
    id: str
    text_ids_path: Path
    codes_path: Path
    condition_path: Path
    emotion_path: Path
    text_len: int
    code_len: int
    language: str


class GPTPairDataset(Dataset):
    def __init__(self, specs: Sequence[ManifestSpec]):
        self.samples: list[Sample] = []
        self.manifest_summaries: list[dict[str, Any]] = []
        for spec in specs:
            self._load_manifest(spec)
        if not self.samples:
            raise RuntimeError("No training pairs were found in the supplied manifests")

    @staticmethod
    def _resolve(root: Path, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    def _load_manifest(self, spec: ManifestSpec) -> None:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        count = 0
        languages: set[str] = set()
        with spec.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get("feature_version", "2.0")) != "2.5":
                    raise ValueError(
                        f"{spec.path}:{line_number} contains non-2.5 features; rerun preprocessing"
                    )
                language = canonical_language(record.get("target_language") or spec.language)
                emotion = record.get("prompt_emo_vec_path") or record.get("target_emo_vec_path")
                if not emotion:
                    raise ValueError(f"{spec.path}:{line_number} is missing an emotion vector")
                self.samples.append(
                    Sample(
                        id=str(record["id"]),
                        text_ids_path=self._resolve(spec.path.parent, record["target_text_ids_path"]),
                        codes_path=self._resolve(spec.path.parent, record["target_codes_path"]),
                        condition_path=self._resolve(spec.path.parent, record["prompt_condition_path"]),
                        emotion_path=self._resolve(spec.path.parent, emotion),
                        text_len=int(record["target_text_len"]),
                        code_len=int(record["target_code_len"]),
                        language=language,
                    )
                )
                languages.add(language)
                count += 1
        self.manifest_summaries.append(
            {"path": str(spec.path), "count": count, "languages": sorted(languages)}
        )
        print(f"[Data] {spec.path}: {count} pairs; languages={sorted(languages)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        text_ids = np.load(sample.text_ids_path, allow_pickle=False).astype(np.int64, copy=False)
        codes = np.load(sample.codes_path, allow_pickle=False).astype(np.int64, copy=False)
        condition = np.load(sample.condition_path, allow_pickle=False).astype(np.float32, copy=False)
        emotion = np.load(sample.emotion_path, allow_pickle=False).astype(np.float32, copy=False)
        if text_ids.ndim != 1 or text_ids.size != sample.text_len:
            raise ValueError(f"{sample.id}: invalid text IDs shape {text_ids.shape}")
        if codes.ndim != 1 or codes.size != sample.code_len:
            raise ValueError(f"{sample.id}: invalid semantic codes shape {codes.shape}")
        condition = condition.reshape(-1)
        emotion = emotion.reshape(-1)
        if condition.shape != (192,):
            raise ValueError(f"{sample.id}: expected CAMPPlus [192], got {condition.shape}")
        if emotion.shape != (1280,):
            raise ValueError(f"{sample.id}: expected emotion [1280], got {emotion.shape}")
        return {
            "id": sample.id,
            "text_ids": torch.from_numpy(text_ids),
            "codes": torch.from_numpy(codes),
            "condition": torch.from_numpy(condition),
            "emotion": torch.from_numpy(emotion),
            "text_len": sample.text_len,
            "code_len": sample.code_len,
            "language_id": lang_to_token(sample.language),
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [item["id"] for item in items],
        "text_ids": pad_sequence(
            [item["text_ids"] for item in items], batch_first=True, padding_value=0
        ),
        "codes": pad_sequence(
            [item["codes"] for item in items], batch_first=True, padding_value=0
        ),
        "condition": torch.stack([item["condition"] for item in items]),
        "emotion": torch.stack([item["emotion"] for item in items]),
        "text_lengths": torch.tensor([item["text_len"] for item in items], dtype=torch.long),
        "code_lengths": torch.tensor([item["code_len"] for item in items], dtype=torch.long),
        "language_ids": torch.tensor([item["language_id"] for item in items], dtype=torch.long),
    }


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return {
        key.replace(".base_layer.", "."): value
        for key, value in state.items()
        if not key.startswith("inference_model.") and ".lora_" not in key and key != "gpt.wte.weight"
    }


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[UnifiedVoice, Any]:
    cfg = OmegaConf.load(config_path)
    if str(cfg.get("version", "")) != "2.5":
        raise ValueError("This trainer requires an IndexTTS config with version: 2.5")
    model = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    missing, unexpected = model.load_state_dict(checkpoint_state(checkpoint_path), strict=False)
    real_missing = [key for key in missing if key != "gpt.wte.weight"]
    if real_missing or unexpected:
        raise RuntimeError(
            "Base checkpoint is not compatible with the 2.5 graph: "
            f"missing={real_missing[:20]}, unexpected={unexpected[:20]}"
        )
    return model.to(device), cfg


def compute_losses(
    model: UnifiedVoice, batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    text_ids = batch["text_ids"].to(device, non_blocking=True)
    codes = batch["codes"].to(device, non_blocking=True)
    speaker = batch["condition"].to(device, non_blocking=True)
    emotion = batch["emotion"].to(device, non_blocking=True)
    text_lengths = batch["text_lengths"].to(device, non_blocking=True)
    code_lengths = batch["code_lengths"].to(device, non_blocking=True)
    language_ids = batch["language_ids"].to(device, non_blocking=True)

    speaker_condition = model.spk_emb_proj(speaker).unsqueeze(1)
    zero_conditions = torch.zeros(
        speaker_condition.shape[0], 2, speaker_condition.shape[2],
        dtype=speaker_condition.dtype, device=device,
    )
    conditions = torch.cat((speaker_condition + emotion.unsqueeze(1), zero_conditions), dim=1)

    text_inputs = model.set_text_padding(text_ids.clone(), text_lengths)
    text_inputs = F.pad(text_inputs, (0, 1), value=model.stop_text_token)
    text_inputs, text_targets = model.build_aligned_inputs_and_targets(
        text_inputs, model.start_text_token, model.stop_text_token
    )
    text_embeddings = model.text_embedding(text_inputs) + model.text_pos_embedding(text_inputs)
    text_embeddings = text_embeddings + model.lang_embedding(language_ids).unsqueeze(1)

    mel_inputs = model.set_mel_padding(codes.clone(), code_lengths)
    mel_inputs = F.pad(mel_inputs, (0, 1), value=model.stop_mel_token)
    mel_inputs, mel_targets = model.build_aligned_inputs_and_targets(
        mel_inputs, model.start_mel_token, model.stop_mel_token
    )
    mel_embeddings = model.mel_embedding(mel_inputs) + model.mel_pos_embedding(mel_inputs)

    text_logits, mel_logits = model.get_logits(
        conditions, text_embeddings, model.text_head, mel_embeddings, model.mel_head
    )
    text_mask = torch.arange(text_targets.size(1), device=device).unsqueeze(0) < (
        text_lengths + 1
    ).unsqueeze(1)
    mel_mask = torch.arange(mel_targets.size(1), device=device).unsqueeze(0) < (
        code_lengths + 1
    ).unsqueeze(1)
    text_per_token = F.cross_entropy(text_logits, text_targets, reduction="none")
    mel_per_token = F.cross_entropy(mel_logits, mel_targets, reduction="none")
    text_loss = (text_per_token * text_mask).sum() / text_mask.sum().clamp_min(1)
    mel_loss = (mel_per_token * mel_mask).sum() / mel_mask.sum().clamp_min(1)
    with torch.no_grad():
        predictions = mel_logits.argmax(dim=1)
        mel_top1 = ((predictions == mel_targets) & mel_mask).sum() / mel_mask.sum().clamp_min(1)
    return text_loss, mel_loss, {"mel_top1": float(mel_top1.item())}


def optimizer_for(model: nn.Module, learning_rate: float, weight_decay: float) -> AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 and not name.endswith("bias") else no_decay).append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
    epoch: int, step: int, recent: list[str], metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch, "step": step, "recent_checkpoints": recent,
        "training": metadata,
    }


@torch.no_grad()
def evaluate(
    model: UnifiedVoice, loader: DataLoader, device: torch.device,
    amp_enabled: bool, amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    totals = {"text_loss": 0.0, "mel_loss": 0.0, "mel_top1": 0.0}
    samples = 0
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            text_loss, mel_loss, metrics = compute_losses(model, batch, device)
        size = len(batch["ids"])
        totals["text_loss"] += text_loss.item() * size
        totals["mel_loss"] += mel_loss.item() * size
        totals["mel_top1"] += metrics["mel_top1"] * size
        samples += size
    model.train()
    return {key: value / max(1, samples) for key, value in totals.items()}


def select_amp_dtype(args: argparse.Namespace, device: torch.device) -> tuple[bool, torch.dtype]:
    if not args.amp or device.type != "cuda":
        return False, torch.float32
    if args.amp_dtype == "bfloat16" or (
        args.amp_dtype == "auto" and torch.cuda.is_bf16_supported()
    ):
        return True, torch.bfloat16
    return True, torch.float16


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.grad_accumulation, args.log_interval, args.save_every) < 1:
        raise ValueError("batch, accumulation, logging, and saving intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = os.environ.get("INDEXTTS_RUN_NAME") or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(output_dir / "logs" / run_name))
    model, cfg = build_model(args.config.expanduser().resolve(), args.base_checkpoint.expanduser().resolve(), device)
    train_data = GPTPairDataset(parse_manifest_specs(args.train_manifests))
    val_data = GPTPairDataset(parse_manifest_specs(args.val_manifests))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_batch, pin_memory=pin_memory, persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_batch, pin_memory=pin_memory, persistent_workers=args.num_workers > 0,
    )
    optimizer = optimizer_for(model, args.learning_rate, args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation)
    total_steps = args.max_steps or max(1, args.epochs * updates_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=min(args.warmup_steps, total_steps), num_training_steps=total_steps
    )
    amp_enabled, amp_dtype = select_amp_dtype(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    metadata = {
        "model_version": "2.5", "config": str(args.config), "base_checkpoint": str(args.base_checkpoint),
        "train_manifests": train_data.manifest_summaries, "val_manifests": val_data.manifest_summaries,
        "amp_dtype": str(amp_dtype),
    }

    epoch_start = global_step = 0
    recent: list[str] = []
    resume_path: Path | None = None
    if args.resume:
        resume_path = output_dir / "latest.pth" if args.resume == "auto" else Path(args.resume)
    if resume_path and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        epoch_start, global_step = int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))
        recent = list(checkpoint.get("recent_checkpoints", []))
        print(f"[Resume] {resume_path}: epoch={epoch_start} step={global_step}")

    print(
        f"[Train] pairs={len(train_data)} validation={len(val_data)} device={device} "
        f"amp={amp_enabled} dtype={amp_dtype} total_steps={total_steps}"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    best_val = math.inf
    stop = False
    last_saved = global_step
    for epoch in range(epoch_start, args.epochs):
        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                text_loss, mel_loss, metrics = compute_losses(model, batch, device)
                loss = args.text_loss_weight * text_loss + args.mel_loss_weight * mel_loss
            scaled_loss = loss / args.grad_accumulation
            scaler.scale(scaled_loss).backward() if scaler.is_enabled() else scaled_loss.backward()
            should_step = (batch_index + 1) % args.grad_accumulation == 0 or batch_index + 1 == len(train_loader)
            if not should_step:
                continue
            if args.grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/text_loss", text_loss.item(), global_step)
                writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                writer.add_scalar("train/mel_top1", metrics["mel_top1"], global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                print(
                    f"[Train] epoch={epoch + 1} step={global_step} loss={loss.item():.4f} "
                    f"text={text_loss.item():.4f} mel={mel_loss.item():.4f} "
                    f"top1={metrics['mel_top1']:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if args.val_interval > 0 and global_step % args.val_interval == 0:
                values = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
                for name, value in values.items():
                    writer.add_scalar(f"val/{name}", value, global_step)
                best_val = min(best_val, values["mel_loss"])
                print(f"[Val] step={global_step} {values} best_mel={best_val:.4f}")

            if global_step % args.save_every == 0:
                path = output_dir / f"model_step{global_step}.pth"
                recent.append(str(path))
                payload = make_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, recent, metadata)
                atomic_save(payload, path)
                atomic_save(payload, output_dir / "latest.pth")
                while len(recent) > args.keep_checkpoints:
                    obsolete = Path(recent.pop(0)).resolve()
                    if obsolete.parent == output_dir and obsolete.name.startswith("model_step"):
                        obsolete.unlink(missing_ok=True)
                last_saved = global_step

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        if stop:
            break
        if args.val_interval == 0:
            values = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
            for name, value in values.items():
                writer.add_scalar(f"val/{name}", value, global_step)
            best_val = min(best_val, values["mel_loss"])
            print(f"[Val] epoch={epoch + 1} step={global_step} {values}")

    if global_step and last_saved != global_step:
        path = output_dir / f"model_step{global_step}.pth"
        recent.append(str(path))
        payload = make_checkpoint(model, optimizer, scheduler, scaler, epoch, global_step, recent, metadata)
        atomic_save(payload, path)
        atomic_save(payload, output_dir / "latest.pth")
    writer.close()
    print(f"Training complete: steps={global_step}; best validation mel loss={best_val:.4f}")


if __name__ == "__main__":
    main()
