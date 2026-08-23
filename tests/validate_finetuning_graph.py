"""Manual, CPU-safe validation for the IndexTTS 2.5 fine-tuning graph."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from indextts.gpt.model_v2 import UnifiedVoice  # noqa: E402
from trainers.train_gpt_v2 import compute_losses  # noqa: E402


def validate_checkpoint_schema() -> None:
    cfg = OmegaConf.load(REPO_ROOT / "checkpoints" / "config.yaml")
    checkpoint = torch.load(
        REPO_ROOT / "checkpoints" / "gpt.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    checkpoint_state = checkpoint.get("model", checkpoint)
    with torch.device("meta"):
        model = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(model_state))
    mismatched = sorted(
        key
        for key in set(model_state) & set(checkpoint_state)
        if model_state[key].shape != checkpoint_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise AssertionError(
            f"2.5 checkpoint mismatch: missing={missing[:10]} "
            f"unexpected={unexpected[:10]} shape={mismatched[:10]}"
        )
    print(f"[OK] released checkpoint matches all {len(model_state)} training tensors")


def tiny_model() -> UnifiedVoice:
    condition_module = {
        "output_size": 16,
        "linear_units": 32,
        "attention_heads": 2,
        "num_blocks": 1,
        "input_layer": "conv2d2",
        "perceiver_mult": 2,
    }
    emotion_module = {
        "output_size": 16,
        "linear_units": 32,
        "attention_heads": 2,
        "num_blocks": 1,
        "input_layer": "conv2d2",
        "perceiver_mult": 2,
    }
    return UnifiedVoice(
        layers=2,
        model_dim=32,
        heads=4,
        max_text_tokens=16,
        max_mel_tokens=24,
        number_text_tokens=128,
        number_mel_codes=66,
        start_mel_token=64,
        stop_mel_token=65,
        condition_type="conformer_perceiver",
        condition_module=condition_module,
        emo_condition_module=emotion_module,
        checkpointing=False,
        use_accel=False,
        spk_cond_mode="campplus",
    )


def validate_loss_and_gradients() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    batch = {
        "ids": ["a", "b"],
        "text_ids": torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0]], dtype=torch.long),
        "codes": torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 0, 0]], dtype=torch.long),
        "condition": torch.randn(2, 192),
        "emotion": torch.randn(2, 32),
        "text_lengths": torch.tensor([3, 2], dtype=torch.long),
        "code_lengths": torch.tensor([4, 3], dtype=torch.long),
        "language_ids": torch.tensor([0, 1], dtype=torch.long),
    }
    text_loss, mel_loss, metrics = compute_losses(model, batch, torch.device("cpu"))
    loss = 0.2 * text_loss + 0.8 * mel_loss
    loss.backward()
    required_gradients = {
        "speaker projection": model.spk_emb_proj.weight.grad,
        "language embedding": model.lang_embedding.weight.grad,
        "text embedding": model.text_embedding.weight.grad,
        "GPT": next(model.gpt.parameters()).grad,
        "mel head": model.mel_head.weight.grad,
    }
    missing = [name for name, gradient in required_gradients.items() if gradient is None]
    if missing:
        raise AssertionError(f"Missing gradients for: {missing}")
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite training loss: {loss.item()}")
    print(
        f"[OK] loss/gradient graph: text={text_loss.item():.4f} "
        f"mel={mel_loss.item():.4f} top1={metrics['mel_top1']:.4f}"
    )


if __name__ == "__main__":
    validate_checkpoint_schema()
    validate_loss_and_gradients()
