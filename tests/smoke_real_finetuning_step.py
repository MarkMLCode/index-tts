"""Run one real 2.5 forward/backward step without saving an optimizer checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trainers.train_gpt_v2 import (  # noqa: E402
    GPTPairDataset,
    ManifestSpec,
    build_model,
    collate_batch,
    compute_losses,
    optimizer_for,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints" / "config.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "gpt.pth"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, _ = build_model(args.config.resolve(), args.checkpoint.resolve(), device)
    dataset = GPTPairDataset([ManifestSpec(args.manifest.resolve())])
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_batch)
    batch = next(iter(loader))
    model.train()
    optimizer = optimizer_for(model, learning_rate=1e-5, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        text_loss, mel_loss, metrics = compute_losses(model, batch, device)
        loss = 0.2 * text_loss + 0.8 * mel_loss
    loss.backward()
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite loss: {loss.item()}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    peak_gib = 0.0
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print(
        f"[OK] real 2.5 optimizer step: loss={loss.item():.4f} text={text_loss.item():.4f} "
        f"mel={mel_loss.item():.4f} top1={metrics['mel_top1']:.4f} peak_vram={peak_gib:.2f} GiB"
    )


if __name__ == "__main__":
    main()
