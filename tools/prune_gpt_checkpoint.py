#!/usr/bin/env python3
"""Convert a training checkpoint into an IndexTTS inference checkpoint."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype", choices=("keep", "float32", "float16", "bfloat16"), default="keep"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def convert_tensor(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if not tensor.is_floating_point() or dtype == "keep":
        return tensor
    target = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    return tensor.to(target)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input.expanduser().resolve(), map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    if not isinstance(source, dict):
        raise TypeError("Expected a state dict or a training checkpoint containing 'model'")
    state: OrderedDict[str, Any] = OrderedDict()
    parameters = 0
    for key, value in source.items():
        if key.startswith("inference_model.") or key.startswith("accel_engine."):
            continue
        if isinstance(value, torch.Tensor):
            value = convert_tensor(value, args.dtype)
            parameters += value.numel()
        state[key] = value
    print(f"[Prune] retained {len(state)} entries / {parameters:,} tensor elements")
    if args.dry_run:
        return
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": state,
        "metadata": {
            "model_version": checkpoint.get("training", {}).get("model_version", "2.5"),
            "source_step": checkpoint.get("step"),
            "source_checkpoint": str(args.input),
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(f"[Prune] wrote {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
