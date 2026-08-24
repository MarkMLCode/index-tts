"""One-shot accelerated IndexTTS-2.5 generation for a local smoke test."""

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REF = r"C:\path\to\reference.wav"
DEFAULT_TEXT = "Replace this placeholder with the text you want to synthesize."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=DEFAULT_REF, help="Speaker reference WAV")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--lang", default="en", help="Language tag passed to infer()")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "generated_audio.wav"),
        help="Output WAV path",
    )
    parser.add_argument("--model-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--no-accel", action="store_true", help="Disable the GPT2 accel engine")
    parser.add_argument("--no-fp16", action="store_true", help="Disable half precision")
    return parser.parse_args()


def main():
    args = parse_args()
    voice = Path(args.voice)
    if not voice.is_file():
        print(f"ERROR: reference audio not found: {voice}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    from indextts.infer_v2_5 import IndexTTS2

    use_half = not args.no_fp16
    use_bf16 = use_half and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_half and not use_bf16:
        print(">> BF16 is not supported on this device, falling back to full precision.")

    print(f">> voice={voice}")
    print(f">> lang={args.lang}")
    print(f">> accel={not args.no_accel} bf16={use_bf16}")
    print(f">> output={output}")

    os.chdir(REPO_ROOT)
    tts = IndexTTS2(
        cfg_path=str(Path(args.model_dir) / "config.yaml"),
        model_dir=args.model_dir,
        use_bf16=use_bf16,
        use_accel=not args.no_accel,
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )
    result = tts.infer(
        spk_audio_prompt=str(voice),
        text=args.text,
        lang=args.lang,
        output_path=str(output),
        verbose=True,
    )
    print(f"Generated: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
