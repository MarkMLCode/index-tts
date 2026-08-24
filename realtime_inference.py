"""Realtime IndexTTS-2.5 playback.

Stream each generated text segment to the speakers. After the model loads,
either synthesize ``--text`` once or keep a prompt open with ``--interactive``.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 22050
DEFAULT_REF = r"Replace this placeholder with the path to your reference audio file."
DEFAULT_TEXT = "Replace this placeholder with the text you want to synthesize."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=DEFAULT_REF, help="Speaker reference WAV")
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesize (ignored with --interactive)",
    )
    parser.add_argument("--lang", default="en", help="Language tag passed to infer()")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "realtime_output.wav"),
        help="Output WAV path for one-shot mode",
    )
    parser.add_argument("--model-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--no-accel", action="store_true", help="Disable the GPT2 accel engine")
    parser.add_argument("--no-fp16", action="store_true", help="Disable half precision")
    parser.add_argument(
        "--seg-tokens",
        type=int,
        default=40,
        help="Max text tokens per streamed segment (smaller = first audio sooner)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Load once, then synthesize each line you type",
    )
    parser.add_argument("--no-play", action="store_true", help="Generate without opening the speakers")
    return parser.parse_args()


def require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "ERROR: sounddevice is required for playback. Install it with:\n"
            "  uv pip install sounddevice",
            file=sys.stderr,
        )
        return None
    return sd


def chunk_to_float32(chunk):
    if chunk is None or not torch.is_tensor(chunk):
        return None
    audio = chunk.detach().float().cpu().numpy()
    if audio.size == 0:
        return None
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio[0]
        else:
            audio = audio.reshape(-1)
    if float(np.max(np.abs(audio))) > 1.5:
        audio = audio / 32767.0
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def play_chunks(chunks, play=True):
    stream = None
    if play:
        sd = require_sounddevice()
        if sd is None:
            return 1
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        stream.start()

    first = True
    started = time.perf_counter()
    try:
        for chunk in chunks:
            audio = chunk_to_float32(chunk)
            if audio is None:
                continue
            if first:
                print(f">> time to first audio: {time.perf_counter() - started:.2f}s")
                first = False
            if stream is not None:
                stream.write(audio.reshape(-1, 1))
    finally:
        if stream is not None:
            stream.stop()
            stream.close()
    if first:
        print(">> no audio chunks were streamed", file=sys.stderr)
        return 1
    return 0


def build_tts(args):
    from indextts.infer_v2_5 import IndexTTS2

    use_half = not args.no_fp16
    use_bf16 = use_half and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_half and not use_bf16:
        print(">> BF16 is not supported on this device, falling back to full precision.")

    print(f">> voice={args.voice}")
    print(f">> lang={args.lang}")
    print(f">> accel={not args.no_accel} bf16={use_bf16}")
    print(f">> seg_tokens={args.seg_tokens} play={not args.no_play}")

    return IndexTTS2(
        cfg_path=str(Path(args.model_dir) / "config.yaml"),
        model_dir=args.model_dir,
        use_bf16=use_bf16,
        use_accel=not args.no_accel,
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_qwen_emo=False,
    )


def synthesize(tts, voice, text, lang, output, seg_tokens, play):
    print(f">> text={text}")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        print(f">> output={output}")
    chunks = tts.infer(
        spk_audio_prompt=str(voice),
        text=text,
        lang=lang,
        output_path=str(output) if output else None,
        stream_return=True,
        verbose=True,
        max_text_tokens_per_segment=seg_tokens,
    )
    return play_chunks(chunks, play=play)


def main():
    args = parse_args()
    voice = Path(args.voice)
    if not voice.is_file():
        print(f"ERROR: reference audio not found: {voice}", file=sys.stderr)
        return 1
    if not args.no_play and require_sounddevice() is None:
        return 1

    os.chdir(REPO_ROOT)
    tts = build_tts(args)

    if not args.interactive:
        return synthesize(
            tts, voice, args.text, args.lang, args.output, args.seg_tokens, not args.no_play
        )

    print(">> interactive mode: type a line and press Enter. Empty line or Ctrl+C to quit.")
    idx = 0
    while True:
        try:
            line = input("tts> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        idx += 1
        output = REPO_ROOT / "outputs" / f"realtime_{idx:03d}.wav"
        code = synthesize(tts, voice, line, args.lang, output, args.seg_tokens, not args.no_play)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
