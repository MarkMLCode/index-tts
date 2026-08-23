# Fine-tuning IndexTTS 2.5

This repository fine-tunes the autoregressive GPT component while retaining the
released 2.5 semantic codec, S2Mel model, and vocoder. The data pipeline mirrors
the 2.5 inference path rather than reusing IndexTTS 2.0 features.

## Source manifest

Use UTF-8 JSONL with one utterance per line:

```json
{"id":"speaker_a_0001","text":"The transcribed sentence.","audio":"audio/speaker_a_0001.wav","speaker":"speaker_a","language":"en"}
```

`id`, `text`, and `audio` are required. `speaker` is strongly recommended and
must identify the same voice across utterances. `language` defaults to the
language supplied to preprocessing. Audio paths may be absolute or relative to
the current directory or source-manifest directory.

Each speaker needs at least two utterances so that the pair builder can use one
utterance as the voice prompt and a different utterance as the training target.
Using the target itself as its prompt encourages content leakage and is not
supported.

## Windows workflow

The checked-in batch files default to the existing example dataset paths. To use
another dataset, set these variables in the same Command Prompt first:

```bat
set INDEXTTS_FINETUNE_DATASET=my_voice
set INDEXTTS_FINETUNE_MANIFEST=C:\datasets\my_voice\manifest.jsonl
```

Then run:

```bat
preprocess_batch.bat
pair_jsonl.bat
train.bat
prune_model.bat
```

The commands produce, respectively:

1. `<name>_processed_data`: 2.5 text IDs, EnhancedCodec semantic codes,
   CAMPPlus speaker embeddings, and emotion vectors.
2. `gpt_pairs_train.jsonl` and `gpt_pairs_val.jsonl`: different-utterance
   prompt/target pairs grouped by speaker.
3. `trained_ckpts_<name>`: resumable training snapshots and TensorBoard logs.
4. `models/<name>_indextts_2_5.pth`: an inference-only GPT checkpoint.

To resume an interrupted training run, add `--resume auto` to `train.bat`.

## What is trained

The trainer constructs `UnifiedVoice` with `spk_cond_mode="campplus"`, exactly
as 2.5 inference does. It optimizes the modules involved in that graph:

- CAMPPlus speaker projection;
- text and language embeddings;
- GPT transformer and positional embeddings;
- semantic-token embedding, final normalization, and prediction heads.

The semantic encoder, EnhancedCodec, CAMPPlus extractor, S2Mel model, and
BigVGAN are fixed feature/decoder components. The first three are used during
preprocessing; S2Mel and BigVGAN are used after GPT generation at inference.

## Important compatibility notes

- IndexTTS 2.0 preprocessed features are not compatible with this trainer.
  Re-run preprocessing so semantic codes, text IDs, and speaker conditions use
  the 2.5 representations.
- A pruned checkpoint can replace `checkpoints/gpt.pth` or be supplied through
  a custom inference/API checkpoint option. Keep the original checkpoint as a
  backup.
- `--amp-dtype auto` uses BF16 on supported CUDA hardware and otherwise FP16.
  BF16 is preferred for numerical stability.
- Samples exceeding the configured GPT text/code capacities are rejected rather
  than truncated against an unchanged transcript.

## Direct commands

Every batch file is a thin wrapper. Run `uv run python <script> --help` for the
complete options:

```text
tools/preprocess_data.py
tools/preprocess_multiproc.py
tools/build_gpt_prompt_pairs.py
tools/generate_gpt_pairs.py
trainers/train_gpt_v2.py
tools/prune_gpt_checkpoint.py
```
