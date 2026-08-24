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

Emotion labels are not required. Preprocessing automatically extracts a
continuous emotion embedding from every recording. During training, the prompt
utterance supplies speaker identity while the target utterance supplies the
emotion embedding aligned with the semantic codes being predicted. Keeping
these sources separate prevents random prompt emotions from teaching the GPT to
ignore emotion control.

## Windows workflow

The batch files read the dataset name and source manifest from environment
variables. Set them in the same Command Prompt first:

```bat
set INDEXTTS_FINETUNE_DATASET=my_voice
set INDEXTTS_FINETUNE_MANIFEST=_datasets\my_voice\manifest.jsonl
```

Then run:

```bat
preprocess_batch.bat
pair_jsonl.bat
train.bat
prune_model.bat
```

The commands produce, respectively:

1. `_processed_data/<name>`: 2.5 text IDs, EnhancedCodec semantic codes,
   CAMPPlus speaker embeddings, and emotion vectors.
2. `gpt_pairs_train.jsonl` and `gpt_pairs_val.jsonl`: different-utterance
   prompt/target pairs grouped by speaker.
3. `_trained/<name>`: resumable training snapshots and TensorBoard logs.
4. `_trained/<name>_indextts_2_5.pth`: an inference-only GPT checkpoint.

The WebUI model selector scans `checkpoints`, `models`, and `_trained`. After
pruning a model, click **Refresh**, choose its checkpoint, and click **Load
Model**. The selector replaces only the GPT weights; the codec, S2Mel model,
and vocoder remain loaded. The last successfully loaded checkpoint is restored
on the next WebUI start. Use a fixed non-negative **seed** for reproducible
sampling, or `-1` for a new random seed. Enable **Realtime streaming playback**
to play each completed text segment immediately while retaining the complete
WAV as the final output.

To resume an interrupted training run, add `--resume auto` to `train.bat`.

## What is trained

The trainer constructs `UnifiedVoice` with `spk_cond_mode="campplus"`, exactly
as 2.5 inference does. It optimizes the adaptable modules involved in that
graph:

- text and language embeddings;
- GPT transformer and positional embeddings;
- semantic-token embedding, final normalization, and prediction heads.

The CAMPPlus speaker projection and pretrained emotion-conditioning interface
are frozen to preserve the coordinate system and emotion sensitivity learned by
the base model. The semantic encoder, EnhancedCodec, CAMPPlus extractor, S2Mel
model, and BigVGAN are also fixed feature/decoder components. The first three
are used during preprocessing; S2Mel and BigVGAN are used after GPT generation
at inference.

## Important compatibility notes

- IndexTTS 2.0 preprocessed features are not compatible with this trainer.
  Re-run preprocessing so semantic codes, text IDs, and speaker conditions use
  the 2.5 representations.
- A pruned checkpoint can replace `checkpoints/gpt.pth` or be selected directly
  in the WebUI. Keep the original checkpoint as a backup.
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
