"""End-of-load synthesis warmup with no persistent prompt changes."""

import numpy as np

from indextts.runtime_logging import file_only_output, timed_stage


def warmup_engine(engine, checkpoint_paths, speaker):
    """Warm each requested GPT plus the shared speech stack; discard the audio.

    The API's model lock must be held throughout. Graphs and immutable head
    caches are retained; speaker/emotion caches and the active model are restored.
    API/WebUI requests reset their seeds before inference, independently of warmup.
    """
    original_path = engine.gpt_path
    reference_cache = {name: value for name, value in vars(engine).items() if name.startswith("cache_")}
    with file_only_output(), timed_stage("Generation warmup total"):
        try:
            for checkpoint in checkpoint_paths:
                with timed_stage(f"Warmup generation: {checkpoint}"):
                    engine.set_gpt_checkpoint(checkpoint, load_if_missing=False)
                    inference_model = getattr(getattr(engine, "gpt", None), "inference_model", None)
                    previous_prefix = getattr(inference_model, "cached_mel_emb", None)
                    try:
                        result = engine.infer(
                            spk_audio_prompt=speaker,
                            text="Hello, world!",
                            lang="EN",
                            output_path=None,
                            verbose=False,
                            max_mel_tokens=256,
                            num_beams=3,
                        )
                        if not isinstance(result, tuple) or len(result) != 2 or np.asarray(result[1]).size == 0:
                            raise RuntimeError("Warmup did not return audio")
                    finally:
                        if inference_model is not None:
                            inference_model.cached_mel_emb = previous_prefix
        finally:
            # Synchronous inference has ended, including on failure. Any unfinished
            # sequences now belong solely to this warmup and must not leak into
            # the next request or prevent restoring the active checkpoint.
            accel = getattr(getattr(engine, "gpt", None), "accel_engine", None)
            if accel is not None:
                accel.current_sequences.clear()
                accel.reset_model_state(weights_changed=False)
                from indextts.accel.attention import reset_forward_context
                reset_forward_context()
            for name, value in reference_cache.items():
                setattr(engine, name, value)
            engine.set_gpt_checkpoint(original_path, load_if_missing=False)
