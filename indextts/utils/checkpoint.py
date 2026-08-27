# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import logging
from pathlib import Path

import torch
import yaml

from indextts.runtime_logging import timed_stage

LOGGER = logging.getLogger("indextts.startup")


def read_checkpoint_state(model_pth, *, mmap=True):
    """Map modern checkpoints lazily; keep support for legacy torch.save files."""
    with timed_stage("GPT checkpoint read/deserialization (CPU)"):
        options = {"map_location": "cpu"}
        if mmap:
            options.update(weights_only=True, mmap=True)
        try:
            checkpoint = torch.load(model_pth, **options)
        except TypeError:
            if not mmap:
                raise
            checkpoint = torch.load(model_pth, map_location="cpu")
        except RuntimeError as exc:
            if not mmap or "mmap can only be used" not in str(exc):
                raise
            options.pop("mmap")
            checkpoint = torch.load(model_pth, **options)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint does not contain a model state dictionary")
        return checkpoint


def load_inference_model(factory, model_pth):
    """Load complete weights into an empty model, retaining real constructor buffers.

    No random parameter initialization or full state-dict copy is needed. Missing
    keys use the original initialized path so partially specified checkpoints
    never leave uninitialized/meta parameters behind. Training loaders are unchanged.
    """
    from accelerate import init_empty_weights

    state = read_checkpoint_state(model_pth)
    rng_state = torch.get_rng_state()
    with timed_stage("GPT module construction"):
        # Keep buffers and unregistered tensors (e.g. Conformer positional
        # encodings) on CPU with their original, deterministic constructor values.
        with init_empty_weights(include_buffers=False):
            model = factory()
    expected = model.state_dict()
    missing = expected.keys() - state.keys()
    if missing:
        LOGGER.info("Incomplete checkpoint; using initialized loader for missing keys: %s", sorted(missing)[:5])
        del expected, model
        torch.set_rng_state(rng_state)
        with timed_stage("GPT compatibility construction (incomplete checkpoint)"):
            model = factory()
        with timed_stage("GPT load_state_dict (CPU)"):
            result = model.load_state_dict(state, strict=False)
    else:
        # assign=True preserves source dtype; match the old copy-based loader's
        # cast to the constructor dtype first (including fp16/fp64 checkpoints).
        assigned = copy.copy(state)
        for name, target in expected.items():
            assigned[name] = state[name].to(dtype=target.dtype)
        with timed_stage("GPT load_state_dict (CPU)"):
            result = model.load_state_dict(assigned, strict=False, assign=True)
        if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
            raise RuntimeError("GPT checkpoint left meta tensors uninitialized")
    if result.unexpected_keys:
        LOGGER.info("GPT checkpoint unused keys: %s", result.unexpected_keys[:5])
    return model


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = read_checkpoint_state(model_pth, mmap=False)
    with timed_stage("GPT load_state_dict (CPU)"):
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f">> load_checkpoint: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f">> load_checkpoint: skipping unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    info_path = str(Path(model_pth).with_suffix('.yaml'))
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
