from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_gpt_prompt_pairs import build_pairs, group_by_speaker, read_manifest  # noqa: E402
from generate_gpt_pairs import generate_for_manifest  # noqa: E402


def record(sample_id: str, language: str = "en") -> dict:
    return {
        "id": sample_id,
        "speaker": "speaker-a",
        "audio_path": f"audio/{sample_id}.wav",
        "text": sample_id,
        "language": language,
        "text_ids_path": f"text_ids/{sample_id}.npy",
        "text_len": 4,
        "codes_path": f"codes/{sample_id}.npy",
        "code_len": 12,
        "condition_path": f"condition/{sample_id}.npy",
        "condition_len": 1,
        "emo_vec_path": f"emo_vec/{sample_id}.npy",
        "feature_version": "2.5",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")


class FineTuningPairTests(unittest.TestCase):
    def test_validation_target_can_use_training_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path = root / "train.jsonl", root / "val.jsonl"
            write_jsonl(train_path, [record("train-1"), record("train-2")])
            write_jsonl(val_path, [record("val-1")])
            targets = group_by_speaker(read_manifest(val_path))
            prompts = group_by_speaker(read_manifest(train_path))
            pairs = build_pairs(
                targets,
                pairs_per_target=2,
                min_text_len=1,
                min_code_len=1,
                prompt_grouped=prompts,
            )
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(pair["target_id"] == "val-1" for pair in pairs))
            self.assertTrue(all(pair["prompt_id"].startswith("train-") for pair in pairs))

    def test_generated_pair_preserves_25_language_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path, output = (
                root / "train.jsonl",
                root / "val.jsonl",
                root / "pairs.jsonl",
            )
            write_jsonl(train_path, [record("train-1", "ja"), record("train-2", "ja")])
            write_jsonl(val_path, [record("val-1", "ja")])
            count = generate_for_manifest(
                val_path,
                output,
                pairs_per_target=1,
                min_text_len=1,
                min_code_len=1,
                max_pairs=None,
                prompt_manifest_path=train_path,
            )
            self.assertEqual(count, 1)
            pair = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(pair["feature_version"], "2.5")
            self.assertEqual(pair["prompt_language"], "ja")
            self.assertEqual(pair["target_language"], "ja")


if __name__ == "__main__":
    unittest.main()
