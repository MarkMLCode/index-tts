from __future__ import annotations

import unittest

import torch

from indextts.accel.kv_manager import KVCacheManager, Seq


class KVCacheManagerResetTests(unittest.TestCase):
    def make_manager(self) -> KVCacheManager:
        return KVCacheManager(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            block_size=2,
            num_blocks=3,
            dtype=torch.float32,
        )

    def test_allocate_and_deallocate_maintain_reference_counts(self) -> None:
        manager = self.make_manager()
        sequence = Seq([1, 2, 3], block_size=2)

        manager.allocate(sequence)
        self.assertEqual([manager.blocks[index].ref_cnt for index in sequence.block_table], [1, 1])

        manager.remove_seq(sequence)
        self.assertTrue(all(block.ref_cnt == 0 for block in manager.blocks))
        self.assertEqual(set(manager.free_block_ids), {0, 1, 2})

    def test_reset_releases_allocated_blocks_for_model_hotswap(self) -> None:
        manager = self.make_manager()
        manager.allocate(Seq([1, 2, 3], block_size=2))

        manager.reset()

        self.assertFalse(manager.used_block_ids)
        self.assertFalse(manager.block_hash_to_id)
        self.assertEqual(list(manager.free_block_ids), [0, 1, 2])
        self.assertTrue(all(block.ref_cnt == 0 for block in manager.blocks))

        replacement = Seq([4, 5, 6], block_size=2)
        manager.allocate(replacement)
        self.assertEqual([manager.blocks[index].ref_cnt for index in replacement.block_table], [1, 1])


if __name__ == "__main__":
    unittest.main()
