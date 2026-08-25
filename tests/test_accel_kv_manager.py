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

    def test_fork_shares_full_blocks_and_copies_partial_block_on_write(self) -> None:
        manager = KVCacheManager(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            block_size=4,
            num_blocks=6,
            dtype=torch.float32,
        )
        parent = Seq([1, 2, 3], block_size=4)
        manager.allocate(parent)
        shared_block_id = parent.block_table[-1]
        manager.kv_cache[:, :, shared_block_id].fill_(7)

        child = manager.fork_sequence(parent)
        self.assertEqual(child.block_table, parent.block_table)
        self.assertEqual(manager.blocks[shared_block_id].ref_cnt, 2)

        child.append_token(4)
        manager.append_to_seq(child)

        child_block_id = child.block_table[-1]
        self.assertNotEqual(child_block_id, shared_block_id)
        self.assertEqual(manager.blocks[shared_block_id].ref_cnt, 1)
        self.assertEqual(manager.blocks[child_block_id].ref_cnt, 1)
        torch.testing.assert_close(
            manager.kv_cache[:, :, child_block_id],
            manager.kv_cache[:, :, shared_block_id],
        )

        manager.remove_seq(parent)
        manager.remove_seq(child)
        self.assertTrue(all(block.ref_cnt == 0 for block in manager.blocks))

    def test_fork_keeps_completed_prefix_block_shared(self) -> None:
        manager = KVCacheManager(
            num_layers=1,
            num_heads=1,
            head_dim=1,
            block_size=2,
            num_blocks=5,
            dtype=torch.float32,
        )
        parent = Seq([1, 2], block_size=2)
        manager.allocate(parent)
        shared_block_id = parent.block_table[0]
        child = manager.fork_sequence(parent)

        child.append_token(3)
        manager.append_to_seq(child)

        self.assertEqual(parent.block_table[0], shared_block_id)
        self.assertEqual(child.block_table[0], shared_block_id)
        self.assertNotEqual(child.block_table[-1], shared_block_id)
        self.assertEqual(manager.blocks[shared_block_id].ref_cnt, 2)


if __name__ == "__main__":
    unittest.main()
