from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from frontier_loader import (
    AsyncDistributedTokenLoader,
    AsyncShuffledDistributedSequenceLoader,
    DistributedTokenLoader,
    ShuffledDistributedSequenceLoader,
)


def _write_shard(path: Path, tokens: list[int]) -> None:
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, dtype="<u2").tobytes())


class FrontierLoaderTest(unittest.TestCase):
    def test_distributed_loader_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "fineweb_train_000000.bin"
            _write_shard(shard, list(range(32)))
            loader = DistributedTokenLoader(str(Path(td) / "fineweb_train_*.bin"), rank=0, world_size=1, device=torch.device("cpu"))
            x1, y1 = loader.next_batch(global_tokens=8, seq_len=4, grad_accum_steps=1)
            x2, y2 = loader.next_batch(global_tokens=8, seq_len=4, grad_accum_steps=1)
            self.assertEqual(tuple(x1.shape), (2, 4))
            self.assertEqual(tuple(y1.shape), (2, 4))
            self.assertFalse(torch.equal(x1, x2))
            self.assertEqual(loader.loader_stats().batches_served, 2)

    def test_async_loader_prefetch_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "fineweb_train_000000.bin"
            _write_shard(shard, list(range(64)))
            loader = AsyncDistributedTokenLoader(
                str(Path(td) / "fineweb_train_*.bin"),
                rank=0,
                world_size=1,
                device=torch.device("cpu"),
                prefetch_size=1,
            )
            try:
                x, y = loader.next_batch(global_tokens=8, seq_len=4, grad_accum_steps=1)
                self.assertEqual(tuple(x.shape), (2, 4))
                self.assertEqual(tuple(y.shape), (2, 4))
                self.assertGreaterEqual(loader.loader_stats().batches_served, 1)
            finally:
                loader.close()

    def test_shuffled_loader_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "fineweb_train_000000.bin"
            _write_shard(shard, list(range(128)))
            loader = ShuffledDistributedSequenceLoader(
                str(Path(td) / "fineweb_train_*.bin"),
                rank=0,
                world_size=1,
                device=torch.device("cpu"),
                seed=7,
            )
            x, y = loader.next_batch(global_tokens=16, seq_len=4, grad_accum_steps=1)
            self.assertEqual(tuple(x.shape), (4, 4))
            self.assertEqual(tuple(y.shape), (4, 4))
            self.assertGreaterEqual(loader.loader_stats().batches_served, 1)

    def test_async_shuffled_loader_prefetch_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shard = Path(td) / "fineweb_train_000000.bin"
            _write_shard(shard, list(range(128)))
            loader = AsyncShuffledDistributedSequenceLoader(
                str(Path(td) / "fineweb_train_*.bin"),
                rank=0,
                world_size=1,
                device=torch.device("cpu"),
                seed=11,
                prefetch_size=1,
            )
            try:
                x, y = loader.next_batch(global_tokens=16, seq_len=4, grad_accum_steps=1)
                self.assertEqual(tuple(x.shape), (4, 4))
                self.assertEqual(tuple(y.shape), (4, 4))
                self.assertGreaterEqual(loader.loader_stats().batches_served, 1)
            finally:
                loader.close()


if __name__ == "__main__":
    unittest.main()
