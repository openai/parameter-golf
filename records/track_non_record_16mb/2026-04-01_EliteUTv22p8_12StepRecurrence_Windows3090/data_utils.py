import glob
import os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False)).pin_memory()


class TokenStream:
    def __init__(self, pattern: str, rank: int = 0, world_size: int = 1):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        
        # ELITE FIX: Distributed Shard Partitioning (Zero Redundancy)
        # Each rank processes a unique subset of the global shards.
        self.files = self.files[rank::world_size]
        if not self.files:
            raise ValueError(f"Rank {rank}/{world_size} has no shards to process!")

        # ELITE FIX: Global Shard Shuffling (Stable across runs with the same seed)
        import random
        rng = random.Random(int(os.environ.get("SEED", 42)))
        rng.shuffle(self.files)
        
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        
        # ELITE FIX: Random Start Offset (To destroy boundary artifacts)
        # Every rank starts at a different, random 'cut' within the first shard.
        seq_len_hint = int(os.environ.get("TRAIN_SEQ_LEN", 256))
        self.pos = rng.randint(0, seq_len_hint - 1)
        print(f"[data] Rank {rank}/{world_size} initialized at file {self.files[0].name}, pos {self.pos}")

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, rank, world_size)

    def next_batch(self, micro_batch_tokens: int, seq_len: int) -> tuple[Tensor, Tensor]:
        """Loads a single micro-batch of tokens for the current rank."""
        per_rank_span = micro_batch_tokens + 1
        # Each rank takes its own span sequentially from the stream
        chunk = self.stream.take(per_rank_span)
        local = chunk.to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
