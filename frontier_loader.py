from __future__ import annotations

import glob
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class LoaderStats:
    batches_served: int
    tokens_served: int
    load_wait_seconds: float
    prefetch_hits: int


_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}


def load_data_shard(file: Path) -> Tensor:
    header_bytes = _SHARD_HEADER_BYTES
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def _read_num_tokens(file: Path) -> int:
    key = str(file)
    cached = _SHARD_NTOKENS_CACHE.get(key)
    if cached is not None:
        return cached
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = num_tokens
    return num_tokens


def _get_shard_memmap(file: Path) -> np.memmap:
    key = str(file)
    cached = _MMAP_CACHE.get(key)
    if cached is not None:
        return cached
    num_tokens = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(num_tokens,))
    _MMAP_CACHE[key] = mm
    return mm


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

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
            take = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + take])
            self.pos += take
            remaining -= take
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def state_dict(self) -> dict[str, int]:
        return {"file_idx": self.file_idx, "pos": self.pos}

    def load_state_dict(self, state: dict[str, int]) -> None:
        file_idx = int(state["file_idx"])
        if not 0 <= file_idx < len(self.files):
            raise ValueError(f"Invalid TokenStream file_idx={file_idx} for {len(self.files)} files")
        self.file_idx = file_idx
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = int(state["pos"])
        if not 0 <= self.pos <= self.tokens.numel():
            raise ValueError(f"Invalid TokenStream pos={self.pos} for shard length {self.tokens.numel()}")


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
        self._batches_served = 0
        self._tokens_served = 0
        self._load_wait_seconds = 0.0
        self._prefetch_hits = 0

    def _batch_from_chunk(self, chunk: Tensor, seq_len: int, local_tokens: int) -> tuple[Tensor, Tensor]:
        per_rank_span = local_tokens + 1
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        t0 = time.perf_counter()
        chunk = self.stream.take(per_rank_span * self.world_size)
        self._load_wait_seconds += time.perf_counter() - t0
        self._batches_served += 1
        self._tokens_served += local_tokens
        return self._batch_from_chunk(chunk, seq_len, local_tokens)

    def state_dict(self) -> dict[str, object]:
        return {
            "stream": self.stream.state_dict(),
            "stats": self.loader_stats().__dict__,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.stream.load_state_dict(state["stream"])

    def loader_stats(self) -> LoaderStats:
        return LoaderStats(
            batches_served=self._batches_served,
            tokens_served=self._tokens_served,
            load_wait_seconds=round(self._load_wait_seconds, 6),
            prefetch_hits=self._prefetch_hits,
        )


class AsyncDistributedTokenLoader(DistributedTokenLoader):
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        prefetch_size: int = 2,
    ):
        super().__init__(pattern, rank, world_size, device)
        self.prefetch_size = max(1, int(prefetch_size))
        self._queue: queue.Queue[tuple[int, int, Tensor]] = queue.Queue(maxsize=self.prefetch_size)
        self._stop_event = threading.Event()
        self._pending_batch: tuple[int, int] | None = None
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            pending = self._pending_batch
            if pending is None:
                time.sleep(0.001)
                continue
            local_tokens, seq_len = pending
            chunk = self.stream.take((local_tokens + 1) * self.world_size)
            try:
                self._queue.put((local_tokens, seq_len, chunk), timeout=0.1)
            except queue.Full:
                continue

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        self._pending_batch = (local_tokens, seq_len)
        t0 = time.perf_counter()
        try:
            prefetched_local_tokens, prefetched_seq_len, chunk = self._queue.get(timeout=5.0)
            if prefetched_local_tokens == local_tokens and prefetched_seq_len == seq_len:
                self._prefetch_hits += 1
            else:
                chunk = self.stream.take((local_tokens + 1) * self.world_size)
        except queue.Empty:
            chunk = self.stream.take((local_tokens + 1) * self.world_size)
        self._load_wait_seconds += time.perf_counter() - t0
        self._batches_served += 1
        self._tokens_served += local_tokens
        return self._batch_from_chunk(chunk, seq_len, local_tokens)

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


class ShuffledDistributedSequenceLoader:
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        seed: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.seed = seed
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.files = self.files[rank::world_size] or self.files
        self.rng = np.random.Generator(np.random.PCG64(seed + rank))
        self.num_tokens = [_read_num_tokens(file) for file in self.files]
        self.start_inds = [[] for _ in self.files]
        self._active_seq_len: int | None = None
        self._batches_served = 0
        self._tokens_served = 0
        self._load_wait_seconds = 0.0
        self._prefetch_hits = 0

    def _reset_shard(self, shard_idx: int, seq_len: int) -> None:
        max_phase = min(seq_len - 1, max(0, self.num_tokens[shard_idx] - seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[shard_idx] - 1 - phase) // seq_len
        order = self.rng.permutation(num_sequences) if num_sequences > 0 else np.asarray([], dtype=np.int64)
        self.start_inds[shard_idx] = (phase + order * seq_len).tolist()

    def _ensure_seq_len(self, seq_len: int) -> None:
        if self._active_seq_len == seq_len:
            return
        self._active_seq_len = seq_len
        for shard_idx in range(len(self.files)):
            self._reset_shard(shard_idx, seq_len)

    def _produce_batch_arrays(self, local_tokens: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        if local_tokens < seq_len:
            raise ValueError(
                f"Batch provides fewer local tokens than seq_len: local_tokens={local_tokens} seq_len={seq_len}"
            )
        self._ensure_seq_len(seq_len)
        device_batch_size = local_tokens // seq_len
        remaining = np.asarray([len(s) for s in self.start_inds], dtype=np.float64)
        x_np = np.empty((device_batch_size, seq_len), dtype=np.int64)
        y_np = np.empty((device_batch_size, seq_len), dtype=np.int64)
        for batch_idx in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for shard_idx in range(len(self.files)):
                    self._reset_shard(shard_idx, seq_len)
                remaining = np.asarray([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            shard_idx = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[shard_idx].pop()
            remaining[shard_idx] -= 1
            window = _get_shard_memmap(self.files[shard_idx])[start_ind : start_ind + seq_len + 1]
            x_np[batch_idx] = window[:-1]
            y_np[batch_idx] = window[1:]
        return x_np, y_np

    def _to_device(self, x_np: np.ndarray, y_np: np.ndarray) -> tuple[Tensor, Tensor]:
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        y = torch.from_numpy(y_np).to(self.device, non_blocking=True)
        return x, y

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        t0 = time.perf_counter()
        x_np, y_np = self._produce_batch_arrays(local_tokens, seq_len)
        self._load_wait_seconds += time.perf_counter() - t0
        self._batches_served += 1
        self._tokens_served += local_tokens
        return self._to_device(x_np, y_np)

    def loader_stats(self) -> LoaderStats:
        return LoaderStats(
            batches_served=self._batches_served,
            tokens_served=self._tokens_served,
            load_wait_seconds=round(self._load_wait_seconds, 6),
            prefetch_hits=self._prefetch_hits,
        )


class AsyncShuffledDistributedSequenceLoader(ShuffledDistributedSequenceLoader):
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        *,
        seed: int = 0,
        prefetch_size: int = 2,
    ):
        super().__init__(pattern, rank, world_size, device, seed=seed)
        self.prefetch_size = max(1, int(prefetch_size))
        self._queue: queue.Queue[tuple[int, int, np.ndarray, np.ndarray]] = queue.Queue(maxsize=self.prefetch_size)
        self._pending_request: tuple[int, int] | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            pending = self._pending_request
            if pending is None:
                time.sleep(0.001)
                continue
            local_tokens, seq_len = pending
            batch = self._produce_batch_arrays(local_tokens, seq_len)
            try:
                self._queue.put((local_tokens, seq_len, *batch), timeout=0.1)
            except queue.Full:
                continue

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        self._pending_request = (local_tokens, seq_len)
        t0 = time.perf_counter()
        try:
            prefetched_local_tokens, prefetched_seq_len, x_np, y_np = self._queue.get(timeout=5.0)
            if prefetched_local_tokens == local_tokens and prefetched_seq_len == seq_len:
                self._prefetch_hits += 1
            else:
                x_np, y_np = self._produce_batch_arrays(local_tokens, seq_len)
        except queue.Empty:
            x_np, y_np = self._produce_batch_arrays(local_tokens, seq_len)
        self._load_wait_seconds += time.perf_counter() - t0
        self._batches_served += 1
        self._tokens_served += local_tokens
        return self._to_device(x_np, y_np)

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


def build_train_loader(
    pattern: str,
    rank: int,
    world_size: int,
    device: torch.device,
    *,
    async_enabled: bool,
    prefetch_size: int = 2,
    shuffled: bool = False,
    seed: int = 0,
) -> DistributedTokenLoader | ShuffledDistributedSequenceLoader:
    if shuffled and async_enabled:
        return AsyncShuffledDistributedSequenceLoader(
            pattern,
            rank,
            world_size,
            device,
            seed=seed,
            prefetch_size=prefetch_size,
        )
    if shuffled:
        return ShuffledDistributedSequenceLoader(pattern, rank, world_size, device, seed=seed)
    if async_enabled:
        return AsyncDistributedTokenLoader(pattern, rank, world_size, device, prefetch_size=prefetch_size)
    return DistributedTokenLoader(pattern, rank, world_size, device)
