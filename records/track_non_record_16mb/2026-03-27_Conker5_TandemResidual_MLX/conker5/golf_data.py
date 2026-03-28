from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob

import mlx.core as mx
import numpy as np

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency in smoke environments
    spm = None


PARAMETER_GOLF_MAGIC = 20240520
PARAMETER_GOLF_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def _load_golf_shard(path: Path) -> np.ndarray:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            token_count = int(header[2])
            payload = np.frombuffer(blob[HEADER_BYTES:], dtype=np.uint16, count=token_count)
            if payload.size == token_count:
                return payload.astype(np.int32, copy=False)
    # Fallback for raw uint16 payloads if the header format changes.
    return np.frombuffer(blob, dtype=np.uint16).astype(np.int32, copy=False)


def _count_shard_tokens(path: Path) -> int:
    blob = path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            return int(header[2])
    return len(blob) // np.dtype(np.uint16).itemsize


class TokenStream:
    def __init__(self, pattern: str, dataset_name: str):
        self.pattern = pattern
        self.dataset_name = dataset_name
        self.files = tuple(Path(path) for path in sorted(glob.glob(pattern)))
        if not self.files:
            raise FileNotFoundError(f"No token shards matched pattern: {pattern}")
        self.file_idx = -1
        self.tokens = np.empty((0,), dtype=np.int32)
        self.pos = 0
        self.reset()

    def reset(self) -> None:
        self.file_idx = -1
        self.tokens = np.empty((0,), dtype=np.int32)
        self.pos = 0
        self.next_file()

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_golf_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            step = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + step])
            self.pos += step
            left -= step
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


@dataclass
class GolfTokenShardDataset:
    train_pattern: str
    test_pattern: str
    vocab_size: int
    tokenizer: str = "sp1024"
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        self.train_stream = TokenStream(self.train_pattern, "train")
        self.test_stream = TokenStream(self.test_pattern, "test")
        self.train_files = self.train_stream.files
        self.test_files = self.test_stream.files
        self.train_token_count = sum(_count_shard_tokens(path) for path in self.train_files)
        self.test_token_count = sum(_count_shard_tokens(path) for path in self.test_files)
        self.source_path = f"{self.train_pattern} :: {self.test_pattern}"
        self.test_tokens_per_byte = None
        self.test_bytes_per_token = None
        if self.tokenizer_path is not None and spm is not None:
            self.test_tokens_per_byte, self.test_bytes_per_token = _compute_tokens_per_byte(
                self.test_files,
                Path(self.tokenizer_path),
                self.vocab_size,
            )

    def batch(self, split: str, batch_size: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = batch_size * seq_len
        if usable <= 0:
            raise ValueError("batch_size * seq_len must be positive")
        stream = self.train_stream if split == "train" else self.test_stream
        chunk = stream.take(usable + 1)
        x = chunk[:-1].reshape(batch_size, seq_len)
        y = chunk[1:].reshape(batch_size, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def build_parameter_golf_dataset(data_root: str | Path, vocab_size: int = 1024) -> GolfTokenShardDataset:
    root = Path(data_root).expanduser()
    tokenizer_path = root.parents[1] / "tokenizers" / "fineweb_1024_bpe.model"
    return GolfTokenShardDataset(
        train_pattern=str(root / "fineweb_train_*.bin"),
        test_pattern=str(root / "fineweb_val_*.bin"),
        vocab_size=vocab_size,
        tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
    )


def _build_sentencepiece_luts(
    sp: "spm.SentencePieceProcessor", vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def _compute_tokens_per_byte(
    shard_paths: tuple[Path, ...], tokenizer_path: Path, vocab_size: int
) -> tuple[float, float]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = _build_sentencepiece_luts(sp, vocab_size)
    tokens = np.ascontiguousarray(np.concatenate([_load_golf_shard(path) for path in shard_paths], axis=0))
    prev_ids = tokens[:-1]
    tgt_ids = tokens[1:]
    bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16, copy=False)
    total_tokens = float(tgt_ids.size)
    total_bytes = float(bytes_np.astype(np.float64).sum())
    tokens_per_byte = total_tokens / total_bytes
    return tokens_per_byte, total_bytes / total_tokens
