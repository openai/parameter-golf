from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import glob

import numpy as np

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency
    spm = None


PARAMETER_GOLF_MAGIC = 20240520
PARAMETER_GOLF_VERSION = 1
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * np.dtype(np.int32).itemsize


def load_golf_shard(path: str | Path) -> np.ndarray:
    shard_path = Path(path)
    blob = shard_path.read_bytes()
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
                return payload.astype(np.int64, copy=False)
    return np.frombuffer(blob, dtype=np.uint16).astype(np.int64, copy=False)


def count_golf_tokens(path: str | Path) -> int:
    shard_path = Path(path)
    blob = shard_path.read_bytes()
    if len(blob) >= HEADER_BYTES:
        header = np.frombuffer(blob[:HEADER_BYTES], dtype=np.int32, count=HEADER_INTS)
        if (
            header.size >= 3
            and int(header[0]) == PARAMETER_GOLF_MAGIC
            and int(header[1]) == PARAMETER_GOLF_VERSION
        ):
            return int(header[2])
    return len(blob) // np.dtype(np.uint16).itemsize


def discover_shards(patterns: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(patterns, (str, Path)):
        candidates = [patterns]
    else:
        candidates = list(patterns)
    paths: list[Path] = []
    for candidate in candidates:
        matches = sorted(glob.glob(str(candidate)))
        paths.extend(Path(match) for match in matches)
    unique = tuple(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"no shards matched: {patterns}")
    return unique


def load_golf_tokens(
    paths: Sequence[str | Path],
    *,
    max_tokens: int | None = None,
) -> np.ndarray:
    shards = [load_golf_shard(path) for path in paths]
    if not shards:
        return np.zeros((0,), dtype=np.int64)
    tokens = np.concatenate(shards, axis=0)
    if max_tokens is not None:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        tokens = tokens[:max_tokens]
    return tokens.astype(np.int64, copy=False)


def write_golf_shard(path: str | Path, tokens: Iterable[int], *, with_header: bool = True) -> Path:
    shard_path = Path(path)
    values = np.asarray(list(tokens), dtype=np.uint16)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if with_header:
        header = np.zeros((HEADER_INTS,), dtype=np.int32)
        header[0] = PARAMETER_GOLF_MAGIC
        header[1] = PARAMETER_GOLF_VERSION
        header[2] = int(values.size)
        payload = header.tobytes() + values.tobytes()
    else:
        payload = values.tobytes()
    shard_path.write_bytes(payload)
    return shard_path


def compute_sentencepiece_bytes_per_token(
    tokens: np.ndarray,
    tokenizer_model: str | Path,
    *,
    vocab_size: int,
) -> float:
    if spm is None:
        raise ImportError("sentencepiece is required to compute bytes-per-token from a tokenizer model")
    token_array = np.asarray(tokens, dtype=np.int64).reshape(-1)
    if token_array.size < 2:
        return 1.0

    sp = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_model)))
    table_size = max(int(sp.vocab_size()), int(vocab_size))
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
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

    prev_ids = token_array[:-1]
    tgt_ids = token_array[1:]
    if tgt_ids.size == 0:
        return 1.0
    if int(np.min(prev_ids)) < 0 or int(np.max(prev_ids)) >= table_size:
        raise ValueError("tokens exceed tokenizer vocabulary table")
    if int(np.min(tgt_ids)) < 0 or int(np.max(tgt_ids)) >= table_size:
        raise ValueError("tokens exceed tokenizer vocabulary table")
    byte_counts = base_bytes_lut[tgt_ids].astype(np.int32, copy=True)
    byte_counts += (
        has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
    ).astype(np.int32, copy=False)
    total_bytes = float(np.sum(byte_counts, dtype=np.float64))
    total_tokens = float(tgt_ids.size)
    if total_bytes <= 0.0 or total_tokens <= 0.0:
        return 1.0
    return total_bytes / total_tokens


__all__ = [
    "PARAMETER_GOLF_MAGIC",
    "PARAMETER_GOLF_VERSION",
    "HEADER_INTS",
    "HEADER_BYTES",
    "count_golf_tokens",
    "compute_sentencepiece_bytes_per_token",
    "discover_shards",
    "load_golf_shard",
    "load_golf_tokens",
    "write_golf_shard",
]
