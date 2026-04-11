from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _is_sequence_of_ints(value: object) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and np.issubdtype(value.dtype, np.integer)
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, int) for item in value)


def ensure_tokens(value: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    if isinstance(value, str):
        return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.uint8).copy()
    if isinstance(value, bytearray):
        return np.frombuffer(bytes(value), dtype=np.uint8).copy()
    if isinstance(value, memoryview):
        return np.frombuffer(value.tobytes(), dtype=np.uint8).copy()
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError("token arrays must be one-dimensional")
        if not np.issubdtype(value.dtype, np.integer):
            raise TypeError("token arrays must contain integers")
        return value.astype(np.int64, copy=True)
    if _is_sequence_of_ints(value):
        return np.asarray(list(value), dtype=np.int64)
    raise TypeError(f"Unsupported token input type: {type(value)!r}")


def ensure_byte_tokens(value: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    tokens = ensure_tokens(value)
    if tokens.size and (int(np.min(tokens)) < 0 or int(np.max(tokens)) > 255):
        raise ValueError("byte tokens must lie in [0, 255]")
    return tokens.astype(np.uint8, copy=False)


class ByteCodec:
    @staticmethod
    def encode_text(text: str, encoding: str = "utf-8") -> np.ndarray:
        return np.frombuffer(text.encode(encoding), dtype=np.uint8).copy()

    @staticmethod
    def decode_text(tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int], encoding: str = "utf-8") -> str:
        return bytes(ensure_byte_tokens(tokens).tolist()).decode(encoding, errors="replace")

    @staticmethod
    def encode_bytes(payload: bytes | bytearray | memoryview) -> np.ndarray:
        return ensure_tokens(payload)

    @staticmethod
    def decode_bytes(tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> bytes:
        return bytes(ensure_byte_tokens(tokens).tolist())
