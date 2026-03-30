from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .codecs import ensure_tokens


@dataclass(frozen=True)
class ByteSequenceDataset:
    sequences: tuple[np.ndarray, ...]

    @classmethod
    def from_items(cls, items: Iterable[str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]]) -> "ByteSequenceDataset":
        return cls(tuple(ensure_tokens(item) for item in items))

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path], encoding: str = "utf-8") -> "ByteSequenceDataset":
        sequences = []
        for path in paths:
            text = Path(path).read_text(encoding=encoding)
            sequences.append(ensure_tokens(text))
        return cls(tuple(sequences))

    def concatenated(self, separator: bytes = b"\n") -> np.ndarray:
        if not self.sequences:
            return np.zeros(0, dtype=np.uint8)
        joined = separator.join(bytes(seq.tolist()) for seq in self.sequences)
        return np.frombuffer(joined, dtype=np.uint8).copy()

