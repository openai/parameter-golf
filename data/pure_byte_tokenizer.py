"""Pure byte-level tokenizer utilities.

Vocabulary layout (fixed):
- 0: <pad>
- 1: <bos>
- 2: <eos>
- 3: <unk>
- 4..259: raw byte values 0..255
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PureByteTokenizer:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + self.byte_count

    def encode(self, text: str) -> np.ndarray:
        data = text.encode("utf-8", errors="replace")
        return np.frombuffer(data, dtype=np.uint8).astype(np.uint16, copy=False) + self.byte_offset

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        out = bytearray()
        for t in token_ids:
            if t in special:
                if skip_special_tokens:
                    continue
                if t == self.unk_id:
                    out.extend(b"?")
                continue
            if self.byte_offset <= t < self.byte_offset + self.byte_count:
                out.append(t - self.byte_offset)
            elif not skip_special_tokens:
                out.extend(b"?")
        return out.decode("utf-8", errors="replace")

    def to_json_dict(self) -> dict:
        return {
            "tokenizer_type": "pure_byte",
            "config": asdict(self),
            "vocab_size": self.vocab_size,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "PureByteTokenizer":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("tokenizer_type") != "pure_byte":
            raise ValueError(f"Unsupported tokenizer_type: {payload.get('tokenizer_type')!r}")
        cfg_raw = payload.get("config", {})
        return cls(
            pad_id=int(cfg_raw.get("pad_id", 0)),
            bos_id=int(cfg_raw.get("bos_id", 1)),
            eos_id=int(cfg_raw.get("eos_id", 2)),
            unk_id=int(cfg_raw.get("unk_id", 3)),
            byte_offset=int(cfg_raw.get("byte_offset", 4)),
            byte_count=int(cfg_raw.get("byte_count", 256)),
        )


def default_pure_byte_tokenizer() -> PureByteTokenizer:
    return PureByteTokenizer()
