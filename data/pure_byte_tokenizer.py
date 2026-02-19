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


@dataclass(frozen=True)
class PureByteTokenizerConfig:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    unk_id: int = 3
    byte_offset: int = 4
    byte_count: int = 256

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + self.byte_count


class PureByteTokenizer:
    def __init__(self, cfg: PureByteTokenizerConfig):
        self.cfg = cfg

    @property
    def pad_id(self) -> int:
        return self.cfg.pad_id

    @property
    def bos_id(self) -> int:
        return self.cfg.bos_id

    @property
    def eos_id(self) -> int:
        return self.cfg.eos_id

    @property
    def unk_id(self) -> int:
        return self.cfg.unk_id

    @property
    def vocab_size(self) -> int:
        return self.cfg.vocab_size

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8", errors="replace")
        return [self.cfg.byte_offset + b for b in data]

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
            if self.cfg.byte_offset <= t < self.cfg.byte_offset + self.cfg.byte_count:
                out.append(t - self.cfg.byte_offset)
            elif not skip_special_tokens:
                out.extend(b"?")
        return out.decode("utf-8", errors="replace")

    def to_json_dict(self) -> dict:
        return {
            "tokenizer_type": "pure_byte",
            "config": asdict(self.cfg),
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
        cfg = PureByteTokenizerConfig(
            pad_id=int(cfg_raw.get("pad_id", 0)),
            bos_id=int(cfg_raw.get("bos_id", 1)),
            eos_id=int(cfg_raw.get("eos_id", 2)),
            unk_id=int(cfg_raw.get("unk_id", 3)),
            byte_offset=int(cfg_raw.get("byte_offset", 4)),
            byte_count=int(cfg_raw.get("byte_count", 256)),
        )
        return cls(cfg)


def default_pure_byte_tokenizer() -> PureByteTokenizer:
    return PureByteTokenizer(PureByteTokenizerConfig())

