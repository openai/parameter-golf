"""
BESE Tokenizer: Base-Efficient Subword Encoding

A novel tokenizer for parameter-constrained language models that uses:
1. Single-token codes for the 8 most frequent English letters (Huffman-inspired)
2. Two-token codes (group + position) for remaining letters, grouped by
   context distinguishability (inspired by QWERTY typewriter separation)
3. Case-insensitive encoding (model learns capitalization from context)

Designed for OpenAI Parameter Golf challenge.
Total vocabulary: 38 tokens (vs 1024 baseline)
Embedding savings: ~504,000 parameters

Origin: independently derived from T9 phone input and typewriter jam prevention,
later recognized as instances of Huffman coding and mutual information minimization.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    from .bese_constants import (
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        ENCODE_TABLE,
        EOS_ID,
        GROUP_START,
        GROUPS,
        OTHER_PUNCT_ID,
        PAD_ID,
        SINGLE_LETTERS,
        UNK_ID,
        VOCAB_SIZE,
    )
except ImportError:  # script: python tokenizer/bese_tokenizer.py
    from bese_constants import (
        BOS_ID,
        BYTES_PER_TOKEN,
        DECODE_TABLE,
        ENCODE_TABLE,
        EOS_ID,
        GROUP_START,
        GROUPS,
        OTHER_PUNCT_ID,
        PAD_ID,
        SINGLE_LETTERS,
        UNK_ID,
        VOCAB_SIZE,
    )


@dataclass(frozen=True)
class BESETokenizer:
    """Base-Efficient Subword Encoding tokenizer."""

    pad_id: int = PAD_ID
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    unk_id: int = UNK_ID

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    def encode(self, text: str) -> np.ndarray:
        """Encode text to token IDs. Case-insensitive."""
        tokens = []
        for ch in text:
            lower = ch.lower()
            if lower in ENCODE_TABLE:
                utf8_len = len(ch.encode("utf-8"))
                mapped_tokens = ENCODE_TABLE[lower]
                mapped_bytes = sum(BYTES_PER_TOKEN[t] for t in mapped_tokens)
                if utf8_len == mapped_bytes:
                    tokens.extend(mapped_tokens)
                else:
                    tokens.extend([OTHER_PUNCT_ID] * utf8_len)
            else:
                utf8_len = len(ch.encode("utf-8"))
                tokens.extend([OTHER_PUNCT_ID] * utf8_len)
        return np.array(tokens, dtype=np.uint16)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(text) for text in texts]

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text (best-effort, lossy on case)."""
        result = []
        i = 0
        while i < len(token_ids):
            tid = token_ids[i]
            if GROUP_START <= tid < GROUP_START + len(GROUPS):
                if i + 1 < len(token_ids):
                    key = (tid, token_ids[i + 1])
                    result.append(DECODE_TABLE.get(key, "?"))
                    i += 2
                    continue
            key = (tid,)
            if key in DECODE_TABLE:
                result.append(DECODE_TABLE[key])
            elif tid not in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
                result.append("?")
            i += 1
        return "".join(result)

    def count_original_bytes(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokenizer_type": "bese",
            "vocab_size": self.vocab_size,
            "config": {
                "single_letters": SINGLE_LETTERS,
                "groups": GROUPS,
                "pad_id": self.pad_id,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "unk_id": self.unk_id,
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def get_bytes_per_token_lut():
    """Return the bytes-per-token lookup table for BPB calculation."""
    return BYTES_PER_TOKEN.copy()


if __name__ == "__main__":
    tok = BESETokenizer()
    for text in ["The cat sat on the mat.", "Hello world!", "Quick brown fox."]:
        enc = tok.encode(text)
        dec = tok.decode_tokens(enc.tolist())
        bpt = sum(BYTES_PER_TOKEN[t] for t in enc)
        utf8 = len(text.encode("utf-8"))
        print(
            f'"{text}" -> {len(enc)} tokens, bytes: {bpt}/{utf8} '
            f'{"OK" if bpt == utf8 else "MISMATCH"}'
        )
        print(f'  decoded: "{dec}"')
