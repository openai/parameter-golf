"""
Shared BESE alphabet constants and lookup tables (38-token base vocabulary).

Used by bese_tokenizer.py and bese_bpe_tokenizer.py — single source of truth.
"""

from __future__ import annotations

import numpy as np

# --- Special tokens ---
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Single-token letters: 8 most frequent in English (e=4 ... r=11)
SINGLE_LETTERS = "etaoinsr"
SINGLE_LETTER_START = 4

# Key groups (group token = 0 bytes; position completes character)
GROUP_START = 12
GROUPS = [
    "jmfg",  # Group 12
    "cqky",  # Group 13
    "zulv",  # Group 14
    "bxhw",  # Group 15
    "dp",    # Group 16
]

POS_START = 17

SPACE_ID = 21
PERIOD_ID = 22
COMMA_ID = 23
NEWLINE_ID = 24
QUESTION_ID = 25
QUOTE_ID = 26
OTHER_PUNCT_ID = 27
DIGIT_START = 28

BASE_VOCAB_SIZE = 38
VOCAB_SIZE = BASE_VOCAB_SIZE  # alias for base-only tokenizer


def build_encode_table() -> dict[str, list[int]]:
    """Character -> base token id(s)."""
    table: dict[str, list[int]] = {}
    for i, ch in enumerate(SINGLE_LETTERS):
        table[ch] = [SINGLE_LETTER_START + i]
    for gi, group in enumerate(GROUPS):
        group_token = GROUP_START + gi
        for pi, ch in enumerate(group):
            pos_token = POS_START + pi
            table[ch] = [group_token, pos_token]
    table[" "] = [SPACE_ID]
    table["."] = [PERIOD_ID]
    table[","] = [COMMA_ID]
    table["\n"] = [NEWLINE_ID]
    table["?"] = [QUESTION_ID]
    for ch in ["'", '"', "\u2018", "\u2019", "\u201c", "\u201d"]:
        table[ch] = [QUOTE_ID]
    for d in range(10):
        table[str(d)] = [DIGIT_START + d]
    return table


def build_decode_table() -> dict[tuple[int, ...], str]:
    """Base token sequence -> single character (best-effort)."""
    table: dict[tuple[int, ...], str] = {}
    for i, ch in enumerate(SINGLE_LETTERS):
        table[(SINGLE_LETTER_START + i,)] = ch
    for gi, group in enumerate(GROUPS):
        group_token = GROUP_START + gi
        for pi, ch in enumerate(group):
            pos_token = POS_START + pi
            table[(group_token, pos_token)] = ch
    table[(SPACE_ID,)] = " "
    table[(PERIOD_ID,)] = "."
    table[(COMMA_ID,)] = ","
    table[(NEWLINE_ID,)] = "\n"
    table[(QUESTION_ID,)] = "?"
    table[(QUOTE_ID,)] = "'"
    table[(OTHER_PUNCT_ID,)] = "?"
    for d in range(10):
        table[(DIGIT_START + d,)] = str(d)
    return table


def build_bytes_per_token() -> np.ndarray:
    """UTF-8 bytes each base token represents (BPB-critical)."""
    bpt = np.zeros(BASE_VOCAB_SIZE, dtype=np.int16)
    for i in range(len(SINGLE_LETTERS)):
        bpt[SINGLE_LETTER_START + i] = 1
    for i in range(4):
        bpt[POS_START + i] = 1
    for tid in (
        SPACE_ID,
        PERIOD_ID,
        COMMA_ID,
        NEWLINE_ID,
        QUESTION_ID,
        QUOTE_ID,
        OTHER_PUNCT_ID,
    ):
        bpt[tid] = 1
    for d in range(10):
        bpt[DIGIT_START + d] = 1
    return bpt


# Eager singletons (import cost is tiny; avoids recomputation)
ENCODE_TABLE = build_encode_table()
DECODE_TABLE = build_decode_table()
BYTES_PER_TOKEN = build_bytes_per_token()
