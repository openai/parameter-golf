"""
Shared BESE alphabet constants and lookup tables (40-token base vocabulary).

Used by bese_tokenizer.py and bese_bpe_tokenizer.py — single source of truth.

v3: Reordered groups by aggregate frequency (most common group → lowest ID).
    Group 15 (ufbz, 6.57%) > Group 16 (cwvj, 6.35%) > Group 17 (mykx, 5.35%) > Group 18 (gpq, 4.0%).
    The group token ID itself now encodes frequency tier information.

v2: Promoted h, d, l to single-token letters (8→11 singles).
    Regrouped remaining 15 letters into 4 groups (5→4 groups).
    Reordered positions within groups by frequency (most freq → P1).
"""

from __future__ import annotations

import numpy as np

# --- Special tokens ---
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Single-token letters: 11 most frequent in English (e=4 ... l=14)
SINGLE_LETTERS = "etaoinsrhdl"
SINGLE_LETTER_START = 4

# Key groups (group token = 0 bytes; position completes character)
# Ordered by frequency within each group (most frequent → P1).
# Letters that commonly co-occur in English bigrams are placed in
# DIFFERENT groups to give BPE cleaner merge patterns.
GROUP_START = 15
GROUPS = [
    "ufbz",  # Group 15: u(2.8%) f(2.2%) b(1.5%) z(0.07%) — 6.57% aggregate
    "cwvj",  # Group 16: c(2.8%) w(2.4%) v(1.0%) j(0.15%) — 6.35% aggregate
    "mykx",  # Group 17: m(2.4%) y(2.0%) k(0.8%) x(0.15%) — 5.35% aggregate
    "gpq",   # Group 18: g(2.0%) p(1.9%) q(0.10%)          — 4.00% aggregate
]

POS_START = 19

SPACE_ID = 23
PERIOD_ID = 24
COMMA_ID = 25
NEWLINE_ID = 26
QUESTION_ID = 27
QUOTE_ID = 28
OTHER_PUNCT_ID = 29
DIGIT_START = 30

BASE_VOCAB_SIZE = 40
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
