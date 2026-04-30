"""Lossless capitalization pre-encoding helpers.

This module provides a narrow, reversible transform that only touches
ASCII capital letters `A-Z`. Each uppercase ASCII letter is rewritten as
`<sentinel><lowercase>`, where `sentinel` is a private-use Unicode
character that is escaped by doubling if it appears literally in the
input text.

Example with the default sentinel `\\uE000`:

    "The NASA Launch" -> "\\uE000the \\uE000n\\uE000a\\uE000s\\uE000a \\uE000launch"

The transform is intentionally simple for v1:

- lowercase ASCII letters are unchanged
- uppercase ASCII letters become sentinel + lowercase letter
- non-ASCII characters are left untouched
- literal sentinel characters are escaped as sentinel + sentinel

This makes the transform exactly invertible while allowing a downstream
tokenizer to reuse lowercase subwords across case variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

LOSSLESS_CAPS_V1 = "lossless_caps_v1"
LOSSLESS_CAPS_V2 = "lossless_caps_v2"
LOSSLESS_CAPS_V3 = "lossless_caps_v3"
LOSSLESS_CAPS_V4 = "lossless_caps_v4"
LOSSLESS_CAPS_V5 = "lossless_caps_v5"
LOSSLESS_CAPS_V6 = "lossless_caps_v6"
LOSSLESS_CAPS_V7 = "lossless_caps_v7"
LOSSLESS_CAPS_CASEOPS_V1 = "lossless_caps_caseops_v1"
IDENTITY = "identity"
DEFAULT_SENTINEL = "\uE000"
DEFAULT_V2_TITLE = "\uE001"
DEFAULT_V2_ALLCAPS = "\uE002"
DEFAULT_V2_CAPNEXT = "\uE003"
DEFAULT_V2_ESC = "\uE004"
DEFAULT_V5_TITLE_MIN_LEN = 7
DEFAULT_V6_ALLCAPS_MIN_LEN = 3
DEFAULT_V7_ALLCAPS_MIN_LEN = 4


class LosslessCapsError(ValueError):
    """Raised when a transformed string is malformed."""


def _is_ascii_upper(ch: str) -> bool:
    return "A" <= ch <= "Z"


def _is_ascii_lower(ch: str) -> bool:
    return "a" <= ch <= "z"


def _is_ascii_alpha(ch: str) -> bool:
    return _is_ascii_lower(ch) or _is_ascii_upper(ch)


def _validate_distinct_single_chars(*chars: str) -> None:
    if any(len(ch) != 1 for ch in chars):
        raise ValueError("all control characters must be exactly one character")
    if len(set(chars)) != len(chars):
        raise ValueError("control characters must be distinct")


def encode_lossless_caps_v1(text: str, *, sentinel: str = DEFAULT_SENTINEL) -> str:
    """Encode ASCII capitals reversibly using a one-character sentinel."""
    if len(sentinel) != 1:
        raise ValueError("sentinel must be exactly one character")
    out: list[str] = []
    for ch in text:
        if ch == sentinel:
            out.append(sentinel)
            out.append(sentinel)
        elif _is_ascii_upper(ch):
            out.append(sentinel)
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def decode_lossless_caps_v1(text: str, *, sentinel: str = DEFAULT_SENTINEL) -> str:
    """Decode the `lossless_caps_v1` transform back to the original text."""
    if len(sentinel) != 1:
        raise ValueError("sentinel must be exactly one character")
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != sentinel:
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            raise LosslessCapsError("dangling capitalization sentinel at end of string")
        nxt = text[i + 1]
        if nxt == sentinel:
            out.append(sentinel)
        elif _is_ascii_lower(nxt):
            out.append(nxt.upper())
        else:
            raise LosslessCapsError(
                f"invalid sentinel escape sequence {sentinel + nxt!r}; "
                "expected doubled sentinel or sentinel + lowercase ASCII letter"
            )
        i += 2
    return "".join(out)


def encode_lossless_caps_v2(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    capnext: str = DEFAULT_V2_CAPNEXT,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Encode ASCII word capitalization with cheap word-level markers.

    Rules over maximal ASCII alphabetic runs:
    - lowercase words stay unchanged
    - TitleCase words become `title + lowercase(word)`
    - ALLCAPS words become `allcaps + lowercase(word)`
    - mixed-case words use:
      - optional `title` when the first letter is uppercase
      - `capnext + lowercase(letter)` for subsequent uppercase letters
    - literal control characters are escaped as `esc + literal`
    """
    _validate_distinct_single_chars(title, allcaps, capnext, esc)
    controls = {title, allcaps, capnext, esc}
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in controls:
            out.append(esc)
            out.append(ch)
            i += 1
            continue
        if not _is_ascii_alpha(ch):
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and _is_ascii_alpha(text[j]):
            j += 1
        word = text[i:j]
        lower_word = word.lower()

        if word.islower():
            out.append(word)
        elif len(word) >= 2 and word.isupper():
            out.append(allcaps)
            out.append(lower_word)
        elif _is_ascii_upper(word[0]) and word[1:].islower():
            out.append(title)
            out.append(lower_word)
        else:
            if _is_ascii_upper(word[0]):
                out.append(title)
            out.append(lower_word[0])
            for orig_ch, lower_ch in zip(word[1:], lower_word[1:], strict=True):
                if _is_ascii_upper(orig_ch):
                    out.append(capnext)
                out.append(lower_ch)
        i = j
    return "".join(out)


def decode_lossless_caps_v2(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    capnext: str = DEFAULT_V2_CAPNEXT,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v2` transform back to the original text."""
    _validate_distinct_single_chars(title, allcaps, capnext, esc)
    out: list[str] = []
    pending_escape = False
    pending_word_mode: str | None = None
    active_allcaps = False
    pending_capnext = False
    in_ascii_word = False

    for ch in text:
        if pending_escape:
            if pending_word_mode is not None and not _is_ascii_alpha(ch):
                raise LosslessCapsError("escaped control char cannot satisfy pending word capitalization mode")
            out.append(ch)
            pending_escape = False
            if _is_ascii_alpha(ch):
                in_ascii_word = True
            else:
                in_ascii_word = False
                active_allcaps = False
            continue

        if ch == esc:
            pending_escape = True
            continue
        if ch == title:
            if pending_word_mode is not None or in_ascii_word or pending_capnext:
                raise LosslessCapsError("invalid title marker placement")
            pending_word_mode = "title"
            continue
        if ch == allcaps:
            if pending_word_mode is not None or in_ascii_word or pending_capnext:
                raise LosslessCapsError("invalid allcaps marker placement")
            pending_word_mode = "allcaps"
            continue
        if ch == capnext:
            if pending_capnext:
                raise LosslessCapsError("duplicate capnext marker")
            pending_capnext = True
            continue

        if _is_ascii_alpha(ch):
            at_word_start = not in_ascii_word
            if at_word_start:
                if pending_word_mode == "allcaps":
                    out.append(ch.upper())
                    active_allcaps = True
                elif pending_word_mode == "title":
                    out.append(ch.upper())
                elif pending_capnext:
                    out.append(ch.upper())
                else:
                    out.append(ch)
                pending_word_mode = None
                pending_capnext = False
                in_ascii_word = True
                continue

            if pending_word_mode is not None:
                raise LosslessCapsError("word capitalization marker leaked into the middle of a word")
            if active_allcaps:
                out.append(ch.upper())
            elif pending_capnext:
                out.append(ch.upper())
            else:
                out.append(ch)
            pending_capnext = False
            continue

        if pending_word_mode is not None or pending_capnext:
            raise LosslessCapsError("capitalization marker not followed by an ASCII letter")
        out.append(ch)
        in_ascii_word = False
        active_allcaps = False

    if pending_escape:
        raise LosslessCapsError("dangling escape marker at end of string")
    if pending_word_mode is not None or pending_capnext:
        raise LosslessCapsError("dangling capitalization marker at end of string")
    return "".join(out)


def encode_lossless_caps_v3(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Encode only common word-level capitalization patterns.

    Rules over maximal ASCII alphabetic runs:
    - lowercase words stay unchanged
    - TitleCase words become `title + lowercase(word)`
    - ALLCAPS words become `allcaps + lowercase(word)`
    - all other mixed-case words are left unchanged
    - literal control characters are escaped as `esc + literal`
    """
    _validate_distinct_single_chars(title, allcaps, esc)
    controls = {title, allcaps, esc}
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in controls:
            out.append(esc)
            out.append(ch)
            i += 1
            continue
        if not _is_ascii_alpha(ch):
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n and _is_ascii_alpha(text[j]):
            j += 1
        word = text[i:j]

        if word.islower():
            out.append(word)
        elif len(word) >= 2 and word.isupper():
            out.append(allcaps)
            out.append(word.lower())
        elif _is_ascii_upper(word[0]) and word[1:].islower():
            out.append(title)
            out.append(word.lower())
        else:
            out.append(word)
        i = j
    return "".join(out)


def decode_lossless_caps_v3(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v3` transform back to the original text."""
    _validate_distinct_single_chars(title, allcaps, esc)
    out: list[str] = []
    pending_escape = False
    pending_word_mode: str | None = None
    active_allcaps = False
    in_ascii_word = False

    for ch in text:
        if pending_escape:
            if pending_word_mode is not None and not _is_ascii_alpha(ch):
                raise LosslessCapsError("escaped control char cannot satisfy pending word capitalization mode")
            out.append(ch)
            pending_escape = False
            if _is_ascii_alpha(ch):
                in_ascii_word = True
            else:
                in_ascii_word = False
                active_allcaps = False
            continue

        if ch == esc:
            pending_escape = True
            continue
        if ch == title:
            if pending_word_mode is not None or in_ascii_word:
                raise LosslessCapsError("invalid title marker placement")
            pending_word_mode = "title"
            continue
        if ch == allcaps:
            if pending_word_mode is not None or in_ascii_word:
                raise LosslessCapsError("invalid allcaps marker placement")
            pending_word_mode = "allcaps"
            continue

        if _is_ascii_alpha(ch):
            at_word_start = not in_ascii_word
            if at_word_start:
                if pending_word_mode == "allcaps":
                    out.append(ch.upper())
                    active_allcaps = True
                elif pending_word_mode == "title":
                    out.append(ch.upper())
                else:
                    out.append(ch)
                pending_word_mode = None
                in_ascii_word = True
                continue

            if pending_word_mode is not None:
                raise LosslessCapsError("word capitalization marker leaked into the middle of a word")
            out.append(ch.upper() if active_allcaps else ch)
            continue

        if pending_word_mode is not None:
            raise LosslessCapsError("capitalization marker not followed by an ASCII letter")
        out.append(ch)
        in_ascii_word = False
        active_allcaps = False

    if pending_escape:
        raise LosslessCapsError("dangling escape marker at end of string")
    if pending_word_mode is not None:
        raise LosslessCapsError("dangling capitalization marker at end of string")
    return "".join(out)


def encode_lossless_caps_v4(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Encode only ALLCAPS ASCII words, leaving all other case untouched."""
    _validate_distinct_single_chars(allcaps, esc)
    controls = {allcaps, esc}
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in controls:
            out.append(esc)
            out.append(ch)
            i += 1
            continue
        if not _is_ascii_alpha(ch):
            out.append(ch)
            i += 1
            continue
        j = i + 1
        while j < n and _is_ascii_alpha(text[j]):
            j += 1
        word = text[i:j]
        if len(word) >= 2 and word.isupper():
            out.append(allcaps)
            out.append(word.lower())
        else:
            out.append(word)
        i = j
    return "".join(out)


def decode_lossless_caps_v4(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v4` transform back to the original text."""
    _validate_distinct_single_chars(allcaps, esc)
    out: list[str] = []
    pending_escape = False
    pending_allcaps = False
    in_ascii_word = False
    active_allcaps = False

    for ch in text:
        if pending_escape:
            if pending_allcaps and not _is_ascii_alpha(ch):
                raise LosslessCapsError("escaped control char cannot satisfy pending allcaps mode")
            out.append(ch)
            pending_escape = False
            if _is_ascii_alpha(ch):
                in_ascii_word = True
            else:
                in_ascii_word = False
                active_allcaps = False
            continue

        if ch == esc:
            pending_escape = True
            continue
        if ch == allcaps:
            if pending_allcaps or in_ascii_word:
                raise LosslessCapsError("invalid allcaps marker placement")
            pending_allcaps = True
            continue

        if _is_ascii_alpha(ch):
            if not in_ascii_word:
                active_allcaps = pending_allcaps
                pending_allcaps = False
                in_ascii_word = True
            out.append(ch.upper() if active_allcaps else ch)
            continue

        if pending_allcaps:
            raise LosslessCapsError("allcaps marker not followed by an ASCII letter")
        out.append(ch)
        in_ascii_word = False
        active_allcaps = False

    if pending_escape:
        raise LosslessCapsError("dangling escape marker at end of string")
    if pending_allcaps:
        raise LosslessCapsError("dangling allcaps marker at end of string")
    return "".join(out)


def encode_lossless_caps_v5(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
    title_min_len: int = DEFAULT_V5_TITLE_MIN_LEN,
