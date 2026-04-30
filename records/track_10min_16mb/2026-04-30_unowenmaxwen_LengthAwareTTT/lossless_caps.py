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
) -> str:
    """Encode ALLCAPS words and only sufficiently long TitleCase words."""
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
        if len(word) >= 2 and word.isupper():
            out.append(allcaps)
            out.append(word.lower())
        elif len(word) >= title_min_len and _is_ascii_upper(word[0]) and word[1:].islower():
            out.append(title)
            out.append(word.lower())
        else:
            out.append(word)
        i = j
    return "".join(out)


def decode_lossless_caps_v5(
    text: str,
    *,
    title: str = DEFAULT_V2_TITLE,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v5` transform back to the original text."""
    return decode_lossless_caps_v3(text, title=title, allcaps=allcaps, esc=esc)


def encode_lossless_caps_v6(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
    allcaps_min_len: int = DEFAULT_V6_ALLCAPS_MIN_LEN,
) -> str:
    """Encode only ALLCAPS words with length >= allcaps_min_len."""
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
        if len(word) >= allcaps_min_len and word.isupper():
            out.append(allcaps)
            out.append(word.lower())
        else:
            out.append(word)
        i = j
    return "".join(out)


def decode_lossless_caps_v6(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v6` transform back to the original text."""
    return decode_lossless_caps_v4(text, allcaps=allcaps, esc=esc)


def encode_lossless_caps_v7(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
    allcaps_min_len: int = DEFAULT_V7_ALLCAPS_MIN_LEN,
) -> str:
    """Encode only ALLCAPS words with length >= 4."""
    return encode_lossless_caps_v6(
        text,
        allcaps=allcaps,
        esc=esc,
        allcaps_min_len=allcaps_min_len,
    )


def decode_lossless_caps_v7(
    text: str,
    *,
    allcaps: str = DEFAULT_V2_ALLCAPS,
    esc: str = DEFAULT_V2_ESC,
) -> str:
    """Decode the `lossless_caps_v7` transform back to the original text."""
    return decode_lossless_caps_v6(text, allcaps=allcaps, esc=esc)


def get_text_transform(name: str | None) -> Callable[[str], str]:
    """Return the forward text transform for the given config name."""
    normalized = IDENTITY if name in {None, "", IDENTITY} else str(name)
    if normalized == IDENTITY:
        return lambda text: text
    if normalized == LOSSLESS_CAPS_V1:
        return encode_lossless_caps_v1
    if normalized == LOSSLESS_CAPS_V2:
        return encode_lossless_caps_v2
    if normalized == LOSSLESS_CAPS_V3:
        return encode_lossless_caps_v3
    if normalized == LOSSLESS_CAPS_V4:
        return encode_lossless_caps_v4
    if normalized == LOSSLESS_CAPS_V5:
        return encode_lossless_caps_v5
    if normalized == LOSSLESS_CAPS_V6:
        return encode_lossless_caps_v6
    if normalized == LOSSLESS_CAPS_V7:
        return encode_lossless_caps_v7
    if normalized == LOSSLESS_CAPS_CASEOPS_V1:
        return encode_lossless_caps_v2
    raise ValueError(f"unsupported text_transform={name!r}")


def get_text_inverse_transform(name: str | None) -> Callable[[str], str]:
    """Return the inverse transform for the given config name."""
    normalized = IDENTITY if name in {None, "", IDENTITY} else str(name)
    if normalized == IDENTITY:
        return lambda text: text
    if normalized == LOSSLESS_CAPS_V1:
        return decode_lossless_caps_v1
    if normalized == LOSSLESS_CAPS_V2:
        return decode_lossless_caps_v2
    if normalized == LOSSLESS_CAPS_V3:
        return decode_lossless_caps_v3
    if normalized == LOSSLESS_CAPS_V4:
        return decode_lossless_caps_v4
    if normalized == LOSSLESS_CAPS_V5:
        return decode_lossless_caps_v5
    if normalized == LOSSLESS_CAPS_V6:
        return decode_lossless_caps_v6
    if normalized == LOSSLESS_CAPS_V7:
        return decode_lossless_caps_v7
    if normalized == LOSSLESS_CAPS_CASEOPS_V1:
        return decode_lossless_caps_v2
    raise ValueError(f"unsupported text_transform={name!r}")


def normalize_text_transform_name(name: str | None) -> str:
    """Normalize empty/None transform names to the identity transform."""
    return IDENTITY if name in {None, "", IDENTITY} else str(name)


def get_text_transform_control_symbols(name: str | None) -> list[str]:
    """Return reserved control symbols used by a transform, if any."""
    normalized = normalize_text_transform_name(name)
    if normalized == IDENTITY:
        return []
    if normalized == LOSSLESS_CAPS_V1:
        return [DEFAULT_SENTINEL]
    if normalized == LOSSLESS_CAPS_V2:
        return [DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC]
    if normalized == LOSSLESS_CAPS_CASEOPS_V1:
        return [DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_CAPNEXT, DEFAULT_V2_ESC]
    if normalized in {LOSSLESS_CAPS_V3, LOSSLESS_CAPS_V5}:
        return [DEFAULT_V2_TITLE, DEFAULT_V2_ALLCAPS, DEFAULT_V2_ESC]
    if normalized in {LOSSLESS_CAPS_V4, LOSSLESS_CAPS_V6, LOSSLESS_CAPS_V7}:
        return [DEFAULT_V2_ALLCAPS, DEFAULT_V2_ESC]
    raise ValueError(f"unsupported text_transform={name!r}")


def infer_text_transform_from_manifest(tokenizer_path: str | Path) -> str:
    """Best-effort lookup of a tokenizer's text transform from a local manifest."""
    tokenizer_path = Path(tokenizer_path).expanduser().resolve()
    manifest_candidates = [
        tokenizer_path.parent.parent / "manifest.json",
        tokenizer_path.parent / "manifest.json",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tokenizers = payload.get("tokenizers")
        if not isinstance(tokenizers, list):
            continue
        for tokenizer_meta in tokenizers:
            if not isinstance(tokenizer_meta, dict):
                continue
            model_path = tokenizer_meta.get("model_path") or tokenizer_meta.get("path")
            if not model_path:
                continue
            candidate = (manifest_path.parent / str(model_path)).resolve()
            if candidate == tokenizer_path:
                return normalize_text_transform_name(tokenizer_meta.get("text_transform"))
    return IDENTITY


def surface_piece_original_byte_counts(
    surfaces: Iterable[str],
    *,
    text_transform_name: str | None = None,
    sentinel: str = DEFAULT_SENTINEL,
) -> list[int]:
    """Return exact original UTF-8 byte counts contributed by each surface piece.

    `surfaces` must be the exact decoded text fragments emitted by SentencePiece
    in order, e.g. `piece.surface` from `encode_as_immutable_proto`.
    """
    normalized = normalize_text_transform_name(text_transform_name)
    if normalized == IDENTITY:
        return [len(surface.encode("utf-8")) for surface in surfaces]
    if normalized == LOSSLESS_CAPS_V1:
        if len(sentinel) != 1:
            raise ValueError("sentinel must be exactly one character")
        sentinel_bytes = len(sentinel.encode("utf-8"))
        pending_sentinel = False
        counts: list[int] = []
        for surface in surfaces:
            piece_bytes = 0
            for ch in surface:
                if pending_sentinel:
                    if ch == sentinel:
                        piece_bytes += sentinel_bytes
                    elif _is_ascii_lower(ch):
                        piece_bytes += 1
                    else:
                        raise LosslessCapsError(
                            f"invalid continuation {ch!r} after capitalization sentinel"
                        )
                    pending_sentinel = False
                    continue
                if ch == sentinel:
                    pending_sentinel = True
                else:
                    piece_bytes += len(ch.encode("utf-8"))
            counts.append(piece_bytes)
        if pending_sentinel:
            raise LosslessCapsError("dangling capitalization sentinel across piece boundary")
        return counts
    if normalized not in {LOSSLESS_CAPS_V2, LOSSLESS_CAPS_V3, LOSSLESS_CAPS_V4, LOSSLESS_CAPS_V5, LOSSLESS_CAPS_V6, LOSSLESS_CAPS_V7, LOSSLESS_CAPS_CASEOPS_V1}:
        raise ValueError(f"unsupported text_transform={text_transform_name!r}")

    title = DEFAULT_V2_TITLE
    allcaps = DEFAULT_V2_ALLCAPS
    capnext = DEFAULT_V2_CAPNEXT
    esc = DEFAULT_V2_ESC
    if normalized in {LOSSLESS_CAPS_V2, LOSSLESS_CAPS_CASEOPS_V1}:
        _validate_distinct_single_chars(title, allcaps, capnext, esc)
    elif normalized in {LOSSLESS_CAPS_V4, LOSSLESS_CAPS_V6, LOSSLESS_CAPS_V7}:
        _validate_distinct_single_chars(allcaps, esc)
    else:
        _validate_distinct_single_chars(title, allcaps, esc)
    pending_escape = False
    pending_word_mode: str | None = None
    active_allcaps = False
    pending_capnext = False
    in_ascii_word = False
    counts: list[int] = []
    for surface in surfaces:
        piece_bytes = 0
        for ch in surface:
            if pending_escape:
                if pending_word_mode is not None and not _is_ascii_alpha(ch):
                    raise LosslessCapsError("escaped control char cannot satisfy pending word capitalization mode")
                piece_bytes += len(ch.encode("utf-8"))
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
            if normalized in {LOSSLESS_CAPS_V2, LOSSLESS_CAPS_V3, LOSSLESS_CAPS_V5, LOSSLESS_CAPS_CASEOPS_V1} and ch == title:
                if pending_word_mode is not None or in_ascii_word or pending_capnext:
                    raise LosslessCapsError("invalid title marker placement")
                pending_word_mode = "title"
                continue
            if ch == allcaps:
                if pending_word_mode is not None or in_ascii_word or pending_capnext:
                    raise LosslessCapsError("invalid allcaps marker placement")
                pending_word_mode = "allcaps"
                continue
            if normalized in {LOSSLESS_CAPS_V2, LOSSLESS_CAPS_CASEOPS_V1} and ch == capnext:
                if pending_capnext:
                    raise LosslessCapsError("duplicate capnext marker")
                pending_capnext = True
                continue

            if _is_ascii_alpha(ch):
                at_word_start = not in_ascii_word
                if at_word_start:
                    piece_bytes += 1
                    active_allcaps = pending_word_mode == "allcaps"
                    pending_word_mode = None
                    pending_capnext = False
                    in_ascii_word = True
                    continue
                if pending_word_mode is not None:
                    raise LosslessCapsError("word capitalization marker leaked into the middle of a word")
                piece_bytes += 1
                pending_capnext = False
                continue

            if pending_word_mode is not None or pending_capnext:
                raise LosslessCapsError("capitalization marker not followed by an ASCII letter")
            piece_bytes += len(ch.encode("utf-8"))
            in_ascii_word = False
            active_allcaps = False
        counts.append(piece_bytes)
    if pending_escape:
        raise LosslessCapsError("dangling escape marker across piece boundary")
    if pending_word_mode is not None or pending_capnext:
        raise LosslessCapsError("dangling capitalization marker across piece boundary")
    return counts
