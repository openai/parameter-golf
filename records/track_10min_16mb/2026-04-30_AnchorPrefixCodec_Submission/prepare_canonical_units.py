from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

DATA_DIR = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"
TOKENIZER_PATH = REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"

OUT_ROOT = HERE / "records" / "canonical_units"
OUT_PER_FILE = OUT_ROOT / "per_file"
OUT_SUMMARY = OUT_ROOT / "summary"

OUT_PER_FILE.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY.mkdir(parents=True, exist_ok=True)

TRAIN_FILE_COUNT = int(os.environ.get("TRAIN_FILE_COUNT", "8"))
MAX_TRAIN_TOKENS_PER_FILE = int(os.environ.get("MAX_TRAIN_TOKENS_PER_FILE", "500000"))
MAX_VAL_TOKENS = int(os.environ.get("MAX_VAL_TOKENS", "500000"))

HEADER_BYTES = 1024
BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

WORD_START_MARK = "▁"
PERIOD_LIKE_CHARS = set(".!?。！？")
QUOTE_LIKE_CHARS = set("\"'“”‘’")
NEWLINE_LIKE_CHARS = set("\n\r")


def read_tokens(path: Path, max_tokens: int) -> np.ndarray:
    arr = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES)
    if max_tokens <= 0:
        return np.asarray(arr, dtype=np.int64)
    return np.asarray(arr[:max_tokens], dtype=np.int64)


def classify_piece(piece: str) -> dict:
    normalized = piece.replace(WORD_START_MARK, " ")
    visible = normalized.strip()
    alnum_count = sum(ch.isalnum() for ch in visible)
    punct_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in visible)

    return {
        "piece": piece,
        "normalized": normalized,
        "is_word_start": piece.startswith(WORD_START_MARK),
        "has_word_start": WORD_START_MARK in piece,
        "is_period_like": any(ch in piece for ch in PERIOD_LIKE_CHARS),
        "is_quote_like": any(ch in piece for ch in QUOTE_LIKE_CHARS),
        "is_newline_like": any(ch in piece for ch in NEWLINE_LIKE_CHARS),
        "is_punctuation_like": punct_count > 0 and alnum_count == 0,
        "char_len": len(piece),
    }


def build_piece_maps(sp):
    piece_by_id = {}
    is_word_start_by_id = {}
    is_punct_by_id = {}
    is_special_by_id = {}

    for token_id in range(sp.get_piece_size()):
        piece = sp.id_to_piece(token_id)
        info = classify_piece(piece)

        piece_by_id[token_id] = piece
        is_word_start_by_id[token_id] = bool(info["is_word_start"])
        is_punct_by_id[token_id] = bool(info["is_punctuation_like"])
        is_special_by_id[token_id] = piece in {"<s>", "</s>", "<unk>"}

    return piece_by_id, is_word_start_by_id, is_punct_by_id, is_special_by_id


def piece_join(token_ids: list[int], piece_by_id: dict[int, str], max_items: int = 24) -> str:
    parts = [piece_by_id.get(int(t), f"<UNK:{t}>") for t in token_ids[:max_items]]
    if len(token_ids) > max_items:
        parts.append("...")
    return " ".join(parts)


def strip_suffix_punctuation(token_ids: list[int], is_punct_by_id: dict[int, bool]) -> tuple[list[int], list[int]]:
    core = list(token_ids)
    suffix = []

    while core and is_punct_by_id.get(int(core[-1]), False):
        suffix.append(int(core.pop()))

    suffix.reverse()
    return core, suffix


def piece_text(piece: str) -> str:
    if piece == "▁":
        return ""
    if piece.startswith("▁"):
        return piece[1:]
    return piece


def looks_url_like(s: str) -> bool:
    low = s.lower()
    return (
        "http" in low
        or "www." in low
        or ".com" in low
        or ".org" in low
        or ".net" in low
        or "://" in low
    )


def looks_num_compound(s: str) -> bool:
    has_digit = any(ch.isdigit() for ch in s)
    has_joiner = any(ch in s for ch in ["-", ".", ",", "/", "%", ":"])
    return has_digit and has_joiner


def split_symbol_prefix_word(canonical_string: str) -> tuple[str, str] | None:
    if not canonical_string:
        return None

    i = 0
    while i < len(canonical_string) and (not canonical_string[i].isalnum()):
        i += 1

    if i == 0 or i >= len(canonical_string):
        return None

    body = canonical_string[i:]
    if not any(ch.isalpha() for ch in body):
        return None

    return canonical_string[:i], body


def canonicalize_pieces(pieces: list[str]) -> tuple[str, str, str]:
    if not pieces:
        return "", "", "EMPTY"

    has_word_start = any(p.startswith("▁") or p == "▁" for p in pieces)
    has_special = any(p.startswith("<") and p.endswith(">") and not BYTE_RE.match(p) for p in pieces)
    has_byte = any(BYTE_RE.match(p) for p in pieces)

    body = "".join(piece_text(p) for p in pieces)

    canonical_unit = "▁" + body if has_word_start else body
    canonical_string = body
    symbol_split = split_symbol_prefix_word(canonical_string)

    if not canonical_string and has_word_start:
        canonical_type = "WORD_BOUNDARY_ONLY"
    elif has_special and len(pieces) > 1:
        canonical_type = "SPECIAL_MIXED"
    elif has_special:
        canonical_type = "SPECIAL"
    elif has_byte:
        canonical_type = "BYTE_NOISE"
    elif symbol_split is not None:
        canonical_type = "SYMBOL_PREFIX_WORD"
    elif looks_url_like(canonical_string):
        canonical_type = "URL_LIKE"
    elif looks_num_compound(canonical_string):
        canonical_type = "NUM_COMPOUND"
    elif canonical_string.isdigit():
        canonical_type = "NUM"
    elif has_word_start:
        canonical_type = "WORD"
    else:
        canonical_type = "FRAGMENT"

    return canonical_unit, canonical_string, canonical_type


def canonicalize_unit_row(unit: dict, piece_by_id: dict[int, str]) -> dict:
    core_token_ids = [int(x) for x in unit.get("core_token_ids", [])]
    raw_token_ids = [int(x) for x in unit.get("raw_token_ids", [])]
    suffix_token_ids = [int(x) for x in unit.get("suffix_punct_token_ids", [])]

    core_pieces = [piece_by_id.get(t, f"<UNK:{t}>") for t in core_token_ids]
    raw_pieces = [piece_by_id.get(t, f"<UNK:{t}>") for t in raw_token_ids]
    suffix_pieces = [piece_by_id.get(t, f"<UNK:{t}>") for t in suffix_token_ids]

    canonical_unit, canonical_string, canonical_type = canonicalize_pieces(core_pieces)

    symbol_split = split_symbol_prefix_word(canonical_string)
    symbol_prefix = symbol_split[0] if symbol_split else None
    symbol_body = symbol_split[1] if symbol_split else None

    is_dropped = canonical_type in {"EMPTY", "WORD_BOUNDARY_ONLY"}

    return {
        **unit,
        "core_pieces": core_pieces,
        "raw_pieces": raw_pieces,
        "suffix_punct_pieces": suffix_pieces,
        "canonical_unit": canonical_unit,
        "canonical_string": canonical_string,
        "canonical_type": canonical_type,
        "symbol_prefix": symbol_prefix,
        "symbol_body": symbol_body,
        "canonical_key": canonical_unit,
        "is_dropped": is_dropped,
    }


def iter_unit_rows(
    file_id: str,
    tokens: np.ndarray,
    piece_by_id: dict[int, str],
    is_word_start_by_id: dict[int, bool],
    is_punct_by_id: dict[int, bool],
    is_special_by_id: dict[int, bool],
):
    current_start = 0
    current_ids: list[int] = []
    unit_id = 0
    n = len(tokens)

    def flush_unit(end_pos: int):
        nonlocal unit_id, current_ids, current_start

        if not current_ids:
            return None

        raw_ids = [int(x) for x in current_ids]
        core_ids, suffix_ids = strip_suffix_punctuation(raw_ids, is_punct_by_id)

        row = {
            "file_id": file_id,
            "unit_id": unit_id,
            "start_token": current_start,
            "end_token": end_pos,
            "token_count": len(raw_ids),
            "core_token_count": len(core_ids),
            "raw_token_ids": raw_ids,
            "core_token_ids": core_ids,
            "suffix_punct_token_ids": suffix_ids,
            "has_suffix_punct": bool(suffix_ids),
            "raw_piece_preview": piece_join(raw_ids, piece_by_id),
            "core_piece_preview": piece_join(core_ids, piece_by_id),
            "suffix_piece_preview": piece_join(suffix_ids, piece_by_id),
        }

        unit_id += 1
        return row

    for pos, token0 in enumerate(tokens):
        token = int(token0)

        starts_new_unit = is_word_start_by_id.get(token, False)
        is_special = is_special_by_id.get(token, False)

        if is_special:
            if current_ids:
                row = flush_unit(pos)
                if row is not None:
                    yield row

            current_start = pos
            current_ids = [token]

            row = flush_unit(pos + 1)
            if row is not None:
                yield row

            current_ids = []
            current_start = pos + 1

        elif starts_new_unit and current_ids:
            row = flush_unit(pos)
            if row is not None:
                yield row

            current_start = pos
            current_ids = [token]

        else:
            if not current_ids:
                current_start = pos
            current_ids.append(token)

    row = flush_unit(n)
    if row is not None:
        yield row


def process_file(
    file_id: str,
    file_name: str,
    path: Path,
    max_tokens: int,
    piece_by_id: dict[int, str],
    is_word_start_by_id: dict[int, bool],
    is_punct_by_id: dict[int, bool],
    is_special_by_id: dict[int, bool],
) -> dict:
    print(f"[prepare] {file_id} from {file_name} max_tokens={max_tokens}")

    tokens = read_tokens(path, max_tokens)
    out_path = OUT_PER_FILE / f"{file_id}_canonical_units_v1.jsonl"

    type_counter = Counter()
    unit_count = 0
    dropped_count = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for unit in iter_unit_rows(
            file_id=file_id,
            tokens=tokens,
            piece_by_id=piece_by_id,
            is_word_start_by_id=is_word_start_by_id,
            is_punct_by_id=is_punct_by_id,
            is_special_by_id=is_special_by_id,
        ):
            row = canonicalize_unit_row(unit, piece_by_id)

            unit_count += 1
            type_counter[row["canonical_type"]] += 1
            if row["is_dropped"]:
                dropped_count += 1

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "file_id": file_id,
        "file_name": file_name,
        "input_path": str(path),
        "output_path": str(out_path),
        "tokens_used": int(len(tokens)),
        "unit_count": int(unit_count),
        "dropped_unit_count": int(dropped_count),
        "canonical_type_counts": dict(type_counter),
    }

    print(f"[done] {file_id} units={unit_count} dropped={dropped_count}")
    return result


def required_outputs_exist() -> bool:
    train_ok = all(
        (OUT_PER_FILE / f"train_{i:06d}_canonical_units_v1.jsonl").exists()
        for i in range(TRAIN_FILE_COUNT)
    )
    val_ok = (OUT_PER_FILE / "val_000000_canonical_units_v1.jsonl").exists()
    return train_ok and val_ok


def main() -> None:
    if required_outputs_exist() and os.environ.get("FORCE_PREPARE_CANONICAL_UNITS", "0") != "1":
        print("[prepare] canonical units already exist; skip")
        return

    if not DATA_DIR.exists():
        raise FileNotFoundError(DATA_DIR)
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))

    piece_by_id, is_word_start_by_id, is_punct_by_id, is_special_by_id = build_piece_maps(sp)

    results = []

    for i in range(TRAIN_FILE_COUNT):
        path = DATA_DIR / f"fineweb_train_{i:06d}.bin"
        if not path.exists():
            raise FileNotFoundError(path)

        results.append(process_file(
            file_id=f"train_{i:06d}",
            file_name=path.name,
            path=path,
            max_tokens=MAX_TRAIN_TOKENS_PER_FILE,
            piece_by_id=piece_by_id,
            is_word_start_by_id=is_word_start_by_id,
            is_punct_by_id=is_punct_by_id,
            is_special_by_id=is_special_by_id,
        ))

    val_path = DATA_DIR / "fineweb_val_000000.bin"
    if not val_path.exists():
        raise FileNotFoundError(val_path)

    results.append(process_file(
        file_id="val_000000",
        file_name=val_path.name,
        path=val_path,
        max_tokens=MAX_VAL_TOKENS,
        piece_by_id=piece_by_id,
        is_word_start_by_id=is_word_start_by_id,
        is_punct_by_id=is_punct_by_id,
        is_special_by_id=is_special_by_id,
    ))

    summary = {
        "stage": "prepare_canonical_units",
        "data_dir": str(DATA_DIR),
        "tokenizer_path": str(TOKENIZER_PATH),
        "train_file_count": TRAIN_FILE_COUNT,
        "max_train_tokens_per_file": MAX_TRAIN_TOKENS_PER_FILE,
        "max_val_tokens": MAX_VAL_TOKENS,
        "files": results,
    }

    out_summary = OUT_SUMMARY / "canonical_unit_summary_v1.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_summary}")


if __name__ == "__main__":
    main()
