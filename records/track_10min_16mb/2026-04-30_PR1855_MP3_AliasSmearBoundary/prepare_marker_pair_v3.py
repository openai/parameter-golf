"""prepare_marker_pair_v3.py — vocab surgery for MP3 marker-pair fusion.

Reads CaseOps-tokenized fineweb shards (output of prepare_caseops_data.py)
and replaces three 2-gram patterns with single alias donor tokens:

  [SPACE_ID=8133, TITLE=4]    ->  ALIAS_TITLE    (donor=8)
  [SPACE_ID=8133, ALLCAPS=5]  ->  ALIAS_ALLCAPS  (donor=9)
  [SPACE_ID=8133, CAPNEXT=6]  ->  ALIAS_CAPNEXT  (donor=10)

The donor IDs come from byte-fallback tokens that occur 0 times in
the CaseOps corpus (audited by full-corpus token-count). The original
byte-fallback meaning is therefore unused and the slot is "free" to
repurpose. Tokenizer is unchanged; the transform is purely a stream
edit on already-tokenized .bin shards.

Word X following the marker pair is PRESERVED — this is what
distinguishes MP3 from earlier full-word-fusion attempts that absorbed
the word into the alias and broke d=1 prediction.

Byte sidecar update (val only):
  byte[ALIAS] = byte[SPACE] + byte[MARKER] = 1 + 0 = 1

I/O:
  Source: ${SRC_DIR:=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}
  Target: ${DST_DIR:=./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3}

Usage (runpod):
  python3 prepare_marker_pair_v3.py
"""

import glob
import json
import os
import sys

import numpy as np


SPACE_ID = 8133
# (donor_id, marker_token_id, name)
MARKER_PAIRS = [
    (8, 4, "TITLE"),
    (9, 5, "ALLCAPS"),
    (10, 6, "CAPNEXT"),
]


def fuse_stream(toks: np.ndarray, bytes_arr: np.ndarray | None = None):
    """Vectorised stream surgery. Input dtype preserved."""
    n = len(toks)
    if n < 2:
        return toks.copy(), (bytes_arr.copy() if bytes_arr is not None else None)

    is_space = toks[:-1] == SPACE_ID
    next_is_marker = np.zeros(n - 1, dtype=bool)
    for _, m, _ in MARKER_PAIRS:
        next_is_marker |= toks[1:] == m

    pair_starts = np.where(is_space & next_is_marker)[0]
    marker_positions = pair_starts + 1

    out_toks = toks.copy()
    out_bytes = bytes_arr.copy() if bytes_arr is not None else None
    for donor, marker_id, _ in MARKER_PAIRS:
        sel = (toks[marker_positions] == marker_id)
        positions = marker_positions[sel]
        if positions.size == 0:
            continue
        out_toks[positions] = donor
        if out_bytes is not None:
            out_bytes[positions] = (
                bytes_arr[positions - 1].astype(np.int32)
                + bytes_arr[positions].astype(np.int32)
            ).astype(out_bytes.dtype)

    keep = np.ones(n, dtype=bool)
    keep[pair_starts] = False  # drop space positions; markers (now donors) stay

    out_toks = out_toks[keep]
    if out_bytes is not None:
        out_bytes = out_bytes[keep]
    return out_toks, out_bytes


SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256          # int32 header slots
HEADER_BYTES = HEADER_INTS * 4


def _read_shard_with_header(path: str):
    """Read a header-prefixed uint16 shard. Returns (header_int32, tokens_uint16)."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"unexpected header for {path}")
    n = int(header[2])
    toks = np.fromfile(path, dtype="<u2", count=n, offset=HEADER_BYTES)
    if toks.size != n:
        raise ValueError(f"short read for {path}: header n={n} got {toks.size}")
    return header, toks


def _write_shard_with_header(path: str, toks: np.ndarray):
    """Write a header-prefixed uint16 shard. Header int32[2] = num_tokens."""
    assert toks.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(toks.size)
    with open(path, "wb") as fh:
        fh.write(header.tobytes())
        fh.write(toks.tobytes())


def process_shards(src_pattern: str, dst_dir: str, with_bytes: bool, label: str):
    os.makedirs(dst_dir, exist_ok=True)
    src_files = sorted(glob.glob(src_pattern))
    if not src_files:
        print(f"  [{label}] no shards matched {src_pattern}")
        return 0, 0
    print(f"  [{label}] {len(src_files)} shards")
    total_in, total_out = 0, 0
    for src in src_files:
        name = os.path.basename(src)
        dst = os.path.join(dst_dir, name)
        _hdr, toks = _read_shard_with_header(src)
        bytes_arr = None
        if with_bytes:
            bp = src.replace("fineweb_val_", "fineweb_val_bytes_")
            if os.path.exists(bp):
                _hdr_b, bytes_arr = _read_shard_with_header(bp)
                if len(bytes_arr) != len(toks):
                    raise RuntimeError(f"length mismatch: {bp} ({len(bytes_arr)}) vs {src} ({len(toks)})")
            else:
                print(f"    WARN: bytes sidecar missing for {name}")
        new_toks, new_bytes = fuse_stream(toks, bytes_arr)
        _write_shard_with_header(dst, new_toks)
        if with_bytes and new_bytes is not None:
            _write_shard_with_header(dst.replace("fineweb_val_", "fineweb_val_bytes_"), new_bytes)
        saved = len(toks) - len(new_toks)
        total_in += len(toks)
        total_out += len(new_toks)
        print(f"    {name}: {len(toks):>11,} -> {len(new_toks):>11,}  (-{saved:,}, {saved / len(toks) * 100:.2f}%)")
    return total_in, total_out


def main():
    src_root = os.environ.get(
        "SRC_DIR",
        "./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved",
    )
    dst_root = os.environ.get(
        "DST_DIR",
        "./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3",
    )

    if not os.path.isdir(src_root):
        # try the alternate layout used by some prepare_caseops_data.py variants
        alt = "./data/datasets/fineweb10B_sp8192_caseops/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
        if os.path.isdir(alt):
            src_root = alt
        else:
            print(f"ERROR: source dir not found: {src_root}")
            print(f"       (also tried: {alt})")
            print("Run prepare_caseops_data.py first.")
            sys.exit(1)

    print("=== marker_pair_v3 vocab surgery ===")
    print(f"  src: {src_root}")
    print(f"  dst: {dst_root}")
    for d, m, n in MARKER_PAIRS:
        print(f"    [SPACE={SPACE_ID}, {n}={m}] -> ALIAS_{n}={d}")

    os.makedirs(dst_root, exist_ok=True)

    print("\n[train shards]")
    tr_in, tr_out = process_shards(f"{src_root}/fineweb_train_*.bin", dst_root, False, "train")
    if tr_in:
        print(f"  train total: {tr_in:,} -> {tr_out:,}  ({(tr_in - tr_out) / tr_in * 100:.2f}% saved)")

    print("\n[val shards + bytes sidecar]")
    val_in, val_out = process_shards(f"{src_root}/fineweb_val_[0-9]*.bin", dst_root, True, "val")
    if val_in:
        print(f"  val total: {val_in:,} -> {val_out:,}  ({(val_in - val_out) / val_in * 100:.2f}% saved)")

    alias_map_data = {
        "alias_map": {
            "marker_pair_space_title": 8,
            "marker_pair_space_allcaps": 9,
            "marker_pair_space_capnext": 10,
        },
        "marker_pairs": [
            {"donor": d, "marker_id": m, "marker_name": n} for d, m, n in MARKER_PAIRS
        ],
        "note": "MP3: 3-marker fusion (TITLE + ALLCAPS + CAPNEXT). Word X preserved.",
    }
    with open(os.path.join(dst_root, "alias_map.json"), "w", encoding="utf-8") as f:
        json.dump(alias_map_data, f, indent=2)
    print(f"\nwrote {dst_root}/alias_map.json")

    print("\n=== done ===")


if __name__ == "__main__":
    main()
