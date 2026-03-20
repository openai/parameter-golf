#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import lzma
import struct
import zlib
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520


def load_val_tokens(val_dir: str) -> np.ndarray:
    files = sorted(glob.glob(str(Path(val_dir) / "fineweb_val_*.bin")))
    if not files:
        raise FileNotFoundError(f"No val files found in {val_dir}")
    parts = []
    for path in files:
        with open(path, "rb") as fh:
            header = np.frombuffer(fh.read(256 * 4), dtype="<i4")
            if header.size != 256 or int(header[0]) != DATAFILE_MAGIC:
                raise ValueError(f"Bad shard header: {path}")
            n_tokens = int(header[2])
            parts.append(np.frombuffer(fh.read(n_tokens * 2), dtype="<u2"))
    return np.concatenate(parts)


def pack_10bit(tokens: np.ndarray) -> bytes:
    n = len(tokens)
    padded = n + (4 - n % 4) % 4
    buf = np.zeros(padded, dtype=np.uint16)
    buf[:n] = tokens
    out = bytearray(struct.pack("<I", n))
    for i in range(0, padded, 4):
        a, b, c, d = map(int, buf[i : i + 4])
        packed = a | (b << 10) | (c << 20) | (d << 30)
        out.extend(struct.pack("<Q", packed)[:5])
    return bytes(out)


def compress_tokens(tokens: np.ndarray, method: str) -> bytes:
    raw = tokens.astype("<u2", copy=False).tobytes()
    if method == "raw":
        return raw
    if method == "zlib9":
        return zlib.compress(raw, 9)
    if method == "lzma6":
        return lzma.compress(raw, preset=6)
    if method == "lzma9":
        return lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    if method == "pack10_zlib":
        return zlib.compress(pack_10bit(tokens), 9)
    if method == "pack10_lzma":
        return lzma.compress(pack_10bit(tokens), preset=9 | lzma.PRESET_EXTREME)
    raise ValueError(f"Unknown method: {method}")


def max_tokens_for_budget(target_tokens: np.ndarray, budget_bytes: int, method: str) -> int:
    lo, hi = 0, len(target_tokens)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if len(compress_tokens(target_tokens[:mid], method)) <= budget_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "raw", "zlib9", "lzma6", "lzma9", "pack10_zlib", "pack10_lzma"],
    )
    args = parser.parse_args()

    val_tokens = load_val_tokens(args.val_dir)
    target_tokens = val_tokens[1:].copy()
    methods = ["lzma6", "lzma9", "pack10_lzma", "pack10_zlib"] if args.method == "auto" else [args.method]

    best_method = methods[0]
    best_n = 0
    for method in methods:
        n = max_tokens_for_budget(target_tokens, args.budget_bytes, method)
        if n > best_n:
            best_n = n
            best_method = method

    blob = compress_tokens(target_tokens[:best_n], best_method)
    Path(args.output).write_bytes(blob)
    print(f"output={args.output}")
    print(f"blob_bytes={len(blob)}")
    print(f"method={best_method}")
    print(f"tokens_covered={best_n}")
    print(f"total_target_tokens={len(target_tokens)}")
    print(f"coverage={best_n / len(target_tokens):.6f}")


if __name__ == "__main__":
    main()
