#!/usr/bin/env python3
"""Fast prefix blob builder — skips binary search, just builds at target token count.

Usage:
  python build_prefix_fast.py --val-dir ./data/datasets/fineweb10B_sp1024/ \
      --num-tokens 15000000 --output prefix.xz
"""
import argparse
import glob
import lzma
import struct
import sys
import time
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520


def load_val_tokens(val_dir: str) -> np.ndarray:
    pattern = str(Path(val_dir) / "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files found: {pattern}")
    all_tokens = []
    for f in files:
        with open(f, "rb") as fh:
            header = np.frombuffer(fh.read(256 * 4), dtype="<i4")
            assert header[0] == DATAFILE_MAGIC
            n_tokens = int(header[2])
            tokens = np.frombuffer(fh.read(n_tokens * 2), dtype="<u2")
            all_tokens.append(tokens)
    result = np.concatenate(all_tokens)
    print(f"Loaded {len(result):,} val tokens from {len(files)} files")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", default="prefix.xz")
    parser.add_argument("--num-tokens", type=int, required=True,
                        help="Number of target tokens to store")
    args = parser.parse_args()

    val_tokens = load_val_tokens(args.val_dir)
    total = len(val_tokens)
    target_tokens = val_tokens[1:args.num_tokens + 1].copy()
    print(f"Target tokens: {len(target_tokens):,} / {total:,} ({len(target_tokens)/total:.1%})")

    raw_data = target_tokens.astype("<u2").tobytes()
    print(f"Raw size: {len(raw_data):,} bytes ({len(raw_data)/1e6:.2f} MB)")

    t0 = time.time()
    compressed = lzma.compress(raw_data, preset=6)  # level 6 is much faster than 9
    dt = time.time() - t0
    print(f"LZMA-6 compressed: {len(compressed):,} bytes ({len(compressed)/1e6:.2f} MB) in {dt:.1f}s")
    print(f"Ratio: {len(raw_data)/len(compressed):.2f}x")

    Path(args.output).write_bytes(compressed)
    coverage = len(target_tokens) / total
    print(f"\nWritten: {args.output}")
    print(f"Coverage: {coverage:.1%} of val tokens")
    print(f"File size: {len(compressed)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
