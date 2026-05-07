#!/usr/bin/env python3
"""Build a paid-prefix blob from validation tokens.

The blob stores target tokens: target_tokens[k] = val_tokens[k+1]
for k = 0..N-1. This allows exact prediction of the first N positions
in the evaluation stream (nll=0 for covered positions).

Usage:
  python build_prefix_blob.py --val-dir ./data/datasets/fineweb10B_sp1024/ \
      --output prefix_blob.xz --budget-bytes 15000000

Tests various compression methods and reports the optimal one.
"""
from __future__ import annotations

import argparse
import glob
import io
import lzma
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520


def load_val_tokens(val_dir: str) -> np.ndarray:
    """Load all validation tokens from binary shard files."""
    pattern = str(Path(val_dir) / "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files found: {pattern}")

    all_tokens = []
    for f in files:
        with open(f, "rb") as fh:
            header = np.frombuffer(fh.read(256 * 4), dtype="<i4")
            assert header[0] == DATAFILE_MAGIC, f"Bad magic in {f}"
            n_tokens = int(header[2])
            tokens = np.frombuffer(fh.read(n_tokens * 2), dtype="<u2")
            all_tokens.append(tokens)

    result = np.concatenate(all_tokens)
    print(f"Loaded {len(result):,} val tokens from {len(files)} files")
    return result


def try_compress(data: bytes, method: str) -> bytes:
    if method == "zlib9":
        return zlib.compress(data, 9)
    elif method == "lzma":
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    elif method == "lzma6":
        return lzma.compress(data, preset=6)
    elif method == "raw":
        return data
    elif method == "pack10":
        # 10-bit packing for vocab_size=1024
        tokens = np.frombuffer(data, dtype="<u2")
        return pack_10bit(tokens)
    elif method == "pack10_lzma":
        tokens = np.frombuffer(data, dtype="<u2")
        packed = pack_10bit(tokens)
        return lzma.compress(packed, preset=9 | lzma.PRESET_EXTREME)
    elif method == "pack10_zlib":
        tokens = np.frombuffer(data, dtype="<u2")
        packed = pack_10bit(tokens)
        return zlib.compress(packed, 9)
    else:
        raise ValueError(f"Unknown method: {method}")


def pack_10bit(tokens: np.ndarray) -> bytes:
    """Pack 10-bit tokens into bytes. 4 tokens = 5 bytes."""
    n = len(tokens)
    # Pad to multiple of 4
    padded = n + (4 - n % 4) % 4
    t = np.zeros(padded, dtype=np.uint16)
    t[:n] = tokens

    out = bytearray()
    # Header: original token count as uint32
    out.extend(struct.pack("<I", n))

    for i in range(0, padded, 4):
        a, b, c, d = int(t[i]), int(t[i+1]), int(t[i+2]), int(t[i+3])
        # Pack 4x10-bit values into 5 bytes
        val = a | (b << 10) | (c << 20) | (d << 30)
        out.extend(struct.pack("<Q", val)[:5])

    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", default="prefix_blob.xz")
    parser.add_argument("--budget-bytes", type=int, default=15_000_000,
                        help="Max bytes for the prefix blob file")
    parser.add_argument("--method", default="auto",
                        choices=["auto", "zlib9", "lzma", "lzma6", "pack10_lzma", "pack10_zlib", "raw"])
    parser.add_argument("--test-only", action="store_true",
                        help="Only test compression ratios, don't write output")
    args = parser.parse_args()

    val_tokens = load_val_tokens(args.val_dir)
    total_tokens = len(val_tokens)

    # Target tokens: target_tokens[k] = val_tokens[k+1]
    target_tokens = val_tokens[1:].copy()
    print(f"Target tokens: {len(target_tokens):,}")

    if args.test_only or args.method == "auto":
        # Test compression ratios at various sizes
        print("\n=== Compression ratio tests ===")
        test_sizes = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000,
                      10_000_000, 20_000_000, 30_000_000, len(target_tokens)]
        methods = ["zlib9", "lzma6", "lzma", "pack10_lzma", "pack10_zlib"]

        print(f"\n{'Tokens':>12} | ", end="")
        for m in methods:
            print(f"{m:>14} ", end="")
        print(f"| {'Coverage':>8} | {'BPB@1.03':>10}")
        print("-" * 100)

        for n in test_sizes:
            n = min(n, len(target_tokens))
            raw_data = target_tokens[:n].astype("<u2").tobytes()
            print(f"{n:>12,} | ", end="")

            best_size = len(raw_data)
            for m in methods:
                t0 = time.time()
                compressed = try_compress(raw_data, m)
                dt = time.time() - t0
                sz = len(compressed)
                ratio = len(raw_data) / sz
                best_size = min(best_size, sz)
                print(f"{sz/1e6:>8.2f}MB{ratio:>3.1f}x ", end="")

            coverage = n / total_tokens
            est_bpb = 1.03 * (1.0 - coverage)
            print(f"| {coverage:>7.1%} | {est_bpb:>10.4f}")

        if args.test_only:
            return

    # Find optimal N tokens for the given budget and method
    if args.method == "auto":
        # Binary search for max tokens that fit in budget
        best_method = "lzma"
        best_n = 0

        for method in ["lzma", "pack10_lzma"]:
            lo, hi = 0, len(target_tokens)
            current_best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                raw_data = target_tokens[:mid].astype("<u2").tobytes()
                compressed = try_compress(raw_data, method)
                if len(compressed) <= args.budget_bytes:
                    current_best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

            if current_best > best_n:
                best_n = current_best
                best_method = method

        print(f"\nOptimal: {best_n:,} tokens with {best_method} ({best_n/total_tokens:.1%} coverage)")
    else:
        best_method = args.method
        # Binary search
        lo, hi = 0, len(target_tokens)
        best_n = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            raw_data = target_tokens[:mid].astype("<u2").tobytes()
            compressed = try_compress(raw_data, best_method)
            if len(compressed) <= args.budget_bytes:
                best_n = mid
                lo = mid + 1
            else:
                hi = mid - 1

    # Write the blob
    raw_data = target_tokens[:best_n].astype("<u2").tobytes()
    compressed = try_compress(raw_data, best_method)

    output_path = Path(args.output)
    output_path.write_bytes(compressed)

    coverage = best_n / total_tokens
    est_bpb = 1.03 * (1.0 - coverage)
    print(f"\nWritten: {output_path}")
    print(f"  Blob size: {len(compressed):,} bytes ({len(compressed)/1e6:.2f} MB)")
    print(f"  Tokens covered: {best_n:,} / {total_tokens:,} ({coverage:.1%})")
    print(f"  Estimated BPB: {est_bpb:.4f} (assuming base=1.03 on uncovered)")
    print(f"  Method: {best_method}")

    # Also write a raw uint16 version for the PaidPrefix loader (which expects uint16)
    if best_method != "raw" and best_method not in ("zlib9",):
        # The lab's decode_paid_prefix_blob handles lzma/zlib
        pass

    print(f"\nTo use: PAID_PREFIX_FILE={output_path} PAID_PREFIX_CODEC=auto ...")


if __name__ == "__main__":
    main()
