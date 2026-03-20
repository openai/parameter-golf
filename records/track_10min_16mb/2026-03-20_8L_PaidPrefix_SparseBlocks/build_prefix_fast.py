#!/usr/bin/env python3
"""Prefix blob builder for contiguous and sparse hard-block paid-prefix strategies.

Examples:
  python build_prefix_fast.py --val-dir ./data/datasets/fineweb10B_sp1024/ \
      --num-tokens 6200000 --output prefix_contiguous.xz

  python build_prefix_fast.py --strategy hard_blocks \
      --val-dir ./data/datasets/fineweb10B_sp1024/ \
      --nll-path ./logs/final_slide_nll.npy \
      --target-bytes 4240472 \
      --block-size 256 \
      --output prefix_sparse_blocks.xz
"""

import argparse
import glob
import lzma
import struct
import time
from pathlib import Path

import numpy as np

DATAFILE_MAGIC = 20240520
SPARSE_PREFIX_MAGIC = b"SPB1"
SPARSE_PREFIX_VERSION = 1
SPARSE_PREFIX_HEADER = struct.Struct("<4sIII")


def load_val_tokens(val_dir: str) -> np.ndarray:
    pattern = str(Path(val_dir) / "fineweb_val_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files found: {pattern}")
    all_tokens = []
    for f in files:
        with open(f, "rb") as fh:
            header = np.frombuffer(fh.read(256 * 4), dtype="<i4")
            if header[0] != DATAFILE_MAGIC:
                raise ValueError(f"Unexpected magic in {f}: {header[0]}")
            n_tokens = int(header[2])
            tokens = np.frombuffer(fh.read(n_tokens * 2), dtype="<u2")
            all_tokens.append(tokens)
    result = np.concatenate(all_tokens)
    print(f"Loaded {len(result):,} val tokens from {len(files)} files")
    return result


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"Cannot encode negative varint: {value}")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def compress_lzma(raw_data: bytes) -> bytes:
    return lzma.compress(raw_data, preset=6)


def build_contiguous_blob(target_tokens: np.ndarray) -> bytes:
    return compress_lzma(target_tokens.astype("<u2", copy=False).tobytes())


def build_sparse_blocks_blob(target_tokens: np.ndarray, block_starts: np.ndarray, block_size: int) -> bytes:
    if block_starts.ndim != 1:
        raise ValueError("block_starts must be a 1D array")
    if len(block_starts) == 0:
        raw = SPARSE_PREFIX_HEADER.pack(SPARSE_PREFIX_MAGIC, SPARSE_PREFIX_VERSION, block_size, 0)
        return compress_lzma(raw)

    starts = np.sort(block_starts.astype(np.int64, copy=False))
    deltas = []
    prev = 0
    for start in starts.tolist():
        deltas.append(start - prev)
        prev = start
    starts_bytes = b"".join(encode_uvarint(delta) for delta in deltas)

    token_blocks = np.stack(
        [target_tokens[start:start + block_size] for start in starts.tolist()],
        axis=0,
    ).astype("<u2", copy=False)
    raw = bytearray()
    raw.extend(SPARSE_PREFIX_HEADER.pack(SPARSE_PREFIX_MAGIC, SPARSE_PREFIX_VERSION, block_size, len(starts)))
    raw.extend(starts_bytes)
    raw.extend(token_blocks.tobytes())
    return compress_lzma(bytes(raw))


def pick_hard_blocks(
    *,
    target_tokens: np.ndarray,
    nll: np.ndarray,
    target_bytes: int,
    block_size: int,
) -> tuple[np.ndarray, bytes]:
    usable_tokens = (len(target_tokens) // block_size) * block_size
    if usable_tokens <= 0:
        raise ValueError(f"Not enough target tokens for block_size={block_size}")
    target_trim = target_tokens[:usable_tokens]
    nll_trim = nll[:usable_tokens]

    num_blocks = usable_tokens // block_size
    block_scores = nll_trim.reshape(num_blocks, block_size).sum(axis=1)
    ranked_blocks = np.argsort(-block_scores, kind="stable").astype(np.int64)
    ranked_starts = ranked_blocks * block_size

    def blob_for_count(count: int) -> bytes:
        return build_sparse_blocks_blob(target_trim, ranked_starts[:count], block_size)

    lo, hi = 0, num_blocks
    best_blob = blob_for_count(0)
    best_count = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        blob = blob_for_count(mid)
        if len(blob) <= target_bytes:
            best_count = mid
            best_blob = blob
            lo = mid + 1
        else:
            hi = mid - 1

    while best_count > 0 and len(best_blob) > target_bytes:
        best_count -= 1
        best_blob = blob_for_count(best_count)

    chosen = np.sort(ranked_starts[:best_count])
    return chosen, best_blob


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", default="prefix.xz")
    parser.add_argument(
        "--strategy",
        choices=("contiguous", "hard_blocks"),
        default="contiguous",
        help="contiguous stores the first N target tokens; hard_blocks stores the highest-NLL fixed blocks",
    )
    parser.add_argument("--num-tokens", type=int, default=None, help="Used by contiguous strategy")
    parser.add_argument("--nll-path", default=None, help="Path to dense per-position NLL .npy dump")
    parser.add_argument("--target-bytes", type=int, default=None, help="Compressed budget for hard_blocks")
    parser.add_argument("--block-size", type=int, default=256)
    args = parser.parse_args()

    val_tokens = load_val_tokens(args.val_dir)
    target_tokens = val_tokens[1:].copy()
    total_targets = len(target_tokens)

    if args.strategy == "contiguous":
        if args.num_tokens is None:
            raise ValueError("--num-tokens is required for --strategy contiguous")
        selected = target_tokens[:args.num_tokens]
        print(f"Target tokens: {len(selected):,} / {total_targets:,} ({len(selected)/total_targets:.1%})")
        raw_size = selected.astype("<u2", copy=False).nbytes
        print(f"Raw size: {raw_size:,} bytes ({raw_size/1e6:.2f} MB)")
        t0 = time.time()
        blob = build_contiguous_blob(selected)
        dt = time.time() - t0
        print(f"LZMA-6 compressed: {len(blob):,} bytes ({len(blob)/1e6:.2f} MB) in {dt:.1f}s")
        print(f"Ratio: {raw_size/max(len(blob), 1):.2f}x")
        coverage = len(selected) / total_targets
    else:
        if args.nll_path is None:
            raise ValueError("--nll-path is required for --strategy hard_blocks")
        if args.target_bytes is None:
            raise ValueError("--target-bytes is required for --strategy hard_blocks")
        if args.block_size <= 0:
            raise ValueError("--block-size must be positive")
        nll = np.load(args.nll_path)
        if nll.ndim != 1:
            raise ValueError(f"NLL dump must be 1D, got shape {nll.shape}")
        if len(nll) != total_targets:
            raise ValueError(f"NLL length mismatch: expected {total_targets}, got {len(nll)}")
        t0 = time.time()
        block_starts, blob = pick_hard_blocks(
            target_tokens=target_tokens,
            nll=nll.astype(np.float32, copy=False),
            target_bytes=args.target_bytes,
            block_size=args.block_size,
        )
        dt = time.time() - t0
        num_blocks = len(block_starts)
        coverage_tokens = num_blocks * args.block_size
        coverage = coverage_tokens / total_targets
        print(
            f"Selected hard blocks: {num_blocks:,} blocks x {args.block_size} = "
            f"{coverage_tokens:,} tokens ({coverage:.1%})"
        )
        print(
            f"Sparse LZMA-6 compressed: {len(blob):,} bytes ({len(blob)/1e6:.2f} MB) "
            f"vs target {args.target_bytes:,} bytes in {dt:.1f}s"
        )
        if num_blocks:
            print(
                f"Block starts: min={int(block_starts[0]):,} max={int(block_starts[-1]):,} "
                f"median={int(np.median(block_starts)):,}"
            )

    Path(args.output).write_bytes(blob)
    print(f"\nWritten: {args.output}")
    print(f"Coverage: {coverage:.1%} of validation targets")
    print(f"File size: {len(blob)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
