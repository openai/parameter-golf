#!/usr/bin/env python3
"""Create tiny data shards for local Mac testing. Extracts a small subset from
the full FineWeb shards so the MLX training script can run without OOM."""

import argparse
import numpy as np
from pathlib import Path


def extract_mini_shard(src: Path, dst: Path, max_tokens: int) -> int:
    header = np.fromfile(src, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {src}")
    full_tokens = int(header[2])
    keep = min(full_tokens, max_tokens)

    # Read only the tokens we need
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(src, dtype="<u2", count=keep, offset=header_bytes)

    # Write mini shard with updated header
    new_header = header.copy()
    new_header[2] = keep
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        new_header.tofile(f)
        tokens.tofile(f)
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--dst", default="./data/datasets/fineweb_mini")
    parser.add_argument("--train-tokens", type=int, default=500_000,
                        help="Max tokens per train shard")
    parser.add_argument("--val-tokens", type=int, default=50_000,
                        help="Max tokens for val shard")
    parser.add_argument("--train-shards", type=int, default=1,
                        help="Number of train shards to create")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    # Train shards
    for i in range(args.train_shards):
        src_file = src / f"fineweb_train_{i:06d}.bin"
        dst_file = dst / f"fineweb_train_{i:06d}.bin"
        if not src_file.exists():
            print(f"  skip {src_file} (not found)")
            continue
        n = extract_mini_shard(src_file, dst_file, args.train_tokens)
        print(f"  train shard {i}: {n:,} tokens -> {dst_file}")

    # Val shard
    src_val = src / "fineweb_val_000000.bin"
    dst_val = dst / "fineweb_val_000000.bin"
    n = extract_mini_shard(src_val, dst_val, args.val_tokens)
    print(f"  val shard: {n:,} tokens -> {dst_val}")

    total_bytes = sum(f.stat().st_size for f in dst.glob("*.bin"))
    print(f"\nDone! Mini dataset at {dst} ({total_bytes / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
