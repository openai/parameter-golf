#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def read_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256:
        raise RuntimeError(f"Invalid shard header: {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * np.dtype("<i4").itemsize)
    if tokens.size != num_tokens:
        raise RuntimeError(f"Token count mismatch in shard: {path}")
    return header, tokens


def write_shard(path: Path, header: np.ndarray, tokens: np.ndarray) -> None:
    out_header = header.copy()
    out_header[2] = int(tokens.size)
    with path.open("wb") as f:
        out_header.astype("<i4").tofile(f)
        tokens.astype("<u2").tofile(f)


def main() -> None:
    p = argparse.ArgumentParser(description="Create a fast local dataset variant for parameter-golf sweeps.")
    p.add_argument("--src", required=True, help="Source dataset dir (contains fineweb_train_*.bin and fineweb_val_*.bin)")
    p.add_argument("--dst", required=True, help="Destination dataset dir")
    p.add_argument("--val-tokens", type=int, default=4_194_304, help="Number of validation tokens to keep")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train_files = sorted(src.glob("fineweb_train_*.bin"))
    val_files = sorted(src.glob("fineweb_val_*.bin"))
    if not train_files or not val_files:
        raise RuntimeError(f"Missing train/val shards in {src}")

    # Keep training shard(s) as hardlinks when possible (zero extra space).
    for train in train_files:
        out = dst / train.name
        if out.exists():
            out.unlink()
        os.link(train, out)

    # Build one truncated validation shard for much faster local eval.
    header, tokens = read_shard(val_files[0])
    keep = min(args.val_tokens, int(tokens.size))
    truncated = tokens[:keep]
    write_shard(dst / "fineweb_val_000000.bin", header, truncated)

    print(f"fast-local dataset created: {dst}")
    print(f"train_shards={len(train_files)} val_tokens={keep}")


if __name__ == "__main__":
    main()

