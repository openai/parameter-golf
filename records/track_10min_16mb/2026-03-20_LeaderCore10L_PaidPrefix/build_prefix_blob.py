#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
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


def write_raw_prefix(val_dir: str, output: str, *, budget_bytes: int | None = None, token_count: int | None = None) -> tuple[int, int, int]:
    if budget_bytes is None and token_count is None:
        raise ValueError("Either budget_bytes or token_count must be set")
    val_tokens = load_val_tokens(val_dir)
    target_tokens = val_tokens[1:]
    if token_count is None:
        if budget_bytes is None:
            raise ValueError("budget_bytes is required when token_count is not set")
        if budget_bytes % 2 != 0:
            raise ValueError("budget_bytes must be even for raw uint16 token storage")
        token_count = budget_bytes // 2
    token_count = min(token_count, target_tokens.size)
    blob = target_tokens[:token_count].astype("<u2", copy=False).tobytes()
    out = Path(output)
    out.write_bytes(blob)
    return len(blob), token_count, target_tokens.size


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a raw paid-prefix blob from validation targets.")
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-bytes", type=int, default=None)
    parser.add_argument("--token-count", type=int, default=None)
    args = parser.parse_args()

    blob_bytes, tokens, total_tokens = write_raw_prefix(
        args.val_dir,
        args.output,
        budget_bytes=args.budget_bytes,
        token_count=args.token_count,
    )
    print(f"output={args.output}")
    print(f"blob_bytes={blob_bytes}")
    print(f"tokens_covered={tokens}")
    print(f"coverage={tokens / total_tokens:.6f}")


if __name__ == "__main__":
    main()
