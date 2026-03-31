#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HEADER_WORDS = 256
MAGIC = 20240520
VERSION = 1
HEADER_BYTES = HEADER_WORDS * np.dtype("<i4").itemsize


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_header(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype="<i4", count=HEADER_WORDS)
    if header.size != HEADER_WORDS or int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    return header


def copy_prefix(src_paths: list[Path], num_tokens: int, out_path: Path) -> None:
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    template_header: np.ndarray | None = None
    chunks: list[np.ndarray] = []
    remaining = num_tokens
    for src_path in src_paths:
        header = load_header(src_path)
        if template_header is None:
            template_header = header.copy()
        available = int(header[2])
        take = min(remaining, available)
        tokens = np.fromfile(src_path, dtype="<u2", count=take, offset=HEADER_BYTES)
        if tokens.size != take:
            raise ValueError(f"Short read for {src_path}: expected {take}, got {tokens.size}")
        chunks.append(tokens)
        remaining -= take
        if remaining == 0:
            break
    if template_header is None or remaining > 0:
        raise ValueError(f"Not enough source tokens to write {num_tokens} tokens to {out_path}")
    out_tokens = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
    template_header[2] = num_tokens
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        template_header.tofile(f)
        out_tokens.tofile(f)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Create a small FineWeb-based proxy split from cached challenge shards.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=root / "data/datasets/fineweb10B_sp1024",
        help="Source dataset directory with fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/datasets/fineweb10B_sp1024_proxy",
        help="Output directory for the proxy split",
    )
    parser.add_argument("--train-tokens", type=int, default=1_048_576, help="Number of train tokens to keep")
    parser.add_argument("--val-tokens", type=int, default=262_144, help="Number of held-out tokens to keep")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --force to overwrite.")

    train_paths = sorted(source_dir.glob("fineweb_train_*.bin"))
    val_paths = sorted(source_dir.glob("fineweb_val_*.bin"))
    if not train_paths or not val_paths:
        raise FileNotFoundError(f"Could not find FineWeb shard files under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_prefix(train_paths, args.train_tokens, output_dir / "fineweb_train_000000.bin")
    copy_prefix(val_paths, args.val_tokens, output_dir / "fineweb_val_000000.bin")

    print(f"proxy dataset written to {output_dir}")
    for path in sorted(output_dir.glob("*.bin")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
