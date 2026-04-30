#!/usr/bin/env python3
"""
Decode human-readable text from Parameter Golf FineWeb .bin shards.

Requires:
  - data from: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
  - sentencepiece

Example:
  python3 scripts/sample_fineweb_tokens.py --shard val --num-samples 3 --length 80
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
DEFAULT_SPM = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"


def load_tokens(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens


def main() -> int:
    p = argparse.ArgumentParser(description="Sample and decode FineWeb challenge tokens.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Dataset directory")
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_SPM, help="SentencePiece .model")
    p.add_argument("--shard", choices=("val", "train"), default="val", help="Which shard file to read")
    p.add_argument("--num-samples", type=int, default=5, help="Number of random windows")
    p.add_argument("--length", type=int, default=64, help="Tokens per window")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (0 = nondeterministic)")
    args = p.parse_args()

    if not args.tokenizer.is_file():
        print(f"Missing tokenizer: {args.tokenizer}")
        return 1
    if not args.data_dir.is_dir():
        print(f"Missing data dir: {args.data_dir}")
        print("Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1")
        return 1

    pattern = "fineweb_val_*.bin" if args.shard == "val" else "fineweb_train_*.bin"
    files = sorted(args.data_dir.glob(pattern))
    if not files:
        print(f"No {pattern} under {args.data_dir}")
        return 1

    path = files[0]
    print(f"shard_file: {path.name} ({path.stat().st_size // 1_000_000} MiB on disk)")
    tokens = load_tokens(path)
    print(f"num_tokens: {tokens.size:,} (uint16 ids, vocab 1024 for sp1024)")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    if int(sp.vocab_size()) != 1024:
        print(f"warning: vocab_size={sp.vocab_size()} (expected 1024 for default challenge)")

    if args.seed:
        random.seed(args.seed)
    ntok = tokens.size
    L = min(args.length, ntok - 1)
    if L < 8:
        print("Shard too short for sampling")
        return 1

    print()
    for i in range(args.num_samples):
        start = random.randint(0, ntok - L - 1)
        chunk = tokens[start : start + L].tolist()
        text = sp.decode_ids(chunk)
        preview = text.replace("\n", "\\n")
        if len(preview) > 500:
            preview = preview[:500] + "…"
        print(f"--- sample {i + 1}  token_offset={start}  len={L} ---")
        print(preview)
        print()

    print(
        "Note: this is raw decoded subword text (BPE). "
        "Training uses next-token prediction on these ids; BPB measures compression vs bytes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
