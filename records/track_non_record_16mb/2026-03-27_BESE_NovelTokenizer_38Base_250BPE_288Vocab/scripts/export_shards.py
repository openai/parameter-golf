#!/usr/bin/env python3
"""
Export FineWeb-style binary token shards compatible with parameter-golf train_gpt.py.

Shard format (matches upstream load_data_shard):
  - 256 x int32 header (little-endian): header[0]=20240520, header[1]=1, header[2]=num_tokens
  - num_tokens x uint16 token ids (little-endian)

Examples:
  python scripts/export_shards.py \\
    --input data/docs_selected.jsonl \\
    --tokenizer data/tokenizers/bese_bpe_250.json \\
    --output-dir data/datasets/fineweb10B_bese250 \\
    --shard-tokens 100000000 \\
    --train-prefix fineweb_train_ \\
    --val-prefix fineweb_val_

Validation: for each encoded document, sum(bytes_per_token) must equal UTF-8 byte length.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tokenizer") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tokenizer"))

from bese_fast_bpe import FastBESEBPETokenizer  # noqa: E402

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def write_shard(path: Path, tokens: np.ndarray) -> None:
    """Write uint16 tokens with fixed header."""
    if tokens.dtype != np.uint16:
        tokens = tokens.astype(np.uint16)
    n = int(tokens.shape[0])
    header = np.zeros(HEADER_INTS, dtype="<i4")  # match train_gpt load_data_shard
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = n
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header.tobytes())
        # uint16 tokens, little-endian (matches upstream)
        f.write(tokens.astype("<u2").tobytes())


def iter_docs(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]


def encode_doc(tok: FastBESEBPETokenizer, text: str, bpt: np.ndarray) -> np.ndarray:
    enc = tok.encode(text)
    tb = int(sum(bpt[t] for t in enc))
    ub = len(text.encode("utf-8"))
    if tb != ub:
        raise ValueError(f"Byte mismatch: token_bytes={tb} utf8={ub} text[:80]={text[:80]!r}")
    return enc


def main() -> None:
    ap = argparse.ArgumentParser(description="Export BESE+BPE token shards for Parameter Golf")
    ap.add_argument("--input", type=Path, required=True, help="JSONL with text field")
    ap.add_argument("--tokenizer", type=Path, required=True, help="BESE+BPE tokenizer JSON")
    ap.add_argument("--output-dir", type=Path, required=True, help="Dataset directory for .bin shards")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000, help="Target tokens per train shard")
    ap.add_argument("--val-docs", type=int, default=50_000, help="First N docs for validation (default 50k)")
    ap.add_argument("--train-prefix", type=str, default="fineweb_train_")
    ap.add_argument("--val-prefix", type=str, default="fineweb_val_")
    args = ap.parse_args()

    tok = FastBESEBPETokenizer.load(str(args.tokenizer))
    bpt = tok.get_bytes_per_token_lut()

    print(f"Tokenizer vocab_size={tok.vocab_size}, merges={len(tok.merges)}")

    # Stream documents: first pass collects val docs, second pass streams training docs.
    # This avoids loading the entire corpus into memory at once.
    val_n = args.val_docs
    val_texts: list[str] = []
    doc_iter = iter_docs(args.input)

    for text in doc_iter:
        val_texts.append(text)
        if len(val_texts) >= val_n:
            break

    if len(val_texts) < val_n:
        print(f"Warning: only {len(val_texts)} docs; adjusting val to {len(val_texts)//10 or 1}")
        actual_val_n = max(1, len(val_texts) // 10)
        # Remaining val texts become training texts
        overflow_texts = val_texts[actual_val_n:]
        val_texts = val_texts[:actual_val_n]
    else:
        overflow_texts = []

    print(f"Encoding {len(val_texts)} validation docs...")
    val_chunks: list[np.ndarray] = []
    for text in val_texts:
        val_chunks.append(encode_doc(tok, text, bpt))
    val_tokens = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=np.uint16)
    del val_chunks  # free memory
    print(f"Val tokens: {val_tokens.shape[0]:,}")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    val_path = out / f"{args.val_prefix}0.bin"
    write_shard(val_path, val_tokens)
    del val_tokens  # free memory
    print(f"Wrote {val_path}")

    # Stream training docs: chain any overflow from val split with remaining docs from file
    import itertools
    train_iter = itertools.chain(overflow_texts, doc_iter)

    print("Encoding training stream...")
    shard_idx = 0
    current: list[np.ndarray] = []
    current_count = 0
    train_doc_count = 0

    def flush_train():
        nonlocal shard_idx, current, current_count
        if not current:
            return
        buf = np.concatenate(current)
        name = f"{args.train_prefix}{shard_idx}.bin"
        write_shard(out / name, buf)
        print(f"Wrote {out / name} ({buf.shape[0]:,} tokens)")
        shard_idx += 1
        current = []
        current_count = 0

    for text in train_iter:
        arr = encode_doc(tok, text, bpt)
        current.append(arr)
        current_count += arr.shape[0]
        train_doc_count += 1
        if current_count >= args.shard_tokens:
            flush_train()

    flush_train()
    print(f"Processed {train_doc_count} training docs.")

    manifest = {
        "tokenizer_name": args.tokenizer.stem,
        "tokenizer_path": str(args.tokenizer.resolve()),
        "vocab_size": tok.vocab_size,
        "shard_magic": SHARD_MAGIC,
        "val_docs": len(val_texts),
        "train_docs": train_doc_count,
    }
    (out / "bese_shard_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Done. Manifest: {out / 'bese_shard_manifest.json'}")


if __name__ == "__main__":
    main()
