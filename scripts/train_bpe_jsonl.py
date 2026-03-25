#!/usr/bin/env python3
"""
Train BESE+BPE merges from a JSONL corpus (e.g. FineWeb docs_selected.jsonl).

Each line must be JSON with a \"text\" field (Parameter Golf convention).

Examples:
  python scripts/train_bpe_jsonl.py --input data/docs_selected.jsonl --output data/tokenizers/bese_bpe_250.json \\
    --num-merges 250 --max-docs 100000

  # Sweep merge counts (writes bese_bpe_{n}.json for each n)
  python scripts/train_bpe_jsonl.py --input data/docs_selected.jsonl --output-dir data/tokenizers \\
    --merge-sweep 100,200,250,300 --max-docs 100000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root: parent of scripts/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tokenizer") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tokenizer"))

from bese_bpe_tokenizer import BESEBPETokenizer, train_bpe_merges  # noqa: E402


def iter_texts(path: Path, max_docs: int | None):
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if max_docs is not None and n >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield obj["text"]
            n += 1


def main() -> None:
    p = argparse.ArgumentParser(description="Train BESE+BPE tokenizer on JSONL text field.")
    p.add_argument("--input", type=Path, required=True, help="JSONL with {\"text\": ...} per line")
    p.add_argument("--output", type=Path, help="Output tokenizer JSON (single run)")
    p.add_argument("--output-dir", type=Path, help="Directory for sweep outputs")
    p.add_argument("--num-merges", type=int, default=250, help="BPE merge count (single run)")
    p.add_argument("--max-docs", type=int, default=100_000, help="Max training documents")
    p.add_argument(
        "--merge-sweep",
        type=str,
        default="",
        help="Comma-separated merge counts (e.g. 100,200,250,300); writes bese_bpe_{n}.json",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    if args.merge_sweep:
        counts = [int(x.strip()) for x in args.merge_sweep.split(",") if x.strip()]
        out_dir = args.output_dir or args.input.parent / "tokenizers"
        out_dir.mkdir(parents=True, exist_ok=True)
        max_docs = args.max_docs
        print(f"Loading up to {max_docs:,} docs from {args.input} ...")
        texts = list(iter_texts(args.input, max_docs))
        print(f"Loaded {len(texts):,} documents.")
        for num in counts:
            print(f"\n=== Training {num} merges ===")
            merges = train_bpe_merges(texts, num_merges=num, verbose=not args.quiet)
            tok = BESEBPETokenizer(merges=merges)
            path = out_dir / f"bese_bpe_{num}.json"
            tok.save(path)
            print(f"Saved {path} (vocab_size={tok.vocab_size})")
        return

    if not args.output:
        raise SystemExit("Provide --output or use --merge-sweep with --output-dir")

    max_docs = args.max_docs
    print(f"Loading up to {max_docs:,} docs from {args.input} ...")
    texts = list(iter_texts(args.input, max_docs))
    print(f"Loaded {len(texts):,} documents.")
    merges = train_bpe_merges(texts, num_merges=args.num_merges, verbose=not args.quiet)
    tok = BESEBPETokenizer(merges=merges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.output)
    print(f"Saved {args.output} (vocab_size={tok.vocab_size})")


if __name__ == "__main__":
    main()
