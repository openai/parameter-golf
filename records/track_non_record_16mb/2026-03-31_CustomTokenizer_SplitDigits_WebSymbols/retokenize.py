"""Train a custom SentencePiece tokenizer and retokenize docs_selected.jsonl into binary shards.

Usage:
    # Full pipeline: train tokenizer + retokenize
    uv run data/retokenize.py

    # Only train tokenizer
    uv run data/retokenize.py --skip-retokenize

    # Only retokenize (reuse existing .model)
    uv run data/retokenize.py --skip-train-tokenizer

    # Limit training shards for faster iteration
    uv run data/retokenize.py --train-shards 10
"""
import argparse
import functools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Force unbuffered output so progress is visible when piped
print = functools.partial(print, flush=True)

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256
DEFAULT_SHARD_SIZE = 100_000_000  # tokens per shard (from manifest)
DEFAULT_NUM_VAL_DOCS = 50_000
DEFAULT_TOKENIZER_TRAIN_DOCS = 5_000_000
DEFAULT_SHUFFLE_SEED = 1337

USER_DEFINED_SYMBOLS = [
    "http://", "https://", "www.",
    ".com", ".org", ".net",
    ".gov", ".html", ".edu", ".co.uk",
]


def iter_jsonl_texts(path: Path, limit: int | None = None):
    """Yield document texts from a JSONL file, one per line."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text:
                    yield text
                    count += 1
            except json.JSONDecodeError:
                continue


def train_tokenizer(
    input_path: Path,
    model_prefix: str,
    vocab_size: int,
    num_train_docs: int,
):
    """Train a SentencePiece BPE tokenizer on the first num_train_docs documents."""
    import sentencepiece as spm

    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size})...")
    print(f"  Training docs: {num_train_docs:,}")
    print(f"  user_defined_symbols: {USER_DEFINED_SYMBOLS}")
    print(f"  split_digits: False")
    print(f"  max_sentence_length: 16384")
    print(f"  Output: {model_prefix}.model")

    # Create an iterator that yields text strings for training
    doc_iter = iter_jsonl_texts(input_path, limit=num_train_docs)

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        sentence_iterator=doc_iter,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        # User's requested settings
        split_digits=False,
        user_defined_symbols=USER_DEFINED_SYMBOLS,
        max_sentence_length=16384,
        # Required for byte-level coverage (BPB scoring uses sp.is_byte())
        byte_fallback=True,
        character_coverage=0.9995,
        # Match original tokenizer control token IDs
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        # Performance
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
    )
    elapsed = time.time() - t0
    print(f"  Tokenizer trained in {elapsed:.1f}s")

    # Verify
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    actual_vocab = sp.vocab_size()
    print(f"  Actual vocab size: {actual_vocab}")
    if actual_vocab != vocab_size:
        print(f"  WARNING: vocab size mismatch! Expected {vocab_size}, got {actual_vocab}")
        sys.exit(1)

    # Verify user_defined_symbols are single tokens
    for sym in USER_DEFINED_SYMBOLS:
        tokens = sp.encode(sym, out_type=str)
        token_strs = [t.replace("\u2581", "_") for t in tokens]  # safe for Windows console
        if len(tokens) != 1 or tokens[0].replace("\u2581", "") != sym:
            encoded_ids = sp.encode(sym)
            print(f"  Note: '{sym}' encodes as {token_strs} (ids: {encoded_ids})")

    print("  Tokenizer training complete.")
    return sp


class ShardWriter:
    """Accumulates token IDs and writes shards in the competition binary format."""

    def __init__(self, output_dir: Path, prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.buffer = []
        self.buffer_len = 0
        self.total_tokens = 0

    def add_tokens(self, ids: list[int]):
        self.buffer.extend(ids)
        self.buffer_len += len(ids)
        while self.buffer_len >= self.shard_size:
            self._flush_shard(self.shard_size)

    def _flush_shard(self, count: int):
        tokens = np.array(self.buffer[:count], dtype="<u2")
        self.buffer = self.buffer[count:]
        self.buffer_len = len(self.buffer)
        self._write_shard(tokens)

    def _write_shard(self, tokens: np.ndarray):
        path = self.output_dir / f"{self.prefix}_{self.shard_idx:06d}.bin"
        header = np.zeros(HEADER_INTS, dtype="<i4")
        header[0] = SHARD_MAGIC
        header[1] = SHARD_VERSION
        header[2] = len(tokens)
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.tobytes())
        self.total_tokens += len(tokens)
        self.shard_idx += 1

    def close(self):
        """Flush any remaining tokens as a final shard."""
        if self.buffer_len > 0:
            tokens = np.array(self.buffer, dtype="<u2")
            self._write_shard(tokens)
            self.buffer = []
            self.buffer_len = 0


def retokenize(
    input_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    shard_size: int,
    num_val_docs: int,
    shuffle_seed: int,
    max_train_shards: int | None,
):
    """Retokenize the full corpus into binary shards."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    print(f"Retokenizing with {tokenizer_path} (vocab_size={sp.vocab_size()})")
    print(f"  Output: {output_dir}")
    print(f"  Shard size: {shard_size:,} tokens")
    print(f"  Val docs: {num_val_docs:,}")
    print(f"  Shuffle seed: {shuffle_seed}")

    # Read all document texts and shuffle to match original ordering
    print("  Reading all documents (this may take a while)...")
    t0 = time.time()
    texts = list(iter_jsonl_texts(input_path))
    print(f"  Read {len(texts):,} documents in {time.time() - t0:.1f}s")

    print(f"  Shuffling with seed {shuffle_seed}...")
    random.seed(shuffle_seed)
    random.shuffle(texts)

    val_texts = texts[:num_val_docs]
    train_texts = texts[num_val_docs:]
    del texts  # free memory

    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize validation set
    print(f"\n  Tokenizing validation set ({len(val_texts):,} docs)...")
    val_writer = ShardWriter(output_dir, "fineweb_val", shard_size)
    t0 = time.time()
    for i, text in enumerate(val_texts):
        ids = sp.encode(text)
        val_writer.add_tokens(ids)
        if (i + 1) % 10_000 == 0:
            print(f"    val: {i+1:,}/{len(val_texts):,} docs")
    val_writer.close()
    val_elapsed = time.time() - t0
    print(f"  Val complete: {val_writer.total_tokens:,} tokens in {val_writer.shard_idx} shard(s) ({val_elapsed:.1f}s)")

    # Tokenize training set
    max_train_tokens = max_train_shards * shard_size if max_train_shards else None
    print(f"\n  Tokenizing training set ({len(train_texts):,} docs)...")
    if max_train_shards:
        print(f"    Limiting to {max_train_shards} shards ({max_train_tokens:,} tokens)")
    train_writer = ShardWriter(output_dir, "fineweb_train", shard_size)
    t0 = time.time()
    for i, text in enumerate(train_texts):
        ids = sp.encode(text)
        train_writer.add_tokens(ids)
        if max_train_tokens and train_writer.total_tokens + train_writer.buffer_len >= max_train_tokens:
            # Stop after enough tokens for the requested number of shards
            break
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(train_texts) - i - 1) / rate if rate > 0 else 0
            print(f"    train: {i+1:,}/{len(train_texts):,} docs ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")
    train_writer.close()
    train_elapsed = time.time() - t0
    print(f"  Train complete: {train_writer.total_tokens:,} tokens in {train_writer.shard_idx} shard(s) ({train_elapsed:.1f}s)")

    # Summary
    total = val_writer.total_tokens + train_writer.total_tokens
    print(f"\n  Summary:")
    print(f"    Total tokens: {total:,}")
    print(f"    Val tokens:   {val_writer.total_tokens:,} ({val_writer.shard_idx} shards)")
    print(f"    Train tokens: {train_writer.total_tokens:,} ({train_writer.shard_idx} shards)")
    print(f"    Avg tokens/doc (val): {val_writer.total_tokens / len(val_texts):,.1f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train custom tokenizer and retokenize corpus")
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).parent / "docs_selected.jsonl",
        help="Path to docs_selected.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "datasets" / "fineweb10B_sp1024_custom",
        help="Output directory for binary shards",
    )
    parser.add_argument(
        "--tokenizer-prefix", type=str,
        default=str(Path(__file__).parent / "tokenizers" / "fineweb_1024_custom"),
        help="Output prefix for tokenizer model (without .model extension)",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--train-docs", type=int, default=DEFAULT_TOKENIZER_TRAIN_DOCS,
                        help="Number of docs for tokenizer training")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help="Tokens per shard")
    parser.add_argument("--num-val-docs", type=int, default=DEFAULT_NUM_VAL_DOCS)
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--train-shards", type=int, default=None,
                        help="Limit number of training shards to produce")
    parser.add_argument("--skip-train-tokenizer", action="store_true",
                        help="Skip tokenizer training, reuse existing .model")
    parser.add_argument("--skip-retokenize", action="store_true",
                        help="Only train tokenizer, skip retokenization")
    return parser


def main():
    args = build_parser().parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Run:")
        print(f"  uv run data/cached_challenge_fineweb.py --with-docs --train-shards 0")
        sys.exit(1)

    tokenizer_model_path = Path(f"{args.tokenizer_prefix}.model")

    # Phase A: Train tokenizer
    if not args.skip_train_tokenizer:
        Path(args.tokenizer_prefix).parent.mkdir(parents=True, exist_ok=True)
        train_tokenizer(
            input_path=args.input,
            model_prefix=args.tokenizer_prefix,
            vocab_size=args.vocab_size,
            num_train_docs=args.train_docs,
        )
    else:
        if not tokenizer_model_path.exists():
            print(f"ERROR: --skip-train-tokenizer but {tokenizer_model_path} not found")
            sys.exit(1)
        print(f"Skipping tokenizer training, using {tokenizer_model_path}")

    # Phase B: Retokenize
    if not args.skip_retokenize:
        retokenize(
            input_path=args.input,
            tokenizer_path=tokenizer_model_path,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            num_val_docs=args.num_val_docs,
            shuffle_seed=args.shuffle_seed,
            max_train_shards=args.train_shards,
        )
    else:
        print("Skipping retokenization.")

    print("\nDone! To train with the custom tokenizer:")
    print(f"  TOKENIZER_PATH={tokenizer_model_path} \\")
    print(f"  DATA_PATH={args.output_dir} \\")
    print(f"  torchrun --standalone --nproc_per_node=1 train_gpt.py")


if __name__ == "__main__":
    main()
