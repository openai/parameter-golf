#!/usr/bin/env python3
"""
Casefold + retokenize FineWeb data for Parameter Golf.

Takes docs_selected.jsonl (raw text), applies NFKC normalization + lowercasing,
then tokenizes with the casefold SP8192 tokenizer and saves as .bin shards.

Usage:
  # Step 1: Train casefold tokenizer on the data
  python3 casefold_retokenize.py --train-tokenizer

  # Step 2: Retokenize all documents
  python3 casefold_retokenize.py --tokenize

  # Or do both:
  python3 casefold_retokenize.py --train-tokenizer --tokenize
"""

import argparse
import json
import os
import struct
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

DOCS_PATH = Path("data/docs_selected.jsonl")
TOKENIZER_DIR = Path("data/tokenizers")
DATASET_DIR = Path("data/datasets/fineweb10B_sp8192_casefold")
VOCAB_SIZE = 8192
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8  # tokens per shard
SP_MODEL_PATH = TOKENIZER_DIR / "fineweb_8192_bpe_casefold.model"


def casefold_text(text: str) -> str:
    """NFKC normalize + lowercase. Same transform as PR #1578/#1585."""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    return text


def train_tokenizer():
    """Train a SentencePiece BPE tokenizer on casefolded text."""
    import sentencepiece as spm
    import tempfile

    print("=== Training casefold SP8192 tokenizer ===")
    print(f"Reading docs from {DOCS_PATH}...")

    # Write casefolded text to temp file for SP training
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    n_docs = 0
    max_train_docs = 500_000  # Use subset for tokenizer training (faster)

    with open(DOCS_PATH) as f:
        for line in f:
            if n_docs >= max_train_docs:
                break
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
            text = casefold_text(text)
            tmp.write(text + "\n")
            n_docs += 1
            if n_docs % 50_000 == 0:
                print(f"  Processed {n_docs:,} docs for tokenizer training...")

    tmp.close()
    print(f"  Total: {n_docs:,} docs written to temp file")

    # Train SentencePiece
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    model_prefix = str(SP_MODEL_PATH).replace('.model', '')

    print(f"  Training SentencePiece BPE (vocab={VOCAB_SIZE})...")
    spm.SentencePieceTrainer.train(
        input=tmp.name,
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type='bpe',
        character_coverage=1.0,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        byte_fallback=True,
    )

    os.unlink(tmp.name)
    print(f"  Tokenizer saved: {SP_MODEL_PATH}")

    # Verify
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))
    print(f"  Vocab size: {sp.get_piece_size()}")
    test = "the quick brown fox jumps over the lazy dog"
    tokens = sp.encode(test)
    print(f"  Test: '{test}' → {len(tokens)} tokens")


def tokenize_docs():
    """Tokenize all documents with casefold + SP8192 and save as .bin shards."""
    import sentencepiece as spm

    print("=== Tokenizing with casefold SP8192 ===")

    if not SP_MODEL_PATH.exists():
        print(f"ERROR: Tokenizer not found at {SP_MODEL_PATH}")
        print("Run with --train-tokenizer first")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL_PATH))
    print(f"  Tokenizer: {SP_MODEL_PATH} (vocab={sp.get_piece_size()})")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Count docs first
    print(f"  Counting docs in {DOCS_PATH}...")
    total_docs = sum(1 for _ in open(DOCS_PATH))
    print(f"  Total docs: {total_docs:,}")

    # Tokenize all docs, split into val (first 50K) and train (rest)
    all_tokens_val = []
    all_tokens_train = []
    val_byte_counts = []
    train_byte_counts = []

    t0 = time.time()
    n_docs = 0
    total_tokens = 0

    with open(DOCS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            # Casefold
            cf_text = casefold_text(text)
            original_bytes = len(text.encode('utf-8'))

            # Tokenize
            tokens = sp.encode(cf_text)
            total_tokens += len(tokens)

            if n_docs < NUM_VAL_DOCS:
                all_tokens_val.extend(tokens)
                val_byte_counts.extend([original_bytes] * len(tokens))
            else:
                all_tokens_train.extend(tokens)
                train_byte_counts.extend([original_bytes] * len(tokens))

            n_docs += 1
            if n_docs % 100_000 == 0:
                elapsed = time.time() - t0
                rate = n_docs / elapsed
                print(f"  {n_docs:,}/{total_docs:,} docs ({rate:.0f} docs/s, {total_tokens:,} tokens)")

    elapsed = time.time() - t0
    print(f"  Done: {n_docs:,} docs, {total_tokens:,} tokens in {elapsed:.1f}s")
    print(f"  Val tokens: {len(all_tokens_val):,}")
    print(f"  Train tokens: {len(all_tokens_train):,}")
    print(f"  Tokens/byte: {total_tokens / sum(len(json.loads(l).get('text','').encode('utf-8')) for l in open(DOCS_PATH) if json.loads(l).get('text','')):.4f}" if False else "")

    # Save val shard
    val_arr = np.array(all_tokens_val, dtype=np.uint16)
    val_path = DATASET_DIR / "fineweb_val_000000.bin"
    val_arr.tofile(str(val_path))
    print(f"  Saved val: {val_path} ({val_arr.nbytes / 1e6:.1f} MB, {len(val_arr):,} tokens)")

    # Save train shards
    train_arr = np.array(all_tokens_train, dtype=np.uint16)
    n_shards = max(1, len(train_arr) // SHARD_SIZE + (1 if len(train_arr) % SHARD_SIZE else 0))

    for i in range(n_shards):
        start = i * SHARD_SIZE
        end = min((i + 1) * SHARD_SIZE, len(train_arr))
        shard = train_arr[start:end]
        shard_path = DATASET_DIR / f"fineweb_train_{i:06d}.bin"
        shard.tofile(str(shard_path))
        print(f"  Saved train shard {i}: {shard_path} ({shard.nbytes / 1e6:.1f} MB, {len(shard):,} tokens)")

    # Copy tokenizer files to expected location
    vocab_path = str(SP_MODEL_PATH).replace('.model', '.vocab')
    if Path(vocab_path).exists():
        import shutil
        shutil.copy(vocab_path, TOKENIZER_DIR / "fineweb_8192_bpe_casefold.vocab")

    print(f"\n=== Done ===")
    print(f"  Dataset: {DATASET_DIR}")
    print(f"  Tokenizer: {SP_MODEL_PATH}")
    print(f"  Val: 1 shard, {len(all_tokens_val):,} tokens")
    print(f"  Train: {n_shards} shards, {len(all_tokens_train):,} tokens")
    print(f"\nTo train:")
    print(f"  VOCAB_SIZE=8192 DATA_PATH={DATASET_DIR}/ TOKENIZER_PATH={SP_MODEL_PATH} USE_ANS=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py")


def main():
    parser = argparse.ArgumentParser(description="Casefold + retokenize for Parameter Golf")
    parser.add_argument("--train-tokenizer", action="store_true", help="Train casefold SP8192 tokenizer")
    parser.add_argument("--tokenize", action="store_true", help="Tokenize all docs with casefold")

    args = parser.parse_args()

    if not args.train_tokenizer and not args.tokenize:
        parser.print_help()
        return

    if not DOCS_PATH.exists():
        print(f"ERROR: {DOCS_PATH} not found.")
        print("Download it first:")
        print("  python3 data/cached_challenge_fineweb.py --variant sp1024 --with-docs --train-shards 0")
        sys.exit(1)

    if args.train_tokenizer:
        train_tokenizer()

    if args.tokenize:
        tokenize_docs()


if __name__ == "__main__":
    main()
