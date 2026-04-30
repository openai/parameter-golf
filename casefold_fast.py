#!/usr/bin/env python3
"""Fast casefold retokenization using multiprocessing."""
import json
import os
import sys
import time
import unicodedata
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import sentencepiece as spm

DOCS_PATH = Path("data/docs_selected.jsonl")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_8192_bpe_casefold.model")
DATASET_DIR = Path("data/datasets/fineweb10B_sp8192_casefold")
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8

# Global tokenizer (loaded per process)
_sp = None

def _init_worker():
    global _sp
    _sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

def _process_line(line):
    doc = json.loads(line)
    text = doc.get("text", "")
    if not text:
        return None
    cf = unicodedata.normalize('NFKC', text).lower()
    tokens = _sp.encode(cf)
    return tokens

def main():
    if not TOKENIZER_PATH.exists():
        print(f"Tokenizer not found: {TOKENIZER_PATH}")
        print("Run: python3 casefold_retokenize.py --train-tokenizer")
        sys.exit(1)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    nproc = cpu_count() or 8
    print(f"=== Fast casefold tokenization ({nproc} workers) ===")

    # Read all lines
    print(f"Reading {DOCS_PATH}...")
    with open(DOCS_PATH) as f:
        lines = f.readlines()
    total = len(lines)
    print(f"Total docs: {total:,}")

    # Process in parallel
    t0 = time.time()
    val_tokens = []
    train_tokens = []

    batch_size = 10000
    processed = 0

    with Pool(nproc, initializer=_init_worker) as pool:
        for i in range(0, total, batch_size):
            batch = lines[i:i+batch_size]
            results = pool.map(_process_line, batch)
            for tokens in results:
                if tokens is None:
                    continue
                if processed < NUM_VAL_DOCS:
                    val_tokens.extend(tokens)
                else:
                    train_tokens.extend(tokens)
                processed += 1

            elapsed = time.time() - t0
            rate = processed / elapsed
            if (i // batch_size) % 10 == 0:
                print(f"  {processed:,}/{total:,} ({rate:.0f} docs/s, {len(val_tokens)+len(train_tokens):,} tokens)")

    elapsed = time.time() - t0
    print(f"\nDone: {processed:,} docs, {len(val_tokens)+len(train_tokens):,} tokens in {elapsed:.0f}s")

    # Save val
    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_path = DATASET_DIR / "fineweb_val_000000.bin"
    val_arr.tofile(str(val_path))
    print(f"Val: {val_path} ({len(val_arr):,} tokens)")

    # Save train shards
    train_arr = np.array(train_tokens, dtype=np.uint16)
    n_shards = max(1, (len(train_arr) + SHARD_SIZE - 1) // SHARD_SIZE)
    for i in range(n_shards):
        s = i * SHARD_SIZE
        e = min(s + SHARD_SIZE, len(train_arr))
        shard = train_arr[s:e]
        path = DATASET_DIR / f"fineweb_train_{i:06d}.bin"
        shard.tofile(str(path))
        print(f"Train shard {i}: {path} ({len(shard):,} tokens)")

    print(f"\nDataset: {DATASET_DIR}")
    print(f"To train:")
    print(f"  VOCAB_SIZE=8192 DATA_PATH={DATASET_DIR}/ TOKENIZER_PATH={TOKENIZER_PATH} USE_ANS=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py")

if __name__ == "__main__":
    main()
