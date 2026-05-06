#!/usr/bin/env python3
"""Stream casefold tokenization — processes docs one at a time, writes shards immediately.
No RAM explosion. Should finish in ~30-60 minutes."""
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm

DOCS_PATH = Path("data/docs_selected.jsonl")
TOKENIZER_PATH = Path("data/tokenizers/fineweb_8192_bpe_casefold.model")
DATASET_DIR = Path("data/datasets/fineweb10B_sp8192_casefold")
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8  # 100M tokens per shard

def main():
    if not TOKENIZER_PATH.exists():
        print(f"ERROR: {TOKENIZER_PATH} not found")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    print(f"Tokenizer: {TOKENIZER_PATH} (vocab={sp.get_piece_size()})")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_docs = 0
    total_tokens = 0

    # Buffers
    val_buf = []
    train_buf = []
    train_shard_idx = 0
    val_written = False

    with open(DOCS_PATH) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue

            # Casefold
            cf = unicodedata.normalize('NFKC', text).lower()
            tokens = sp.encode(cf)
            total_tokens += len(tokens)

            if n_docs < NUM_VAL_DOCS:
                val_buf.extend(tokens)
            else:
                train_buf.extend(tokens)

                # Write shard when buffer is full
                if len(train_buf) >= SHARD_SIZE:
                    arr = np.array(train_buf[:SHARD_SIZE], dtype=np.uint16)
                    path = DATASET_DIR / f"fineweb_train_{train_shard_idx:06d}.bin"
                    arr.tofile(str(path))
                    print(f"  Shard {train_shard_idx}: {path} ({len(arr):,} tokens)")
                    train_buf = train_buf[SHARD_SIZE:]
                    train_shard_idx += 1

            n_docs += 1

            # Write val when done with val docs
            if n_docs == NUM_VAL_DOCS and not val_written:
                arr = np.array(val_buf, dtype=np.uint16)
                path = DATASET_DIR / "fineweb_val_000000.bin"
                arr.tofile(str(path))
                print(f"  Val: {path} ({len(arr):,} tokens)")
                val_buf = []
                val_written = True

            if n_docs % 100_000 == 0:
                elapsed = time.time() - t0
                rate = n_docs / elapsed
                print(f"  {n_docs:,}/15,368,808 ({rate:.0f} docs/s, {total_tokens:,} tokens, shards={train_shard_idx})")

    # Write remaining train tokens
    if train_buf:
        arr = np.array(train_buf, dtype=np.uint16)
        path = DATASET_DIR / f"fineweb_train_{train_shard_idx:06d}.bin"
        arr.tofile(str(path))
        print(f"  Shard {train_shard_idx}: {path} ({len(arr):,} tokens)")
        train_shard_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone: {n_docs:,} docs, {total_tokens:,} tokens in {elapsed:.0f}s")
    print(f"Val: 1 shard, Train: {train_shard_idx} shards")
    print(f"\nTo train:")
    print(f"  VOCAB_SIZE=8192 DATA_PATH={DATASET_DIR}/ TOKENIZER_PATH={TOKENIZER_PATH} USE_ANS=1 torchrun --standalone --nproc_per_node=8 train_gpt_ans.py")

if __name__ == "__main__":
    main()
