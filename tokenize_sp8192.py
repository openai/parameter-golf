"""Tokenize docs_selected.jsonl with SP8192 tokenizer into challenge format shards."""
import json
import os
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

MAGIC = 20240520
VERSION = 1
HEADER_INTS = 256
NUM_VAL_DOCS = 50_000
SHARD_SIZE = 10**8
SP_BATCH = 2048
SP_THREADS = min(os.cpu_count() or 8, 128)

DOCS = Path("data/docs_selected.jsonl")
DST = Path("data/datasets/fineweb10B_sp8192")
MODEL = "data/tokenizers/fineweb_8192_bpe.model"


def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def main():
    DST.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=MODEL)
    bos = sp.bos_id()

    buf = np.empty((SHARD_SIZE,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}
    doc_count = 0
    total_tokens = {"val": 0, "train": 0}

    def flush():
        nonlocal fill
        if fill == 0:
            return
        path = DST / f"fineweb_{split}_{shards[split]:06d}.bin"
        write_shard(path, buf[:fill])
        print(f"  {path.name}: {fill:,} tokens", flush=True)
        shards[split] += 1
        fill = 0

    def add_tokens(toks):
        nonlocal fill
        pos = 0
        while pos < len(toks):
            take = min(SHARD_SIZE - fill, len(toks) - pos)
            buf[fill:fill + take] = toks[pos:pos + take]
            fill += take
            pos += take
            if fill == SHARD_SIZE:
                flush()

    t0 = time.time()
    batch_texts = []

    with open(DOCS, "r", encoding="utf-8") as f:
        for line in f:
            batch_texts.append(json.loads(line)["text"])
            if len(batch_texts) >= SP_BATCH:
                encoded_batch = sp.encode(batch_texts, out_type=int, num_threads=SP_THREADS)
                for encoded in encoded_batch:
                    if doc_count == NUM_VAL_DOCS:
                        flush()
                        split = "train"
                    toks = np.array([bos] + encoded, dtype=np.uint16)
                    total_tokens[split] += len(toks)
                    add_tokens(toks)
                    doc_count += 1
                    if doc_count % 500_000 == 0:
                        elapsed = time.time() - t0
                        tps = sum(total_tokens.values()) / elapsed
                        print(f"  docs: {doc_count:,}, tokens: {sum(total_tokens.values()):,}, "
                              f"time: {elapsed:.0f}s, {tps / 1e6:.1f}M tok/s", flush=True)
                batch_texts = []

    if batch_texts:
        encoded_batch = sp.encode(batch_texts, out_type=int, num_threads=SP_THREADS)
        for encoded in encoded_batch:
            if doc_count == NUM_VAL_DOCS:
                flush()
                split = "train"
            toks = np.array([bos] + encoded, dtype=np.uint16)
            total_tokens[split] += len(toks)
            add_tokens(toks)
            doc_count += 1

    flush()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Docs: {doc_count:,} (val: {min(doc_count, NUM_VAL_DOCS):,}, "
          f"train: {max(0, doc_count - NUM_VAL_DOCS):,})")
    print(f"Val tokens: {total_tokens['val']:,} in {shards['val']} shards")
    print(f"Train tokens: {total_tokens['train']:,} in {shards['train']} shards")
    print(f"Total tokens: {sum(total_tokens.values()):,}")


if __name__ == "__main__":
    main()
