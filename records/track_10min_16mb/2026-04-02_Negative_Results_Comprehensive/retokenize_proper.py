"""Proper Scylla retokenization: split train/val from raw docs, no SP1024 roundtrip.
Matches the official manifest: shuffle with seed 1337, last 50K docs = val.
Memory-efficient: workers read from disk, not from in-memory lists."""
import json
import os
import sys
import time
import random
import math
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue

HEADER_INTS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1
TOKENS_PER_SHARD = 100_000_000
NUM_VAL_DOCS = 50000
SHUFFLE_SEED = 1337


def write_shard(path, tokens):
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = HEADER_MAGIC
    header[1] = HEADER_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


def worker_fn(worker_id, line_indices, docs_path, vocab_path, out_dir, result_queue):
    """Tokenize docs at given line indices (contiguous range) into shards."""
    import tokenmonster
    vocab = tokenmonster.load_multiprocess_safe(vocab_path)

    # Read only our lines from the JSONL
    line_set = set(line_indices)
    buffer = []
    shard_count = 0

    with open(docs_path, "r") as f:
        for line_num, line in enumerate(f):
            if line_num not in line_set:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            if not text:
                continue
            tokens = vocab.tokenize(text)
            buffer.extend(tokens)

            while len(buffer) >= TOKENS_PER_SHARD:
                shard_tokens = np.array(buffer[:TOKENS_PER_SHARD], dtype=np.uint16)
                write_shard(
                    Path(out_dir) / f"fineweb_train_w{worker_id:02d}_{shard_count:04d}.bin",
                    shard_tokens,
                )
                buffer = buffer[TOKENS_PER_SHARD:]
                shard_count += 1

    if buffer:
        write_shard(
            Path(out_dir) / f"fineweb_train_w{worker_id:02d}_{shard_count:04d}.bin",
            np.array(buffer, dtype=np.uint16),
        )
        shard_count += 1

    result_queue.put((worker_id, shard_count))
    print(f"  Worker {worker_id}: {shard_count} shards", flush=True)


def main():
    vocab_path = os.environ.get("VOCAB_PATH", "/workspace/candidate.vocab")
    docs_path = os.environ.get("DOCS_PATH", "/workspace/raw_docs/datasets/docs_selected.jsonl")
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/workspace/fineweb_scylla"))
    num_workers = int(os.environ.get("NUM_WORKERS", "16"))

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Count lines and determine split ---
    print("Counting docs...", flush=True)
    t0 = time.time()
    total_lines = 0
    with open(docs_path, "r") as f:
        for _ in f:
            total_lines += 1
    print(f"Total: {total_lines} docs in {time.time()-t0:.0f}s", flush=True)

    # Shuffle indices (matching official manifest)
    print(f"Shuffling with seed {SHUFFLE_SEED}...", flush=True)
    indices = list(range(total_lines))
    random.seed(SHUFFLE_SEED)
    random.shuffle(indices)

    val_indices = set(indices[-NUM_VAL_DOCS:])
    train_indices = indices[:-NUM_VAL_DOCS]
    print(f"Train: {len(train_indices)} docs, Val: {len(val_indices)} docs", flush=True)

    # --- Step 2: Tokenize val (single process) ---
    print("Tokenizing val docs...", flush=True)
    import tokenmonster
    vocab = tokenmonster.load(vocab_path)
    val_tokens = []
    with open(docs_path, "r") as f:
        for line_num, line in enumerate(f):
            if line_num not in val_indices:
                continue
            doc = json.loads(line)
            text = doc.get("text", "")
            if text:
                val_tokens.extend(vocab.tokenize(text))
    write_shard(out_dir / "fineweb_val_000000.bin", np.array(val_tokens, dtype=np.uint16))
    print(f"Val: {len(val_tokens)} tokens", flush=True)
    del val_tokens, vocab

    # --- Step 3: Split train indices into contiguous chunks for workers ---
    # Sort train indices so each worker processes docs in file order (fast sequential read)
    train_indices.sort()
    chunk_size = math.ceil(len(train_indices) / num_workers)
    chunks = [train_indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_workers)]
    del train_indices

    # --- Step 4: Launch parallel workers ---
    print(f"Tokenizing train with {num_workers} workers...", flush=True)
    t0 = time.time()
    result_queue = Queue()
    workers = []
    for i in range(num_workers):
        p = Process(target=worker_fn, args=(i, chunks[i], docs_path, vocab_path, str(out_dir), result_queue))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort()
    total_shards = sum(r[1] for r in results)
    print(f"Workers done: {total_shards} shards in {time.time()-t0:.0f}s", flush=True)

    # --- Step 5: Rename to sequential ---
    shard_files = []
    for wid in range(num_workers):
        worker_files = sorted(out_dir.glob(f"fineweb_train_w{wid:02d}_*.bin"))
        shard_files.extend(worker_files)

    for idx, f in enumerate(shard_files):
        f.rename(f.parent / f"fineweb_train_{idx:06d}.bin")

    print(f"Done! {len(shard_files)} train + 1 val shards", flush=True)


if __name__ == "__main__":
    main()
