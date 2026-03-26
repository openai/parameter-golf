#!/usr/bin/env python3
"""
All-in-one RunPod script: decode text, train BESE BPE, export shards, run comparison.
Designed to run as a single command on the pod with no interruptions.

Usage (on the pod):
  cd /workspace && python3 bese/scripts/runpod_all_in_one.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Paths
PG_DIR = Path("/workspace/parameter-golf")
BESE_DIR = Path("/workspace/bese")
SP_MODEL = PG_DIR / "data/tokenizers/fineweb_1024_bpe.model"
SHARD_DIR = PG_DIR / "data/datasets/fineweb10B_sp1024"
TOK_DIR = BESE_DIR / "tokenizers"
BESE_SHARD_DIR = Path("/workspace/bese_shards")

MAX_DOCS = 10_000
NUM_MERGES = 250

sys.path.insert(0, str(BESE_DIR / "tokenizer"))


def step(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def decode_shards():
    """Decode SP binary shard back to text JSONL."""
    step(f"STEP 1: Decoding up to {MAX_DOCS} docs from SP shard")
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    bos = sp.bos_id()

    shard = SHARD_DIR / "fineweb_train_000000.bin"
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(shard, dtype="<i4", count=256)
    n = int(header[2])
    tokens = np.fromfile(shard, dtype="<u2", count=n, offset=header_bytes).tolist()
    print(f"  Loaded {n:,} tokens")

    docs = []
    current = []
    for t in tokens:
        if t == bos:
            if current:
                text = sp.decode(current)
                if len(text.strip()) > 50:
                    docs.append(text)
                    if len(docs) >= MAX_DOCS:
                        break
            current = []
        else:
            current.append(t)

    out = BESE_DIR / "decoded_docs.jsonl"
    with out.open("w") as f:
        for d in docs:
            f.write(json.dumps({"text": d}) + "\n")

    print(f"  Decoded {len(docs)} documents -> {out}")
    return out, docs


def train_bpe(texts):
    """Train BESE BPE on decoded texts."""
    step(f"STEP 2: Training BESE BPE ({NUM_MERGES} merges on {len(texts)} docs)")
    from bese_bpe_tokenizer import BESEBPETokenizer, train_bpe_merges

    t0 = time.time()
    merges = train_bpe_merges(texts, num_merges=NUM_MERGES, verbose=True)
    tok = BESEBPETokenizer(merges=merges)
    elapsed = time.time() - t0
    print(f"  BPE training took {elapsed:.1f}s")
    print(f"  Vocab size: {tok.vocab_size}")

    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = TOK_DIR / f"bese_bpe_{NUM_MERGES}.json"
    tok.save(tok_path)
    print(f"  Saved {tok_path}")

    bpt = tok.get_bytes_per_token_lut()
    ok = 0
    fail = 0
    for text in texts[:100]:
        enc = tok.encode(text)
        tb = int(sum(bpt[t] for t in enc))
        ub = len(text.encode("utf-8"))
        if tb == ub:
            ok += 1
        else:
            fail += 1
    print(f"  Byte check: {ok} OK, {fail} FAIL (out of 100)")

    return tok, tok_path


def export_shards(tok, tok_path, texts):
    """Export BESE+BPE binary shards."""
    step(f"STEP 3: Exporting BESE shards ({len(texts)} docs)")
    bpt = tok.get_bytes_per_token_lut()

    BESE_SHARD_DIR.mkdir(parents=True, exist_ok=True)
    HEADER_INTS = 256

    val_n = min(2000, len(texts) // 5)
    val_texts = texts[:val_n]
    train_texts = texts[val_n:]

    def encode_texts(text_list):
        chunks = []
        for text in text_list:
            enc = tok.encode(text)
            chunks.append(enc.astype(np.uint16))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16)

    def write_shard(path, tokens):
        header = np.zeros(HEADER_INTS, dtype="<i4")
        header[0] = 20240520
        header[1] = 1
        header[2] = int(tokens.shape[0])
        with open(path, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens.astype("<u2").tobytes())

    print(f"  Encoding {len(val_texts)} val docs...")
    val_tokens = encode_texts(val_texts)
    val_path = BESE_SHARD_DIR / "fineweb_val_000000.bin"
    write_shard(val_path, val_tokens)
    print(f"  Val shard: {val_tokens.shape[0]:,} tokens")

    print(f"  Encoding {len(train_texts)} train docs...")
    train_tokens = encode_texts(train_texts)
    shard_size = 10_000_000
    for i in range(0, len(train_tokens), shard_size):
        chunk = train_tokens[i : i + shard_size]
        shard_path = BESE_SHARD_DIR / f"fineweb_train_{i // shard_size:06d}.bin"
        write_shard(shard_path, chunk)
        print(f"  Train shard {i // shard_size}: {chunk.shape[0]:,} tokens")

    print(f"  Shards written to {BESE_SHARD_DIR}")


def run_baseline():
    """Run baseline SP1024 training for 10 minutes."""
    step("STEP 4a: Running BASELINE (SP1024) training - 10 min")
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "baseline_10min",
            "DATA_PATH": str(SHARD_DIR),
            "TOKENIZER_PATH": str(SP_MODEL),
            "VOCAB_SIZE": "1024",
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": "200",
        }
    )
    r = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=1", str(PG_DIR / "train_gpt.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1000:])
    return r.stdout


def run_bese(tok_path):
    """Run BESE+BPE training for 10 minutes."""
    step("STEP 4b: Running BESE+BPE training - 10 min")
    from bese_bpe_tokenizer import BESEBPETokenizer

    tok = BESEBPETokenizer.load(str(tok_path))

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "bese_10min",
            "DATA_PATH": str(BESE_SHARD_DIR),
            "TOKENIZER_PATH": str(tok_path),
            "VOCAB_SIZE": str(tok.vocab_size),
            "VAL_LOSS_EVERY": "0",
            "TRAIN_LOG_EVERY": "200",
            "BESE_TOKENIZER_ROOT": str(BESE_DIR / "tokenizer"),
        }
    )
    r = subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            str(BESE_DIR / "integration" / "train_gpt_bese.py"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    print(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1000:])
    return r.stdout


def extract_bpb(output):
    for line in output.strip().split("\n"):
        if "final_int8_zlib_roundtrip_exact" in line and "val_bpb" in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    return float(part.split(":")[1])
        if "val_bpb:" in line and "final" not in line:
            for part in line.split():
                if part.startswith("val_bpb:"):
                    return float(part.split(":")[1])
    return None


def main():
    t_start = time.time()

    jsonl_path, texts = decode_shards()
    tok, tok_path = train_bpe(texts)
    export_shards(tok, tok_path, texts)

    baseline_out = run_baseline()
    bese_out = run_bese(tok_path)

    step("RESULTS")
    baseline_bpb = extract_bpb(baseline_out)
    bese_bpb = extract_bpb(bese_out)
    print(f"  Baseline val_bpb: {baseline_bpb}")
    print(f"  BESE     val_bpb: {bese_bpb}")
    if baseline_bpb and bese_bpb:
        diff = bese_bpb - baseline_bpb
        print(f"  Difference: {diff:+.4f} ({'BESE better' if diff < 0 else 'Baseline better'})")

    total = time.time() - t_start
    print(f"\n  Total wall time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
