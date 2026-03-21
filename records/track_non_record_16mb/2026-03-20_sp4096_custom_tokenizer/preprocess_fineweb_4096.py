"""
preprocess_fineweb_4096.py — Tokenize FineWeb with our custom 4096-vocab tokenizer
Team: Parameter-Golf

Reads:  data/docs_selected.jsonl
        data/tokenizers/fineweb_4096_bpe.model
Writes: data/datasets/fineweb10B_sp4096/fineweb_train_000000.bin ... (N shards)
        data/datasets/fineweb10B_sp4096/fineweb_val_000000.bin

Binary format is identical to the official sp1024 shards:
  Header: 256 × int32  [magic=20240520, version=1, num_tokens, ...]
  Tokens: uint16 array
"""

import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm


# ── Config ────────────────────────────────────────────────────────────────────
DOCS_PATH        = "data/docs_selected.jsonl"
TOKENIZER_PATH   = "data/tokenizers/fineweb_4096_bpe.model"
OUTPUT_DIR       = "data/datasets/fineweb10B_sp4096"
TOKENS_PER_SHARD = 100_000_000   # 100M tokens per shard (~200MB as uint16)
NUM_TRAIN_SHARDS = 10
VAL_DOCS         = 50_000        # first 50k docs go to validation (matches sp1024)
SHARD_MAGIC      = 20240520
SHARD_VERSION    = 1
HEADER_INTS      = 256


def write_shard(path: Path, tokens: np.ndarray) -> None:
    """Write tokens to a binary shard file in the official format."""
    assert tokens.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    size_mb = path.stat().st_size / 1e6
    print(f"  Written: {path.name}  ({len(tokens):,} tokens, {size_mb:.0f} MB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_PATH}")
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vocab_size = int(sp.vocab_size())
    print(f"Vocab size: {vocab_size}")
    assert vocab_size == 4096, f"Expected 4096, got {vocab_size}"

    eos_id = sp.eos_id()   # = 2
    print(f"EOS id: {eos_id}")
    print()

    print(f"Source: {DOCS_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Val docs: {VAL_DOCS:,}")
    print(f"Train shards: {NUM_TRAIN_SHARDS} × {TOKENS_PER_SHARD/1e6:.0f}M tokens")
    print()

    # ── Stream docs, tokenize, write shards ──────────────────────────────────
    val_tokens_list   = []
    train_tokens_list = []
    shard_idx         = 0
    doc_idx           = 0
    train_token_count = 0
    val_token_count   = 0
    t0                = time.perf_counter()

    with open(DOCS_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text") or obj.get("content") or ""
            if not text or len(text) < 10:
                continue

            # Tokenize and append EOS
            tokens = sp.encode(text) + [eos_id]
            tokens_np = np.array(tokens, dtype=np.uint16)

            is_val = doc_idx < VAL_DOCS

            if is_val:
                val_tokens_list.append(tokens_np)
                val_token_count += len(tokens_np)
            else:
                train_tokens_list.append(tokens_np)
                train_token_count += len(tokens_np)

                # Flush train shard when full
                if train_token_count >= TOKENS_PER_SHARD:
                    if shard_idx < NUM_TRAIN_SHARDS:
                        shard_path = Path(OUTPUT_DIR) / f"fineweb_train_{shard_idx:06d}.bin"
                        shard_tokens = np.concatenate(train_tokens_list)[:TOKENS_PER_SHARD]
                        write_shard(shard_path, shard_tokens.astype(np.uint16))
                        # Keep leftover tokens for next shard
                        leftover = np.concatenate(train_tokens_list)[TOKENS_PER_SHARD:]
                        train_tokens_list = [leftover] if len(leftover) > 0 else []
                        train_token_count = len(leftover)
                        shard_idx += 1
                    else:
                        break  # enough shards

            doc_idx += 1

            if doc_idx % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  doc {doc_idx:,}  "
                      f"val_tokens={val_token_count/1e6:.1f}M  "
                      f"train_tokens={train_token_count/1e6:.1f}M  "
                      f"shards={shard_idx}  "
                      f"[{elapsed:.0f}s]")

            if shard_idx >= NUM_TRAIN_SHARDS:
                break

    # ── Write remaining train shard ───────────────────────────────────────────
    if train_tokens_list and shard_idx < NUM_TRAIN_SHARDS:
        shard_path = Path(OUTPUT_DIR) / f"fineweb_train_{shard_idx:06d}.bin"
        shard_tokens = np.concatenate(train_tokens_list)
        write_shard(shard_path, shard_tokens.astype(np.uint16))
        shard_idx += 1

    # ── Write validation shard ────────────────────────────────────────────────
    print()
    print("Writing validation shard...")
    val_path   = Path(OUTPUT_DIR) / "fineweb_val_000000.bin"
    val_tokens = np.concatenate(val_tokens_list).astype(np.uint16)
    write_shard(val_path, val_tokens)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.0f}s")
    print(f"Train shards written: {shard_idx}")
    print(f"Val tokens:           {val_token_count:,}")
    print(f"Total docs processed: {doc_idx:,}")
    print()

    # Verify shards
    import glob
    shards = sorted(glob.glob(f"{OUTPUT_DIR}/fineweb_train_*.bin"))
    print(f"Train shards verified: {len(shards)}")
    print(f"Val shard verified:    {val_path.exists()}")
    print()
    print("✅ Preprocessing complete.")
    print()
    print("Next step — run training:")
    print("  RUN_ID=sp4096_run1 \\")
    print(f"  DATA_PATH={OUTPUT_DIR} \\")
    print(f"  TOKENIZER_PATH={TOKENIZER_PATH} \\")
    print("  VOCAB_SIZE=4096 \\")
    print("  torchrun --standalone --nproc_per_node=1 train_gpt.py")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    main()
