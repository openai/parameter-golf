"""
train_tokenizer.py — Train a 4096-vocab BPE tokenizer on FineWeb docs
Team: Parameter-Golf

Why 4096 vocab:
  BPB = (loss / ln2) * (tokens / bytes)
  Larger vocab → fewer tokens per byte → lower BPB directly
  4096 vocab encodes ~2x more bytes per token vs 1024 vocab
  This improves BPB score independent of model quality.

Output:
  data/tokenizers/fineweb_4096_bpe.model
  data/tokenizers/fineweb_4096_bpe.vocab
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import sentencepiece as spm


def main():
    docs_path  = "data/docs_selected.jsonl"
    output_dir = "data/tokenizers"
    model_name = "fineweb_4096_bpe"
    vocab_size = 4096

    # How many docs to use for tokenizer training
    # 2M docs is plenty — sentencepiece doesn't need all 48GB
    max_docs   = 2_000_000

    os.makedirs(output_dir, exist_ok=True)

    print(f"Training {vocab_size}-vocab BPE tokenizer")
    print(f"Source: {docs_path}")
    print(f"Max docs: {max_docs:,}")
    print()

    # ── Step 1: Extract text from JSONL into a flat text file ────────────────
    # SentencePiece trains from a plain text file, one document per line.
    print("Step 1/2: Extracting text from JSONL...")
    t0 = time.perf_counter()

    tmp_txt = Path(output_dir) / "tokenizer_train_corpus.txt"

    docs_written = 0
    bytes_written = 0

    with open(docs_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(tmp_txt, "w", encoding="utf-8") as fout:

        for line in fin:
            if docs_written >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Handle both {"text": "..."} and {"content": "..."} formats
            text = obj.get("text") or obj.get("content") or ""
            if not text:
                continue

            # Write as single line (replace internal newlines with space)
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            if len(clean) < 10:
                continue

            fout.write(clean + "\n")
            docs_written += 1
            bytes_written += len(clean.encode("utf-8"))

            if docs_written % 100_000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {docs_written:,} docs written "
                      f"({bytes_written/1e9:.2f} GB) "
                      f"[{elapsed:.0f}s]")

    elapsed = time.perf_counter() - t0
    print(f"Done: {docs_written:,} docs, "
          f"{bytes_written/1e9:.2f} GB, "
          f"{elapsed:.0f}s")
    print(f"Corpus saved to: {tmp_txt}")
    print()

    # ── Step 2: Train SentencePiece BPE tokenizer ────────────────────────────
    print("Step 2/2: Training SentencePiece BPE tokenizer...")
    print(f"  vocab_size = {vocab_size}")
    t1 = time.perf_counter()

    model_prefix = str(Path(output_dir) / model_name)

    spm.SentencePieceTrainer.train(
        input            = str(tmp_txt),
        model_prefix     = model_prefix,
        vocab_size       = vocab_size,
        model_type       = "bpe",

        # Match the baseline tokenizer settings exactly
        character_coverage      = 0.9995,
        pad_id                  = 0,
        unk_id                  = 3,
        bos_id                  = 1,
        eos_id                  = 2,

        # Important: add dummy whitespace prefix so tokenizer
        # matches the baseline's ▁ prefix convention
        add_dummy_prefix        = True,

        # Allow all byte fallback for OOV characters
        byte_fallback           = True,

        # Training corpus size limits
        input_sentence_size     = max_docs,
        shuffle_input_sentence  = True,

        # Normalisation — match baseline
        normalization_rule_name = "identity",

        # Speed up training
        num_threads             = os.cpu_count() or 4,
        num_sub_iterations      = 2,
    )

    elapsed = time.perf_counter() - t1
    print(f"Tokenizer trained in {elapsed:.0f}s")
    print(f"Saved: {model_prefix}.model")
    print(f"Saved: {model_prefix}.vocab")
    print()

    # ── Step 3: Verify the tokenizer ─────────────────────────────────────────
    print("Verifying tokenizer...")
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    assert int(sp.vocab_size()) == vocab_size, \
        f"Vocab size mismatch: {sp.vocab_size()} != {vocab_size}"

    # Test encoding
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens    = sp.encode(test_text)
    decoded   = sp.decode(tokens)

    print(f"  Vocab size:    {sp.vocab_size()}")
    print(f"  Test text:     '{test_text}'")
    print(f"  Token count:   {len(tokens)}")
    print(f"  Tokens:        {tokens}")
    print(f"  Decoded:       '{decoded}'")
    print(f"  Bytes/token:   {len(test_text.encode())/len(tokens):.2f}")
    print()

    # Compare bytes/token with baseline 1024-vocab tokenizer
    sp_baseline = spm.SentencePieceProcessor(
        model_file="data/tokenizers/fineweb_1024_bpe.model")
    tokens_baseline = sp_baseline.encode(test_text)
    print(f"  Baseline 1024-vocab bytes/token: "
          f"{len(test_text.encode())/len(tokens_baseline):.2f}")
    print(f"  New     4096-vocab bytes/token:  "
          f"{len(test_text.encode())/len(tokens):.2f}")
    print()
    print("✅ Tokenizer training complete.")
    print()
    print("Next step: preprocess FineWeb data with new tokenizer")
    print("Run: python3 data/cached_challenge_fineweb.py "
          "--variant sp4096 --train-shards 10")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    main()
