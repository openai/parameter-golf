#!/usr/bin/env python3
"""Tokenizer benchmark — compare BPE/Unigram/ByteLevel on FineWeb sample.

Measures compression ratio (bytes per token) for different tokenizer configs.
Byte-level (vocab_size=256) is the baseline; SentencePiece 4096 is the
alternative used by PR #1333.

Usage:
    python scripts/benchmark_tokenizers.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_fineweb_train


def main():
    print("Loading sample data (1MB)...")
    data = load_fineweb_train(1_000_000)
    text = bytes(data).decode("utf-8", errors="replace")

    print(f"Sample: {len(data):,} bytes, {len(text):,} chars\n")

    results = []

    # Byte-level baseline
    n_tokens_byte = len(data)
    bpt_byte = len(data) / n_tokens_byte  # always 1.0
    results.append(("byte-level (256)", 256, n_tokens_byte, bpt_byte))
    print(f"byte-level:  {n_tokens_byte:>10,} tokens, {bpt_byte:.2f} bytes/token")

    # tiktoken byte-level BPE variants
    try:
        import tiktoken
        for enc_name in ["cl100k_base", "o200k_base"]:
            enc = tiktoken.get_encoding(enc_name)
            tokens = enc.encode(text)
            bpt = len(data) / len(tokens)
            results.append((enc_name, enc.max_token_value, len(tokens), bpt))
            print(f"{enc_name:>16s}:  {len(tokens):>10,} tokens, {bpt:.2f} bytes/token, "
                  f"vocab={enc.max_token_value}")
    except ImportError:
        print("(tiktoken not available, skipping)")

    # SentencePiece (if model available)
    try:
        import sentencepiece as spm
        sp_models = list(Path(PROJECT_ROOT / "data" / "tokenizers").glob("*.model"))
        for sp_path in sp_models:
            sp = spm.SentencePieceProcessor(model_file=str(sp_path))
            tokens = sp.encode(text)
            bpt = len(data) / len(tokens)
            results.append((sp_path.stem, sp.vocab_size(), len(tokens), bpt))
            print(f"{sp_path.stem:>16s}:  {len(tokens):>10,} tokens, {bpt:.2f} bytes/token, "
                  f"vocab={sp.vocab_size()}")
    except ImportError:
        print("(sentencepiece not available, skipping)")

    print(f"\n{'='*60}")
    print(f"{'Tokenizer':<20} {'Vocab':>8} {'Tokens':>12} {'Bytes/Tok':>10}")
    print(f"{'-'*60}")
    for name, vocab, n_tok, bpt in results:
        print(f"{name:<20} {vocab:>8,} {n_tok:>12,} {bpt:>10.2f}")

    # Embedding budget analysis
    print(f"\n{'='*60}")
    print("Embedding budget at d=512:")
    print(f"{'Tokenizer':<20} {'Emb Params':>12} {'Emb MB (fp16)':>14} {'Emb MB (int6)':>14}")
    print(f"{'-'*60}")
    d = 512
    for name, vocab, _, _ in results:
        emb_params = vocab * d
        emb_fp16 = emb_params * 2 / (1024 * 1024)
        emb_int6 = emb_params * 0.75 / (1024 * 1024)
        print(f"{name:<20} {emb_params:>12,} {emb_fp16:>14.2f} {emb_int6:>14.2f}")


if __name__ == "__main__":
    main()
