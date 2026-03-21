#!/usr/bin/env python3
"""
Tokenizer Trainer for Parameter Golf Competition

Trains custom SentencePiece tokenizers on FineWeb data, evaluates quality,
and optionally exports binary shards compatible with train_gpt.py.

Supports both BPE and Unigram model types. Research suggests Unigram often
outperforms BPE at small vocab sizes (512-4096) due to its top-down pruning
strategy selecting globally-useful tokens vs BPE's greedy bottom-up merges.

Usage:
    python train_tokenizer.py --vocab-size 1024 --model-type bpe
    python train_tokenizer.py --vocab-size 1024 --model-type unigram
    python train_tokenizer.py --compare                # Compare BPE vs Unigram across vocab sizes
    python train_tokenizer.py --vocab-size 1024 --export-shards  # Train + export .bin shards
    python train_tokenizer.py --evaluate path/to/model.model     # Evaluate existing tokenizer

Integration:
    The trained .model file plugs directly into train_gpt.py:
        VOCAB_SIZE=1024 python train_gpt.py --tokenizer-path ./tokenizers/spm_bpe_1024.model
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

FINEWEB_DOCS_PATH = Path(os.environ.get(
    "DOCS_JSONL_PATH",
    "./data/docs_selected.jsonl"  # Default for RunPod; override for local use
))
OUTPUT_DIR = Path("./tokenizers")

# Binary shard format (must match train_gpt.py / download_hf_docs_and_tokenize.py)
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
SHARD_SIZE = 10**8  # 100M tokens per shard
NUM_VAL_DOCS = 50_000
APPEND_EOS = False

# Training data size recommendations by vocab size
# Larger vocabs need more data to see enough merge candidates
TRAIN_DOCS_COUNT = {
    512: 50_000,
    1024: 100_000,
    2048: 200_000,
    4096: 500_000,
}

# Sample sentences for manual inspection of tokenization quality
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require careful hyperparameter tuning.",
    "In 2024, researchers published 3,847 papers on language models.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The café served crème brûlée for €12.50 — absolutely délicieux!",
    "HTTP/1.1 200 OK\nContent-Type: application/json\n{\"status\": \"success\"}",
    "∀x ∈ ℝ, |sin(x)| ≤ 1",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def iter_docs_jsonl(docs_path: Path, max_docs: Optional[int] = None):
    """Iterate over texts from FineWeb docs_selected.jsonl (streaming, low memory)."""
    if not docs_path.exists():
        raise FileNotFoundError(
            f"{docs_path} not found!\n"
            "Download FineWeb data first:\n"
            "  cd ../parameter-golf && python data/cached_challenge_fineweb.py --variant sp1024"
        )
    count = 0
    with open(docs_path) as f:
        for line in f:
            if max_docs is not None and count >= max_docs:
                break
            line = line.strip()
            if line:
                yield json.loads(line)["text"]
                count += 1
    print(f"Loaded {count:,} documents from {docs_path}")


def iter_sentences_for_spm(docs_path: Path, max_docs: Optional[int] = None):
    """Yield individual sentences for SentencePiece sentence_iterator.

    SentencePiece's sentence_iterator expects one sentence per yield.
    Splitting on newlines gives cleaner training signal than full documents.
    """
    for text in iter_docs_jsonl(docs_path, max_docs):
        for line in text.split("\n"):
            line = line.strip()
            if line:
                yield line


# =============================================================================
# SENTENCEPIECE TRAINER
# =============================================================================

def train_sentencepiece(
    docs_path: Path,
    vocab_size: int,
    model_type: str,
    output_dir: Path,
    max_docs: Optional[int] = None,
    *,
    character_coverage: float = 0.995,
    max_sentencepiece_length: int = 16,
    min_frequency: int = 2,
    input_sentence_size: int = 0,
) -> Dict[str, Any]:
    """Train a SentencePiece tokenizer (BPE or Unigram).

    Args:
        docs_path: Path to docs_selected.jsonl
        vocab_size: Target vocabulary size
        model_type: 'bpe' or 'unigram'
        output_dir: Directory to save model files
        max_docs: Max documents to use for training
        character_coverage: Unicode character coverage (0.995 recommended for small vocabs)
        max_sentencepiece_length: Max token length in chars (prevents overly long tokens)
        min_frequency: Minimum frequency for a token to be kept
        input_sentence_size: Max sentences to use (0 = all). Set to 5M+ for large corpora.
    """
    try:
        import sentencepiece as spm
    except ImportError:
        print("ERROR: sentencepiece not installed. Run: pip install sentencepiece")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = str(output_dir / f"spm_{model_type}_{vocab_size}")

    print(f"\n{'=' * 60}")
    print(f"Training SentencePiece {model_type.upper()} (vocab={vocab_size})")
    print(f"{'=' * 60}")

    # Build training kwargs — uses sentence_iterator (streaming) instead of
    # writing a temp file, matching the official data pipeline.
    kwargs: Dict[str, Any] = {
        "sentence_iterator": iter_sentences_for_spm(docs_path, max_docs),
        "model_prefix": model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        # --- Coverage & fallback ---
        # 0.995 saves vocab slots vs 0.9995; rare Unicode falls back to bytes.
        "character_coverage": character_coverage,
        # Critical for small vocab: reserves 256 byte tokens so any byte sequence
        # is representable (no <unk> output). Costs 256 vocab slots.
        "byte_fallback": True,
        # --- Splitting rules ---
        "split_digits": True,           # Each digit 0-9 is its own token
        "split_by_unicode_script": True, # Prevent cross-script merges
        "split_by_number": True,         # Prevent number-letter merges
        # --- Normalization ---
        # nmt_nfkc collapses Unicode variants (fullwidth chars, ligatures, etc.)
        # saving vocab slots. Must match at inference time.
        "normalization_rule_name": "nmt_nfkc",
        # False matches the official data pipeline. Means "Hello" stays "Hello"
        # (no leading ▁), but " Hello" gets ▁.
        "add_dummy_prefix": False,
        # --- Special token IDs (must match train_gpt.py) ---
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        # --- Vocab constraints ---
        "hard_vocab_limit": False,
        "max_sentencepiece_length": max_sentencepiece_length,
    }

    if min_frequency > 0:
        # SentencePiece: --min_frequency is only used in BPE mode (ignored for unigram)
        kwargs["min_frequency"] = min_frequency

    if input_sentence_size > 0:
        kwargs["input_sentence_size"] = input_sentence_size
        kwargs["shuffle_input_sentence"] = True

    start_time = time.time()
    spm.SentencePieceTrainer.train(**kwargs)
    train_time = time.time() - start_time

    print(f"Training completed in {train_time:.1f}s")

    # Load and verify
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    # Count token types
    n_byte = sum(1 for i in range(sp.vocab_size()) if sp.is_byte(i))
    n_control = sum(1 for i in range(sp.vocab_size()) if sp.is_control(i))
    n_unknown = sum(1 for i in range(sp.vocab_size()) if sp.is_unknown(i))
    n_learned = sp.vocab_size() - n_byte - n_control - n_unknown

    print(f"Actual vocab size: {sp.vocab_size()}")
    print(f"  Learned subword tokens: {n_learned}")
    print(f"  Byte fallback tokens:   {n_byte}")
    print(f"  Control tokens:         {n_control}")
    print(f"  Unknown tokens:         {n_unknown}")

    return {
        "method": f"sentencepiece_{model_type}",
        "model_type": model_type,
        "vocab_size": vocab_size,
        "actual_vocab": sp.vocab_size(),
        "learned_tokens": n_learned,
        "byte_tokens": n_byte,
        "train_time_sec": train_time,
        "model_path": f"{model_prefix}.model",
        "vocab_path": f"{model_prefix}.vocab",
    }


# =============================================================================
# TOKENIZER EVALUATION
# =============================================================================

def evaluate_tokenizer(model_path: str, docs_path: Path, n_eval_docs: int = 5000) -> Dict[str, Any]:
    """Comprehensive evaluation of a trained SentencePiece tokenizer.

    Computes:
        - Compression ratio (bytes per token) — higher is better
        - Fertility (tokens per word) — lower is better
        - Token length distribution
        - Coverage analysis
        - Sample tokenizations for manual inspection
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_path}")
    print(f"Vocab size: {sp.vocab_size()}, Eval docs: {n_eval_docs:,}")
    print(f"{'=' * 60}")

    total_bytes = 0
    total_tokens = 0
    total_words = 0
    token_lengths: List[int] = []  # UTF-8 byte length of each token's text
    token_freq: Counter = Counter()

    start = time.time()
    for text in iter_docs_jsonl(docs_path, max_docs=n_eval_docs):
        text_bytes = len(text.encode("utf-8"))
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        words = text.split()

        total_bytes += text_bytes
        total_tokens += len(ids)
        total_words += len(words)

        for piece in pieces:
            clean = piece.lstrip("▁")
            token_lengths.append(len(clean.encode("utf-8")))

        token_freq.update(ids)

    eval_time = time.time() - start

    # Core metrics
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0
    fertility = total_tokens / total_words if total_words else 0
    bits_per_byte_ceiling = math.log2(sp.vocab_size())  # Theoretical worst case

    # Token length distribution
    lengths_arr = np.array(token_lengths)
    pct_single_byte = np.sum(lengths_arr <= 1) / len(lengths_arr) * 100
    pct_multi_char = np.sum(lengths_arr >= 3) / len(lengths_arr) * 100

    # Vocab utilization: how many unique tokens actually appear
    unique_used = len(token_freq)
    vocab_utilization = unique_used / sp.vocab_size() * 100

    # Top tokens
    top_20 = token_freq.most_common(20)

    print(f"\n--- Compression ---")
    print(f"  Bytes per token:    {bytes_per_token:.2f} (higher = better)")
    print(f"  Tokens per word:    {fertility:.2f} (lower = better)")
    print(f"  Bits/byte ceiling:  {bits_per_byte_ceiling:.2f} (log2(vocab_size))")
    print(f"  Effective BPB:      ~{bits_per_byte_ceiling / bytes_per_token:.2f} (ceiling / compression)")

    print(f"\n--- Token Length Distribution ---")
    print(f"  Mean token length:  {lengths_arr.mean():.1f} bytes")
    print(f"  Median:             {np.median(lengths_arr):.0f} bytes")
    print(f"  Single-byte tokens: {pct_single_byte:.1f}%")
    print(f"  Multi-char (≥3B):   {pct_multi_char:.1f}%")

    print(f"\n--- Vocab Utilization ---")
    print(f"  Unique tokens used: {unique_used:,} / {sp.vocab_size()} ({vocab_utilization:.1f}%)")

    print(f"\n--- Top 20 Tokens ---")
    for token_id, count in top_20:
        piece = sp.id_to_piece(token_id)
        print(f"  {token_id:5d} | {piece:20s} | {count:,}")

    # Sample tokenizations
    print(f"\n--- Sample Tokenizations ---")
    for sentence in SAMPLE_SENTENCES:
        pieces = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)
        print(f"\n  Input:  {sentence[:80]}")
        print(f"  Tokens: {len(ids)}")
        print(f"  Pieces: {' | '.join(pieces[:30])}")

    print(f"\n  Eval time: {eval_time:.1f}s")

    return {
        "model_path": model_path,
        "vocab_size": sp.vocab_size(),
        "bytes_per_token": bytes_per_token,
        "fertility": fertility,
        "bits_per_byte_ceiling": bits_per_byte_ceiling,
        "mean_token_length_bytes": float(lengths_arr.mean()),
        "pct_single_byte_tokens": pct_single_byte,
        "pct_multi_char_tokens": pct_multi_char,
        "vocab_utilization_pct": vocab_utilization,
        "eval_docs": n_eval_docs,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
    }


# =============================================================================
# BINARY SHARD EXPORT (compatible with train_gpt.py)
# =============================================================================

def write_datafile(path: Path, toks: np.ndarray) -> None:
    """Write a binary shard file matching train_gpt.py format."""
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = toks.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def export_binary_shards(
    model_path: str,
    docs_path: Path,
    output_dir: Path,
    num_val_docs: int = NUM_VAL_DOCS,
    shard_size: int = SHARD_SIZE,
) -> Dict[str, int]:
    """Export tokenized binary shards for train_gpt.py.

    Format: [256-int32 header][uint16 token stream]
    Each document: [BOS_ID] [encoded tokens] (no EOS by default)
    First `num_val_docs` go to val shards, rest to train shards.
    """
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=model_path)
    vocab_size = sp.vocab_size()

    if vocab_size > 2**16:
        raise ValueError(f"vocab_size={vocab_size} too large for uint16 shard storage")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale shards
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    stats = {k: 0 for k in [
        "docs_total", "docs_val", "docs_train",
        "files_total", "files_val", "files_train",
        "tokens_total", "tokens_val", "tokens_train",
    ]}

    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    split = "val"
    shards = {"val": 0, "train": 0}

    def flush():
        nonlocal fill
        if fill == 0:
            return
        path = output_dir / f"fineweb_{split}_{shards[split]:06d}.bin"
        write_datafile(path, buf[:fill])
        stats["files_total"] += 1
        stats[f"files_{split}"] += 1
        shards[split] += 1
        fill = 0

    bos_id = sp.bos_id()

    print(f"\nExporting binary shards to {output_dir}/")
    print(f"  Tokenizer: {model_path} (vocab={vocab_size})")
    print(f"  Val docs: {num_val_docs:,}, Shard size: {shard_size:,} tokens")

    for text in iter_docs_jsonl(docs_path):
        doc_split = "val" if stats["docs_total"] < num_val_docs else "train"
        if doc_split != split:
            flush()
            split = doc_split

        encoded = np.asarray(sp.encode(text, out_type=int), dtype=np.int32)
        toks = np.empty((encoded.size + 1 + int(APPEND_EOS),), dtype=np.int32)
        toks[0] = bos_id
        toks[1: 1 + encoded.size] = encoded
        if APPEND_EOS:
            toks[-1] = sp.eos_id()

        if not ((0 <= toks).all() and (toks < vocab_size).all()):
            bad = int(toks[(toks < 0) | (toks >= vocab_size)][0])
            raise ValueError(f"token id {bad} outside vocab_size={vocab_size}")
        toks = toks.astype("<u2", copy=False)

        stats["docs_total"] += 1
        stats[f"docs_{split}"] += 1
        stats["tokens_total"] += len(toks)
        stats[f"tokens_{split}"] += len(toks)

        pos = 0
        while pos < len(toks):
            take = min(shard_size - fill, len(toks) - pos)
            buf[fill: fill + take] = toks[pos: pos + take]
            fill += take
            pos += take
            if fill == shard_size:
                flush()

        if stats["docs_total"] % 100_000 == 0:
            print(f"  {stats['docs_total']:,} docs exported...", flush=True)

    flush()

    print(f"\nExport complete:")
    print(f"  Total docs:   {stats['docs_total']:,}")
    print(f"  Val shards:   {stats['files_val']} ({stats['tokens_val']:,} tokens)")
    print(f"  Train shards: {stats['files_train']} ({stats['tokens_train']:,} tokens)")
    print(f"  Total tokens: {stats['tokens_total']:,}")

    return stats


# =============================================================================
# EMBEDDING SIZE CALCULATOR
# =============================================================================

def calc_embedding_budget(vocab_size: int, model_dim: int = 512) -> Dict[str, Any]:
    """Calculate embedding size impact on the 16MB artifact budget."""
    bytes_fp16 = vocab_size * model_dim * 2
    artifact_budget = 16_000_000
    baseline_vocab = 1024
    baseline_bytes = baseline_vocab * model_dim * 2

    return {
        "vocab_size": vocab_size,
        "model_dim": model_dim,
        "embedding_bytes_fp16": bytes_fp16,
        "embedding_mb_fp16": bytes_fp16 / 1_000_000,
        "budget_remaining": artifact_budget - bytes_fp16,
        "vs_baseline_1024": bytes_fp16 - baseline_bytes,
        "pct_of_budget": bytes_fp16 / artifact_budget * 100,
    }


# =============================================================================
# COMPARISON
# =============================================================================

def run_comparison(docs_path: Path, vocab_sizes: List[int], model_types: List[str],
                   max_docs: Optional[int] = None, n_eval_docs: int = 5000):
    """Train and evaluate all combinations, print comparison table."""
    results = []

    for vocab_size in vocab_sizes:
        n_docs = max_docs or TRAIN_DOCS_COUNT.get(vocab_size, 100_000)
        for model_type in model_types:
            info = train_sentencepiece(docs_path, vocab_size, model_type, OUTPUT_DIR, max_docs=n_docs)
            if not info:
                continue
            eval_result = evaluate_tokenizer(info["model_path"], docs_path, n_eval_docs=n_eval_docs)
            budget = calc_embedding_budget(vocab_size)
            results.append({**info, **eval_result, "budget": budget})

    # Print comparison table
    print(f"\n{'=' * 100}")
    print("COMPARISON TABLE")
    print(f"{'=' * 100}")
    print(f"{'Method':<20} {'Vocab':>6} {'Actual':>6} {'Learned':>7} {'B/Tok':>6} "
          f"{'Fert':>6} {'Emb MB':>7} {'Budget%':>8} {'Train(s)':>9}")
    print("-" * 100)

    for r in results:
        print(f"{r['method']:<20} {r['vocab_size']:>6} {r['actual_vocab']:>6} "
              f"{r['learned_tokens']:>7} {r['bytes_per_token']:>6.2f} "
              f"{r['fertility']:>6.2f} {r['budget']['embedding_mb_fp16']:>7.2f} "
              f"{r['budget']['pct_of_budget']:>7.1f}% {r['train_time_sec']:>8.1f}")

    # Save results
    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Recommendation
    if results:
        best = max(results, key=lambda r: r["bytes_per_token"])
        print(f"\nBest compression: {best['method']} vocab={best['vocab_size']} "
              f"({best['bytes_per_token']:.2f} bytes/token)")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train custom SentencePiece tokenizers for Parameter Golf",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train BPE tokenizer with vocab 1024
  python train_tokenizer.py --vocab-size 1024 --model-type bpe

  # Train Unigram tokenizer (often better for small vocabs)
  python train_tokenizer.py --vocab-size 1024 --model-type unigram

  # Compare BPE vs Unigram across vocab sizes
  python train_tokenizer.py --compare

  # Train + export binary shards for train_gpt.py
  python train_tokenizer.py --vocab-size 1024 --model-type bpe --export-shards

  # Evaluate an existing tokenizer
  python train_tokenizer.py --evaluate ./tokenizers/spm_bpe_1024.model
        """,
    )
    parser.add_argument("--vocab-size", type=int, default=1024, help="Target vocabulary size (default: 1024)")
    parser.add_argument("--model-type", type=str, default="bpe", choices=["bpe", "unigram"],
                        help="SentencePiece model type (default: bpe)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare BPE vs Unigram across vocab sizes [512, 1024, 2048, 4096]")
    parser.add_argument("--evaluate", type=str, default=None, metavar="MODEL_PATH",
                        help="Evaluate an existing .model file instead of training")
    parser.add_argument("--export-shards", action="store_true",
                        help="Export binary shards after training (for train_gpt.py)")
    parser.add_argument("--shard-output-dir", type=str, default=None,
                        help="Output directory for binary shards (default: alongside tokenizer)")
    parser.add_argument("--docs-path", type=str, default=None, help="Path to docs_selected.jsonl")
    parser.add_argument("--max-docs", type=int, default=None, help="Max docs for tokenizer training")
    parser.add_argument("--eval-docs", type=int, default=5000, help="Docs for evaluation (default: 5000)")
    parser.add_argument("--character-coverage", type=float, default=0.995,
                        help="Unicode character coverage (default: 0.995)")
    parser.add_argument("--max-token-length", type=int, default=16,
                        help="Max SentencePiece token length in chars (default: 16)")
    parser.add_argument("--input-sentence-size", type=int, default=0,
                        help="Max sentences for SPM training (0=all, set >0 for large corpora)")
    args = parser.parse_args()

    docs_path = Path(args.docs_path) if args.docs_path else FINEWEB_DOCS_PATH
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Evaluate existing model ---
    if args.evaluate:
        if not Path(args.evaluate).exists():
            print(f"ERROR: {args.evaluate} not found!")
            sys.exit(1)
        evaluate_tokenizer(args.evaluate, docs_path, n_eval_docs=args.eval_docs)
        return

    # --- Compare mode ---
    if args.compare:
        run_comparison(
            docs_path,
            vocab_sizes=[512, 1024, 2048, 4096],
            model_types=["bpe", "unigram"],
            max_docs=args.max_docs,
            n_eval_docs=args.eval_docs,
        )
        return

    # --- Train single tokenizer ---
    max_docs = args.max_docs or TRAIN_DOCS_COUNT.get(args.vocab_size, 100_000)

    info = train_sentencepiece(
        docs_path,
        args.vocab_size,
        args.model_type,
        OUTPUT_DIR,
        max_docs=max_docs,
        character_coverage=args.character_coverage,
        max_sentencepiece_length=args.max_token_length,
        input_sentence_size=args.input_sentence_size,
    )

    if not info:
        print("ERROR: Training failed!")
        sys.exit(1)

    # Evaluate
    eval_result = evaluate_tokenizer(info["model_path"], docs_path, n_eval_docs=args.eval_docs)
    budget = calc_embedding_budget(args.vocab_size)

    # Print budget info
    print(f"\n--- Embedding Budget Impact (d_model=512, FP16) ---")
    print(f"  Embedding size:    {budget['embedding_mb_fp16']:.2f} MB")
    print(f"  % of 16MB budget:  {budget['pct_of_budget']:.1f}%")
    print(f"  vs baseline (1024): {budget['vs_baseline_1024']:+,} bytes")

    # Save results
    all_results = {**info, **eval_result, "budget": budget}
    results_path = OUTPUT_DIR / f"result_{args.model_type}_{args.vocab_size}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Export shards if requested
    if args.export_shards:
        shard_dir = Path(args.shard_output_dir) if args.shard_output_dir else (
            OUTPUT_DIR / f"shards_{args.model_type}_{args.vocab_size}"
        )
        export_binary_shards(info["model_path"], docs_path, shard_dir)

    # Print integration instructions
    print(f"\n--- Integration ---")
    print(f"  To use with train_gpt.py:")
    print(f"    VOCAB_SIZE={info['actual_vocab']} python train_gpt.py \\")
    print(f"      --tokenizer-path {info['model_path']}")
    if not args.export_shards:
        print(f"\n  To export binary shards:")
        print(f"    python train_tokenizer.py --vocab-size {args.vocab_size} "
              f"--model-type {args.model_type} --export-shards")


if __name__ == "__main__":
    main()
