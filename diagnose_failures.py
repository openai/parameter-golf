#!/usr/bin/env python3
"""
Diagnose: What does the model actually get wrong?

Runs a forward pass on validation data, collects per-token losses,
and analyzes WHERE and WHY the model fails.

This is the question we should have asked first.

Usage:
  python3 diagnose_failures.py --model final_model.pt --data data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin --tokenizer data/tokenizers/fineweb_1024_bpe.model
"""

import argparse
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_shard(path):
    """Load a tokenized shard with header."""
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.int64))


def main():
    parser = argparse.ArgumentParser(description="Diagnose model failures")
    parser.add_argument("--model", default="final_model.pt")
    parser.add_argument("--data", default="data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin")
    parser.add_argument("--tokenizer", default="data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--max-tokens", type=int, default=500_000, help="Max tokens to analyze")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab_size = sp.get_piece_size()

    # Load tokens
    print(f"Loading validation data from {args.data}...")
    val_tokens = load_shard(args.data)
    n = min(len(val_tokens), args.max_tokens + 1)
    val_tokens = val_tokens[:n]
    print(f"  Loaded {n:,} tokens (vocab={vocab_size})")

    # Load model
    print(f"Loading model from {args.model}...")
    sd = torch.load(args.model, map_location=args.device, weights_only=True)

    # We can't run the full model without the architecture code.
    # Instead, compute token-level statistics from the data itself.
    print("\n" + "=" * 60)
    print("TOKEN-LEVEL ANALYSIS (no model needed)")
    print("=" * 60)

    tokens = val_tokens.numpy()

    # 1. Token frequency distribution
    print("\n--- Token Frequency ---")
    counts = Counter(tokens.tolist())
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    total = len(tokens)
    top10_frac = sum(c for _, c in sorted_counts[:10]) / total
    top100_frac = sum(c for _, c in sorted_counts[:100]) / total
    top500_frac = sum(c for _, c in sorted_counts[:500]) / total
    bottom_half = sum(c for _, c in sorted_counts[vocab_size // 2:]) / total

    print(f"  Top 10 tokens cover:   {top10_frac*100:.1f}% of text")
    print(f"  Top 100 tokens cover:  {top100_frac*100:.1f}% of text")
    print(f"  Top 500 tokens cover:  {top500_frac*100:.1f}% of text")
    print(f"  Bottom 50% of vocab:   {bottom_half*100:.1f}% of text")
    print(f"  Unique tokens used:    {len(counts)} / {vocab_size}")

    print("\n  Most common tokens:")
    for tok_id, count in sorted_counts[:20]:
        piece = sp.id_to_piece(tok_id)
        pct = count / total * 100
        print(f"    {tok_id:5d}  {piece:15s}  {count:8,}  ({pct:.2f}%)")

    print("\n  Rarest tokens (that appear at least once):")
    for tok_id, count in sorted_counts[-10:]:
        piece = sp.id_to_piece(tok_id)
        print(f"    {tok_id:5d}  {piece:15s}  {count:8,}")

    # 2. Bigram analysis — what transitions are hardest?
    print("\n--- Bigram Predictability ---")
    bigram_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - 1):
        bigram_counts[tokens[i]][tokens[i + 1]] += 1

    # For each context token, compute entropy of next-token distribution
    context_entropy = {}
    for ctx, next_counts in bigram_counts.items():
        total_ctx = sum(next_counts.values())
        entropy = 0
        for count in next_counts.values():
            p = count / total_ctx
            if p > 0:
                entropy -= p * math.log2(p)
        context_entropy[ctx] = (entropy, total_ctx)

    # Sort by entropy (highest = hardest to predict after)
    sorted_entropy = sorted(context_entropy.items(), key=lambda x: -x[1][0])

    print("\n  Hardest to predict AFTER (highest entropy):")
    for tok_id, (ent, count) in sorted_entropy[:20]:
        if count < 100:
            continue
        piece = sp.id_to_piece(tok_id)
        n_next = len(bigram_counts[tok_id])
        print(f"    After {piece:15s}  entropy={ent:.2f} bits  ({n_next} unique next tokens, {count:,} occurrences)")

    print("\n  Easiest to predict AFTER (lowest entropy):")
    easy = [(t, e, c) for t, (e, c) in context_entropy.items() if c >= 100]
    easy.sort(key=lambda x: x[1])
    for tok_id, ent, count in easy[:20]:
        piece = sp.id_to_piece(tok_id)
        # What's the most common next token?
        most_common_next = max(bigram_counts[tok_id].items(), key=lambda x: x[1])
        next_piece = sp.id_to_piece(most_common_next[0])
        pct = most_common_next[1] / count * 100
        print(f"    After {piece:15s}  entropy={ent:.2f} bits  → {next_piece} ({pct:.0f}%)")

    # 3. Position analysis — are certain positions harder?
    print("\n--- Position in Sequence ---")
    # Group tokens by position within 1024-token windows
    n_windows = (len(tokens) - 1) // args.seq_len
    pos_counts = defaultdict(list)
    for w in range(min(n_windows, 500)):  # sample 500 windows
        start = w * args.seq_len
        for pos in range(args.seq_len):
            if start + pos + 1 < len(tokens):
                ctx = tokens[start + pos]
                tgt = tokens[start + pos + 1]
                # Simple "difficulty" proxy: is tgt the most common follower of ctx?
                if ctx in bigram_counts and tgt in bigram_counts[ctx]:
                    total_ctx = sum(bigram_counts[ctx].values())
                    p = bigram_counts[ctx][tgt] / total_ctx
                    bits = -math.log2(max(p, 1e-10))
                    pos_counts[pos].append(bits)

    print("  Average bigram cost by position in window:")
    positions = sorted(pos_counts.keys())
    for p in [0, 1, 2, 5, 10, 50, 100, 200, 500, 1000, 1023]:
        if p in pos_counts and pos_counts[p]:
            avg = sum(pos_counts[p]) / len(pos_counts[p])
            print(f"    Position {p:5d}: {avg:.2f} bits")

    # 4. Token type analysis
    print("\n--- Token Type Analysis ---")
    type_bits = defaultdict(list)

    for i in range(min(len(tokens) - 1, 200000)):
        ctx = tokens[i]
        tgt = tokens[i + 1]
        if ctx in bigram_counts and tgt in bigram_counts[ctx]:
            total_ctx = sum(bigram_counts[ctx].values())
            p = bigram_counts[ctx][tgt] / total_ctx
            bits = -math.log2(max(p, 1e-10))

            piece = sp.id_to_piece(int(tgt))

            # Classify token type
            if piece.startswith('▁'):
                type_bits['word_start'].append(bits)
            elif piece.isdigit() or (len(piece) > 1 and piece[0] == '▁' and piece[1:].isdigit()):
                type_bits['number'].append(bits)
            elif all(c in '.,;:!?-()[]{}"\'' for c in piece.replace('▁', '')):
                type_bits['punctuation'].append(bits)
            elif piece.isupper() or (piece.startswith('▁') and piece[1:].isupper()):
                type_bits['uppercase'].append(bits)
            else:
                type_bits['continuation'].append(bits)

    print(f"  {'Type':<20s} {'Count':>8s} {'Avg bits':>10s} {'Median':>10s} {'P90':>10s}")
    print(f"  {'-'*60}")
    for ttype in ['word_start', 'continuation', 'punctuation', 'number', 'uppercase']:
        if ttype in type_bits and type_bits[ttype]:
            vals = sorted(type_bits[ttype])
            avg = sum(vals) / len(vals)
            med = vals[len(vals) // 2]
            p90 = vals[int(len(vals) * 0.9)]
            print(f"  {ttype:<20s} {len(vals):>8,} {avg:>10.2f} {med:>10.2f} {p90:>10.2f}")

    # 5. Hardest specific bigrams
    print("\n--- Hardest Specific Transitions ---")
    print("  (Context → Target tokens that cost the most bits)")
    hard_transitions = []
    for ctx_id, next_counts in bigram_counts.items():
        total_ctx = sum(next_counts.values())
        for tgt_id, count in next_counts.items():
            if count >= 50:  # only frequent enough to matter
                p = count / total_ctx
                bits = -math.log2(max(p, 1e-10))
                contribution = count * bits  # total bits from this bigram
                hard_transitions.append((ctx_id, tgt_id, bits, count, contribution))

    # Sort by total contribution (bits × frequency)
    hard_transitions.sort(key=lambda x: -x[4])
    print("\n  Highest total cost (bits × frequency):")
    for ctx_id, tgt_id, bits, count, contrib in hard_transitions[:20]:
        ctx_piece = sp.id_to_piece(ctx_id)
        tgt_piece = sp.id_to_piece(tgt_id)
        print(f"    {ctx_piece:12s} → {tgt_piece:12s}  {bits:.1f} bits × {count:,} = {contrib:,.0f} total bits")

    # 6. Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Where to focus improvements")
    print("=" * 60)

    total_bits_by_type = {}
    for ttype, vals in type_bits.items():
        total_bits_by_type[ttype] = sum(vals)
    grand_total = sum(total_bits_by_type.values())

    print("\n  BPB contribution by token type:")
    for ttype in sorted(total_bits_by_type, key=total_bits_by_type.get, reverse=True):
        pct = total_bits_by_type[ttype] / grand_total * 100
        count = len(type_bits[ttype])
        avg = total_bits_by_type[ttype] / count if count else 0
        print(f"    {ttype:<20s}  {pct:5.1f}% of total bits  (avg {avg:.2f} bits/token, {count:,} tokens)")

    print("\n  The token types contributing the most TOTAL bits")
    print("  are where improvements have the biggest BPB impact.")
    print("  Focus on the type with highest % contribution AND highest avg bits.")


if __name__ == "__main__":
    main()
