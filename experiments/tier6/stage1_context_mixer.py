#!/usr/bin/env python3
"""
Tier 6 — Stage 1: Pure token-level context mixer (PPM) evaluator.

Measures standalone BPB of a classical PPM context mixer on the FineWeb
validation set. No neural model, no GPU, no training. Pure statistics.

Two modes:
  - per-document: statistics reset each sequence (clean, conservative)
  - cumulative:   statistics accumulate across sequences (stronger, still causal)

Usage:
  python3 experiments/tier6/stage1_context_mixer.py [--max-order 4] [--seq-len 1024]
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Data loading (copied from train_gpt.py to stay self-contained)
# ---------------------------------------------------------------------------

def load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    return np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)


def load_validation_tokens(pattern: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    return np.concatenate([load_data_shard(Path(f)) for f in files])


def build_byte_lut(sp: spm.SentencePieceProcessor, vocab_size: int):
    """Build per-token byte counts matching train_gpt.py's BPB logic."""
    base_bytes = np.zeros(vocab_size, dtype=np.int16)
    has_leading_space = np.zeros(vocab_size, dtype=np.bool_)
    is_boundary = np.ones(vocab_size, dtype=np.bool_)
    for tid in range(min(int(sp.vocab_size()), vocab_size)):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):  # ▁
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


# ---------------------------------------------------------------------------
# PPM Context Mixer
# ---------------------------------------------------------------------------

class PPMContextMixer:
    """
    Prediction by Partial Matching (PPM) at the token level.

    Uses Method C blending: at each order, the escape probability is
    e_k = unique(context) / (total(context) + unique(context)), and the
    symbol probability is count(sym) / (total + unique). On escape, we
    back off to order k-1.

    Order -1 is a uniform distribution over the full vocabulary.
    """

    def __init__(self, max_order: int, vocab_size: int):
        self.max_order = max_order
        self.vocab_size = vocab_size
        # counts[k] maps context_tuple -> {token_id: count}
        # totals[k] maps context_tuple -> total_count
        # uniques[k] maps context_tuple -> unique_symbol_count
        self.counts: list[dict[tuple, dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)
        ]
        self.totals: list[dict[tuple, int]] = [
            defaultdict(int) for _ in range(max_order + 1)
        ]
        self.uniques: list[dict[tuple, int]] = [
            defaultdict(int) for _ in range(max_order + 1)
        ]

    def reset(self):
        for k in range(self.max_order + 1):
            self.counts[k].clear()
            self.totals[k].clear()
            self.uniques[k].clear()

    def update(self, context: tuple, symbol: int):
        """Update all order models with an observed (context, symbol) pair."""
        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            c = self.counts[k][ctx]
            if symbol not in c:
                self.uniques[k][ctx] += 1
            c[symbol] += 1
            self.totals[k][ctx] += 1

    def predict(self, context: tuple, symbol: int) -> float:
        """
        Return p(symbol | context) under PPM-C blending.

        Only computes the probability for the specific target symbol,
        not the full 1024-way distribution. This is O(max_order) per call.
        """
        # Walk from highest order down to -1
        prob = 0.0
        escape_product = 1.0  # cumulative probability of escaping to this level

        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            total = self.totals[k].get(ctx, 0)
            unique = self.uniques[k].get(ctx, 0)

            if total == 0:
                # No data at this order, escape with probability 1
                continue

            denom = total + unique
            sym_count = self.counts[k][ctx].get(symbol, 0)

            if sym_count > 0:
                # Symbol found at this order
                p_sym = sym_count / denom
                prob += escape_product * p_sym
                # We also need to account for the remaining escape mass
                # that eventually resolves. But in PPM-C, once we find
                # the symbol, we return the blended probability so far
                # plus the probability of finding it at lower orders
                # (exclusion variant would stop here, but without exclusion
                # we need the full blend).
                #
                # For simplicity and correctness, use the full PPM-C blend:
                # p_escape at this level contributes to lower-order terms
                escape_product *= unique / denom
            else:
                # Symbol not found, full escape
                escape_product *= unique / denom

        # Order -1: uniform over vocab
        prob += escape_product * (1.0 / self.vocab_size)
        return prob


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    tokens: np.ndarray,
    seq_len: int,
    max_order: int,
    vocab_size: int,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary: np.ndarray,
    cumulative: bool,
) -> tuple[float, float, float]:
    """
    Evaluate PPM mixer on validation tokens.

    Returns (bpb, avg_neg_log2_p, runtime_seconds).
    """
    mixer = PPMContextMixer(max_order, vocab_size)
    total_seqs = (len(tokens) - 1) // seq_len
    total_bits = 0.0
    total_bytes = 0.0
    total_tokens_scored = 0

    t0 = time.perf_counter()

    for seq_idx in range(total_seqs):
        if not cumulative:
            mixer.reset()

        start = seq_idx * seq_len
        seq = tokens[start: start + seq_len + 1]

        for t in range(seq_len):
            prev_id = int(seq[t])
            target_id = int(seq[t + 1])

            # Build context (up to max_order previous tokens within this sequence)
            ctx_start = max(0, t + 1 - max_order)
            context = tuple(int(seq[i]) for i in range(ctx_start, t + 1))

            # Predict
            p = mixer.predict(context, target_id)
            if p <= 0:
                p = 1e-12  # safety floor

            # BPB accounting (matching train_gpt.py exactly)
            bits = -math.log2(p)
            byte_count = int(base_bytes[target_id])
            if has_leading_space[target_id] and not is_boundary[prev_id]:
                byte_count += 1
            total_bits += bits
            total_bytes += byte_count
            total_tokens_scored += 1

            # Update mixer with observed token (causal: we've now seen this token)
            mixer.update(context, target_id)

        if (seq_idx + 1) % 2000 == 0 or seq_idx == 0:
            elapsed = time.perf_counter() - t0
            running_bpb = total_bits / max(total_bytes, 1)
            seqs_per_sec = (seq_idx + 1) / elapsed
            eta = (total_seqs - seq_idx - 1) / seqs_per_sec
            print(
                f"  seq {seq_idx + 1}/{total_seqs}  "
                f"running_bpb:{running_bpb:.4f}  "
                f"seqs/s:{seqs_per_sec:.1f}  "
                f"eta:{eta:.0f}s"
            )

    elapsed = time.perf_counter() - t0
    bpb = total_bits / total_bytes
    avg_bits = total_bits / total_tokens_scored
    return bpb, avg_bits, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1: PPM Context Mixer Eval")
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--max-seqs", type=int, default=0, help="0 = all sequences")
    args = parser.parse_args()

    print(f"=== Tier 6 Stage 1: PPM Context Mixer ===")
    print(f"max_order={args.max_order} seq_len={args.seq_len} vocab_size={args.vocab_size}")

    # Load tokenizer and build byte LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes, has_leading_space, is_boundary = build_byte_lut(sp, args.vocab_size)

    # Load validation tokens
    val_pattern = f"{args.data_path}/fineweb_val_*.bin"
    tokens = load_validation_tokens(val_pattern)
    total_seqs = (len(tokens) - 1) // args.seq_len
    if args.max_seqs > 0:
        limit = min(args.max_seqs, total_seqs) * args.seq_len + 1
        tokens = tokens[:limit]
        total_seqs = min(args.max_seqs, total_seqs)
    print(f"val_tokens={len(tokens)}  sequences={total_seqs}")

    # --- Per-document mode ---
    print(f"\n--- Per-document PPM (order 0..{args.max_order}) ---")
    bpb_pd, avg_bits_pd, time_pd = evaluate(
        tokens, args.seq_len, args.max_order, args.vocab_size,
        base_bytes, has_leading_space, is_boundary,
        cumulative=False,
    )
    print(f"RESULT per_doc  bpb:{bpb_pd:.4f}  avg_bits/tok:{avg_bits_pd:.4f}  time:{time_pd:.1f}s")

    # --- Cumulative mode ---
    print(f"\n--- Cumulative PPM (order 0..{args.max_order}) ---")
    bpb_cum, avg_bits_cum, time_cum = evaluate(
        tokens, args.seq_len, args.max_order, args.vocab_size,
        base_bytes, has_leading_space, is_boundary,
        cumulative=True,
    )
    print(f"RESULT cumul    bpb:{bpb_cum:.4f}  avg_bits/tok:{avg_bits_cum:.4f}  time:{time_cum:.1f}s")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Baseline neural (post-quant):  1.2244 BPB")
    print(f"PPM per-document:              {bpb_pd:.4f} BPB")
    print(f"PPM cumulative:                {bpb_cum:.4f} BPB")
    print(f"{'='*60}")
    if bpb_pd < 3.0:
        print("GO: Per-doc mixer is useful. Proceed to Stage 2.")
    elif bpb_cum < 3.0:
        print("GO (cumulative only): Cumulative mixer is useful. Stage 2 should use cumulative mode.")
    else:
        print("STOP: Mixer alone is too weak. Reconsider approach before Stage 2.")


if __name__ == "__main__":
    main()
