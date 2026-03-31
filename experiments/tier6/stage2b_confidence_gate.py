#!/usr/bin/env python3
"""
Tier 6 — Stage 2b: Confidence-gated mixture (no learned params).

Instead of a fixed alpha, the mixing weight adapts per-token based on
the neural model's confidence. When the neural model is confident,
trust it more. When uncertain, lean on the mixer.

Confidence proxy: p_neural(target) — the probability the neural model
assigned to the correct token. High p = confident, low p = uncertain.

Gate function:
  alpha(t) = alpha_min + (alpha_max - alpha_min) * sigmoid(k * (p_neural(t) - threshold))

This gives a smooth S-curve: below threshold -> alpha_min (trust mixer),
above threshold -> alpha_max (trust neural).

No learned parameters — all knobs are grid-searched offline.

Usage:
  python3 experiments/tier6/stage2b_confidence_gate.py \
      --neural-logprobs experiments/tier6/neural_logprobs_2k.npz
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Data loading + PPM mixer (shared with Stage 2a)
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
        if piece.startswith("\u2581"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


class PPMContextMixer:
    def __init__(self, max_order: int, vocab_size: int):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(max_order + 1)]
        self.totals = [defaultdict(int) for _ in range(max_order + 1)]
        self.uniques = [defaultdict(int) for _ in range(max_order + 1)]

    def reset(self):
        for k in range(self.max_order + 1):
            self.counts[k].clear()
            self.totals[k].clear()
            self.uniques[k].clear()

    def update(self, context: tuple, symbol: int):
        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            c = self.counts[k][ctx]
            if symbol not in c:
                self.uniques[k][ctx] += 1
            c[symbol] += 1
            self.totals[k][ctx] += 1

    def predict(self, context: tuple, symbol: int) -> float:
        prob = 0.0
        escape_product = 1.0
        for k in range(min(len(context), self.max_order), -1, -1):
            ctx = context[-k:] if k > 0 else ()
            total = self.totals[k].get(ctx, 0)
            unique = self.uniques[k].get(ctx, 0)
            if total == 0:
                continue
            denom = total + unique
            sym_count = self.counts[k][ctx].get(symbol, 0)
            if sym_count > 0:
                prob += escape_product * (sym_count / denom)
            escape_product *= unique / denom
        prob += escape_product * (1.0 / self.vocab_size)
        return prob


def compute_mixer_logprobs(tokens, seq_len, max_order, vocab_size, cumulative):
    mixer = PPMContextMixer(max_order, vocab_size)
    total_seqs = (len(tokens) - 1) // seq_len
    result = np.zeros(total_seqs * seq_len, dtype=np.float32)
    idx = 0
    for seq_idx in range(total_seqs):
        if not cumulative:
            mixer.reset()
        start = seq_idx * seq_len
        seq = tokens[start: start + seq_len + 1]
        for t in range(seq_len):
            target_id = int(seq[t + 1])
            ctx_start = max(0, t + 1 - max_order)
            context = tuple(int(seq[i]) for i in range(ctx_start, t + 1))
            p = mixer.predict(context, target_id)
            if p <= 0:
                p = 1e-12
            result[idx] = -math.log2(p)
            mixer.update(context, target_id)
            idx += 1
        if (seq_idx + 1) % 5000 == 0:
            print(f"  mixer: {seq_idx + 1}/{total_seqs}")
    print(f"  mixer done: {total_seqs} seqs")
    return result


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def confidence_gated_bpb(
    neural_neg_log2_p: np.ndarray,
    mixer_neg_log2_p: np.ndarray,
    target_ids: np.ndarray,
    prev_ids: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary: np.ndarray,
    alpha_min: float,
    alpha_max: float,
    threshold: float,
    steepness: float,
) -> float:
    """Compute BPB with per-token confidence-gated mixing."""
    # Byte counts
    token_bytes = base_bytes[target_ids].astype(np.float64)
    space_mask = has_leading_space[target_ids] & ~is_boundary[prev_ids]
    token_bytes += space_mask.astype(np.float64)
    total_bytes = token_bytes.sum()

    # Neural confidence = p_neural(target)
    neural_p = np.power(2.0, -np.clip(neural_neg_log2_p, 0, 50).astype(np.float64))
    mixer_p = np.power(2.0, -np.clip(mixer_neg_log2_p, 0, 50).astype(np.float64))

    # Per-token alpha via sigmoid gate
    alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(steepness * (neural_p - threshold))

    # Mix
    mixed_p = alpha * neural_p + (1.0 - alpha) * mixer_p
    mixed_p = np.maximum(mixed_p, 1e-30)
    neg_log2_mixed = -np.log2(mixed_p)
    return float(neg_log2_mixed.sum() / total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2b: Confidence gate sweep")
    parser.add_argument("--neural-logprobs", required=True)
    parser.add_argument("--data-path", default="./data/datasets/fineweb10B_sp1024")
    parser.add_argument("--tokenizer-path", default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--max-seqs", type=int, default=0)
    args = parser.parse_args()

    print("=== Tier 6 Stage 2b: Confidence-Gated Mixture ===")

    # Load neural log-probs
    data = np.load(args.neural_logprobs)
    neural_neg_log2_p = data["neg_log2_p"]
    target_ids = data["target_ids"]
    prev_ids = data["prev_ids"]
    saved_seq_len = int(data["seq_len"])
    if saved_seq_len != args.seq_len:
        raise ValueError(f"Seq-len mismatch: NPZ={saved_seq_len} vs --seq-len={args.seq_len}")

    total_tokens = len(neural_neg_log2_p)
    total_seqs = total_tokens // saved_seq_len
    if args.max_seqs > 0 and args.max_seqs < total_seqs:
        limit = args.max_seqs * saved_seq_len
        neural_neg_log2_p = neural_neg_log2_p[:limit]
        target_ids = target_ids[:limit]
        prev_ids = prev_ids[:limit]
        total_seqs = args.max_seqs
        total_tokens = limit
    print(f"  {total_tokens} tokens, {total_seqs} sequences")

    # Byte LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes, has_leading_space, is_boundary = build_byte_lut(sp, args.vocab_size)

    # Neural-only baseline
    token_byte_counts = base_bytes[target_ids].astype(np.float64)
    space_mask = has_leading_space[target_ids] & ~is_boundary[prev_ids]
    token_byte_counts += space_mask.astype(np.float64)
    total_bytes = token_byte_counts.sum()
    neural_bpb = float(neural_neg_log2_p.sum() / total_bytes)
    print(f"  neural-only BPB: {neural_bpb:.4f}")

    # Load val tokens for mixer
    val_pattern = f"{args.data_path}/fineweb_val_*.bin"
    tokens = load_validation_tokens(val_pattern)
    if args.max_seqs > 0:
        tokens = tokens[:args.max_seqs * args.seq_len + 1]

    # Neural confidence distribution (for setting grid ranges)
    neural_p = np.power(2.0, -np.clip(neural_neg_log2_p, 0, 50).astype(np.float64))
    p10, p25, p50, p75, p90 = np.percentile(neural_p, [10, 25, 50, 75, 90])
    print(f"  neural p(target) percentiles: p10={p10:.4f} p25={p25:.4f} "
          f"p50={p50:.4f} p75={p75:.4f} p90={p90:.4f}")

    for cumulative in [False, True]:
        mode = "cumulative" if cumulative else "per-doc"
        print(f"\n{'='*60}")
        print(f"  Mode: {mode} PPM (order {args.max_order})")
        print(f"{'='*60}")

        mixer_logprobs = compute_mixer_logprobs(
            tokens, args.seq_len, args.max_order, args.vocab_size, cumulative
        )
        if len(mixer_logprobs) > total_tokens:
            mixer_logprobs = mixer_logprobs[:total_tokens]

        # First: fixed-alpha baseline (best from Stage 2a) for comparison
        best_fixed_alpha = 0.95 if not cumulative else 0.85
        mixer_p_arr = np.power(2.0, -np.clip(mixer_logprobs, 0, 50).astype(np.float64))
        fixed_mixed = best_fixed_alpha * neural_p + (1.0 - best_fixed_alpha) * mixer_p_arr
        fixed_mixed = np.maximum(fixed_mixed, 1e-30)
        fixed_bpb = float((-np.log2(fixed_mixed)).sum() / total_bytes)
        print(f"\n  Fixed alpha={best_fixed_alpha:.2f} baseline: {fixed_bpb:.4f} BPB")

        # Grid search over gate parameters
        # alpha_min: how much to trust neural when UNconfident
        # alpha_max: how much to trust neural when confident
        # threshold: p_neural cutoff for the sigmoid center
        # steepness: how sharp the transition is
        print(f"\n  Grid search:")
        print(f"  {'alpha_min':>9} {'alpha_max':>9} {'threshold':>9} {'steepness':>9} {'bpb':>8} {'gain':>8}")

        best_bpb = neural_bpb
        best_config = None

        for alpha_min in [0.50, 0.60, 0.70, 0.80]:
            for alpha_max in [0.95, 0.97, 0.99, 1.00]:
                if alpha_min >= alpha_max:
                    continue
                for threshold in [p25, p50, p75]:
                    for steepness in [5.0, 10.0, 20.0, 50.0]:
                        bpb = confidence_gated_bpb(
                            neural_neg_log2_p, mixer_logprobs,
                            target_ids, prev_ids,
                            base_bytes, has_leading_space, is_boundary,
                            alpha_min, alpha_max, threshold, steepness,
                        )
                        gain = neural_bpb - bpb
                        if bpb < best_bpb:
                            best_bpb = bpb
                            best_config = (alpha_min, alpha_max, threshold, steepness)
                            print(f"  {alpha_min:9.2f} {alpha_max:9.2f} {threshold:9.4f} "
                                  f"{steepness:9.1f} {bpb:8.4f} {gain:+8.4f} *")

        if best_config:
            print(f"\n  Best gated ({mode}): {best_bpb:.4f} BPB  config={best_config}")
            print(f"  vs fixed-alpha:     {fixed_bpb:.4f} BPB")
            print(f"  vs neural-only:     {neural_bpb:.4f} BPB")
            print(f"  Gate improvement over fixed: {fixed_bpb - best_bpb:.4f} BPB")
            print(f"  Gate improvement over neural: {neural_bpb - best_bpb:.4f} BPB")
        else:
            print(f"\n  No gated config beat neural-only. Fixed alpha is still best.")

    print(f"\n{'='*60}")
    print("Stage 2b complete.")


if __name__ == "__main__":
    main()
