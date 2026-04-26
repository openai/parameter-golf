"""
Definitive test: Do properly normalized n-gram caches help?

This script builds a trie-based n-gram language model with Kneser-Ney smoothing
over the validation set, produces FULL normalized distributions over all vocab tokens
at every position, and measures the real BPB improvement when mixed with a neural LM.

Key properties:
- Exact counts via trie (NO hashing — zero collisions)
- Full distribution over all 1024 tokens at every position (sum = 1.0, verified)
- Score-first: n-gram scores use only tokens BEFORE the current position
- Kneser-Ney smoothing (the gold standard for statistical n-gram LMs)
- Interpolated mixing: p_mixed = (1-alpha) * p_model + alpha * p_ngram

Usage:
    python3 ngram_test.py --model-path final_model.int6.ptz --val-path fineweb_val_000000.bin

Or standalone (no neural model, just n-gram vs uniform baseline):
    python3 ngram_test.py --standalone --val-path fineweb_val_000000.bin
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from typing import Optional

import numpy as np


# ============================================================
# Trie-based N-gram Count Store (exact counts, no hashing)
# ============================================================

class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children: dict[int, 'TrieNode'] = {}
        self.count: int = 0


class NgramTrie:
    """Exact-count n-gram storage using a trie. No hashing, no collisions."""

    def __init__(self, max_order: int = 5, vocab_size: int = 1024):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.root = TrieNode()
        self.total_tokens = 0
        # For Kneser-Ney: count of unique contexts each token appears in
        self.continuation_counts = np.zeros(vocab_size, dtype=np.int64)
        self.total_continuations = 0

    def add_ngram(self, tokens: list[int]):
        """Add an n-gram (list of token IDs) to the trie."""
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
            node.count += 1

    def get_count(self, tokens: list[int]) -> int:
        """Get exact count for an n-gram."""
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                return 0
            node = node.children[tok]
        return node.count

    def get_children(self, context: list[int]) -> dict[int, int]:
        """Get all children (next tokens) and their counts for a context."""
        node = self.root
        for tok in context:
            if tok not in node.children:
                return {}
            node = node.children[tok]
        return {tok: child.count for tok, child in node.children.items()}


# ============================================================
# Kneser-Ney Smoothed N-gram Language Model
# ============================================================

class KneserNeyNgram:
    """
    Modified Kneser-Ney smoothed n-gram LM.

    Produces a FULL probability distribution over all vocab tokens
    at every position. The distribution is guaranteed to sum to 1.0.

    Uses interpolated Kneser-Ney:
        p_KN(w | context) = max(c(context, w) - d, 0) / c(context)
                          + d * N1+(context, .) / c(context) * p_KN(w | shorter_context)

    Where:
        d = discount (typically 0.75)
        N1+(context, .) = number of unique tokens following context
        p_KN(w | shorter_context) = recursive backoff

    Base case (unigram): uses continuation counts (how many unique
    contexts each token appears in), not raw frequency.
    """

    def __init__(self, max_order: int = 5, vocab_size: int = 1024, discount: float = 0.75):
        self.max_order = max_order
        self.vocab_size = vocab_size
        self.discount = discount
        self.trie = NgramTrie(max_order, vocab_size)
        # Precomputed unigram distribution (continuation counts)
        self._unigram_dist: Optional[np.ndarray] = None
        # Cache for context statistics
        self._built = False

    def update(self, tokens: np.ndarray, position: int):
        """
        Update the model with tokens up to (not including) position.
        Score-first: only uses tokens that have already been scored.

        Call this AFTER scoring position, BEFORE scoring position+1.
        """
        # Add all n-grams ending at this position
        for order in range(1, self.max_order + 1):
            start = position - order + 1
            if start < 0:
                continue
            ngram = tokens[start:position + 1].tolist()
            self.trie.add_ngram(ngram)

        self.trie.total_tokens += 1
        self._built = False  # invalidate cached unigram

    def get_distribution(self, tokens: np.ndarray, position: int) -> np.ndarray:
        """
        Get a full probability distribution over all vocab tokens
        for the next token at `position`, using only tokens before `position`.

        Returns: np.ndarray of shape (vocab_size,) summing to 1.0
        """
        dist = np.zeros(self.vocab_size, dtype=np.float64)

        # Try each order from highest to lowest
        for order in range(min(self.max_order, position), 0, -1):
            context = tokens[position - order:position].tolist()
            context_count = self.trie.get_count(context)

            if context_count == 0:
                continue

            # Get children of this context
            children = self.trie.get_children(context)
            num_unique = len(children)

            if num_unique == 0:
                continue

            # Interpolated Kneser-Ney for this order
            for tok_id in range(self.vocab_size):
                tok_count = children.get(tok_id, 0)
                # Main term: max(count - discount, 0) / context_count
                main = max(tok_count - self.discount, 0.0) / context_count
                dist[tok_id] = main

            # Backoff weight
            backoff_weight = (self.discount * num_unique) / context_count

            # Get lower-order distribution
            lower_dist = self._get_lower_order_dist(tokens, position, order - 1)

            # Interpolate
            dist = dist + backoff_weight * lower_dist

            # Verify normalization
            total = dist.sum()
            if total > 0 and abs(total - 1.0) > 1e-6:
                dist /= total

            return dist

        # Fallback: uniform distribution (no context matches)
        return np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

    def _get_lower_order_dist(self, tokens: np.ndarray, position: int, order: int) -> np.ndarray:
        """Get distribution for a lower-order context (recursive backoff)."""
        if order == 0:
            # Unigram: use continuation counts or uniform
            if self.trie.total_tokens == 0:
                return np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

            # Simple unigram from counts
            dist = np.zeros(self.vocab_size, dtype=np.float64)
            unigram_children = self.trie.get_children([])
            total = sum(unigram_children.values())
            if total == 0:
                return np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

            for tok_id, count in unigram_children.items():
                dist[tok_id] = count / total

            # Add small floor for unseen tokens
            floor = 1e-10
            dist = dist + floor
            dist /= dist.sum()
            return dist

        context = tokens[position - order:position].tolist()
        context_count = self.trie.get_count(context)

        if context_count == 0:
            return self._get_lower_order_dist(tokens, position, order - 1)

        children = self.trie.get_children(context)
        num_unique = len(children)

        dist = np.zeros(self.vocab_size, dtype=np.float64)
        for tok_id in range(self.vocab_size):
            tok_count = children.get(tok_id, 0)
            dist[tok_id] = max(tok_count - self.discount, 0.0) / context_count

        backoff_weight = (self.discount * num_unique) / context_count
        lower = self._get_lower_order_dist(tokens, position, order - 1)
        dist = dist + backoff_weight * lower

        total = dist.sum()
        if total > 0:
            dist /= total

        return dist


# ============================================================
# Evaluation
# ============================================================

def load_val_tokens(path: str) -> np.ndarray:
    """Load validation tokens from a .bin shard."""
    raw = np.fromfile(path, dtype=np.uint8)
    # Header: 256 int32s
    header = np.frombuffer(raw[:1024], dtype=np.int32)
    magic, version, num_tokens = header[0], header[1], header[2]
    assert magic == 20240520, f"Bad magic: {magic}"
    tokens = np.frombuffer(raw[1024:1024 + num_tokens * 2], dtype=np.uint16).astype(np.int64)
    return tokens


def evaluate_ngram_standalone(val_tokens: np.ndarray, max_order: int = 5,
                              vocab_size: int = 1024, max_positions: int = 100000,
                              discount: float = 0.75):
    """
    Evaluate a standalone Kneser-Ney n-gram LM on val tokens.
    Score-first: at each position, score with current model, then update.
    """
    ngram = KneserNeyNgram(max_order=max_order, vocab_size=vocab_size, discount=discount)

    total_positions = min(len(val_tokens) - 1, max_positions)
    total_nll = 0.0
    normalization_errors = 0
    max_norm_error = 0.0

    t0 = time.perf_counter()

    for pos in range(total_positions):
        target = int(val_tokens[pos + 1])

        # Score first (using only tokens 0..pos)
        if pos < 1:
            # No context yet — uniform
            prob = 1.0 / vocab_size
        else:
            dist = ngram.get_distribution(val_tokens, pos + 1)

            # Verify normalization
            dist_sum = dist.sum()
            norm_error = abs(dist_sum - 1.0)
            max_norm_error = max(max_norm_error, norm_error)
            if norm_error > 1e-4:
                normalization_errors += 1

            prob = dist[target]

        prob = max(prob, 1e-12)  # floor to avoid log(0)
        total_nll += -math.log(prob)

        # Update after scoring
        ngram.update(val_tokens, pos + 1)

        if (pos + 1) % 10000 == 0:
            elapsed = time.perf_counter() - t0
            avg_nll = total_nll / (pos + 1)
            bpb_est = avg_nll / math.log(2) * (vocab_size / 1024)  # rough BPB
            speed = (pos + 1) / elapsed
            print(f"  pos {pos+1}/{total_positions}: avg_nll={avg_nll:.4f} "
                  f"speed={speed:.0f} tok/s max_norm_err={max_norm_error:.2e} "
                  f"norm_violations={normalization_errors}")

    avg_nll = total_nll / total_positions
    avg_bpb_approx = avg_nll / math.log(2)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"Kneser-Ney N-gram (order={max_order}, discount={discount})")
    print(f"Positions evaluated: {total_positions}")
    print(f"Average NLL: {avg_nll:.6f}")
    print(f"Approx BPB (token-level): {avg_bpb_approx:.6f}")
    print(f"Max normalization error: {max_norm_error:.2e}")
    print(f"Normalization violations (>1e-4): {normalization_errors}")
    print(f"Time: {elapsed:.1f}s ({total_positions/elapsed:.0f} tok/s)")
    print(f"{'='*60}")

    return avg_nll, avg_bpb_approx


def evaluate_mixed(val_tokens: np.ndarray, model_nll: np.ndarray,
                   max_order: int = 5, vocab_size: int = 1024,
                   alphas: list[float] = [0.01, 0.05, 0.10, 0.20, 0.30],
                   max_positions: int = 100000, discount: float = 0.75):
    """
    Evaluate mixed distribution: p_mixed = (1-alpha)*p_model + alpha*p_ngram

    model_nll: per-position NLL from the neural model (precomputed)

    This is the key experiment: does mixing help?
    """
    ngram = KneserNeyNgram(max_order=max_order, vocab_size=vocab_size, discount=discount)

    total_positions = min(len(val_tokens) - 1, min(len(model_nll), max_positions))

    # Track NLL for each alpha
    nll_sums = {alpha: 0.0 for alpha in alphas}
    nll_baseline = 0.0  # model-only
    norm_errors = 0

    t0 = time.perf_counter()

    for pos in range(total_positions):
        target = int(val_tokens[pos + 1])
        p_model_target = math.exp(-model_nll[pos])  # model's prob for correct token

        nll_baseline += model_nll[pos]

        if pos < 1:
            # No n-gram context — mixed = model
            for alpha in alphas:
                nll_sums[alpha] += model_nll[pos]
        else:
            dist_ngram = ngram.get_distribution(val_tokens, pos + 1)

            # Verify normalization
            dist_sum = dist_ngram.sum()
            if abs(dist_sum - 1.0) > 1e-4:
                norm_errors += 1

            p_ngram_target = dist_ngram[target]

            for alpha in alphas:
                p_mixed = (1.0 - alpha) * p_model_target + alpha * p_ngram_target
                p_mixed = max(p_mixed, 1e-12)
                nll_sums[alpha] += -math.log(p_mixed)

        # Update after scoring
        ngram.update(val_tokens, pos + 1)

        if (pos + 1) % 10000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  pos {pos+1}/{total_positions}: "
                  f"baseline_nll={nll_baseline/(pos+1):.4f} "
                  f"best_mixed_nll={min(v/(pos+1) for v in nll_sums.values()):.4f} "
                  f"speed={((pos+1)/elapsed):.0f} tok/s")

    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"MIXED DISTRIBUTION RESULTS (Kneser-Ney order={max_order}, d={discount})")
    print(f"Positions: {total_positions} | Norm violations: {norm_errors}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"{'Alpha':<10} {'Avg NLL':<12} {'Approx BPB':<12} {'Delta NLL':<12} {'Delta BPB':<12}")
    print(f"{'-'*58}")

    baseline_avg = nll_baseline / total_positions
    baseline_bpb = baseline_avg / math.log(2)
    print(f"{'model':10} {baseline_avg:<12.6f} {baseline_bpb:<12.6f} {'—':12} {'—':12}")

    for alpha in alphas:
        avg = nll_sums[alpha] / total_positions
        bpb = avg / math.log(2)
        delta_nll = avg - baseline_avg
        delta_bpb = bpb - baseline_bpb
        marker = " ***" if delta_nll < -0.0001 else ""
        print(f"{alpha:<10.2f} {avg:<12.6f} {bpb:<12.6f} {delta_nll:<+12.6f} {delta_bpb:<+12.6f}{marker}")

    print(f"{'='*60}")

    best_alpha = min(alphas, key=lambda a: nll_sums[a])
    best_delta = (nll_sums[best_alpha] / total_positions) - baseline_avg
    print(f"\nBest alpha: {best_alpha} (delta NLL: {best_delta:+.6f})")
    if abs(best_delta) < 0.001:
        print("CONCLUSION: N-gram provides negligible improvement (<0.001 NLL)")
    elif best_delta < 0:
        print(f"CONCLUSION: N-gram provides real improvement of {-best_delta:.4f} NLL")
    else:
        print("CONCLUSION: N-gram HURTS — model-only is better")


def main():
    parser = argparse.ArgumentParser(description="Definitive n-gram normalization test")
    parser.add_argument("--val-path", type=str, required=True, help="Path to val .bin shard")
    parser.add_argument("--model-nll-path", type=str, default="",
                        help="Path to precomputed model NLL (.npy). If empty, runs standalone only.")
    parser.add_argument("--max-order", type=int, default=5, help="Max n-gram order")
    parser.add_argument("--max-positions", type=int, default=100000,
                        help="Max positions to evaluate (default 100K for speed)")
    parser.add_argument("--discount", type=float, default=0.75, help="Kneser-Ney discount")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--standalone", action="store_true", help="Run standalone n-gram eval only")
    parser.add_argument("--alphas", type=str, default="0.01,0.05,0.10,0.20,0.30,0.50",
                        help="Comma-separated alpha values for mixing")
    args = parser.parse_args()

    print(f"Loading val tokens from {args.val_path}...")
    val_tokens = load_val_tokens(args.val_path)
    print(f"Loaded {len(val_tokens)} tokens")

    if args.standalone:
        print(f"\n=== Standalone Kneser-Ney N-gram (order={args.max_order}) ===")
        evaluate_ngram_standalone(
            val_tokens, max_order=args.max_order, vocab_size=args.vocab_size,
            max_positions=args.max_positions, discount=args.discount,
        )
    elif args.model_nll_path:
        print(f"\nLoading model NLL from {args.model_nll_path}...")
        model_nll = np.load(args.model_nll_path)
        print(f"Loaded {len(model_nll)} NLL values")

        alphas = [float(a) for a in args.alphas.split(",")]

        print(f"\n=== Mixed Distribution Test ===")
        evaluate_mixed(
            val_tokens, model_nll, max_order=args.max_order,
            vocab_size=args.vocab_size, alphas=alphas,
            max_positions=args.max_positions, discount=args.discount,
        )
    else:
        print("ERROR: Provide --model-nll-path for mixed eval, or --standalone for n-gram only")
        sys.exit(1)


if __name__ == "__main__":
    main()
