"""
Online N-gram Predictor with PPM (Prediction by Partial Matching) Backoff.

Builds n-gram statistics incrementally as tokens are revealed during evaluation.
Zero storage in the model artifact — everything is computed at eval time.

Usage:
    ngram = SparseNgramPredictor(vocab_size=1024, max_order=6)
    for token in tokens:
        probs = ngram.predict(context)  # returns [vocab_size] numpy array
        ngram.update(context + [token])
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class SparseNgramPredictor:
    """Sparse online n-gram model with PPM escape-based backoff."""

    def __init__(self, vocab_size: int = 1024, max_order: int = 6):
        self.vocab_size = vocab_size
        self.max_order = max_order
        # tables[n] maps context tuple → {token_id: count}
        # n=0: unigram (empty context), n=1: bigram, ..., n=max_order-1
        self.tables: list[dict[tuple[int, ...], dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(max_order)
        ]
        self._total_tokens = 0
        self._unigram_counts = np.zeros(vocab_size, dtype=np.float64)

    def update_token(self, history: list[int]) -> None:
        """Update counts after a single token is revealed.
        
        Args:
            history: list of token IDs ending with the newly revealed token.
                     We extract the last token as target and preceding tokens as context.
        """
        if len(history) < 1:
            return
        target = history[-1]
        self._total_tokens += 1
        self._unigram_counts[target] += 1

        for n in range(self.max_order):
            ctx_len = n  # context length for order n+1
            if len(history) < ctx_len + 1:
                break
            ctx = tuple(history[-(ctx_len + 1):-1]) if ctx_len > 0 else ()
            self.tables[n][ctx][target] += 1

    def update_batch(self, token_ids: list[int], start_pos: int = 0) -> None:
        """Update counts for a batch of consecutive tokens.
        
        More efficient than calling update_token repeatedly.
        
        Args:
            token_ids: full sequence of token IDs
            start_pos: position to start updating from (tokens before this
                       are assumed to already be in the model)
        """
        for i in range(max(start_pos, 0), len(token_ids)):
            ctx_start = max(0, i - self.max_order + 1)
            self.update_token(token_ids[ctx_start:i + 1])

    def predict(self, context: list[int], alpha: float = 0.5) -> np.ndarray:
        """Predict next token distribution using PPM escape.

        P(w|ctx_n) = (c(ctx_n, w) + alpha * P(w|ctx_{n-1})) / (C(ctx_n) + alpha)

        Falls back to unigram, then uniform.

        Args:
            context: list of preceding token IDs (most recent last)
            alpha: escape probability weight (higher = more smoothing)

        Returns:
            Probability distribution over vocab, shape [vocab_size]
        """
        # Start with uniform fallback
        probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        # Layer on unigram
        if self._total_tokens > 0:
            probs = (self._unigram_counts + alpha * probs) / (self._total_tokens + alpha)

        # Layer on higher-order n-grams (bigram, trigram, ...)
        for n in range(1, self.max_order):
            ctx_len = n
            if len(context) < ctx_len:
                break
            ctx = tuple(context[-ctx_len:])
            if ctx not in self.tables[n]:
                continue
            counts = self.tables[n][ctx]
            total = sum(counts.values())
            if total == 0:
                continue

            # PPM escape: mix current order with lower-order backoff
            new_probs = alpha * probs.copy()
            for tok, c in counts.items():
                new_probs[tok] += c
            new_probs /= (total + alpha)
            probs = new_probs

        return probs

    def predict_log_probs(self, context: list[int], alpha: float = 0.5) -> np.ndarray:
        """Return log probabilities (natural log)."""
        probs = self.predict(context, alpha)
        return np.log(np.maximum(probs, 1e-30))

    def stats(self) -> dict[str, int]:
        """Return memory usage statistics."""
        total_contexts = sum(len(t) for t in self.tables)
        total_entries = sum(
            sum(len(v) for v in t.values()) for t in self.tables
        )
        return {
            "total_tokens_seen": self._total_tokens,
            "total_unique_contexts": total_contexts,
            "total_count_entries": total_entries,
            "estimated_bytes": total_entries * 16 + total_contexts * 64,
        }
