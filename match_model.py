"""Match Model — Eval-time longest-match predictor for Parameter Golf.

Core idea: maintain a hash index of all token sequences seen during evaluation.
When predicting token t at position p, find the longest suffix of the context
that appeared earlier in the data. Then use the empirical distribution of the
token following that suffix as the prediction.

V2: Stores {context_hash → {next_token → count}} instead of position lists.
This reduces memory by ~10x and makes predict() O(1) per order level.
Hash collisions at order ≥ 6 are negligible (< 10⁻⁶ per lookup).
"""

from collections import defaultdict
import numpy as np


class MatchModel:
    """Hash-based longest-match predictor (count-based, memory-efficient).

    For each position, finds the longest exact prefix match in history and
    returns the empirical next-token distribution from accumulated counts.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        min_order: Minimum context length to consider a match.
        max_order: Maximum context length to search for.
        min_count: Minimum total count to trust the prediction.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        min_order: int = 4,
        max_order: int = 12,
        min_count: int = 1,
    ):
        self.vocab_size = vocab_size
        self.min_order = min_order
        self.max_order = max_order
        self.min_count = min_count

        # tables[k - min_order]: context_hash → {next_token_id → count}
        self.tables: list[dict[int, dict[int, int]]] = [
            {} for _ in range(max_order - min_order + 1)
        ]

        # Rolling context buffer (ring buffer for recent tokens)
        self.context_buf: list[int] = []
        self.total_tokens = 0

        # Stats
        self.n_predictions = 0
        self.n_matches = 0
        self.match_lengths: list[int] = []

        # Preallocated buffer for predict() — avoids 10M+ np.zeros() allocations
        self._pred_buf = np.zeros(vocab_size, dtype=np.float64)

    def _context_hash(self, tokens: list[int], start: int, length: int) -> int:
        """Hash a subsequence of the context buffer."""
        return hash(tuple(tokens[start:start + length]))

    def update(self, token: int) -> None:
        """Add a token to history and update all hash tables.

        For each order k, the context is the k tokens BEFORE `token`,
        and we record that `token` followed that context.
        """
        self.context_buf.append(token)
        self.total_tokens += 1
        pos = len(self.context_buf) - 1  # index of token just added

        # For each order k, register context[pos-k : pos] → token
        for k in range(self.min_order, min(self.max_order + 1, pos + 1)):
            ctx = tuple(self.context_buf[pos - k : pos])
            h = hash(ctx)
            table = self.tables[k - self.min_order]
            if h not in table:
                table[h] = {}
            counts = table[h]
            counts[token] = counts.get(token, 0) + 1

    def predict(self, context: list[int]) -> tuple[np.ndarray | None, int]:
        """Predict next token distribution given context.

        Args:
            context: List of preceding tokens (the current context window).

        Returns:
            (probs, match_length) where probs is a normalized probability
            distribution over vocab_size if a match was found, else None.
            match_length is the length of the longest match found.
        """
        self.n_predictions += 1

        if len(context) < self.min_order:
            return None, 0

        # Search from longest to shortest context
        for k in range(min(self.max_order, len(context)), self.min_order - 1, -1):
            ctx = tuple(context[-k:])
            h = hash(ctx)
            table = self.tables[k - self.min_order]
            counts = table.get(h)

            if counts is not None:
                total = sum(counts.values())
                if total >= self.min_count:
                    # Build probability distribution from counts (reuse buffer)
                    buf = self._pred_buf
                    buf[:] = 0.0
                    for tok_id, count in counts.items():
                        if 0 <= tok_id < self.vocab_size:
                            buf[tok_id] = count
                    buf /= buf.sum()
                    # Return a copy so caller can safely store the result
                    probs = buf.copy()

                    self.n_matches += 1
                    self.match_lengths.append(k)
                    return probs, k

        return None, 0

    def batch_update(self, tokens: list[int]) -> None:
        """Efficiently add multiple tokens to history."""
        for token in tokens:
            self.update(token)

    def stats(self) -> dict[str, float]:
        """Return prediction statistics."""
        total_entries = sum(len(table) for table in self.tables)
        total_counts = sum(
            sum(sum(c.values()) for c in table.values())
            for table in self.tables
        )
        result = {
            "n_predictions": self.n_predictions,
            "n_matches": self.n_matches,
            "match_rate": self.n_matches / max(self.n_predictions, 1),
            "total_tokens": self.total_tokens,
            "avg_match_length": (
                float(np.mean(self.match_lengths)) if self.match_lengths else 0.0
            ),
            "max_match_length": (
                max(self.match_lengths) if self.match_lengths else 0
            ),
            "total_unique_contexts": total_entries,
            "total_count_entries": total_counts,
            # Memory estimate: each dict entry ~120 bytes (key + inner dict + counts)
            "estimated_memory_mb": total_entries * 120 / 1e6,
        }
        return result

    def reset(self) -> None:
        """Clear all state."""
        self.tables = [{} for _ in range(self.max_order - self.min_order + 1)]
        self.context_buf.clear()
        self.total_tokens = 0
        self.n_predictions = 0
        self.n_matches = 0
        self.match_lengths.clear()
