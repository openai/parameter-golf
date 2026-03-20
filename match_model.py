"""Match Model — Eval-time longest-match predictor for Parameter Golf.

Core idea: maintain a hash index of all token sequences seen during evaluation.
When predicting token t at position p, find the longest suffix of the context
(tokens[0:p]) that appeared earlier in the data. Then use the empirical
distribution of the token following that suffix as the prediction.

This exploits the massive redundancy in web text: HTML boilerplate, navigation
bars, cookie notices, legal disclaimers, etc. repeat verbatim across documents.
The transformer does soft attention over ~1024 tokens; the Match Model does
EXACT matching over the ENTIRE history (~10M tokens).

Expected BPB improvement: -0.01 to -0.02 (our core differentiator).
"""

from collections import defaultdict
import numpy as np


class MatchModel:
    """Hash-based longest-match predictor.

    For each position, finds the longest exact prefix match in history and
    returns the empirical next-token distribution from all matching positions.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        min_order: Minimum context length to consider a match.
        max_order: Maximum context length to search for.
        min_count: Minimum number of matching positions to trust the prediction.
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

        # hash(context_tuple) → list of positions where this context appeared
        # Indexed by order: tables[k - min_order] stores k-gram contexts
        self.tables: list[dict[int, list[int]]] = [
            defaultdict(list) for _ in range(max_order - min_order + 1)
        ]

        # Full history of tokens seen
        self.history: list[int] = []

        # Stats
        self.n_predictions = 0
        self.n_matches = 0
        self.match_lengths: list[int] = []

    def _context_hash(self, context: list[int] | tuple[int, ...]) -> int:
        """Fast hash for a context sequence."""
        # Use Python's built-in tuple hash (very fast, good distribution)
        return hash(tuple(context))

    def update(self, token: int) -> None:
        """Add a token to history and update all hash tables."""
        self.history.append(token)
        pos = len(self.history) - 1  # position of the token just added

        # For each order k, register the context of length k ending BEFORE this token.
        # The context is history[pos-k : pos], and the "next token" is history[pos].
        for k in range(self.min_order, min(self.max_order + 1, pos + 1)):
            ctx = tuple(self.history[pos - k : pos])
            h = self._context_hash(ctx)
            self.tables[k - self.min_order][h].append(pos)

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
        best_length = 0
        best_positions: list[int] = []

        for k in range(min(self.max_order, len(context)), self.min_order - 1, -1):
            ctx = tuple(context[-k:])
            h = self._context_hash(ctx)
            positions = self.tables[k - self.min_order].get(h, [])

            if len(positions) >= self.min_count:
                # Verify at least one position is a true match (not hash collision)
                # by checking the actual context. This is O(matches) but matches
                # are typically few for long contexts.
                verified = []
                for p in positions:
                    if p >= k and tuple(self.history[p - k : p]) == ctx:
                        verified.append(p)

                if len(verified) >= self.min_count:
                    best_length = k
                    best_positions = verified
                    break  # Found longest match, no need to search shorter

        if best_length == 0:
            return None, 0

        # Build empirical distribution from the tokens following each match
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for p in best_positions:
            if p < len(self.history):
                next_token = self.history[p]
                counts[next_token] += 1

        total = counts.sum()
        if total == 0:
            return None, 0

        self.n_matches += 1
        self.match_lengths.append(best_length)

        probs = counts / total
        return probs, best_length

    def batch_update(self, tokens: list[int]) -> None:
        """Efficiently add multiple tokens to history."""
        for token in tokens:
            self.update(token)

    def stats(self) -> dict[str, float]:
        """Return prediction statistics."""
        result = {
            "n_predictions": self.n_predictions,
            "n_matches": self.n_matches,
            "match_rate": self.n_matches / max(self.n_predictions, 1),
            "history_size": len(self.history),
            "avg_match_length": (
                np.mean(self.match_lengths) if self.match_lengths else 0.0
            ),
            "max_match_length": (
                max(self.match_lengths) if self.match_lengths else 0
            ),
        }
        # Memory usage estimate
        total_entries = sum(
            sum(len(v) for v in table.values()) for table in self.tables
        )
        result["total_hash_entries"] = total_entries
        result["estimated_memory_mb"] = (
            total_entries * 8 + len(self.history) * 4
        ) / 1e6
        return result

    def reset(self) -> None:
        """Clear all state."""
        self.tables = [
            defaultdict(list) for _ in range(self.max_order - self.min_order + 1)
        ]
        self.history.clear()
        self.n_predictions = 0
        self.n_matches = 0
        self.match_lengths.clear()
