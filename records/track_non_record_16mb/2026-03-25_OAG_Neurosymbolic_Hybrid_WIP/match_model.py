"""
LZ77-style longest substring match predictor for eval-time BPB improvement.

At eval time, maintains a buffer of already-scored tokens. For each new position,
finds the longest match in the history and predicts the next token from the
match continuation. Zero artifact cost (no learned parameters).

Reference: Ziv & Lempel (1977), "A Universal Algorithm for Sequential Data Compression"

Legal: purely backward-looking. Only uses tokens that have already been scored.
"""
import torch
from typing import Optional, Tuple


class MatchModel:
    """LZ77-style longest substring match predictor.

    Maintains a suffix-based index of all previously scored tokens for
    O(max_order * history_size) lookup. Predicts next token based on
    what followed the longest matching context in history.
    """

    def __init__(self, vocab_size: int, max_order: int = 32, min_match: int = 3):
        """
        Args:
            vocab_size: Size of token vocabulary.
            max_order: Maximum context length to search for matches.
            min_match: Minimum match length to produce a prediction.
        """
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.min_match = min_match
        self.history: list[int] = []
        # Suffix index: maps tuple of tokens -> list of positions where that tuple starts
        self._suffix_index: dict[tuple, list[int]] = {}
        self._max_index_order = 8  # Index up to this length for fast lookup

    def observe(self, token_ids: list[int]) -> None:
        """Add scored tokens to history and update suffix index."""
        start_pos = len(self.history)
        self.history.extend(token_ids)

        # Update suffix index for new tokens
        for i in range(start_pos, len(self.history)):
            for order in range(1, min(self._max_index_order + 1, i + 1)):
                key = tuple(self.history[i - order + 1 : i + 1])
                if key not in self._suffix_index:
                    self._suffix_index[key] = []
                self._suffix_index[key].append(i - order + 1)

    def predict(self, context: list[int]) -> Tuple[Optional[torch.Tensor], float]:
        """Find longest match in history and predict continuation.

        Args:
            context: Recent token IDs (the context to match against).

        Returns:
            (probs, confidence) where probs is a distribution over vocab_size,
            or (None, 0.0) if no sufficient match found.
        """
        if len(self.history) < self.min_match + 1 or len(context) < self.min_match:
            return None, 0.0

        best_match_len = 0
        best_continuations: dict[int, int] = {}  # token -> count

        # Try decreasing context lengths to find longest match
        ctx = context[-self.max_order:]
        for match_len in range(min(len(ctx), self.max_order), self.min_match - 1, -1):
            query = tuple(ctx[-match_len:])

            # Use suffix index for short queries, brute force for long ones
            if match_len <= self._max_index_order:
                positions = self._suffix_index.get(query, [])
            else:
                # For longer queries, start from indexed shorter prefix and filter
                short_key = tuple(query[:self._max_index_order])
                candidates = self._suffix_index.get(short_key, [])
                positions = []
                for pos in candidates:
                    if pos + match_len < len(self.history):
                        if tuple(self.history[pos:pos + match_len]) == query:
                            positions.append(pos)

            # Collect continuations from all matching positions
            continuations: dict[int, int] = {}
            for pos in positions:
                next_pos = pos + match_len
                if next_pos < len(self.history):
                    tok = self.history[next_pos]
                    continuations[tok] = continuations.get(tok, 0) + 1

            if continuations:
                if match_len > best_match_len:
                    best_match_len = match_len
                    best_continuations = continuations
                    if match_len >= self._max_index_order:
                        break  # Long match found, good enough

        if not best_continuations or best_match_len < self.min_match:
            return None, 0.0

        # Build probability distribution from continuations
        total = sum(best_continuations.values())
        probs = torch.zeros(self.vocab_size, dtype=torch.float32)
        for tok, count in best_continuations.items():
            if tok < self.vocab_size:
                probs[tok] = count / total

        # Smooth slightly to avoid zero probs
        smoothing = 1e-6
        probs = probs + smoothing
        probs = probs / probs.sum()

        # Confidence based on match length and number of examples
        match_conf = min(0.9, 0.3 + 0.1 * best_match_len)
        count_conf = min(0.95, total / (total + 2))
        confidence = match_conf * count_conf

        return probs, confidence

    def reset(self) -> None:
        """Clear all history and index."""
        self.history.clear()
        self._suffix_index.clear()
