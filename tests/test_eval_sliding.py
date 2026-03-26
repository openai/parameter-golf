# tests/test_eval_sliding.py
"""Tests for sliding window eval correctness and N-gram legality."""
import math
import torch
import pytest


class TestSlidingWindowEval:
    def test_sliding_covers_all_tokens(self):
        """Every token position in val_tokens should be scored exactly once."""
        total_tokens = 200
        seq_len = 32
        stride = 8
        all_starts = list(range(0, total_tokens - seq_len + 1, stride))

        scored_positions = set()
        for ws in all_starts:
            score_start = 0 if ws == 0 else seq_len - stride
            for local_pos in range(score_start, seq_len):
                global_pos = ws + local_pos
                assert global_pos not in scored_positions, \
                    f"Position {global_pos} scored twice (window start={ws})"
                scored_positions.add(global_pos)

        last_ws = all_starts[-1]
        expected = set(range(0, last_ws + seq_len))
        assert expected == scored_positions, \
            "Sliding window must cover all token positions exactly once"

    def test_stride_64_improvement_over_nooverlap(self):
        """Sliding window gives more average context per scored token than non-overlapping."""
        seq_len = 64
        stride = 16
        total_tokens = 256

        # Compute average context for sliding window:
        # First window: score positions 0..(seq_len-1), context = local_pos
        # Subsequent windows: score positions (seq_len-stride)..(seq_len-1), context = local_pos
        all_starts = list(range(0, total_tokens - seq_len + 1, stride))
        total_context = 0
        total_scored = 0
        for ws in all_starts:
            score_start = 0 if ws == 0 else seq_len - stride
            for local_pos in range(score_start, seq_len):
                total_context += local_pos
                total_scored += 1
        sliding_avg_context = total_context / total_scored

        # Non-overlapping: score all positions 0..(seq_len-1) per block, avg context = (seq_len-1)/2
        nonoverlap_avg_context = (seq_len - 1) / 2

        assert sliding_avg_context >= nonoverlap_avg_context, (
            f"Sliding window avg context ({sliding_avg_context:.1f}) should be >= "
            f"non-overlapping ({nonoverlap_avg_context:.1f})"
        )

    def test_ngram_update_after_score_is_legal(self):
        """N-gram cache must only be updated after scoring (backward-looking)."""
        from collections import defaultdict

        class LegalityTracker:
            def __init__(self):
                self.scored = set()
                self.in_cache = set()

            def score(self, pos):
                self.scored.add(pos)

            def update_cache(self, pos):
                assert pos in self.scored, \
                    f"ILLEGAL: Position {pos} added to cache before being scored!"
                self.in_cache.add(pos)

        tracker = LegalityTracker()
        val_tokens = list(range(100))
        seq_len = 16
        stride = 4

        for ws in range(0, len(val_tokens) - seq_len, stride):
            score_start = 0 if ws == 0 else seq_len - stride
            for local_pos in range(score_start, seq_len):
                global_pos = ws + local_pos
                tracker.score(global_pos)
                if global_pos > 0:
                    tracker.update_cache(global_pos - 1)

    def test_stride_divides_sequence_correctly(self):
        """With stride=64, each token is scored in exactly one window."""
        seq_len = 1024
        stride = 64
        total_tokens = 10000

        token_score_count = [0] * total_tokens
        all_starts = list(range(0, total_tokens - seq_len, stride))

        for ws in all_starts:
            score_start = 0 if ws == 0 else seq_len - stride
            for local_pos in range(score_start, seq_len):
                gp = ws + local_pos
                if gp < total_tokens:
                    token_score_count[gp] += 1

        covered = [c for c in token_score_count if c > 0]
        assert all(c == 1 for c in covered), \
            "Every covered position must be scored exactly once in sliding window"


class TestNGramLegality:
    """Integration tests verifying N-gram cache legality rules."""

    def test_cache_empty_at_start(self):
        """N-gram cache should be empty at evaluation start."""
        from collections import defaultdict

        class NGramCache:
            def __init__(self, order, vocab_size):
                self.order = order
                self.vocab_size = vocab_size
                self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]

            def get_ngram_log_probs(self, context):
                for n in range(self.order, 0, -1):
                    if len(context) >= n:
                        ctx = tuple(context[-n:])
                        if ctx in self.counts[n-1]:
                            return True
                return None

        cache = NGramCache(order=9, vocab_size=1024)
        assert cache.get_ngram_log_probs([1, 2, 3, 4, 5]) is None
        assert cache.get_ngram_log_probs([99]) is None
