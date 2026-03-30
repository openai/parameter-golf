# tests/test_ngram_cache.py
"""Tests for NGramCache — the highest-priority Track B+ component."""
import math
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict


class NGramCache:
    def __init__(self, order=9, vocab_size=1024):
        self.order = order
        self.vocab_size = vocab_size
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]

    def reset(self):
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.order)]

    def update(self, context, next_token):
        for n in range(1, self.order + 1):
            if len(context) >= n:
                ctx = tuple(context[-n:])
                self.counts[n - 1][ctx][next_token] += 1

    def get_ngram_log_probs(self, context):
        for n in range(self.order, 0, -1):
            if len(context) >= n:
                ctx = tuple(context[-n:])
                if ctx in self.counts[n - 1]:
                    c = self.counts[n - 1][ctx]
                    total = sum(c.values())
                    p = torch.zeros(self.vocab_size)
                    for tok, cnt in c.items():
                        p[tok] = cnt / total
                    return p.clamp(min=1e-10).log()
        return None

    def interpolate(self, neural_logits, context, device=torch.device("cpu")):
        ngram_lp = self.get_ngram_log_probs(context)
        if ngram_lp is None:
            return neural_logits
        neural_p = torch.softmax(neural_logits.float(), dim=-1)
        H = -(neural_p * (neural_p + 1e-10).log()).sum()
        max_H = math.log(self.vocab_size)
        alpha = float((H / max_H).clamp(0.0, 0.95))
        ngram_p = ngram_lp.exp().to(device=device)
        mixed_p = (1.0 - alpha) * neural_p + alpha * ngram_p
        return mixed_p.clamp(min=1e-10).log()


class TestNGramCache:
    def test_empty_cache_returns_none(self):
        cache = NGramCache(order=3, vocab_size=10)
        result = cache.get_ngram_log_probs([1, 2, 3])
        assert result is None, "Empty cache should return None"

    def test_update_then_retrieve(self):
        cache = NGramCache(order=2, vocab_size=10)
        cache.update([1, 2], next_token=5)
        lp = cache.get_ngram_log_probs([1, 2])
        assert lp is not None
        assert lp[5] == lp.max(), "Updated token should have highest log-prob"

    def test_backoff_to_lower_order(self):
        cache = NGramCache(order=3, vocab_size=10)
        cache.update([7, 8], next_token=3)
        lp = cache.get_ngram_log_probs([99, 7, 8])
        assert lp is not None, "Should backoff to bigram match"
        assert lp[3] == lp.max(), "Bigram prediction should dominate"

    def test_update_is_backward_looking(self):
        """update() must be called AFTER scoring to be legal."""
        cache = NGramCache(order=2, vocab_size=10)
        ctx = [1, 2]
        next_tok = 5
        assert cache.get_ngram_log_probs(ctx) is None
        cache.update(ctx, next_tok)
        assert cache.get_ngram_log_probs(ctx) is not None

    def test_log_probs_sum_to_one(self):
        cache = NGramCache(order=2, vocab_size=5)
        for tok in [0, 1, 2, 3]:
            cache.update([9], next_token=tok)
        lp = cache.get_ngram_log_probs([9])
        probs = lp.exp()
        assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilities must sum to 1"

    def test_interpolate_returns_log_probs(self):
        cache = NGramCache(order=2, vocab_size=10)
        cache.update([1, 2], next_token=3)
        neural = torch.randn(10)
        result = cache.interpolate(neural, [1, 2])
        assert (result <= 0).all(), "Log-probs must be <= 0"
        probs = result.exp()
        assert abs(probs.sum().item() - 1.0) < 1e-4, "Interpolated probs must sum to 1"

    def test_interpolate_without_cache_returns_neural(self):
        cache = NGramCache(order=2, vocab_size=10)
        neural = torch.randn(10)
        result = cache.interpolate(neural, [99, 88])
        assert torch.allclose(result, neural), "No cache hit should return neural logits unchanged"

    def test_reset_clears_all_counts(self):
        cache = NGramCache(order=3, vocab_size=10)
        cache.update([1, 2, 3], next_token=5)
        cache.reset()
        assert cache.get_ngram_log_probs([1, 2, 3]) is None, "After reset, cache must be empty"

    def test_high_entropy_gives_higher_alpha(self):
        """High neural entropy → trust N-gram more (higher alpha)."""
        cache = NGramCache(order=2, vocab_size=100)
        for t in range(100):
            cache.update([0], next_token=t)

        neural_uniform = torch.ones(100)
        neural_peaked = torch.zeros(100)
        neural_peaked[0] = 10.0

        result_uniform = cache.interpolate(neural_uniform, [0])
        result_peaked = cache.interpolate(neural_peaked, [0])
        peaked_probs = result_peaked.exp()
        uniform_probs = result_uniform.exp()
        assert peaked_probs[0] > uniform_probs[0], \
            "High-confidence neural should dominate over n-gram more than low-confidence"
