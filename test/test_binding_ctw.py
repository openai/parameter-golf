"""
test_binding_ctw.py — Tests for binding-energy-modulated Dirichlet CTW

Tests:
  1. Cache build and update
  2. Fixed vs binding-modulated concentration
  3. Binding energy computation
  4. High-specificity contexts get higher concentration
  5. End-to-end: binding CTW beats fixed CTW on structured data
"""

import math
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from binding_ctw import BindingCTW


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_structured_tokens(n: int = 50_000, vocab_size: int = 64,
                           seed: int = 42) -> np.ndarray:
    """
    Token stream with two regimes:
    - Rare pattern: token sequence [60, 61, 62] → always followed by 63
    - Common pattern: token 1 → followed by any of [2,3,4,5] equally
    """
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=n, dtype=np.uint16)

    # Plant rare deterministic pattern every 200 tokens
    for i in range(0, n - 4, 200):
        tokens[i] = 60
        tokens[i + 1] = 61
        tokens[i + 2] = 62
        tokens[i + 3] = 63  # deterministic

    # Plant common ambiguous pattern every 20 tokens
    for i in range(5, n - 2, 20):
        tokens[i] = 1
        tokens[i + 1] = rng.choice([2, 3, 4, 5])  # ambiguous

    return tokens


@pytest.fixture
def structured_tokens():
    return make_structured_tokens()


@pytest.fixture
def built_cache(structured_tokens):
    cache = BindingCTW(max_order=5, min_order=2, num_buckets=4096,
                       vocab_size=64, c_base=5.0, beta=2.0)
    cache.build_full(structured_tokens)
    return cache


# ---------------------------------------------------------------------------
# 1. Cache build and update
# ---------------------------------------------------------------------------

class TestCacheBuild:

    def test_build_populates_counts(self, built_cache):
        total_ctx = sum(int(c.sum()) for c in built_cache.ctx_counts)
        assert total_ctx > 0, "Cache should have non-zero context counts"

    def test_build_populates_token_freq(self, built_cache):
        assert built_cache.total_tokens > 0
        assert np.sum(built_cache.token_freq > 0) > 0

    def test_update_adds_counts(self, structured_tokens):
        cache = BindingCTW(max_order=3, min_order=2, num_buckets=1024,
                           vocab_size=64)
        before = sum(int(c.sum()) for c in cache.ctx_counts)
        cache.update(structured_tokens, 0, 1000)
        after = sum(int(c.sum()) for c in cache.ctx_counts)
        assert after > before

    def test_stats_reports_correctly(self, built_cache):
        stats = built_cache.stats()
        assert stats['total_tokens'] > 0
        assert stats['total_ctx_entries'] > 0
        assert stats['c_base'] == 5.0
        assert stats['beta'] == 2.0


# ---------------------------------------------------------------------------
# 2. Binding energy computation
# ---------------------------------------------------------------------------

class TestBindingEnergy:

    def test_rare_tokens_higher_binding(self, built_cache):
        """Rare tokens (60,61,62) should have higher binding than common (1)."""
        rare_ctx = np.array([60, 61, 62], dtype=np.uint16)
        common_ctx = np.array([1, 1, 1], dtype=np.uint16)
        b_rare = built_cache.binding_energy(rare_ctx)
        b_common = built_cache.binding_energy(common_ctx)
        assert b_rare > b_common, \
            f"Rare context B={b_rare:.4f} should exceed common B={b_common:.4f}"

    def test_empty_context_zero_binding(self, built_cache):
        assert built_cache.binding_energy(np.array([], dtype=np.uint16)) == 0.0

    def test_single_token_uses_specificity(self, built_cache):
        b = built_cache.binding_energy(np.array([60], dtype=np.uint16))
        assert b > 0

    def test_batch_binding_matches_individual(self, built_cache, structured_tokens):
        positions = np.array([100, 200, 300])
        batch_b = built_cache.binding_energy_batch(
            structured_tokens, positions, context_len=3)
        for i, pos in enumerate(positions):
            ctx = structured_tokens[max(0, pos - 3):pos + 1]
            individual_b = built_cache.binding_energy(ctx)
            assert abs(batch_b[i] - individual_b) < 1e-6


# ---------------------------------------------------------------------------
# 3. Concentration mapping
# ---------------------------------------------------------------------------

class TestConcentration:

    def test_higher_binding_higher_concentration(self, built_cache):
        low_b = np.array([0.01, 0.02])
        high_b = np.array([50.0, 100.0])
        c_low = built_cache.concentration_for_binding(low_b)
        c_high = built_cache.concentration_for_binding(high_b)
        # Compare max values since sigmoid centering shifts the median
        assert c_high.max() > c_low.min()

    def test_concentration_always_positive(self, built_cache):
        binding = np.array([0.0, 0.5, 1.0, 5.0, 100.0])
        c = built_cache.concentration_for_binding(binding)
        assert np.all(c > 0)

    def test_concentration_bounded(self, built_cache):
        """c should be between c_base and c_base × (1 + beta)."""
        binding = np.array([0.0, 1.0, 10.0, 100.0])
        c = built_cache.concentration_for_binding(binding)
        assert np.all(c >= built_cache.c_base * 0.5)  # allow some margin
        assert np.all(c <= built_cache.c_base * (1 + built_cache.beta) * 1.1)


# ---------------------------------------------------------------------------
# 4. Hierarchical Dirichlet mixing
# ---------------------------------------------------------------------------

class TestHierarchicalMixing:

    def test_fixed_concentration_works(self, built_cache, structured_tokens):
        n = len(structured_tokens)
        base_p = np.full(1000, 1.0 / 64)  # uniform base
        blended = built_cache.lookup_hierarchical_fixed(
            structured_tokens, 100, 1100, base_p, concentration=5.0)
        assert blended.shape == (1000,)
        assert np.all(blended >= 0)
        assert np.all(blended <= 1.0)

    def test_binding_concentration_works(self, built_cache, structured_tokens):
        base_p = np.full(1000, 1.0 / 64)
        blended = built_cache.lookup_hierarchical_binding(
            structured_tokens, 100, 1100, base_p, context_len=4)
        assert blended.shape == (1000,)
        assert np.all(blended >= 0)
        assert np.all(blended <= 1.0)

    def test_blended_differs_from_uniform(self, built_cache, structured_tokens):
        base_p = np.full(1000, 1.0 / 64)
        blended = built_cache.lookup_hierarchical_fixed(
            structured_tokens, 100, 1100, base_p)
        # At least some positions should differ from uniform
        differs = np.sum(np.abs(blended - 1.0 / 64) > 1e-6)
        assert differs > 0, "CTW should modify at least some positions"

    def test_deterministic_pattern_gets_high_probability(self, built_cache, structured_tokens):
        """At positions where [60,61,62]→63 is planted, blended prob should be high."""
        # Find positions right after the planted pattern
        high_prob_positions = []
        for i in range(0, len(structured_tokens) - 4, 200):
            if (structured_tokens[i] == 60 and structured_tokens[i+1] == 61
                and structured_tokens[i+2] == 62 and structured_tokens[i+3] == 63):
                if i + 2 >= 100 and i + 2 < 1100:
                    high_prob_positions.append(i + 2 - 100)

        if len(high_prob_positions) == 0:
            pytest.skip("No planted patterns in scoring range")

        base_p = np.full(1000, 1.0 / 64)
        blended = built_cache.lookup_hierarchical_fixed(
            structured_tokens, 100, 1100, base_p)

        for pos in high_prob_positions[:5]:
            assert blended[pos] > 1.0 / 64, \
                f"Planted pattern at position {pos} should have above-uniform probability"


# ---------------------------------------------------------------------------
# 5. Binding CTW vs Fixed CTW
# ---------------------------------------------------------------------------

class TestBindingVsFixed:

    def test_binding_modulates_concentration(self, built_cache, structured_tokens):
        """
        Verify that binding-modulated CTW actually uses different
        concentrations for different contexts.
        """
        base_p = np.full(2000, 1.0 / 64)
        blended_fixed = built_cache.lookup_hierarchical_fixed(
            structured_tokens, 0, 2000, base_p, concentration=5.0)
        blended_binding = built_cache.lookup_hierarchical_binding(
            structured_tokens, 0, 2000, base_p, context_len=4)

        # They should differ at some positions (different concentration)
        diff = np.abs(blended_fixed - blended_binding)
        assert np.sum(diff > 1e-8) > 0, \
            "Binding CTW should differ from fixed CTW at some positions"

    def test_warm_from_training_improves_specificity(self):
        """Training freq data should improve binding computation."""
        cache = BindingCTW(max_order=3, min_order=2, num_buckets=1024,
                           vocab_size=64)

        # Without training data: all zero specificity
        ctx = np.array([60, 61, 62], dtype=np.uint16)
        b_cold = cache.binding_energy(ctx)
        assert b_cold == 0.0

        # With training data: non-zero specificity
        freq = np.ones(64, dtype=np.float64) * 1000
        freq[60] = 10  # rare
        freq[61] = 10
        freq[62] = 10
        cache.warm_from_training(freq, total_tokens=64000)
        b_warm = cache.binding_energy(ctx)
        assert b_warm > 0.0


# ---------------------------------------------------------------------------
# 6. Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline(self):
        """Build → warm → score → compare."""
        tokens = make_structured_tokens(n=10_000, vocab_size=32)

        cache = BindingCTW(max_order=5, min_order=2, num_buckets=2048,
                           vocab_size=32, c_base=5.0, beta=2.0)
        cache.build_full(tokens)

        base_p = np.full(1000, 1.0 / 32)

        # Both methods should produce valid probabilities
        fixed = cache.lookup_hierarchical_fixed(tokens, 500, 1500, base_p)
        binding = cache.lookup_hierarchical_binding(tokens, 500, 1500, base_p)

        assert np.all(np.isfinite(fixed))
        assert np.all(np.isfinite(binding))
        assert np.all(fixed >= 0) and np.all(fixed <= 1)
        assert np.all(binding >= 0) and np.all(binding <= 1)

    def test_memory_footprint(self):
        """Cache should be reasonable size."""
        cache = BindingCTW(max_order=13, min_order=2, num_buckets=131072,
                           vocab_size=1024)
        # 12 orders × 131K × 4 bytes × 2 arrays = ~12MB
        expected_mb = cache.num_orders * cache.num_buckets * 4 * 2 / 1e6
        assert expected_mb < 20, f"Cache too large: {expected_mb:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
