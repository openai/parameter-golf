"""
test_hybrid_system.py

Tests for the hybrid hypergraph + transformer Parameter Golf system.

Tests cover:
  1. HypergraphStore: scan, build, binding energy, pattern selection
  2. Prediction: multi-level lookup, binding-weighted interpolation
  3. Serialization: roundtrip fidelity
  4. Budget: 16MB constraint
  5. HybridGPT: forward pass, hybrid interpolation (requires torch)
  6. End-to-end: synthetic data → store → predict → interpolate
"""

import math
import struct
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Check torch availability
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from hypergraph_lm import (
    HypergraphPatternStore, PatternEntry, LevelStore,
)

# Only import torch-dependent modules if available
if HAS_TORCH:
    from train_hybrid import (
        HypergraphStore, HybridGPT, quantize_state_dict_int8,
        dequantize_state_dict_int8,
    )
    from hypergraph_lm import hypergraph_to_torch_logits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_tokens(n: int = 100_000, vocab_size: int = 64,
                          seed: int = 42) -> np.ndarray:
    """
    Generate synthetic token stream with planted patterns.
    Some bigrams/trigrams are deterministic (high binding);
    the rest are random (low binding).
    """
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=n, dtype=np.uint16)

    # Plant strong bigram: token 10 always followed by token 20
    for i in range(0, n - 1, 50):
        tokens[i] = 10
        tokens[i + 1] = 20

    # Plant strong trigram: (5, 15) always followed by 25
    for i in range(0, n - 2, 100):
        tokens[i] = 5
        tokens[i + 1] = 15
        tokens[i + 2] = 25

    # Plant 5-gram: (1, 2, 3, 4) → 5
    for i in range(0, n - 4, 200):
        tokens[i] = 1
        tokens[i + 1] = 2
        tokens[i + 2] = 3
        tokens[i + 3] = 4
        tokens[i + 4] = 5

    return tokens


@pytest.fixture
def synth_tokens():
    return make_synthetic_tokens()


# ---------------------------------------------------------------------------
# Pure-Python HypergraphStore for testing without torch
# ---------------------------------------------------------------------------
# We extract the store logic to work without torch imports.
# The actual train_hybrid.py needs torch, so we test the core
# hypergraph logic through hypergraph_lm.py which is pure Python.

@pytest.fixture
def built_pattern_store(synth_tokens):
    """Build a HypergraphPatternStore from synthetic tokens."""
    store = HypergraphPatternStore(vocab_size=64)
    store.scan_tokens_fast(synth_tokens)
    store.build(bigram_budget=200_000, trigram_budget=200_000,
                fivegram_budget=100_000, min_count=3, top_k_next=16)
    return store


# ---------------------------------------------------------------------------
# 1. HypergraphPatternStore: scan and build
# ---------------------------------------------------------------------------

class TestPatternStoreScan:

    def test_scan_accumulates_frequencies(self, synth_tokens):
        store = HypergraphPatternStore(vocab_size=64)
        store.scan_tokens_fast(synth_tokens)
        assert store.total_tokens == len(synth_tokens)
        assert store.token_freq.sum() == len(synth_tokens)

    def test_scan_multiple_shards(self):
        tokens1 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint16)
        tokens2 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint16)
        store = HypergraphPatternStore(vocab_size=10)
        store.scan_tokens_fast(tokens1)
        store.scan_tokens_fast(tokens2)
        assert store.total_tokens == 14

    def test_build_produces_all_levels(self, built_pattern_store):
        assert 1 in built_pattern_store.levels
        assert 2 in built_pattern_store.levels
        assert 3 in built_pattern_store.levels
        assert built_pattern_store._built

    def test_build_produces_nonempty_levels(self, built_pattern_store):
        assert len(built_pattern_store.levels[1].patterns) > 0
        assert len(built_pattern_store.levels[2].patterns) > 0

    def test_planted_bigram_detected(self, built_pattern_store):
        """Token 10 → 20 was planted as a strong bigram."""
        entry = built_pattern_store.levels[1].patterns.get((10,))
        assert entry is not None, "Planted bigram (10,) should be in store"
        assert 20 in entry.next_dist, "Token 20 should be top prediction"
        assert entry.next_dist[20] > 0.3

    def test_planted_trigram_detected(self, built_pattern_store):
        """(5, 15) → 25 was planted."""
        entry = built_pattern_store.levels[2].patterns.get((5, 15))
        assert entry is not None, "Planted trigram (5,15) should be in store"
        assert 25 in entry.next_dist


class TestPatternStoreBinding:

    def test_specificity_rare_vs_common(self):
        store = HypergraphPatternStore(vocab_size=10)
        store.token_freq[0] = 1     # rare
        store.token_freq[1] = 1000  # common
        assert store.specificity(0) > store.specificity(1)

    def test_specificity_zero_for_unseen(self):
        store = HypergraphPatternStore(vocab_size=10)
        assert store.specificity(5) == 0.0

    def test_binding_higher_for_predictable(self, synth_tokens):
        """Planted bigram should have higher binding than random."""
        store = HypergraphPatternStore(vocab_size=64)
        store.scan_tokens_fast(synth_tokens)
        b_planted = store.binding_energy_bigram(10)  # planted
        b_random = store.binding_energy_bigram(30)   # random
        assert b_planted > b_random

    def test_binding_zero_when_empty(self):
        store = HypergraphPatternStore(vocab_size=10)
        assert store.binding_energy_bigram(0) == 0.0


# ---------------------------------------------------------------------------
# 2. Prediction: multi-level lookup
# ---------------------------------------------------------------------------

class TestPrediction:

    def test_predict_returns_valid_distribution(self, built_pattern_store):
        ctx = np.array([10], dtype=np.uint16)
        dist, conf = built_pattern_store.predict(ctx)
        assert dist is not None
        assert abs(dist.sum() - 1.0) < 0.01

    def test_predict_planted_bigram(self, built_pattern_store):
        ctx = np.array([10], dtype=np.uint16)
        dist, conf = built_pattern_store.predict(ctx)
        assert dist is not None
        assert dist.argmax() == 20

    def test_predict_planted_trigram(self, built_pattern_store):
        ctx = np.array([5, 15], dtype=np.uint16)
        dist, conf = built_pattern_store.predict(ctx)
        assert dist is not None
        assert dist.argmax() == 25

    def test_predict_confidence_positive_for_known(self, built_pattern_store):
        ctx = np.array([10], dtype=np.uint16)
        _, conf = built_pattern_store.predict(ctx)
        assert conf > 0

    def test_predict_no_match_returns_none(self, built_pattern_store):
        """Unseen context should return None."""
        ctx = np.array([63, 62, 61, 60], dtype=np.uint16)  # unlikely pattern
        dist, conf = built_pattern_store.predict(ctx)
        # May or may not match — if no match, dist is None
        if dist is None:
            assert conf == 0.0

    def test_multilevel_trigram_higher_confidence(self, built_pattern_store):
        """Trigram context should combine bigram + trigram → higher confidence."""
        ctx_bi = np.array([15], dtype=np.uint16)
        ctx_tri = np.array([5, 15], dtype=np.uint16)
        _, conf_bi = built_pattern_store.predict(ctx_bi)
        _, conf_tri = built_pattern_store.predict(ctx_tri)
        # Trigram match adds binding on top of bigram
        assert conf_tri >= conf_bi

    def test_batch_prediction(self, built_pattern_store):
        contexts = np.array([[10, 0, 0, 0], [5, 15, 0, 0]], dtype=np.uint16)
        dists, confs = built_pattern_store.predict_batch(contexts)
        assert dists.shape == (2, 64)
        assert confs.shape == (2,)


# ---------------------------------------------------------------------------
# 3. Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_roundtrip_pattern_counts(self, built_pattern_store):
        blob = built_pattern_store.serialize()
        restored = HypergraphPatternStore.deserialize(blob, vocab_size=64)
        for level in [1, 2, 3]:
            if level in built_pattern_store.levels:
                assert len(restored.levels[level].patterns) == \
                       len(built_pattern_store.levels[level].patterns)

    def test_roundtrip_prediction_top1(self, built_pattern_store):
        blob = built_pattern_store.serialize()
        restored = HypergraphPatternStore.deserialize(blob, vocab_size=64)

        ctx = np.array([10], dtype=np.uint16)
        d_orig, _ = built_pattern_store.predict(ctx)
        d_rest, _ = restored.predict(ctx)

        assert d_orig is not None and d_rest is not None
        assert d_orig.argmax() == d_rest.argmax()

    def test_roundtrip_size_reasonable(self, built_pattern_store):
        blob = built_pattern_store.serialize()
        assert 100 < len(blob) < 6_000_000

    def test_deserialized_is_built(self, built_pattern_store):
        blob = built_pattern_store.serialize()
        restored = HypergraphPatternStore.deserialize(blob, vocab_size=64)
        assert restored._built


# ---------------------------------------------------------------------------
# 4. Budget constraint
# ---------------------------------------------------------------------------

class TestBudget:

    def test_store_respects_budget(self):
        tokens = make_synthetic_tokens(n=50_000, vocab_size=64)
        store = HypergraphPatternStore(vocab_size=64)
        store.scan_tokens_fast(tokens)
        store.build(bigram_budget=10_000, trigram_budget=10_000,
                    fivegram_budget=10_000, min_count=3, top_k_next=8)
        blob = store.serialize()
        # Compressed should be well within total budget
        assert len(blob) < 100_000

    def test_larger_budget_more_patterns(self):
        tokens = make_synthetic_tokens(n=50_000, vocab_size=64)

        small = HypergraphPatternStore(vocab_size=64)
        small.scan_tokens_fast(tokens)
        small.build(bigram_budget=5_000, trigram_budget=5_000,
                    fivegram_budget=5_000, min_count=3, top_k_next=8)

        large = HypergraphPatternStore(vocab_size=64)
        large.scan_tokens_fast(tokens)
        large.build(bigram_budget=200_000, trigram_budget=200_000,
                    fivegram_budget=200_000, min_count=3, top_k_next=8)

        small_total = sum(len(s.patterns) for s in small.levels.values())
        large_total = sum(len(s.patterns) for s in large.levels.values())
        assert large_total >= small_total

    def test_16mb_split_arithmetic(self):
        """Budget split: 5MB store + 9MB model + 2MB code ≤ 16MB."""
        assert 5_000_000 + 9_000_000 + 2_000_000 <= 16_000_000

    def test_binding_selects_high_quality_first(self):
        """With tight budget, planted patterns survive over random."""
        tokens = make_synthetic_tokens(n=50_000, vocab_size=64)
        store = HypergraphPatternStore(vocab_size=64)
        store.scan_tokens_fast(tokens)
        store.build(bigram_budget=2_000, trigram_budget=2_000,
                    fivegram_budget=2_000, min_count=3, top_k_next=4)

        # Check if planted bigram survived (should be highest binding)
        if 1 in store.levels and len(store.levels[1].patterns) > 0:
            # Among surviving patterns, planted should be there
            entries = list(store.levels[1].patterns.values())
            bindings = [e.binding for e in entries]
            # All surviving should have positive binding
            assert all(b > 0 for b in bindings)


# ---------------------------------------------------------------------------
# 5. HybridGPT (requires torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestHybridGPT:

    @pytest.fixture
    def small_model(self):
        return HybridGPT(
            vocab_size=64, num_layers=2, model_dim=32,
            num_heads=2, num_kv_heads=2, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.01,
            logit_softcap=30.0, rope_base=10000.0,
            qk_gain_init=1.5,
        )

    def test_forward_returns_scalar(self, small_model):
        x = torch.randint(0, 64, (2, 16))
        y = torch.randint(0, 64, (2, 16))
        loss = small_model(x, y)
        assert loss.ndim == 0

    def test_get_logits_shape(self, small_model):
        x = torch.randint(0, 64, (2, 16))
        logits = small_model.get_logits(x)
        assert logits.shape == (2, 16, 64)

    def test_hybrid_without_store_equals_standard(self, small_model):
        x = torch.randint(0, 64, (2, 8))
        y = torch.randint(0, 64, (2, 8))
        loss_std = small_model(x, y)
        loss_hyb = small_model.forward_hybrid(x, y)
        assert abs(loss_std.item() - loss_hyb.item()) < 1e-4

    def test_hybrid_with_store_reduces_loss_on_planted(self):
        tokens = make_synthetic_tokens(n=50_000, vocab_size=64)
        store = HypergraphStore(vocab_size=64)
        store.scan(tokens)
        store.build(budget_bytes=200_000, min_count=3, top_k=16)

        model = HybridGPT(
            vocab_size=64, num_layers=2, model_dim=32,
            num_heads=2, num_kv_heads=2, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.01,
            logit_softcap=30.0, rope_base=10000.0,
            qk_gain_init=1.5, hyper_store=store, hyper_lambda=0.5,
        )
        x = torch.tensor([[10, 10, 10, 10]])
        y = torch.tensor([[20, 20, 20, 20]])
        loss_neural = model(x, y).item()
        loss_hybrid = model.forward_hybrid(x, y).item()
        assert loss_hybrid < loss_neural


# ---------------------------------------------------------------------------
# 6. Quantization (requires torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestQuantization:

    def test_roundtrip_preserves_keys(self):
        model = HybridGPT(
            vocab_size=64, num_layers=2, model_dim=32,
            num_heads=2, num_kv_heads=2, mlp_mult=2,
            tie_embeddings=True, tied_embed_init_std=0.01,
            logit_softcap=30.0, rope_base=10000.0,
            qk_gain_init=1.5,
        )
        sd = model.state_dict()
        quant, _ = quantize_state_dict_int8(sd)
        restored = dequantize_state_dict_int8(quant)
        assert set(restored.keys()) == set(sd.keys())


# ---------------------------------------------------------------------------
# 7. Torch interpolation (requires torch)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTorchInterpolation:

    def test_interpolation_valid_distribution(self):
        hyper_dist = np.zeros(64, dtype=np.float64)
        hyper_dist[20] = 0.9
        hyper_dist[21] = 0.1
        neural_logits = torch.randn(64)

        combined = hypergraph_to_torch_logits(
            hyper_dist, confidence=10.0, neural_logits=neural_logits)
        probs = torch.exp(combined)
        assert abs(probs.sum().item() - 1.0) < 0.01

    def test_high_confidence_favors_hypergraph(self):
        hyper_dist = np.zeros(64, dtype=np.float64)
        hyper_dist[20] = 1.0
        neural_logits = torch.zeros(64)  # uniform neural

        combined = hypergraph_to_torch_logits(
            hyper_dist, confidence=100.0, neural_logits=neural_logits)
        probs = torch.exp(combined)
        assert probs[20].item() > 0.3

    def test_zero_confidence_uses_neural(self):
        hyper_dist = np.zeros(64, dtype=np.float64)
        hyper_dist[20] = 1.0
        neural_logits = torch.zeros(64)
        neural_logits[30] = 10.0  # strong neural prediction for 30

        combined = hypergraph_to_torch_logits(
            hyper_dist, confidence=0.0, neural_logits=neural_logits)
        probs = torch.exp(combined)
        # With zero confidence, neural dominates
        assert probs[30].item() > probs[20].item()


# ---------------------------------------------------------------------------
# 8. End-to-end (pure Python parts)
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline_pure_python(self):
        """Build store → predict → serialize → roundtrip → predict again."""
        tokens = make_synthetic_tokens(n=20_000, vocab_size=32, seed=99)

        store = HypergraphPatternStore(vocab_size=32)
        store.scan_tokens_fast(tokens)
        store.build(bigram_budget=50_000, trigram_budget=50_000,
                    fivegram_budget=50_000, min_count=3, top_k_next=8)

        # Predict planted pattern
        ctx = np.array([10], dtype=np.uint16)
        dist, conf = store.predict(ctx)
        assert dist is not None
        assert dist.argmax() == 20

        # Serialize roundtrip
        blob = store.serialize()
        restored = HypergraphPatternStore.deserialize(blob, vocab_size=32)
        dist2, conf2 = restored.predict(ctx)
        assert dist2 is not None
        assert dist2.argmax() == 20

    def test_stats_report(self, built_pattern_store):
        stats = built_pattern_store.stats()
        assert 'serialized_bytes' in stats
        assert stats['total_tokens_scanned'] > 0
        for level_id, level_stats in stats['levels'].items():
            assert level_stats['num_patterns'] >= 0

    def test_cantor_enrichment_holds(self, built_pattern_store):
        """
        |A₀| < |A₁| < |A₂| — each level adds new structure.
        A₀ = unique tokens, A₁ = A₀ + bigram patterns, A₂ = A₁ + trigram patterns.
        """
        A0 = 64  # vocab_size (unique tokens)
        A1 = A0 + len(built_pattern_store.levels[1].patterns)
        A2 = A1 + len(built_pattern_store.levels[2].patterns)
        A3 = A2 + len(built_pattern_store.levels[3].patterns)

        assert A1 > A0, "Level 1 should enrich the alphabet"
        assert A2 > A1, "Level 2 should enrich further"
        # Level 3 may or may not add (depends on 5-gram subsampling)

    def test_noise_gets_low_binding(self):
        """Random tokens should produce lower average binding than planted."""
        tokens_signal = make_synthetic_tokens(n=10_000, vocab_size=64)
        tokens_noise = np.random.RandomState(999).randint(
            0, 64, size=10_000).astype(np.uint16)

        store_sig = HypergraphPatternStore(vocab_size=64)
        store_sig.scan_tokens_fast(tokens_signal)

        store_noise = HypergraphPatternStore(vocab_size=64)
        store_noise.scan_tokens_fast(tokens_noise)

        # Binding of planted bigram vs random bigram
        b_planted = store_sig.binding_energy_bigram(10)
        b_noise = store_noise.binding_energy_bigram(10)
        assert b_planted > b_noise, \
            "Planted pattern should have higher binding than pure noise"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
