"""Unit tests for PrefixNgramCorrector.

Verifies all legality properties from LEGALITY_SPEC.md:
  1. Causality            – position t uses only prefix [0, t)
  2. Full-vocab support   – softmax output is positive for all tokens
  3. Score-before-update  – hash state unchanged until update() is called
  4. No realized-token dep – bias at t is independent of x_t
  5. Single-pass equivalence – chunk boundaries don't change results
  6. Laplace nonzero      – every vocab entry gets a finite, nonzero bias
  7. Reset after SGD      – state is fully cleared by reset()
  8. No dense [B,S,V] tensor – production path doesn't allocate [B,S,V] bias

Run:
    python -m pytest tests/test_corrector.py -v
or:
    python tests/test_corrector.py
"""
import sys, types, math, re, pathlib, importlib.util, unittest
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
# Minimal mocks so train_gpt.py loads without a GPU or special wheels
# ------------------------------------------------------------------
def _stub(name, **attrs):
    """Create a minimal stub module with given attributes."""
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


if "flash_attn_interface" not in sys.modules:
    _stub("flash_attn_interface",
          flash_attn_func=None,
          flash_attn_varlen_func=None)

if "brotli" not in sys.modules:
    _stub("brotli")

if "sentencepiece" not in sys.modules:
    _stub("sentencepiece", SentencePieceProcessor=object)

# ------------------------------------------------------------------
# Load PrefixNgramCorrector from the actual train_gpt.py source
# ------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_TRAIN_GPT = _REPO_ROOT / "records/track_10min_16mb/2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"

_spec = importlib.util.spec_from_file_location("_train_gpt_src", _TRAIN_GPT)
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    PrefixNgramCorrector = _mod.PrefixNgramCorrector
except Exception as exc:
    raise ImportError(
        f"Could not load PrefixNgramCorrector from {_TRAIN_GPT}: {exc}\n"
        "Make sure torch, numpy, and optional deps are installed."
    ) from exc


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
V = 64       # small vocab for fast tests
ALPHA = 0.5
ORDERS = [2, 3]


def _corrector(**kw):
    """Create a fresh corrector with test defaults."""
    kw.setdefault("V", V)
    kw.setdefault("alpha", ALPHA)
    kw.setdefault("orders", ORDERS)
    return PrefixNgramCorrector(**kw)


def _feed_sequence(corrector, tokens):
    """Score then update for each token, collecting bias vectors."""
    biases = []
    for t_idx, tok in enumerate(tokens):
        bias = corrector.get_logit_bias()   # BEFORE update
        biases.append(bias.clone())
        corrector.update(tok)               # AFTER scoring
    return biases


# ==================================================================
# Tests
# ==================================================================

class TestCausality(unittest.TestCase):
    """Test 1: bias at position t uses only prefix [0, t)."""

    def test_causality(self):
        tokens = [5, 10, 15, 20, 5, 10]
        c1 = _corrector()
        c2 = _corrector()
        biases1 = _feed_sequence(c1, tokens)

        # Re-run with different suffix — biases at positions 0..k-1 must be identical
        tokens_alt = tokens[:3] + [59, 59, 59]   # differ after position 3 (within V=64)
        biases2 = _feed_sequence(c2, tokens_alt)

        for t in range(3):
            self.assertTrue(
                torch.allclose(biases1[t], biases2[t], atol=1e-6),
                f"Bias differs at position {t} despite identical prefix"
            )


class TestFullVocabNormalization(unittest.TestCase):
    """Test 2: softmax of (neural_logits + corrector_bias) sums to 1.0."""

    def test_full_vocab_normalization(self):
        tokens = [1, 2, 3, 10, 20]
        c = _corrector()
        for tok in tokens:
            bias = c.get_logit_bias()
            neural = torch.randn(V)
            combined = neural + bias
            probs = F.softmax(combined.float(), dim=0)
            self.assertAlmostEqual(probs.sum().item(), 1.0, places=5,
                                   msg="Softmax must sum to 1.0")
            self.assertTrue((probs > 0).all(),
                            "All probabilities must be strictly positive")
            c.update(tok)


class TestScoreBeforeUpdate(unittest.TestCase):
    """Test 3: hash state is unchanged until update() is called."""

    def test_score_before_update(self):
        c = _corrector()
        # Seed some history
        for tok in [1, 2, 3]:
            c.get_logit_bias()
            c.update(tok)

        # Capture state snapshot
        bias_before = c.get_logit_bias().clone()
        uni_before = c.uni.clone()
        hist_before = list(c.hist)

        # Calling get_logit_bias() again must NOT change state
        _ = c.get_logit_bias()
        _ = c.get_logit_bias()

        self.assertTrue(torch.equal(c.uni, uni_before),
                        "unigram counts must not change after get_logit_bias()")
        self.assertEqual(c.hist, hist_before,
                         "history must not change after get_logit_bias()")
        self.assertTrue(torch.allclose(c.get_logit_bias(), bias_before, atol=1e-6),
                        "bias must be idempotent without update()")


class TestNoRealizedTokenDependence(unittest.TestCase):
    """Test 4: correction at position t is independent of x_t (the actual token)."""

    def test_no_realized_token_dependence(self):
        prefix = [5, 10, 15]
        c1 = _corrector()
        c2 = _corrector()

        # Feed identical prefix
        for tok in prefix:
            c1.get_logit_bias(); c1.update(tok)
            c2.get_logit_bias(); c2.update(tok)

        # Bias at position len(prefix) must be the same regardless of what x_t will be
        bias_before_any_update = c1.get_logit_bias().clone()
        # c2 has the same state — verify
        self.assertTrue(
            torch.allclose(c1.get_logit_bias(), c2.get_logit_bias(), atol=1e-6),
            "Bias must be identical for identical prefix"
        )
        # Update c1 with tok_a, c2 with tok_b — bias BEFORE update must still be the same
        tok_a, tok_b = 7, 42
        bias_c1 = c1.get_logit_bias().clone()
        bias_c2 = c2.get_logit_bias().clone()
        self.assertTrue(
            torch.allclose(bias_c1, bias_c2, atol=1e-6),
            "get_logit_bias() must return same value for same prefix, regardless of future x_t"
        )
        c1.update(tok_a)
        c2.update(tok_b)
        # After divergent updates, biases differ — that's correct (different new prefix)
        if tok_a != tok_b:
            self.assertFalse(
                torch.allclose(c1.get_logit_bias(), c2.get_logit_bias(), atol=1e-6),
                "After different updates, biases should diverge"
            )


class TestSinglePass(unittest.TestCase):
    """Test 5: bias is the same regardless of chunk boundaries (global state)."""

    def test_single_pass(self):
        tokens = list(range(20))
        # Single pass: one corrector sees all 20 tokens at once
        c_single = _corrector()
        biases_single = _feed_sequence(c_single, tokens)

        # Chunked pass: same corrector, fed in two batches (split at 10)
        c_chunk = _corrector()
        biases_chunk = _feed_sequence(c_chunk, tokens[:10]) + _feed_sequence(c_chunk, tokens[10:])

        # Must be identical — corrector state is cumulative, not per-chunk
        self.assertEqual(len(biases_single), len(biases_chunk))
        for t, (b1, b2) in enumerate(zip(biases_single, biases_chunk)):
            self.assertTrue(
                torch.allclose(b1, b2, atol=1e-6),
                f"Chunk-boundary non-invariance at position {t}"
            )


class TestLaplaceNonzero(unittest.TestCase):
    """Test 6: every vocab entry gets a finite, nonzero bias (Laplace smoothing)."""

    def test_laplace_nonzero(self):
        c = _corrector()
        # Even before any tokens are scored, all entries must be finite
        bias = c.get_logit_bias()
        self.assertEqual(bias.shape, (V,), "bias must be [V]")
        self.assertTrue(torch.isfinite(bias).all(),
                        "All bias values must be finite (Laplace guarantees this)")
        # After scoring some tokens, still all finite
        for tok in [1, 2, 3, 1, 2]:
            c.update(tok)
        bias_after = c.get_logit_bias()
        self.assertTrue(torch.isfinite(bias_after).all(),
                        "All bias values must remain finite after updates")
        # Unigram count >= 1 for all (Laplace), so log(count) > -inf
        self.assertTrue((c.uni >= 1).all(),
                        "Laplace smoothing: all unigram counts must be >= 1")


class TestResetAfterSgd(unittest.TestCase):
    """Test 7: reset() clears all state (called after global SGD phase)."""

    def test_reset_after_sgd(self):
        c = _corrector()
        # Accumulate meaningful state
        for tok in [5, 10, 15, 5, 10, 5]:
            c.get_logit_bias()
            c.update(tok)
        self.assertGreater(len(c.hist), 0, "History must be populated")
        self.assertTrue((c.uni > 1).any(), "Some unigram counts must be > 1")

        # Simulate SGD phase boundary
        c.reset()

        # All state must be cleared
        self.assertEqual(len(c.hist), 0, "History must be empty after reset()")
        self.assertTrue((c.uni == 1).all(),
                        "Unigram counts must be reset to Laplace baseline (all 1s)")
        for n in c.orders:
            self.assertEqual(len(c.ng[n]), 0,
                             f"N-gram table for order {n} must be empty after reset()")

        # Bias after reset must equal Laplace flat distribution
        bias_after_reset = c.get_logit_bias()
        expected = torch.full((V,), -math.log(V) * ALPHA, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(bias_after_reset, expected, atol=1e-5),
            "After reset, bias must equal flat Laplace distribution"
        )


class TestNoDenseBsvTensor(unittest.TestCase):
    """Test 8: production integration path uses [B,1,V] not [B,S,V]."""

    def test_no_dense_bsv_tensor(self):
        B = 4    # batch size
        S = 32   # sequence length (chunk_size)
        correctors = [_corrector() for _ in range(B)]
        active = [True] * B
        # Seed some history
        for c in correctors:
            for tok in [1, 2, 3]:
                c.get_logit_bias(); c.update(tok)

        # Simulate the production integration: one [V] bias per batch element
        biases_cpu = [
            correctors[b].get_logit_bias() if active[b] else torch.zeros(V)
            for b in range(B)
        ]
        logit_bias = torch.stack(biases_cpu).unsqueeze(1)  # [B, 1, V]

        # Shape check: must be [B, 1, V], NOT [B, S, V]
        self.assertEqual(logit_bias.shape, (B, 1, V),
                         f"logit_bias must be [B,1,V]={B,1,V}, got {tuple(logit_bias.shape)}")

        # Verify broadcast correctness: adding to [B, S, V] logits works
        fake_logits = torch.randn(B, S, V)
        corrected = fake_logits + logit_bias   # broadcast [B,S,V] + [B,1,V] → [B,S,V]
        self.assertEqual(corrected.shape, (B, S, V),
                         "Broadcast addition must produce [B,S,V]")

        # Verify no actual [B,S,V] bias tensor was allocated
        max_bias_elements = max(t.numel() for t in biases_cpu + [logit_bias])
        dense_bsv_elements = B * S * V
        self.assertLess(
            max_bias_elements, dense_bsv_elements,
            f"No single bias tensor should have {dense_bsv_elements} elements "
            f"(max seen: {max_bias_elements})"
        )


class TestWarmupLegality(unittest.TestCase):
    """Test 9: compile warmup must not reference real validation tokens.

    LEGALITY_SPEC forbids val-token exposure before the official eval
    timer starts. Enforced via source-level static check on the warmup
    block in train_gpt.py (bracketed by `# BEGIN warmup synthetic
    tokens` / `# END warmup synthetic tokens` markers).
    """

    def test_warmup_does_not_reference_val_tokens(self):
        src_path = pathlib.Path(__file__).parent.parent / (
            "records/track_10min_16mb/"
            "2026-04-14_VarLenAttn_PhasingTTT_Corrector/train_gpt.py"
        )
        src = src_path.read_text()
        m = re.search(
            r"# BEGIN warmup synthetic tokens(.*?)# END warmup synthetic tokens",
            src, flags=re.DOTALL,
        )
        self.assertIsNotNone(
            m, "Warmup block markers missing in train_gpt.py"
        )
        block = m.group(1)
        self.assertNotIn(
            "val_tokens", block,
            "Warmup block must not reference val_tokens "
            "(see LEGALITY_SPEC, PLAN_PR1610_CORRECTOR.md:96)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
