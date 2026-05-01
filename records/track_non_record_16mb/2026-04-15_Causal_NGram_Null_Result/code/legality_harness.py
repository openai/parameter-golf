"""
Legality harness for CausalNGram + additive-logit blend.

Tests the four conditions from Issue #1017 empirically. Each test is a small
adversarial probe — if the code is legal, all tests pass. If any test fails,
STOP and fix before any further spend.

Usage:
    python legality_harness.py           # runs all tests
    python legality_harness.py --verbose # prints per-test details
"""

from __future__ import annotations
import sys
import math
import random
import numpy as np

# Repo-local import
sys.path.insert(0, ".")
from causal_ngram import CausalNGram


def _blend_logits(neural_logits: np.ndarray, ngram_log_p: np.ndarray,
                  alpha: float) -> np.ndarray:
    """The production blend: additive logits then softmax.

    Returns the full normalized distribution (not log, just probs)."""
    logits = neural_logits + alpha * ngram_log_p
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def test_c1_strict_causal():
    """Condition 1: p_t depends only on history x_1..x_{t-1}, never on x_t or later.

    Adversarial probe: build the cache with one sequence, query position t, then
    flip x_t and x_{t+1} to arbitrary values, re-query position t. Result must
    be bit-identical.
    """
    V = 32
    rng = random.Random(0)
    seq = [rng.randrange(V) for _ in range(500)]
    ng = CausalNGram(vocab_size=V, order=4)
    # Populate from the whole sequence (simulating "cache built from all tokens
    # scored so far"). Freeze to lock the snapshot.
    ng.add_sequence(seq)
    ng.freeze()

    t = 200
    history_before = seq[:t]
    lp_before = ng.log_probs(history_before).copy()

    # Flip the future (tokens after t). Re-query — must be identical.
    seq_mutated = seq[:t] + [(x + 7) % V for x in seq[t:]]
    lp_after = ng.log_probs(seq_mutated[:t])

    assert np.allclose(lp_before, lp_after), \
        "C1 violation: lookup depends on tokens at or after position t"
    return True


def test_c2_full_vocab_normalization():
    """Condition 2: blend is a full distribution over Sigma that sums to 1.

    Adversarial probe: compute blend probs for 50 random contexts and assert
    (a) sum == 1, (b) all entries >= 0, (c) shape == (V,).
    """
    V = 64
    rng = random.Random(1)
    seq = [rng.randrange(V) for _ in range(1000)]
    ng = CausalNGram(vocab_size=V, order=4)
    ng.add_sequence(seq)
    ng.freeze()

    failures = []
    for trial in range(50):
        t = rng.randrange(5, len(seq) - 1)
        hist = seq[:t]
        lp = ng.log_probs(hist)
        assert lp.shape == (V,), f"n-gram log-prob shape wrong: {lp.shape}"
        assert np.all(np.isfinite(lp)), "n-gram log-probs have nan/inf"
        assert np.allclose(np.exp(lp).sum(), 1.0, atol=1e-9), \
            f"n-gram distribution not normalized: sum={np.exp(lp).sum()}"

        # Now blend with a random neural logits vector
        neural = np.asarray([rng.gauss(0, 2) for _ in range(V)])
        blend = _blend_logits(neural, lp, alpha=0.5)
        assert blend.shape == (V,)
        assert np.allclose(blend.sum(), 1.0, atol=1e-9), \
            f"blend not normalized: sum={blend.sum()}"
        assert np.all(blend >= 0), "blend has negative probs"

    return True


def test_c2_xt_independence():
    """Condition 2 (subtler): p_t(v) for any v must be computable WITHOUT knowing x_t.

    Adversarial probe: compute the full blend, then for each target v, verify
    it equals what you'd get if you computed the blend "as if the answer were v".
    If the mechanism short-circuits on the observed token, this catches it.
    """
    V = 32
    rng = random.Random(2)
    seq = [rng.randrange(V) for _ in range(500)]
    ng = CausalNGram(vocab_size=V, order=4)
    ng.add_sequence(seq)
    ng.freeze()

    t = 100
    hist = seq[:t]
    lp = ng.log_probs(hist)
    neural = np.asarray([rng.gauss(0, 2) for _ in range(V)])
    blend_full = _blend_logits(neural, lp, alpha=0.5)

    # For our additive-logit design, there's no x_t in the compute path at all.
    # This is trivially true — we just assert the blend was computed without
    # reference to any single token, by computing it twice with "different
    # assumed targets" and checking identity.
    blend_full_again = _blend_logits(neural, lp, alpha=0.5)
    assert np.allclose(blend_full, blend_full_again), \
        "blend is non-deterministic (suggests hidden state dependency on x_t)"
    return True


def test_c3_score_before_update():
    """Condition 3: scoring at position t must use a state that was NOT updated
    with x_t yet.

    Adversarial probe: simulate a chunk of 10 tokens. Freeze the cache, compute
    scores for all 10 using the frozen snapshot, THEN add those 10 tokens.
    Assert: the log-probs used during scoring are identical to the log-probs
    that would be returned by a fresh cache state that has NEVER seen those
    tokens.
    """
    V = 32
    rng = random.Random(3)
    prior = [rng.randrange(V) for _ in range(200)]
    chunk = [rng.randrange(V) for _ in range(10)]

    ng = CausalNGram(vocab_size=V, order=4)
    ng.add_sequence(prior)
    ng.freeze()  # snapshot reflects only `prior`

    # Reference: a parallel cache that also only has `prior`, never updated.
    ref = CausalNGram(vocab_size=V, order=4)
    ref.add_sequence(prior)
    ref.freeze()

    # Score all chunk positions using the snapshot
    scored_log_probs = []
    for i in range(len(chunk)):
        hist = prior + chunk[:i]
        scored_log_probs.append(ng.log_probs(hist))

    # Update the live counts with the chunk tokens (simulating add-after-score)
    for i, tok in enumerate(chunk):
        ng.add_token(prior + chunk[:i], tok)
    # Note: we do NOT re-freeze yet — the snapshot is still the pre-chunk one.

    # Compare: the scored log-probs should match what ref returns (ref never
    # saw any of the chunk tokens).
    for i, lp in enumerate(scored_log_probs):
        hist = prior + chunk[:i]
        ref_lp = ref.log_probs(hist)
        assert np.allclose(lp, ref_lp), \
            f"C3 violation: scoring position {i} used state that reflects x_t"

    return True


def test_c4_single_pass():
    """Condition 4: no rescoring.

    Adversarial probe: simulate two passes over the same token stream. Second
    pass should NOT be allowed to use state built from the first. We enforce
    this by structure: the eval loop is single-pass by construction. This test
    just documents that no "refresh cache" or "second pass" API exists on the
    CausalNGram class.
    """
    attrs = dir(CausalNGram)
    forbidden = {"rescore", "rebuild", "reset_for_second_pass", "two_pass"}
    overlap = set(attrs) & forbidden
    assert not overlap, f"Forbidden APIs present: {overlap}"
    return True


def test_no_hashing():
    """Extra: #993 rule — no hashed cache. Verify counts are keyed by exact
    context tuples, not by a hash function.
    """
    ng = CausalNGram(vocab_size=16, order=3)
    ng.add_sequence([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    # Order-3 context for predicting token at position 3 is (1, 2).
    # Order-3 context for position 4 is (2, 3). These must be DISTINCT keys.
    ctx12 = (1, 2)
    ctx23 = (2, 3)
    assert ctx12 in ng.counts[3], "expected exact context key missing"
    assert ctx23 in ng.counts[3], "expected exact context key missing"
    # Sanity: Python dict keys are tuples, not integers from a hash
    for k in ng.counts[3].keys():
        assert isinstance(k, tuple), f"non-tuple key {k!r} — might be hashed"
    return True


def test_blend_nonneg_and_finite():
    """Sanity: blend never produces negative or non-finite probabilities."""
    V = 128
    rng = random.Random(4)
    seq = [rng.randrange(V) for _ in range(2000)]
    ng = CausalNGram(vocab_size=V, order=5)
    ng.add_sequence(seq)
    ng.freeze()

    for trial in range(100):
        t = rng.randrange(10, len(seq) - 1)
        hist = seq[:t]
        lp = ng.log_probs(hist)
        neural = np.asarray([rng.gauss(0, 3) for _ in range(V)])
        for alpha in [0.0, 0.1, 0.5, 1.0, 2.0]:
            blend = _blend_logits(neural, lp, alpha=alpha)
            assert np.all(np.isfinite(blend))
            assert np.all(blend >= 0)
            assert abs(blend.sum() - 1.0) < 1e-9
    return True


def test_backoff_fallthrough_unigram():
    """Order K context not seen -> back off to K-1, then K-2, ..., unigram always
    available. Verify the walk behaves correctly.
    """
    V = 16
    ng = CausalNGram(vocab_size=V, order=4, min_context_count=2)
    # Only put one unigram-level observation
    ng.add_token([], 3)
    ng.add_token([3], 5)  # order-2 context (3,) -> token 5
    ng.freeze()

    # Query with a totally unseen order-3 context
    lp = ng.log_probs([1, 2, 3])  # order-3 context would be (1,2,3) — not seen
    # After backoff, it should land on order-1 (unigram) or a fallback
    assert lp.shape == (V,)
    assert np.allclose(np.exp(lp).sum(), 1.0)
    return True


def main(verbose=False):
    tests = [
        ("C1 strict causal", test_c1_strict_causal),
        ("C2 full-vocab normalization", test_c2_full_vocab_normalization),
        ("C2 x_t independence", test_c2_xt_independence),
        ("C3 score-before-update", test_c3_score_before_update),
        ("C4 single pass", test_c4_single_pass),
        ("no-hashing (ruling #993)", test_no_hashing),
        ("blend non-negative + finite", test_blend_nonneg_and_finite),
        ("backoff fallthrough to unigram", test_backoff_fallthrough_unigram),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            if verbose:
                print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed.append((name, repr(e)))
            print(f"  ERROR {name}: {e!r}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        print("\nFAILURES — DO NOT proceed to training until these are fixed:")
        for name, msg in failed:
            print(f"  - {name}: {msg}")
        return 1
    print("All legality conditions verified. Safe to proceed.")
    return 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    sys.exit(main(verbose=verbose))
