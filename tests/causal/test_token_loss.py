"""Tests for scripts/causal/token_loss_decompose.py."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.causal.token_loss_decompose import (
    build_frequency_buckets,
    classify_boundary_tokens,
    compute_category_stats,
    verify_decomposition,
)


def test_decomposition_check():
    """Mean of per-token losses must match aggregate within 1e-6."""
    rng = np.random.default_rng(42)
    per_token_losses = rng.uniform(0.5, 3.0, size=1000).astype(np.float64)
    aggregate_loss = float(np.mean(per_token_losses))

    result = verify_decomposition(per_token_losses, aggregate_loss)
    assert result["passed"] is True
    assert abs(result["delta"]) < 1e-6

    # Deliberately wrong aggregate should fail
    result_bad = verify_decomposition(per_token_losses, aggregate_loss + 1.0)
    assert result_bad["passed"] is False


def test_frequency_bucketing():
    """Known vocab frequencies produce correct bucket assignments."""
    # Simulate a vocabulary of 1024 tokens with known frequencies
    rng = np.random.default_rng(42)
    vocab_freqs = rng.zipf(1.5, size=1024).astype(np.float64)
    # Sort descending to assign ranks
    sorted_indices = np.argsort(-vocab_freqs)

    buckets = build_frequency_buckets(vocab_freqs)

    # Top-100 tokens (by frequency) should be in "top_100" bucket
    for idx in sorted_indices[:100]:
        assert buckets[idx] == "top_100"
    # 100-500 range
    for idx in sorted_indices[100:500]:
        assert buckets[idx] == "mid_100_500"
    # 500-1024 range
    for idx in sorted_indices[500:]:
        assert buckets[idx] == "tail_500_1024"


def test_bpb_contribution_summation():
    """All category contributions must sum to the total BPB."""
    rng = np.random.default_rng(42)
    n_tokens = 2000
    per_token_losses = rng.uniform(0.5, 3.0, size=n_tokens).astype(np.float64)
    token_ids = rng.integers(0, 1024, size=n_tokens)

    vocab_freqs = rng.zipf(1.5, size=1024).astype(np.float64)
    buckets = build_frequency_buckets(vocab_freqs)

    stats = compute_category_stats(per_token_losses, token_ids, buckets)

    total_contribution = sum(s["bpb_contribution"] for s in stats.values())
    total_mean = float(np.mean(per_token_losses))
    # Contributions should sum to total mean loss (within floating point)
    assert abs(total_contribution - total_mean) < 1e-6


def test_boundary_classification():
    """Boundary tokens (first after whitespace) classified correctly."""
    # Token IDs: 0=space, 1='hello', 2='world', 3=' the'
    # whitespace_tokens = {0, 3}
    # Sequence: [1, 0, 2, 3, 1]
    #
    # Index 0: token_id=1 ('hello'), first token => boundary
    # Index 1: token_id=0 ('space'), prev=1 ('hello', not WS) => mid-sequence
    # Index 2: token_id=2 ('world'), prev=0 ('space', WS) => boundary
    # Index 3: token_id=3 (' the'), prev=2 ('world', not WS) => mid-sequence
    # Index 4: token_id=1 ('hello'), prev=3 (' the', WS) => boundary
    token_ids = np.array([1, 0, 2, 3, 1])
    is_boundary = classify_boundary_tokens(token_ids, whitespace_tokens={0, 3})

    assert is_boundary[0] is True   # first token
    assert is_boundary[1] is False  # prev=1 (not whitespace)
    assert is_boundary[2] is True   # prev=0 (whitespace)
    assert is_boundary[3] is False  # prev=2 (not whitespace)
    assert is_boundary[4] is True   # prev=3 (whitespace)
