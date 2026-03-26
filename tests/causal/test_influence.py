"""Tests for scripts/causal/influence_proxy.py."""
from __future__ import annotations

import json
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — import the module under test
# ---------------------------------------------------------------------------

from scripts.causal import influence_proxy as ip


# ---------------------------------------------------------------------------
# T17 Tests
# ---------------------------------------------------------------------------


def test_dot_product_mock():
    """Small mock gradient tensors, verify dot product computation."""
    # Simulate two flat gradient dicts with known values
    val_grad = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0])}
    shard_grad = {"a": np.array([0.5, 0.5, 0.5]), "b": np.array([1.0, 1.0])}

    # Expected: (1*0.5 + 2*0.5 + 3*0.5) + (4*1 + 5*1) = 3.0 + 9.0 = 12.0
    result = ip.compute_dot_product(val_grad, shard_grad)
    assert abs(result - 12.0) < 1e-6


def test_cv_calculation():
    """Known scores, verify CV = std/mean."""
    scores = [10.0, 20.0, 30.0]
    mean_val = np.mean(scores)
    std_val = np.std(scores, ddof=0)
    expected_cv = std_val / mean_val

    result = ip.compute_cv(scores)
    assert abs(result - expected_cv) < 1e-6


def test_skip_threshold():
    """CV < 0.1 -> recommendation='skip', curriculum_skipped=true."""
    # All scores nearly identical -> low CV
    scores = [10.0, 10.01, 9.99, 10.0, 10.0]
    result = ip.build_variance_check(scores)

    assert result["cv"] < 0.1
    assert result["recommendation"] == "skip"


def test_proceed_threshold():
    """CV >= 0.1 -> recommendation='proceed'."""
    # Diverse scores -> high CV
    scores = [1.0, 10.0, 100.0, 50.0, 5.0]
    result = ip.build_variance_check(scores)

    assert result["cv"] >= 0.1
    assert result["recommendation"] == "proceed"


def test_cv_all_zeros():
    """If all scores are zero, CV should be 0 and recommendation skip."""
    scores = [0.0, 0.0, 0.0]
    result = ip.build_variance_check(scores)
    assert result["cv"] == 0.0
    assert result["recommendation"] == "skip"


def test_output_schema():
    """Verify the output dict has the correct schema keys."""
    scores_list = [
        {"shard": "shard_0", "influence_score": 5.0},
        {"shard": "shard_1", "influence_score": 3.0},
    ]
    output = ip.build_output(
        checkpoint="ckpt.safetensors",
        scores=scores_list,
        variance_check={"mean": 4.0, "std": 1.0, "cv": 0.25, "recommendation": "proceed"},
    )

    assert output["checkpoint"] == "ckpt.safetensors"
    assert output["n_shards_scored"] == 2
    assert output["scores"] == scores_list
    assert output["variance_check"]["recommendation"] == "proceed"
    assert output["curriculum_skipped"] is False
    assert output["reason"] is None


def test_output_schema_skipped():
    """Verify schema when curriculum is skipped."""
    output = ip.build_output(
        checkpoint="ckpt.safetensors",
        scores=[],
        variance_check={"mean": 1.0, "std": 0.01, "cv": 0.01, "recommendation": "skip"},
    )
    assert output["curriculum_skipped"] is True
    assert output["reason"] == "CV < 0.1"
