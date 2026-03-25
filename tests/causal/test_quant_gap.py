"""Tests for scripts/causal/quant_gap_analysis.py."""
from __future__ import annotations

import pytest

from scripts.causal.quant_gap_analysis import (
    compute_quant_gap,
    check_threshold,
)


def test_gap_computation():
    """Gap = post_quant_bpb - pre_quant_bpb."""
    pre_bpb = 1.120
    post_bpb = 1.137
    gap = compute_quant_gap(pre_bpb, post_bpb)
    assert gap == pytest.approx(0.017, abs=1e-6)


def test_threshold_check_true():
    """Gap > 3x effect => True."""
    gap = 0.060
    largest_effect = 0.010  # 3x = 0.03, gap=0.06 > 0.03
    assert check_threshold(gap, largest_effect) is True


def test_threshold_check_false():
    """Gap <= 3x effect => False."""
    gap = 0.020
    largest_effect = 0.010  # 3x = 0.03, gap=0.02 <= 0.03
    assert check_threshold(gap, largest_effect) is False


def test_threshold_check_no_effect():
    """When largest_effect is None, threshold check defaults to False."""
    gap = 0.017
    assert check_threshold(gap, None) is False


def test_gap_negative():
    """If post < pre (unlikely but possible), gap is negative."""
    gap = compute_quant_gap(1.14, 1.13)
    assert gap < 0
