from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from core.metric_core import compute_loss_byte_deltas, compute_token_bytes, compute_val_bpb, finalize_eval_result


@st.composite
def byte_accounting_case(draw):
    vocab_size = draw(st.integers(min_value=1, max_value=12))
    length = draw(st.integers(min_value=1, max_value=24))
    prev_ids = draw(st.lists(st.integers(min_value=0, max_value=vocab_size - 1), min_size=length, max_size=length))
    tgt_ids = draw(st.lists(st.integers(min_value=0, max_value=vocab_size - 1), min_size=length, max_size=length))
    base_bytes = draw(st.lists(st.integers(min_value=1, max_value=6), min_size=vocab_size, max_size=vocab_size))
    has_space = draw(st.lists(st.booleans(), min_size=vocab_size, max_size=vocab_size))
    is_boundary = draw(st.lists(st.booleans(), min_size=vocab_size, max_size=vocab_size))
    losses = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
            min_size=length,
            max_size=length,
        )
    )
    return {
        "prev_ids": torch.tensor(prev_ids, dtype=torch.int64),
        "tgt_ids": torch.tensor(tgt_ids, dtype=torch.int64),
        "base_bytes_lut": torch.tensor(base_bytes, dtype=torch.int16),
        "has_leading_space_lut": torch.tensor(has_space, dtype=torch.bool),
        "is_boundary_token_lut": torch.tensor(is_boundary, dtype=torch.bool),
        "losses": torch.tensor(losses, dtype=torch.float32),
    }


@given(byte_accounting_case())
def test_compute_token_bytes_matches_manual_accounting(case):
    token_bytes = compute_token_bytes(
        case["prev_ids"],
        case["tgt_ids"],
        case["base_bytes_lut"],
        case["has_leading_space_lut"],
        case["is_boundary_token_lut"],
    )
    expected = []
    for prev_id, tgt_id in zip(case["prev_ids"].tolist(), case["tgt_ids"].tolist(), strict=True):
        bytes_for_token = int(case["base_bytes_lut"][tgt_id].item())
        if bool(case["has_leading_space_lut"][tgt_id].item()) and not bool(case["is_boundary_token_lut"][prev_id].item()):
            bytes_for_token += 1
        expected.append(bytes_for_token)
    assert token_bytes.tolist() == expected


@given(byte_accounting_case(), st.integers(min_value=1, max_value=23))
def test_loss_and_byte_deltas_are_additive(case, split_idx):
    if split_idx >= case["tgt_ids"].numel():
        split_idx = case["tgt_ids"].numel() - 1

    full_loss, full_bytes, full_tokens = compute_loss_byte_deltas(
        case["losses"],
        case["prev_ids"],
        case["tgt_ids"],
        case["base_bytes_lut"],
        case["has_leading_space_lut"],
        case["is_boundary_token_lut"],
    )
    left = compute_loss_byte_deltas(
        case["losses"][:split_idx],
        case["prev_ids"][:split_idx],
        case["tgt_ids"][:split_idx],
        case["base_bytes_lut"],
        case["has_leading_space_lut"],
        case["is_boundary_token_lut"],
    )
    right = compute_loss_byte_deltas(
        case["losses"][split_idx:],
        case["prev_ids"][split_idx:],
        case["tgt_ids"][split_idx:],
        case["base_bytes_lut"],
        case["has_leading_space_lut"],
        case["is_boundary_token_lut"],
    )
    assert full_loss.item() == pytest.approx(left[0].item() + right[0].item())
    assert full_bytes.item() == pytest.approx(left[1].item() + right[1].item())
    assert full_tokens == left[2] + right[2]


def test_compute_val_bpb_matches_formula():
    loss_sum = 13.5
    byte_count = 9
    assert compute_val_bpb(loss_sum, byte_count) == pytest.approx((loss_sum / math.log(2.0)) / byte_count)


def test_compute_val_bpb_rejects_nonpositive_byte_count():
    with pytest.raises(ValueError):
        compute_val_bpb(1.0, 0)


def test_finalize_eval_result_matches_formula():
    result = finalize_eval_result(loss_sum=13.5, token_count=6, byte_count=9)
    assert result.val_loss == pytest.approx(13.5 / 6)
    assert result.val_bpb == pytest.approx((13.5 / math.log(2.0)) / 9)
    assert result.loss_sum == pytest.approx(13.5)
    assert result.token_count == pytest.approx(6.0)
    assert result.byte_count == pytest.approx(9.0)


def test_finalize_eval_result_rejects_nonpositive_counts():
    with pytest.raises(ValueError):
        finalize_eval_result(loss_sum=1.0, token_count=0, byte_count=2)
    with pytest.raises(ValueError):
        finalize_eval_result(loss_sum=1.0, token_count=2, byte_count=0)
