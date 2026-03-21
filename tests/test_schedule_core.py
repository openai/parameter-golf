from __future__ import annotations

import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from core.schedule_core import compute_chunk_window, find_docs


@given(
    pred_len=st.integers(min_value=1, max_value=128),
    chunk_size=st.integers(min_value=1, max_value=32),
    eval_seq_len=st.integers(min_value=1, max_value=64),
)
def test_chunk_windows_cover_prediction_range_exactly(pred_len: int, chunk_size: int, eval_seq_len: int):
    assume(eval_seq_len >= min(chunk_size, pred_len))
    num_chunks = (pred_len + chunk_size - 1) // chunk_size
    covered: list[int] = []
    for ci in range(num_chunks):
        window = compute_chunk_window(ci, pred_len, num_chunks, chunk_size, eval_seq_len)
        covered.extend(range(ci * chunk_size, ci * chunk_size + window.chunk_len))
        assert 0 <= window.win_start
        assert 0 < window.win_len <= min(pred_len, eval_seq_len)
        assert 0 <= window.chunk_offset < window.win_len
        assert window.chunk_offset + window.chunk_len <= window.win_len
    assert covered == list(range(pred_len))


def test_find_docs_respects_bos_boundaries():
    bos_id = 1
    tokens = torch.tensor([1, 11, 12, 1, 21, 22, 23, 1, 31, 32], dtype=torch.int64)
    docs = find_docs(tokens, bos_id=bos_id)
    assert docs == [(0, 4), (3, 5), (7, 3)]


def test_find_docs_without_next_bos_excludes_boundary_token():
    bos_id = 1
    tokens = torch.tensor([1, 11, 12, 1, 21, 22], dtype=torch.int64)
    docs = find_docs(tokens, bos_id=bos_id, include_next_bos=False)
    assert docs == [(0, 3), (3, 3)]


def test_find_docs_with_next_bos_preserves_flat_prediction_count():
    bos_id = 1
    tokens = torch.tensor([1, 11, 12, 1, 21, 22], dtype=torch.int64)
    docs = find_docs(tokens, bos_id=bos_id, include_next_bos=True)
    assert sum(doc_len - 1 for _, doc_len in docs) == tokens.numel() - 1
