#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from mlx_eval_utils import flatten_validation_docs, iter_doc_stride_windows, iter_doc_windows, iter_validation_docs


def iter_flat_chunks(tokens: np.ndarray, seq_len: int):
    usable = ((tokens.size - 1) // seq_len) * seq_len
    for start in range(0, usable, seq_len):
        yield tokens[start : start + seq_len + 1]


def check_flat_can_mix_context() -> None:
    bos_id = 1
    tokens = np.array([bos_id, 11, 12, bos_id, 21, 22, bos_id, 31], dtype=np.int32)
    docs = tuple(iter_validation_docs(tokens, bos_id=bos_id))
    flat_chunk = next(iter_flat_chunks(flatten_validation_docs(docs), seq_len=4))
    assert flat_chunk.tolist() == [1, 11, 12, 1, 21]


def check_doc_eval_respects_boundaries() -> None:
    bos_id = 1
    tokens = np.array([bos_id, 11, 12, bos_id, 21, 22, bos_id, 31], dtype=np.int32)
    docs = tuple(iter_validation_docs(tokens, bos_id=bos_id))
    doc_windows = list(iter_doc_windows(docs, seq_len=4))
    second_doc_window = next(window for window in doc_windows if window.doc_index == 1)
    assert second_doc_window.chunk.tolist() == [1, 21, 22]
    assert 11 not in second_doc_window.chunk
    assert 12 not in second_doc_window.chunk


def check_doc_stride_scores_once_causally() -> None:
    doc = np.array([1, 101, 102, 103, 104, 105, 106], dtype=np.int32)
    windows = list(iter_doc_stride_windows((doc,), seq_len=4, stride=2))
    total_targets = doc.size - 1
    seen = [0] * total_targets
    for window in windows:
        chunk_start = window.target_start - window.score_start
        for offset, target_pos in enumerate(range(window.target_start, window.target_stop)):
            seen[target_pos] += 1
            local_idx = window.score_start + offset
            assert np.array_equal(window.chunk[: local_idx + 1], doc[chunk_start : chunk_start + local_idx + 1])
    assert seen == [1] * total_targets


def main() -> None:
    check_flat_can_mix_context()
    print("flat_mixes_docs: ok")
    check_doc_eval_respects_boundaries()
    print("doc_eval_respects_boundaries: ok")
    check_doc_stride_scores_once_causally()
    print("doc_stride_scores_once_causally: ok")


if __name__ == "__main__":
    main()
