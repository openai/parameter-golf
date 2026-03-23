from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationWindow:
    doc_index: int
    target_start: int
    target_stop: int
    chunk: np.ndarray
    score_start: int
    score_stop: int


def iter_validation_docs(
    tokens: np.ndarray,
    *,
    bos_id: int,
    max_docs: int | None = None,
) -> Iterator[np.ndarray]:
    starts = np.flatnonzero(tokens == bos_id)
    if starts.size == 0 or int(starts[0]) != 0:
        raise ValueError("validation tokens must begin with a BOS token for document-aware eval")
    limit = starts.size if max_docs is None else min(int(max_docs), int(starts.size))
    for i in range(limit):
        start = int(starts[i])
        end = int(starts[i + 1]) if i + 1 < starts.size else int(tokens.size)
        yield tokens[start:end]


def flatten_validation_docs(docs: Iterable[np.ndarray]) -> np.ndarray:
    doc_list = [np.asarray(doc, dtype=np.int32) for doc in docs]
    if not doc_list:
        return np.empty((0,), dtype=np.int32)
    return np.ascontiguousarray(np.concatenate(doc_list, axis=0))


def count_doc_windows(docs: Iterable[np.ndarray], *, seq_len: int) -> int:
    total = 0
    for doc in docs:
        total_targets = max(int(doc.size) - 1, 0)
        total += (total_targets + seq_len - 1) // seq_len
    return total


def count_doc_stride_windows(docs: Iterable[np.ndarray], *, seq_len: int, stride: int) -> int:
    total = 0
    for doc in docs:
        total_targets = max(int(doc.size) - 1, 0)
        if total_targets <= 0:
            continue
        if total_targets <= seq_len:
            total += 1
            continue
        total += (total_targets + stride - 1) // stride
    return total


def iter_doc_windows(docs: Iterable[np.ndarray], *, seq_len: int) -> Iterator[ValidationWindow]:
    for doc_index, doc in enumerate(docs):
        total_targets = max(int(doc.size) - 1, 0)
        for target_start in range(0, total_targets, seq_len):
            target_stop = min(target_start + seq_len, total_targets)
            yield ValidationWindow(
                doc_index=doc_index,
                target_start=target_start,
                target_stop=target_stop,
                chunk=doc[target_start : target_stop + 1],
                score_start=0,
                score_stop=target_stop - target_start,
            )


def iter_doc_stride_windows(
    docs: Iterable[np.ndarray],
    *,
    seq_len: int,
    stride: int,
) -> Iterator[ValidationWindow]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    for doc_index, doc in enumerate(docs):
        total_targets = max(int(doc.size) - 1, 0)
        if total_targets <= 0:
            continue
        if total_targets <= seq_len:
            yield ValidationWindow(
                doc_index=doc_index,
                target_start=0,
                target_stop=total_targets,
                chunk=doc[: total_targets + 1],
                score_start=0,
                score_stop=total_targets,
            )
            continue
        for target_start in range(0, total_targets, stride):
            target_stop = min(target_start + stride, total_targets)
            window_start = max(target_stop - seq_len, 0)
            yield ValidationWindow(
                doc_index=doc_index,
                target_start=target_start,
                target_stop=target_stop,
                chunk=doc[window_start : target_stop + 1],
                score_start=target_start - window_start,
                score_stop=target_stop - window_start,
            )
