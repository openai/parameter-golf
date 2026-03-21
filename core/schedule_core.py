from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class ChunkWindow:
    win_start: int
    win_len: int
    chunk_offset: int
    chunk_len: int


def find_docs(all_tokens: Tensor, bos_id: int, include_next_bos: bool = True) -> list[tuple[int, int]]:
    """Return (start_offset, length) pairs for BOS-delimited documents."""
    bos_positions = (all_tokens == bos_id).nonzero(as_tuple=True)[0].tolist()
    docs = []
    for i, start in enumerate(bos_positions):
        end = bos_positions[i + 1] if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        if end - start < 2:
            raise ValueError(f"Document at offset {start} is too short: length={end - start}")
        docs.append((int(start), int(end - start)))
    return docs


def compute_chunk_window(
    ci: int,
    pred_len: int,
    num_chunks: int,
    chunk_size: int,
    eval_seq_len: int,
) -> ChunkWindow:
    """Return the legal evaluation window and score range for chunk `ci`."""
    if ci < 0:
        raise ValueError(f"ci must be non-negative, got {ci}")
    if pred_len <= 0:
        raise ValueError(f"pred_len must be positive, got {pred_len}")
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if eval_seq_len <= 0:
        raise ValueError(f"eval_seq_len must be positive, got {eval_seq_len}")
    if ci >= num_chunks:
        raise ValueError(f"ci={ci} must be less than num_chunks={num_chunks}")

    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start

    if chunk_start < 0 or chunk_end > pred_len or chunk_start >= chunk_end:
        raise AssertionError("Computed invalid chunk range")
    if win_start < 0 or win_len <= 0 or win_start + win_len != chunk_end:
        raise AssertionError("Computed invalid window range")
    if chunk_offset < 0 or chunk_offset + chunk_len > win_len:
        raise AssertionError("Chunk score range is not contained within the window")

    return ChunkWindow(
        win_start=win_start,
        win_len=win_len,
        chunk_offset=chunk_offset,
        chunk_len=chunk_len,
    )
