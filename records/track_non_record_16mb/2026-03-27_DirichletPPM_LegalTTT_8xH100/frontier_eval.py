from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class WindowShardPlan:
    windows: list[int]
    prefix_start: int | None
    prefix_end: int | None
    suffix_start: int | None
    suffix_end: int | None
    local_start: int | None
    local_end: int | None


def sliding_window_starts(
    total_tokens: int,
    seq_len: int,
    stride: int,
    *,
    require_full_stride: bool,
) -> list[int]:
    starts: list[int] = []
    for ws in range(0, total_tokens, stride):
        wlen = min(ws + seq_len, total_tokens) - ws
        if require_full_stride:
            if wlen >= stride or ws == 0:
                starts.append(ws)
        elif wlen >= 1:
            starts.append(ws)
    return starts


def window_score_bounds(
    window_start: int,
    total_tokens: int,
    seq_len: int,
    stride: int,
) -> tuple[int, int]:
    end = min(window_start + seq_len, total_tokens)
    wlen = end - window_start
    score_offset = 0 if window_start == 0 else max(wlen - stride, 0)
    first_target_pos = window_start + score_offset + 1
    last_target_pos = window_start + wlen
    return first_target_pos, last_target_pos


def assign_windows_to_score_chunks(
    window_starts: Sequence[int],
    total_tokens: int,
    seq_len: int,
    stride: int,
    chunk_tokens: int,
) -> list[list[int]]:
    if not window_starts:
        return []
    num_chunks = max(1, (total_tokens + chunk_tokens - 1) // chunk_tokens)
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        first_target_pos, _last_target_pos = window_score_bounds(ws, total_tokens, seq_len, stride)
        chunk_index = min((first_target_pos - 1) // chunk_tokens, num_chunks - 1)
        chunk_windows[chunk_index].append(ws)
    return chunk_windows


def contiguous_rank_slice(total_items: int, rank: int, world_size: int) -> tuple[int, int]:
    return (total_items * rank) // world_size, (total_items * (rank + 1)) // world_size


def score_range_for_windows(
    window_starts: Sequence[int],
    total_tokens: int,
    seq_len: int,
    stride: int,
) -> tuple[int | None, int | None]:
    if not window_starts:
        return None, None
    first_target_pos, _ = window_score_bounds(window_starts[0], total_tokens, seq_len, stride)
    _, last_target_pos = window_score_bounds(window_starts[-1], total_tokens, seq_len, stride)
    return first_target_pos, last_target_pos


def plan_distributed_window_shard(
    window_starts: Sequence[int],
    total_tokens: int,
    seq_len: int,
    stride: int,
    rank: int,
    world_size: int,
) -> WindowShardPlan:
    shard_start, shard_end = contiguous_rank_slice(len(window_starts), rank, world_size)
    my_windows = list(window_starts[shard_start:shard_end])
    global_start, global_end = score_range_for_windows(window_starts, total_tokens, seq_len, stride)
    local_start, local_end = score_range_for_windows(my_windows, total_tokens, seq_len, stride)
    if local_start is None or local_end is None:
        return WindowShardPlan(
            windows=my_windows,
            prefix_start=global_start,
            prefix_end=global_end,
            suffix_start=None,
            suffix_end=None,
            local_start=None,
            local_end=None,
        )
    prefix_start = global_start
    prefix_end = local_start - 1 if global_start is not None and local_start > global_start else None
    suffix_start = local_end + 1 if global_end is not None and local_end < global_end else None
    suffix_end = global_end if suffix_start is not None else None
    return WindowShardPlan(
        windows=my_windows,
        prefix_start=prefix_start,
        prefix_end=prefix_end,
        suffix_start=suffix_start,
        suffix_end=suffix_end,
        local_start=local_start,
        local_end=local_end,
    )


def commit_position_range(
    cache,
    token_stream: np.ndarray,
    start_pos: int | None,
    end_pos: int | None,
    *,
    block_size: int = 131_072,
) -> None:
    if cache is None or start_pos is None or end_pos is None or end_pos < start_pos:
        return
    for block_start in range(start_pos, end_pos + 1, block_size):
        block_end = min(block_start + block_size - 1, end_pos)
        positions = np.arange(block_start, block_end + 1, dtype=np.int64)
        cache.replay_scored_positions(token_stream, positions)


def reduce_eval_totals(
    loss_sum: torch.Tensor,
    token_count: torch.Tensor,
    byte_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    return loss_sum, token_count, byte_count
