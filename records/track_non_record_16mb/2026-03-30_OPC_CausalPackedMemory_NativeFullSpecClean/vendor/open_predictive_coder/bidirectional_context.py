from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .codecs import ensure_tokens


@dataclass(frozen=True)
class BidirectionalContextConfig:
    left_order: int = 2
    right_order: int = 2

    def __post_init__(self) -> None:
        if self.left_order < 0:
            raise ValueError("left_order must be >= 0")
        if self.right_order < 0:
            raise ValueError("right_order must be >= 0")


@dataclass(frozen=True)
class BidirectionalContextNeighborhood:
    position: int
    token: int
    left_context: tuple[int, ...]
    right_context: tuple[int, ...]
    left_support: int
    right_support: int
    pair_support: int
    candidate_tokens: tuple[int, ...]
    candidate_count: int
    deterministic: bool


@dataclass(frozen=True)
class BidirectionalContextStats:
    sequence_length: int
    neighborhood_count: int
    left_context_count: int
    right_context_count: int
    pair_context_count: int
    deterministic_fraction: float
    candidate_le_2_rate: float
    candidate_le_4_rate: float
    candidate_le_8_rate: float
    mean_candidate_size: float
    median_candidate_size: float
    max_candidate_size: int
    mean_left_support: float
    mean_right_support: float
    mean_pair_support: float
    candidate_sizes: tuple[int, ...]
    neighborhoods: tuple[BidirectionalContextNeighborhood, ...]


@dataclass(frozen=True)
class BidirectionalLeaveOneOutStats:
    position: int
    token: int
    left_context: tuple[int, ...]
    right_context: tuple[int, ...]
    left_support: int
    right_support: int
    pair_support: int
    candidate_tokens: tuple[int, ...]
    candidate_count: int
    deterministic: bool


def _coerce_tokens(
    data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[Any],
) -> np.ndarray:
    tokens = ensure_tokens(data).astype(np.int64, copy=False)
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    return tokens


def _left_context(tokens: np.ndarray, position: int, order: int) -> tuple[int, ...]:
    if order == 0:
        return ()
    start = max(0, position - order)
    return tuple(int(token) for token in tokens[start:position])


def _right_context(tokens: np.ndarray, position: int, order: int) -> tuple[int, ...]:
    if order == 0:
        return ()
    stop = min(tokens.size, position + order + 1)
    return tuple(int(token) for token in tokens[position + 1 : stop])


def _build_support_maps(
    tokens: np.ndarray,
    config: BidirectionalContextConfig,
) -> tuple[
    dict[tuple[int, ...], int],
    dict[tuple[int, ...], int],
    dict[tuple[tuple[int, ...], tuple[int, ...]], int],
    dict[tuple[tuple[int, ...], tuple[int, ...]], dict[int, int]],
]:
    left_support: dict[tuple[int, ...], int] = {}
    right_support: dict[tuple[int, ...], int] = {}
    pair_support: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    pair_candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[int, int]] = {}

    for position, token in enumerate(tokens):
        left = _left_context(tokens, position, config.left_order)
        right = _right_context(tokens, position, config.right_order)
        pair = (left, right)

        left_support[left] = left_support.get(left, 0) + 1
        right_support[right] = right_support.get(right, 0) + 1
        pair_support[pair] = pair_support.get(pair, 0) + 1

        candidates = pair_candidates.setdefault(pair, {})
        candidate_token = int(token)
        candidates[candidate_token] = candidates.get(candidate_token, 0) + 1

    return left_support, right_support, pair_support, pair_candidates


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


class BidirectionalContextProbe:
    def __init__(self, config: BidirectionalContextConfig | None = None):
        self.config = config or BidirectionalContextConfig()
        self._last_tokens: np.ndarray | None = None
        self._last_stats: BidirectionalContextStats | None = None
        self._last_pair_candidates: dict[tuple[tuple[int, ...], tuple[int, ...]], dict[int, int]] | None = None
        self._last_left_support: dict[tuple[int, ...], int] | None = None
        self._last_right_support: dict[tuple[int, ...], int] | None = None
        self._last_pair_support: dict[tuple[tuple[int, ...], tuple[int, ...]], int] | None = None

    def scan(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[Any],
    ) -> BidirectionalContextStats:
        sequence = _coerce_tokens(tokens)
        if sequence.size == 0:
            raise ValueError("tokens must contain at least one item")

        left_support, right_support, pair_support, pair_candidates = _build_support_maps(sequence, self.config)
        neighborhoods: list[BidirectionalContextNeighborhood] = []
        candidate_sizes: list[int] = []
        left_support_values: list[int] = []
        right_support_values: list[int] = []
        pair_support_values: list[int] = []

        for position, token in enumerate(sequence):
            left = _left_context(sequence, position, self.config.left_order)
            right = _right_context(sequence, position, self.config.right_order)
            pair = (left, right)
            candidates = pair_candidates[pair]
            candidate_tokens = tuple(sorted(int(candidate) for candidate in candidates))
            candidate_count = len(candidate_tokens)
            left_count = left_support[left]
            right_count = right_support[right]
            pair_count = pair_support[pair]
            neighborhoods.append(
                BidirectionalContextNeighborhood(
                    position=position,
                    token=int(token),
                    left_context=left,
                    right_context=right,
                    left_support=left_count,
                    right_support=right_count,
                    pair_support=pair_count,
                    candidate_tokens=candidate_tokens,
                    candidate_count=candidate_count,
                    deterministic=candidate_count == 1,
                )
            )
            candidate_sizes.append(candidate_count)
            left_support_values.append(left_count)
            right_support_values.append(right_count)
            pair_support_values.append(pair_count)

        candidate_array = np.asarray(candidate_sizes, dtype=np.float64)
        stats = BidirectionalContextStats(
            sequence_length=int(sequence.size),
            neighborhood_count=len(neighborhoods),
            left_context_count=len(left_support),
            right_context_count=len(right_support),
            pair_context_count=len(pair_support),
            deterministic_fraction=float(np.mean(candidate_array == 1.0)),
            candidate_le_2_rate=float(np.mean(candidate_array <= 2.0)),
            candidate_le_4_rate=float(np.mean(candidate_array <= 4.0)),
            candidate_le_8_rate=float(np.mean(candidate_array <= 8.0)),
            mean_candidate_size=float(np.mean(candidate_array)),
            median_candidate_size=float(np.median(candidate_array)),
            max_candidate_size=int(np.max(candidate_array)),
            mean_left_support=_mean(left_support_values),
            mean_right_support=_mean(right_support_values),
            mean_pair_support=_mean(pair_support_values),
            candidate_sizes=tuple(int(size) for size in candidate_sizes),
            neighborhoods=tuple(neighborhoods),
        )

        self._last_tokens = sequence
        self._last_stats = stats
        self._last_pair_candidates = pair_candidates
        self._last_left_support = left_support
        self._last_right_support = right_support
        self._last_pair_support = pair_support
        return stats

    def determinism_stats(self) -> BidirectionalContextStats:
        if self._last_stats is None:
            raise ValueError("scan must be called before determinism_stats")
        return self._last_stats

    def leave_one_out(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[Any],
        position: int,
    ) -> BidirectionalLeaveOneOutStats:
        sequence = _coerce_tokens(tokens)
        if sequence.size == 0:
            raise ValueError("tokens must contain at least one item")
        if position < 0 or position >= sequence.size:
            raise IndexError("position out of range")

        left_support, right_support, pair_support, pair_candidates = _build_support_maps(sequence, self.config)
        left = _left_context(sequence, position, self.config.left_order)
        right = _right_context(sequence, position, self.config.right_order)
        pair = (left, right)
        token = int(sequence[position])
        candidates = dict(pair_candidates[pair])
        if token in candidates:
            if candidates[token] <= 1:
                del candidates[token]
            else:
                candidates[token] -= 1
        candidate_tokens = tuple(sorted(int(candidate) for candidate in candidates))
        return BidirectionalLeaveOneOutStats(
            position=position,
            token=token,
            left_context=left,
            right_context=right,
            left_support=left_support[left],
            right_support=right_support[right],
            pair_support=max(pair_support[pair] - 1, 0),
            candidate_tokens=candidate_tokens,
            candidate_count=len(candidate_tokens),
            deterministic=len(candidate_tokens) == 1,
        )


__all__ = [
    "BidirectionalContextConfig",
    "BidirectionalContextLeaveOneOutStats",
    "BidirectionalContextNeighborhood",
    "BidirectionalContextProbe",
    "BidirectionalContextStats",
    "BidirectionalLeaveOneOutStats",
]

BidirectionalContextLeaveOneOutStats = BidirectionalLeaveOneOutStats
