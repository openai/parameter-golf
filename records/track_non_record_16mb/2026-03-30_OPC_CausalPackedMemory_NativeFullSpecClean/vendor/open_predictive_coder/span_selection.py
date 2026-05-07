from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifacts import ArtifactMetadata, ReplaySpan, make_replay_span


@dataclass(frozen=True)
class SpanSelectionConfig:
    threshold: float
    min_span: int = 1
    max_gap: int = 0

    def __post_init__(self) -> None:
        if self.min_span < 1:
            raise ValueError("min_span must be >= 1")
        if self.max_gap < 0:
            raise ValueError("max_gap must be >= 0")


@dataclass(frozen=True)
class ScoredSpan:
    start: int
    stop: int
    mean_score: float
    max_score: float
    count: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.stop < self.start:
            raise ValueError("stop must be >= start")
        if self.count < 0:
            raise ValueError("count must be >= 0")

    @property
    def length(self) -> int:
        return self.stop - self.start


def select_scored_spans(
    scores: np.ndarray,
    config: SpanSelectionConfig,
) -> tuple[ScoredSpan, ...]:
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if flat_scores.size == 0:
        return ()

    indices = np.flatnonzero(flat_scores >= config.threshold)
    if indices.size == 0:
        return ()

    spans: list[ScoredSpan] = []
    selected: list[int] = [int(indices[0])]
    start = int(indices[0])
    previous = int(indices[0])

    def flush() -> None:
        if not selected:
            return
        stop = previous + 1
        if stop - start < config.min_span:
            return
        values = flat_scores[np.asarray(selected, dtype=np.int64)]
        spans.append(
            ScoredSpan(
                start=start,
                stop=stop,
                mean_score=float(np.mean(values)),
                max_score=float(np.max(values)),
                count=len(selected),
            )
        )

    for raw_index in indices[1:]:
        index = int(raw_index)
        if index - previous - 1 <= config.max_gap:
            selected.append(index)
            previous = index
            continue
        flush()
        selected = [index]
        start = index
        previous = index
    flush()
    return tuple(spans)


def replay_spans_from_scores(
    scores: np.ndarray,
    config: SpanSelectionConfig,
    *,
    label: str | None = None,
    metadata: ArtifactMetadata | Mapping[str, Any] | None = None,
    **updates: Any,
) -> tuple[ReplaySpan, ...]:
    return tuple(
        make_replay_span(
            span.start,
            span.stop,
            label=label,
            metadata=metadata,
            **updates,
        )
        for span in select_scored_spans(scores, config)
    )


__all__ = [
    "ReplaySpan",
    "ScoredSpan",
    "SpanSelectionConfig",
    "replay_spans_from_scores",
    "select_scored_spans",
]
