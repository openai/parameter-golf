from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

SummaryReduction = Literal["identity", "last", "mean", "mean_abs", "max_abs"]


@dataclass(frozen=True)
class ControllerSummaryConfig:
    reduction: SummaryReduction = "identity"
    normalize: bool = False
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class ControllerSummary:
    values: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.size < 1:
            raise ValueError("ControllerSummary must contain at least one value")
        object.__setattr__(self, "values", values)

    @property
    def dim(self) -> int:
        return int(self.values.shape[0])


class ControllerSummaryBuilder:
    def __init__(self, config: ControllerSummaryConfig | None = None):
        self.config = config or ControllerSummaryConfig()

    def _reduce(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 0:
            return signal.reshape(1)
        if signal.ndim == 1:
            return signal
        if self.config.reduction == "identity":
            return signal.reshape(-1)
        if self.config.reduction == "last":
            return signal[-1]
        if self.config.reduction == "mean":
            return np.mean(signal, axis=0)
        if self.config.reduction == "mean_abs":
            return np.mean(np.abs(signal), axis=0)
        if self.config.reduction == "max_abs":
            return np.max(np.abs(signal), axis=0)
        raise ValueError(f"unknown summary reduction: {self.config.reduction}")

    def encode(
        self,
        signal: float | Sequence[float] | np.ndarray,
        *,
        name: str | None = None,
    ) -> ControllerSummary:
        array = np.asarray(signal, dtype=np.float64)
        summary = self._reduce(array).reshape(-1)
        if self.config.normalize:
            norm = float(np.linalg.norm(summary))
            if norm > self.config.eps:
                summary = summary / norm
        return ControllerSummary(values=summary, name=name)


def stack_summaries(summaries: Sequence[ControllerSummary]) -> np.ndarray:
    if not summaries:
        raise ValueError("stack_summaries requires at least one summary")
    dim = summaries[0].dim
    for summary in summaries[1:]:
        if summary.dim != dim:
            raise ValueError("all controller summaries must share the same dimension")
    return np.vstack([summary.values for summary in summaries])


__all__ = [
    "ControllerSummary",
    "ControllerSummaryBuilder",
    "ControllerSummaryConfig",
    "SummaryReduction",
    "stack_summaries",
]
