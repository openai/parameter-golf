from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .control import ControllerSummary, ControllerSummaryBuilder, ControllerSummaryConfig

def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


@dataclass(frozen=True)
class PathwayGateConfig:
    refresh_stride: int = 1
    summary: ControllerSummaryConfig = field(default_factory=ControllerSummaryConfig)
    fast_to_mid_index: int = 0
    mid_to_slow_index: int = 0
    fast_to_mid_bias: float = 0.0
    fast_to_mid_scale: float = 1.0
    mid_to_slow_bias: float = 0.0
    mid_to_slow_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.refresh_stride < 1:
            raise ValueError("refresh_stride must be >= 1")
        if self.fast_to_mid_index < 0:
            raise ValueError("fast_to_mid_index must be >= 0")
        if self.mid_to_slow_index < 0:
            raise ValueError("mid_to_slow_index must be >= 0")
        if self.fast_to_mid_scale < 0.0:
            raise ValueError("fast_to_mid_scale must be >= 0")
        if self.mid_to_slow_scale < 0.0:
            raise ValueError("mid_to_slow_scale must be >= 0")


@dataclass(frozen=True)
class PathwayGateValues:
    fast_to_mid: float
    mid_to_slow: float
    step: int
    refreshed: bool
    summary_name: str | None = None

    def __post_init__(self) -> None:
        fast_to_mid = float(np.clip(self.fast_to_mid, 0.0, 1.0))
        mid_to_slow = float(np.clip(self.mid_to_slow, 0.0, 1.0))
        object.__setattr__(self, "fast_to_mid", fast_to_mid)
        object.__setattr__(self, "mid_to_slow", mid_to_slow)


@dataclass(frozen=True)
class PathwayGateState:
    last_refresh_step: int
    values: PathwayGateValues


class PathwayGateController:
    def __init__(
        self,
        config: PathwayGateConfig | None = None,
        *,
        summary_builder: ControllerSummaryBuilder | None = None,
    ):
        self.config = config or PathwayGateConfig()
        self.summary_builder = summary_builder or ControllerSummaryBuilder(self.config.summary)

    def initial_state(self) -> PathwayGateState:
        return PathwayGateState(
            last_refresh_step=-self.config.refresh_stride,
            values=PathwayGateValues(
                fast_to_mid=0.0,
                mid_to_slow=0.0,
                step=-1,
                refreshed=False,
                summary_name=None,
            ),
        )

    def _coerce_summary(
        self,
        summary: ControllerSummary | float | Sequence[float] | np.ndarray,
        *,
        name: str | None = None,
    ) -> ControllerSummary:
        if isinstance(summary, ControllerSummary):
            return summary
        return self.summary_builder.encode(summary, name=name)

    def _summary_signals(self, summary: ControllerSummary) -> tuple[float, float]:
        values = summary.values
        max_index = max(self.config.fast_to_mid_index, self.config.mid_to_slow_index)
        if max_index >= values.size:
            raise ValueError("summary does not contain enough dimensions for the configured gate indices")
        return (
            float(values[self.config.fast_to_mid_index]),
            float(values[self.config.mid_to_slow_index]),
        )

    def _compute_values(self, summary: ControllerSummary, step: int, refreshed: bool) -> PathwayGateValues:
        fast_signal, slow_signal = self._summary_signals(summary)
        fast_to_mid = _sigmoid(self.config.fast_to_mid_bias + self.config.fast_to_mid_scale * fast_signal)
        mid_to_slow = _sigmoid(self.config.mid_to_slow_bias + self.config.mid_to_slow_scale * slow_signal)
        return PathwayGateValues(
            fast_to_mid=fast_to_mid,
            mid_to_slow=mid_to_slow,
            step=step,
            refreshed=refreshed,
            summary_name=summary.name,
        )

    def advance(
        self,
        state: PathwayGateState,
        summary: ControllerSummary | float | Sequence[float] | np.ndarray,
        *,
        step: int,
        name: str | None = None,
    ) -> PathwayGateState:
        controller_summary = self._coerce_summary(summary, name=name)
        should_refresh = (step - state.last_refresh_step) >= self.config.refresh_stride
        if should_refresh:
            values = self._compute_values(controller_summary, step, refreshed=True)
            return PathwayGateState(last_refresh_step=step, values=values)

        values = PathwayGateValues(
            fast_to_mid=state.values.fast_to_mid,
            mid_to_slow=state.values.mid_to_slow,
            step=step,
            refreshed=False,
            summary_name=controller_summary.name,
        )
        return PathwayGateState(last_refresh_step=state.last_refresh_step, values=values)


__all__ = [
    "PathwayGateConfig",
    "PathwayGateController",
    "PathwayGateState",
    "PathwayGateValues",
]
