from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .control import ControllerSummary, ControllerSummaryBuilder, ControllerSummaryConfig


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


@dataclass(frozen=True)
class HormoneModulationConfig:
    refresh_stride: int = 1
    summary: ControllerSummaryConfig = field(default_factory=ControllerSummaryConfig)
    hormone_count: int = 4
    summary_scale: float = 1.0
    hormone_bias: float = 0.0
    hormone_scale: float = 1.0
    output_indices: tuple[int, ...] = (0, 1)
    output_biases: tuple[float, ...] = (0.0, 0.0)
    output_scales: tuple[float, ...] = (1.0, 1.0)
    seed: int = 7

    def __post_init__(self) -> None:
        if self.refresh_stride < 1:
            raise ValueError("refresh_stride must be >= 1")
        if self.hormone_count < 1:
            raise ValueError("hormone_count must be >= 1")
        if self.summary_scale < 0.0:
            raise ValueError("summary_scale must be >= 0")
        if self.hormone_scale < 0.0:
            raise ValueError("hormone_scale must be >= 0")
        if not self.output_indices:
            raise ValueError("output_indices must contain at least one index")
        if len(self.output_indices) != len(self.output_biases) or len(self.output_indices) != len(self.output_scales):
            raise ValueError("output_indices, output_biases, and output_scales must have the same length")
        if any(index < 0 for index in self.output_indices):
            raise ValueError("output_indices must be >= 0")
        if any(scale < 0.0 for scale in self.output_scales):
            raise ValueError("output_scales must be >= 0")


@dataclass(frozen=True)
class HormoneState:
    hormones: np.ndarray
    outputs: np.ndarray
    step: int
    refreshed: bool
    last_refresh_step: int
    summary_name: str | None = None

    def __post_init__(self) -> None:
        hormones = np.asarray(self.hormones, dtype=np.float64).reshape(-1)
        outputs = np.asarray(self.outputs, dtype=np.float64).reshape(-1)
        if hormones.size < 1:
            raise ValueError("HormoneState requires at least one hormone value")
        if outputs.size < 1:
            raise ValueError("HormoneState requires at least one output value")
        object.__setattr__(self, "hormones", hormones)
        object.__setattr__(self, "outputs", outputs)


class HormoneModulator:
    def __init__(self, summary_dim: int, config: HormoneModulationConfig | None = None):
        if summary_dim < 1:
            raise ValueError("summary_dim must be >= 1")
        self.summary_dim = summary_dim
        self.config = config or HormoneModulationConfig()
        self.summary_builder = ControllerSummaryBuilder(self.config.summary)
        rng = np.random.default_rng(self.config.seed)
        self._summary_projection = rng.standard_normal((summary_dim, self.config.hormone_count)).astype(np.float64)
        self._summary_projection /= np.sqrt(max(summary_dim, 1))
        self._output_indices = np.asarray(self.config.output_indices, dtype=np.int64)
        self._output_biases = np.asarray(self.config.output_biases, dtype=np.float64)
        self._output_scales = np.asarray(self.config.output_scales, dtype=np.float64)

    @property
    def output_count(self) -> int:
        return int(self._output_indices.shape[0])

    def initial_state(self) -> HormoneState:
        return HormoneState(
            hormones=np.zeros(self.config.hormone_count, dtype=np.float64),
            outputs=np.zeros(self.output_count, dtype=np.float64),
            step=-1,
            refreshed=False,
            last_refresh_step=-self.config.refresh_stride,
            summary_name=None,
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

    def _project_hormones(self, summary: ControllerSummary) -> np.ndarray:
        if summary.dim != self.summary_dim:
            raise ValueError("summary does not match the configured summary_dim")
        projection = summary.values @ self._summary_projection
        return np.tanh(self.config.hormone_bias + self.config.summary_scale * projection)

    def _project_outputs(self, hormones: np.ndarray) -> np.ndarray:
        selected = hormones[self._output_indices]
        outputs = self._output_biases + (self._output_scales * selected * self.config.hormone_scale)
        return np.asarray(_sigmoid(outputs), dtype=np.float64)

    def advance(
        self,
        state: HormoneState,
        summary: ControllerSummary | float | Sequence[float] | np.ndarray,
        *,
        step: int,
        name: str | None = None,
    ) -> HormoneState:
        controller_summary = self._coerce_summary(summary, name=name)
        should_refresh = (step - state.last_refresh_step) >= self.config.refresh_stride
        if not should_refresh:
            return HormoneState(
                hormones=state.hormones,
                outputs=state.outputs,
                step=step,
                refreshed=False,
                last_refresh_step=state.last_refresh_step,
                summary_name=controller_summary.name,
            )

        hormones = self._project_hormones(controller_summary)
        outputs = self._project_outputs(hormones)
        return HormoneState(
            hormones=hormones,
            outputs=outputs,
            step=step,
            refreshed=True,
            last_refresh_step=step,
            summary_name=controller_summary.name,
        )

    def project(
        self,
        summary: ControllerSummary | float | Sequence[float] | np.ndarray,
        *,
        name: str | None = None,
    ) -> HormoneState:
        return self.advance(self.initial_state(), summary, step=0, name=name)


__all__ = [
    "HormoneModulationConfig",
    "HormoneModulator",
    "HormoneState",
]
