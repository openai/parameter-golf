from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .control import ControllerSummary, ControllerSummaryConfig

SummaryMode = Literal["residual", "surprise"]


@dataclass(frozen=True)
class PredictiveSurpriseConfig:
    summary: ControllerSummaryConfig = field(
        default_factory=lambda: ControllerSummaryConfig(reduction="mean_abs", normalize=False)
    )
    feature_mode: SummaryMode = "surprise"
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")


@dataclass(frozen=True)
class PredictionState:
    predicted: np.ndarray
    actual: np.ndarray
    residual: np.ndarray
    surprise: np.ndarray
    summary: ControllerSummary
    step: int | None = None

    @property
    def surprise_score(self) -> float:
        return float(np.mean(self.surprise))

    @property
    def residual_score(self) -> float:
        return float(np.mean(np.abs(self.residual)))


class PredictiveSurpriseController:
    def __init__(self, config: PredictiveSurpriseConfig | None = None):
        self.config = config or PredictiveSurpriseConfig()

    @staticmethod
    def _coerce_vector(signal: np.ndarray | float | list[float] | tuple[float, ...]) -> np.ndarray:
        array = np.asarray(signal, dtype=np.float64).reshape(-1)
        if array.size < 1:
            raise ValueError("signal must contain at least one value")
        return array

    def _summary_from_signal(self, signal: np.ndarray, *, name: str | None = None) -> ControllerSummary:
        signal = np.asarray(signal, dtype=np.float64).reshape(-1)
        summary = np.asarray(
            [
                float(np.mean(np.abs(signal))),
                float(np.max(np.abs(signal))),
            ],
            dtype=np.float64,
        )
        if self.config.summary.normalize:
            norm = float(np.linalg.norm(summary))
            if norm > self.config.summary.eps:
                summary = summary / norm
        return ControllerSummary(values=summary, name=name)

    def observe(
        self,
        predicted: np.ndarray | float | list[float] | tuple[float, ...],
        actual: np.ndarray | float | list[float] | tuple[float, ...],
        *,
        step: int | None = None,
        name: str | None = None,
    ) -> PredictionState:
        predicted_vector = self._coerce_vector(predicted)
        actual_vector = self._coerce_vector(actual)
        if predicted_vector.shape != actual_vector.shape:
            raise ValueError("predicted and actual must share the same shape")

        residual = actual_vector - predicted_vector
        surprise = np.abs(residual)
        summary_signal = residual if self.config.feature_mode == "residual" else surprise
        summary = self._summary_from_signal(summary_signal, name=name)
        return PredictionState(
            predicted=predicted_vector,
            actual=actual_vector,
            residual=residual,
            surprise=surprise,
            summary=summary,
            step=step,
        )

    def feature_vector(self, state: PredictionState) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    [
                        float(np.mean(state.predicted)),
                        float(np.mean(state.actual)),
                        float(np.mean(state.residual)),
                        float(np.mean(np.abs(state.residual))),
                        float(np.mean(np.square(state.residual))),
                        float(np.mean(np.square(state.surprise))),
                    ],
                    dtype=np.float64,
                ),
                state.summary.values,
            ]
        )

    @property
    def feature_dim(self) -> int:
        return 8


__all__ = [
    "PredictionState",
    "PredictiveSurpriseConfig",
    "PredictiveSurpriseController",
    "SummaryMode",
]
