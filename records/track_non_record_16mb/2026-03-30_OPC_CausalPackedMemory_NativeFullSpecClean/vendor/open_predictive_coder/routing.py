from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .control import ControllerSummary

RoutingMode = Literal["equal", "static", "projection"]


@dataclass(frozen=True)
class RoutingConfig:
    mode: RoutingMode = "equal"
    static_logits: tuple[float, ...] = ()
    projection_weights: tuple[float, ...] = ()
    route_biases: tuple[float, ...] = ()
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be > 0")


@dataclass(frozen=True)
class RoutingDecision:
    mode: RoutingMode
    route_names: tuple[str, ...]
    logits: np.ndarray
    weights: np.ndarray
    selected_index: int

    def __post_init__(self) -> None:
        logits = np.asarray(self.logits, dtype=np.float64).reshape(-1)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if logits.size < 1:
            raise ValueError("RoutingDecision requires at least one logit")
        if logits.shape != weights.shape:
            raise ValueError("logits and weights must have the same shape")
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "weights", weights)
        if not 0 <= self.selected_index < weights.size:
            raise ValueError("selected_index out of range")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return exp / total


class SummaryRouter:
    def __init__(self, config: RoutingConfig | None = None):
        self.config = config or RoutingConfig()

    def route(
        self,
        summaries: Sequence[ControllerSummary],
        *,
        names: Sequence[str] | None = None,
    ) -> RoutingDecision:
        if not summaries:
            raise ValueError("route requires at least one summary")
        dim = summaries[0].dim
        for summary in summaries[1:]:
            if summary.dim != dim:
                raise ValueError("all controller summaries must share the same dimension")

        route_names = tuple(names) if names is not None else tuple(
            summary.name if summary.name is not None else f"branch_{index}"
            for index, summary in enumerate(summaries)
        )
        if len(route_names) != len(summaries):
            raise ValueError("names must match the number of summaries")

        if self.config.mode == "equal":
            logits = np.zeros(len(summaries), dtype=np.float64)
        elif self.config.mode == "static":
            logits = np.asarray(self.config.static_logits, dtype=np.float64)
            if logits.size == 0:
                logits = np.zeros(len(summaries), dtype=np.float64)
            if logits.size != len(summaries):
                raise ValueError("static_logits must match the number of summaries")
        elif self.config.mode == "projection":
            weights = np.asarray(self.config.projection_weights, dtype=np.float64)
            if weights.size == 0:
                raise ValueError("projection mode requires projection_weights")
            if weights.size != dim:
                raise ValueError("projection_weights must match the summary dimension")
            route_biases = np.asarray(self.config.route_biases, dtype=np.float64)
            if route_biases.size == 0:
                route_biases = np.zeros(len(summaries), dtype=np.float64)
            if route_biases.size != len(summaries):
                raise ValueError("route_biases must match the number of summaries")
            logits = np.asarray(
                [float(summary.values @ weights) + float(route_biases[index]) for index, summary in enumerate(summaries)],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unknown routing mode: {self.config.mode}")

        route_weights = _softmax(logits / self.config.temperature)
        selected_index = int(np.argmax(route_weights))
        return RoutingDecision(
            mode=self.config.mode,
            route_names=route_names,
            logits=logits,
            weights=route_weights,
            selected_index=selected_index,
        )


__all__ = [
    "RoutingConfig",
    "RoutingDecision",
    "RoutingMode",
    "SummaryRouter",
]
