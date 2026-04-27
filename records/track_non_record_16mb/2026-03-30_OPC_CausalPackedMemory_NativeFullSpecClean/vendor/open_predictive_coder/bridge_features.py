from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probability_diagnostics import (
    ProbabilityDiagnosticsConfig,
    probability_diagnostics,
)


@dataclass(frozen=True)
class BridgeFeatureConfig:
    candidate_count: int = 4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")


@dataclass(frozen=True)
class BridgeFeatureArrays:
    entropy: np.ndarray
    peak: np.ndarray
    candidate4: np.ndarray
    agreement: np.ndarray
    agreement_mass: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "entropy": self.entropy,
            "peak": self.peak,
            "candidate4": self.candidate4,
            "agreement": self.agreement,
            "agreement_mass": self.agreement_mass,
        }


def _coerce_probabilities(probabilities: np.ndarray | list[float] | tuple[float, ...], vocab_size: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("probability arrays must have at least one dimension")
    if array.shape[-1] != vocab_size:
        raise ValueError("last dimension must match vocab_size")
    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative")
    totals = array.sum(axis=-1, keepdims=True)
    normalized = np.divide(
        array,
        totals,
        out=np.full_like(array, 1.0 / float(vocab_size)),
        where=totals > 0.0,
    )
    return normalized


def bridge_feature_arrays(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
    vocab_size: int,
    *,
    config: BridgeFeatureConfig | None = None,
) -> BridgeFeatureArrays:
    config = config or BridgeFeatureConfig()
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1")

    base = _coerce_probabilities(base_probs, vocab_size)
    proxy = _coerce_probabilities(proxy_probs, vocab_size)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")

    diagnostics = probability_diagnostics(
        base,
        proxy,
        config=ProbabilityDiagnosticsConfig(
            top_k=config.candidate_count,
            epsilon=config.epsilon,
        ),
    )

    return BridgeFeatureArrays(
        entropy=np.asarray(diagnostics.entropy, dtype=np.float64),
        peak=np.asarray(diagnostics.peak, dtype=np.float64),
        candidate4=np.asarray(diagnostics.top_k_mass, dtype=np.float64),
        agreement=np.asarray(diagnostics.overlap, dtype=np.float64),
        agreement_mass=np.asarray(diagnostics.shared_top_k_mass, dtype=np.float64),
    )


__all__ = [
    "BridgeFeatureArrays",
    "BridgeFeatureConfig",
    "bridge_feature_arrays",
]
