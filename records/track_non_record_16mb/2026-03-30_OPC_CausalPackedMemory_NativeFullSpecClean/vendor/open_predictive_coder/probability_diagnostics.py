from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProbabilityDiagnosticsConfig:
    top_k: int = 4
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")


@dataclass(frozen=True)
class ProbabilityDiagnostics:
    entropy: np.ndarray
    peak: np.ndarray
    top_k_mass: np.ndarray
    overlap: np.ndarray
    top1_agreement: np.ndarray
    shared_top_k_mass: np.ndarray
    top2_margin: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "entropy": self.entropy,
            "peak": self.peak,
            "top_k_mass": self.top_k_mass,
            "overlap": self.overlap,
            "top1_agreement": self.top1_agreement,
            "shared_top_k_mass": self.shared_top_k_mass,
            "top2_margin": self.top2_margin,
        }


def _coerce_probabilities(probabilities: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("probability arrays must have at least one dimension")
    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative")
    return array


def _normalize_probabilities(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    vocab_size = probabilities.shape[-1]
    totals = np.sum(probabilities, axis=-1, keepdims=True)
    return np.divide(
        probabilities,
        totals,
        out=np.full_like(probabilities, 1.0 / float(vocab_size)),
        where=totals > epsilon,
    )


def normalized_entropy(probabilities: np.ndarray | list[float] | tuple[float, ...], *, epsilon: float = 1e-12) -> np.ndarray:
    array = _normalize_probabilities(_coerce_probabilities(probabilities), epsilon=epsilon)
    vocab_size = array.shape[-1]
    log_vocab = np.log(float(vocab_size))
    entropy = -np.sum(array * np.log(array + epsilon), axis=-1)
    return np.asarray(entropy / log_vocab if log_vocab > 0.0 else np.zeros_like(entropy), dtype=np.float64)


def top1_peak(probabilities: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = _normalize_probabilities(_coerce_probabilities(probabilities), epsilon=1e-12)
    return np.asarray(np.max(array, axis=-1), dtype=np.float64)


def top_k_mass(
    probabilities: np.ndarray | list[float] | tuple[float, ...],
    *,
    top_k: int = 4,
) -> np.ndarray:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    array = _normalize_probabilities(_coerce_probabilities(probabilities), epsilon=1e-12)
    k = min(top_k, array.shape[-1])
    if k == array.shape[-1]:
        mass = np.sum(array, axis=-1)
    else:
        partition = np.partition(array, kth=-k, axis=-1)
        mass = np.sum(partition[..., -k:], axis=-1)
    return np.asarray(mass, dtype=np.float64)


def top2_margin(probabilities: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    array = _normalize_probabilities(_coerce_probabilities(probabilities), epsilon=1e-12)
    if array.shape[-1] < 2:
        return np.zeros(array.shape[:-1], dtype=np.float64)
    partition = np.partition(array, kth=-2, axis=-1)
    top2 = np.sort(partition[..., -2:], axis=-1)
    return np.asarray(top2[..., 1] - top2[..., 0], dtype=np.float64)


def overlap_mass(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    base = _normalize_probabilities(_coerce_probabilities(base_probs), epsilon=1e-12)
    proxy = _normalize_probabilities(_coerce_probabilities(proxy_probs), epsilon=1e-12)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")
    overlap = np.sum(np.minimum(base, proxy), axis=-1)
    return np.asarray(overlap, dtype=np.float64)


def top1_agreement(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    base = _normalize_probabilities(_coerce_probabilities(base_probs), epsilon=1e-12)
    proxy = _normalize_probabilities(_coerce_probabilities(proxy_probs), epsilon=1e-12)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")
    agreement = np.argmax(base, axis=-1) == np.argmax(proxy, axis=-1)
    return np.asarray(agreement, dtype=np.float64)


def shared_top_k_mass(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
    *,
    top_k: int = 4,
) -> np.ndarray:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    base = _normalize_probabilities(_coerce_probabilities(base_probs), epsilon=1e-12)
    proxy = _normalize_probabilities(_coerce_probabilities(proxy_probs), epsilon=1e-12)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")

    k = min(top_k, base.shape[-1])
    if k == base.shape[-1]:
        shared_mask = np.ones_like(base, dtype=bool)
    else:
        base_indices = np.argpartition(base, kth=-k, axis=-1)[..., -k:]
        proxy_indices = np.argpartition(proxy, kth=-k, axis=-1)[..., -k:]
        base_mask = np.zeros_like(base, dtype=bool)
        proxy_mask = np.zeros_like(proxy, dtype=bool)
        np.put_along_axis(base_mask, base_indices, True, axis=-1)
        np.put_along_axis(proxy_mask, proxy_indices, True, axis=-1)
        shared_mask = base_mask & proxy_mask

    mass = 0.5 * (np.sum(base * shared_mask, axis=-1) + np.sum(proxy * shared_mask, axis=-1))
    return np.asarray(mass, dtype=np.float64)


def probability_diagnostics(
    base_probs: np.ndarray | list[float] | tuple[float, ...],
    proxy_probs: np.ndarray | list[float] | tuple[float, ...],
    *,
    config: ProbabilityDiagnosticsConfig | None = None,
) -> ProbabilityDiagnostics:
    config = config or ProbabilityDiagnosticsConfig()
    base = _normalize_probabilities(_coerce_probabilities(base_probs), epsilon=config.epsilon)
    proxy = _normalize_probabilities(_coerce_probabilities(proxy_probs), epsilon=config.epsilon)
    if base.shape != proxy.shape:
        raise ValueError("base_probs and proxy_probs must have the same shape")

    entropy = 0.5 * (normalized_entropy(base, epsilon=config.epsilon) + normalized_entropy(proxy, epsilon=config.epsilon))
    peak = 0.5 * (top1_peak(base) + top1_peak(proxy))
    mass = 0.5 * (top_k_mass(base, top_k=config.top_k) + top_k_mass(proxy, top_k=config.top_k))
    overlap = overlap_mass(base, proxy)
    agreement = top1_agreement(base, proxy)
    shared_mass = shared_top_k_mass(base, proxy, top_k=config.top_k)
    margin = 0.5 * (top2_margin(base) + top2_margin(proxy))

    return ProbabilityDiagnostics(
        entropy=np.asarray(entropy, dtype=np.float64),
        peak=np.asarray(peak, dtype=np.float64),
        top_k_mass=np.asarray(mass, dtype=np.float64),
        overlap=np.asarray(overlap, dtype=np.float64),
        top1_agreement=np.asarray(agreement, dtype=np.float64),
        shared_top_k_mass=np.asarray(shared_mass, dtype=np.float64),
        top2_margin=np.asarray(margin, dtype=np.float64),
    )


__all__ = [
    "ProbabilityDiagnostics",
    "ProbabilityDiagnosticsConfig",
    "normalized_entropy",
    "overlap_mass",
    "probability_diagnostics",
    "shared_top_k_mass",
    "top1_agreement",
    "top1_peak",
    "top2_margin",
    "top_k_mass",
]
