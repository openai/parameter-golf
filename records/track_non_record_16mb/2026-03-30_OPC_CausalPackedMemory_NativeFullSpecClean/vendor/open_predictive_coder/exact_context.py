from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .codecs import ensure_tokens
from .metrics import softmax


@dataclass(frozen=True)
class ExactContextConfig:
    vocabulary_size: int = 256
    max_order: int = 3
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.max_order < 1:
            raise ValueError("max_order must be >= 1")
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0")


@dataclass(frozen=True)
class ExactContextPrediction:
    name: str
    order: int
    context: tuple[int, ...]
    probabilities: np.ndarray
    support: float
    total: float


@dataclass(frozen=True)
class ExactContextFitReport:
    sequences: int
    tokens: int
    contexts_by_order: tuple[int, ...]


@dataclass(frozen=True)
class SupportMixConfig:
    base_bias: float = 2.0
    expert_bias: float = -1.0
    support_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.support_scale < 0.0:
            raise ValueError("support_scale must be >= 0")


@dataclass(frozen=True)
class SupportBlend:
    probabilities: np.ndarray
    component_names: tuple[str, ...]
    weights: np.ndarray
    supports: np.ndarray


class ExactContextMemory:
    def __init__(self, config: ExactContextConfig | None = None):
        self.config = config or ExactContextConfig()
        self._unigram = np.zeros(self.config.vocabulary_size, dtype=np.float64)
        self._tables = [dict[tuple[int, ...], np.ndarray]() for _ in range(self.config.max_order)]

    @staticmethod
    def _coerce_sequences(
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> tuple[np.ndarray, ...]:
        if isinstance(data, (str, bytes, bytearray, memoryview, np.ndarray)):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence) and data and all(isinstance(item, int) for item in data):
            return (ensure_tokens(data),)
        if isinstance(data, Sequence):
            return tuple(ensure_tokens(item) for item in data)
        return (ensure_tokens(data),)

    def clear(self) -> None:
        self._unigram.fill(0.0)
        self._tables = [dict[tuple[int, ...], np.ndarray]() for _ in range(self.config.max_order)]

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ExactContextFitReport:
        self.clear()
        sequences = self._coerce_sequences(data)
        total_tokens = 0
        for sequence in sequences:
            tokens = ensure_tokens(sequence).astype(np.int64, copy=False)
            if tokens.size == 0:
                continue
            total_tokens += int(tokens.size)
            self._unigram += np.bincount(tokens, minlength=self.config.vocabulary_size)
            for index in range(tokens.size):
                target = int(tokens[index])
                max_order = min(self.config.max_order, index)
                for order in range(1, max_order + 1):
                    context = tuple(int(token) for token in tokens[index - order : index])
                    table = self._tables[order - 1]
                    counts = table.get(context)
                    if counts is None:
                        counts = np.zeros(self.config.vocabulary_size, dtype=np.float64)
                        table[context] = counts
                    counts[target] += 1.0
        contexts_by_order = tuple(len(table) for table in self._tables)
        return ExactContextFitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            contexts_by_order=contexts_by_order,
        )

    def _normalize(self, distribution: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(distribution, dtype=np.float64), 1e-12, None)
        total = float(np.sum(clipped))
        if total <= 0.0:
            return np.full(self.config.vocabulary_size, 1.0 / self.config.vocabulary_size, dtype=np.float64)
        return clipped / total

    def _smooth_counts(self, counts: np.ndarray) -> np.ndarray:
        alpha = self.config.alpha
        total = float(np.sum(counts))
        if total == 0.0 and alpha == 0.0:
            return np.full(self.config.vocabulary_size, 1.0 / self.config.vocabulary_size, dtype=np.float64)
        probs = (counts + (alpha / self.config.vocabulary_size)) / (total + alpha)
        return self._normalize(probs)

    def unigram_probabilities(self) -> np.ndarray:
        return self._smooth_counts(self._unigram)

    def experts(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> tuple[ExactContextPrediction, ...]:
        tokens = ensure_tokens(context).astype(np.int64, copy=False)
        predictions: list[ExactContextPrediction] = []
        max_order = min(self.config.max_order, int(tokens.size))
        for order in range(1, max_order + 1):
            key = tuple(int(token) for token in tokens[-order:])
            counts = self._tables[order - 1].get(key)
            if counts is None:
                counts = np.zeros(self.config.vocabulary_size, dtype=np.float64)
            total = float(np.sum(counts))
            predictions.append(
                ExactContextPrediction(
                    name=f"exact{order}",
                    order=order,
                    context=key,
                    probabilities=self._smooth_counts(counts),
                    support=float(np.log1p(total)),
                    total=total,
                )
            )
        return tuple(predictions)

    def predictive_distribution(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        predictions = self.experts(context)
        for prediction in reversed(predictions):
            if prediction.total > 0.0:
                return prediction.probabilities
        return self.unigram_probabilities()


class SupportWeightedMixer:
    def __init__(self, config: SupportMixConfig | None = None):
        self.config = config or SupportMixConfig()

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
        return clipped / np.sum(clipped)

    def mix(
        self,
        *,
        base_probs: np.ndarray | None = None,
        experts: Sequence[ExactContextPrediction] = (),
        base_name: str = "base",
        base_support: float = 1.0,
    ) -> SupportBlend:
        component_names: list[str] = []
        component_probs: list[np.ndarray] = []
        supports: list[float] = []
        biases: list[float] = []

        if base_probs is not None:
            component_names.append(base_name)
            component_probs.append(self._normalize(base_probs))
            supports.append(float(base_support))
            biases.append(self.config.base_bias)

        for expert in experts:
            component_names.append(expert.name)
            component_probs.append(self._normalize(expert.probabilities))
            supports.append(float(expert.support))
            biases.append(self.config.expert_bias)

        if not component_probs:
            raise ValueError("mix requires at least one component")

        logits = np.asarray(biases, dtype=np.float64) + self.config.support_scale * np.asarray(supports, dtype=np.float64)
        weights = softmax(logits[None, :], axis=-1)[0]
        stacked = np.stack(component_probs, axis=0)
        mixed = np.sum(weights[:, None] * stacked, axis=0)
        return SupportBlend(
            probabilities=self._normalize(mixed),
            component_names=tuple(component_names),
            weights=weights,
            supports=np.asarray(supports, dtype=np.float64),
        )


__all__ = [
    "ExactContextConfig",
    "ExactContextFitReport",
    "ExactContextMemory",
    "ExactContextPrediction",
    "SupportBlend",
    "SupportMixConfig",
    "SupportWeightedMixer",
]
