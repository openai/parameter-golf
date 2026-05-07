from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .codecs import ensure_tokens
from .metrics import bits_per_byte_from_probabilities
from .readout import RidgeReadout
from .substrates import TokenSubstrate


@dataclass(frozen=True)
class ExpertFitReport:
    sequences: int
    tokens: int
    bits_per_byte: float


@dataclass(frozen=True)
class ExpertScore:
    tokens: int
    bits_per_byte: float


class FrozenReadoutExpert:
    def __init__(
        self,
        *,
        name: str,
        substrate: TokenSubstrate,
        feature_dim: int,
        vocabulary_size: int,
        feature_fn: Callable[[np.ndarray, np.ndarray | None], np.ndarray],
        alpha: float = 1e-3,
    ):
        self.name = name
        self.substrate = substrate
        self.feature_fn = feature_fn
        self.vocabulary_size = vocabulary_size
        self.readout = RidgeReadout(
            input_dim=feature_dim,
            output_dim=vocabulary_size,
            alpha=alpha,
        )

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

    def _trace(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")
        state = self.substrate.initial_state()
        features: list[np.ndarray] = []
        for token in tokens[:-1]:
            previous_state = state.copy()
            state = self.substrate.step(state, int(token))
            features.append(self.feature_fn(state, previous_state))
        return np.vstack(features), tokens[1:].astype(np.int64, copy=False)

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> ExpertFitReport:
        sequences = self._coerce_sequences(data)
        feature_batches = []
        target_batches = []
        total_tokens = 0
        for sequence in sequences:
            features, targets = self._trace(sequence)
            feature_batches.append(features)
            target_batches.append(targets)
            total_tokens += int(ensure_tokens(sequence).size)
        design = np.concatenate(feature_batches, axis=0)
        labels = np.concatenate(target_batches, axis=0)
        self.readout.fit(design, labels)
        return ExpertFitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            bits_per_byte=bits_per_byte_from_probabilities(self.readout.probabilities(design), labels),
        )

    def sequence_probabilities(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        features, targets = self._trace(ensure_tokens(sequence))
        return self.readout.probabilities(features), targets

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> ExpertScore:
        probabilities, targets = self.sequence_probabilities(sequence)
        return ExpertScore(
            tokens=int(targets.size + 1),
            bits_per_byte=bits_per_byte_from_probabilities(probabilities, targets),
        )

    def predict_proba(self, prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")
        state = self.substrate.initial_state()
        feature = None
        for token in tokens:
            previous_state = state.copy()
            state = self.substrate.step(state, int(token))
            feature = self.feature_fn(state, previous_state)
        assert feature is not None
        return self.readout.probabilities(feature[None, :])[0]


__all__ = [
    "ExpertFitReport",
    "ExpertScore",
    "FrozenReadoutExpert",
]
