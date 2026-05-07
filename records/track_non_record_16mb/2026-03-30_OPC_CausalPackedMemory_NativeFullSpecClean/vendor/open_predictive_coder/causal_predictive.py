from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .artifacts import (
    ArtifactAccounting,
    ArtifactMetadata,
    make_artifact_accounting,
    make_replay_span,
)
from .codecs import ensure_tokens
from .exact_context import (
    ExactContextFitReport,
    ExactContextMemory,
    ExactContextPrediction,
    SupportWeightedMixer,
)
from .experts import ExpertFitReport, FrozenReadoutExpert
from .metrics import bits_per_byte_from_probabilities
from .ngram_memory import NgramMemory, NgramMemoryReport


@dataclass(frozen=True)
class CausalPredictiveFitReport:
    sequences: int
    tokens: int
    train_bits_per_byte: float
    exact_fit: ExactContextFitReport
    expert_fits: tuple[ExpertFitReport, ...]
    ngram_fit: NgramMemoryReport | None
    accounting: ArtifactAccounting

    @property
    def bits_per_byte(self) -> float:
        return self.train_bits_per_byte


@dataclass(frozen=True)
class CausalPredictiveScore:
    tokens: int
    bits_per_byte: float
    exact_bits_per_byte: float
    auxiliary_bits_per_byte: float | None
    ngram_bits_per_byte: float | None
    exact_support: float
    accounting: ArtifactAccounting


@dataclass(frozen=True)
class _ComponentPrediction:
    name: str
    probabilities: np.ndarray
    support: float


class CausalPredictiveAdapter:
    def __init__(
        self,
        exact_context: ExactContextMemory | None = None,
        *,
        experts: Sequence[FrozenReadoutExpert] = (),
        ngram_memory: NgramMemory | None = None,
        mixer: SupportWeightedMixer | None = None,
        artifact_name: str = "causal_predictive",
        metadata: ArtifactMetadata | None = None,
    ):
        self.exact_context = exact_context or ExactContextMemory()
        self.experts = tuple(experts)
        self.ngram_memory = ngram_memory
        self.mixer = mixer or SupportWeightedMixer()
        self.artifact_name = artifact_name
        self.metadata = metadata or ArtifactMetadata()
        self._last_fit_accounting = make_artifact_accounting(
            self.artifact_name,
            0,
            0,
            metadata=self.metadata,
            tokens=0,
            supported_tokens=0,
            exact_orders=0,
        )
        self._validate_experts()
        self._validate_ngram_memory()

    def _validate_experts(self) -> None:
        vocab_size = self.exact_context.config.vocabulary_size
        for expert in self.experts:
            if expert.vocabulary_size != vocab_size:
                raise ValueError(
                    f"expert {expert.name!r} has vocabulary_size={expert.vocabulary_size}, "
                    f"expected {vocab_size}"
                )

    def _validate_ngram_memory(self) -> None:
        if self.ngram_memory is None:
            return
        vocab_size = self.exact_context.config.vocabulary_size
        if self.ngram_memory.config.vocabulary_size != vocab_size:
            raise ValueError(
                "ngram_memory vocabulary_size must match exact_context vocabulary_size"
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

    @staticmethod
    def _normalize(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
        return clipped / np.sum(clipped)

    def _auxiliary_predictions(self, prompt: np.ndarray) -> tuple[_ComponentPrediction, ...]:
        components: list[_ComponentPrediction] = []
        for expert in self.experts:
            probabilities = self._normalize(expert.predict_proba(prompt))
            support = float(np.max(probabilities))
            components.append(
                _ComponentPrediction(
                    name=expert.name,
                    probabilities=probabilities,
                    support=support,
                )
            )
        return tuple(components)

    def _ngram_prediction(self, prompt: np.ndarray) -> ExactContextPrediction | None:
        if self.ngram_memory is None:
            return None

        tokens = ensure_tokens(prompt).astype(np.int64, copy=False)
        if tokens.size == 0:
            probabilities = self.ngram_memory.unigram_probs()
            context: tuple[int, ...] = ()
            order = 0
        elif tokens.size == 1:
            probabilities = self.ngram_memory.bigram_probs(int(tokens[-1]))
            context = (int(tokens[-1]),)
            order = 1
        else:
            probabilities = self.ngram_memory.trigram_probs(int(tokens[-2]), int(tokens[-1]))
            context = (int(tokens[-2]), int(tokens[-1]))
            order = 2

        support = float(np.max(probabilities))
        return ExactContextPrediction(
            name="ngram",
            order=order,
            context=context,
            probabilities=np.asarray(probabilities, dtype=np.float64),
            support=support,
            total=support,
        )

    @staticmethod
    def _pack_prediction(
        name: str,
        probabilities: np.ndarray,
        support: float,
        *,
        order: int = 0,
        context: tuple[int, ...] = (),
    ) -> ExactContextPrediction:
        return ExactContextPrediction(
            name=name,
            order=order,
            context=context,
            probabilities=np.asarray(probabilities, dtype=np.float64),
            support=float(support),
            total=float(max(support, 0.0)),
        )

    def _blend_prefix(self, prefix: np.ndarray) -> tuple[np.ndarray, float]:
        base_probs = self.exact_context.predictive_distribution(prefix)
        exact_predictions = self.exact_context.experts(prefix)
        base_support = max((prediction.support for prediction in exact_predictions), default=0.0)
        aux_predictions = self._auxiliary_predictions(prefix)
        ngram_prediction = self._ngram_prediction(prefix)
        packed_aux = tuple(
            self._pack_prediction(component.name, component.probabilities, component.support)
            for component in aux_predictions
        )
        ngram_expert: tuple[ExactContextPrediction, ...] = ()
        if ngram_prediction is not None:
            ngram_expert = (ngram_prediction,)
        blend = self.mixer.mix(
            base_probs=base_probs,
            experts=tuple(exact_predictions) + packed_aux + ngram_expert,
            base_name="exact_context",
            base_support=base_support,
        )
        return blend.probabilities, base_support

    def accounting(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | None = None,
    ) -> ArtifactAccounting:
        if sequence is None:
            return self._last_fit_accounting

        tokens = ensure_tokens(sequence)
        if tokens.size == 0:
            return make_artifact_accounting(
                self.artifact_name,
                0,
                0,
                metadata=self.metadata,
                tokens=0,
                supported_tokens=0,
                exact_orders=0,
            )

        replay_spans = []
        current_start: int | None = None
        supported_tokens = 0
        exact_orders = 0

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            predictions = self.exact_context.experts(prefix)
            support = max((prediction.total for prediction in predictions), default=0.0)
            exact_orders = max(exact_orders, len(predictions))
            if support > 0.0:
                supported_tokens += 1
                if current_start is None:
                    current_start = index
            elif current_start is not None:
                replay_spans.append(
                    make_replay_span(
                        current_start,
                        index,
                        label="exact_context",
                        supported_tokens=index - current_start,
                        exact_orders=exact_orders,
                    )
                )
                current_start = None

        if current_start is not None:
            replay_spans.append(
                make_replay_span(
                    current_start,
                    int(tokens.size),
                    label="exact_context",
                    supported_tokens=int(tokens.size) - current_start,
                    exact_orders=exact_orders,
                )
            )

        return make_artifact_accounting(
            self.artifact_name,
            int(tokens.size),
            supported_tokens,
            replay_spans=tuple(replay_spans),
            metadata=self.metadata,
            tokens=int(tokens.size),
            supported_tokens=supported_tokens,
            exact_orders=exact_orders,
        )

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> CausalPredictiveFitReport:
        sequences = self._coerce_sequences(data)
        exact_fit = self.exact_context.fit(sequences)
        ngram_fit = self.ngram_memory.fit(sequences) if self.ngram_memory is not None else None

        fit_sequences = tuple(sequence for sequence in sequences if ensure_tokens(sequence).size >= 2)
        expert_fits: list[ExpertFitReport] = []
        for expert in self.experts:
            if fit_sequences:
                expert_fits.append(expert.fit(fit_sequences))

        total_tokens = 0
        total_effective_tokens = 0
        weighted_bits = 0.0
        artifact_bytes = 0
        replay_bytes = 0
        replay_spans = []
        offset = 0

        for sequence in sequences:
            tokens = ensure_tokens(sequence)
            score = self.score(tokens)
            total_tokens += int(tokens.size)
            effective_tokens = max(int(tokens.size) - 1, 0)
            total_effective_tokens += effective_tokens
            weighted_bits += score.bits_per_byte * effective_tokens
            accounting = score.accounting
            artifact_bytes += accounting.artifact_bytes
            replay_bytes += accounting.replay_bytes
            replay_spans.extend(
                make_replay_span(
                    span.start + offset,
                    span.stop + offset,
                    label=span.label,
                    metadata=span.metadata,
                )
                for span in accounting.replay_spans
            )
            offset += int(tokens.size)

        train_bits = 0.0 if total_effective_tokens == 0 else weighted_bits / total_effective_tokens
        accounting = make_artifact_accounting(
            self.artifact_name,
            artifact_bytes,
            replay_bytes,
            replay_spans=tuple(replay_spans),
            metadata=self.metadata,
            sequences=len(sequences),
            tokens=artifact_bytes,
            supported_tokens=replay_bytes,
            exact_orders=self.exact_context.config.max_order,
        )
        self._last_fit_accounting = accounting
        return CausalPredictiveFitReport(
            sequences=len(sequences),
            tokens=total_tokens,
            train_bits_per_byte=train_bits,
            exact_fit=exact_fit,
            expert_fits=tuple(expert_fits),
            ngram_fit=ngram_fit,
            accounting=accounting,
        )

    def predict_proba(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        tokens = ensure_tokens(prompt)
        if tokens.size < 1:
            raise ValueError("prompt must contain at least one token")
        probabilities, _ = self._blend_prefix(tokens)
        return probabilities

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> CausalPredictiveScore:
        tokens = ensure_tokens(sequence)
        if tokens.size < 2:
            raise ValueError("sequence must contain at least two tokens")

        exact_rows: list[np.ndarray] = []
        final_rows: list[np.ndarray] = []
        exact_support = 0.0
        ngram_rows: list[np.ndarray] = []

        for index in range(1, tokens.size):
            prefix = tokens[:index]
            exact_probs = self.exact_context.predictive_distribution(prefix)
            exact_predictions = self.exact_context.experts(prefix)
            base_support = max((prediction.support for prediction in exact_predictions), default=0.0)
            aux_predictions = self._auxiliary_predictions(prefix)
            ngram_prediction = self._ngram_prediction(prefix)
            packed_aux = tuple(
                self._pack_prediction(component.name, component.probabilities, component.support)
                for component in aux_predictions
            )
            ngram_expert: tuple[ExactContextPrediction, ...] = ()
            if ngram_prediction is not None:
                ngram_rows.append(np.asarray(ngram_prediction.probabilities, dtype=np.float64))
                ngram_expert = (ngram_prediction,)
            exact_blend = self.mixer.mix(
                base_probs=exact_probs,
                experts=tuple(exact_predictions),
                base_name="exact_context",
                base_support=base_support,
            )
            final_blend = self.mixer.mix(
                base_probs=exact_probs,
                experts=tuple(exact_predictions) + packed_aux + ngram_expert,
                base_name="exact_context",
                base_support=base_support,
            )
            exact_rows.append(exact_blend.probabilities)
            final_rows.append(final_blend.probabilities)
            exact_support = base_support

        targets = tokens[1:].astype(np.int64, copy=False)
        exact_bpb = bits_per_byte_from_probabilities(np.vstack(exact_rows), targets)
        final_bpb = bits_per_byte_from_probabilities(np.vstack(final_rows), targets)
        ngram_bpb = (
            None
            if not ngram_rows
            else bits_per_byte_from_probabilities(np.vstack(ngram_rows), targets[: len(ngram_rows)])
        )
        auxiliary_bpb = (
            None
            if not self.experts
            else float(np.mean([expert.score(tokens).bits_per_byte for expert in self.experts]))
        )
        return CausalPredictiveScore(
            tokens=int(tokens.size),
            bits_per_byte=final_bpb,
            exact_bits_per_byte=exact_bpb,
            auxiliary_bits_per_byte=auxiliary_bpb,
            ngram_bits_per_byte=ngram_bpb,
            exact_support=exact_support,
            accounting=self.accounting(tokens),
        )


__all__ = [
    "CausalPredictiveAdapter",
    "CausalPredictiveFitReport",
    "CausalPredictiveScore",
]
