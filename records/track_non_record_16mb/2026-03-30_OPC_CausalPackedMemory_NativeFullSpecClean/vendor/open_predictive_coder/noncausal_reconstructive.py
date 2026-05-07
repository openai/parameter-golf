from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .artifacts import ArtifactAccounting, ArtifactMetadata, make_artifact_accounting
from .bidirectional_context import BidirectionalContextConfig, BidirectionalContextProbe, BidirectionalContextStats
from .codecs import ByteCodec, ensure_tokens
from .exact_context import ExactContextConfig, ExactContextFitReport, ExactContextMemory
from .span_selection import ReplaySpan, SpanSelectionConfig, replay_spans_from_scores


def _coerce_tokens(
    data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
) -> np.ndarray:
    return ensure_tokens(data).astype(np.uint8, copy=False)


def _reverse_tokens(tokens: np.ndarray) -> np.ndarray:
    return np.asarray(tokens, dtype=np.uint8)[::-1].copy()


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return probs
    clipped = np.clip(probs, 1e-12, None)
    total = float(np.sum(clipped))
    if total <= 0.0:
        return np.full_like(clipped, 1.0 / clipped.size, dtype=np.float64)
    return clipped / total


def _blend_probabilities(left: np.ndarray, right: np.ndarray, temperature: float) -> np.ndarray:
    left = np.clip(np.asarray(left, dtype=np.float64), 1e-12, None)
    right = np.clip(np.asarray(right, dtype=np.float64), 1e-12, None)
    logits = 0.5 * (np.log(left) + np.log(right))
    logits = logits / temperature
    logits = logits - float(np.max(logits))
    return _normalize(np.exp(logits))


@dataclass(frozen=True)
class NoncausalReconstructiveConfig:
    vocabulary_size: int = 256
    exact_max_order: int = 3
    exact_alpha: float = 0.05
    bidirectional_left_order: int = 2
    bidirectional_right_order: int = 2
    blend_temperature: float = 1.0
    agreement_threshold: float = 0.75
    replay_threshold: float = 0.55
    min_replay_span: int = 2

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.exact_max_order < 1:
            raise ValueError("exact_max_order must be >= 1")
        if self.exact_alpha < 0.0:
            raise ValueError("exact_alpha must be >= 0")
        if self.bidirectional_left_order < 0:
            raise ValueError("bidirectional_left_order must be >= 0")
        if self.bidirectional_right_order < 0:
            raise ValueError("bidirectional_right_order must be >= 0")
        if self.blend_temperature <= 0.0:
            raise ValueError("blend_temperature must be > 0")
        if not 0.0 <= self.agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold must be in [0, 1]")
        if not 0.0 <= self.replay_threshold <= 1.0:
            raise ValueError("replay_threshold must be in [0, 1]")
        if self.min_replay_span < 1:
            raise ValueError("min_replay_span must be >= 1")


@dataclass(frozen=True)
class NoncausalReconstructiveFitReport:
    forward: ExactContextFitReport
    reverse: ExactContextFitReport
    bidirectional_context: BidirectionalContextStats
    accounting: ArtifactAccounting


@dataclass(frozen=True)
class NoncausalReconstructiveTrace:
    tokens: int
    source_tokens: np.ndarray
    left_probs: np.ndarray
    right_probs: np.ndarray
    blended_probs: np.ndarray
    reconstructed_tokens: np.ndarray
    agreement_mask: np.ndarray
    replay_mask: np.ndarray
    replay_spans: tuple[ReplaySpan, ...]
    bidirectional_context: BidirectionalContextStats
    accounting: ArtifactAccounting

    @property
    def steps(self) -> int:
        return int(self.source_tokens.size)


@dataclass(frozen=True)
class NoncausalReconstructiveReport:
    tokens: int
    steps: int
    left_bits_per_byte: float
    right_bits_per_byte: float
    blended_bits_per_byte: float
    agreement_rate: float
    replay_rate: float
    replay_span_count: int
    reconstructed_text: str
    bidirectional_context: BidirectionalContextStats
    accounting: ArtifactAccounting

    @property
    def bits_per_byte(self) -> float:
        return float(self.blended_bits_per_byte)


class NoncausalReconstructiveAdapter:
    def __init__(
        self,
        config: NoncausalReconstructiveConfig | None = None,
        *,
        artifact_name: str = "noncausal_reconstructive",
        metadata: ArtifactMetadata | None = None,
    ):
        self.config = config or NoncausalReconstructiveConfig()
        exact_config = ExactContextConfig(
            vocabulary_size=self.config.vocabulary_size,
            max_order=self.config.exact_max_order,
            alpha=self.config.exact_alpha,
        )
        self.forward_memory = ExactContextMemory(exact_config)
        self.reverse_memory = ExactContextMemory(exact_config)
        self.bidirectional_probe = BidirectionalContextProbe(
            BidirectionalContextConfig(
                left_order=self.config.bidirectional_left_order,
                right_order=self.config.bidirectional_right_order,
            )
        )
        self.artifact_name = artifact_name
        self.metadata = metadata or ArtifactMetadata()
        self._last_fit_accounting = make_artifact_accounting(
            self.artifact_name,
            0,
            0,
            metadata=self.metadata,
            tokens=0,
            replay_positions=0,
        )

    @classmethod
    def build(cls, **kwargs: object) -> "NoncausalReconstructiveAdapter":
        return cls(NoncausalReconstructiveConfig(**kwargs))

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> NoncausalReconstructiveFitReport:
        tokens = _coerce_tokens(data)
        if tokens.size == 0:
            raise ValueError("data must contain at least one token")

        forward = self.forward_memory.fit(tokens)
        reverse = self.reverse_memory.fit(_reverse_tokens(tokens))
        bidirectional_context = self.bidirectional_probe.scan(tokens)
        replay_mask = np.asarray(bidirectional_context.candidate_sizes, dtype=np.int64) <= 1
        replay_spans = replay_spans_from_scores(
            replay_mask.astype(np.float64, copy=False),
            SpanSelectionConfig(threshold=0.5, min_span=self.config.min_replay_span, max_gap=0),
            label="replay",
        )
        accounting = make_artifact_accounting(
            self.artifact_name,
            int(tokens.size),
            int(np.sum(replay_mask)),
            metadata=self.metadata,
            tokens=int(tokens.size),
            replay_positions=int(np.sum(replay_mask)),
            replay_spans=replay_spans,
        )
        self._last_fit_accounting = accounting
        return NoncausalReconstructiveFitReport(
            forward=forward,
            reverse=reverse,
            bidirectional_context=bidirectional_context,
            accounting=accounting,
        )

    def _distributions_for_position(
        self,
        tokens: np.ndarray,
        position: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        left_context = tokens[:position]
        right_context = _reverse_tokens(tokens[position + 1 :])
        left_probs = self.forward_memory.predictive_distribution(left_context)
        right_probs = self.reverse_memory.predictive_distribution(right_context)
        blended = _blend_probabilities(left_probs, right_probs, self.config.blend_temperature)
        return left_probs, right_probs, blended

    def trace(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> NoncausalReconstructiveTrace:
        tokens = _coerce_tokens(sequence)
        if tokens.size == 0:
            raise ValueError("sequence must contain at least one token")

        bidirectional_context = self.bidirectional_probe.scan(tokens)
        left_rows: list[np.ndarray] = []
        right_rows: list[np.ndarray] = []
        blended_rows: list[np.ndarray] = []
        reconstructed: list[int] = []
        agreement_mask: list[bool] = []
        replay_mask: list[bool] = []

        for position, token in enumerate(tokens):
            left_probs, right_probs, blended_probs = self._distributions_for_position(tokens, position)
            left_rows.append(left_probs)
            right_rows.append(right_probs)
            blended_rows.append(blended_probs)
            reconstructed.append(int(np.argmax(blended_probs)))
            agreement = int(np.argmax(left_probs)) == int(np.argmax(right_probs))
            agreement_mask.append(agreement)
            confidence = float(np.max(blended_probs))
            agreement_strength = 0.5 * (float(np.max(left_probs)) + float(np.max(right_probs)))
            replay_mask.append(
                agreement
                and confidence >= self.config.replay_threshold
                and agreement_strength >= self.config.agreement_threshold
            )

        replay_mask_array = np.asarray(replay_mask, dtype=bool)
        replay_spans = replay_spans_from_scores(
            replay_mask_array.astype(np.float64, copy=False),
            SpanSelectionConfig(threshold=0.5, min_span=self.config.min_replay_span, max_gap=0),
            label="replay",
        )
        accounting = make_artifact_accounting(
            self.artifact_name,
            int(tokens.size),
            int(np.sum(replay_mask_array)),
            replay_spans=replay_spans,
            metadata=self.metadata,
            tokens=int(tokens.size),
            replay_positions=int(np.sum(replay_mask_array)),
            replay_spans_count=len(replay_spans),
        )

        return NoncausalReconstructiveTrace(
            tokens=int(tokens.size),
            source_tokens=tokens,
            left_probs=np.vstack(left_rows),
            right_probs=np.vstack(right_rows),
            blended_probs=np.vstack(blended_rows),
            reconstructed_tokens=np.asarray(reconstructed, dtype=np.uint8),
            agreement_mask=np.asarray(agreement_mask, dtype=bool),
            replay_mask=replay_mask_array,
            replay_spans=replay_spans,
            bidirectional_context=bidirectional_context,
            accounting=accounting,
        )

    def reconstruct(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> np.ndarray:
        return self.trace(sequence).reconstructed_tokens

    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> NoncausalReconstructiveReport:
        trace = self.trace(sequence)
        source_tokens = trace.source_tokens
        targets = source_tokens
        row_indices = np.arange(targets.size)
        left_bits = -np.log2(np.clip(trace.left_probs[row_indices, targets], 1e-12, 1.0))
        right_bits = -np.log2(np.clip(trace.right_probs[row_indices, targets], 1e-12, 1.0))
        blended_bits = -np.log2(np.clip(trace.blended_probs[row_indices, targets], 1e-12, 1.0))

        reconstructed_text = ByteCodec.decode_text(trace.reconstructed_tokens)
        return NoncausalReconstructiveReport(
            tokens=trace.tokens,
            steps=trace.steps,
            left_bits_per_byte=float(np.mean(left_bits)),
            right_bits_per_byte=float(np.mean(right_bits)),
            blended_bits_per_byte=float(np.mean(blended_bits)),
            agreement_rate=float(np.mean(trace.agreement_mask)),
            replay_rate=float(np.mean(trace.replay_mask)),
            replay_span_count=len(trace.replay_spans),
            reconstructed_text=reconstructed_text,
            bidirectional_context=trace.bidirectional_context,
            accounting=trace.accounting,
        )

    def accounting(self) -> ArtifactAccounting:
        return self._last_fit_accounting


NoncausalReconstructiveModel = NoncausalReconstructiveAdapter


__all__ = [
    "NoncausalReconstructiveAdapter",
    "NoncausalReconstructiveConfig",
    "NoncausalReconstructiveFitReport",
    "NoncausalReconstructiveModel",
    "NoncausalReconstructiveReport",
    "NoncausalReconstructiveTrace",
]
