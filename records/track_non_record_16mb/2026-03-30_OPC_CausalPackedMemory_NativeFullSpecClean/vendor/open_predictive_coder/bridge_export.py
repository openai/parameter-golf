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
from .bridge_features import BridgeFeatureArrays, BridgeFeatureConfig, bridge_feature_arrays
from .codecs import ensure_tokens
from .metrics import bits_per_byte_from_probabilities
from .span_selection import SpanSelectionConfig, replay_spans_from_scores


def _coerce_probability_array(
    probabilities: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain non-negative values")
    return array


def _coerce_targets(targets: object, expected_tokens: int) -> np.ndarray:
    token_array = ensure_tokens(targets).astype(np.int64, copy=False).reshape(-1)
    if token_array.size != expected_tokens:
        raise ValueError("targets must align with the probability rows")
    return token_array


@dataclass(frozen=True)
class BridgeExportConfig:
    vocabulary_size: int = 256
    candidate_count: int = 4
    epsilon: float = 1e-12
    replay_threshold: float = 0.0
    source_names: tuple[str, str] = ("base", "proxy")

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.replay_threshold < 0.0:
            raise ValueError("replay_threshold must be >= 0")
        if len(self.source_names) != 2 or any(not name for name in self.source_names):
            raise ValueError("source_names must contain two non-empty names")


@dataclass(frozen=True)
class BridgeExportReport:
    tokens: int
    source_names: tuple[str, str]
    features: BridgeFeatureArrays
    mean_entropy: float
    mean_peak: float
    mean_candidate4: float
    mean_agreement: float
    mean_agreement_mass: float
    base_bits_per_byte: float | None
    proxy_bits_per_byte: float | None
    mean_bits_per_byte: float | None
    accounting: ArtifactAccounting

    @property
    def bits_per_byte(self) -> float:
        return 0.0 if self.mean_bits_per_byte is None else float(self.mean_bits_per_byte)


@dataclass(frozen=True)
class BridgeExportFitReport:
    sequences: int
    tokens: int
    report: BridgeExportReport
    accounting: ArtifactAccounting

    @property
    def bits_per_byte(self) -> float:
        return self.report.bits_per_byte


class BridgeExportAdapter:
    def __init__(
        self,
        config: BridgeExportConfig | None = None,
        *,
        artifact_name: str = "bridge_export",
        metadata: ArtifactMetadata | None = None,
    ):
        self.config = config or BridgeExportConfig()
        self.feature_config = BridgeFeatureConfig(
            candidate_count=self.config.candidate_count,
            epsilon=self.config.epsilon,
        )
        self.artifact_name = artifact_name
        self.metadata = metadata or ArtifactMetadata()
        self._last_fit_accounting = make_artifact_accounting(
            self.artifact_name,
            0,
            0,
            metadata=self.metadata,
            tokens=0,
            bridge_rows=0,
        )

    def _resolve_vocabulary_size(self, base: np.ndarray, proxy: np.ndarray) -> int:
        observed = int(base.shape[-1])
        if proxy.shape[-1] != observed:
            raise ValueError("base_probs and proxy_probs must have the same vocabulary size")
        configured = self.config.vocabulary_size
        if configured == observed:
            return observed
        if configured == 256 and observed != 256:
            return observed
        raise ValueError(
            f"configured vocabulary_size={configured} does not match input vocabulary_size={observed}"
        )

    def _build_accounting(
        self,
        features: BridgeFeatureArrays,
        *,
        tokens: int,
        source_names: tuple[str, str],
    ) -> ArtifactAccounting:
        feature_rows = np.asarray(features.entropy, dtype=np.float64).reshape(-1)
        replay_scores = np.asarray(features.agreement_mass, dtype=np.float64).reshape(-1)
        replay_mask = replay_scores > self.config.replay_threshold
        replay_spans = replay_spans_from_scores(
            replay_scores,
            SpanSelectionConfig(threshold=self.config.replay_threshold, min_span=1, max_gap=0),
            label="bridge_export",
            source_names=source_names,
        )

        return make_artifact_accounting(
            self.artifact_name,
            int(tokens),
            int(replay_mask.sum()),
            replay_spans=replay_spans,
            metadata=self.metadata,
            tokens=int(tokens),
            bridge_rows=int(feature_rows.size),
            source_names=source_names,
        )

    def export(
        self,
        base_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        proxy_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> BridgeExportReport:
        source_names = source_names or self.config.source_names
        base = _coerce_probability_array(base_probs, name="base_probs")
        proxy = _coerce_probability_array(proxy_probs, name="proxy_probs")
        if base.shape != proxy.shape:
            raise ValueError("base_probs and proxy_probs must have the same shape")
        vocab_size = self._resolve_vocabulary_size(base, proxy)

        features = bridge_feature_arrays(
            base,
            proxy,
            vocab_size,
            config=self.feature_config,
        )
        flattened_rows = int(np.prod(base.shape[:-1], dtype=np.int64)) if base.ndim > 1 else 1
        flat_base = np.reshape(base, (flattened_rows, vocab_size))
        flat_proxy = np.reshape(proxy, (flattened_rows, vocab_size))

        target_array = None
        base_bits_per_byte = None
        proxy_bits_per_byte = None
        mean_bits_per_byte = None
        if targets is not None:
            target_array = _coerce_targets(targets, flattened_rows)
            base_bits_per_byte = bits_per_byte_from_probabilities(flat_base, target_array)
            proxy_bits_per_byte = bits_per_byte_from_probabilities(flat_proxy, target_array)
            mean_bits_per_byte = float(0.5 * (base_bits_per_byte + proxy_bits_per_byte))

        accounting = self._build_accounting(
            features,
            tokens=flattened_rows,
            source_names=source_names,
        )
        return BridgeExportReport(
            tokens=flattened_rows,
            source_names=source_names,
            features=features,
            mean_entropy=float(np.mean(np.asarray(features.entropy, dtype=np.float64))) if flattened_rows else 0.0,
            mean_peak=float(np.mean(np.asarray(features.peak, dtype=np.float64))) if flattened_rows else 0.0,
            mean_candidate4=float(np.mean(np.asarray(features.candidate4, dtype=np.float64))) if flattened_rows else 0.0,
            mean_agreement=float(np.mean(np.asarray(features.agreement, dtype=np.float64))) if flattened_rows else 0.0,
            mean_agreement_mass=float(np.mean(np.asarray(features.agreement_mass, dtype=np.float64))) if flattened_rows else 0.0,
            base_bits_per_byte=base_bits_per_byte,
            proxy_bits_per_byte=proxy_bits_per_byte,
            mean_bits_per_byte=mean_bits_per_byte,
            accounting=accounting,
        )

    def score(
        self,
        base_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        proxy_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> BridgeExportReport:
        return self.export(
            base_probs,
            proxy_probs,
            targets=targets,
            source_names=source_names,
        )

    def fit(
        self,
        base_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        proxy_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> BridgeExportFitReport:
        report = self.export(
            base_probs,
            proxy_probs,
            targets=targets,
            source_names=source_names,
        )
        self._last_fit_accounting = report.accounting
        return BridgeExportFitReport(
            sequences=1,
            tokens=report.tokens,
            report=report,
            accounting=report.accounting,
        )

    def accounting(self) -> ArtifactAccounting:
        return self._last_fit_accounting


__all__ = [
    "BridgeExportAdapter",
    "BridgeExportConfig",
    "BridgeExportFitReport",
    "BridgeExportReport",
]
