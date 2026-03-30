from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .artifacts import ArtifactAccounting, ArtifactMetadata, coerce_artifact_metadata


@dataclass(frozen=True)
class SequenceTrace:
    features: np.ndarray
    targets: np.ndarray
    boundaries: np.ndarray
    tokens: int
    patches: int


@dataclass(frozen=True)
class SequenceReport:
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float
    bits_per_byte: float


@dataclass(frozen=True)
class FitReport:
    sequences: int
    tokens: int
    patches: int
    mean_patch_size: float
    compression_ratio: float
    train_bits_per_byte: float


@dataclass(frozen=True)
class CausalTrace:
    trace: SequenceTrace
    artifact_accounting: ArtifactAccounting | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def tokens(self) -> int:
        return int(self.trace.tokens)

    @property
    def patches(self) -> int:
        return int(self.trace.patches)


@dataclass(frozen=True)
class CausalSequenceReport:
    report: SequenceReport
    artifact_accounting: ArtifactAccounting | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def tokens(self) -> int:
        return int(self.report.tokens)

    @property
    def patches(self) -> int:
        return int(self.report.patches)

    @property
    def bits_per_byte(self) -> float:
        return float(self.report.bits_per_byte)


@dataclass(frozen=True)
class CausalFitReport:
    report: FitReport
    artifact_accounting: ArtifactAccounting | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def sequences(self) -> int:
        return int(self.report.sequences)

    @property
    def tokens(self) -> int:
        return int(self.report.tokens)

    @property
    def patches(self) -> int:
        return int(self.report.patches)

    @property
    def train_bits_per_byte(self) -> float:
        return float(self.report.train_bits_per_byte)


def tag_metadata(
    metadata: ArtifactMetadata | Mapping[str, Any] | None = None,
    /,
    **updates: Any,
) -> ArtifactMetadata:
    return coerce_artifact_metadata(metadata, **updates)


__all__ = [
    "CausalFitReport",
    "CausalSequenceReport",
    "CausalTrace",
    "FitReport",
    "SequenceReport",
    "SequenceTrace",
    "tag_metadata",
]
