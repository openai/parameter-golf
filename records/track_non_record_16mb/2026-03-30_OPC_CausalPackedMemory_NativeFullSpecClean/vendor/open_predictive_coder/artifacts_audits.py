from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .artifacts import ArtifactAccounting, ArtifactMetadata, ReplaySpan, coerce_artifact_metadata


@dataclass(frozen=True)
class ArtifactAuditRecord:
    accounting: ArtifactAccounting
    side_data_count: int = 0
    side_data_bytes: int = 0
    payload_bytes: int | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    def __post_init__(self) -> None:
        if self.side_data_count < 0:
            raise ValueError("side_data_count must be >= 0")
        if self.side_data_bytes < 0:
            raise ValueError("side_data_bytes must be >= 0")
        if self.payload_bytes is not None and self.payload_bytes < 0:
            raise ValueError("payload_bytes must be >= 0")
        if self.payload_bytes is None:
            object.__setattr__(self, "payload_bytes", self.accounting.artifact_bytes + self.side_data_bytes)
        object.__setattr__(self, "metadata", coerce_artifact_metadata(self.metadata))

    @property
    def artifact_name(self) -> str:
        return self.accounting.artifact_name

    @property
    def artifact_bytes(self) -> int:
        return self.accounting.artifact_bytes

    @property
    def replay_bytes(self) -> int:
        return self.accounting.replay_bytes

    @property
    def replay_spans(self) -> tuple[ReplaySpan, ...]:
        return self.accounting.replay_spans

    @property
    def replay_span_count(self) -> int:
        return self.accounting.replay_span_count

    @property
    def replay_span_length(self) -> int:
        return self.accounting.replay_span_length

    @property
    def coverage_ratio(self) -> float:
        return self.accounting.coverage_ratio

    @property
    def payload_coverage_ratio(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.replay_bytes / float(self.payload_bytes)

    @property
    def side_data_ratio(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.side_data_bytes / float(self.payload_bytes)

    @property
    def artifact_gap_bytes(self) -> int:
        return self.accounting.artifact_gap_bytes


@dataclass(frozen=True)
class ArtifactAuditSummary:
    records: tuple[ArtifactAuditRecord, ...] = ()
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def artifact_bytes(self) -> int:
        return sum(record.artifact_bytes for record in self.records)

    @property
    def replay_bytes(self) -> int:
        return sum(record.replay_bytes for record in self.records)

    @property
    def payload_bytes(self) -> int:
        return sum(int(record.payload_bytes) for record in self.records)

    @property
    def side_data_bytes(self) -> int:
        return sum(record.side_data_bytes for record in self.records)

    @property
    def side_data_count(self) -> int:
        return sum(record.side_data_count for record in self.records)

    @property
    def replay_span_count(self) -> int:
        return sum(record.replay_span_count for record in self.records)

    @property
    def replay_span_length(self) -> int:
        return sum(record.replay_span_length for record in self.records)

    @property
    def coverage_ratio(self) -> float:
        if self.artifact_bytes == 0:
            return 0.0
        return self.replay_bytes / float(self.artifact_bytes)

    @property
    def payload_coverage_ratio(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.replay_bytes / float(self.payload_bytes)

    @property
    def side_data_ratio(self) -> float:
        if self.payload_bytes == 0:
            return 0.0
        return self.side_data_bytes / float(self.payload_bytes)

    @property
    def artifact_gap_bytes(self) -> int:
        return max(self.artifact_bytes - self.replay_bytes, 0)


def audit_artifact(
    accounting: ArtifactAccounting,
    *,
    side_data_count: int = 0,
    side_data_bytes: int = 0,
    payload_bytes: int | None = None,
    metadata: ArtifactMetadata | dict[str, Any] | None = None,
    **updates: Any,
) -> ArtifactAuditRecord:
    return ArtifactAuditRecord(
        accounting=accounting,
        side_data_count=side_data_count,
        side_data_bytes=side_data_bytes,
        payload_bytes=payload_bytes,
        metadata=coerce_artifact_metadata(metadata, **updates),
    )


def summarize_artifact_audits(
    records: Sequence[ArtifactAuditRecord],
    *,
    metadata: ArtifactMetadata | dict[str, Any] | None = None,
    **updates: Any,
) -> ArtifactAuditSummary:
    return ArtifactAuditSummary(
        records=tuple(records),
        metadata=coerce_artifact_metadata(metadata, **updates),
    )


__all__ = [
    "ArtifactAuditRecord",
    "ArtifactAuditSummary",
    "audit_artifact",
    "summarize_artifact_audits",
]
