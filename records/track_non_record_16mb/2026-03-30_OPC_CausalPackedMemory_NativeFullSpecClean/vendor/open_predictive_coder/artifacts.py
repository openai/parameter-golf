from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _validate_metadata_value(value: Any, *, path: str = "metadata") -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_metadata_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_metadata_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            _validate_metadata_value(item, path=f"{path}.{key}")
        return
    raise TypeError(f"{path} must contain JSON-serializable values")


@dataclass(frozen=True)
class ArtifactMetadata:
    items: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        normalized: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for key, value in self.items:
            if not isinstance(key, str) or not key:
                raise ValueError("metadata keys must be non-empty strings")
            if key in seen:
                raise ValueError("metadata keys must be unique")
            _validate_metadata_value(value, path=f"metadata.{key}")
            normalized.append((key, value))
            seen.add(key)
        object.__setattr__(self, "items", tuple(normalized))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ArtifactMetadata":
        return cls(items=tuple(mapping.items()))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def merged(self, **updates: Any) -> "ArtifactMetadata":
        payload = self.to_dict()
        payload.update(updates)
        return ArtifactMetadata.from_mapping(payload)


def coerce_artifact_metadata(
    metadata: ArtifactMetadata | Mapping[str, Any] | None = None,
    /,
    **updates: Any,
) -> ArtifactMetadata:
    if metadata is None:
        base = ArtifactMetadata()
    elif isinstance(metadata, ArtifactMetadata):
        base = metadata
    elif isinstance(metadata, Mapping):
        base = ArtifactMetadata.from_mapping(metadata)
    else:
        raise TypeError("metadata must be an ArtifactMetadata, mapping, or None")
    return base.merged(**updates) if updates else base


@dataclass(frozen=True)
class ReplaySpan:
    start: int
    stop: int
    label: str | None = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.stop < self.start:
            raise ValueError("stop must be >= start")

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def is_empty(self) -> bool:
        return self.length == 0


def make_replay_span(
    start: int,
    stop: int,
    *,
    label: str | None = None,
    metadata: ArtifactMetadata | Mapping[str, Any] | None = None,
    **updates: Any,
) -> ReplaySpan:
    return ReplaySpan(
        start=start,
        stop=stop,
        label=label,
        metadata=coerce_artifact_metadata(metadata, **updates),
    )


@dataclass(frozen=True)
class ArtifactAccounting:
    artifact_name: str
    artifact_bytes: int
    replay_bytes: int
    replay_spans: tuple[ReplaySpan, ...] = ()
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)

    def __post_init__(self) -> None:
        if not self.artifact_name:
            raise ValueError("artifact_name must be non-empty")
        if self.artifact_bytes < 0:
            raise ValueError("artifact_bytes must be >= 0")
        if self.replay_bytes < 0:
            raise ValueError("replay_bytes must be >= 0")
        object.__setattr__(self, "replay_spans", tuple(self.replay_spans))
        if any(not isinstance(span, ReplaySpan) for span in self.replay_spans):
            raise TypeError("replay_spans must contain ReplaySpan instances")

    @property
    def replay_span_count(self) -> int:
        return len(self.replay_spans)

    @property
    def replay_span_length(self) -> int:
        return sum(span.length for span in self.replay_spans)

    @property
    def coverage_ratio(self) -> float:
        if self.artifact_bytes == 0:
            return 0.0
        return self.replay_bytes / float(self.artifact_bytes)

    @property
    def artifact_gap_bytes(self) -> int:
        return max(self.artifact_bytes - self.replay_bytes, 0)


def make_artifact_accounting(
    artifact_name: str,
    artifact_bytes: int,
    replay_bytes: int,
    *,
    replay_spans: Sequence[ReplaySpan] = (),
    metadata: ArtifactMetadata | Mapping[str, Any] | None = None,
    **updates: Any,
) -> ArtifactAccounting:
    return ArtifactAccounting(
        artifact_name=artifact_name,
        artifact_bytes=artifact_bytes,
        replay_bytes=replay_bytes,
        replay_spans=tuple(replay_spans),
        metadata=coerce_artifact_metadata(metadata, **updates),
    )


__all__ = [
    "ArtifactAccounting",
    "ArtifactMetadata",
    "coerce_artifact_metadata",
    "make_artifact_accounting",
    "make_replay_span",
    "ReplaySpan",
]
