from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import bits_per_byte_from_probabilities
from .probability_diagnostics import ProbabilityDiagnostics, ProbabilityDiagnosticsConfig, probability_diagnostics


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


def _normalize_probabilities(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    vocab_size = probabilities.shape[-1]
    totals = np.sum(probabilities, axis=-1, keepdims=True)
    return np.divide(
        probabilities,
        totals,
        out=np.full_like(probabilities, 1.0 / float(vocab_size)),
        where=totals > epsilon,
    )


def _coerce_labels(labels: object, expected_rows: int, vocab_size: int) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    if array.size != expected_rows:
        raise ValueError("labels must align with the probability rows")
    if np.any(array < 0) or np.any(array >= vocab_size):
        raise ValueError("labels must be valid vocabulary indices")
    return array


@dataclass(frozen=True)
class TeacherExportConfig:
    vocabulary_size: int = 256
    source_names: tuple[str, str] = ("teacher", "student")
    diagnostics: ProbabilityDiagnosticsConfig = ProbabilityDiagnosticsConfig()

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if len(self.source_names) != 2 or any(not name for name in self.source_names):
            raise ValueError("source_names must contain two non-empty names")


@dataclass(frozen=True)
class TeacherExportRecord:
    tokens: int
    source_names: tuple[str, str]
    teacher_probs: np.ndarray
    student_probs: np.ndarray
    teacher_labels: np.ndarray
    student_labels: np.ndarray
    diagnostics: ProbabilityDiagnostics

    @property
    def steps(self) -> int:
        return int(self.teacher_probs.shape[0])

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "teacher_probs": self.teacher_probs,
            "student_probs": self.student_probs,
            "teacher_labels": self.teacher_labels,
            "student_labels": self.student_labels,
            **self.diagnostics.as_dict(),
        }


@dataclass(frozen=True)
class TeacherExportReport:
    record: TeacherExportRecord
    teacher_bits_per_byte: float | None
    student_bits_per_byte: float | None
    mean_bits_per_byte: float | None
    label_flip_rate: float
    label_agreement_rate: float
    mean_entropy: float
    mean_peak: float
    mean_top_k_mass: float
    mean_overlap: float
    mean_shared_top_k_mass: float
    mean_top2_margin: float

    @property
    def tokens(self) -> int:
        return self.record.tokens

    @property
    def source_names(self) -> tuple[str, str]:
        return self.record.source_names

    @property
    def steps(self) -> int:
        return self.record.steps


class TeacherExportAdapter:
    def __init__(self, config: TeacherExportConfig | None = None):
        self.config = config or TeacherExportConfig()

    def _resolve_vocabulary_size(self, teacher_probs: np.ndarray, student_probs: np.ndarray) -> int:
        observed = int(teacher_probs.shape[-1])
        if student_probs.shape[-1] != observed:
            raise ValueError("teacher_probs and student_probs must have the same vocabulary size")
        configured = self.config.vocabulary_size
        if configured == observed:
            return observed
        if configured == 256 and observed != 256:
            return observed
        raise ValueError(
            f"configured vocabulary_size={configured} does not match input vocabulary_size={observed}"
        )

    def record(
        self,
        teacher_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        student_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        source_names: tuple[str, str] | None = None,
    ) -> TeacherExportRecord:
        source_names = source_names or self.config.source_names
        teacher = _coerce_probability_array(teacher_probs, name="teacher_probs")
        student = _coerce_probability_array(student_probs, name="student_probs")
        if teacher.shape != student.shape:
            raise ValueError("teacher_probs and student_probs must have the same shape")
        vocab_size = self._resolve_vocabulary_size(teacher, student)

        teacher = _normalize_probabilities(teacher, epsilon=self.config.diagnostics.epsilon)
        student = _normalize_probabilities(student, epsilon=self.config.diagnostics.epsilon)

        flattened_rows = int(np.prod(teacher.shape[:-1], dtype=np.int64)) if teacher.ndim > 1 else 1
        teacher_rows = np.reshape(teacher, (flattened_rows, vocab_size))
        student_rows = np.reshape(student, (flattened_rows, vocab_size))
        teacher_labels = np.argmax(teacher_rows, axis=-1).astype(np.int64, copy=False)
        student_labels = np.argmax(student_rows, axis=-1).astype(np.int64, copy=False)
        diagnostics = probability_diagnostics(
            teacher_rows,
            student_rows,
            config=self.config.diagnostics,
        )
        return TeacherExportRecord(
            tokens=flattened_rows,
            source_names=source_names,
            teacher_probs=teacher_rows,
            student_probs=student_rows,
            teacher_labels=teacher_labels,
            student_labels=student_labels,
            diagnostics=diagnostics,
        )

    def export(
        self,
        teacher_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        student_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> TeacherExportReport:
        record = self.record(teacher_probs, student_probs, source_names=source_names)
        target_array = None
        teacher_bits_per_byte = None
        student_bits_per_byte = None
        mean_bits_per_byte = None
        if targets is not None:
            target_array = _coerce_labels(targets, record.steps, record.teacher_probs.shape[-1])
            teacher_bits_per_byte = bits_per_byte_from_probabilities(record.teacher_probs, target_array)
            student_bits_per_byte = bits_per_byte_from_probabilities(record.student_probs, target_array)
            mean_bits_per_byte = float(0.5 * (teacher_bits_per_byte + student_bits_per_byte))

        label_flip_rate = float(np.mean(record.teacher_labels != record.student_labels)) if record.steps else 0.0
        label_agreement_rate = float(1.0 - label_flip_rate)
        diagnostics = record.diagnostics
        return TeacherExportReport(
            record=record,
            teacher_bits_per_byte=teacher_bits_per_byte,
            student_bits_per_byte=student_bits_per_byte,
            mean_bits_per_byte=mean_bits_per_byte,
            label_flip_rate=label_flip_rate,
            label_agreement_rate=label_agreement_rate,
            mean_entropy=float(np.mean(diagnostics.entropy)) if record.steps else 0.0,
            mean_peak=float(np.mean(diagnostics.peak)) if record.steps else 0.0,
            mean_top_k_mass=float(np.mean(diagnostics.top_k_mass)) if record.steps else 0.0,
            mean_overlap=float(np.mean(diagnostics.overlap)) if record.steps else 0.0,
            mean_shared_top_k_mass=float(np.mean(diagnostics.shared_top_k_mass)) if record.steps else 0.0,
            mean_top2_margin=float(np.mean(diagnostics.top2_margin)) if record.steps else 0.0,
        )

    def score(
        self,
        teacher_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        student_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> TeacherExportReport:
        return self.export(
            teacher_probs,
            student_probs,
            targets=targets,
            source_names=source_names,
        )

    def fit(
        self,
        teacher_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        student_probs: np.ndarray | Sequence[float] | Sequence[Sequence[float]],
        *,
        targets: object | None = None,
        source_names: tuple[str, str] | None = None,
    ) -> TeacherExportReport:
        return self.export(
            teacher_probs,
            student_probs,
            targets=targets,
            source_names=source_names,
        )


__all__ = [
    "TeacherExportAdapter",
    "TeacherExportConfig",
    "TeacherExportRecord",
    "TeacherExportReport",
]
