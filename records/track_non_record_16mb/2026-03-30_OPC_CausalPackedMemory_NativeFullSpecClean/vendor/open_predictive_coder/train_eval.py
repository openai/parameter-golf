from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .codecs import ensure_tokens
from .eval import RolloutMode


@runtime_checkable
class SupportsDatasetFit(Protocol):
    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> Any: ...


@runtime_checkable
class SupportsSequenceScoring(Protocol):
    def score(
        self,
        sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> Any: ...


@runtime_checkable
class SupportsPredictProba(Protocol):
    def predict_proba(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class DatasetEvaluation:
    sequences: int
    tokens: int
    effective_tokens: int
    bits_per_byte: float
    sequence_bits_per_byte: np.ndarray

    @property
    def steps(self) -> int:
        return self.effective_tokens


@dataclass(frozen=True)
class RolloutCurvePoint:
    step: int
    bits_per_byte: float
    match_rate: float | None


@dataclass(frozen=True)
class RolloutCurveEvaluation:
    mode: RolloutMode
    prompt_tokens: np.ndarray
    target_tokens: np.ndarray
    generated_tokens: np.ndarray
    sequence_tokens: np.ndarray
    checkpoints: tuple[RolloutCurvePoint, ...]

    @property
    def continuation_tokens(self) -> np.ndarray:
        return self.target_tokens

    @property
    def predicted_tokens(self) -> np.ndarray:
        return self.generated_tokens


@dataclass(frozen=True)
class TransferProbeReport:
    source_fit: Any
    source_eval: DatasetEvaluation
    target_zero_shot: DatasetEvaluation
    target_scratch_fit: Any | None
    target_scratch_eval: DatasetEvaluation | None

    @property
    def source_fit_bits_per_byte(self) -> float | None:
        value = getattr(self.source_fit, "train_bits_per_byte", None)
        return None if value is None else float(value)

    @property
    def target_fit_bits_per_byte(self) -> float | None:
        if self.target_scratch_fit is None:
            return None
        value = getattr(self.target_scratch_fit, "train_bits_per_byte", None)
        return None if value is None else float(value)

    @property
    def source_evaluation(self) -> DatasetEvaluation:
        return self.source_eval

    @property
    def target_from_source(self) -> DatasetEvaluation:
        return self.target_zero_shot

    @property
    def target_scratch(self) -> DatasetEvaluation | None:
        return self.target_scratch_eval

    @property
    def transfer_gap_bits_per_byte(self) -> float | None:
        if self.target_scratch_eval is None:
            return None
        return self.target_zero_shot.bits_per_byte - self.target_scratch_eval.bits_per_byte


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


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
    return clipped / np.sum(clipped)


def _sample_next_token(
    probabilities: np.ndarray,
    *,
    temperature: float,
    greedy: bool,
    rng: np.random.Generator,
) -> int:
    if greedy:
        return int(np.argmax(probabilities))
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    scaled = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
    stabilized = np.exp(scaled - np.max(scaled))
    sample_probs = stabilized / np.sum(stabilized)
    return int(rng.choice(sample_probs.shape[0], p=sample_probs))


def evaluate_dataset(
    model: SupportsSequenceScoring,
    data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
) -> DatasetEvaluation:
    sequences = _coerce_sequences(data)
    sequence_scores: list[float] = []
    total_tokens = 0
    total_effective_tokens = 0
    weighted_bits = 0.0

    for sequence in sequences:
        report = model.score(sequence)
        tokens = int(getattr(report, "tokens"))
        effective_tokens = max(tokens - 1, 0)
        bits_per_byte = float(getattr(report, "bits_per_byte"))
        sequence_scores.append(bits_per_byte)
        total_tokens += tokens
        total_effective_tokens += effective_tokens
        weighted_bits += bits_per_byte * effective_tokens

    mean_bits = 0.0 if total_effective_tokens == 0 else weighted_bits / total_effective_tokens
    return DatasetEvaluation(
        sequences=len(sequences),
        tokens=total_tokens,
        effective_tokens=total_effective_tokens,
        bits_per_byte=mean_bits,
        sequence_bits_per_byte=np.asarray(sequence_scores, dtype=np.float64),
    )


def evaluate_rollout_curve(
    model: SupportsSequenceScoring | SupportsPredictProba,
    prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    continuation: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | None = None,
    *,
    mode: RolloutMode = "teacher_forced",
    checkpoints: Sequence[int] | None = None,
    steps: int | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    seed: int | None = None,
) -> RolloutCurveEvaluation:
    if not hasattr(model, "predict_proba"):
        raise TypeError("evaluate_rollout_curve requires a model with predict_proba(...)")

    prompt_tokens = ensure_tokens(prompt)
    if prompt_tokens.size < 1:
        raise ValueError("prompt must contain at least one token")

    if continuation is not None:
        target_tokens = ensure_tokens(continuation)
    else:
        target_tokens = np.asarray([], dtype=np.uint8)

    if mode == "teacher_forced":
        if target_tokens.size == 0:
            raise ValueError("teacher_forced mode requires a continuation")
        total_steps = int(target_tokens.size)
    elif mode == "closed_loop":
        if target_tokens.size > 0:
            if steps is None:
                steps = int(target_tokens.size)
            elif steps != int(target_tokens.size):
                raise ValueError("closed_loop steps must match continuation length when both are provided")
        if steps is None:
            raise ValueError("closed_loop mode requires steps or a continuation")
        total_steps = int(steps)
    else:
        raise ValueError(f"unknown rollout mode: {mode}")

    checkpoint_values = tuple(sorted(set(checkpoints or (total_steps,))))
    if not checkpoint_values:
        raise ValueError("checkpoints must not be empty")
    if checkpoint_values[0] < 1 or checkpoint_values[-1] > total_steps:
        raise ValueError("checkpoints must lie within the rollout length")

    rng = np.random.default_rng(seed)
    generated: list[int] = []
    context = prompt_tokens.astype(np.uint8, copy=True).tolist()
    checkpoint_set = set(checkpoint_values)
    match_count = 0
    points: list[RolloutCurvePoint] = []

    for step in range(total_steps):
        prefix = np.asarray(context, dtype=np.uint8)
        probabilities = _normalize_probabilities(np.asarray(model.predict_proba(prefix), dtype=np.float64))
        predicted_token = int(np.argmax(probabilities))

        if mode == "teacher_forced":
            next_token = int(target_tokens[step])
        else:
            next_token = _sample_next_token(probabilities, temperature=temperature, greedy=greedy, rng=rng)

        if target_tokens.size > step and predicted_token == int(target_tokens[step]):
            match_count += 1

        generated.append(next_token)
        context.append(next_token)
        step_count = step + 1
        if step_count in checkpoint_set:
            sequence_tokens = np.asarray(context, dtype=np.uint8)
            score = model.score(sequence_tokens)
            match_rate = None
            if target_tokens.size > 0:
                match_rate = match_count / float(step_count)
            points.append(
                RolloutCurvePoint(
                    step=step_count,
                    bits_per_byte=float(score.bits_per_byte),
                    match_rate=match_rate,
                )
            )

    generated_tokens = np.asarray(generated, dtype=np.uint8)
    return RolloutCurveEvaluation(
        mode=mode,
        prompt_tokens=prompt_tokens,
        target_tokens=target_tokens,
        generated_tokens=generated_tokens,
        sequence_tokens=np.asarray(context, dtype=np.uint8),
        checkpoints=tuple(points),
    )


def evaluate_transfer_probe(
    model_factory: Callable[[], SupportsDatasetFit | SupportsSequenceScoring],
    source_train: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    target_train: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object] | None = None,
    *,
    source_eval: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object] | None = None,
    target_eval: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object] | None = None,
) -> TransferProbeReport:
    if target_train is None and target_eval is None:
        raise ValueError("transfer probe requires target_train or target_eval")

    source_model = model_factory()
    if not hasattr(source_model, "fit") or not hasattr(source_model, "score"):
        raise TypeError("transfer probe requires models with fit(...) and score(...)")

    source_fit = source_model.fit(source_train)
    source_eval_report = evaluate_dataset(source_model, source_train if source_eval is None else source_eval)
    target_eval_data = target_eval if target_eval is not None else target_train
    assert target_eval_data is not None
    target_zero_shot = evaluate_dataset(source_model, target_eval_data)

    target_scratch_fit = None
    target_scratch_eval = None
    if target_train is not None:
        scratch_model = model_factory()
        if not hasattr(scratch_model, "fit") or not hasattr(scratch_model, "score"):
            raise TypeError("transfer probe requires models with fit(...) and score(...)")
        target_scratch_fit = scratch_model.fit(target_train)
        target_scratch_eval = evaluate_dataset(scratch_model, target_eval_data)

    return TransferProbeReport(
        source_fit=source_fit,
        source_eval=source_eval_report,
        target_zero_shot=target_zero_shot,
        target_scratch_fit=target_scratch_fit,
        target_scratch_eval=target_scratch_eval,
    )


RolloutCurveMode = RolloutMode
RolloutCheckpoint = RolloutCurvePoint
RolloutCurve = RolloutCurveEvaluation
TransferEvaluation = TransferProbeReport
SupportsNextTokenProbabilities = SupportsPredictProba
SupportsSequenceScore = SupportsSequenceScoring
score_dataset = evaluate_dataset


__all__ = [
    "DatasetEvaluation",
    "RolloutCheckpoint",
    "RolloutCurve",
    "RolloutCurveMode",
    "RolloutCurveEvaluation",
    "RolloutCurvePoint",
    "SupportsDatasetFit",
    "SupportsNextTokenProbabilities",
    "SupportsPredictProba",
    "SupportsSequenceScore",
    "SupportsSequenceScoring",
    "TransferEvaluation",
    "TransferProbeReport",
    "evaluate_dataset",
    "evaluate_rollout_curve",
    "evaluate_transfer_probe",
    "score_dataset",
]
