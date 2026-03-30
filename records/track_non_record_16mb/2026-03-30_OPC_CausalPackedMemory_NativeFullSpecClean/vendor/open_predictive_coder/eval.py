from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .codecs import ensure_tokens

RolloutMode = Literal["teacher_forced", "closed_loop"]


@runtime_checkable
class SupportsSequenceScoring(Protocol):
    def score(self, sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]): ...


@runtime_checkable
class SupportsRolloutGeneration(Protocol):
    def generate(
        self,
        prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
        steps: int,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class NextStepScore:
    tokens: int
    bits_per_byte: float


@dataclass(frozen=True)
class RolloutEvaluation:
    mode: RolloutMode
    prompt_tokens: np.ndarray
    continuation_tokens: np.ndarray
    sequence_tokens: np.ndarray
    bits_per_byte: float
    total_tokens: int


def score_next_step(
    model: SupportsSequenceScoring,
    sequence: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
) -> NextStepScore:
    tokens = ensure_tokens(sequence)
    if tokens.size < 2:
        raise ValueError("sequence must contain at least two tokens")

    report = model.score(tokens)
    return NextStepScore(tokens=int(getattr(report, "tokens", tokens.size)), bits_per_byte=float(report.bits_per_byte))


def evaluate_rollout(
    model: SupportsSequenceScoring | SupportsRolloutGeneration,
    prompt: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    continuation: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | None = None,
    *,
    mode: RolloutMode = "teacher_forced",
    steps: int | None = None,
    temperature: float = 1.0,
    greedy: bool = False,
    seed: int | None = None,
) -> RolloutEvaluation:
    prompt_tokens = ensure_tokens(prompt)
    if prompt_tokens.size < 1:
        raise ValueError("prompt must contain at least one token")

    if mode == "teacher_forced":
        if continuation is None:
            raise ValueError("teacher_forced mode requires a continuation")
        continuation_tokens = ensure_tokens(continuation)
        sequence_tokens = np.concatenate([prompt_tokens, continuation_tokens])
    elif mode == "closed_loop":
        if continuation is not None:
            continuation_tokens = ensure_tokens(continuation)
            if steps is None:
                steps = int(continuation_tokens.size)
        else:
            continuation_tokens = np.asarray([], dtype=np.uint8)
        if steps is None:
            raise ValueError("closed_loop mode requires steps or a continuation length")
        if not hasattr(model, "generate"):
            raise TypeError("closed_loop mode requires a model with generate(...)")
        sequence_tokens = np.asarray(
            model.generate(
                prompt_tokens,
                steps=steps,
                temperature=temperature,
                greedy=greedy,
                seed=seed,
            ),
            dtype=np.uint8,
        )
        continuation_tokens = sequence_tokens[prompt_tokens.size :]
    else:
        raise ValueError(f"unknown rollout mode: {mode}")

    if sequence_tokens.size < 2:
        raise ValueError("the evaluated sequence must contain at least two tokens")

    score = score_next_step(model, sequence_tokens)
    return RolloutEvaluation(
        mode=mode,
        prompt_tokens=prompt_tokens,
        continuation_tokens=continuation_tokens,
        sequence_tokens=sequence_tokens,
        bits_per_byte=score.bits_per_byte,
        total_tokens=score.tokens,
    )


__all__ = [
    "NextStepScore",
    "RolloutEvaluation",
    "RolloutMode",
    "SupportsRolloutGeneration",
    "SupportsSequenceScoring",
    "evaluate_rollout",
    "score_next_step",
]
