from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bootstrap import add_local_sources

add_local_sources()

from open_predictive_coder.artifacts import ArtifactMetadata, make_artifact_accounting, make_replay_span
from open_predictive_coder.artifacts_audits import ArtifactAuditRecord, audit_artifact
from open_predictive_coder.codecs import ensure_tokens
from open_predictive_coder.metrics import bits_per_token_from_probabilities
from open_predictive_coder.ngram_memory import NgramMemory, NgramMemoryConfig


def _coerce_tokens(data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> np.ndarray:
    return ensure_tokens(data).astype(np.int64, copy=False)


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(values.shape[-1], 1.0 / float(values.shape[-1]), dtype=np.float64)
    return values / total


@dataclass(frozen=True)
class GolfSubmissionModelConfig:
    vocabulary_size: int = 256
    bigram_alpha: float = 0.5
    trigram_alpha: float = 0.5
    trigram_bucket_count: int = 2048
    calibration_fraction: float = 0.1
    calibration_min_tokens: int = 65_536
    simplex_grid_denominator: int = 20
    score_block_tokens: int = 2_000_000

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.bigram_alpha < 0.0:
            raise ValueError("bigram_alpha must be >= 0")
        if self.trigram_alpha < 0.0:
            raise ValueError("trigram_alpha must be >= 0")
        if self.trigram_bucket_count < 1:
            raise ValueError("trigram_bucket_count must be >= 1")
        if not 0.0 < self.calibration_fraction < 1.0:
            raise ValueError("calibration_fraction must lie in (0, 1)")
        if self.calibration_min_tokens < 16:
            raise ValueError("calibration_min_tokens must be >= 16")
        if self.simplex_grid_denominator < 1:
            raise ValueError("simplex_grid_denominator must be >= 1")
        if self.score_block_tokens < 1024:
            raise ValueError("score_block_tokens must be >= 1024")


@dataclass(frozen=True)
class GolfSubmissionFitReport:
    train_tokens: int
    train_bits_per_token: float
    ngram_bytes: int
    mixture_weights: np.ndarray
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class GolfSubmissionScore:
    tokens: int
    unigram_bits_per_token: float
    bigram_bits_per_token: float
    trigram_bits_per_token: float
    mixed_bits_per_token: float
    mixture_weights: np.ndarray


class GolfSubmissionModel:
    SOURCE_NAMES = ("unigram", "bigram", "trigram")

    def __init__(self, config: GolfSubmissionModelConfig | None = None):
        self.config = config or GolfSubmissionModelConfig()
        self.ngram_memory = NgramMemory(
            NgramMemoryConfig(
                vocabulary_size=self.config.vocabulary_size,
                bigram_alpha=self.config.bigram_alpha,
                trigram_alpha=self.config.trigram_alpha,
                trigram_bucket_count=self.config.trigram_bucket_count,
            )
        )
        self._mixture_weights = np.asarray([0.1, 0.25, 0.65], dtype=np.float64)

    def _split_training_tokens(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if tokens.size < max(self.config.calibration_min_tokens * 2, 128):
            return tokens, tokens
        calibration_tokens = max(int(tokens.size * self.config.calibration_fraction), self.config.calibration_min_tokens)
        calibration_start = max(tokens.size - calibration_tokens, 1)
        return tokens[:calibration_start], tokens[calibration_start:]

    def _simplex_candidates(self) -> np.ndarray:
        denom = int(self.config.simplex_grid_denominator)
        rows = []
        for unigram_weight in range(denom + 1):
            for bigram_weight in range(denom - unigram_weight + 1):
                trigram_weight = denom - unigram_weight - bigram_weight
                rows.append([unigram_weight, bigram_weight, trigram_weight])
        values = np.asarray(rows, dtype=np.float64) / float(denom)
        return values

    def _fit_mixture_weights(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.size == 0:
            return self._mixture_weights.copy()
        unigram = self.ngram_memory.chosen_probs(tokens, order="unigram")
        bigram = self.ngram_memory.chosen_probs(tokens, order="bigram")
        trigram = self.ngram_memory.chosen_probs(tokens, order="trigram")
        grid = self._simplex_candidates()
        best_weights = self._mixture_weights.copy()
        best_loss = float("inf")
        stacked = np.stack([unigram, bigram, trigram], axis=0)
        for weights in grid:
            mixed = np.clip(np.tensordot(weights, stacked, axes=(0, 0)), 1e-300, 1.0)
            loss = float(-np.mean(np.log(mixed)))
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
        return best_weights

    def fit(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> GolfSubmissionFitReport:
        tokens = _coerce_tokens(data)
        memory_tokens, calibration_tokens = self._split_training_tokens(tokens)
        if memory_tokens.size == 0:
            memory_tokens = tokens
        self.ngram_memory.fit(memory_tokens)
        self._mixture_weights = self._fit_mixture_weights(calibration_tokens)
        train_bits_per_token = 0.0
        if calibration_tokens.size > 0:
            train_bits_per_token = float(self.score(calibration_tokens).mixed_bits_per_token)
        return GolfSubmissionFitReport(
            train_tokens=int(tokens.size),
            train_bits_per_token=train_bits_per_token,
            ngram_bytes=self.ngram_memory.report().total_bytes,
            mixture_weights=self._mixture_weights.copy(),
            source_names=self.SOURCE_NAMES,
        )

    def _score_block(self, tokens: np.ndarray) -> tuple[float, float, float, float, int]:
        unigram = self.ngram_memory.chosen_probs(tokens, order="unigram")
        bigram = self.ngram_memory.chosen_probs(tokens, order="bigram")
        trigram = self.ngram_memory.chosen_probs(tokens, order="trigram")
        mixed = np.clip(
            self._mixture_weights[0] * unigram
            + self._mixture_weights[1] * bigram
            + self._mixture_weights[2] * trigram,
            1e-300,
            1.0,
        )
        return (
            float(np.sum(-np.log(unigram), dtype=np.float64)),
            float(np.sum(-np.log(bigram), dtype=np.float64)),
            float(np.sum(-np.log(trigram), dtype=np.float64)),
            float(np.sum(-np.log(mixed), dtype=np.float64)),
            int(tokens.size),
        )

    def score(self, data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int]) -> GolfSubmissionScore:
        tokens = _coerce_tokens(data)
        if tokens.size == 0:
            raise ValueError("score requires at least one token")

        unigram_loss = 0.0
        bigram_loss = 0.0
        trigram_loss = 0.0
        mixed_loss = 0.0
        total_tokens = 0
        block = int(self.config.score_block_tokens)
        for start in range(0, int(tokens.size), block):
            end = min(start + block, int(tokens.size))
            left = max(start - 2, 0)
            block_tokens = tokens[left:end]
            block_unigram, block_bigram, block_trigram, block_mixed, block_count = self._score_block(block_tokens)
            drop = start - left
            if drop > 0:
                trimmed = block_tokens[drop:]
                block_unigram, block_bigram, block_trigram, block_mixed, block_count = self._score_block(trimmed)
            unigram_loss += block_unigram
            bigram_loss += block_bigram
            trigram_loss += block_trigram
            mixed_loss += block_mixed
            total_tokens += block_count

        scale = np.log(2.0) * max(total_tokens, 1)
        return GolfSubmissionScore(
            tokens=total_tokens,
            unigram_bits_per_token=float(unigram_loss / scale),
            bigram_bits_per_token=float(bigram_loss / scale),
            trigram_bits_per_token=float(trigram_loss / scale),
            mixed_bits_per_token=float(mixed_loss / scale),
            mixture_weights=self._mixture_weights.copy(),
        )

    def predictive_distribution(
        self,
        context: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int],
    ) -> np.ndarray:
        tokens = _coerce_tokens(context)
        unigram = self.ngram_memory.unigram_probs()
        bigram = self.ngram_memory.bigram_probs(int(tokens[-1])) if tokens.size >= 1 else unigram
        trigram = self.ngram_memory.trigram_probs(int(tokens[-2]), int(tokens[-1])) if tokens.size >= 2 else bigram
        return _normalize(
            self._mixture_weights[0] * unigram
            + self._mixture_weights[1] * bigram
            + self._mixture_weights[2] * trigram
        )

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        return {
            "mixture_weights": self._mixture_weights.astype(np.float32, copy=True),
            "config_vocabulary_size": np.asarray([self.config.vocabulary_size], dtype=np.int32),
            "config_bigram_alpha": np.asarray([self.config.bigram_alpha], dtype=np.float32),
            "config_trigram_alpha": np.asarray([self.config.trigram_alpha], dtype=np.float32),
            "config_trigram_bucket_count": np.asarray([self.config.trigram_bucket_count], dtype=np.int32),
            "ngram_unigram_counts": self.ngram_memory.unigram_counts.astype(np.uint32, copy=True),
            "ngram_bigram_counts": self.ngram_memory.bigram_counts.astype(np.uint32, copy=True),
            "ngram_trigram_counts": self.ngram_memory.trigram_counts.astype(np.uint32, copy=True),
        }

    def save_artifact(
        self,
        path: str | Path,
        *,
        reference_tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactAuditRecord:
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(artifact_path, **self.artifact_arrays())

        replay_spans = ()
        if reference_tokens is not None:
            tokens = _coerce_tokens(reference_tokens)
            if tokens.size >= 1:
                replay_spans = (make_replay_span(0, int(tokens.size), label="causal_prefix_replay"),)

        artifact_bytes = int(artifact_path.stat().st_size)
        accounting = make_artifact_accounting(
            artifact_name="opc_causal_packed_memory_submission",
            artifact_bytes=artifact_bytes,
            replay_bytes=artifact_bytes,
            replay_spans=replay_spans,
            metadata=ArtifactMetadata.from_mapping(
                {
                    "vocabulary_size": self.config.vocabulary_size,
                    "trigram_bucket_count": self.config.trigram_bucket_count,
                    "source_names": list(self.SOURCE_NAMES),
                    "mixture_weights": self._mixture_weights.tolist(),
                    **(metadata or {}),
                }
            ),
        )
        return audit_artifact(
            accounting,
            payload_bytes=artifact_bytes,
            side_data_count=0,
            side_data_bytes=0,
        )

    @classmethod
    def load_artifact(cls, path: str | Path) -> "GolfSubmissionModel":
        with np.load(Path(path), allow_pickle=False) as data:
            config = GolfSubmissionModelConfig(
                vocabulary_size=int(np.asarray(data["config_vocabulary_size"]).reshape(-1)[0]),
                bigram_alpha=float(np.asarray(data["config_bigram_alpha"]).reshape(-1)[0]),
                trigram_alpha=float(np.asarray(data["config_trigram_alpha"]).reshape(-1)[0]),
                trigram_bucket_count=int(np.asarray(data["config_trigram_bucket_count"]).reshape(-1)[0]),
            )
            model = cls(config)
            model._mixture_weights = np.asarray(data["mixture_weights"], dtype=np.float64).reshape(-1)
            model.ngram_memory.unigram_counts = np.asarray(data["ngram_unigram_counts"], dtype=np.float64)
            model.ngram_memory.bigram_counts = np.asarray(data["ngram_bigram_counts"], dtype=np.float64)
            model.ngram_memory.trigram_counts = np.asarray(data["ngram_trigram_counts"], dtype=np.float64)
        model.ngram_memory._unigram_total = float(np.sum(model.ngram_memory.unigram_counts))
        model.ngram_memory._bigram_totals = np.sum(model.ngram_memory.bigram_counts, axis=1)
        model.ngram_memory._trigram_totals = np.sum(model.ngram_memory.trigram_counts, axis=1)
        return model


__all__ = [
    "GolfSubmissionFitReport",
    "GolfSubmissionModel",
    "GolfSubmissionModelConfig",
    "GolfSubmissionScore",
]
