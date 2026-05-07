from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .codecs import ensure_tokens


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


def _coerce_context_token(context: object) -> int:
    if isinstance(context, (int, np.integer)):
        token = int(context)
        if token < 0:
            raise ValueError("context token must be >= 0")
        return token
    tokens = ensure_tokens(context)
    if tokens.size < 1:
        raise ValueError("context must contain at least one token")
    return int(tokens[-1])


def _laplace_smooth(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    width = counts.shape[-1]
    smoothed = counts + (alpha / float(width))
    total = float(np.sum(smoothed))
    if total <= 0.0:
        return np.full(width, 1.0 / float(width), dtype=np.float64)
    return smoothed / total


@dataclass(frozen=True)
class NgramMemoryConfig:
    vocabulary_size: int = 256
    bigram_alpha: float = 0.5
    trigram_alpha: float = 0.5
    trigram_bucket_count: int = 4096

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if self.bigram_alpha < 0.0:
            raise ValueError("bigram_alpha must be >= 0")
        if self.trigram_alpha < 0.0:
            raise ValueError("trigram_alpha must be >= 0")
        if self.trigram_bucket_count < 1:
            raise ValueError("trigram_bucket_count must be >= 1")


@dataclass(frozen=True)
class NgramMemoryReport:
    sequences: int
    tokens: int
    vocabulary_size: int
    bigram_contexts: int
    trigram_buckets_used: int
    unigram_bytes: int
    bigram_bytes: int
    trigram_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.unigram_bytes + self.bigram_bytes + self.trigram_bytes


class NgramMemory:
    def __init__(self, config: NgramMemoryConfig | None = None):
        self.config = config or NgramMemoryConfig()
        self.unigram_counts = np.zeros(self.config.vocabulary_size, dtype=np.float64)
        self.bigram_counts = np.zeros((self.config.vocabulary_size, self.config.vocabulary_size), dtype=np.float64)
        self.trigram_counts = np.zeros((self.config.trigram_bucket_count, self.config.vocabulary_size), dtype=np.float64)
        self._unigram_total = 0.0
        self._bigram_totals = np.zeros((self.config.vocabulary_size,), dtype=np.float64)
        self._trigram_totals = np.zeros((self.config.trigram_bucket_count,), dtype=np.float64)
        self._tokens_seen = 0
        self._sequences_seen = 0

    def clear(self) -> None:
        self.unigram_counts.fill(0.0)
        self.bigram_counts.fill(0.0)
        self.trigram_counts.fill(0.0)
        self._unigram_total = 0.0
        self._bigram_totals.fill(0.0)
        self._trigram_totals.fill(0.0)
        self._tokens_seen = 0
        self._sequences_seen = 0

    def _check_tokens(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 1:
            raise ValueError("tokens must be rank-1")
        if tokens.size == 0:
            return tokens
        if int(np.min(tokens)) < 0 or int(np.max(tokens)) >= self.config.vocabulary_size:
            raise ValueError("tokens must lie within the configured vocabulary")
        return tokens

    def _trigram_bucket(self, left: int, right: int) -> int:
        value = (np.uint64(left) * np.uint64(1_315_423_911)) ^ (np.uint64(right) * np.uint64(2_654_435_761))
        return int(value % np.uint64(self.config.trigram_bucket_count))

    def _trigram_buckets(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_u = np.asarray(left, dtype=np.uint64)
        right_u = np.asarray(right, dtype=np.uint64)
        values = (left_u * np.uint64(1_315_423_911)) ^ (right_u * np.uint64(2_654_435_761))
        return np.asarray(values % np.uint64(self.config.trigram_bucket_count), dtype=np.int64)

    def update(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> NgramMemoryReport:
        sequences = _coerce_sequences(data)
        for sequence in sequences:
            tokens = self._check_tokens(ensure_tokens(sequence).astype(np.int64, copy=False))
            self._sequences_seen += 1
            self._tokens_seen += int(tokens.size)
            if tokens.size == 0:
                continue

            unigram_delta = np.bincount(tokens, minlength=self.config.vocabulary_size).astype(np.float64, copy=False)
            self.unigram_counts += unigram_delta
            self._unigram_total += float(tokens.size)

            if tokens.size >= 2:
                prev = tokens[:-1]
                curr = tokens[1:]
                np.add.at(self.bigram_counts, (prev, curr), 1.0)
                self._bigram_totals += np.bincount(prev, minlength=self.config.vocabulary_size).astype(
                    np.float64,
                    copy=False,
                )

            if tokens.size >= 3:
                left = tokens[:-2]
                right = tokens[1:-1]
                target = tokens[2:]
                buckets = self._trigram_buckets(left, right)
                np.add.at(self.trigram_counts, (buckets, target), 1.0)
                self._trigram_totals += np.bincount(buckets, minlength=self.config.trigram_bucket_count).astype(
                    np.float64,
                    copy=False,
                )

        return self.report()

    def fit(
        self,
        data: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> NgramMemoryReport:
        self.clear()
        return self.update(data)

    def report(self) -> NgramMemoryReport:
        return NgramMemoryReport(
            sequences=self._sequences_seen,
            tokens=self._tokens_seen,
            vocabulary_size=self.config.vocabulary_size,
            bigram_contexts=int(np.count_nonzero(self._bigram_totals)),
            trigram_buckets_used=int(np.count_nonzero(self._trigram_totals)),
            unigram_bytes=int(self.unigram_counts.nbytes),
            bigram_bytes=int(self.bigram_counts.nbytes),
            trigram_bytes=int(self.trigram_counts.nbytes),
        )

    def unigram_probs(self) -> np.ndarray:
        return _laplace_smooth(self.unigram_counts, self.config.bigram_alpha)

    def bigram_probs(self, context: object) -> np.ndarray:
        index = _coerce_context_token(context)
        counts = self.bigram_counts[index]
        return _laplace_smooth(counts, self.config.bigram_alpha)

    def trigram_probs(self, context2: object, context1: object) -> np.ndarray:
        left = _coerce_context_token(context2)
        right = _coerce_context_token(context1)
        bucket = self._trigram_bucket(left, right)
        counts = self.trigram_counts[bucket]
        return _laplace_smooth(counts, self.config.trigram_alpha)

    def chosen_probs(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
        *,
        order: str = "max",
    ) -> np.ndarray:
        sequence = _coerce_sequences(tokens)[0]
        sequence = self._check_tokens(sequence.astype(np.int64, copy=False))
        if sequence.size == 0:
            return np.zeros((0,), dtype=np.float64)

        values = np.empty(sequence.size, dtype=np.float64)
        vocab = float(self.config.vocabulary_size)
        unigram_alpha = float(self.config.bigram_alpha)
        unigram_denom = max(self._unigram_total + unigram_alpha, np.finfo(np.float64).tiny)
        values[:] = (self.unigram_counts[sequence] + (unigram_alpha / vocab)) / unigram_denom

        if sequence.size >= 2 and order in {"bigram", "trigram", "max"}:
            prev = sequence[:-1]
            curr = sequence[1:]
            bigram_counts = self.bigram_counts[prev, curr]
            bigram_totals = self._bigram_totals[prev]
            values[1:] = (bigram_counts + (self.config.bigram_alpha / vocab)) / np.maximum(
                bigram_totals + self.config.bigram_alpha,
                np.finfo(np.float64).tiny,
            )

        if sequence.size >= 3 and order in {"trigram", "max"}:
            left = sequence[:-2]
            right = sequence[1:-1]
            target = sequence[2:]
            buckets = self._trigram_buckets(left, right)
            trigram_counts = self.trigram_counts[buckets, target]
            trigram_totals = self._trigram_totals[buckets]
            values[2:] = (trigram_counts + (self.config.trigram_alpha / vocab)) / np.maximum(
                trigram_totals + self.config.trigram_alpha,
                np.finfo(np.float64).tiny,
            )

        if order not in {"unigram", "bigram", "trigram", "max"}:
            raise ValueError(f"unknown order: {order}")
        return np.clip(values, 1e-300, 1.0)

    def chosen_log_probs(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
        *,
        order: str = "max",
    ) -> np.ndarray:
        return np.log(self.chosen_probs(tokens, order=order))

    def log_probs(
        self,
        tokens: str | bytes | bytearray | memoryview | np.ndarray | Sequence[int] | Sequence[object],
    ) -> np.ndarray:
        return self.chosen_log_probs(tokens, order="max")


__all__ = [
    "NgramMemory",
    "NgramMemoryConfig",
    "NgramMemoryReport",
]
