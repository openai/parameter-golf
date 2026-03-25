#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import pickle
import time
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sentencepiece as spm


HEADER_INTS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = HEADER_INTS * HEADER_DTYPE.itemsize
    token_bytes = TOKEN_DTYPE.itemsize
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != HEADER_MAGIC or int(header[1]) != HEADER_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected shard size for {path}: {path.stat().st_size} != {expected_size}")
    return np.fromfile(path, dtype=TOKEN_DTYPE, count=num_tokens, offset=header_bytes)


def load_validation_tokens(pattern: str, max_tokens: int | None) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    shards = [load_data_shard(path) for path in files]
    tokens = np.concatenate(shards).astype(np.uint16, copy=False)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    if tokens.size < 2:
        raise ValueError("Need at least two tokens for next-token evaluation")
    return tokens


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token = np.ones((table_size,), dtype=np.bool_)

    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))

    return base_bytes, has_leading_space, is_boundary_token


def compute_target_byte_counts(
    tokens: np.ndarray,
    base_bytes: np.ndarray,
    has_leading_space: np.ndarray,
    is_boundary_token: np.ndarray,
) -> np.ndarray:
    prev_ids = tokens[:-1].astype(np.int32, copy=False)
    tgt_ids = tokens[1:].astype(np.int32, copy=False)
    token_bytes = base_bytes[tgt_ids].astype(np.int32, copy=True)
    token_bytes += (has_leading_space[tgt_ids] & ~is_boundary_token[prev_ids]).astype(np.int32)
    return token_bytes


class Expert:
    name: str

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del tokens, position_offset
        return None

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        raise NotImplementedError

    def cached_prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None,
    ) -> float:
        if cache is None:
            return self.prob(tokens, pos, token, None)
        key = (id(self), token)
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = self.prob(tokens, pos, token, cache)
        cache[key] = value
        return value

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        raise NotImplementedError

    def reset_state(self) -> None:
        return None

    def state_dict(self) -> dict | None:
        return None

    def load_state_dict(self, state: dict | None) -> None:
        del state
        return None


@dataclass
class DiscountCountTable:
    context_totals: dict[int, int] = field(default_factory=dict)
    pair_counts: dict[int, int] = field(default_factory=dict)
    n1: dict[int, int] = field(default_factory=dict)
    n2: dict[int, int] = field(default_factory=dict)
    n3p: dict[int, int] = field(default_factory=dict)


@dataclass
class FollowerCountTable:
    context_totals: dict[int, int] = field(default_factory=dict)
    pair_counts: dict[int, int] = field(default_factory=dict)
    followers: dict[int, list[int]] = field(default_factory=dict)


class UnigramExpert(Expert):
    def __init__(self, vocab_size: int, alpha: float = 0.5):
        self.name = "unigram_kt"
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.counts = np.zeros((vocab_size,), dtype=np.uint32)
        self.total = 0

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        del tokens, pos
        return (float(self.counts[token]) + self.alpha) / (float(self.total) + self.alpha * self.vocab_size)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        del tokens, pos
        self.counts[token] += 1
        self.total += 1

    def state_dict(self) -> dict:
        return {
            "counts": self.counts.copy(),
            "total": self.total,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        self.counts = np.array(state["counts"], dtype=np.uint32, copy=True)
        self.total = int(state["total"])


class BigramExpert(Expert):
    def __init__(
        self,
        vocab_size: int,
        backoff: Expert,
        alpha: float = 4.0,
        discount: float = 0.0,
        use_continuation_unigram: bool = False,
    ):
        self.name = "bigram_interp"
        self.vocab_size = vocab_size
        self.backoff = backoff
        self.alpha = alpha
        self.discount = discount
        self.use_continuation_unigram = use_continuation_unigram
        self.counts = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        self.row_totals = np.zeros((vocab_size,), dtype=np.uint32)
        self.distinct_followers = np.zeros((vocab_size,), dtype=np.uint32)
        self.continuation_counts = np.zeros((vocab_size,), dtype=np.uint32)
        self.total_distinct_bigrams = 0

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        prev_token = int(tokens[pos - 1])
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        row_total = float(self.row_totals[prev_token])
        pair_count = float(self.counts[prev_token, token])
        if row_total <= 0.0:
            return base_prob
        if self.discount > 0.0:
            if self.use_continuation_unigram and self.total_distinct_bigrams > 0:
                base_prob = float(self.continuation_counts[token]) / float(self.total_distinct_bigrams)
            backoff_mass = self.discount * float(self.distinct_followers[prev_token]) / row_total
            discounted = max(pair_count - self.discount, 0.0) / row_total
            return discounted + backoff_mass * base_prob
        return (pair_count + self.alpha * base_prob) / (row_total + self.alpha)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        prev_token = int(tokens[pos - 1])
        if self.counts[prev_token, token] == 0:
            self.distinct_followers[prev_token] += 1
            self.continuation_counts[token] += 1
            self.total_distinct_bigrams += 1
        self.counts[prev_token, token] += 1
        self.row_totals[prev_token] += 1

    def state_dict(self) -> dict:
        return {
            "counts": self.counts.copy(),
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        self.counts = np.array(state["counts"], dtype=np.uint32, copy=True)
        self.row_totals = self.counts.sum(axis=1, dtype=np.uint32)
        self.distinct_followers = (self.counts > 0).sum(axis=1, dtype=np.uint32)
        self.continuation_counts = (self.counts > 0).sum(axis=0, dtype=np.uint32)
        self.total_distinct_bigrams = int(self.continuation_counts.sum(dtype=np.uint64))


class HashedNgramExpert(Expert):
    def __init__(
        self,
        context_len: int,
        vocab_size: int,
        backoff: Expert,
        alpha: float = 1.0,
        discount: float = 0.0,
        count_scale_limit: int = 0,
        seed: int = 1469598103934665603,
        base: int = 1099511628211,
    ):
        self.name = f"ngram_{context_len + 1}"
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.backoff = backoff
        self.alpha = alpha
        self.discount = discount
        self.count_scale_limit = count_scale_limit
        self.seed = seed
        self.base = base
        self.mask = (1 << 64) - 1
        self.token_shift = max(1, (vocab_size - 1).bit_length())
        self.context_totals: dict[int, int] = {}
        self.pair_counts: dict[int, int] = {}
        self.distinct_followers: dict[int, int] = {}
        self.followers: dict[int, list[int]] = {}
        self.base_pow = pow(self.base, self.context_len - 1, 1 << 64)
        self.sequence_keys: np.ndarray | None = None

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del position_offset
        if tokens.size <= self.context_len:
            self.sequence_keys = None
            return
        window_hash = 0
        for idx in range(self.context_len):
            window_hash = (window_hash * self.base + int(tokens[idx]) + 1) & self.mask
        keys = np.zeros((tokens.size,), dtype=np.uint64)
        keys[self.context_len] = window_hash ^ self.seed
        for pos in range(self.context_len + 1, tokens.size):
            old_token = int(tokens[pos - self.context_len - 1]) + 1
            new_token = int(tokens[pos - 1]) + 1
            window_hash = (window_hash - (old_token * self.base_pow)) & self.mask
            window_hash = (window_hash * self.base + new_token) & self.mask
            keys[pos] = window_hash ^ self.seed
        self.sequence_keys = keys

    def _context_key(self, tokens: np.ndarray, pos: int) -> int | None:
        if pos < self.context_len:
            return None
        del tokens
        if self.sequence_keys is None:
            return None
        return int(self.sequence_keys[pos])

    def _pair_key(self, context_key: int, token: int) -> int:
        return (context_key << self.token_shift) | token

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        context_key = self._context_key(tokens, pos)
        if context_key is None:
            return base_prob
        total = float(self.context_totals.get(context_key, 0))
        count = float(self.pair_counts.get(self._pair_key(context_key, token), 0))
        if total <= 0.0:
            return base_prob
        if self.discount > 0.0:
            distinct = float(self.distinct_followers.get(context_key, 0))
            backoff_mass = self.discount * distinct / total
            discounted = max(count - self.discount, 0.0) / total
            return discounted + backoff_mass * base_prob
        return (count + self.alpha * base_prob) / (total + self.alpha)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        context_key = self._context_key(tokens, pos)
        if context_key is None:
            return
        pair_key = self._pair_key(context_key, token)
        prev = self.pair_counts.get(pair_key, 0)
        if prev == 0:
            self.distinct_followers[context_key] = self.distinct_followers.get(context_key, 0) + 1
            self.followers.setdefault(context_key, []).append(token)
        self.pair_counts[pair_key] = prev + 1
        new_total = self.context_totals.get(context_key, 0) + 1
        self.context_totals[context_key] = new_total
        if self.count_scale_limit > 0 and new_total > self.count_scale_limit:
            self._scale_context(context_key)

    def _scale_context(self, context_key: int) -> None:
        followers = self.followers.get(context_key)
        if not followers:
            return
        kept_followers: list[int] = []
        total = 0
        distinct = 0
        for token in followers:
            pair_key = self._pair_key(context_key, token)
            count = self.pair_counts.get(pair_key, 0)
            if count <= 0:
                continue
            scaled = count // 2
            if scaled <= 0:
                self.pair_counts.pop(pair_key, None)
                continue
            self.pair_counts[pair_key] = scaled
            kept_followers.append(token)
            total += scaled
            distinct += 1
        if kept_followers:
            self.followers[context_key] = kept_followers
            self.context_totals[context_key] = total
            self.distinct_followers[context_key] = distinct
        else:
            self.followers.pop(context_key, None)
            self.context_totals.pop(context_key, None)
            self.distinct_followers.pop(context_key, None)

    def state_dict(self) -> dict:
        size = len(self.pair_counts)
        contexts = np.empty((size,), dtype=np.uint64)
        tokens = np.empty((size,), dtype=np.uint16)
        counts = np.empty((size,), dtype=np.uint32)
        token_mask = (1 << self.token_shift) - 1
        for idx, (pair_key, count) in enumerate(self.pair_counts.items()):
            contexts[idx] = pair_key >> self.token_shift
            tokens[idx] = pair_key & token_mask
            counts[idx] = count
        if size:
            order = np.argsort(contexts, kind="stable")
            contexts = contexts[order]
            tokens = tokens[order]
            counts = counts[order]
        return {
            "pair_contexts": contexts,
            "pair_tokens": tokens,
            "pair_counts": counts,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        pair_contexts = np.array(state["pair_contexts"], dtype=np.uint64, copy=False)
        pair_tokens = np.array(state["pair_tokens"], dtype=np.uint16, copy=False)
        pair_counts = np.array(state["pair_counts"], dtype=np.uint32, copy=False)
        self.context_totals = {}
        self.pair_counts = {}
        self.distinct_followers = {}
        self.followers = {}
        track_followers = self.count_scale_limit > 0
        for context_key_raw, token_raw, count_raw in zip(pair_contexts, pair_tokens, pair_counts, strict=True):
            context_key = int(context_key_raw)
            token = int(token_raw)
            count = int(count_raw)
            pair_key = self._pair_key(context_key, token)
            self.pair_counts[pair_key] = count
            self.context_totals[context_key] = self.context_totals.get(context_key, 0) + count
            self.distinct_followers[context_key] = self.distinct_followers.get(context_key, 0) + 1
            if track_followers:
                self.followers.setdefault(context_key, []).append(token)


class ModifiedKneserNeyExpert(Expert):
    def __init__(
        self,
        max_context: int,
        vocab_size: int,
        base_unigram: Expert,
        unigram_alpha: float = 1.0,
        discounts: tuple[float, float, float] = (0.7, 1.0, 1.3),
        seed: int = 1469598103934665603,
        base: int = 1099511628211,
    ):
        if max_context < 1:
            raise ValueError("ModifiedKneserNeyExpert requires max_context >= 1")
        self.name = f"mkn_{max_context + 1}"
        self.context_len = max_context
        self.max_context = max_context
        self.vocab_size = vocab_size
        self.base_unigram = base_unigram
        self.unigram_alpha = unigram_alpha
        self.discount1, self.discount2, self.discount3p = discounts
        self.seed = seed
        self.base = base
        self.mask = (1 << 64) - 1
        self.token_shift = max(1, (vocab_size - 1).bit_length())
        self.base_pows = [0] * (self.max_context + 1)
        for context_len in range(1, self.max_context + 1):
            self.base_pows[context_len] = pow(self.base, context_len - 1, 1 << 64)
        self.tables = [DiscountCountTable() for _ in range(self.max_context + 1)]
        self.unigram_continuations = np.zeros((vocab_size,), dtype=np.uint32)
        self.total_unigram_continuations = 0
        self.sequence_keys: list[np.ndarray | None] = [None] * (self.max_context + 1)

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del position_offset
        for context_len in range(1, self.max_context + 1):
            if tokens.size <= context_len:
                self.sequence_keys[context_len] = None
                continue
            window_hash = 0
            for idx in range(context_len):
                window_hash = (window_hash * self.base + int(tokens[idx]) + 1) & self.mask
            keys = np.zeros((tokens.size,), dtype=np.uint64)
            keys[context_len] = window_hash ^ self.seed
            base_pow = self.base_pows[context_len]
            for pos in range(context_len + 1, tokens.size):
                old_token = int(tokens[pos - context_len - 1]) + 1
                new_token = int(tokens[pos - 1]) + 1
                window_hash = (window_hash - (old_token * base_pow)) & self.mask
                window_hash = (window_hash * self.base + new_token) & self.mask
                keys[pos] = window_hash ^ self.seed
            self.sequence_keys[context_len] = keys

    def _context_key(self, context_len: int, pos: int) -> int | None:
        if pos < context_len:
            return None
        keys = self.sequence_keys[context_len]
        if keys is None:
            return None
        return int(keys[pos])

    def _pair_key(self, context_key: int, token: int) -> int:
        return (context_key << self.token_shift) | token

    def _discount_for_count(self, count: int) -> float:
        if count <= 0:
            return 0.0
        if count == 1:
            return self.discount1
        if count == 2:
            return self.discount2
        return self.discount3p

    def _increment_table(self, context_len: int, context_key: int, token: int) -> bool:
        table = self.tables[context_len]
        pair_key = self._pair_key(context_key, token)
        prev = table.pair_counts.get(pair_key, 0)
        table.pair_counts[pair_key] = prev + 1
        table.context_totals[context_key] = table.context_totals.get(context_key, 0) + 1

        if prev == 0:
            table.n1[context_key] = table.n1.get(context_key, 0) + 1
            return True
        if prev == 1:
            table.n1[context_key] -= 1
            table.n2[context_key] = table.n2.get(context_key, 0) + 1
            return False
        if prev == 2:
            table.n2[context_key] -= 1
            table.n3p[context_key] = table.n3p.get(context_key, 0) + 1
            return False
        return False

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        prob = self.base_unigram.cached_prob(tokens, pos, token, cache)
        if self.total_unigram_continuations > 0:
            prob = (
                float(self.unigram_continuations[token]) + self.unigram_alpha * prob
            ) / (float(self.total_unigram_continuations) + self.unigram_alpha)

        max_depth = min(self.max_context, pos)
        for context_len in range(1, max_depth + 1):
            context_key = self._context_key(context_len, pos)
            if context_key is None:
                continue
            table = self.tables[context_len]
            total = float(table.context_totals.get(context_key, 0))
            if total <= 0.0:
                continue

            count = table.pair_counts.get(self._pair_key(context_key, token), 0)
            discount = self._discount_for_count(count)
            discounted = max(float(count) - discount, 0.0) / total
            backoff_mass = (
                self.discount1 * float(table.n1.get(context_key, 0))
                + self.discount2 * float(table.n2.get(context_key, 0))
                + self.discount3p * float(table.n3p.get(context_key, 0))
            ) / total
            backoff_mass = min(max(backoff_mass, 0.0), 1.0)
            prob = discounted + backoff_mass * prob
        return prob

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        max_depth = min(self.max_context, pos)
        propagate = True
        for context_len in range(max_depth, 0, -1):
            if not propagate:
                break
            context_key = self._context_key(context_len, pos)
            if context_key is None:
                continue
            propagate = self._increment_table(context_len, context_key, token)

        if propagate:
            self.total_unigram_continuations += 1
            self.unigram_continuations[token] += 1

    def state_dict(self) -> dict:
        tables = []
        for table in self.tables:
            tables.append(
                {
                    "context_totals": dict(table.context_totals),
                    "pair_counts": dict(table.pair_counts),
                    "n1": dict(table.n1),
                    "n2": dict(table.n2),
                    "n3p": dict(table.n3p),
                }
            )
        return {
            "tables": tables,
            "unigram_continuations": self.unigram_continuations.copy(),
            "total_unigram_continuations": self.total_unigram_continuations,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        self.tables = []
        for saved in state["tables"]:
            self.tables.append(
                DiscountCountTable(
                    context_totals={int(key): int(value) for key, value in saved["context_totals"].items()},
                    pair_counts={int(key): int(value) for key, value in saved["pair_counts"].items()},
                    n1={int(key): int(value) for key, value in saved["n1"].items()},
                    n2={int(key): int(value) for key, value in saved["n2"].items()},
                    n3p={int(key): int(value) for key, value in saved["n3p"].items()},
                )
            )
        self.unigram_continuations = np.array(state["unigram_continuations"], dtype=np.uint32, copy=True)
        self.total_unigram_continuations = int(state["total_unigram_continuations"])


class PPMExpert(Expert):
    def __init__(
        self,
        max_context: int,
        vocab_size: int,
        single_counting: bool = True,
        seed: int = 1469598103934665603,
        base: int = 1099511628211,
    ):
        if max_context < 1:
            raise ValueError("PPMExpert requires max_context >= 1")
        self.name = f"ppm_c_{max_context + 1}"
        self.context_len = max_context
        self.max_context = max_context
        self.vocab_size = vocab_size
        self.single_counting = single_counting
        self.seed = seed
        self.base = base
        self.mask = (1 << 64) - 1
        self.token_shift = max(1, (vocab_size - 1).bit_length())
        self.base_pows = [0] * (self.max_context + 1)
        for context_len in range(1, self.max_context + 1):
            self.base_pows[context_len] = pow(self.base, context_len - 1, 1 << 64)
        self.tables = [FollowerCountTable() for _ in range(self.max_context + 1)]
        self.sequence_keys: list[np.ndarray | None] = [None] * (self.max_context + 1)
        self.unigram_counts = np.zeros((vocab_size,), dtype=np.uint32)
        self.total_unigrams = 0
        self.total_distinct_unigrams = 0
        self.unigram_seen = np.zeros((vocab_size,), dtype=np.bool_)
        self.unigram_tokens: list[int] = []

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del position_offset
        for context_len in range(1, self.max_context + 1):
            if tokens.size <= context_len:
                self.sequence_keys[context_len] = None
                continue
            window_hash = 0
            for idx in range(context_len):
                window_hash = (window_hash * self.base + int(tokens[idx]) + 1) & self.mask
            keys = np.zeros((tokens.size,), dtype=np.uint64)
            keys[context_len] = window_hash ^ self.seed
            base_pow = self.base_pows[context_len]
            for pos in range(context_len + 1, tokens.size):
                old_token = int(tokens[pos - context_len - 1]) + 1
                new_token = int(tokens[pos - 1]) + 1
                window_hash = (window_hash - (old_token * base_pow)) & self.mask
                window_hash = (window_hash * self.base + new_token) & self.mask
                keys[pos] = window_hash ^ self.seed
            self.sequence_keys[context_len] = keys

    def _context_key(self, context_len: int, pos: int) -> int | None:
        if pos < context_len:
            return None
        keys = self.sequence_keys[context_len]
        if keys is None:
            return None
        return int(keys[pos])

    def _pair_key(self, context_key: int, token: int) -> int:
        return (context_key << self.token_shift) | token

    def _prob_zero_order(self, token: int, excluded: set[int]) -> float:
        available_total = 0
        available_distinct = 0
        token_count = 0
        for follower in self.unigram_tokens:
            if follower in excluded:
                continue
            count = int(self.unigram_counts[follower])
            available_total += count
            available_distinct += 1
            if follower == token:
                token_count = count
        if available_distinct == 0:
            remaining = self.vocab_size - self.total_distinct_unigrams
            return 1.0 / max(remaining, 1)
        denom = float(available_total + available_distinct)
        if token_count > 0:
            return float(token_count) / denom
        remaining = self.vocab_size - self.total_distinct_unigrams
        if remaining <= 0:
            return 1.0 / self.vocab_size
        return (float(available_distinct) / denom) * (1.0 / float(remaining))

    def _prob_order(self, context_len: int, pos: int, token: int, excluded: set[int]) -> float:
        if context_len <= 0:
            return self._prob_zero_order(token, excluded)

        context_key = self._context_key(context_len, pos)
        if context_key is None:
            return self._prob_order(context_len - 1, pos, token, excluded)

        table = self.tables[context_len]
        followers = table.followers.get(context_key)
        if not followers:
            return self._prob_order(context_len - 1, pos, token, excluded)

        available_total = 0
        available_distinct = 0
        token_count = 0
        for follower in followers:
            if follower in excluded:
                continue
            count = table.pair_counts[self._pair_key(context_key, follower)]
            available_total += count
            available_distinct += 1
            if follower == token:
                token_count = count

        if available_distinct == 0:
            excluded.update(followers)
            return self._prob_order(context_len - 1, pos, token, excluded)

        denom = float(available_total + available_distinct)
        if token_count > 0:
            return float(token_count) / denom

        excluded.update(followers)
        return (float(available_distinct) / denom) * self._prob_order(context_len - 1, pos, token, excluded)

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        del tokens
        max_depth = min(self.max_context, pos)
        return self._prob_order(max_depth, pos, token, excluded=set())

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        del tokens
        max_depth = min(self.max_context, pos)
        min_update = 0
        if self.single_counting:
            min_update = -1
            for context_len in range(max_depth, 0, -1):
                context_key = self._context_key(context_len, pos)
                if context_key is None:
                    continue
                if self.tables[context_len].pair_counts.get(self._pair_key(context_key, token), 0) > 0:
                    min_update = context_len
                    break
            if min_update < 0 and self.unigram_counts[token] > 0:
                min_update = 0
            if min_update < 0:
                min_update = 0

        for context_len in range(max_depth, 0, -1):
            if self.single_counting and context_len < min_update:
                break
            context_key = self._context_key(context_len, pos)
            if context_key is None:
                continue
            table = self.tables[context_len]
            pair_key = self._pair_key(context_key, token)
            prev = table.pair_counts.get(pair_key, 0)
            if prev == 0:
                table.followers.setdefault(context_key, []).append(token)
            table.pair_counts[pair_key] = prev + 1
            table.context_totals[context_key] = table.context_totals.get(context_key, 0) + 1

        if not self.single_counting or min_update == 0:
            if not self.unigram_seen[token]:
                self.unigram_seen[token] = True
                self.unigram_tokens.append(token)
                self.total_distinct_unigrams += 1
            self.unigram_counts[token] += 1
            self.total_unigrams += 1

    def state_dict(self) -> dict:
        tables = []
        for table in self.tables:
            tables.append(
                {
                    "context_totals": dict(table.context_totals),
                    "pair_counts": dict(table.pair_counts),
                    "followers": {key: list(value) for key, value in table.followers.items()},
                }
            )
        return {
            "tables": tables,
            "unigram_counts": self.unigram_counts.copy(),
            "total_unigrams": self.total_unigrams,
            "total_distinct_unigrams": self.total_distinct_unigrams,
            "unigram_seen": self.unigram_seen.copy(),
            "unigram_tokens": list(self.unigram_tokens),
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        self.tables = []
        for saved in state["tables"]:
            self.tables.append(
                FollowerCountTable(
                    context_totals={int(key): int(value) for key, value in saved["context_totals"].items()},
                    pair_counts={int(key): int(value) for key, value in saved["pair_counts"].items()},
                    followers={int(key): [int(token) for token in value] for key, value in saved["followers"].items()},
                )
            )
        self.unigram_counts = np.array(state["unigram_counts"], dtype=np.uint32, copy=True)
        self.total_unigrams = int(state["total_unigrams"])
        self.total_distinct_unigrams = int(state["total_distinct_unigrams"])
        self.unigram_seen = np.array(state["unigram_seen"], dtype=np.bool_, copy=True)
        self.unigram_tokens = [int(token) for token in state["unigram_tokens"]]


class SlidingWindowExpert(Expert):
    def __init__(
        self,
        vocab_size: int,
        window: int,
        backoff: Expert,
        alpha: float = 1.0,
        reset_token: int | None = None,
        name_prefix: str = "cache",
    ):
        self.name = f"{name_prefix}_{window}"
        self.vocab_size = vocab_size
        self.window = window
        self.backoff = backoff
        self.alpha = alpha
        self.reset_token = reset_token
        self.counts = np.zeros((vocab_size,), dtype=np.uint32)
        self.items: deque[int] = deque()

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        total = float(len(self.items))
        hit_count = float(self.counts[token])
        return (hit_count + self.alpha * base_prob) / (total + self.alpha)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        del tokens, pos
        if self.reset_token is not None and token == self.reset_token:
            self.reset_state()
            return
        self.items.append(token)
        self.counts[token] += 1
        if len(self.items) > self.window:
            old = self.items.popleft()
            self.counts[old] -= 1

    def reset_state(self) -> None:
        self.counts.fill(0)
        self.items.clear()


class RecentMatchExpert(Expert):
    def __init__(
        self,
        context_len: int,
        max_gap: int,
        max_matches: int,
        max_stored_matches: int,
        backoff: Expert,
        alpha: float = 1.0,
        decay_power: float = 0.6,
        reset_token: int | None = None,
        name_prefix: str = "copy_ctx",
    ):
        self.name = f"{name_prefix}{context_len}"
        self.context_len = context_len
        self.max_gap = max_gap
        self.max_matches = max_matches
        self.backoff = backoff
        self.alpha = alpha
        self.decay_power = decay_power
        self.reset_token = reset_token
        self.index: dict[int, deque[tuple[int, int]]] = defaultdict(deque)
        self.max_stored_matches = max_stored_matches
        self.seed = 1469598103934665603
        self.base = 1099511628211
        self.mask = (1 << 64) - 1
        self.base_pow = pow(self.base, self.context_len - 1, 1 << 64)
        self.sequence_keys: np.ndarray | None = None
        self.position_offset = 0

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        self.position_offset = position_offset
        if tokens.size <= self.context_len:
            self.sequence_keys = None
            return
        window_hash = 0
        for idx in range(self.context_len):
            window_hash = (window_hash * self.base + int(tokens[idx]) + 1) & self.mask
        keys = np.zeros((tokens.size,), dtype=np.uint64)
        keys[self.context_len] = window_hash ^ self.seed
        for pos in range(self.context_len + 1, tokens.size):
            old_token = int(tokens[pos - self.context_len - 1]) + 1
            new_token = int(tokens[pos - 1]) + 1
            window_hash = (window_hash - (old_token * self.base_pow)) & self.mask
            window_hash = (window_hash * self.base + new_token) & self.mask
            keys[pos] = window_hash ^ self.seed
        self.sequence_keys = keys

    def _context_key(self, tokens: np.ndarray, pos: int) -> int | None:
        if pos < self.context_len:
            return None
        del tokens
        if self.sequence_keys is None:
            return None
        return int(self.sequence_keys[pos])

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        key = self._context_key(tokens, pos)
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        if key is None:
            return base_prob

        matches = self.index.get(key)
        if not matches:
            return base_prob

        absolute_pos = self.position_offset + pos
        min_pos = absolute_pos - self.max_gap
        weighted_hits = 0.0
        weighted_total = 0.0
        used = 0

        for hit_pos, next_token in reversed(matches):
            if hit_pos < min_pos:
                break
            gap = absolute_pos - hit_pos
            weight = 1.0 / math.pow(gap + 1.0, self.decay_power)
            weighted_total += weight
            if next_token == token:
                weighted_hits += weight
            used += 1
            if used >= self.max_matches:
                break

        if weighted_total == 0.0:
            return base_prob
        return (weighted_hits + self.alpha * base_prob) / (weighted_total + self.alpha)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        if self.reset_token is not None and token == self.reset_token:
            self.reset_state()
            return
        key = self._context_key(tokens, pos)
        if key is None:
            return

        entries = self.index[key]
        absolute_pos = self.position_offset + pos
        entries.append((absolute_pos, token))
        min_pos = absolute_pos - self.max_gap
        while entries and entries[0][0] < min_pos:
            entries.popleft()
        while len(entries) > self.max_stored_matches:
            entries.popleft()

    def reset_state(self) -> None:
        self.index.clear()


class AdaptiveMixer:
    def __init__(self, experts: list[Expert], eta: float, share: float, min_prob: float):
        if not experts:
            raise ValueError("Need at least one expert")
        self.experts = experts
        self.eta = eta
        self.share = share
        self.min_prob = min_prob
        self.weights = np.full((len(experts),), 1.0 / len(experts), dtype=np.float64)
        self.prior = self.weights.copy()
        self.expert_logloss = np.zeros((len(experts),), dtype=np.float64)

    def step(self, tokens: np.ndarray, pos: int, token: int) -> tuple[float, np.ndarray]:
        probs = np.empty((len(self.experts),), dtype=np.float64)
        prob_cache: dict[tuple[int, int], float] = {}
        for idx, expert in enumerate(self.experts):
            probs[idx] = max(self.min_prob, expert.cached_prob(tokens, pos, token, prob_cache))

        mix_prob = float(np.dot(self.weights, probs))
        mix_prob = max(mix_prob, self.min_prob)
        self.expert_logloss += -np.log2(probs)

        posterior = self.weights * np.power(probs, self.eta)
        total = float(posterior.sum())
        if not np.isfinite(total) or total <= 0.0:
            posterior = self.prior.copy()
        else:
            posterior /= total
        self.weights = (1.0 - self.share) * posterior + self.share * self.prior

        for expert in self.experts:
            expert.update(tokens, pos, token)

        return mix_prob, probs

    def observe(self, tokens: np.ndarray, pos: int, token: int) -> None:
        for expert in self.experts:
            expert.update(tokens, pos, token)

    def reset_weights(self) -> None:
        self.weights = self.prior.copy()
        self.expert_logloss.fill(0.0)

    def reset_ephemeral_state(self) -> None:
        for expert in self.experts:
            expert.reset_state()

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del position_offset
        for expert in self.experts:
            expert.set_sequence(tokens)

    def set_sequence_with_offset(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        for expert in self.experts:
            expert.set_sequence(tokens, position_offset=position_offset)

    def state_dict(self) -> dict:
        return {
            "expert_names": [expert.name for expert in self.experts],
            "weights": self.weights.copy(),
            "prior": self.prior.copy(),
            "expert_logloss": self.expert_logloss.copy(),
            "experts": [expert.state_dict() for expert in self.experts],
        }

    def load_state_dict(self, state: dict) -> None:
        expected_names = [expert.name for expert in self.experts]
        saved_names = list(state["expert_names"])
        if saved_names != expected_names:
            raise ValueError(f"State expert mismatch: saved={saved_names} current={expected_names}")
        self.weights = np.array(state["weights"], dtype=np.float64, copy=True)
        self.prior = np.array(state["prior"], dtype=np.float64, copy=True)
        self.expert_logloss = np.array(state["expert_logloss"], dtype=np.float64, copy=True)
        for expert, expert_state in zip(self.experts, state["experts"], strict=True):
            expert.load_state_dict(expert_state)


class CompositeLMExpert(Expert):
    def __init__(self, prob_expert: Expert, components: list[Expert]):
        self.name = prob_expert.name
        self.prob_expert = prob_expert
        self.components = components
        self.context_len = getattr(prob_expert, "context_len", 0)

    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        for expert in self.components:
            expert.set_sequence(tokens, position_offset=position_offset)

    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: dict[tuple[int, int], float] | None = None,
    ) -> float:
        return self.prob_expert.cached_prob(tokens, pos, token, cache)

    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        for expert in self.components:
            expert.update(tokens, pos, token)

    def reset_state(self) -> None:
        for expert in self.components:
            expert.reset_state()

    def state_dict(self) -> dict:
        return {
            "component_names": [expert.name for expert in self.components],
            "component_states": [expert.state_dict() for expert in self.components],
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:
            return
        expected = [expert.name for expert in self.components]
        saved = list(state["component_names"])
        if saved != expected:
            raise ValueError(f"Composite state mismatch: saved={saved} current={expected}")
        for expert, expert_state in zip(self.components, state["component_states"], strict=True):
            expert.load_state_dict(expert_state)


def build_experts(args: argparse.Namespace, vocab_size: int) -> list[Expert]:
    unigram = UnigramExpert(vocab_size=vocab_size, alpha=args.unigram_alpha)
    lm_components: list[Expert] = [unigram]
    experts: list[Expert] = []

    if args.ppm:
        max_context = max(args.ngram_contexts) if args.ngram_contexts else 1
        lm_backoff = PPMExpert(
            max_context=max_context,
            vocab_size=vocab_size,
            single_counting=bool(args.ppm_single_counting),
        )
        lm_components.append(lm_backoff)
    elif args.modified_kn:
        discounts = args.mkn_discounts
        if len(discounts) == 1:
            discounts = [discounts[0], discounts[0], discounts[0]]
        if len(discounts) != 3:
            raise ValueError("--mkn-discounts must provide either 1 or 3 values")
        max_context = max(args.ngram_contexts) if args.ngram_contexts else 1
        lm_backoff = ModifiedKneserNeyExpert(
            max_context=max_context,
            vocab_size=vocab_size,
            base_unigram=unigram,
            unigram_alpha=args.mkn_unigram_alpha,
            discounts=(discounts[0], discounts[1], discounts[2]),
        )
        lm_components.append(lm_backoff)
    else:
        bigram = BigramExpert(
            vocab_size=vocab_size,
            backoff=unigram,
            alpha=args.bigram_alpha,
            discount=args.absolute_discount,
            use_continuation_unigram=bool(args.continuation_unigram),
        )

        lm_components.append(bigram)
        lm_backoff = bigram
        for context_len in args.ngram_contexts:
            ngram = HashedNgramExpert(
                context_len=context_len,
                vocab_size=vocab_size,
                backoff=lm_backoff,
                alpha=args.ngram_alpha,
                discount=args.absolute_discount,
                count_scale_limit=args.ngram_count_scale_limit,
            )
            lm_components.append(ngram)
            lm_backoff = ngram

    if args.mix_backoff_experts:
        experts.extend(lm_components)
    else:
        experts.append(CompositeLMExpert(prob_expert=lm_backoff, components=lm_components))

    for window in args.cache_windows:
        experts.append(
            SlidingWindowExpert(
                vocab_size=vocab_size,
                window=window,
                backoff=lm_backoff,
                alpha=args.cache_alpha,
            )
        )
    for window in args.doc_cache_windows:
        experts.append(
            SlidingWindowExpert(
                vocab_size=vocab_size,
                window=window,
                backoff=lm_backoff,
                alpha=args.cache_alpha,
                reset_token=args.doc_reset_token,
                name_prefix="doc_cache",
            )
        )
    for context_len in args.copy_contexts:
        experts.append(
            RecentMatchExpert(
                context_len=context_len,
                max_gap=args.copy_window,
                max_matches=args.copy_max_matches,
                max_stored_matches=args.copy_store_limit,
                backoff=lm_backoff,
                alpha=args.copy_alpha,
                decay_power=args.copy_decay_power,
            )
        )
    for context_len in args.doc_copy_contexts:
        experts.append(
            RecentMatchExpert(
                context_len=context_len,
                max_gap=args.copy_window,
                max_matches=args.copy_max_matches,
                max_stored_matches=args.copy_store_limit,
                backoff=lm_backoff,
                alpha=args.copy_alpha,
                decay_power=args.copy_decay_power,
                reset_token=args.doc_reset_token,
                name_prefix="doc_copy_ctx",
            )
        )
    return experts


def parse_csv_ints(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(piece) for piece in raw.split(",") if piece.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(piece) for piece in raw.split(",") if piece.strip()]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a non-neural token compressor on challenge val data")
    parser.add_argument(
        "--data-pattern",
        default="data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        help="Glob for validation shards.",
    )
    parser.add_argument(
        "--train-pattern",
        default="",
        help="Optional glob for training shards used only for warm start.",
    )
    parser.add_argument(
        "--load-state",
        default="",
        help="Optional zlib-compressed model-state path to load before warmup/eval.",
    )
    parser.add_argument(
        "--save-state",
        default="",
        help="Optional zlib-compressed model-state path to write after warmup and before validation.",
    )
    parser.add_argument(
        "--skip-validation",
        type=int,
        default=0,
        help="If set, stop after optional warmup/state export without loading validation shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="data/tokenizers/fineweb_1024_bpe.model",
        help="SentencePiece model path used for tokenizer-aware byte accounting.",
    )
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000,
        help="Optional cap on loaded tokens for local iteration. Includes the initial BOS token. Use 0 for full validation.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help="Number of training tokens to feed into the model before validation scoring.",
    )
    parser.add_argument(
        "--warmup-adapt-weights",
        type=int,
        default=0,
        help="If set, also adapt mixture weights during warm start instead of only updating expert state.",
    )
    parser.add_argument(
        "--reset-weights-after-warmup",
        type=int,
        default=1,
        help="If set, reset mixture weights before validation after warm start.",
    )
    parser.add_argument(
        "--reset-ephemeral-after-warmup",
        type=int,
        default=1,
        help="If set, clear recency-only expert state before validation after warm start.",
    )
    parser.add_argument("--report-every", type=int, default=100_000)
    parser.add_argument("--eta", type=float, default=0.7, help="Exponentiated-gradient update strength.")
    parser.add_argument("--share", type=float, default=0.03, help="Fixed-share reset rate for expert weights.")
    parser.add_argument("--min-prob", type=float, default=1e-12)
    parser.add_argument("--unigram-alpha", type=float, default=0.5)
    parser.add_argument("--bigram-alpha", type=float, default=4.0)
    parser.add_argument("--ngram-alpha", type=float, default=1.0)
    parser.add_argument("--ngram-count-scale-limit", type=int, default=0)
    parser.add_argument(
        "--mix-backoff-experts",
        type=int,
        default=1,
        help="If set, include unigram/bigram/intermediate n-grams as separate experts in the adaptive mixture.",
    )
    parser.add_argument("--absolute-discount", type=float, default=0.0)
    parser.add_argument("--continuation-unigram", type=int, default=1)
    parser.add_argument("--ppm", type=int, default=0)
    parser.add_argument("--ppm-single-counting", type=int, default=1)
    parser.add_argument("--modified-kn", type=int, default=0)
    parser.add_argument("--mkn-unigram-alpha", type=float, default=1.0)
    parser.add_argument("--mkn-discounts", type=parse_csv_floats, default=parse_csv_floats("0.7,1.0,1.3"))
    parser.add_argument("--cache-alpha", type=float, default=1.0)
    parser.add_argument("--copy-alpha", type=float, default=1.0)
    parser.add_argument("--copy-decay-power", type=float, default=0.6)
    parser.add_argument("--ngram-contexts", type=parse_csv_ints, default=parse_csv_ints("2,3,4"))
    parser.add_argument("--cache-windows", type=parse_csv_ints, default=parse_csv_ints("64,512,4096,32768"))
    parser.add_argument("--copy-contexts", type=parse_csv_ints, default=parse_csv_ints("2,4,8,12"))
    parser.add_argument("--doc-cache-windows", type=parse_csv_ints, default=parse_csv_ints(""))
    parser.add_argument("--doc-copy-contexts", type=parse_csv_ints, default=parse_csv_ints(""))
    parser.add_argument("--doc-reset-token", type=int, default=1)
    parser.add_argument("--copy-window", type=int, default=200_000)
    parser.add_argument("--copy-max-matches", type=int, default=32)
    parser.add_argument("--copy-store-limit", type=int, default=32)
    return parser


def max_required_context(experts: list[Expert]) -> int:
    max_context = 1
    for expert in experts:
        context_len = getattr(expert, "context_len", 0)
        if context_len > max_context:
            max_context = context_len
    return max_context


STATE_VERSION = 1


def save_mixer_state(path: Path, args: argparse.Namespace, mixer: AdaptiveMixer, warmup_seen: int) -> int:
    payload = {
        "version": STATE_VERSION,
        "args": vars(args).copy(),
        "warmup_seen": int(warmup_seen),
        "mixer": mixer.state_dict(),
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, level=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return len(compressed)


def load_mixer_state(path: Path, mixer: AdaptiveMixer) -> tuple[int, dict]:
    compressed = path.read_bytes()
    payload = pickle.loads(zlib.decompress(compressed))
    if int(payload.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported state version in {path}")
    mixer.load_state_dict(payload["mixer"])
    return len(compressed), payload


def warmup_mixer(args: argparse.Namespace, mixer: AdaptiveMixer) -> tuple[int, float]:
    if not args.train_pattern or args.warmup_tokens <= 0:
        return 0, 0.0

    files = [Path(p) for p in sorted(glob.glob(args.train_pattern))]
    if not files:
        raise FileNotFoundError(f"No training files found for pattern: {args.train_pattern}")

    max_context = max_required_context(mixer.experts)
    remaining = int(args.warmup_tokens)
    total_seen = 0
    tail = np.empty((0,), dtype=np.uint16)
    start = time.perf_counter()

    for path in files:
        if remaining <= 0:
            break
        shard_tokens = load_data_shard(path)
        if shard_tokens.size > remaining:
            shard_tokens = shard_tokens[:remaining]
        if tail.size:
            tokens = np.concatenate((tail, shard_tokens))
            start_pos = tail.size
        else:
            tokens = shard_tokens
            start_pos = 1
        position_offset = total_seen + 1 - start_pos
        mixer.set_sequence_with_offset(tokens, position_offset=position_offset)

        for pos in range(start_pos, tokens.size):
            token = int(tokens[pos])
            if args.warmup_adapt_weights:
                mixer.step(tokens, pos, token)
            else:
                mixer.observe(tokens, pos, token)
            total_seen += 1

        remaining -= int(shard_tokens.size)
        if tokens.size > max_context:
            tail = tokens[-max_context:].copy()
        else:
            tail = tokens.copy()

    if args.reset_weights_after_warmup:
        mixer.reset_weights()
    if args.reset_ephemeral_after_warmup:
        mixer.reset_ephemeral_state()

    return total_seen, time.perf_counter() - start


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    start = time.perf_counter()
    experts = build_experts(args, args.vocab_size)
    mixer = AdaptiveMixer(experts=experts, eta=args.eta, share=args.share, min_prob=args.min_prob)
    loaded_state_bytes = 0
    loaded_warmup_seen = 0
    if args.load_state:
        loaded_state_bytes, payload = load_mixer_state(Path(args.load_state), mixer)
        loaded_warmup_seen = int(payload.get("warmup_seen", 0))

    warmup_seen, warmup_elapsed = warmup_mixer(args, mixer)
    total_warmup_seen = loaded_warmup_seen + warmup_seen
    saved_state_bytes = 0
    if args.save_state:
        saved_state_bytes = save_mixer_state(Path(args.save_state), args, mixer, total_warmup_seen)

    if args.skip_validation:
        if loaded_state_bytes:
            print(f"loaded_state_bytes={loaded_state_bytes}")
        if loaded_warmup_seen:
            print(f"loaded_warmup_predictions={loaded_warmup_seen}")
        if warmup_seen:
            print(f"warmup_predictions={warmup_seen}")
            print(f"warmup_elapsed_seconds={warmup_elapsed:.2f}")
        if saved_state_bytes:
            print(f"saved_state_bytes={saved_state_bytes}")
        print(f"elapsed_seconds={time.perf_counter() - start:.2f}")
        return

    max_tokens = args.max_tokens if args.max_tokens > 0 else None
    tokens = load_validation_tokens(args.data_pattern, max_tokens)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes, has_leading_space, is_boundary_token = build_sentencepiece_luts(sp, args.vocab_size)
    target_bytes = compute_target_byte_counts(tokens, base_bytes, has_leading_space, is_boundary_token)

    total_bits = 0.0
    total_bytes = 0.0
    processed = 0
    loop_start = time.perf_counter()
    mixer.set_sequence_with_offset(tokens, position_offset=total_warmup_seen)

    for pos in range(1, tokens.size):
        token = int(tokens[pos])
        mix_prob, _ = mixer.step(tokens, pos, token)
        total_bits += -math.log2(mix_prob)
        total_bytes += float(target_bytes[pos - 1])
        processed += 1

        if args.report_every and processed % args.report_every == 0:
            elapsed = time.perf_counter() - loop_start
            bpb = total_bits / max(total_bytes, 1.0)
            tps = processed / max(elapsed, 1e-9)
            top = np.argsort(mixer.weights)[::-1][:5]
            summary = ", ".join(f"{experts[i].name}={mixer.weights[i]:.3f}" for i in top)
            print(
                f"step={processed} bpb={bpb:.6f} tok_per_s={tps:,.0f} "
                f"weights=[{summary}]"
            , flush=True)

    elapsed = time.perf_counter() - start
    if loaded_state_bytes:
        print(f"loaded_state_bytes={loaded_state_bytes}")
    if loaded_warmup_seen:
        print(f"loaded_warmup_predictions={loaded_warmup_seen}")
    if warmup_seen:
        print(f"warmup_predictions={warmup_seen}")
        print(f"warmup_elapsed_seconds={warmup_elapsed:.2f}")
    if saved_state_bytes:
        print(f"saved_state_bytes={saved_state_bytes}")
    bpb = total_bits / total_bytes
    print(f"tokens_loaded={tokens.size}")
    print(f"predictions={processed}")
    print(f"total_bytes={int(total_bytes)}")
    print(f"val_bpb={bpb:.8f}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print("expert_weights:")
    for idx in np.argsort(mixer.weights)[::-1]:
        print(f"  {experts[idx].name}: weight={mixer.weights[idx]:.6f} avg_logloss_bits={mixer.expert_logloss[idx]/processed:.6f}")


if __name__ == "__main__":
    main()
