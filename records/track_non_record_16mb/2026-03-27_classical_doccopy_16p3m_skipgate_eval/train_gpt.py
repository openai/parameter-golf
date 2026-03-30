#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import lzma
import math
import pickle
import time
import zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import sentencepiece as spm
REPO_ROOT = Path(__file__).resolve().parents[3]
HEADER_INTS = 256
HEADER_MAGIC = 20240520
HEADER_VERSION = 1
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")
def resolve_repo_path(raw: str) -> str:
    if not raw:
        return raw
    path = Path(raw)
    if path.is_absolute():
        return raw
    return str(REPO_ROOT / raw)
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
def build_sequence_keys(tokens: np.ndarray, context_len: int, base: int, seed: int, mask: int) -> np.ndarray | None:
    if tokens.size <= context_len:
        return None
    span = tokens.size - context_len
    values = tokens.astype(np.uint64, copy=False) + np.uint64(1)
    hashes = values[:span].copy()
    base_u64 = np.uint64(base)
    mask_u64 = np.uint64(mask)
    for offset in range(1, context_len):
        hashes = (hashes * base_u64 + values[offset : offset + span]) & mask_u64
    keys = np.zeros((tokens.size,), dtype=np.uint64)
    keys[context_len:] = hashes ^ np.uint64(seed)
    return keys
def build_skip_sequence_keys(tokens: np.ndarray, offsets: tuple[int, ...], base: int, seed: int, mask: int) -> np.ndarray | None:
    if not offsets:
        return None
    max_offset = max(offsets)
    if tokens.size <= max_offset:
        return None
    span = tokens.size - max_offset
    values = tokens.astype(np.uint64, copy=False) + np.uint64(1)
    hashes = np.full((span,), np.uint64(seed), dtype=np.uint64)
    base_u64 = np.uint64(base)
    mask_u64 = np.uint64(mask)
    for offset in offsets:
        hashes = (hashes * base_u64 + values[max_offset - offset : tokens.size - offset]) & mask_u64
    keys = np.zeros((tokens.size,), dtype=np.uint64)
    keys[max_offset:] = hashes
    return keys
def pack_u10_tokens(tokens: np.ndarray) -> tuple[bytes, int]:
    size = int(tokens.size)
    if size == 0:
        return b"", 0
    arr = np.asarray(tokens, dtype=np.uint16)
    pad = (-size) % 4
    if pad:
        arr = np.pad(arr, (0, pad))
    g = arr.reshape(-1, 4)
    out = np.empty((g.shape[0], 5), dtype=np.uint8)
    out[:, 0] = g[:, 0] >> 2; out[:, 1] = ((g[:, 0] & 0x3) << 6) | (g[:, 1] >> 4); out[:, 2] = ((g[:, 1] & 0xF) << 4) | (g[:, 2] >> 6)
    out[:, 3] = ((g[:, 2] & 0x3F) << 2) | (g[:, 3] >> 8); out[:, 4] = g[:, 3] & 0xFF
    return out.tobytes(), size
def unpack_u10_tokens(packed: bytes, size: int) -> np.ndarray:
    if size <= 0:
        return np.empty((0,), dtype=np.uint16)
    buf = np.frombuffer(packed, dtype=np.uint8)
    if buf.size % 5 != 0:
        raise ValueError("Packed token buffer is not a multiple of 5 bytes")
    g = buf.reshape(-1, 5)
    out = np.empty((g.shape[0], 4), dtype=np.uint16)
    out[:, 0] = (g[:, 0].astype(np.uint16) << 2) | (g[:, 1] >> 6); out[:, 1] = ((g[:, 1].astype(np.uint16) & 0x3F) << 4) | (g[:, 2] >> 4)
    out[:, 2] = ((g[:, 2].astype(np.uint16) & 0x0F) << 6) | (g[:, 3] >> 2); out[:, 3] = ((g[:, 3].astype(np.uint16) & 0x03) << 8) | g[:, 4]
    return out.reshape(-1)[:size].copy()
def compress_state_payload(raw: bytes, codec: str) -> bytes:
    return lzma.compress(raw, preset=6) if codec == "lzma" else zlib.compress(raw, level=9)
def decompress_state_payload(blob: bytes) -> bytes:
    return lzma.decompress(blob) if blob.startswith(b"\xfd7zXZ\x00") else zlib.decompress(blob)
class Expert:
    name: str
    supports_online_sequence = False
    cache_slot = 0
    frozen_updates = False
    is_lm_expert = False
    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del tokens, position_offset
        return None
    def begin_sequence(self, position_offset: int = 0) -> None:
        del position_offset
        return None
    def prime(self, token: int) -> None:
        del token
        return None
    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: list[float] | None = None,
    ) -> float:
        raise NotImplementedError
    def cached_prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: list[float] | None,
    ) -> float:
        if cache is None:
            return self.prob(tokens, pos, token, None)
        cached = cache[self.cache_slot]
        if cached == cached:
            return cached
        value = self.prob(tokens, pos, token, cache)
        cache[self.cache_slot] = value
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
class UnigramExpert(Expert):
    supports_online_sequence = True
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
        cache: list[float] | None = None,
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
    supports_online_sequence = True
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
        cache: list[float] | None = None,
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
        prune_min_count: int = 0,
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
        self.prune_min_count = prune_min_count
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
        self.sequence_keys = build_sequence_keys(
            tokens=tokens,
            context_len=self.context_len,
            base=self.base,
            seed=self.seed,
            mask=self.mask,
        )
    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: list[float] | None = None,
    ) -> float:
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        del tokens
        sequence_keys = self.sequence_keys
        if sequence_keys is None or pos < self.context_len:
            return base_prob
        context_key = int(sequence_keys[pos])
        context_totals = self.context_totals
        total = float(context_totals.get(context_key, 0))
        if total <= 0.0:
            return base_prob
        pair_counts = self.pair_counts
        pair_key = (context_key << self.token_shift) | token
        count = float(pair_counts.get(pair_key, 0))
        if self.discount > 0.0:
            distinct = float(self.distinct_followers.get(context_key, 0))
            backoff_mass = self.discount * distinct / total
            discounted = max(count - self.discount, 0.0) / total
            return discounted + backoff_mass * base_prob
        return (count + self.alpha * base_prob) / (total + self.alpha)
    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        del tokens
        sequence_keys = self.sequence_keys
        if sequence_keys is None or pos < self.context_len:
            return
        context_key = int(sequence_keys[pos])
        pair_key = (context_key << self.token_shift) | token
        pair_counts = self.pair_counts
        prev = pair_counts.get(pair_key, 0)
        if prev == 0:
            self.distinct_followers[context_key] = self.distinct_followers.get(context_key, 0) + 1
            if self.count_scale_limit > 0:
                self.followers.setdefault(context_key, []).append(token)
        pair_counts[pair_key] = prev + 1
        context_totals = self.context_totals
        new_total = context_totals.get(context_key, 0) + 1
        context_totals[context_key] = new_total
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
        if self.prune_min_count > 1:
            items = [(pair_key, count) for pair_key, count in self.pair_counts.items() if count >= self.prune_min_count]
        else:
            items = list(self.pair_counts.items())
        size = len(items)
        contexts = np.empty((size,), dtype=np.uint64)
        tokens = np.empty((size,), dtype=np.uint16)
        counts = np.empty((size,), dtype=np.uint32)
        token_mask = (1 << self.token_shift) - 1
        for idx, (pair_key, count) in enumerate(items):
            contexts[idx] = pair_key >> self.token_shift
            tokens[idx] = pair_key & token_mask
            counts[idx] = count
        if size:
            order = np.argsort(contexts, kind="stable")
            contexts = contexts[order]
            tokens = tokens[order]
            counts = counts[order]
        packed_tokens, token_count = pack_u10_tokens(tokens)
        if size and int(counts.max()) <= np.iinfo(np.uint16).max: counts = counts.astype(np.uint16, copy=False)
        return {
            "pair_contexts": contexts,
            "pair_tokens_packed": packed_tokens,
            "pair_tokens_count": token_count,
            "pair_counts": counts,
        }
    def load_state_dict(self, state: dict | None) -> None:
        if state is None: return
        pair_contexts = np.asarray(state["pair_contexts"], dtype=np.uint64)
        pair_tokens = unpack_u10_tokens(state["pair_tokens_packed"], int(state["pair_tokens_count"])) if "pair_tokens_packed" in state else np.asarray(state["pair_tokens"], dtype=np.uint16)
        pair_counts = np.asarray(state["pair_counts"], dtype=np.uint32)
        if self.prune_min_count > 1:
            keep = pair_counts >= self.prune_min_count
            pair_contexts = pair_contexts[keep]
            pair_tokens = pair_tokens[keep]
            pair_counts = pair_counts[keep]
        self.followers = {}
        if pair_counts.size == 0:
            self.context_totals = {}
            self.pair_counts = {}
            self.distinct_followers = {}
            return
        context_key_list = pair_contexts.tolist()
        token_list = pair_tokens.tolist()
        count_list = pair_counts.tolist()
        pair_keys = [(context_key << self.token_shift) | token for context_key, token in zip(context_key_list, token_list, strict=True)]
        self.pair_counts = dict(zip(pair_keys, count_list, strict=True))
        unique_contexts, first_idx, distinct_counts = np.unique(
            pair_contexts,
            return_index=True,
            return_counts=True,
        )
        totals = np.add.reduceat(pair_counts.astype(np.uint64, copy=False), first_idx)
        context_list = unique_contexts.tolist()
        self.context_totals = dict(zip(context_list, totals.tolist(), strict=True))
        self.distinct_followers = dict(zip(context_list, distinct_counts.tolist(), strict=True))
        if self.count_scale_limit > 0:
            followers: dict[int, list[int]] = {}
            for ctx, tok in zip(context_key_list, token_list, strict=True):
                followers.setdefault(ctx, []).append(tok)
            self.followers = followers
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
        self.index: dict[int, list[tuple[int, int]]] = {}
        self.max_stored_matches = max_stored_matches
        self.seed = 1469598103934665603
        self.base = 1099511628211
        self.mask = (1 << 64) - 1
        self.base_pow = pow(self.base, self.context_len - 1, 1 << 64)
        self.sequence_keys: np.ndarray | None = None
        self.position_offset = 0
        self.decay_weights = np.power(np.arange(1, self.max_gap + 2, dtype=np.float64), -self.decay_power)
    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        self.position_offset = position_offset
        self.sequence_keys = build_sequence_keys(
            tokens=tokens,
            context_len=self.context_len,
            base=self.base,
            seed=self.seed,
            mask=self.mask,
        )
    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: list[float] | None = None,
    ) -> float:
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache)
        del tokens
        sequence_keys = self.sequence_keys
        if sequence_keys is None or pos < self.context_len:
            return base_prob
        key = int(sequence_keys[pos])
        matches = self.index.get(key)
        if not matches:
            return base_prob
        absolute_pos = self.position_offset + pos
        min_pos = absolute_pos - self.max_gap
        weighted_hits = 0.0
        weighted_total = 0.0
        used = 0
        decay_weights = self.decay_weights
        for hit_pos, next_token in reversed(matches):
            if hit_pos < min_pos:
                break
            gap = absolute_pos - hit_pos
            weight = float(decay_weights[gap])
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
        del tokens
        sequence_keys = self.sequence_keys
        if sequence_keys is None or pos < self.context_len:
            return
        key = int(sequence_keys[pos])
        entries = self.index.get(key)
        if entries is None:
            entries = []
            self.index[key] = entries
        absolute_pos = self.position_offset + pos
        entries.append((absolute_pos, token))
        min_pos = absolute_pos - self.max_gap
        while entries and entries[0][0] < min_pos:
            entries.pop(0)
        while len(entries) > self.max_stored_matches:
            entries.pop(0)
    def reset_state(self) -> None:
        self.index.clear()
class SkipRecentMatchExpert(Expert):
    def __init__(self, offsets: tuple[int, ...], max_gap: int, max_matches: int, max_stored_matches: int, backoff: Expert, alpha: float = 1.0, decay_power: float = 0.6, reset_token: int | None = None, name_prefix: str = "skip_copy"):
        ordered_offsets = tuple(sorted(int(offset) for offset in offsets))
        if not ordered_offsets: raise ValueError("SkipRecentMatchExpert requires at least one offset")
        self.offsets = ordered_offsets; self.name = f"{name_prefix}_{'-'.join(str(offset) for offset in ordered_offsets)}"; self.context_len = max(ordered_offsets)
        self.max_gap = max_gap; self.max_matches = max_matches; self.backoff = backoff; self.alpha = alpha; self.decay_power = decay_power; self.reset_token = reset_token
        self.index: dict[int, list[tuple[int, int]]] = {}; self.max_stored_matches = max_stored_matches
        self.seed = 1469598103934665603 ^ (self.context_len << 9) ^ len(ordered_offsets); self.base = 1099511628211; self.mask = (1 << 64) - 1
        self.sequence_keys: np.ndarray | None = None; self.position_offset = 0
        self.decay_weights = np.power(np.arange(1, self.max_gap + 2, dtype=np.float64), -self.decay_power)
    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        self.position_offset = position_offset
        self.sequence_keys = build_skip_sequence_keys(tokens=tokens, offsets=self.offsets, base=self.base, seed=self.seed, mask=self.mask)
    def prob(self, tokens: np.ndarray, pos: int, token: int, cache: list[float] | None = None) -> float:
        base_prob = self.backoff.cached_prob(tokens, pos, token, cache); del tokens
        if self.sequence_keys is None or pos < self.context_len: return base_prob
        matches = self.index.get(int(self.sequence_keys[pos]))
        if not matches: return base_prob
        absolute_pos = self.position_offset + pos; min_pos = absolute_pos - self.max_gap; weighted_hits = weighted_total = 0.0; used = 0
        for hit_pos, next_token in reversed(matches):
            if hit_pos < min_pos: break
            gap = absolute_pos - hit_pos; weight = float(self.decay_weights[gap]); weighted_total += weight
            if next_token == token: weighted_hits += weight
            used += 1
            if used >= self.max_matches: break
        return base_prob if weighted_total == 0.0 else (weighted_hits + self.alpha * base_prob) / (weighted_total + self.alpha)
    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        if self.reset_token is not None and token == self.reset_token: self.reset_state(); return
        del tokens
        if self.sequence_keys is None or pos < self.context_len: return
        key = int(self.sequence_keys[pos]); entries = self.index.get(key)
        if entries is None: entries = []; self.index[key] = entries
        absolute_pos = self.position_offset + pos; entries.append((absolute_pos, token)); min_pos = absolute_pos - self.max_gap
        while entries and entries[0][0] < min_pos: entries.pop(0)
        while len(entries) > self.max_stored_matches: entries.pop(0)
    def reset_state(self) -> None:
        self.index.clear()
class AdaptiveMixer:
    def __init__(self, experts: list[Expert], eta: float, share: float, min_prob: float, instantaneous_eta: float = 0.0):
        if not experts: raise ValueError("Need at least one expert")
        self.experts = experts; self.eta = eta; self.share = share; self.min_prob = min_prob; self.instantaneous_eta = instantaneous_eta
        self.weights = np.full((len(experts),), 1.0 / len(experts), dtype=np.float64); self.prior = self.weights.copy(); self.expert_logloss = np.zeros((len(experts),), dtype=np.float64)
        self.fixed_weights = eta == 0.0 and share == 0.0; self.fixed_active = np.arange(len(experts), dtype=np.int64); self.active = self.fixed_active.copy()
        self.supports_online_sequence = all(expert.supports_online_sequence for expert in experts); self.graph_experts = iter_expert_graph(self.experts)
        for idx, expert in enumerate(self.graph_experts): expert.cache_slot = idx
        self.cache_size = len(self.graph_experts); self.set_weights(self.weights)
    def set_weights(self, weights: np.ndarray) -> None:
        normalized = np.array(weights, dtype=np.float64, copy=True); total = float(normalized.sum())
        if total <= 0.0 or not np.isfinite(total): raise ValueError("weights must sum to a positive finite value")
        normalized /= total; self.weights = normalized; self.prior = normalized.copy(); self.active = np.flatnonzero(normalized > 0.0)
        if self.active.size == 0: raise ValueError("mixer needs at least one positive-weight expert")
        if self.fixed_weights: self.fixed_active = self.active.copy()
    def step(self, tokens: np.ndarray, pos: int, token: int) -> float:
        prob_cache = [math.nan] * self.cache_size
        if self.fixed_weights:
            if self.fixed_active.size == 1:
                idx = int(self.fixed_active[0]); mix_prob = max(self.min_prob, self.experts[idx].cached_prob(tokens, pos, token, prob_cache)); self.expert_logloss[idx] += -math.log2(mix_prob)
            else:
                active = self.fixed_active
                if self.instantaneous_eta > 0.0:
                    numerator = denominator = 0.0
                    for idx in active:
                        prob = max(self.min_prob, self.experts[int(idx)].cached_prob(tokens, pos, token, prob_cache)); self.expert_logloss[idx] += -math.log2(prob)
                        base = float(self.weights[idx]) * (prob ** self.instantaneous_eta); denominator += base; numerator += base * prob
                    mix_prob = max(numerator / denominator if denominator > 0.0 else self.min_prob, self.min_prob)
                else:
                    mix_prob = 0.0
                    for idx in active:
                        prob = max(self.min_prob, self.experts[int(idx)].cached_prob(tokens, pos, token, prob_cache)); mix_prob += float(self.weights[idx] * prob); self.expert_logloss[idx] += -math.log2(prob)
                    mix_prob = max(mix_prob, self.min_prob)
        else:
            active = self.active; probs = np.empty((active.size,), dtype=np.float64)
            for active_pos, idx in enumerate(active): probs[active_pos] = max(self.min_prob, self.experts[int(idx)].cached_prob(tokens, pos, token, prob_cache))
            mix_weights = self.weights[active]
            if self.instantaneous_eta > 0.0:
                dynamic_weights = mix_weights * np.power(probs, self.instantaneous_eta); total = float(dynamic_weights.sum())
                if np.isfinite(total) and total > 0.0: mix_weights = dynamic_weights / total
            mix_prob = max(float(np.dot(mix_weights, probs)), self.min_prob); self.expert_logloss[active] += -np.log2(probs)
            posterior = self.weights[active] * np.power(probs, self.eta); total = float(posterior.sum())
            if not np.isfinite(total) or total <= 0.0:
                self.weights = self.prior.copy()
            else:
                posterior /= total; self.weights = self.prior.copy(); self.weights[active] = (1.0 - self.share) * posterior + self.share * self.prior[active]
        for expert in self.experts:
            if not expert.frozen_updates: expert.update(tokens, pos, token)
        return mix_prob
    def observe(self, tokens: np.ndarray, pos: int, token: int) -> None:
        for expert in self.experts:
            if not expert.frozen_updates: expert.update(tokens, pos, token)
    def reset_weights(self) -> None:
        self.weights = self.prior.copy(); self.expert_logloss.fill(0.0)
    def reset_ephemeral_state(self) -> None:
        for expert in self.experts: expert.reset_state()
    def begin_sequence(self, position_offset: int = 0) -> None:
        for expert in self.experts: expert.begin_sequence(position_offset=position_offset)
    def prime(self, token: int) -> None:
        for expert in self.experts: expert.prime(token)
    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        del position_offset
        for expert in self.experts: expert.set_sequence(tokens)
    def set_sequence_with_offset(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        for expert in self.experts: expert.set_sequence(tokens, position_offset=position_offset)
    def state_dict(self) -> dict:
        return {"expert_names": [expert.name for expert in self.experts], "weights": self.weights.copy(), "prior": self.prior.copy(), "expert_logloss": self.expert_logloss.copy(), "experts": [expert.state_dict() for expert in self.experts]}
    def load_state_dict(self, state: dict) -> None:
        expected_names = [expert.name for expert in self.experts]; saved_names = list(state["expert_names"])
        if saved_names != expected_names: raise ValueError(f"State expert mismatch: saved={saved_names} current={expected_names}")
        self.weights = np.array(state["weights"], dtype=np.float64, copy=True); self.prior = np.array(state["prior"], dtype=np.float64, copy=True); self.active = np.flatnonzero(self.prior > 0.0)
        if self.fixed_weights: self.fixed_active = self.active.copy()
        self.expert_logloss = np.array(state["expert_logloss"], dtype=np.float64, copy=True)
        for expert, expert_state in zip(self.experts, state["experts"], strict=True): expert.load_state_dict(expert_state)
class CompositeLMExpert(Expert):
    def __init__(self, prob_expert: Expert, components: list[Expert]):
        self.name = prob_expert.name
        self.prob_expert = prob_expert
        self.components = components
        self.context_len = getattr(prob_expert, "context_len", 0)
        self.supports_online_sequence = all(expert.supports_online_sequence for expert in components)
        self.is_lm_expert = True
    def set_sequence(self, tokens: np.ndarray, position_offset: int = 0) -> None:
        for expert in self.components:
            expert.set_sequence(tokens, position_offset=position_offset)
    def begin_sequence(self, position_offset: int = 0) -> None:
        for expert in self.components:
            expert.begin_sequence(position_offset=position_offset)
    def prime(self, token: int) -> None:
        for expert in self.components:
            expert.prime(token)
    def prob(
        self,
        tokens: np.ndarray,
        pos: int,
        token: int,
        cache: list[float] | None = None,
    ) -> float:
        return self.prob_expert.cached_prob(tokens, pos, token, cache)
    def update(self, tokens: np.ndarray, pos: int, token: int) -> None:
        for expert in self.components:
            if not expert.frozen_updates:
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
    unigram.is_lm_expert = True
    lm_components: list[Expert] = [unigram]
    bigram = BigramExpert(vocab_size=vocab_size, backoff=unigram, alpha=args.bigram_alpha, discount=args.absolute_discount, use_continuation_unigram=bool(args.continuation_unigram))
    bigram.is_lm_expert = True; lm_components.append(bigram); lm_backoff = bigram
    for context_len in args.ngram_contexts:
        ngram = HashedNgramExpert(context_len=context_len, vocab_size=vocab_size, backoff=lm_backoff, alpha=args.ngram_alpha, discount=args.absolute_discount, count_scale_limit=args.ngram_count_scale_limit, prune_min_count=args.ngram_prune_min_count)
        ngram.is_lm_expert = True; lm_components.append(ngram); lm_backoff = ngram
    experts: list[Expert] = [CompositeLMExpert(prob_expert=lm_backoff, components=lm_components)]
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
    for offsets in args.doc_skip_copy_contexts:
        experts.append(
            SkipRecentMatchExpert(
                offsets=offsets,
                max_gap=args.copy_window,
                max_matches=args.copy_max_matches,
                max_stored_matches=args.copy_store_limit,
                backoff=lm_backoff,
                alpha=args.copy_alpha,
                decay_power=args.copy_decay_power,
                reset_token=args.doc_reset_token,
                name_prefix="doc_skip_copy",
            )
        )
    return experts
def parse_csv_ints(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(piece) for piece in raw.split(",") if piece.strip()]
def parse_skip_contexts(raw: str) -> list[tuple[int, ...]]:
    if not raw.strip():
        return []
    contexts: list[tuple[int, ...]] = []
    for piece in raw.split(","):
        offsets = tuple(int(bit) for bit in piece.strip().split("-") if bit.strip())
        if offsets: contexts.append(offsets)
    return contexts
def parse_csv_floats(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(piece) for piece in raw.split(",") if piece.strip()]
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a non-neural token compressor on challenge val data")
    parser.add_argument(
        "--data-pattern",
        default=str(REPO_ROOT / "data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"),
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
    parser.add_argument("--state-compression", choices=("zlib", "lzma"), default="zlib")
    parser.add_argument(
        "--skip-validation",
        type=int,
        default=0,
        help="If set, stop after optional warmup/state export without loading validation shards.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=str(REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model"),
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
    parser.add_argument("--instantaneous-eta", type=float, default=0.0)
    parser.add_argument("--min-prob", type=float, default=1e-12)
    parser.add_argument("--unigram-alpha", type=float, default=0.5)
    parser.add_argument("--bigram-alpha", type=float, default=4.0)
    parser.add_argument("--ngram-alpha", type=float, default=1.0)
    parser.add_argument("--ngram-count-scale-limit", type=int, default=0)
    parser.add_argument("--ngram-prune-min-count", type=int, default=0)
    parser.add_argument("--absolute-discount", type=float, default=0.0)
    parser.add_argument("--continuation-unigram", type=int, default=1)
    parser.add_argument("--copy-alpha", type=float, default=1.0)
    parser.add_argument("--copy-decay-power", type=float, default=0.6)
    parser.add_argument("--ngram-contexts", type=parse_csv_ints, default=parse_csv_ints("2,3,4"))
    parser.add_argument("--doc-copy-contexts", type=parse_csv_ints, default=parse_csv_ints(""))
    parser.add_argument("--doc-skip-copy-contexts", type=parse_skip_contexts, default=parse_skip_contexts(""))
    parser.add_argument("--doc-reset-token", type=int, default=1)
    parser.add_argument("--copy-window", type=int, default=200_000)
    parser.add_argument("--copy-max-matches", type=int, default=32)
    parser.add_argument("--copy-store-limit", type=int, default=32)
    parser.add_argument("--expert-weights", type=parse_csv_floats, default=parse_csv_floats(""))
    parser.add_argument("--eval-chunk-tokens", type=int, default=0)
    return parser
def max_required_context(experts: list[Expert]) -> int:
    max_context = 1
    for expert in experts:
        context_len = getattr(expert, "context_len", 0)
        if context_len > max_context:
            max_context = context_len
    return max_context
def iter_expert_graph(experts: list[Expert]) -> list[Expert]:
    seen: set[int] = set()
    ordered: list[Expert] = []
    stack = list(reversed(experts))
    while stack:
        expert = stack.pop()
        expert_id = id(expert)
        if expert_id in seen:
            continue
        seen.add(expert_id)
        ordered.append(expert)
        backoff = getattr(expert, "backoff", None)
        if isinstance(backoff, Expert):
            stack.append(backoff)
        base_unigram = getattr(expert, "base_unigram", None)
        if isinstance(base_unigram, Expert):
            stack.append(base_unigram)
        prob_expert = getattr(expert, "prob_expert", None)
        if isinstance(prob_expert, Expert):
            stack.append(prob_expert)
        components = getattr(expert, "components", None)
        if components:
            stack.extend(reversed([child for child in components if isinstance(child, Expert)]))
    return ordered
STATE_VERSION = 1
def save_mixer_state(path: Path, args: argparse.Namespace, mixer: AdaptiveMixer, warmup_seen: int) -> int:
    payload = {
        "version": STATE_VERSION,
        "args": vars(args).copy(),
        "warmup_seen": int(warmup_seen),
        "mixer": mixer.state_dict(),
    }
    raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = compress_state_payload(raw, args.state_compression)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return len(compressed)
def load_mixer_state(path: Path, mixer: AdaptiveMixer) -> tuple[int, dict]:
    compressed = path.read_bytes()
    payload = pickle.loads(decompress_state_payload(compressed))
    if int(payload.get("version", -1)) != STATE_VERSION:
        raise ValueError(f"Unsupported state version in {path}")
    mixer.load_state_dict(payload["mixer"])
    return len(compressed), payload
def warmup_mixer(args: argparse.Namespace, mixer: AdaptiveMixer, position_offset: int = 0) -> tuple[int, float]:
    if not args.train_pattern or args.warmup_tokens <= 0:
        return 0, 0.0
    files = [Path(p) for p in sorted(glob.glob(args.train_pattern))]
    if not files:
        raise FileNotFoundError(f"No training files found for pattern: {args.train_pattern}")
    remaining = int(args.warmup_tokens)
    total_seen = 0
    start = time.perf_counter()
    max_context = max_required_context(mixer.experts)
    tail = np.empty((0,), dtype=np.uint16)
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
        sequence_offset = total_seen + position_offset + 1 - start_pos
        mixer.set_sequence_with_offset(tokens, position_offset=sequence_offset)
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
    args.data_pattern = resolve_repo_path(args.data_pattern)
    args.train_pattern = resolve_repo_path(args.train_pattern)
    args.tokenizer_path = resolve_repo_path(args.tokenizer_path)
    start = time.perf_counter()
    experts = build_experts(args, args.vocab_size)
    mixer = AdaptiveMixer(experts=experts, eta=args.eta, share=args.share, min_prob=args.min_prob, instantaneous_eta=args.instantaneous_eta)
    loaded_state_bytes = 0
    loaded_warmup_seen = 0
    if args.load_state:
        loaded_state_bytes, payload = load_mixer_state(Path(args.load_state), mixer)
        loaded_warmup_seen = int(payload.get("warmup_seen", 0))
    warmup_seen, warmup_elapsed = warmup_mixer(args, mixer, position_offset=loaded_warmup_seen)
    total_warmup_seen = loaded_warmup_seen + warmup_seen
    saved_state_bytes = 0
    if args.save_state:
        saved_state_bytes = save_mixer_state(Path(args.save_state), args, mixer, total_warmup_seen)
    if args.expert_weights:
        weights = np.array(args.expert_weights, dtype=np.float64)
        if weights.size != len(mixer.experts):
            raise ValueError(
                f"--expert-weights length mismatch: got {weights.size}, expected {len(mixer.experts)}"
            )
        mixer.set_weights(weights)
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
    chunk_size = args.eval_chunk_tokens if args.eval_chunk_tokens > 0 else tokens.size
    if chunk_size >= tokens.size:
        mixer.set_sequence_with_offset(tokens, position_offset=total_warmup_seen)
        for pos in range(1, tokens.size):
            token = int(tokens[pos])
            mix_prob = mixer.step(tokens, pos, token)
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
    else:
        overlap = max_required_context(mixer.experts)
        chunk_start = 0
        while chunk_start < tokens.size - 1:
            effective_overlap = 0 if chunk_start == 0 else overlap
            sequence_start = chunk_start - effective_overlap
            chunk_end = min(tokens.size, chunk_start + chunk_size)
            chunk_tokens = tokens[sequence_start:chunk_end]
            mixer.set_sequence_with_offset(chunk_tokens, position_offset=total_warmup_seen + sequence_start)
            local_start = 1 if sequence_start == 0 else effective_overlap
            for pos in range(local_start, chunk_tokens.size):
                token = int(chunk_tokens[pos])
                mix_prob = mixer.step(chunk_tokens, pos, token)
                global_pos = sequence_start + pos
                total_bits += -math.log2(mix_prob)
                total_bytes += float(target_bytes[global_pos - 1])
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
            chunk_start += chunk_size
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
