#!/usr/bin/env python3
"""Exact Path A token-normalized PPM-D evaluator scaffold.

This script implements the mathematically valid Path A scoring rule discussed
in ``plans/ppmd-legality-proof.md``:

    q_t(v) = P_PPMD(bytes_t(v) | byte_history_before_t)
    p_ppm_t(v) = q_t(v) / sum_u q_t(u)
    p_mix_t(v) = lambda_t * p_nn_t(v) + (1 - lambda_t) * p_ppm_t(v)

The core is intentionally import-light so unit tests can run on CPU-only login
nodes.  The optional fresh-eval hooks import the exp_1876 model code only when
requested on a GPU pod.

Performance note: the exact Path A PPM pass is sequential over validation
positions because the PPM state updates after every scored token.  This module
uses a token-byte trie to share candidate-prefix work and supports prefix
smokes.  Full validation in Python is expected to be slow; use it as a
correctness reference or as a scaffold for a compiled persistent kernel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP1876_SOURCE = REPO_ROOT / "results" / "exp_1876_ppmd" / "train_gpt_merged.py"
EXP1876_PROD = REPO_ROOT / "results" / "exp_1876_ppmd" / "prod_8gpu_s42v2"
EXP1876_MODEL = EXP1876_PROD / "final_model.int6.ptz"
DEFAULT_OUTPUT = EXP1876_PROD / "path_a_ppmd_eval.json"
KNOWN_TARGET_TOKENS = 40_540_160
KNOWN_TARGET_BYTES = 151_078_222


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CandidateBytes:
    """Two candidate byte strings for one token.

    ``after_boundary`` is used when the previous token is a boundary token and
    therefore a SentencePiece leading-space marker does not emit a literal
    space byte.  ``after_non_boundary`` is used otherwise.
    """

    token_id: int
    after_boundary: bytes
    after_non_boundary: bytes
    emittable: bool = True


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"] = field(default_factory=dict)
    terminals: List[int] = field(default_factory=list)


def candidate_bytes_for_token(
    token_id: int,
    token_bytes_lut: Sequence[bytes],
    has_leading_space_lut: Sequence[bool],
    *,
    emittable: bool = True,
) -> CandidateBytes:
    """Build candidate bytes for both previous-boundary cases.

    This mirrors the exp_1876 denominator semantics: the SentencePiece ``▁``
    marker is stripped from ``token_bytes_lut`` and reintroduced as a literal
    space only when the previous token is not a boundary.
    """

    base = token_bytes_lut[token_id] if 0 <= token_id < len(token_bytes_lut) else b""
    has_space = bool(has_leading_space_lut[token_id]) if 0 <= token_id < len(has_leading_space_lut) else False
    after_boundary = bytes(base)
    after_non_boundary = (b" " + bytes(base)) if has_space else bytes(base)
    return CandidateBytes(
        token_id=token_id,
        after_boundary=after_boundary,
        after_non_boundary=after_non_boundary,
        emittable=emittable,
    )


def build_candidate_tries(candidates: Sequence[CandidateBytes]) -> Tuple[TrieNode, TrieNode]:
    """Build separate tries for previous-boundary and non-boundary contexts."""

    boundary_root = TrieNode()
    non_boundary_root = TrieNode()
    for candidate in candidates:
        if not candidate.emittable:
            continue
        _insert_candidate(boundary_root, candidate.after_boundary, candidate.token_id)
        _insert_candidate(non_boundary_root, candidate.after_non_boundary, candidate.token_id)
    return boundary_root, non_boundary_root


def _insert_candidate(root: TrieNode, seq: bytes, token_id: int) -> None:
    node = root
    for b in seq:
        node = node.children.setdefault(int(b), TrieNode())
    node.terminals.append(int(token_id))


def trie_stats(root: TrieNode) -> Dict[str, int]:
    nodes = 0
    edges = 0
    terminals = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        terminals += len(node.terminals)
        edges += len(node.children)
        stack.extend(node.children.values())
    return {"nodes": nodes, "edges": edges, "terminals": terminals}


class PPMDState:
    """Proper PPM-D state with update exclusion and normalized byte probs."""

    def __init__(self, order: int = 5):
        if order < 0:
            raise ValueError("PPM order must be non-negative")
        self.order = int(order)
        self.ctx_counts: Dict[bytes, Dict[int, int]] = {}
        self.window = bytearray()

    def clone_virtual(self) -> "VirtualPPMDState":
        return VirtualPPMDState(self.ctx_counts, bytearray(self.window), self.order)

    def update_byte(self, b: int) -> None:
        b = int(b)
        for k in range(0, min(self.order, len(self.window)) + 1):
            ctx = bytes(self.window[-k:]) if k > 0 else b""
            counts = self.ctx_counts.setdefault(ctx, {})
            counts[b] = counts.get(b, 0) + 1
        self.window.append(b)
        if len(self.window) > self.order:
            del self.window[0]

    def update_bytes(self, data: bytes) -> None:
        for b in data:
            self.update_byte(b)

    def byte_probs(self) -> List[float]:
        return _ppmd_byte_probs_with_provider(lambda ctx: self.ctx_counts.get(ctx), self.window, self.order)

    def byte_prob(self, b: int) -> float:
        return _ppmd_byte_prob_with_provider(lambda ctx: self.ctx_counts.get(ctx), self.window, self.order, int(b))

    def confidence(self) -> float:
        """Prefix-only confidence gate compatible with exp_1876 intent.

        This depends only on the current PPM state, not on the target byte or
        token.  It uses the longest available context and computes max_count /
        (total + unique), matching the production confidence shape.
        """

        for k in range(min(self.order, len(self.window)), -1, -1):
            ctx = bytes(self.window[-k:]) if k > 0 else b""
            counts = self.ctx_counts.get(ctx)
            if counts:
                total = sum(counts.values())
                unique = len(counts)
                return max(counts.values()) / float(total + unique)
        return 0.0

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(bytes(self.window))
        for ctx in sorted(self.ctx_counts.keys()):
            digest.update(len(ctx).to_bytes(2, "little"))
            digest.update(ctx)
            counts = self.ctx_counts[ctx]
            for b in sorted(counts.keys()):
                digest.update(int(b).to_bytes(1, "little"))
                digest.update(int(counts[b]).to_bytes(8, "little", signed=False))
        return digest.hexdigest()


class VirtualPPMDState:
    """Candidate-local PPM-D overlay used for token byte-string scoring."""

    def __init__(
        self,
        base_counts: Mapping[bytes, Mapping[int, int]],
        window: bytearray,
        order: int,
        overlay_counts: Optional[Dict[bytes, Dict[int, int]]] = None,
    ):
        self.base_counts = base_counts
        self.window = window
        self.order = int(order)
        self.overlay_counts = overlay_counts if overlay_counts is not None else {}

    def fork_and_update(self, b: int) -> "VirtualPPMDState":
        overlay = {ctx: dict(counts) for ctx, counts in self.overlay_counts.items()}
        window = bytearray(self.window)
        b = int(b)
        for k in range(0, min(self.order, len(window)) + 1):
            ctx = bytes(window[-k:]) if k > 0 else b""
            counts = overlay.setdefault(ctx, {})
            counts[b] = counts.get(b, 0) + 1
        window.append(b)
        if len(window) > self.order:
            del window[0]
        return VirtualPPMDState(self.base_counts, window, self.order, overlay)

    def combined_counts(self, ctx: bytes) -> Optional[Dict[int, int]]:
        base = self.base_counts.get(ctx)
        overlay = self.overlay_counts.get(ctx)
        if not base and not overlay:
            return None
        merged: Dict[int, int] = {}
        if base:
            for b, c in base.items():
                merged[int(b)] = int(c)
        if overlay:
            for b, c in overlay.items():
                merged[int(b)] = merged.get(int(b), 0) + int(c)
        return merged

    def byte_probs(self) -> List[float]:
        return _ppmd_byte_probs_with_provider(self.combined_counts, self.window, self.order)

    def byte_prob(self, b: int) -> float:
        return _ppmd_byte_prob_with_provider(self.combined_counts, self.window, self.order, int(b))


def _ppmd_byte_probs_with_provider(counts_for_ctx: Any, window: bytearray, order: int) -> List[float]:
    """Compute normalized PPM-D byte probabilities with update exclusion."""

    probs = [0.0] * 256
    assigned = set()
    escape_mass = 1.0
    for k in range(min(order, len(window)), -1, -1):
        ctx = bytes(window[-k:]) if k > 0 else b""
        counts = counts_for_ctx(ctx)
        if not counts:
            continue
        active_counts = {int(b): int(c) for b, c in counts.items() if int(b) not in assigned and int(c) > 0}
        if not active_counts:
            continue
        active_alphabet_size = 256 - len(assigned)
        if len(active_counts) == active_alphabet_size:
            active_total = sum(active_counts.values())
            if active_total <= 0:
                continue
            for b, c in active_counts.items():
                probs[b] = escape_mass * (c / active_total)
                assigned.add(b)
            escape_mass = 0.0
            break
        active_unique = len(active_counts)
        active_total = sum(active_counts.values())
        denom = active_total + active_unique
        for b, c in active_counts.items():
            probs[b] = escape_mass * (c / denom)
            assigned.add(b)
        escape_mass *= active_unique / denom

    remaining_count = 256 - len(assigned)
    if remaining_count:
        per_byte = escape_mass / remaining_count
        for b in range(256):
            if b not in assigned:
                probs[b] = per_byte
    return probs


def _ppmd_byte_prob_with_provider(counts_for_ctx: Any, window: bytearray, order: int, target_b: int) -> float:
    """Compute one exact PPM-D byte probability with update exclusion.

    This is equivalent to ``_ppmd_byte_probs_with_provider(...)[target_b]`` but
    avoids building a 256-entry distribution at every trie edge.  The escape
    mass still depends on the full set of bytes assigned at higher contexts, so
    the implementation tracks the exclusion set while returning as soon as the
    requested byte is assigned.
    """

    target_b = int(target_b)
    assigned = set()
    escape_mass = 1.0
    for k in range(min(order, len(window)), -1, -1):
        ctx = bytes(window[-k:]) if k > 0 else b""
        counts = counts_for_ctx(ctx)
        if not counts:
            continue
        active_counts = {int(b): int(c) for b, c in counts.items() if int(b) not in assigned and int(c) > 0}
        if not active_counts:
            continue
        active_alphabet_size = 256 - len(assigned)
        active_total = sum(active_counts.values())
        if active_total <= 0:
            continue
        if len(active_counts) == active_alphabet_size:
            if target_b in active_counts:
                return escape_mass * (active_counts[target_b] / active_total)
            return 0.0
        active_unique = len(active_counts)
        denom = active_total + active_unique
        if target_b in active_counts:
            return escape_mass * (active_counts[target_b] / denom)
        assigned.update(active_counts.keys())
        escape_mass *= active_unique / denom

    if target_b in assigned:
        return 0.0
    return escape_mass / (256 - len(assigned))


def sequence_probability(state: PPMDState, seq: bytes) -> float:
    """Exact PPM-D probability of a candidate byte string with virtual updates."""

    virtual = state.clone_virtual()
    prob = 1.0
    for b in seq:
        prob *= virtual.byte_prob(int(b))
        virtual = virtual.fork_and_update(int(b))
    return prob


def trie_partial_z_and_target(
    root: TrieNode,
    state: PPMDState,
    target_id: int,
    *,
    shard_start: int = 0,
    shard_end: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Compute partial Z and target q for a token shard via trie DFS."""

    if shard_end is None:
        shard_end = sys.maxsize
    z = 0.0
    target_q = 0.0
    terminal_count = 0
    stack: List[Tuple[TrieNode, VirtualPPMDState, float]] = [(root, state.clone_virtual(), 1.0)]
    while stack:
        node, virtual, prefix_prob = stack.pop()
        if node.terminals:
            for token_id in node.terminals:
                if shard_start <= token_id < shard_end:
                    z += prefix_prob
                    terminal_count += 1
                    if token_id == target_id:
                        target_q += prefix_prob
        if node.children:
            for b, child in node.children.items():
                p = virtual.byte_prob(int(b))
                if p <= 0.0:
                    continue
                stack.append((child, virtual.fork_and_update(int(b)), prefix_prob * p))
    return z, target_q, terminal_count


def combine_path_a_partials(partials: Iterable[Tuple[float, float, int]]) -> Tuple[float, float, int]:
    z = 0.0
    target_q = 0.0
    count = 0
    for part_z, part_target_q, part_count in partials:
        z += float(part_z)
        target_q += float(part_target_q)
        count += int(part_count)
    return z, target_q, count


def path_a_score_position(
    state: PPMDState,
    boundary_root: TrieNode,
    non_boundary_root: TrieNode,
    target_id: int,
    prev_is_boundary: bool,
    neural_nll_nats: float,
    actual_target_bytes: bytes,
    *,
    lambda_hi: float = 0.9,
    lambda_lo: float = 0.05,
    conf_threshold: float = 0.9,
    shard_start: int = 0,
    shard_end: Optional[int] = None,
    distributed_partials: Optional[Iterable[Tuple[float, float, int]]] = None,
) -> Dict[str, Any]:
    """Score one token with exact token-normalized Path A, then update state.

    If ``distributed_partials`` is supplied, it is combined instead of computing
    a local trie shard.  This is useful for tests and for external distributed
    orchestration.
    """

    before_digest = state.state_digest()
    trie = boundary_root if prev_is_boundary else non_boundary_root
    if distributed_partials is None:
        if shard_start != 0 or shard_end is not None:
            raise ValueError(
                "Refusing to normalize over a single vocab shard. Pass complete distributed_partials "
                "or leave shard_start/shard_end unset for full-vocab scoring."
            )
        partials = [trie_partial_z_and_target(trie, state, target_id, shard_start=shard_start, shard_end=shard_end)]
    else:
        partials = list(distributed_partials)
    z, target_q, terminal_count = combine_path_a_partials(partials)
    if z <= 0.0:
        raise ValueError("Path A normalization constant Z is non-positive")
    p_ppm_target = target_q / z
    if p_ppm_target < 0.0:
        raise ValueError("Negative target PPM probability")
    p_nn_target = math.exp(-float(neural_nll_nats))
    confidence = state.confidence()
    lam = lambda_lo if confidence >= conf_threshold else lambda_hi
    p_mix = lam * p_nn_target + (1.0 - lam) * p_ppm_target
    if p_mix <= 0.0:
        raise ValueError("Mixture assigned zero probability to target")
    score_digest = state.state_digest()
    if score_digest != before_digest:
        raise AssertionError("PPM state changed during scoring")
    loss_bits = -math.log(p_mix, 2.0)
    state.update_bytes(actual_target_bytes)
    after_digest = state.state_digest()
    return {
        "loss_bits": loss_bits,
        "p_mix_target": p_mix,
        "p_nn_target": p_nn_target,
        "p_ppm_target": p_ppm_target,
        "q_target": target_q,
        "z": z,
        "lambda": lam,
        "confidence": confidence,
        "terminal_count": terminal_count,
        "state_digest_before": before_digest,
        "state_digest_after_score": score_digest,
        "state_digest_after_update": after_digest,
        "state_changed_only_after_update": before_digest == score_digest and before_digest != after_digest,
    }


def actual_bytes_for_position(candidates: Sequence[CandidateBytes], target_id: int, prev_is_boundary: bool) -> bytes:
    candidate = candidates[int(target_id)]
    return candidate.after_boundary if prev_is_boundary else candidate.after_non_boundary


# --- Phase 5: C++ backend dispatcher (Path A) ---------------------------------
#
# The default backend remains "python" so existing tests under /bin/python3.8
# continue to pass without the C++ extension. The "cpp" backend dispatches to
# `_ppmd_cpp.score_path_a_arrays(...)`, which lives in `.venv-smoke` (Python
# 3.12) at scripts/ppmd_cpp/_ppmd_cpp*.so. The adapter below packs the
# CandidateBytes vocab into the flat (uint8 bytes + int32 offsets) format the
# C++ scorer expects.

_PPMD_CPP_BUILD_DIR = REPO_ROOT / "scripts" / "ppmd_cpp"


def _pack_vocab_for_cpp(
    candidates: Sequence[CandidateBytes],
):
    """Pack a list of CandidateBytes into the flat arrays the C++ scorer wants.

    Returns (boundary_flat, boundary_offsets, nonboundary_flat,
    nonboundary_offsets, emittable_u8, vocab_size).

    Offsets are int32 of length V+1; bytes are uint8 of length sum(len(...)).
    The is_boundary array is NOT vocab-derived; callers pass it through
    separately because it is the per-token boundary LUT, not vocab geometry.
    """

    import numpy as _np  # local import; keep top-of-file import-light

    vocab_size = len(candidates)
    bnd_flat = bytearray()
    nbnd_flat = bytearray()
    bnd_off = [0]
    nbnd_off = [0]
    emit = bytearray(vocab_size)
    for tid, cand in enumerate(candidates):
        b = bytes(cand.after_boundary)
        nb = bytes(cand.after_non_boundary)
        bnd_flat.extend(b)
        nbnd_flat.extend(nb)
        bnd_off.append(len(bnd_flat))
        nbnd_off.append(len(nbnd_flat))
        emit[tid] = 1 if cand.emittable else 0
    return (
        _np.frombuffer(bytes(bnd_flat), dtype=_np.uint8).copy(),
        _np.asarray(bnd_off, dtype=_np.int32),
        _np.frombuffer(bytes(nbnd_flat), dtype=_np.uint8).copy(),
        _np.asarray(nbnd_off, dtype=_np.int32),
        _np.frombuffer(bytes(emit), dtype=_np.uint8).copy(),
        vocab_size,
    )


def _import_ppmd_cpp(*, abort_on_failure: bool):
    """Lazy-import _ppmd_cpp from the build dir.

    On ImportError, return None unless abort_on_failure is True (in which case
    raise). Callers are responsible for printing a stderr fallback warning when
    abort_on_failure is False.
    """

    if str(_PPMD_CPP_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_PPMD_CPP_BUILD_DIR))
    try:
        import _ppmd_cpp  # type: ignore[import-not-found]
        return _ppmd_cpp
    except ImportError as exc:
        if abort_on_failure:
            raise
        print(
            f"WARN: --backend cpp requested but _ppmd_cpp import failed ({exc!r}); "
            "falling back to python backend.",
            file=sys.stderr,
        )
        return None


def _score_path_a_arrays_cpp(
    *,
    target_ids,
    prev_ids,
    nll_nats,
    candidates: Sequence[CandidateBytes],
    is_boundary_token_lut: Sequence[bool],
    order: int,
    lambda_hi: float,
    lambda_lo: float,
    conf_threshold: float,
    abort_on_import_failure: bool,
):
    """Run the C++ Path A scorer end-to-end on the same inputs as Python.

    Returns the same dict shape as `score_path_a_arrays(..., python)` for the
    keys callers actually use: total_bits, total_bytes, bpb, positions.
    Other diagnostic keys (samples, candidate_trie_stats) are absent in the
    C++ path.
    """

    import numpy as _np

    _ppmd_cpp = _import_ppmd_cpp(abort_on_failure=abort_on_import_failure)
    if _ppmd_cpp is None:
        return None  # caller will fall back to python

    target_ids_np = _np.ascontiguousarray(target_ids, dtype=_np.int32)
    prev_ids_np = _np.ascontiguousarray(prev_ids, dtype=_np.int32)
    nll_nats_np = _np.ascontiguousarray(nll_nats, dtype=_np.float64)

    (
        bnd_flat,
        bnd_off,
        nbnd_flat,
        nbnd_off,
        emit_arr,
        vocab_size,
    ) = _pack_vocab_for_cpp(candidates)

    isb_arr = _np.zeros(vocab_size, dtype=_np.uint8)
    for tid in range(vocab_size):
        if 0 <= tid < len(is_boundary_token_lut) and bool(is_boundary_token_lut[tid]):
            isb_arr[tid] = 1

    out = _ppmd_cpp.score_path_a_arrays(
        target_ids_np,
        prev_ids_np,
        nll_nats_np,
        bnd_flat,
        bnd_off,
        nbnd_flat,
        nbnd_off,
        emit_arr,
        isb_arr,
        {
            "order": int(order),
            "lambda_hi": float(lambda_hi),
            "lambda_lo": float(lambda_lo),
            "conf_threshold": float(conf_threshold),
            "update_after_score": True,
        },
    )
    return {
        "mode": "path-a-array-score-cpp",
        "positions": int(out["positions"]),
        "total_bits": float(out["total_bits"]),
        "total_bytes": int(out["total_bytes"]),
        "bpb": float(out["bpb"]) if out.get("bpb") is not None else None,
        "backend": "cpp",
    }


def _import_ppmd_cuda(*, abort_on_failure: bool):
    """Lazy-import _ppmd_cuda from the build dir.

    On ImportError, return None unless abort_on_failure is True.
    """
    if str(_PPMD_CPP_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_PPMD_CPP_BUILD_DIR))
    try:
        import _ppmd_cuda  # type: ignore[import-not-found]
        return _ppmd_cuda
    except ImportError as exc:
        if abort_on_failure:
            raise
        print(
            f"WARN: --backend cuda requested but _ppmd_cuda import failed ({exc!r}); "
            "falling back to python backend.",
            file=sys.stderr,
        )
        return None


def _score_path_a_arrays_cuda(
    *,
    target_ids,
    prev_ids,
    nll_nats,
    candidates: Sequence[CandidateBytes],
    is_boundary_token_lut: Sequence[bool],
    order: int,
    lambda_hi: float,
    lambda_lo: float,
    conf_threshold: float,
    abort_on_import_failure: bool,
):
    """Run the CUDA Path A scorer end-to-end.

    Returns the same dict shape as `_score_path_a_arrays_cpp`.
    """
    import numpy as _np

    _ppmd_cuda = _import_ppmd_cuda(abort_on_failure=abort_on_import_failure)
    if _ppmd_cuda is None:
        return None

    if not _ppmd_cuda.cuda.available():
        if abort_on_import_failure:
            raise RuntimeError("CUDA backend requested but no CUDA device found")
        print(
            "WARN: --backend cuda requested but no CUDA device found; "
            "falling back to python backend.",
            file=sys.stderr,
        )
        return None

    target_ids_np = _np.ascontiguousarray(target_ids, dtype=_np.int32)
    prev_ids_np = _np.ascontiguousarray(prev_ids, dtype=_np.int32)
    nll_nats_np = _np.ascontiguousarray(nll_nats, dtype=_np.float64)

    (
        bnd_flat,
        bnd_off,
        nbnd_flat,
        nbnd_off,
        emit_arr,
        vocab_size,
    ) = _pack_vocab_for_cpp(candidates)

    isb_arr = _np.zeros(vocab_size, dtype=_np.uint8)
    for tid in range(vocab_size):
        if 0 <= tid < len(is_boundary_token_lut) and bool(is_boundary_token_lut[tid]):
            isb_arr[tid] = 1

    out = _ppmd_cuda.cuda.score_path_a_arrays_cuda(
        target_ids_np,
        prev_ids_np,
        nll_nats_np,
        bnd_flat,
        bnd_off,
        nbnd_flat,
        nbnd_off,
        emit_arr,
        isb_arr,
        {
            "order": int(order),
            "lambda_hi": float(lambda_hi),
            "lambda_lo": float(lambda_lo),
            "conf_threshold": float(conf_threshold),
            "update_after_score": True,
        },
    )
    return {
        "mode": "path-a-array-score-cuda",
        "positions": int(out["positions"]),
        "total_bits": float(out["total_bits"]),
        "total_bytes": int(out["total_bytes"]),
        "bpb": float(out["bpb"]) if out.get("bpb") is not None else None,
        "backend": "cuda",
    }


def _score_path_a_arrays_dispatch(
    backend: str,
    *,
    target_ids,
    prev_ids,
    nll_nats,
    candidates: Sequence[CandidateBytes],
    is_boundary_token_lut: Sequence[bool],
    order: int = 5,
    lambda_hi: float = 0.9,
    lambda_lo: float = 0.05,
    conf_threshold: float = 0.9,
    max_positions: Optional[int] = None,
    normalization_sample_every: int = 0,
    abort_on_import_failure: bool = False,
) -> Dict[str, Any]:
    """Dispatch to the python or cpp Path A scorer.

    Default 'python' must be byte-for-byte identical to a direct call into
    score_path_a_arrays(...) -- callers that already use score_path_a_arrays
    do not need to route through this dispatcher.

    'cpp' lazy-imports _ppmd_cpp; if abort_on_import_failure is True the
    failure raises; otherwise a warning is printed to stderr and we fall
    back to the python backend.

    Note: the cpp backend ignores max_positions and normalization_sample_every
    (the C++ scorer always processes the full provided arrays and does not
    emit normalization samples). Trim arrays before dispatching if needed.
    """

    if backend == "python":
        return score_path_a_arrays(
            target_ids,
            prev_ids,
            nll_nats,
            candidates,
            is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            max_positions=max_positions,
            normalization_sample_every=normalization_sample_every,
        )
    if backend == "cpp":
        out = _score_path_a_arrays_cpp(
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            abort_on_import_failure=abort_on_import_failure,
        )
        if out is None:
            # Soft fallback (only reached when abort_on_import_failure=False).
            return score_path_a_arrays(
                target_ids,
                prev_ids,
                nll_nats,
                candidates,
                is_boundary_token_lut,
                order=order,
                lambda_hi=lambda_hi,
                lambda_lo=lambda_lo,
                conf_threshold=conf_threshold,
                max_positions=max_positions,
                normalization_sample_every=normalization_sample_every,
            )
        return out
    if backend == "cuda":
        out = _score_path_a_arrays_cuda(
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            abort_on_import_failure=abort_on_import_failure,
        )
        if out is None:
            # Soft fallback.
            return score_path_a_arrays(
                target_ids,
                prev_ids,
                nll_nats,
                candidates,
                is_boundary_token_lut,
                order=order,
                lambda_hi=lambda_hi,
                lambda_lo=lambda_lo,
                conf_threshold=conf_threshold,
                max_positions=max_positions,
                normalization_sample_every=normalization_sample_every,
            )
        return out
    if backend == "auto":
        # Try cuda first, then cpp, then python.
        cuda_out = _score_path_a_arrays_cuda(
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            abort_on_import_failure=False,
        )
        if cuda_out is not None:
            return cuda_out
        cpp_out = _score_path_a_arrays_cpp(
            target_ids=target_ids,
            prev_ids=prev_ids,
            nll_nats=nll_nats,
            candidates=candidates,
            is_boundary_token_lut=is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            abort_on_import_failure=False,
        )
        if cpp_out is not None:
            return cpp_out
        return score_path_a_arrays(
            target_ids,
            prev_ids,
            nll_nats,
            candidates,
            is_boundary_token_lut,
            order=order,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
            max_positions=max_positions,
            normalization_sample_every=normalization_sample_every,
        )
    raise ValueError(f"unknown backend {backend!r}; expected 'python', 'cpp', 'cuda', or 'auto'")


def score_path_a_arrays(
    target_ids: Sequence[int],
    prev_ids: Sequence[int],
    nll_nats: Sequence[float],
    candidates: Sequence[CandidateBytes],
    is_boundary_token_lut: Sequence[bool],
    *,
    order: int = 5,
    lambda_hi: float = 0.9,
    lambda_lo: float = 0.05,
    conf_threshold: float = 0.9,
    max_positions: Optional[int] = None,
    normalization_sample_every: int = 0,
) -> Dict[str, Any]:
    """Score a canonical target/prev/NLL stream with exact Path A.

    This is the handoff point for a fresh eval: collect neural NLLs in absolute
    prediction-position order, then call this function on rank 0 (or replace the
    trie traversal with a compiled backend using the same semantics).
    """

    total_positions = min(len(target_ids), len(prev_ids), len(nll_nats))
    if max_positions is not None:
        total_positions = min(total_positions, int(max_positions))
    boundary_root, non_boundary_root = build_candidate_tries(candidates)
    state = PPMDState(order=order)
    total_bits = 0.0
    total_bytes = 0
    samples: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for pos in range(total_positions):
        target_id = int(target_ids[pos])
        prev_id = int(prev_ids[pos])
        prev_is_boundary = prev_id < 0 or (
            bool(is_boundary_token_lut[prev_id]) if 0 <= prev_id < len(is_boundary_token_lut) else True
        )
        actual_bytes = actual_bytes_for_position(candidates, target_id, prev_is_boundary)
        rec = path_a_score_position(
            state,
            boundary_root,
            non_boundary_root,
            target_id,
            prev_is_boundary,
            float(nll_nats[pos]),
            actual_bytes,
            lambda_hi=lambda_hi,
            lambda_lo=lambda_lo,
            conf_threshold=conf_threshold,
        )
        total_bits += rec["loss_bits"]
        total_bytes += len(actual_bytes)
        if normalization_sample_every and pos % normalization_sample_every == 0:
            samples.append(
                {
                    "position": pos,
                    "z": rec["z"],
                    "target_id": target_id,
                    "p_mix_target": rec["p_mix_target"],
                    "p_ppm_target": rec["p_ppm_target"],
                    "lambda": rec["lambda"],
                    "score_first": rec["state_changed_only_after_update"],
                }
            )

    elapsed = time.perf_counter() - t0
    return {
        "mode": "path-a-array-score",
        "positions": total_positions,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
        "bpb": total_bits / total_bytes if total_bytes else None,
        "elapsed_seconds": elapsed,
        "positions_per_second": total_positions / elapsed if elapsed > 0 else None,
        "normalization_samples": samples,
        "candidate_trie_stats": {
            "boundary": trie_stats(boundary_root),
            "non_boundary": trie_stats(non_boundary_root),
        },
    }


def estimate_path_a_cost(
    *,
    target_tokens: int = KNOWN_TARGET_TOKENS,
    vocab_size: int = 8192,
    avg_candidate_bytes: float = 3.73,
    order: int = 5,
) -> Dict[str, Any]:
    candidate_token_evals = target_tokens * vocab_size
    candidate_byte_extensions = int(candidate_token_evals * avg_candidate_bytes)
    context_probes = int(candidate_byte_extensions * (order + 1))
    return {
        "target_tokens": target_tokens,
        "vocab_size": vocab_size,
        "avg_candidate_bytes_assumption": avg_candidate_bytes,
        "order": order,
        "candidate_token_evals": candidate_token_evals,
        "candidate_byte_extensions": candidate_byte_extensions,
        "worst_case_context_probes": context_probes,
        "rough_wallclock_seconds_by_probe_rate": {
            "1e8_probes_per_sec": context_probes / 1e8,
            "1e9_probes_per_sec": context_probes / 1e9,
            "5e9_probes_per_sec": context_probes / 5e9,
        },
    }


def import_exp1876_module(source_path: Path = EXP1876_SOURCE) -> Any:
    spec = importlib.util.spec_from_file_location("exp1876_train_gpt_merged", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_torch_device() -> Tuple[Any, Any, Any]:
    """Import torch lazily and initialize CUDA/distributed if available."""

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Fresh exp_1876 eval requires CUDA/FA3; use --core-smoke on CPU-only hosts")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return torch, dist, F


def _configure_exp_h(module: Any, args: argparse.Namespace) -> Any:
    h = module.Hyperparameters()
    h.data_dir = str(args.data_dir)
    h.datasets_dir = os.path.join(h.data_dir, "datasets", f"fineweb10B_sp{h.vocab_size}")
    h.val_files = os.path.join(h.datasets_dir, "fineweb_val_*.bin")
    h.tokenizer_path = os.path.join(h.data_dir, "tokenizers", f"fineweb_{h.vocab_size}_bpe.model")
    h.quantized_model_path = str(args.model_path.resolve())
    h.ppm_enabled = True
    h.logfile = None
    h.rank = int(os.environ.get("RANK", "0"))
    h.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    h.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    h.is_main_process = h.rank == 0
    h.grad_accum_steps = max(1, 8 // max(1, h.world_size))
    return h


def collect_neural_sliding_arrays(
    module: Any,
    h: Any,
    device: Any,
    val_data: Any,
    model: Any,
    *,
    max_positions: Optional[int],
    batch_seqs: int,
) -> Optional[Dict[str, Any]]:
    """Collect canonical per-position target/prev/NLL arrays for sliding eval."""

    torch, dist, F = _init_torch_device()
    model.eval()
    logits_fn = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = int(val_data.val_tokens.numel() - 1)
    cap = total_tokens if max_positions is None else min(total_tokens, int(max_positions))
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]

    nll_buf = torch.zeros((cap,), dtype=torch.float64, device=device)
    tgt_buf = torch.zeros((cap,), dtype=torch.int64, device=device)
    prev_buf = torch.zeros((cap,), dtype=torch.int64, device=device)
    written = torch.zeros((cap,), dtype=torch.int32, device=device)
    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    byte_count = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            if not batch_ws:
                continue
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: List[int] = []
            for i, ws in enumerate(batch_ws):
                if ws >= cap:
                    wlens.append(0)
                    continue
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                if wlen <= 0:
                    continue
                s = 0 if ws == 0 else context_size
                abs_s = ws + s
                if abs_s >= cap:
                    continue
                n = min(wlen - s, cap - abs_s)
                if n <= 0:
                    continue
                abs_e = abs_s + n
                scored_nll = nll[i, s : s + n].to(torch.float64)
                tgt = y_batch[i, s : s + n]
                prev = x_batch[i, s : s + n]
                nll_buf[abs_s:abs_e] = scored_nll
                tgt_buf[abs_s:abs_e] = tgt.to(torch.int64)
                prev_buf[abs_s:abs_e] = prev.to(torch.int64)
                written[abs_s:abs_e] = 1
                loss_sum += scored_nll.sum()
                token_count += float(n)
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(nll_buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(tgt_buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(prev_buf, op=dist.ReduceOp.SUM)
        dist.all_reduce(written, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    if h.rank != 0:
        return None
    import numpy as np

    written_np = written.detach().cpu().numpy()
    missing = int(np.count_nonzero(written_np[:cap] == 0))
    if missing:
        raise RuntimeError(f"Neural sliding collection missed {missing} of {cap} prefix positions")
    return {
        "target_ids": tgt_buf.detach().cpu().numpy().astype(np.int64),
        "prev_ids": prev_buf.detach().cpu().numpy().astype(np.int64),
        "nll_nats": nll_buf.detach().cpu().numpy().astype(np.float64),
        "neural_loss_sum_nats": float(loss_sum.item()),
        "neural_token_count": float(token_count.item()),
        "neural_byte_count": float(byte_count.item()),
        "cap": cap,
    }


def _build_candidates_from_val_data(val_data: Any, h: Any) -> Tuple[List[CandidateBytes], List[bool]]:
    token_bytes_lut = val_data.token_bytes_py
    if token_bytes_lut is None:
        token_bytes_lut = []
    has_space = val_data.has_leading_space_lut.detach().cpu().numpy().astype(bool).tolist()
    is_boundary = val_data.is_boundary_token_lut.detach().cpu().numpy().astype(bool).tolist()
    candidates: List[CandidateBytes] = []
    for token_id in range(h.vocab_size):
        emittable = True
        try:
            emittable = not (
                val_data.sp.is_control(token_id) or val_data.sp.is_unknown(token_id) or val_data.sp.is_unused(token_id)
            )
        except Exception:
            emittable = True
        candidates.append(candidate_bytes_for_token(token_id, token_bytes_lut, has_space, emittable=emittable))
    return candidates, is_boundary


def run_core_smoke(output: Path, max_positions: int) -> Dict[str, Any]:
    """Run a tiny synthetic Path A smoke without importing torch/FA3."""

    token_bytes = [b"a", b"b", b"ab", b""]
    has_space = [False, True, False, False]
    candidates = [candidate_bytes_for_token(i, token_bytes, has_space) for i in range(len(token_bytes))]
    boundary_root, non_boundary_root = build_candidate_tries(candidates)
    state = PPMDState(order=2)
    targets = [0, 1, 2, 0][:max_positions]
    prev_boundary = [True, False, False, False][:max_positions]
    nll = [math.log(4.0), math.log(4.0), math.log(4.0), math.log(4.0)][:max_positions]
    total_bits = 0.0
    positions = []
    for i, target in enumerate(targets):
        actual = candidates[target].after_boundary if prev_boundary[i] else candidates[target].after_non_boundary
        rec = path_a_score_position(
            state,
            boundary_root,
            non_boundary_root,
            target,
            prev_boundary[i],
            nll[i],
            actual,
            lambda_hi=0.5,
            lambda_lo=0.5,
        )
        total_bits += rec["loss_bits"]
        positions.append(rec)
    report = {
        "generated_at": now_iso(),
        "mode": "core-smoke",
        "positions": len(positions),
        "total_bits": total_bits,
        "candidate_trie_stats": {
            "boundary": trie_stats(boundary_root),
            "non_boundary": trie_stats(non_boundary_root),
        },
        "all_score_first": all(p["state_changed_only_after_update"] for p in positions if p["state_digest_before"] != p["state_digest_after_update"]),
        "cost_estimate": estimate_path_a_cost(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_fresh_eval(args: argparse.Namespace) -> Dict[str, Any]:
    """Fresh eval entrypoint.

    This currently performs environment/artifact validation and then raises a
    deliberate error unless ``--core-smoke`` is used.  The exact Path A scoring
    core above is complete and tested; full neural NLL collection is left as a
    RunPod execution-plan step because importing the exp model requires FA3 and
    CUDA and full Path A Python evaluation is expected to be very slow.
    """

    if not args.allow_slow_python_full_eval:
        raise RuntimeError(
            "Fresh full Path A eval is intentionally guarded. Run with --core-smoke for local correctness, "
            "or pass --allow-slow-python-full-eval on a GPU pod after reviewing the RunPod plan."
        )
    os.environ.setdefault("COMPRESSOR", "brotli")
    os.environ.setdefault("PPM_ENABLED", "1")
    torch, dist, _ = _init_torch_device()
    module = import_exp1876_module(args.exp_source)
    h = _configure_exp_h(module, args)
    if hasattr(module, "set_logging_hparams"):
        module.set_logging_hparams(h)
    val_data = module.ValidationData(h, torch.device("cuda", h.local_rank))
    model = module.deserialize(h, torch.device("cuda", h.local_rank))
    if getattr(h, "num_loops", 0) > 0:
        model.looping_active = True
    arrays = collect_neural_sliding_arrays(
        module,
        h,
        torch.device("cuda", h.local_rank),
        val_data,
        model,
        max_positions=args.max_positions,
        batch_seqs=args.batch_seqs,
    )
    if h.rank != 0:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return {"mode": "fresh-eval-worker", "rank": h.rank, "generated_at": now_iso(), "skip_output_write": True}
    candidates, is_boundary = _build_candidates_from_val_data(val_data, h)
    backend = getattr(args, "backend", "python")
    score = _score_path_a_arrays_dispatch(
        backend,
        target_ids=arrays["target_ids"],
        prev_ids=arrays["prev_ids"],
        nll_nats=arrays["nll_nats"],
        candidates=candidates,
        is_boundary_token_lut=is_boundary,
        order=args.ppm_order,
        lambda_hi=args.ppm_lambda_hi,
        lambda_lo=args.ppm_lambda_lo,
        conf_threshold=args.ppm_conf_threshold,
        max_positions=args.max_positions,
        normalization_sample_every=args.normalization_sample_every,
        abort_on_import_failure=bool(getattr(args, "backend_equiv_check", 0)),
    )

    equiv_check_report: Optional[Dict[str, Any]] = None
    equiv_k = int(getattr(args, "backend_equiv_check", 0) or 0)
    if equiv_k > 0 and backend in ("cpp", "cuda", "auto"):
        # Re-score the first K positions with python AND the selected backend.
        k = min(equiv_k, len(arrays["target_ids"]))
        ti = arrays["target_ids"][:k]
        pi = arrays["prev_ids"][:k]
        nl = arrays["nll_nats"][:k]
        py_slice = _score_path_a_arrays_dispatch(
            "python",
            target_ids=ti, prev_ids=pi, nll_nats=nl,
            candidates=candidates, is_boundary_token_lut=is_boundary,
            order=args.ppm_order, lambda_hi=args.ppm_lambda_hi,
            lambda_lo=args.ppm_lambda_lo, conf_threshold=args.ppm_conf_threshold,
        )
        backend_slice = _score_path_a_arrays_dispatch(
            backend,
            target_ids=ti, prev_ids=pi, nll_nats=nl,
            candidates=candidates, is_boundary_token_lut=is_boundary,
            order=args.ppm_order, lambda_hi=args.ppm_lambda_hi,
            lambda_lo=args.ppm_lambda_lo, conf_threshold=args.ppm_conf_threshold,
            abort_on_import_failure=True,
        )
        py_bpb = float(py_slice["bpb"]) if py_slice.get("bpb") is not None else float("nan")
        backend_bpb = float(backend_slice["bpb"]) if backend_slice.get("bpb") is not None else float("nan")
        diff = abs(py_bpb - backend_bpb)
        equiv_check_report = {
            "k": k,
            "py_bpb": py_bpb,
            "backend_bpb": backend_bpb,
            "backend": backend,
            "abs_bpb_diff": diff,
            "tolerance": 1e-10,
        }
        if not (diff <= 1e-10):
            print(
                f"FATAL: --backend-equiv-check failed at k={k}: "
                f"py_bpb={py_bpb!r} {backend}_bpb={backend_bpb!r} diff={diff!r} > 1e-10",
                file=sys.stderr,
            )
            sys.exit(3)
    report = {
        "generated_at": now_iso(),
        "mode": "fresh-eval-sliding-prefix-path-a",
        "exp_source": str(args.exp_source),
        "exp_source_sha256": sha256_file(args.exp_source),
        "model_path": str(args.model_path),
        "model_sha256": sha256_file(args.model_path),
        "neural_mode": "sliding",
        "neural_collection": {k: v for k, v in arrays.items() if k not in {"target_ids", "prev_ids", "nll_nats"}},
        "path_a_score": score,
        "backend": backend,
        "backend_equiv_check": equiv_check_report,
        "known_full_target_tokens": KNOWN_TARGET_TOKENS,
        "known_full_target_bytes": KNOWN_TARGET_BYTES,
        "warning": "This is exact Python Path A; use prefix smoke first. Full validation likely needs a compiled backend.",
    }
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-smoke", action="store_true", help="run a local synthetic Path A correctness smoke")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="optional prefix positions; omit for full fresh eval, core-smoke defaults to 4",
    )
    parser.add_argument("--batch-seqs", type=int, default=32, help="sliding neural eval batch sequences per rank")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON output path")
    parser.add_argument("--exp-source", type=Path, default=EXP1876_SOURCE, help="exp_1876 train_gpt_merged.py path")
    parser.add_argument("--model-path", type=Path, default=EXP1876_MODEL, help="compressed model artifact path")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data", help="data directory containing datasets/tokenizers")
    parser.add_argument("--ppm-order", type=int, default=5)
    parser.add_argument("--ppm-lambda-hi", type=float, default=0.9)
    parser.add_argument("--ppm-lambda-lo", type=float, default=0.05)
    parser.add_argument("--ppm-conf-threshold", type=float, default=0.9)
    parser.add_argument("--normalization-sample-every", type=int, default=1000)
    parser.add_argument(
        "--backend",
        choices=("python", "cpp", "cuda", "auto"),
        default="python",
        help=(
            "Path A scorer backend (default: python). "
            "'cpp' dispatches to _ppmd_cpp.score_path_a_arrays. "
            "'cuda' dispatches to _ppmd_cuda.cuda.score_path_a_arrays_cuda. "
            "'auto' prefers cuda > cpp > python (uses best available)."
        ),
    )
    parser.add_argument(
        "--backend-equiv-check",
        type=int,
        default=0,
        help="When > 0 AND --backend cpp or cuda, score the first K positions "
             "through BOTH the selected backend and the python backend and "
             "assert |bpb_py - bpb_backend| <= 1e-10. Aborts on mismatch or "
             "import failure.",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=None,
        help="Optional cap on positions loaded for a real-slice eval; alias of "
             "--max-positions for SLURM scripts that prefer the shorter name.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default=None,
        help="Optional base name (no extension) for the output JSON file under "
             "results/ppmd_cpp_eval/. Used by SLURM scripts to namespace per "
             "job ID.",
    )
    parser.add_argument(
        "--allow-slow-python-full-eval",
        action="store_true",
        help="acknowledge that full exact Path A in Python is a slow reference path",
    )
    parser.add_argument("--estimate-cost", action="store_true", help="print Path A cost estimate and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # --positions is a SLURM-friendly alias of --max-positions; if both are set
    # and disagree we take the smaller (most conservative) cap.
    if args.positions is not None:
        if args.max_positions is None:
            args.max_positions = int(args.positions)
        else:
            args.max_positions = min(int(args.max_positions), int(args.positions))

    # --results-name overrides the default --output destination, namespacing
    # under results/ppmd_cpp_eval/<name>.json (used by the real-slice SLURM
    # script).
    if args.results_name:
        args.output = REPO_ROOT / "results" / "ppmd_cpp_eval" / f"{args.results_name}.json"

    # Validate equiv-check semantics: requires --backend cpp or cuda.
    if args.backend_equiv_check and args.backend not in ("cpp", "cuda", "auto"):
        print(
            f"FATAL: --backend-equiv-check {args.backend_equiv_check} requires "
            f"--backend cpp or cuda (got --backend {args.backend!r}).",
            file=sys.stderr,
        )
        return 2
    # In equiv-check mode import failures must abort, not soft-fall back.
    if args.backend in ("cpp", "auto") and args.backend_equiv_check:
        _import_ppmd_cpp(abort_on_failure=True)
    if args.backend in ("cuda", "auto") and args.backend_equiv_check:
        _import_ppmd_cuda(abort_on_failure=True)

    if args.estimate_cost:
        print(json.dumps(estimate_path_a_cost(), indent=2, sort_keys=True))
        return 0
    if args.core_smoke:
        report = run_core_smoke(args.output, max(1, args.max_positions or 4))
    else:
        report = run_fresh_eval(args)
        if not report.get("skip_output_write"):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": report.get("mode"), "generated_at": report.get("generated_at")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())