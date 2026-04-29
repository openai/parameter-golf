#!/usr/bin/env python3
"""Path B byte-trie, PPM-D, audit metadata, and dry-run CLI helpers.

This module provides import-light pieces for a proper byte-level Path B
evaluator, including the Phase 1/2 CPU reference primitives:

    p_nn(next_byte | byte_prefix)
        = sum_v p_nn(v) * 1[bytes(v) has byte_prefix + next_byte]
          / sum_v p_nn(v) * 1[bytes(v) strictly extends byte_prefix]

The denominator is the *continuable* token mass at the prefix.  Tokens that end
exactly at the prefix, including zero-byte special/control tokens stored as
root terminals, are excluded from that denominator because they cannot emit a
next byte.

The Phase 3 surface area adds a standalone, safe-by-default command line
interface for planning a fresh non-record Path B eval of the exp_1876 PPM-D
artifact.  ``--dry-run`` validates source/artifact paths and writes JSON audit
metadata without importing torch-heavy model code or scoring validation data.
The explicit future ``--eval`` path is intentionally guarded by a clear
``NotImplementedError`` until the distributed 8xH100 evaluator is completed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


SENTENCEPIECE_SPACE = "▁"
BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
PATH_B_VERSION = "path_b_ppmd_phase3_skeleton_v1"
SCHEMA_VERSION = 1
DEFAULT_SUBSET_TOKENS = 8_000_000
KNOWN_FULL_VALIDATION_BYTES = 151_078_222
KNOWN_FIRST_8M_TOKEN_BYTES = 29_365_687
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PYTHON_PATH = REPO_ROOT / "results" / "exp_1876_ppmd" / "train_gpt_merged.py"
DEFAULT_ARTIFACT_PATH = (
    REPO_ROOT / "results" / "exp_1876_ppmd" / "prod_8gpu_s42v2" / "final_model.int6.ptz"
)
DEFAULT_OUTPUT_JSON_PATH = REPO_ROOT / "results" / "exp_1876_ppmd" / "path_b_ppmd_eval_plan.json"
EVAL_KIND_SLIDING = "sliding"
EVAL_KIND_TTT = "ttt"
SUPPORTED_EVAL_KINDS = (EVAL_KIND_SLIDING, EVAL_KIND_TTT)
MODE_FLAG_TO_NAME = {0: "boundary", 1: "non_boundary"}
MODE_NAME_TO_FLAG = {name: flag for flag, name in MODE_FLAG_TO_NAME.items()}


@dataclass
class PathBEvalConfig:
    """Configuration for Path B dry-run planning and future explicit eval.

    The defaults target the exp_1876 production artifact while remaining safe
    to instantiate locally.  Real evaluation must be requested with ``--eval``
    and is currently a guarded skeleton.
    """

    source_python_path: Path = field(default_factory=lambda: DEFAULT_SOURCE_PYTHON_PATH)
    artifact_path: Path = field(default_factory=lambda: DEFAULT_ARTIFACT_PATH)
    output_json_path: Optional[Path] = field(default_factory=lambda: DEFAULT_OUTPUT_JSON_PATH)
    subset_tokens: int = DEFAULT_SUBSET_TOKENS
    ppmd_order: int = 5
    ppmd_lambda: float = 0.35
    ppmd_lambda_hi: float = 0.90
    ppmd_lambda_lo: float = 0.05
    ppmd_conf_threshold: float = 0.90
    ppmd_confidence_gating: bool = True
    eval_kind: str = EVAL_KIND_SLIDING
    full_eval: bool = False

    def __post_init__(self) -> None:
        self.source_python_path = Path(self.source_python_path)
        self.artifact_path = Path(self.artifact_path)
        self.output_json_path = None if self.output_json_path is None else Path(self.output_json_path)
        self.subset_tokens = int(self.subset_tokens)
        self.ppmd_order = int(self.ppmd_order)
        self.ppmd_lambda = float(self.ppmd_lambda)
        self.ppmd_lambda_hi = float(self.ppmd_lambda_hi)
        self.ppmd_lambda_lo = float(self.ppmd_lambda_lo)
        self.ppmd_conf_threshold = float(self.ppmd_conf_threshold)
        self.eval_kind = normalize_eval_kind(self.eval_kind)
        self.full_eval = bool(self.full_eval)
        if self.subset_tokens <= 0:
            raise ValueError("subset_tokens must be positive")
        if self.ppmd_order < 0:
            raise ValueError("ppmd_order must be non-negative")
        for name, value in (
            ("ppmd_lambda", self.ppmd_lambda),
            ("ppmd_lambda_hi", self.ppmd_lambda_hi),
            ("ppmd_lambda_lo", self.ppmd_lambda_lo),
            ("ppmd_conf_threshold", self.ppmd_conf_threshold),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def normalize_eval_kind(eval_kind: str) -> str:
    """Normalize and validate the explicit heavy-eval flavor."""

    value = str(eval_kind).strip().lower().replace("-", "_")
    if value not in SUPPORTED_EVAL_KINDS:
        raise ValueError(
            f"unknown eval_kind {eval_kind!r}; supported eval kinds are "
            f"{', '.join(SUPPORTED_EVAL_KINDS)}"
        )
    return value


def guard_explicit_eval_kind(config: PathBEvalConfig) -> None:
    """Guard future explicit eval modes so incomplete paths cannot fake BPB."""

    eval_kind = normalize_eval_kind(config.eval_kind)
    if eval_kind == EVAL_KIND_TTT:
        raise NotImplementedError(
            "TTT Path B eval is not implemented in this safe utility layer yet; "
            "future work must add explicit test-time-training accounting before use."
        )


@dataclass(frozen=True)
class ByteLogprobRecord:
    """One future distributed byte-score record keyed by absolute position."""

    absolute_token_position: int
    byte_offset_in_token: int
    byte_value: int
    neural_logprob: float

    def __post_init__(self) -> None:
        if int(self.absolute_token_position) < 0:
            raise ValueError("absolute_token_position must be non-negative")
        if int(self.byte_offset_in_token) < 0:
            raise ValueError("byte_offset_in_token must be non-negative")
        _validate_byte_value(self.byte_value)
        if not math.isfinite(float(self.neural_logprob)):
            raise ValueError("neural_logprob must be finite")

    @property
    def order_key(self) -> Tuple[int, int]:
        """Stable absolute ordering key used by shard merge."""

        return (int(self.absolute_token_position), int(self.byte_offset_in_token))

    def to_json_dict(self) -> Dict[str, Union[int, float]]:
        """Serialize to a compact JSON-compatible record."""

        return {
            "absolute_token_position": int(self.absolute_token_position),
            "byte_offset_in_token": int(self.byte_offset_in_token),
            "byte_value": int(self.byte_value),
            "neural_logprob": float(self.neural_logprob),
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "ByteLogprobRecord":
        """Parse a record previously emitted by ``to_json_dict``."""

        return cls(
            absolute_token_position=int(data["absolute_token_position"]),
            byte_offset_in_token=int(data["byte_offset_in_token"]),
            byte_value=int(data["byte_value"]),
            neural_logprob=float(data["neural_logprob"]),
        )


@dataclass(frozen=True)
class TokenByteSequences:
    """The two possible byte strings emitted by one token.

    SentencePiece uses ``▁`` as a word-boundary marker.  For challenge byte
    accounting, that marker is never encoded as its own UTF-8 bytes.  Instead,
    the marker is stripped from ``base_bytes`` and reintroduced as one literal
    ASCII space only when the previous token is *not* a boundary token.
    """

    token_id: int
    piece: str
    base_bytes: bytes
    after_boundary: bytes
    after_non_boundary: bytes
    has_leading_space: bool = False
    is_special: bool = False

    def bytes_for_mode(self, mode: str) -> bytes:
        if mode == "boundary":
            return self.after_boundary
        if mode == "non_boundary":
            return self.after_non_boundary
        raise ValueError("mode must be 'boundary' or 'non_boundary'")


@dataclass
class ByteTrieNode:
    """Byte trie node with terminal IDs and subtree token membership."""

    children: Dict[int, "ByteTrieNode"] = field(default_factory=dict)
    terminal_token_ids: List[int] = field(default_factory=list)
    subtree_token_ids: Set[int] = field(default_factory=set)


@dataclass(frozen=True)
class OptimizedTrieTables:
    """Flattened trie tables for interval/cumsum neural byte marginalization.

    ``token_order`` is a DFS terminal order.  For every node, the token IDs in
    that node's subtree occupy ``[subtree_starts[node], subtree_ends[node])`` in
    ``token_order``.  Exact terminals at the node occupy
    ``[terminal_starts[node], terminal_ends[node])``.  A cumsum over neural
    probabilities gathered in this order can therefore recover both numerator
    and denominator masses with two indexed reads per interval.
    """

    token_order: List[int]
    subtree_starts: List[int]
    subtree_ends: List[int]
    terminal_starts: List[int]
    terminal_ends: List[int]
    children_by_node: List[Dict[int, int]]

    @property
    def num_nodes(self) -> int:
        return len(self.subtree_starts)


@dataclass(frozen=True)
class TokenBytePath:
    """Trie path metadata for every emitted byte of one token/mode."""

    token_id: int
    mode: str
    byte_values: Tuple[int, ...]
    prefix_node_ids: Tuple[int, ...]
    child_node_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.byte_values) != len(self.prefix_node_ids):
            raise ValueError("byte_values and prefix_node_ids must have equal length")
        if len(self.byte_values) != len(self.child_node_ids):
            raise ValueError("byte_values and child_node_ids must have equal length")
        for byte_value in self.byte_values:
            _validate_byte_value(byte_value)


@dataclass(frozen=True)
class TokenPathMetadata:
    """Per-mode token byte paths used by vectorized target-byte scoring."""

    paths_by_mode: Dict[int, Dict[int, TokenBytePath]]
    mode_names_by_flag: Dict[int, str]

    def path_for(self, mode_flag: Union[int, str], token_id: int) -> TokenBytePath:
        flag = _normalize_mode_flag(mode_flag)
        paths = self.paths_by_mode.get(flag)
        if paths is None:
            raise KeyError(f"no token path metadata for mode flag {mode_flag!r}")
        try:
            return paths[int(token_id)]
        except KeyError as exc:
            raise KeyError(f"no token path metadata for token_id {token_id!r} in mode {flag}") from exc


@dataclass(frozen=True)
class ByteScore:
    """Probability/log-probability for one scored byte."""

    byte_value: int
    probability: float
    logprob: float


@dataclass(frozen=True)
class MixtureByteDistribution:
    """Dense 256-way mixture distribution and optional target score."""

    probs: List[float]
    target_byte: Optional[int] = None
    target_prob: Optional[float] = None
    target_logprob: Optional[float] = None


@dataclass(frozen=True)
class PPMDStreamScoreSummary:
    """Aggregate nll/BPB metrics for streaming neural+PPM-D byte scoring."""

    byte_count: int
    mix_nll: float
    ppm_nll: float
    nn_nll: float
    mix_bpb: float
    ppm_bpb: float
    nn_bpb: float
    lambdas: List[float]
    ppmd_history: bytes


def token_byte_sequences_from_piece(
    token_id: int,
    piece: str,
    *,
    is_special: bool = False,
    is_control: bool = False,
    is_unknown: bool = False,
    is_unused: bool = False,
) -> TokenByteSequences:
    """Construct boundary/non-boundary byte strings for a token piece.

    Special/control/unknown/unused tokens are represented as zero-byte terminal
    tokens.  They are still inserted into tries by callers so their neural mass
    can be explicitly excluded from byte-distribution denominators.

    Byte fallback pieces such as ``<0x20>`` map to their literal byte value.
    """

    special = bool(is_special or is_control or is_unknown or is_unused)
    if special:
        return TokenByteSequences(
            token_id=int(token_id),
            piece=str(piece),
            base_bytes=b"",
            after_boundary=b"",
            after_non_boundary=b"",
            has_leading_space=False,
            is_special=True,
        )

    text = str(piece)
    has_leading_space = text.startswith(SENTENCEPIECE_SPACE)
    payload = text[1:] if has_leading_space else text
    base = piece_payload_bytes(payload)
    after_boundary = bytes(base)
    after_non_boundary = (b" " + base) if has_leading_space else bytes(base)
    return TokenByteSequences(
        token_id=int(token_id),
        piece=text,
        base_bytes=bytes(base),
        after_boundary=after_boundary,
        after_non_boundary=after_non_boundary,
        has_leading_space=has_leading_space,
        is_special=False,
    )


def piece_payload_bytes(piece_payload: str) -> bytes:
    """Return raw bytes for a SentencePiece payload without a leading marker."""

    match = BYTE_FALLBACK_RE.match(piece_payload)
    if match:
        return bytes([int(match.group(1), 16)])
    return piece_payload.encode("utf-8")


def _normalize_mode_flag(mode_flag: Union[int, str]) -> int:
    """Normalize a trie mode flag to 0=boundary or 1=non_boundary."""

    if isinstance(mode_flag, str):
        name = mode_flag.strip().lower().replace("-", "_")
        if name not in MODE_NAME_TO_FLAG:
            raise ValueError("mode flag must be 0/'boundary' or 1/'non_boundary'")
        return MODE_NAME_TO_FLAG[name]
    flag = int(mode_flag)
    if flag not in MODE_FLAG_TO_NAME:
        raise ValueError("mode flag must be 0/'boundary' or 1/'non_boundary'")
    return flag


def _mode_name_from_flag(mode_flag: Union[int, str]) -> str:
    return MODE_FLAG_TO_NAME[_normalize_mode_flag(mode_flag)]


def _lookup_by_mode(mapping: Any, mode_flag: Union[int, str]) -> Any:
    """Look up a mode-keyed mapping by numeric flag or canonical name."""

    flag = _normalize_mode_flag(mode_flag)
    if isinstance(mapping, (list, tuple)):
        return mapping[flag]
    if flag in mapping:
        return mapping[flag]
    name = MODE_FLAG_TO_NAME[flag]
    if name in mapping:
        return mapping[name]
    raise KeyError(f"mapping has no entry for mode {flag}/{name}")


def build_byte_trie(sequences: Sequence[TokenByteSequences], *, mode: str) -> ByteTrieNode:
    """Build a byte trie for one previous-token boundary mode."""

    if mode not in {"boundary", "non_boundary"}:
        raise ValueError("mode must be 'boundary' or 'non_boundary'")
    root = ByteTrieNode()
    for token in sequences:
        insert_token_bytes(root, token.bytes_for_mode(mode), token.token_id)
    return root


def build_mode_tries(sequences: Sequence[TokenByteSequences]) -> Tuple[ByteTrieNode, ByteTrieNode]:
    """Build the previous-boundary and previous-non-boundary tries."""

    return build_byte_trie(sequences, mode="boundary"), build_byte_trie(sequences, mode="non_boundary")


def insert_token_bytes(root: ByteTrieNode, byte_sequence: bytes, token_id: int) -> None:
    """Insert one token byte sequence, preserving terminals at every prefix."""

    node = root
    tid = int(token_id)
    node.subtree_token_ids.add(tid)
    for byte_value in byte_sequence:
        b = int(byte_value)
        node = node.children.setdefault(b, ByteTrieNode())
        node.subtree_token_ids.add(tid)
    node.terminal_token_ids.append(tid)


def find_prefix_node(root: ByteTrieNode, prefix: bytes) -> Optional[ByteTrieNode]:
    """Find the trie node corresponding to ``prefix``; return ``None`` if absent."""

    node = root
    for byte_value in prefix:
        node = node.children.get(int(byte_value))
        if node is None:
            return None
    return node


def token_id_mass(token_ids: Sequence[int], token_probs: Sequence[float]) -> float:
    """Sum neural token probabilities for the given token IDs."""

    total = 0.0
    n = len(token_probs)
    for token_id in token_ids:
        tid = int(token_id)
        if tid < 0 or tid >= n:
            raise IndexError(f"token_id {tid} has no probability entry")
        total += float(token_probs[tid])
    return total


def subtree_mass(node: ByteTrieNode, token_probs: Sequence[float]) -> float:
    """Mass of all token IDs in this prefix subtree."""

    return token_id_mass(sorted(node.subtree_token_ids), token_probs)


def terminal_mass(node: ByteTrieNode, token_probs: Sequence[float]) -> float:
    """Mass of tokens that end exactly at this prefix."""

    return token_id_mass(node.terminal_token_ids, token_probs)


def continuable_mass(node: ByteTrieNode, token_probs: Sequence[float]) -> float:
    """Mass of tokens that strictly extend this prefix and can emit a next byte."""

    return subtree_mass(node, token_probs) - terminal_mass(node, token_probs)


def neural_byte_distribution(
    root: ByteTrieNode,
    token_probs: Sequence[float],
    prefix: bytes = b"",
) -> Dict[int, float]:
    """CPU reference Path B neural next-byte distribution for ``prefix``.

    The denominator is ``subtree_mass(prefix) - terminal_mass(prefix)``.  That
    excludes exact terminal tokens, including zero-byte specials at the root,
    from the next-byte distribution while preserving their mass for explicit
    accounting/auditing.
    """

    node = find_prefix_node(root, prefix)
    if node is None:
        raise KeyError(f"prefix {prefix!r} is not present in the byte trie")
    denom = continuable_mass(node, token_probs)
    if denom <= 0.0:
        raise ValueError(f"prefix {prefix!r} has no positive continuable token mass")

    dist: Dict[int, float] = {}
    for byte_value, child in sorted(node.children.items()):
        mass = subtree_mass(child, token_probs)
        if mass > 0.0:
            dist[int(byte_value)] = mass / denom
    return dist


reference_neural_byte_distribution = neural_byte_distribution


def build_optimized_trie_tables(root: ByteTrieNode) -> OptimizedTrieTables:
    """Flatten a byte trie into interval tables for cumsum mass queries.

    The table construction does not require token IDs to be contiguous in the
    original vocabulary.  Instead, each terminal token ID is placed in a DFS
    order that makes every trie subtree a contiguous interval.  Runtime scoring
    gathers neural token probabilities into this order once, builds a cumsum,
    and obtains child-prefix masses by interval subtraction.
    """

    token_order: List[int] = []
    subtree_starts: List[int] = []
    subtree_ends: List[int] = []
    terminal_starts: List[int] = []
    terminal_ends: List[int] = []
    children_by_node: List[Dict[int, int]] = []

    def visit(node: ByteTrieNode) -> int:
        node_id = len(subtree_starts)
        subtree_starts.append(0)
        subtree_ends.append(0)
        terminal_starts.append(0)
        terminal_ends.append(0)
        children_by_node.append({})

        subtree_start = len(token_order)
        terminal_start = len(token_order)
        token_order.extend(int(token_id) for token_id in node.terminal_token_ids)
        terminal_end = len(token_order)

        child_map: Dict[int, int] = {}
        for byte_value, child in sorted(node.children.items()):
            child_map[int(byte_value)] = visit(child)

        subtree_starts[node_id] = subtree_start
        subtree_ends[node_id] = len(token_order)
        terminal_starts[node_id] = terminal_start
        terminal_ends[node_id] = terminal_end
        children_by_node[node_id] = child_map
        return node_id

    root_id = visit(root)
    if root_id != 0:
        raise AssertionError("internal error: root node was not assigned ID 0")
    if len(token_order) != len(root.subtree_token_ids):
        raise ValueError("trie terminal count does not match root subtree token membership")
    return OptimizedTrieTables(
        token_order=token_order,
        subtree_starts=subtree_starts,
        subtree_ends=subtree_ends,
        terminal_starts=terminal_starts,
        terminal_ends=terminal_ends,
        children_by_node=children_by_node,
    )


def build_token_path_metadata(
    sequences: Sequence[TokenByteSequences],
    trie: ByteTrieNode,
    tables: OptimizedTrieTables,
    *,
    mode: str,
) -> Dict[int, TokenBytePath]:
    """Build per-token byte path metadata by walking an existing trie.

    For every emitted byte, the metadata stores the byte value, the node ID for
    the prefix before that byte, and the child node ID after consuming it.  The
    node IDs refer to ``tables`` and are later used for interval mass gathers.
    """

    if mode not in {"boundary", "non_boundary"}:
        raise ValueError("mode must be 'boundary' or 'non_boundary'")

    paths: Dict[int, TokenBytePath] = {}
    for token in sequences:
        byte_values: List[int] = []
        prefix_node_ids: List[int] = []
        child_node_ids: List[int] = []
        node = trie
        node_id = 0

        for byte_value in token.bytes_for_mode(mode):
            b = _validate_byte_value(byte_value)
            child = node.children.get(b)
            if child is None:
                raise ValueError(f"token {token.token_id} byte path is absent from the {mode} trie")
            child_id = tables.children_by_node[node_id].get(b)
            if child_id is None:
                raise ValueError(f"token {token.token_id} byte path is absent from optimized {mode} tables")
            byte_values.append(b)
            prefix_node_ids.append(node_id)
            child_node_ids.append(child_id)
            node = child
            node_id = child_id

        if int(token.token_id) not in node.terminal_token_ids:
            raise ValueError(f"token {token.token_id} is not terminal at its {mode} trie path")
        paths[int(token.token_id)] = TokenBytePath(
            token_id=int(token.token_id),
            mode=mode,
            byte_values=tuple(byte_values),
            prefix_node_ids=tuple(prefix_node_ids),
            child_node_ids=tuple(child_node_ids),
        )
    return paths


def build_mode_token_path_metadata(
    sequences: Sequence[TokenByteSequences],
    tries_by_mode: Mapping[Union[int, str], ByteTrieNode],
    tables_by_mode: Mapping[Union[int, str], OptimizedTrieTables],
) -> TokenPathMetadata:
    """Build token path metadata for boundary and non-boundary modes."""

    paths_by_mode: Dict[int, Dict[int, TokenBytePath]] = {}
    mode_names_by_flag: Dict[int, str] = {}
    for mode_flag, mode_name in MODE_FLAG_TO_NAME.items():
        trie = _lookup_by_mode(tries_by_mode, mode_flag)
        tables = _lookup_by_mode(tables_by_mode, mode_flag)
        paths_by_mode[mode_flag] = build_token_path_metadata(sequences, trie, tables, mode=mode_name)
        mode_names_by_flag[mode_flag] = mode_name
    return TokenPathMetadata(paths_by_mode=paths_by_mode, mode_names_by_flag=mode_names_by_flag)


build_token_path_metadata_from_trie = build_token_path_metadata


def _find_prefix_node_id(tables: OptimizedTrieTables, prefix: bytes) -> Optional[int]:
    node_id = 0
    for byte_value in bytes(prefix):
        node_id = tables.children_by_node[node_id].get(int(byte_value))
        if node_id is None:
            return None
    return node_id


def _optional_torch():
    try:
        import torch  # type: ignore
    except Exception:
        return None
    return torch


def _optional_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


def _validate_token_prob_entries(token_order: Sequence[int], token_probs: Sequence[float]) -> None:
    n = len(token_probs)
    for token_id in token_order:
        tid = int(token_id)
        if tid < 0 or tid >= n:
            raise IndexError(f"token_id {tid} has no probability entry")


def _python_interval_prefix_sums(tables: OptimizedTrieTables, token_probs: Sequence[float]) -> List[float]:
    _validate_token_prob_entries(tables.token_order, token_probs)
    cumsum = [0.0]
    total = 0.0
    for token_id in tables.token_order:
        total += float(token_probs[int(token_id)])
        cumsum.append(total)
    return cumsum


def _python_interval_mass(cumsum: Sequence[float], start: int, end: int) -> float:
    return float(cumsum[int(end)] - cumsum[int(start)])


def optimized_neural_byte_distribution(
    tables: OptimizedTrieTables,
    token_probs: Sequence[float],
    prefix: bytes = b"",
    *,
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Optimized Path B neural next-byte distribution using interval masses.

    When PyTorch is importable, this uses torch tensors and ``cumsum`` on the
    requested device (CPU by default, CUDA later if supplied by callers).  If
    torch is unavailable, it falls back to the same interval algorithm in pure
    Python.  Returned probabilities are Python floats to keep the Phase 1 API
    style intact.
    """

    node_id = _find_prefix_node_id(tables, prefix)
    if node_id is None:
        raise KeyError(f"prefix {prefix!r} is not present in the byte trie")

    torch = _optional_torch()
    if torch is not None:
        probs_tensor = torch.as_tensor(token_probs, dtype=torch.float64, device=device)
        order = torch.as_tensor(tables.token_order, dtype=torch.long, device=probs_tensor.device)
        ordered = probs_tensor.index_select(0, order)
        zero = torch.zeros(1, dtype=ordered.dtype, device=ordered.device)
        cumsum = torch.cat((zero, torch.cumsum(ordered, dim=0)))

        def mass(start: int, end: int) -> float:
            return float((cumsum[int(end)] - cumsum[int(start)]).item())

    else:
        cumsum_py = _python_interval_prefix_sums(tables, token_probs)

        def mass(start: int, end: int) -> float:
            return _python_interval_mass(cumsum_py, start, end)

    subtree = mass(tables.subtree_starts[node_id], tables.subtree_ends[node_id])
    terminal = mass(tables.terminal_starts[node_id], tables.terminal_ends[node_id])
    denom = subtree - terminal
    if denom <= 0.0:
        raise ValueError(f"prefix {prefix!r} has no positive continuable token mass")

    dist: Dict[int, float] = {}
    for byte_value, child_id in sorted(tables.children_by_node[node_id].items()):
        numerator = mass(tables.subtree_starts[child_id], tables.subtree_ends[child_id])
        if numerator > 0.0:
            dist[int(byte_value)] = numerator / denom
    return dist


def _as_cpu_list(values: Any, *, dtype: str) -> List[Any]:
    """Convert a tensor/list-like object to a small Python list for metadata lookup."""

    if hasattr(values, "detach"):
        raw = values.detach().cpu().tolist()
    else:
        raw = list(values)
    if dtype == "int":
        return [int(value) for value in raw]
    return raw


def vectorized_target_path_logprobs(
    probability_tensor: Any,
    target_ids: Any,
    mode_flags: Any,
    tables_by_mode: Mapping[Union[int, str], OptimizedTrieTables],
    path_metadata: TokenPathMetadata,
    *,
    absolute_token_positions: Optional[Sequence[int]] = None,
) -> List[ByteLogprobRecord]:
    """Extract neural log-probs for every emitted byte of target token paths.

    ``probability_tensor`` is ``[N, V]`` token probabilities.  For each row, the
    target token path is decoded in the selected mode.  The routine computes a
    single ordered-token cumsum per mode and gathers child-subtree numerators and
    continuable-prefix denominators, where denominators exclude terminal mass at
    the current prefix.
    """

    torch = _optional_torch()
    if torch is None:
        raise ImportError("torch is required for vectorized target path logprob extraction")

    probs = torch.as_tensor(probability_tensor)
    if probs.ndim != 2:
        raise ValueError(f"probability_tensor must have shape [N, V], got {tuple(probs.shape)!r}")
    if not probs.is_floating_point():
        probs = probs.to(dtype=torch.float64)

    target_list = _as_cpu_list(target_ids, dtype="int")
    raw_mode_list = _as_cpu_list(mode_flags, dtype="raw")
    mode_list = [_normalize_mode_flag(value) for value in raw_mode_list]
    row_count = int(probs.shape[0])
    if len(target_list) != row_count:
        raise ValueError(f"target_ids length {len(target_list)} does not match probability rows {row_count}")
    if len(mode_list) != row_count:
        raise ValueError(f"mode_flags length {len(mode_list)} does not match probability rows {row_count}")

    if absolute_token_positions is None:
        absolute_positions = list(range(row_count))
    else:
        absolute_positions = [int(value) for value in absolute_token_positions]
        if len(absolute_positions) != row_count:
            raise ValueError(
                f"absolute_token_positions length {len(absolute_positions)} does not match probability rows {row_count}"
            )

    record_rows: List[int] = []
    record_modes: List[int] = []
    record_prefix_nodes: List[int] = []
    record_child_nodes: List[int] = []
    record_positions: List[int] = []
    record_offsets: List[int] = []
    record_bytes: List[int] = []

    vocab_size = int(probs.shape[1])
    for row_index, (target_id, mode_flag) in enumerate(zip(target_list, mode_list)):
        if target_id < 0 or target_id >= vocab_size:
            raise IndexError(f"target_id {target_id} is outside probability vocabulary size {vocab_size}")
        path = path_metadata.path_for(mode_flag, target_id)
        for byte_offset, byte_value in enumerate(path.byte_values):
            record_rows.append(row_index)
            record_modes.append(mode_flag)
            record_prefix_nodes.append(path.prefix_node_ids[byte_offset])
            record_child_nodes.append(path.child_node_ids[byte_offset])
            record_positions.append(absolute_positions[row_index])
            record_offsets.append(byte_offset)
            record_bytes.append(int(byte_value))

    if not record_rows:
        return []

    logprobs: List[Optional[float]] = [None] * len(record_rows)
    device = probs.device
    for mode_flag in sorted(set(record_modes)):
        tables = _lookup_by_mode(tables_by_mode, mode_flag)
        mode_record_indices = [index for index, value in enumerate(record_modes) if value == mode_flag]

        order = torch.as_tensor(tables.token_order, dtype=torch.long, device=device)
        if order.numel() == 0:
            raise ValueError(f"optimized trie tables for mode {mode_flag} have empty token order")
        if int(order.max().item()) >= vocab_size or int(order.min().item()) < 0:
            raise IndexError(f"optimized trie tables for mode {mode_flag} reference token IDs outside vocabulary")

        ordered_probs = probs.index_select(1, order)
        zero = torch.zeros((row_count, 1), dtype=ordered_probs.dtype, device=device)
        cumsum = torch.cat((zero, torch.cumsum(ordered_probs, dim=1)), dim=1)

        rows = torch.as_tensor([record_rows[i] for i in mode_record_indices], dtype=torch.long, device=device)
        prefix_nodes = [record_prefix_nodes[i] for i in mode_record_indices]
        child_nodes = [record_child_nodes[i] for i in mode_record_indices]

        prefix_subtree_starts = torch.as_tensor(
            [tables.subtree_starts[node_id] for node_id in prefix_nodes], dtype=torch.long, device=device
        )
        prefix_subtree_ends = torch.as_tensor(
            [tables.subtree_ends[node_id] for node_id in prefix_nodes], dtype=torch.long, device=device
        )
        prefix_terminal_starts = torch.as_tensor(
            [tables.terminal_starts[node_id] for node_id in prefix_nodes], dtype=torch.long, device=device
        )
        prefix_terminal_ends = torch.as_tensor(
            [tables.terminal_ends[node_id] for node_id in prefix_nodes], dtype=torch.long, device=device
        )
        child_starts = torch.as_tensor(
            [tables.subtree_starts[node_id] for node_id in child_nodes], dtype=torch.long, device=device
        )
        child_ends = torch.as_tensor(
            [tables.subtree_ends[node_id] for node_id in child_nodes], dtype=torch.long, device=device
        )

        prefix_subtree_mass = cumsum[rows, prefix_subtree_ends] - cumsum[rows, prefix_subtree_starts]
        prefix_terminal_mass = cumsum[rows, prefix_terminal_ends] - cumsum[rows, prefix_terminal_starts]
        denominator = prefix_subtree_mass - prefix_terminal_mass
        numerator = cumsum[rows, child_ends] - cumsum[rows, child_starts]

        if bool((denominator <= 0).detach().cpu().any().item()):
            raise ValueError(f"mode {mode_flag} has a target byte with no positive continuable prefix mass")
        if bool((numerator <= 0).detach().cpu().any().item()):
            raise ValueError(f"mode {mode_flag} has a target byte with no positive child subtree mass")

        mode_logprobs = torch.log(numerator / denominator).detach().cpu().tolist()
        for output_index, logprob in zip(mode_record_indices, mode_logprobs):
            logprobs[output_index] = float(logprob)

    records: List[ByteLogprobRecord] = []
    for index, logprob in enumerate(logprobs):
        if logprob is None:
            raise AssertionError("internal error: missing vectorized target byte logprob")
        records.append(
            ByteLogprobRecord(
                absolute_token_position=record_positions[index],
                byte_offset_in_token=record_offsets[index],
                byte_value=record_bytes[index],
                neural_logprob=logprob,
            )
        )
    return records


extract_vectorized_target_byte_logprobs = vectorized_target_path_logprobs


ByteDistributionInput = Union[Mapping[int, float], Sequence[float]]


def dense_byte_distribution(dist: ByteDistributionInput, *, normalize: bool = True) -> List[float]:
    """Convert a sparse mapping or dense sequence into a 256-way byte vector."""

    if isinstance(dist, Mapping):
        dense = [0.0] * 256
        for byte_value, prob in dist.items():
            b = _validate_byte_value(byte_value)
            p = float(prob)
            if p < 0.0:
                raise ValueError(f"negative probability for byte {byte_value!r}: {p}")
            dense[b] = p
    else:
        values = dist
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()  # type: ignore[assignment]
        dense = [float(prob) for prob in values]
        if len(dense) != 256:
            raise ValueError(f"dense byte distribution must have length 256, got {len(dense)}")
        if any(prob < 0.0 for prob in dense):
            raise ValueError("dense byte distribution contains negative probabilities")

    if normalize:
        total = sum(dense)
        if total <= 0.0:
            raise ValueError("byte distribution has no positive mass")
        dense = [prob / total for prob in dense]
    return dense


def optimized_neural_byte_distribution_dense(
    tables: OptimizedTrieTables,
    token_probs: Sequence[float],
    prefix: bytes = b"",
    *,
    device: Optional[str] = None,
) -> List[float]:
    """Dense 256-way adapter for ``optimized_neural_byte_distribution``."""

    return dense_byte_distribution(
        optimized_neural_byte_distribution(tables, token_probs, prefix, device=device),
        normalize=True,
    )


def bruteforce_neural_byte_distribution(
    sequences: Sequence[TokenByteSequences],
    token_probs: Sequence[float],
    prefix: bytes = b"",
    *,
    mode: str,
) -> Dict[int, float]:
    """Simple O(V) reference implementation used to audit trie results."""

    if mode not in {"boundary", "non_boundary"}:
        raise ValueError("mode must be 'boundary' or 'non_boundary'")
    prefix = bytes(prefix)
    numerators: Dict[int, float] = {}
    denom = 0.0
    for token in sequences:
        seq = token.bytes_for_mode(mode)
        if not seq.startswith(prefix) or len(seq) <= len(prefix):
            continue
        prob = float(token_probs[token.token_id])
        denom += prob
        next_byte = int(seq[len(prefix)])
        numerators[next_byte] = numerators.get(next_byte, 0.0) + prob
    if denom <= 0.0:
        raise ValueError(f"prefix {prefix!r} has no positive continuable token mass")
    return {byte_value: mass / denom for byte_value, mass in sorted(numerators.items()) if mass > 0.0}


def assert_distribution_normalized(dist: Mapping[int, float], *, tol: float = 1e-12) -> None:
    """Raise ``AssertionError`` if a byte distribution is malformed."""

    total = 0.0
    for byte_value, prob in dist.items():
        if int(byte_value) < 0 or int(byte_value) > 255:
            raise AssertionError(f"invalid byte value {byte_value!r}")
        p = float(prob)
        if p < -tol:
            raise AssertionError(f"negative probability for byte {byte_value!r}: {p}")
        total += p
    if abs(total - 1.0) > tol:
        raise AssertionError(f"distribution sums to {total:.17g}, not 1.0")


def _validate_byte_value(byte_value: int) -> int:
    b = int(byte_value)
    if b < 0 or b > 255:
        raise ValueError(f"byte value must be in [0, 255], got {byte_value!r}")
    return b


def assert_dense_distribution_normalized(dist: Sequence[float], *, tol: float = 1e-12) -> None:
    """Raise ``AssertionError`` if a dense 256-way byte distribution is malformed."""

    if len(dist) != 256:
        raise AssertionError(f"dense byte distribution has length {len(dist)}, not 256")
    total = 0.0
    for byte_value, prob in enumerate(dist):
        p = float(prob)
        if p < -tol:
            raise AssertionError(f"negative probability for byte {byte_value!r}: {p}")
        total += p
    if abs(total - 1.0) > tol:
        raise AssertionError(f"distribution sums to {total:.17g}, not 1.0")


@dataclass
class PPMDByteModel:
    """Small byte-level PPM-D model with exclusion over a 256-symbol alphabet.

    The method-D estimate used at each context assigns symbol weights
    ``2 * count - 1`` and escape weight ``unique`` over denominator
    ``2 * total``.  Symbols already emitted by higher-order contexts are
    excluded from lower-order contexts.  If a context covers every remaining
    byte, escape is suppressed and the context weights are renormalized over
    the remaining alphabet so the final dense distribution is always proper.
    """

    order: int = 4
    context_counts: Dict[bytes, Dict[int, int]] = field(default_factory=dict)
    history: bytearray = field(default_factory=bytearray)

    def __post_init__(self) -> None:
        self.order = int(self.order)
        if self.order < 0:
            raise ValueError("PPM-D order must be non-negative")

    def update(self, byte_value: int) -> None:
        """Update the model after a byte has already been scored."""

        b = _validate_byte_value(byte_value)
        hist = bytes(self.history)
        max_context = min(self.order, len(hist))
        for context_len in range(max_context + 1):
            context = hist[len(hist) - context_len :] if context_len else b""
            counts = self.context_counts.setdefault(context, {})
            counts[b] = counts.get(b, 0) + 1
        self.history.append(b)
        if len(self.history) > self.order:
            del self.history[: len(self.history) - self.order]

    def update_bytes(self, data: bytes) -> None:
        """Update the model sequentially with already-scored bytes."""

        for byte_value in bytes(data):
            self.update(int(byte_value))

    def distribution(self, history: Optional[bytes] = None) -> List[float]:
        """Return ``p(next_byte | history)`` as a normalized 256-way list."""

        raw_history = bytes(self.history if history is None else bytes(history))
        hist = raw_history[-self.order :] if self.order > 0 else b""
        probs = [0.0] * 256
        excluded: Set[int] = set()
        escape_mass = 1.0
        max_context = min(self.order, len(hist))

        for context_len in range(max_context, -1, -1):
            context = hist[len(hist) - context_len :] if context_len else b""
            counts = self.context_counts.get(context)
            if not counts:
                continue

            available = [(int(sym), int(count)) for sym, count in sorted(counts.items()) if sym not in excluded]
            available = [(sym, count) for sym, count in available if count > 0]
            if not available:
                continue

            unique = len(available)
            total = sum(count for _, count in available)
            remaining_before = 256 - len(excluded)
            weights = [(sym, (2 * count) - 1) for sym, count in available]

            if unique >= remaining_before:
                denom = float(sum(weight for _, weight in weights))
                next_escape = 0.0
            else:
                denom = float(2 * total)
                next_escape = unique / denom

            if denom <= 0.0:
                continue

            for sym, weight in weights:
                probs[sym] += escape_mass * (float(weight) / denom)
            escape_mass *= next_escape
            excluded.update(sym for sym, _ in available)

            if escape_mass <= 0.0:
                break

        remaining = 256 - len(excluded)
        if escape_mass > 0.0 and remaining > 0:
            fallback = escape_mass / remaining
            for byte_value in range(256):
                if byte_value not in excluded:
                    probs[byte_value] += fallback

        return dense_byte_distribution(probs, normalize=True)


def score_ppmd_byte_then_update(model: PPMDByteModel, byte_value: int) -> ByteScore:
    """Score one byte from the old PPM-D state, then update the model."""

    b = _validate_byte_value(byte_value)
    dist = model.distribution()
    probability = float(dist[b])
    model.update(b)
    return ByteScore(byte_value=b, probability=probability, logprob=math.log(probability))


def mixture_byte_distribution(
    neural_dist: ByteDistributionInput,
    ppmd_dist: ByteDistributionInput,
    *,
    ppmd_lambda: float,
    target_byte: Optional[int] = None,
) -> MixtureByteDistribution:
    """Mix normalized neural and PPM-D byte distributions.

    ``ppmd_lambda`` is the PPM-D weight: ``0`` means neural-only and ``1``
    means PPM-D-only.  Sparse mappings are accepted for the neural side and are
    densified over all 256 bytes before mixing.
    """

    lam = float(ppmd_lambda)
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"ppmd_lambda must be in [0, 1], got {ppmd_lambda!r}")

    neural = dense_byte_distribution(neural_dist, normalize=True)
    ppmd = dense_byte_distribution(ppmd_dist, normalize=True)
    probs = [((1.0 - lam) * neural[i]) + (lam * ppmd[i]) for i in range(256)]
    probs = dense_byte_distribution(probs, normalize=True)

    if target_byte is None:
        return MixtureByteDistribution(probs=probs)
    b = _validate_byte_value(target_byte)
    target_prob = float(probs[b])
    return MixtureByteDistribution(
        probs=probs,
        target_byte=b,
        target_prob=target_prob,
        target_logprob=math.log(target_prob) if target_prob > 0.0 else -math.inf,
    )


def _logaddexp(a: float, b: float) -> float:
    """Small Python 3.8-compatible two-term logaddexp helper."""

    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))


def _log_mixture_probability(nn_logprob: float, ppm_prob: float, ppmd_lambda: float) -> float:
    """Return log((1-lambda)*p_nn + lambda*p_ppm) without probability underflow."""

    lam = float(ppmd_lambda)
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"ppmd_lambda must be in [0, 1], got {ppmd_lambda!r}")
    if ppm_prob <= 0.0:
        raise ValueError(f"PPM-D probability must be positive, got {ppm_prob!r}")
    if lam == 0.0:
        return float(nn_logprob)
    if lam == 1.0:
        return math.log(float(ppm_prob))
    return _logaddexp(math.log1p(-lam) + float(nn_logprob), math.log(lam) + math.log(float(ppm_prob)))


def ppmd_prefix_confidence(model: PPMDByteModel, history: Optional[bytes] = None) -> float:
    """Return a prefix-only confidence score for the current PPM-D state."""

    dist = model.distribution(history)
    return float(max(dist))


def ppmd_prefix_lambda(
    model: PPMDByteModel,
    target_byte: Optional[int] = None,
    *,
    base_lambda: float = 0.35,
    lambda_hi: float = 0.90,
    lambda_lo: float = 0.05,
    conf_threshold: float = 0.90,
    confidence_gating: bool = True,
    history: Optional[bytes] = None,
) -> float:
    """Choose the PPM-D mixture weight from prefix confidence only.

    ``target_byte`` is accepted for call-site readability but intentionally not
    consulted; the gate must not peek at the target symbol before scoring it.
    """

    if target_byte is not None:
        _validate_byte_value(target_byte)
    for name, value in (
        ("base_lambda", base_lambda),
        ("lambda_hi", lambda_hi),
        ("lambda_lo", lambda_lo),
        ("conf_threshold", conf_threshold),
    ):
        v = float(value)
        if v < 0.0 or v > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    if not confidence_gating:
        return float(base_lambda)
    confidence = ppmd_prefix_confidence(model, history)
    return float(lambda_hi if confidence >= float(conf_threshold) else lambda_lo)


def score_ppmd_stream(
    records: Sequence[Union[ByteLogprobRecord, Mapping[str, Any]]],
    *,
    ppmd_order: int = 5,
    ppmd_lambda: float = 0.35,
    ppmd_lambda_hi: float = 0.90,
    ppmd_lambda_lo: float = 0.05,
    ppmd_conf_threshold: float = 0.90,
    ppmd_confidence_gating: bool = True,
    initial_bytes: bytes = b"",
) -> PPMDStreamScoreSummary:
    """Stream sorted byte records through PPM-D, then update after scoring.

    Returns natural-log NLLs and BPB values for the neural-only, PPM-D-only, and
    mixture scores.  The neural probability for each byte is read from the
    record's precomputed ``neural_logprob``.
    """

    ordered = merge_shard_records([[ _coerce_record(record) for record in records ]])
    if not ordered:
        raise ValueError("cannot score an empty byte-record stream")

    model = PPMDByteModel(order=ppmd_order)
    model.update_bytes(bytes(initial_bytes))

    mix_nll = 0.0
    ppm_nll = 0.0
    nn_nll = 0.0
    lambdas: List[float] = []

    for record in ordered:
        b = _validate_byte_value(record.byte_value)
        ppmd_dist = model.distribution()
        ppm_prob = float(ppmd_dist[b])
        if ppm_prob <= 0.0:
            raise ValueError(f"PPM-D assigned non-positive probability to byte {b}")
        nn_logprob = float(record.neural_logprob)
        lam = ppmd_prefix_lambda(
            model,
            b,
            base_lambda=ppmd_lambda,
            lambda_hi=ppmd_lambda_hi,
            lambda_lo=ppmd_lambda_lo,
            conf_threshold=ppmd_conf_threshold,
            confidence_gating=ppmd_confidence_gating,
        )
        mix_logprob = _log_mixture_probability(nn_logprob, ppm_prob, lam)

        nn_nll -= nn_logprob
        ppm_nll -= math.log(ppm_prob)
        mix_nll -= mix_logprob
        lambdas.append(lam)
        model.update(b)

    byte_count = len(ordered)
    denom = math.log(2.0) * float(byte_count)
    return PPMDStreamScoreSummary(
        byte_count=byte_count,
        mix_nll=float(mix_nll),
        ppm_nll=float(ppm_nll),
        nn_nll=float(nn_nll),
        mix_bpb=float(mix_nll / denom),
        ppm_bpb=float(ppm_nll / denom),
        nn_bpb=float(nn_nll / denom),
        lambdas=lambdas,
        ppmd_history=bytes(model.history),
    )


def trie_stats(root: ByteTrieNode) -> Dict[str, int]:
    """Small introspection helper for tests and future audit reports."""

    nodes = 0
    edges = 0
    terminals = 0
    token_refs = 0
    stack = [root]
    while stack:
        node = stack.pop()
        nodes += 1
        edges += len(node.children)
        terminals += len(node.terminal_token_ids)
        token_refs += len(node.subtree_token_ids)
        stack.extend(node.children.values())
    return {"nodes": nodes, "edges": edges, "terminals": terminals, "subtree_token_refs": token_refs}


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path without requiring it to exist."""

    return Path(path).expanduser().resolve(strict=False)


def _require_existing_file(path: Union[str, Path], *, label: str) -> Path:
    """Return an absolute path or raise a clear file validation error."""

    resolved = _resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    return resolved


def import_exp1876_source_module(
    source_python_path: Union[str, Path] = DEFAULT_SOURCE_PYTHON_PATH,
    *,
    module_name: str = "path_b_exp1876_train_gpt",
) -> Any:
    """Dynamically import the exp_1876 source module for future eval mode.

    This helper intentionally is not used by dry-run paths, because importing
    the source pulls in torch, FlashAttention, SentencePiece, and distributed
    setup code.  Callers must opt into this only from explicit eval workflows.
    """

    source = _require_existing_file(source_python_path, label="source Python path")
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def known_path_b_denominators() -> Dict[str, Union[int, str]]:
    """Return audited Path B denominator constants for exp_1876 planning.

    These are metadata constants only.  They do not imply that this module has
    evaluated validation tokens; heavy scoring remains explicitly disabled.
    """

    return {
        "full_validation_bytes": KNOWN_FULL_VALIDATION_BYTES,
        "first_8m_token_bytes": KNOWN_FIRST_8M_TOKEN_BYTES,
        "first_8m_token_token_count": DEFAULT_SUBSET_TOKENS,
        "description": "Known production target byte denominators for Path B audit planning.",
    }


def byte_denominator_from_token_byte_lengths(
    token_byte_lengths: Iterable[int],
    *,
    token_limit: Optional[int] = None,
) -> int:
    """Compute a small synthetic byte denominator from token byte lengths.

    This helper is deliberately simple and is used only for local regression
    tests or future audits that already have token-to-byte lengths available.
    """

    total = 0
    for index, length in enumerate(token_byte_lengths):
        if token_limit is not None and index >= int(token_limit):
            break
        value = int(length)
        if value < 0:
            raise ValueError(f"token byte length at index {index} is negative: {value}")
        total += value
    return int(total)


def _record_order_key(record: ByteLogprobRecord) -> Tuple[int, int]:
    return record.order_key


def _coerce_record(record: Union[ByteLogprobRecord, Mapping[str, Any]]) -> ByteLogprobRecord:
    if isinstance(record, ByteLogprobRecord):
        return record
    return ByteLogprobRecord.from_json_dict(record)


def merge_shard_records(
    shard_records: Sequence[Sequence[Union[ByteLogprobRecord, Mapping[str, Any]]]]
) -> List[ByteLogprobRecord]:
    """Deterministically merge per-rank byte records by absolute order.

    Each individual shard must already be strictly increasing by
    ``(absolute_token_position, byte_offset_in_token)``.  The final merge sorts
    across shards by that same key and rejects duplicate absolute byte records.
    """

    merged: List[ByteLogprobRecord] = []
    for shard_index, records in enumerate(shard_records):
        previous_key: Optional[Tuple[int, int]] = None
        for raw_record in records:
            record = _coerce_record(raw_record)
            key = _record_order_key(record)
            if previous_key is not None:
                if key == previous_key:
                    raise ValueError(
                        f"duplicate record in shard {shard_index} at absolute token position "
                        f"{key[0]} byte offset {key[1]}"
                    )
                if key < previous_key:
                    raise ValueError(
                        f"out-of-order records in shard {shard_index}: {key} appeared after {previous_key}"
                    )
            previous_key = key
            merged.append(record)

    merged.sort(key=_record_order_key)
    previous_key = None
    for record in merged:
        key = _record_order_key(record)
        if previous_key is not None and key == previous_key:
            raise ValueError(
                f"duplicate record across shards at absolute token position {key[0]} byte offset {key[1]}"
            )
        previous_key = key
    return merged


def write_records_jsonl(path: Union[str, Path], records: Sequence[ByteLogprobRecord]) -> None:
    """Write byte logprob records as JSONL in deterministic order."""

    output = _resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = merge_shard_records([records])
    with output.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True, separators=(",", ":")) + "\n")


def read_records_jsonl(path: Union[str, Path]) -> List[ByteLogprobRecord]:
    """Read byte logprob records from JSONL."""

    input_path = _require_existing_file(path, label="records JSONL path")
    records: List[ByteLogprobRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {input_path}:{line_number}: {exc}") from exc
            records.append(ByteLogprobRecord.from_json_dict(payload))
    return merge_shard_records([records])


def write_records_npz(path: Union[str, Path], records: Sequence[ByteLogprobRecord]) -> None:
    """Write byte logprob records as a compact compressed ``.npz`` shard."""

    np = _optional_numpy()
    if np is None:
        raise ImportError("numpy is required to write NPZ byte logprob shards")
    output = _resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = merge_shard_records([records])
    np.savez_compressed(
        str(output),
        token_position=np.asarray([r.absolute_token_position for r in ordered], dtype=np.int64),
        byte_offset=np.asarray([r.byte_offset_in_token for r in ordered], dtype=np.int64),
        byte_value=np.asarray([r.byte_value for r in ordered], dtype=np.uint8),
        neural_logprob=np.asarray([r.neural_logprob for r in ordered], dtype=np.float64),
    )


def read_records_npz(path: Union[str, Path]) -> List[ByteLogprobRecord]:
    """Read a compact ``.npz`` byte logprob shard."""

    np = _optional_numpy()
    if np is None:
        raise ImportError("numpy is required to read NPZ byte logprob shards")
    input_path = _require_existing_file(path, label="records NPZ path")
    with np.load(str(input_path)) as payload:
        required = ("token_position", "byte_offset", "byte_value", "neural_logprob")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"records NPZ path {input_path} is missing arrays: {', '.join(missing)}")
        token_position = payload["token_position"]
        byte_offset = payload["byte_offset"]
        byte_value = payload["byte_value"]
        neural_logprob = payload["neural_logprob"]
        lengths = {len(token_position), len(byte_offset), len(byte_value), len(neural_logprob)}
        if len(lengths) != 1:
            raise ValueError(f"records NPZ path {input_path} has arrays with inconsistent lengths")
        records = [
            ByteLogprobRecord(
                absolute_token_position=int(token_position[index]),
                byte_offset_in_token=int(byte_offset[index]),
                byte_value=int(byte_value[index]),
                neural_logprob=float(neural_logprob[index]),
            )
            for index in range(len(token_position))
        ]
    return merge_shard_records([records])


def merge_record_npz_shards(paths: Sequence[Union[str, Path]]) -> List[ByteLogprobRecord]:
    """Read and deterministically merge multiple NPZ byte-record shards."""

    return merge_shard_records([read_records_npz(path) for path in paths])


write_binary_shard_records = write_records_npz
read_binary_shard_records = read_records_npz


def future_runpod_command_suggestions(config: PathBEvalConfig) -> List[str]:
    """Return non-secret command suggestions for future manual RunPod eval."""

    output = config.output_json_path or DEFAULT_OUTPUT_JSON_PATH
    return [
        "python3 scripts/eval_path_b_ppmd.py --dry-run "
        f"--source-python {config.source_python_path} --artifact-path {config.artifact_path} "
        f"--output-json {output} --eval-kind {config.eval_kind}",
        "torchrun --standalone --nproc_per_node=8 scripts/eval_path_b_ppmd.py --eval "
        f"--source-python {config.source_python_path} --artifact-path {config.artifact_path} "
        f"--output-json {output} --subset-tokens {config.subset_tokens} --eval-kind {config.eval_kind}",
    ]


def build_output_schema_metadata(
    *,
    source_python_path: Union[str, Path],
    artifact_path: Union[str, Path],
    artifact_size_bytes: Optional[int],
    subset_tokens: int,
    config: PathBEvalConfig,
    mode: str,
    dry_run_imported_source: bool = False,
) -> Dict[str, Any]:
    """Build the JSON audit schema used by dry-run and future eval outputs."""

    source = _resolve_path(source_python_path)
    artifact = _resolve_path(artifact_path)
    size = artifact_size_bytes
    if size is None and artifact.exists() and artifact.is_file():
        size = int(artifact.stat().st_size)

    return {
        "schema_version": SCHEMA_VERSION,
        "path_b_version": PATH_B_VERSION,
        "mode": str(mode),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_python_path": str(source),
        "source_python_exists": bool(source.exists()),
        "artifact_path": str(artifact),
        "artifact_exists": bool(artifact.exists()),
        "artifact_size_bytes": size,
        "subset_tokens": int(subset_tokens),
        "eval_kind": str(config.eval_kind),
        "denominator_constants": known_path_b_denominators(),
        "normalizer_description": (
            "token-trie marginalization with terminal mass exclusion + PPM-D with exclusion; "
            "bits-per-byte denominators must count true emitted target bytes only"
        ),
        "distributed_merge_strategy": (
            "Per-rank ByteLogprobRecord JSONL shards are merged by absolute token position and "
            "byte offset within token; duplicate absolute byte records are rejected and each shard "
            "must be locally sorted before the deterministic global merge."
        ),
        "ppmd": {
            "order": int(config.ppmd_order),
            "method": "PPM-D with exclusion over 256 bytes",
        },
        "mixture": {
            "ppmd_lambda": float(config.ppmd_lambda),
            "ppmd_lambda_hi": float(config.ppmd_lambda_hi),
            "ppmd_lambda_lo": float(config.ppmd_lambda_lo),
            "ppmd_conf_threshold": float(config.ppmd_conf_threshold),
            "ppmd_confidence_gating": bool(config.ppmd_confidence_gating),
        },
        "dry_run_imported_source": bool(dry_run_imported_source),
        "eval_status": "not_run" if mode == "dry_run" else "not_implemented",
        "metrics": {
            "val_bpb": None,
            "total_nll_bits": None,
            "byte_denominator": None,
        },
        "future_runpod_command_suggestions": future_runpod_command_suggestions(config),
        "notes": [
            "Dry-run validates file paths only; it does not import torch-heavy model code.",
            "Real eval must be launched explicitly and must re-audit byte denominators before claiming BPB.",
        ],
    }


def write_output_json(path: Union[str, Path], payload: Mapping[str, Any]) -> Path:
    """Write a JSON payload and return the resolved output path."""

    output = _resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_dry_run(config: PathBEvalConfig) -> Dict[str, Any]:
    """Validate paths and write a JSON plan without importing heavy model code."""

    source = _require_existing_file(config.source_python_path, label="source Python path")
    artifact = _require_existing_file(config.artifact_path, label="artifact path")
    payload = build_output_schema_metadata(
        source_python_path=source,
        artifact_path=artifact,
        artifact_size_bytes=int(artifact.stat().st_size),
        subset_tokens=config.subset_tokens,
        config=config,
        mode="dry_run",
        dry_run_imported_source=False,
    )
    if config.output_json_path is not None:
        write_output_json(config.output_json_path, payload)
    return payload


RANK_SHARD_FILENAME_TEMPLATE = "path_b_sliding_rank{rank}.npz"
RANK_ACCOUNTING_FILENAME_TEMPLATE = "path_b_sliding_accounting_rank{rank}.json"
MERGE_MANIFEST_FILENAME = "path_b_sliding_merge_manifest.json"
ACCOUNTING_AUDIT_FILENAME = "path_b_sliding_accounting_audit.json"
SLIDING_FULL_FILENAME = "path_b_sliding_full.json"
SLIDING_SUBSET_FILENAME_TEMPLATE = "path_b_sliding_subset_{n}.json"
DENOMINATOR_FORMULA = (
    "scored_byte_count = sum_{i in scored_positions} ("
    "base_bytes_lut[target_id_i] + (has_leading_space[target_id_i] & ~is_boundary_token[prev_id_i]))"
)


def plan_sliding_window_starts(total_tokens: int, seq_len: int, stride: int) -> List[int]:
    """Mirror exp_1876 ``eval_val_sliding`` window planning exactly."""

    total = int(total_tokens)
    sl = int(seq_len)
    st = int(stride)
    if total <= 0:
        raise ValueError("total_tokens must be positive")
    if sl <= 0 or st <= 0:
        raise ValueError("seq_len and stride must be positive")
    if st > sl:
        raise ValueError("stride must not exceed seq_len")
    context_size = sl - st
    return [ws for ws in range(0, total, st) if ws + context_size < total]


def slice_window_starts_for_rank(
    window_starts: Sequence[int], *, rank: int, world_size: int
) -> List[int]:
    """Mirror exp_1876 per-rank window slicing (n*r//W : n*(r+1)//W)."""

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} out of range for world_size {world_size}")
    n = len(window_starts)
    s = n * int(rank) // int(world_size)
    e = n * (int(rank) + 1) // int(world_size)
    return list(window_starts[s:e])


def rank_shard_filename(rank: int) -> str:
    if int(rank) < 0:
        raise ValueError("rank must be non-negative")
    return RANK_SHARD_FILENAME_TEMPLATE.format(rank=int(rank))


def rank_accounting_filename(rank: int) -> str:
    if int(rank) < 0:
        raise ValueError("rank must be non-negative")
    return RANK_ACCOUNTING_FILENAME_TEMPLATE.format(rank=int(rank))


def merged_eval_result_filename(*, subset_tokens: Optional[int], full_eval: bool) -> str:
    if full_eval or subset_tokens is None:
        return SLIDING_FULL_FILENAME
    return SLIDING_SUBSET_FILENAME_TEMPLATE.format(n=int(subset_tokens))


def filter_records_by_subset(
    records: Sequence[ByteLogprobRecord], *, subset_tokens: Optional[int]
) -> List[ByteLogprobRecord]:
    """Keep only records with absolute token position < subset_tokens."""

    if subset_tokens is None:
        return list(records)
    cap = int(subset_tokens)
    if cap < 0:
        raise ValueError("subset_tokens must be non-negative or None")
    return [r for r in records if int(r.absolute_token_position) < cap]


def emitted_token_byte_count(
    target_id: int,
    prev_id: int,
    base_bytes_lut: Sequence[int],
    has_leading_space_lut: Sequence[bool],
    is_boundary_token_lut: Sequence[bool],
) -> int:
    """Mirror eval_val_sliding's per-token byte count formula.

    base_bytes_lut[target] + (has_leading_space[target] & ~is_boundary_token[prev]).
    Special/control/unknown/unused tokens have base_bytes=0 and has_leading_space=False
    so they correctly contribute zero bytes.
    """

    tid = int(target_id)
    pid = int(prev_id)
    base = int(base_bytes_lut[tid]) if 0 <= tid < len(base_bytes_lut) else 0
    has_space = bool(has_leading_space_lut[tid]) if 0 <= tid < len(has_leading_space_lut) else False
    if pid < 0:
        prev_is_boundary = True
    elif 0 <= pid < len(is_boundary_token_lut):
        prev_is_boundary = bool(is_boundary_token_lut[pid])
    else:
        prev_is_boundary = True
    return int(base) + (1 if (has_space and not prev_is_boundary) else 0)


def expected_denominator_for_eval(subset_tokens: Optional[int], *, full_eval: bool) -> Optional[int]:
    """Return the audited expected byte denominator for known subset sizes."""

    if full_eval:
        return KNOWN_FULL_VALIDATION_BYTES
    if subset_tokens is not None and int(subset_tokens) == DEFAULT_SUBSET_TOKENS:
        return KNOWN_FIRST_8M_TOKEN_BYTES
    return None


def build_per_rank_accounting(
    *,
    rank: int,
    scored_token_count: int,
    scored_byte_count: int,
    zero_byte_token_count: int,
    min_absolute_token_position: Optional[int],
    max_absolute_token_position: Optional[int],
    shard_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Per-rank accounting JSON payload."""

    return {
        "schema_version": SCHEMA_VERSION,
        "path_b_version": PATH_B_VERSION,
        "rank": int(rank),
        "scored_token_count": int(scored_token_count),
        "scored_byte_count": int(scored_byte_count),
        "zero_byte_token_count": int(zero_byte_token_count),
        "min_absolute_token_position": (
            None if min_absolute_token_position is None else int(min_absolute_token_position)
        ),
        "max_absolute_token_position": (
            None if max_absolute_token_position is None else int(max_absolute_token_position)
        ),
        "shard_path": (None if shard_path is None else str(shard_path)),
    }


def build_merge_manifest(
    shard_entries: Sequence[Mapping[str, Any]], *, world_size: int
) -> Dict[str, Any]:
    """Cross-shard merge manifest used by rank-0 audit."""

    shards: List[Dict[str, Any]] = []
    total_tokens = 0
    total_bytes = 0
    for entry in shard_entries:
        rank = int(entry["rank"])
        scored_tokens = int(entry["scored_tokens"])
        scored_bytes = int(entry["scored_bytes"])
        record: Dict[str, Any] = {
            "rank": rank,
            "scored_tokens": scored_tokens,
            "scored_bytes": scored_bytes,
            "file_path": str(entry["file_path"]),
        }
        if "sha256" in entry and entry["sha256"] is not None:
            record["sha256"] = str(entry["sha256"])
        shards.append(record)
        total_tokens += scored_tokens
        total_bytes += scored_bytes
    shards.sort(key=lambda r: r["rank"])
    return {
        "schema_version": SCHEMA_VERSION,
        "path_b_version": PATH_B_VERSION,
        "world_size": int(world_size),
        "shards": shards,
        "total_scored_tokens": int(total_tokens),
        "total_scored_bytes": int(total_bytes),
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def build_sliding_eval_result(
    *,
    config: PathBEvalConfig,
    source_module_path: Union[str, Path],
    artifact_path: Union[str, Path],
    artifact_size_bytes: Optional[int],
    rank: int,
    world_size: int,
    subset_tokens: Optional[int],
    full_eval: bool,
    scored_token_count: int,
    scored_byte_count: int,
    zero_byte_token_count: int,
    runtime_seconds: float,
    summary: Optional[Any],
    shard_manifest_path: Optional[Union[str, Path]],
    accounting_audit_path: Optional[Union[str, Path]],
    warnings: Sequence[str],
    error: Optional[str],
) -> Dict[str, Any]:
    """Build the canonical sliding-eval result JSON."""

    expected = expected_denominator_for_eval(subset_tokens, full_eval=full_eval)
    if expected is None:
        denominator_match: Optional[bool] = None
    else:
        denominator_match = bool(int(scored_byte_count) == int(expected))
    metric_gate = (
        error is None
        and summary is not None
        and (expected is None or denominator_match is True)
        and int(scored_byte_count) > 0
        and int(scored_token_count) > 0
    )

    def _metric(value: float) -> Optional[float]:
        return float(value) if metric_gate else None

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "path_b_version": PATH_B_VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_module_path": str(source_module_path),
        "artifact_path": str(artifact_path),
        "artifact_size_bytes": (None if artifact_size_bytes is None else int(artifact_size_bytes)),
        "eval_kind": EVAL_KIND_SLIDING,
        "rank": int(rank),
        "world_size": int(world_size),
        "subset_tokens": (None if (full_eval or subset_tokens is None) else int(subset_tokens)),
        "full_eval": bool(full_eval),
        "scored_token_count": int(scored_token_count),
        "scored_byte_count": int(scored_byte_count),
        "expected_denominator": (None if expected is None else int(expected)),
        "denominator_match": denominator_match,
        "denominator_formula": DENOMINATOR_FORMULA,
        "zero_byte_token_count": int(zero_byte_token_count),
        "ppm_d_config": {
            "order": int(config.ppmd_order),
            "method": "PPM-D with exclusion over 256 bytes",
            "score_before_update": True,
        },
        "lambda_gating_config": {
            "ppmd_lambda": float(config.ppmd_lambda),
            "ppmd_lambda_hi": float(config.ppmd_lambda_hi),
            "ppmd_lambda_lo": float(config.ppmd_lambda_lo),
            "ppmd_conf_threshold": float(config.ppmd_conf_threshold),
            "ppmd_confidence_gating": bool(config.ppmd_confidence_gating),
            "prefix_only": True,
        },
        "shard_manifest_path": (None if shard_manifest_path is None else str(shard_manifest_path)),
        "accounting_audit_path": (
            None if accounting_audit_path is None else str(accounting_audit_path)
        ),
        "runtime_seconds": float(runtime_seconds),
        "neural_only_bpb": _metric(getattr(summary, "nn_bpb", 0.0)) if summary is not None else None,
        "ppm_d_only_bpb": _metric(getattr(summary, "ppm_bpb", 0.0)) if summary is not None else None,
        "mixture_bpb": _metric(getattr(summary, "mix_bpb", 0.0)) if summary is not None else None,
        "warnings": list(warnings),
        "claim_ready": bool(metric_gate),
        "error": (None if error is None else str(error)),
    }
    return payload


def execute_sliding_eval(
    config: PathBEvalConfig, *, output_json_path: Optional[Path]
) -> Dict[str, Any]:
    """Real distributed sliding eval. Heavy: requires torch + CUDA + the artifact.

    Mirrors ``eval_val_sliding`` windowing exactly. For each scored position,
    softmaxes logits, gathers target/prev token IDs and the boundary mode flag,
    then calls ``vectorized_target_path_logprobs`` on the precomputed
    boundary/non-boundary tries to extract Path B byte-level neural log-probs.

    Per-rank NPZ shards + accounting JSON are written. Rank 0 merges across
    shards, streams once through ``score_ppmd_stream``, and writes the final
    audited result. If anything fails the caller writes claim_ready=false.
    """

    import time
    import hashlib

    torch = _optional_torch()
    if torch is None:
        raise ImportError("torch is required for the real Path B sliding eval")
    np = _optional_numpy()
    if np is None:
        raise ImportError("numpy is required for the real Path B sliding eval")

    import torch.nn.functional as F  # type: ignore
    import torch.distributed as dist  # type: ignore
    import os

    src_module = import_exp1876_source_module(config.source_python_path)
    artifact = _require_existing_file(config.artifact_path, label="artifact path")
    artifact_size = int(artifact.stat().st_size)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_run = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real sliding eval")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed_run and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    h = src_module.Hyperparameters()
    h.quantized_model_path = str(artifact)
    h.ppm_enabled = False
    h.rank = rank
    h.world_size = world_size
    h.local_rank = local_rank

    started = time.perf_counter()
    val_data = src_module.ValidationData(h, device)
    eval_model = src_module.deserialize(h, device)
    if getattr(h, "num_loops", 0) > 0:
        eval_model.looping_active = True
    eval_model.eval()

    seq_len = int(h.eval_seq_len)
    stride = int(h.eval_stride)
    total_tokens = int(val_data.val_tokens.numel()) - 1
    context_size = seq_len - stride
    window_starts = plan_sliding_window_starts(total_tokens, seq_len, stride)
    my_windows = slice_window_starts_for_rank(window_starts, rank=rank, world_size=world_size)

    # Build SP token byte sequences once. SP-leading-space handling is performed
    # inside token_byte_sequences_from_piece. Special/control/unknown/unused
    # tokens become zero-byte terminals.
    sp = val_data.sp
    vocab_size = int(h.vocab_size)
    sp_vocab = int(sp.vocab_size())
    sequences: List[TokenByteSequences] = []
    for token_id in range(min(sp_vocab, vocab_size)):
        sequences.append(
            token_byte_sequences_from_piece(
                token_id,
                sp.id_to_piece(token_id),
                is_special=False,
                is_control=bool(sp.is_control(token_id)),
                is_unknown=bool(sp.is_unknown(token_id)),
                is_unused=bool(sp.is_unused(token_id)),
            )
        )
    boundary_trie, non_boundary_trie = build_mode_tries(sequences)
    boundary_tables = build_optimized_trie_tables(boundary_trie)
    non_boundary_tables = build_optimized_trie_tables(non_boundary_trie)
    tries_by_mode = {0: boundary_trie, 1: non_boundary_trie}
    tables_by_mode = {0: boundary_tables, 1: non_boundary_tables}
    path_metadata = build_mode_token_path_metadata(sequences, tries_by_mode, tables_by_mode)

    base_bytes_lut_cpu = val_data.base_bytes_lut.detach().cpu().tolist()
    has_leading_space_lut_cpu = val_data.has_leading_space_lut.detach().cpu().tolist()
    is_boundary_token_lut_cpu = val_data.is_boundary_token_lut.detach().cpu().tolist()

    subset_tokens = None if config.full_eval else int(config.subset_tokens)

    rank_records: List[ByteLogprobRecord] = []
    rank_scored_token_count = 0
    rank_scored_byte_count = 0
    rank_zero_byte_token_count = 0
    rank_min_pos: Optional[int] = None
    rank_max_pos: Optional[int] = None
    batch_seqs = int(os.environ.get("PATH_B_BATCH_SEQS", "16"))

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi : bi + batch_seqs]
            if subset_tokens is not None:
                # Short-circuit batches whose first scored position is already
                # past subset_tokens. For each window, the first scored absolute
                # position is ws + (0 if ws == 0 else context_size).
                if all(
                    (ws + (0 if ws == 0 else context_size)) >= subset_tokens
                    for ws in batch_ws
                ):
                    break
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: List[int] = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws : we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = eval_model.forward_logits(x_batch)
            probs_full = torch.softmax(logits.float(), dim=-1)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                if s >= wlen:
                    continue
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                row_probs = probs_full[i, s:wlen, :].to(dtype=torch.float64)
                positions = list(range(ws + s, ws + wlen))
                if subset_tokens is not None:
                    keep_mask = [p < subset_tokens for p in positions]
                    if not any(keep_mask):
                        continue
                    keep_idx = [j for j, k in enumerate(keep_mask) if k]
                    row_probs = row_probs[keep_idx, :]
                    tgt = tgt[keep_idx]
                    prev = prev[keep_idx]
                    positions = [positions[j] for j in keep_idx]
                if positions == []:
                    continue
                target_ids_cpu = tgt.detach().cpu().tolist()
                prev_ids_cpu = prev.detach().cpu().tolist()
                mode_flags = [
                    0 if bool(is_boundary_token_lut_cpu[int(pid)]) else 1
                    for pid in prev_ids_cpu
                ]
                # Update zero-byte token accounting (e.g. SP specials inside the
                # validation stream). They contribute zero records and zero bytes.
                for tid_v, pid_v in zip(target_ids_cpu, prev_ids_cpu):
                    n_b = emitted_token_byte_count(
                        tid_v, pid_v,
                        base_bytes_lut_cpu, has_leading_space_lut_cpu, is_boundary_token_lut_cpu,
                    )
                    rank_scored_token_count += 1
                    rank_scored_byte_count += int(n_b)
                    if n_b == 0:
                        rank_zero_byte_token_count += 1
                if rank_min_pos is None:
                    rank_min_pos = positions[0]
                rank_max_pos = positions[-1]
                # vectorized_target_path_logprobs already skips zero-byte tokens
                # implicitly because their TokenBytePath has empty byte_values.
                batch_records = vectorized_target_path_logprobs(
                    row_probs,
                    target_ids_cpu,
                    mode_flags,
                    tables_by_mode,
                    path_metadata,
                    absolute_token_positions=positions,
                )
                rank_records.extend(batch_records)

    # Sort rank-local records.
    rank_records = merge_shard_records([rank_records])

    output_dir = (output_json_path.parent if output_json_path is not None else Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_shard_path = output_dir / rank_shard_filename(rank)
    rank_acc_path = output_dir / rank_accounting_filename(rank)
    write_records_npz(rank_shard_path, rank_records)
    rank_accounting = build_per_rank_accounting(
        rank=rank,
        scored_token_count=rank_scored_token_count,
        scored_byte_count=rank_scored_byte_count,
        zero_byte_token_count=rank_zero_byte_token_count,
        min_absolute_token_position=rank_min_pos,
        max_absolute_token_position=rank_max_pos,
        shard_path=rank_shard_path,
    )
    write_output_json(rank_acc_path, rank_accounting)

    if distributed_run and dist.is_initialized():
        dist.barrier()

    if rank != 0:
        return {
            "rank": rank,
            "world_size": world_size,
            "shard_path": str(rank_shard_path),
            "accounting_path": str(rank_acc_path),
            "claim_ready": False,
            "note": "non-rank-0 worker; final result emitted on rank 0",
        }

    # Rank 0 merge + audit.
    shard_paths = [output_dir / rank_shard_filename(r) for r in range(world_size)]
    accounting_paths = [output_dir / rank_accounting_filename(r) for r in range(world_size)]
    merged = merge_record_npz_shards(shard_paths)

    shard_entries: List[Dict[str, Any]] = []
    total_scored_tokens = 0
    total_scored_bytes = 0
    total_zero_byte_tokens = 0
    for r, (sp_path, acc_path) in enumerate(zip(shard_paths, accounting_paths)):
        acc = json.loads(Path(acc_path).read_text(encoding="utf-8"))
        sha = hashlib.sha256(Path(sp_path).read_bytes()).hexdigest()
        shard_entries.append({
            "rank": r,
            "scored_tokens": int(acc["scored_token_count"]),
            "scored_bytes": int(acc["scored_byte_count"]),
            "file_path": str(sp_path),
            "sha256": sha,
        })
        total_scored_tokens += int(acc["scored_token_count"])
        total_scored_bytes += int(acc["scored_byte_count"])
        total_zero_byte_tokens += int(acc["zero_byte_token_count"])
    manifest = build_merge_manifest(shard_entries, world_size=world_size)
    manifest_path = output_dir / MERGE_MANIFEST_FILENAME
    write_output_json(manifest_path, manifest)

    accounting_audit = {
        "schema_version": SCHEMA_VERSION,
        "path_b_version": PATH_B_VERSION,
        "world_size": world_size,
        "subset_tokens": subset_tokens,
        "full_eval": bool(config.full_eval),
        "denominator_formula": DENOMINATOR_FORMULA,
        "expected_denominator": expected_denominator_for_eval(subset_tokens, full_eval=config.full_eval),
        "scored_token_count": total_scored_tokens,
        "scored_byte_count": total_scored_bytes,
        "merged_record_count": len(merged),
        "zero_byte_token_count": total_zero_byte_tokens,
    }
    audit_path = output_dir / ACCOUNTING_AUDIT_FILENAME
    write_output_json(audit_path, accounting_audit)

    summary = score_ppmd_stream(
        merged,
        ppmd_order=int(config.ppmd_order),
        ppmd_lambda=float(config.ppmd_lambda),
        ppmd_lambda_hi=float(config.ppmd_lambda_hi),
        ppmd_lambda_lo=float(config.ppmd_lambda_lo),
        ppmd_conf_threshold=float(config.ppmd_conf_threshold),
        ppmd_confidence_gating=bool(config.ppmd_confidence_gating),
    )

    runtime = time.perf_counter() - started
    result = build_sliding_eval_result(
        config=config,
        source_module_path=str(_resolve_path(config.source_python_path)),
        artifact_path=str(artifact),
        artifact_size_bytes=artifact_size,
        rank=rank,
        world_size=world_size,
        subset_tokens=subset_tokens,
        full_eval=bool(config.full_eval),
        scored_token_count=total_scored_tokens,
        scored_byte_count=total_scored_bytes,
        zero_byte_token_count=total_zero_byte_tokens,
        runtime_seconds=runtime,
        summary=summary,
        shard_manifest_path=manifest_path,
        accounting_audit_path=audit_path,
        warnings=[],
        error=None,
    )

    final_filename = merged_eval_result_filename(
        subset_tokens=subset_tokens, full_eval=bool(config.full_eval)
    )
    final_path = (output_json_path if output_json_path is not None else (output_dir / final_filename))
    write_output_json(final_path, result)
    if distributed_run and dist.is_initialized():
        dist.destroy_process_group()
    return result


def run_explicit_eval(
    config: PathBEvalConfig,
    *,
    sliding_executor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Real explicit Path B eval entry point.

    For ``--eval-kind sliding``: runs the distributed byte-level evaluator. On
    failure (executor raises, denominator mismatch, etc.) emits a JSON payload
    with ``claim_ready=false`` and ``mixture_bpb=null`` so no caller can ever
    receive a fake BPB.

    For ``--eval-kind ttt``: still NotImplementedError.
    """

    guard_explicit_eval_kind(config)
    _require_existing_file(config.source_python_path, label="source Python path")
    artifact = _require_existing_file(config.artifact_path, label="artifact path")

    if config.eval_kind != EVAL_KIND_SLIDING:
        # Defensive fallback; guard_explicit_eval_kind already handles ttt.
        raise NotImplementedError(f"eval kind {config.eval_kind!r} is not implemented")

    executor = sliding_executor if sliding_executor is not None else execute_sliding_eval
    output_json_path = (
        None if config.output_json_path is None else _resolve_path(config.output_json_path)
    )

    try:
        result = executor(config, output_json_path=output_json_path)
        if isinstance(result, dict):
            return result
        # Defensive: executor returned something unexpected.
        raise RuntimeError("sliding executor returned a non-dict result")
    except Exception as exc:
        artifact_size = int(artifact.stat().st_size) if artifact.exists() and artifact.is_file() else None
        import os as _os
        rank = int(_os.environ.get("RANK", "0"))
        world_size = int(_os.environ.get("WORLD_SIZE", "1"))
        subset_tokens = None if config.full_eval else int(config.subset_tokens)
        failed = build_sliding_eval_result(
            config=config,
            source_module_path=str(_resolve_path(config.source_python_path)),
            artifact_path=str(artifact),
            artifact_size_bytes=artifact_size,
            rank=rank,
            world_size=world_size,
            subset_tokens=subset_tokens,
            full_eval=bool(config.full_eval),
            scored_token_count=0,
            scored_byte_count=0,
            zero_byte_token_count=0,
            runtime_seconds=0.0,
            summary=None,
            shard_manifest_path=None,
            accounting_audit_path=None,
            warnings=[f"sliding executor raised: {type(exc).__name__}"],
            error=str(exc),
        )
        if output_json_path is not None:
            write_output_json(output_json_path, failed)
        return failed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Plan a Path B PPM-D eval safely by default. Use --eval only for the future "
            "explicit real evaluator; dry-run never imports torch-heavy source code."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate paths and write a JSON plan/schema")
    mode.add_argument("--eval", action="store_true", help="explicit future real eval path; currently not implemented")
    parser.add_argument("--source-python", type=Path, default=DEFAULT_SOURCE_PYTHON_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    parser.add_argument("--subset-tokens", type=int, default=DEFAULT_SUBSET_TOKENS)
    parser.add_argument("--ppmd-order", type=int, default=5)
    parser.add_argument("--ppmd-lambda", type=float, default=0.35)
    parser.add_argument("--ppmd-lambda-hi", type=float, default=0.90)
    parser.add_argument("--ppmd-lambda-lo", type=float, default=0.05)
    parser.add_argument("--ppmd-conf-threshold", type=float, default=0.90)
    parser.add_argument(
        "--eval-kind",
        choices=SUPPORTED_EVAL_KINDS,
        default=EVAL_KIND_SLIDING,
        help="future explicit eval flavor to guard: sliding or ttt",
    )
    parser.add_argument(
        "--disable-ppmd-confidence-gating",
        action="store_true",
        help="record a fixed-lambda future plan instead of confidence-gated mixture metadata",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="evaluate over the full validation set (denominator should match 151,078,222 bytes)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PathBEvalConfig:
    """Create a validated config from parsed CLI arguments."""

    return PathBEvalConfig(
        source_python_path=args.source_python,
        artifact_path=args.artifact_path,
        output_json_path=args.output_json,
        subset_tokens=args.subset_tokens,
        ppmd_order=args.ppmd_order,
        ppmd_lambda=args.ppmd_lambda,
        ppmd_lambda_hi=args.ppmd_lambda_hi,
        ppmd_lambda_lo=args.ppmd_lambda_lo,
        ppmd_conf_threshold=args.ppmd_conf_threshold,
        ppmd_confidence_gating=not bool(args.disable_ppmd_confidence_gating),
        eval_kind=getattr(args, "eval_kind", EVAL_KIND_SLIDING),
        full_eval=bool(getattr(args, "full_eval", False)),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Defaults to safe dry-run behavior."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if args.eval:
        try:
            payload = run_explicit_eval(config)
        except NotImplementedError as exc:
            artifact = _resolve_path(config.artifact_path)
            payload = build_output_schema_metadata(
                source_python_path=config.source_python_path,
                artifact_path=config.artifact_path,
                artifact_size_bytes=int(artifact.stat().st_size) if artifact.exists() and artifact.is_file() else None,
                subset_tokens=config.subset_tokens,
                config=config,
                mode="eval_not_implemented",
                dry_run_imported_source=False,
            )
            payload["error"] = str(exc)
            if config.output_json_path is not None:
                write_output_json(config.output_json_path, payload)
            print(str(exc), file=sys.stderr)
            return 2
        if config.output_json_path is not None:
            print(f"Wrote Path B sliding eval result: {_resolve_path(config.output_json_path)}")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        # Non-rank-0 workers always succeed if they reached here without raising;
        # claim_ready is only set on rank 0 after the merge phase. Exiting non-zero
        # on workers triggers torchrun to SIGTERM rank 0 mid-merge.
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            return 0
        return 0 if payload.get("claim_ready", False) else 3

    payload = run_dry_run(config)
    if config.output_json_path is not None:
        print(f"Wrote Path B dry-run plan: {_resolve_path(config.output_json_path)}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
