"""ConvexTok tokenizer utilities for the ToricGT OAI baseline path.

The tokenizer is deliberately byte-exact.  Candidate tokens are byte strings,
encoding is a min-plus shortest path over byte-boundary DAGs, and LP training
uses a sparse relaxation of the vocabulary-selection/path-routing integer
program on a configured tokenizer-training sample.
"""

from __future__ import annotations

import base64
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
BYTE_OFFSET = 4
BYTE_COUNT = 256
PRICED_OFFSET = BYTE_OFFSET + BYTE_COUNT


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def _text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8", errors="replace")


@dataclass(frozen=True)
class Candidate:
    data: bytes
    count: int
    score: float

    @property
    def length(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class ConvexTokTrainingResult:
    tokenizer: "ConvexTokTokenizer"
    lp_objective: float
    lp_lower_bound_tokens: float
    lp_status: str
    lp_success: bool
    selected_candidate_count: int
    candidate_count: int
    sample_doc_count: int
    sample_byte_count: int
    rounding: str


class ConvexTokTokenizer:
    """Byte-exact rounded ConvexTok tokenizer.

    Tokens 0..3 are special ids.  Tokens 4..259 are literal byte tokens.
    Tokens from 260 upward are priced substring tokens selected by LP rounding.
    """

    tokenizer_type = "convextok"

    def __init__(
        self,
        *,
        vocab_size: int,
        priced_tokens: list[bytes],
        lp_scores: list[float] | None = None,
        rounding: str = "det",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if vocab_size < PRICED_OFFSET:
            raise ValueError(f"vocab_size must be at least {PRICED_OFFSET}, got {vocab_size}")
        max_priced = vocab_size - PRICED_OFFSET
        if len(priced_tokens) > max_priced:
            raise ValueError(f"too many priced tokens: {len(priced_tokens)} > {max_priced}")
        if any(len(tok) < 2 for tok in priced_tokens):
            raise ValueError("priced ConvexTok tokens must have byte length >= 2")
        if len(set(priced_tokens)) != len(priced_tokens):
            raise ValueError("duplicate priced tokens")
        self.vocab_size = int(vocab_size)
        self.rounding = str(rounding)
        self.metadata = dict(metadata or {})
        self._token_bytes: list[bytes] = [b"" for _ in range(self.vocab_size)]
        for byte_value in range(BYTE_COUNT):
            self._token_bytes[BYTE_OFFSET + byte_value] = bytes([byte_value])
        for idx, tok in enumerate(priced_tokens):
            self._token_bytes[PRICED_OFFSET + idx] = bytes(tok)
        self._priced_tokens = list(priced_tokens)
        scores = list(lp_scores or [0.0] * len(priced_tokens))
        if len(scores) != len(priced_tokens):
            raise ValueError("lp_scores length must match priced_tokens length")
        self._lp_scores = [float(x) for x in scores]
        self._token_to_id = {tok: PRICED_OFFSET + i for i, tok in enumerate(priced_tokens)}
        for byte_value in range(BYTE_COUNT):
            self._token_to_id[bytes([byte_value])] = BYTE_OFFSET + byte_value
        self._trie = self._build_trie()

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def token_bytes(self, token_id: int) -> bytes:
        if not (0 <= int(token_id) < self.vocab_size):
            raise ValueError(f"token id outside vocab: {token_id}")
        return self._token_bytes[int(token_id)]

    def token_byte_lengths(self) -> np.ndarray:
        return np.asarray([len(tok) for tok in self._token_bytes], dtype=np.int16)

    def token_lp_scores(self) -> np.ndarray:
        scores = np.zeros((self.vocab_size,), dtype=np.float32)
        for i, value in enumerate(self._lp_scores):
            scores[PRICED_OFFSET + i] = float(value)
        return scores

    def token_rank_scores(self) -> np.ndarray:
        ranks = np.zeros((self.vocab_size,), dtype=np.float32)
        n = max(len(self._priced_tokens), 1)
        for i in range(len(self._priced_tokens)):
            ranks[PRICED_OFFSET + i] = 1.0 - float(i) / float(n)
        return ranks

    def token_is_priced(self) -> np.ndarray:
        flags = np.zeros((self.vocab_size,), dtype=np.float32)
        flags[PRICED_OFFSET : PRICED_OFFSET + len(self._priced_tokens)] = 1.0
        return flags

    def priced_tokens(self) -> list[bytes]:
        return list(self._priced_tokens)

    def _build_trie(self) -> dict[int, Any]:
        root: dict[int, Any] = {}
        for token_id in range(BYTE_OFFSET, PRICED_OFFSET + len(self._priced_tokens)):
            data = self._token_bytes[token_id]
            node = root
            for byte in data:
                node = node.setdefault(byte, {})
            node.setdefault("_ids", []).append(token_id)
        return root

    def _candidate_edges_at(self, data: bytes, start: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        node = self._trie
        pos = start
        while pos < len(data):
            nxt = node.get(data[pos])
            if nxt is None:
                break
            node = nxt
            pos += 1
            for token_id in node.get("_ids", []):
                out.append((pos, int(token_id)))
        return out

    def encode_bytes_with_trace(self, data: bytes) -> tuple[list[int], dict[str, Any]]:
        n = len(data)
        inf = 10**12
        dp = [inf] * (n + 1)
        prev: list[tuple[int, int] | None] = [None] * (n + 1)
        margins = [0.0] * (n + 1)
        entropy = [0.0] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if dp[i] >= inf:
                continue
            candidates: list[tuple[int, int, int]] = []
            for end, token_id in self._candidate_edges_at(data, i):
                candidates.append((dp[i] + 1, end, token_id))
            for cost, end, token_id in candidates:
                current = dp[end]
                better = cost < current
                if cost == current and prev[end] is not None:
                    # Prefer longer tokens, then lower ids for deterministic paths.
                    old_start, old_id = prev[end]
                    better = (end - i, -token_id) > (end - old_start, -old_id)
                if better:
                    dp[end] = cost
                    prev[end] = (i, token_id)
            for end in {edge[1] for edge in candidates}:
                local_costs = [float(cost) for cost, e, _ in candidates if e == end]
                if len(local_costs) >= 2:
                    local_costs.sort()
                    margins[end] = float(local_costs[1] - local_costs[0])
                elif local_costs:
                    margins[end] = float("inf")
                if local_costs:
                    arr = np.asarray(local_costs, dtype=np.float64)
                    arr = np.exp(-(arr - arr.min()))
                    probs = arr / max(float(arr.sum()), 1e-12)
                    entropy[end] = float(-(probs * np.log(probs + 1e-12)).sum())
        if prev[n] is None and n > 0:
            raise RuntimeError("ConvexTok DP failed despite byte fallback tokens")
        ids: list[int] = []
        selected_edges: list[dict[str, Any]] = []
        pos = n
        while pos > 0:
            item = prev[pos]
            if item is None:
                raise RuntimeError("broken ConvexTok predecessor chain")
            start, token_id = item
            ids.append(token_id)
            selected_edges.append(
                {
                    "source": int(start),
                    "target": int(pos),
                    "token_id": int(token_id),
                    "byte_length": int(pos - start),
                    "lp_score": float(self.token_lp_scores()[token_id]),
                    "priced": bool(token_id >= PRICED_OFFSET),
                }
            )
            pos = start
        ids.reverse()
        selected_edges.reverse()
        finite_margins = [x for x in margins if math.isfinite(x)]
        trace = {
            "byte_length": int(n),
            "token_count": int(len(ids)),
            "selected_edges": selected_edges,
            "mean_tropical_margin": float(np.mean(finite_margins)) if finite_margins else 0.0,
            "min_tropical_margin": float(np.min(finite_margins)) if finite_margins else 0.0,
            "active_path_entropy": float(np.mean([x for x in entropy if x > 0.0])) if any(x > 0.0 for x in entropy) else 0.0,
            "priced_token_fraction": float(sum(1 for x in ids if x >= PRICED_OFFSET) / max(len(ids), 1)),
        }
        return ids, trace

    def encode_bytes(self, data: bytes) -> list[int]:
        n = len(data)
        inf = 10**12
        dp = [inf] * (n + 1)
        prev: list[tuple[int, int] | None] = [None] * (n + 1)
        dp[0] = 0
        for i in range(n):
            base = dp[i]
            if base >= inf:
                continue
            cost = base + 1
            for end, token_id in self._candidate_edges_at(data, i):
                current = dp[end]
                better = cost < current
                if cost == current and prev[end] is not None:
                    old_start, old_id = prev[end]
                    better = (end - i, -token_id) > (end - old_start, -old_id)
                if better:
                    dp[end] = cost
                    prev[end] = (i, token_id)
        if prev[n] is None and n > 0:
            raise RuntimeError("ConvexTok DP failed despite byte fallback tokens")
        ids: list[int] = []
        pos = n
        while pos > 0:
            item = prev[pos]
            if item is None:
                raise RuntimeError("broken ConvexTok predecessor chain")
            start, token_id = item
            ids.append(token_id)
            pos = start
        ids.reverse()
        return ids

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(_text_to_bytes(text))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_ids(self, ids: Iterable[int]) -> bytes:
        chunks: list[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in {PAD_ID, BOS_ID, EOS_ID}:
                continue
            if token_id == UNK_ID:
                chunks.append(b"\xef\xbf\xbd")
            else:
                chunks.append(self.token_bytes(token_id))
        return b"".join(chunks)

    def decode(self, ids: Iterable[int]) -> str:
        return self.decode_ids(ids).decode("utf-8", errors="replace")

    def to_json_dict(self) -> dict[str, Any]:
        priced = []
        for i, data in enumerate(self._priced_tokens):
            priced.append(
                {
                    "id": PRICED_OFFSET + i,
                    "bytes_b64": _b64(data),
                    "byte_length": len(data),
                    "lp_score": float(self._lp_scores[i]),
                    "rank_score": float(1.0 - i / max(len(self._priced_tokens), 1)),
                }
            )
        return {
            "tokenizer_type": self.tokenizer_type,
            "format_version": 1,
            "vocab_size": int(self.vocab_size),
            "special_tokens": list(SPECIAL_TOKENS),
            "byte_offset": BYTE_OFFSET,
            "byte_count": BYTE_COUNT,
            "priced_offset": PRICED_OFFSET,
            "rounding": self.rounding,
            "priced_tokens": priced,
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ConvexTokTokenizer":
        if payload.get("tokenizer_type") != "convextok":
            raise ValueError("not a ConvexTok tokenizer JSON")
        entries = sorted(payload.get("priced_tokens", []), key=lambda item: int(item["id"]))
        priced_tokens = [_unb64(str(item["bytes_b64"])) for item in entries]
        lp_scores = [float(item.get("lp_score", 0.0)) for item in entries]
        return cls(
            vocab_size=int(payload["vocab_size"]),
            priced_tokens=priced_tokens,
            lp_scores=lp_scores,
            rounding=str(payload.get("rounding", "det")),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ConvexTokTokenizer":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def collect_candidates(
    texts: Iterable[str],
    *,
    min_len: int = 2,
    max_len: int = 24,
    max_candidates: int = 2000,
    max_docs: int | None = None,
    max_doc_bytes: int = 4096,
) -> tuple[list[Candidate], list[bytes]]:
    counts: dict[bytes, int] = {}
    docs: list[bytes] = []
    for doc_idx, text in enumerate(texts):
        if max_docs is not None and doc_idx >= max_docs:
            break
        data = _text_to_bytes(text)[:max_doc_bytes]
        if not data:
            continue
        docs.append(data)
        n = len(data)
        for start in range(n):
            upper = min(n, start + max_len)
            for end in range(start + min_len, upper + 1):
                piece = data[start:end]
                counts[piece] = counts.get(piece, 0) + 1
    candidates: list[Candidate] = []
    for data, count in counts.items():
        saving = float(count * (len(data) - 1))
        if saving <= 0:
            continue
        candidates.append(Candidate(data=data, count=int(count), score=saving))
    candidates.sort(key=lambda c: (-c.score, -c.length, c.data))
    return candidates[:max_candidates], docs


def _candidate_occurrences(docs: list[bytes], candidates: list[Candidate]) -> list[list[tuple[int, int, int]]]:
    by_first: dict[int, list[tuple[int, bytes]]] = {}
    for cid, cand in enumerate(candidates):
        by_first.setdefault(cand.data[0], []).append((cid, cand.data))
    occurrences: list[list[tuple[int, int, int]]] = []
    for data in docs:
        doc_occ: list[tuple[int, int, int]] = []
        n = len(data)
        for start in range(n):
            for cid, piece in by_first.get(data[start], []):
                end = start + len(piece)
                if end <= n and data[start:end] == piece:
                    doc_occ.append((start, end, cid))
        occurrences.append(doc_occ)
    return occurrences


def solve_convextok_lp(
    docs: list[bytes],
    candidates: list[Candidate],
    *,
    budget: int,
) -> tuple[np.ndarray, float, str, bool]:
    if budget <= 0 or not candidates:
        return np.zeros((len(candidates),), dtype=np.float64), float(sum(len(doc) for doc in docs)), "empty", True
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
    except Exception as exc:  # pragma: no cover - exercised in environment failures.
        raise RuntimeError("SciPy with HiGHS support is required for exact ConvexTok LP training") from exc

    occurrences = _candidate_occurrences(docs, candidates)
    edge_doc: list[int] = []
    edge_start: list[int] = []
    edge_end: list[int] = []
    edge_cid: list[int] = []
    for doc_id, data in enumerate(docs):
        for pos in range(len(data)):
            edge_doc.append(doc_id)
            edge_start.append(pos)
            edge_end.append(pos + 1)
            edge_cid.append(-1)
        for start, end, cid in occurrences[doc_id]:
            edge_doc.append(doc_id)
            edge_start.append(start)
            edge_end.append(end)
            edge_cid.append(cid)

    edge_count = len(edge_doc)
    cand_count = len(candidates)
    total_vars = edge_count + cand_count
    c_obj = np.zeros((total_vars,), dtype=np.float64)
    c_obj[:edge_count] = 1.0

    vertex_offsets: list[int] = []
    total_vertices = 0
    for data in docs:
        vertex_offsets.append(total_vertices)
        total_vertices += len(data) + 1
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b_eq = np.zeros((total_vertices,), dtype=np.float64)
    for doc_id, data in enumerate(docs):
        base = vertex_offsets[doc_id]
        b_eq[base] = -1.0
        b_eq[base + len(data)] = 1.0
    for eidx, (doc_id, start, end) in enumerate(zip(edge_doc, edge_start, edge_end, strict=True)):
        base = vertex_offsets[doc_id]
        rows.extend([base + start, base + end])
        cols.extend([eidx, eidx])
        vals.extend([-1.0, 1.0])
    a_eq = coo_matrix((vals, (rows, cols)), shape=(total_vertices, total_vars)).tocsr()

    ineq_rows: list[int] = []
    ineq_cols: list[int] = []
    ineq_vals: list[float] = []
    b_ub: list[float] = []
    row = 0
    for eidx, cid in enumerate(edge_cid):
        if cid < 0:
            continue
        ineq_rows.extend([row, row])
        ineq_cols.extend([eidx, edge_count + cid])
        ineq_vals.extend([1.0, -1.0])
        b_ub.append(0.0)
        row += 1
    for cid in range(cand_count):
        ineq_rows.append(row)
        ineq_cols.append(edge_count + cid)
        ineq_vals.append(1.0)
    b_ub.append(float(budget))
    a_ub = coo_matrix((ineq_vals, (ineq_rows, ineq_cols)), shape=(row + 1, total_vars)).tocsr()

    result = linprog(
        c_obj,
        A_ub=a_ub,
        b_ub=np.asarray(b_ub, dtype=np.float64),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
    )
    scores = np.zeros((cand_count,), dtype=np.float64)
    if result.x is not None:
        scores = np.asarray(result.x[edge_count : edge_count + cand_count], dtype=np.float64)
    return scores, float(result.fun) if result.fun is not None else float("nan"), str(result.message), bool(result.success)


def rounded_candidate_indices(
    candidates: list[Candidate],
    lp_scores: np.ndarray,
    *,
    budget: int,
    rounding: str,
) -> list[int]:
    rounding = rounding.lower()
    if budget <= 0:
        return []
    if rounding == "int":
        selected = [i for i, score in enumerate(lp_scores) if float(score) >= 1.0 - 1e-8]
        if len(selected) >= budget:
            return selected[:budget]
        remaining = [i for i in range(len(candidates)) if i not in set(selected)]
        remaining.sort(key=lambda i: (-float(lp_scores[i]), -candidates[i].score, -candidates[i].length, candidates[i].data))
        return (selected + remaining)[:budget]
    if rounding == "bias":
        key = lambda i: (-float(lp_scores[i]) / max(float(candidates[i].length), 1.0), -float(lp_scores[i]), candidates[i].data)
    elif rounding == "det":
        key = lambda i: (-float(lp_scores[i]), -candidates[i].score, -candidates[i].length, candidates[i].data)
    else:
        raise ValueError("rounding must be one of: det, bias, int")
    order = list(range(len(candidates)))
    order.sort(key=key)
    return order[: min(budget, len(order))]


def train_convextok(
    texts: Iterable[str],
    *,
    vocab_size: int,
    rounding: str = "det",
    min_len: int = 2,
    max_len: int = 24,
    max_candidates: int = 2000,
    max_docs: int | None = 2048,
    max_doc_bytes: int = 4096,
) -> ConvexTokTrainingResult:
    budget = int(vocab_size) - PRICED_OFFSET
    if budget <= 0:
        raise ValueError(f"vocab_size={vocab_size} leaves no priced-token budget after byte fallback")
    candidates, docs = collect_candidates(
        texts,
        min_len=min_len,
        max_len=max_len,
        max_candidates=max_candidates,
        max_docs=max_docs,
        max_doc_bytes=max_doc_bytes,
    )
    if not docs:
        raise ValueError("no nonempty tokenizer-training documents")
    lp_scores, objective, status, success = solve_convextok_lp(docs, candidates, budget=budget)
    selected_indices = rounded_candidate_indices(candidates, lp_scores, budget=budget, rounding=rounding)
    priced = [candidates[i].data for i in selected_indices]
    selected_scores = [float(lp_scores[i]) for i in selected_indices]
    metadata = {
        "training_algorithm": "convextok_sparse_lp",
        "rounding": rounding,
        "lp_status": status,
        "lp_success": success,
        "lp_objective": objective,
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected_indices),
        "sample_doc_count": len(docs),
        "sample_byte_count": int(sum(len(doc) for doc in docs)),
        "max_len": int(max_len),
        "max_candidates": int(max_candidates),
    }
    tok = ConvexTokTokenizer(
        vocab_size=vocab_size,
        priced_tokens=priced,
        lp_scores=selected_scores,
        rounding=rounding,
        metadata=metadata,
    )
    return ConvexTokTrainingResult(
        tokenizer=tok,
        lp_objective=objective,
        lp_lower_bound_tokens=objective,
        lp_status=status,
        lp_success=success,
        selected_candidate_count=len(selected_indices),
        candidate_count=len(candidates),
        sample_doc_count=len(docs),
        sample_byte_count=int(sum(len(doc) for doc in docs)),
        rounding=rounding,
    )


def tokenisation_dag_payload(
    tokenizer: ConvexTokTokenizer,
    text: str,
    *,
    max_bytes: int = 512,
) -> dict[str, Any]:
    data = _text_to_bytes(text)[:max_bytes]
    ids, trace = tokenizer.encode_bytes_with_trace(data)
    selected = {(edge["source"], edge["target"], edge["token_id"]) for edge in trace["selected_edges"]}
    nodes = [{"id": f"b{i}", "type": "byte_boundary", "byte_position": i} for i in range(len(data) + 1)]
    edges: list[dict[str, Any]] = []
    for i, byte_value in enumerate(data):
        token_id = BYTE_OFFSET + int(byte_value)
        edges.append(
            {
                "source": f"b{i}",
                "target": f"b{i+1}",
                "type": "free_byte_edge",
                "token_id": token_id,
                "byte_length": 1,
                "selected": (i, i + 1, token_id) in selected,
                "lp_score": 0.0,
            }
        )
    for start in range(len(data)):
        for end, token_id in tokenizer._candidate_edges_at(data, start):
            if token_id < PRICED_OFFSET:
                continue
            edges.append(
                {
                    "source": f"b{start}",
                    "target": f"b{end}",
                    "type": "priced_candidate_token_edge",
                    "token_id": int(token_id),
                    "byte_length": int(end - start),
                    "selected": (start, end, token_id) in selected,
                    "lp_score": float(tokenizer.token_lp_scores()[token_id]),
                    "token_preview": tokenizer.token_bytes(token_id).decode("utf-8", errors="replace"),
                }
            )
    return {
        "schema": "toricgt.convextok_tokenisation_dag.v1",
        "byte_length": len(data),
        "encoded_token_ids": ids,
        "trace": trace,
        "nodes": nodes,
        "edges": edges,
    }


def analyze_tokenizer_regret(
    texts: list[str],
    tokenizer: ConvexTokTokenizer,
    *,
    lp_vocab_size: int | None = None,
    lp_rounding: str = "det",
    max_candidates: int = 1500,
    max_doc_bytes: int = 2048,
) -> dict[str, float]:
    if not texts:
        raise ValueError("texts must be nonempty")
    lp_result = train_convextok(
        texts,
        vocab_size=int(lp_vocab_size or tokenizer.vocab_size),
        rounding=lp_rounding,
        max_candidates=max_candidates,
        max_docs=len(texts),
        max_doc_bytes=max_doc_bytes,
    )
    total_bytes = 0
    conv_tokens = 0
    byte_tokens = 0
    margins: list[float] = []
    entropies: list[float] = []
    priced_fracs: list[float] = []
    used: set[int] = set()
    for text in texts:
        data = _text_to_bytes(text)[:max_doc_bytes]
        ids, trace = tokenizer.encode_bytes_with_trace(data)
        total_bytes += len(data)
        conv_tokens += len(ids)
        byte_tokens += len(data)
        used.update(ids)
        margins.append(float(trace["mean_tropical_margin"]))
        entropies.append(float(trace["active_path_entropy"]))
        priced_fracs.append(float(trace["priced_token_fraction"]))
    lower = max(float(lp_result.lp_lower_bound_tokens), 1e-9)
    return {
        "tokenizer_regret/sample_docs": float(len(texts)),
        "tokenizer_regret/sample_bytes": float(total_bytes),
        "tokenizer_regret/byte_path_tokens": float(byte_tokens),
        "tokenizer_regret/convextok_path_tokens": float(conv_tokens),
        "tokenizer_regret/lp_lower_bound_tokens": float(lower),
        "tokenizer_regret/convextok_gap_ratio": float(conv_tokens / lower),
        "tokenizer_regret/byte_gap_ratio": float(byte_tokens / lower),
        "tokenizer_regret/tokens_per_byte": float(conv_tokens / max(total_bytes, 1)),
        "tokenizer_regret/vocab_utilization": float(len(used) / max(tokenizer.vocab_size, 1)),
        "tokenizer_tropical/mean_margin": float(np.mean(margins)) if margins else 0.0,
        "tokenizer_tropical/active_path_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "tokenizer_tropical/priced_token_fraction": float(np.mean(priced_fracs)) if priced_fracs else 0.0,
        "tokenizer_toric/priced_vocab_fraction": float(len(tokenizer._priced_tokens) / max(tokenizer.vocab_size, 1)),
        "tokenizer_toric/lp_face_mass": float(np.sum(tokenizer.token_lp_scores())),
        "tokenizer_toric/lp_face_entropy": float(_entropy(tokenizer.token_lp_scores())),
        "tokenizer_toric/mean_token_byte_length": float(np.mean(tokenizer.token_byte_lengths())),
    }


def _entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())
