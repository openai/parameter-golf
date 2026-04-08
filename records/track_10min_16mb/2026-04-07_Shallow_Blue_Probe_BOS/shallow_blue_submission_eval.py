from __future__ import annotations

import glob
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist

from shallow_blue_probe_runtime import (
    ProbePrefixState,
    build_live_probe_feature_vector,
    load_probe_runtime_artifact,
)


@dataclass(frozen=True)
class SparsePrediction:
    ids: np.ndarray
    probs: np.ndarray
    support: int
    entropy: float
    source: str
    metadata: dict[str, float | int | str] | None = None


@dataclass(frozen=True)
class ShallowBlueEvalSummary:
    docs: int
    scored_positions: int
    total_bytes: int
    baseline_bpb: float
    ngram_emitted: int
    repeat_emitted: int
    safe_delta_bpb: float
    safe_mixed_bpb: float
    safe_bits_saved: float
    probe_delta_bpb: float
    probe_mixed_bpb: float
    probe_bits_saved: float
    probe_mean_alpha: float
    probe_alpha_rows: int
    probe_boosted_rows: int
    elapsed_seconds: float


class NgramExpert:
    def __init__(
        self,
        min_order: int = 3,
        max_order: int = 3,
        top_k: int = 3,
        min_support: int = 2,
    ) -> None:
        self.min_order = int(min_order)
        self.max_order = int(max_order)
        self.top_k = int(top_k)
        self.min_support = int(min_support)
        self.reset()

    def reset(self) -> None:
        self.history: deque[int] = deque(maxlen=self.max_order - 1)
        self.counts: dict[int, defaultdict[tuple[int, ...], Counter[int]]] = {
            order: defaultdict(Counter)
            for order in range(self.min_order, self.max_order + 1)
        }

    def update(self, token_id: int) -> None:
        tok = int(token_id)
        hist = tuple(self.history)
        for order in range(self.min_order, self.max_order + 1):
            ctx_len = order - 1
            if len(hist) < ctx_len:
                continue
            ctx = hist[-ctx_len:]
            self.counts[order][ctx][tok] += 1
        self.history.append(tok)

    def predict(self) -> SparsePrediction | None:
        hist = tuple(self.history)
        for order in range(self.max_order, self.min_order - 1, -1):
            ctx_len = order - 1
            if len(hist) < ctx_len:
                continue
            ctx = hist[-ctx_len:]
            cnts = self.counts[order].get(ctx)
            if not cnts:
                continue
            support = sum(cnts.values())
            if support < self.min_support:
                continue
            top = cnts.most_common(self.top_k)
            ids = np.array([tid for tid, _ in top], dtype=np.int32)
            probs = np.array([count / support for _, count in top], dtype=np.float64)
            entropy = 0.0
            for count in cnts.values():
                if count <= 0:
                    continue
                prob = count / support
                entropy -= prob * math.log2(prob)
            return SparsePrediction(
                ids=ids,
                probs=probs,
                support=support,
                entropy=entropy,
                source=f"NGRAM_{order}",
                metadata={
                    "prediction_token_id": int(ids[0]),
                    "prediction_prob": float(probs[0]),
                    "top2_prediction_prob": float(probs[1]) if len(probs) > 1 else 0.0,
                    "unique_continuations": int(len(cnts)),
                },
            )
        return None


class ExactRepeatExpert:
    def __init__(
        self,
        min_match_len: int = 4,
        max_match_len: int = 8,
        top_k: int = 3,
        min_support: int = 2,
    ) -> None:
        self.min_match_len = int(min_match_len)
        self.max_match_len = int(max_match_len)
        self.top_k = int(top_k)
        self.min_support = int(min_support)
        self.reset()

    def reset(self) -> None:
        self.history: deque[int] = deque(maxlen=self.max_match_len)
        self.counts: dict[int, defaultdict[tuple[int, ...], Counter[int]]] = {
            match_len: defaultdict(Counter)
            for match_len in range(self.min_match_len, self.max_match_len + 1)
        }

    def update(self, token_id: int) -> None:
        tok = int(token_id)
        hist = tuple(self.history)
        for match_len in range(self.min_match_len, self.max_match_len + 1):
            if len(hist) < match_len:
                continue
            ctx = hist[-match_len:]
            self.counts[match_len][ctx][tok] += 1
        self.history.append(tok)

    def predict(self) -> SparsePrediction | None:
        hist = tuple(self.history)
        for match_len in range(self.max_match_len, self.min_match_len - 1, -1):
            if len(hist) < match_len:
                continue
            ctx = hist[-match_len:]
            cnts = self.counts[match_len].get(ctx)
            if not cnts:
                continue
            support = sum(cnts.values())
            if support < self.min_support:
                continue
            top = cnts.most_common(self.top_k)
            ids = np.array([tid for tid, _ in top], dtype=np.int32)
            probs = np.array([count / support for _, count in top], dtype=np.float64)
            entropy = 0.0
            for count in cnts.values():
                if count <= 0:
                    continue
                prob = count / support
                entropy -= prob * math.log2(prob)
            return SparsePrediction(
                ids=ids,
                probs=probs,
                support=support,
                entropy=entropy,
                source=f"REPEAT_{match_len}",
                metadata={
                    "prediction_token_id": int(ids[0]),
                    "prediction_prob": float(probs[0]),
                    "match_len": int(match_len),
                },
            )
        return None


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


def load_validation_tokens(path_glob: str) -> np.ndarray:
    files = [Path(path) for path in sorted(glob.glob(path_glob))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {path_glob}")
    tokens = np.concatenate([load_data_shard(path) for path in files]).astype(
        np.int32,
        copy=False,
    )
    if tokens.size < 2:
        raise ValueError("Validation split is too short to score")
    return tokens


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_size = max(int(sp.vocab_size()), int(vocab_size))
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(int(sp.vocab_size())):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


def find_bos_documents(tokens: np.ndarray, bos_id: int) -> list[tuple[int, int]]:
    bos_positions = np.flatnonzero(tokens == int(bos_id))
    if bos_positions.size == 0:
        raise ValueError(f"no BOS documents found for bos_id={bos_id}")
    docs: list[tuple[int, int]] = []
    for idx, start in enumerate(bos_positions.tolist()):
        end = int(bos_positions[idx + 1]) if idx + 1 < bos_positions.size else int(tokens.size)
        length = end - int(start)
        if length >= 2:
            docs.append((int(start), length))
    if not docs:
        raise ValueError("no scoreable BOS documents found")
    return docs


def truncate_docs(
    docs: list[tuple[int, int]],
    *,
    max_docs: int,
    max_val_tokens: int,
) -> list[tuple[int, int]]:
    selected = docs
    if max_val_tokens > 0:
        selected = [
            (start, length)
            for start, length in selected
            if (start + length) <= max_val_tokens
        ]
    if max_docs > 0:
        selected = selected[:max_docs]
    if not selected:
        raise ValueError("document selection is empty after applying limits")
    return selected


def rank_slice(items: list[tuple[int, int]], rank: int, world_size: int) -> list[tuple[int, int]]:
    start = (len(items) * rank) // world_size
    end = (len(items) * (rank + 1)) // world_size
    return items[start:end]


def iter_eval_blocks(num_tokens: int, window: int, stride: int) -> list[tuple[int, int, int, int]]:
    if num_tokens < 2:
        return []
    blocks: list[tuple[int, int, int, int]] = []
    max_transition = num_tokens - 1
    score_start = 0
    score_end = min(window, max_transition)
    while score_start < max_transition:
        context_end = score_end
        context_start = max(0, context_end - window)
        blocks.append((context_start, context_end, score_start, score_end))
        score_start = score_end
        score_end = min(score_end + stride, max_transition)
    return blocks


def collect_block_batch(
    blocks: list[tuple[int, int, int, int]],
    doc_tokens: np.ndarray,
    start_idx: int,
    batch_windows: int,
) -> tuple[list[int], np.ndarray | None]:
    block_indices: list[int] = []
    token_views: list[np.ndarray] = []
    target_len: int | None = None
    for block_idx in range(start_idx, len(blocks)):
        context_start, context_end, _, _ = blocks[block_idx]
        context_len = context_end - context_start
        if target_len is None:
            target_len = context_len
        elif context_len != target_len:
            break
        block_indices.append(block_idx)
        token_views.append(doc_tokens[context_start:context_end])
        if len(block_indices) >= batch_windows:
            break
    if not block_indices:
        return [], None
    return block_indices, np.ascontiguousarray(np.stack(token_views, axis=0), dtype=np.int32)


def sparse_mixed_token_prob_from_row(
    lp_row: np.ndarray,
    next_tid: int,
    ids: np.ndarray,
    probs: np.ndarray,
    alpha: float,
) -> float:
    p_next = float(np.exp(float(lp_row[int(next_tid)])))
    if alpha <= 0.0:
        return max(p_next, 1e-15)
    qv = np.asarray(probs, dtype=np.float64)
    ids_np = np.asarray(ids, dtype=np.int32)
    mask = qv > 0.0
    if not np.any(mask):
        return max(p_next, 1e-15)
    ids_sel = ids_np[mask]
    qv_sel = qv[mask]
    p_sel = np.exp(lp_row[ids_sel].astype(np.float64, copy=False))
    taken = float(np.sum(p_sel))
    resid = max(1e-10, 1.0 - float(np.sum(qv_sel)))
    tail = max(1e-10, 1.0 - taken)
    matches = np.flatnonzero(ids_sel == int(next_tid))
    if matches.size:
        q_next = float(qv_sel[int(matches[0])])
    else:
        q_next = p_next * (resid / tail)
    mixed = p_next + float(alpha) * (q_next - p_next)
    return max(float(mixed), 1e-15)


def build_model_logprob_fn(
    model: torch.nn.Module,
    device: torch.device,
):
    def get_logprobs_batch(token_chunk: np.ndarray) -> np.ndarray:
        batch = np.asarray(token_chunk, dtype=np.int32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        ids = torch.tensor(batch, dtype=torch.long, device=device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                logits = model.forward_logits(ids)
            logits = logits.float()
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        vocab_size = logits.shape[-1]
        seq_len = batch.shape[1]
        return logits.view(batch.shape[0], seq_len, vocab_size).cpu().numpy().astype(
            np.float32,
            copy=False,
        )

    return get_logprobs_batch


def _empty_local_stats() -> dict[str, float | int]:
    return {
        "docs": 0,
        "positions": 0,
        "bytes": 0,
        "baseline_bits": 0.0,
        "ngram_emitted": 0,
        "repeat_emitted": 0,
        "safe_delta_bits": 0.0,
        "safe_rows": 0,
        "probe_delta_bits": 0.0,
        "probe_rows": 0,
        "probe_alpha_sum": 0.0,
        "probe_boosted": 0,
        "elapsed_seconds_local": 0.0,
    }


def _merge_stats(items: list[dict[str, float | int]]) -> dict[str, float | int]:
    merged = _empty_local_stats()
    for item in items:
        for key in merged:
            if key == "elapsed_seconds_local":
                merged[key] = max(float(merged[key]), float(item[key]))
            elif isinstance(merged[key], float):
                merged[key] = float(merged[key]) + float(item[key])
            else:
                merged[key] = int(merged[key]) + int(item[key])
    return merged


def evaluate_shallow_blue_submission(
    *,
    model_path: str,
    device: torch.device,
    tokenizer_path: str,
    val_files: str,
    probe_artifact_path: str,
    rank: int,
    world_size: int,
    vocab_size: int = 1024,
    window: int = 1024,
    stride: int = 1024,
    batch_windows: int = 32,
    alpha: float = 0.20,
    ngram_min_order: int = 3,
    ngram_max_order: int = 3,
    ngram_top_k: int = 3,
    ngram_min_support: int = 2,
    repeat_min_match_len: int = 4,
    repeat_max_match_len: int = 8,
    repeat_top_k: int = 3,
    repeat_min_support: int = 2,
    max_docs: int = 0,
    max_val_tokens: int = 0,
) -> ShallowBlueEvalSummary:
    t_start = time.perf_counter()
    from train_gpt import Hyperparameters, build_gpt_model, load_state_dict_artifact

    eval_model = build_gpt_model(Hyperparameters()).to(device)
    eval_model.load_state_dict(load_state_dict_artifact(Path(model_path)), strict=True)
    eval_model.eval()
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    bos_id = int(sp.bos_id())
    if bos_id < 0:
        raise ValueError(f"tokenizer {tokenizer_path} does not define BOS")
    base_bytes, has_leading_space, is_boundary = build_sentencepiece_luts(sp, vocab_size)
    all_tokens = load_validation_tokens(val_files)
    docs = truncate_docs(
        find_bos_documents(all_tokens, bos_id),
        max_docs=max_docs,
        max_val_tokens=max_val_tokens,
    )
    rank_docs = rank_slice(docs, rank, world_size)
    artifact = load_probe_runtime_artifact(probe_artifact_path)
    get_logprobs_batch = build_model_logprob_fn(eval_model, device)
    local = _empty_local_stats()
    ngram = NgramExpert(
        min_order=ngram_min_order,
        max_order=ngram_max_order,
        top_k=ngram_top_k,
        min_support=ngram_min_support,
    )
    repeat = ExactRepeatExpert(
        min_match_len=repeat_min_match_len,
        max_match_len=repeat_max_match_len,
        top_k=repeat_top_k,
        min_support=repeat_min_support,
    )
    prefix_state = ProbePrefixState()
    safe_alpha = float(np.clip(float(alpha) * float(artifact.safe_alpha_scale), 0.0, 1.0))
    feature_buffer = np.empty(len(artifact.feature_names), dtype=np.float64)

    for start, length in rank_docs:
        local["docs"] += 1
        doc_tokens = np.asarray(all_tokens[start:start + length], dtype=np.int32)
        blocks = iter_eval_blocks(doc_tokens.size, window, stride)
        ngram.reset()
        repeat.reset()
        prefix_state.reset()
        block_cursor = 0
        while block_cursor < len(blocks):
            block_indices, token_batch = collect_block_batch(
                blocks,
                doc_tokens,
                block_cursor,
                batch_windows,
            )
            if not block_indices or token_batch is None:
                raise RuntimeError(f"failed to build block batch at doc offset {block_cursor}")
            lp_batch = get_logprobs_batch(token_batch)
            for batch_idx, block_idx in enumerate(block_indices):
                context_start, _context_end, score_start, score_end = blocks[block_idx]
                lp_np = lp_batch[batch_idx]
                for abs_t in range(score_start, score_end):
                    curr_tid = int(doc_tokens[abs_t])
                    next_tid = int(doc_tokens[abs_t + 1])
                    if bool(is_boundary[curr_tid]):
                        ngram.reset()
                        repeat.reset()
                        prefix_state.reset()
                        continue
                    ngram.update(curr_tid)
                    repeat.update(curr_tid)
                    if bool(is_boundary[next_tid]):
                        continue
                    rel_t = abs_t - context_start
                    lp_row = lp_np[rel_t]
                    byte_count = int(base_bytes[next_tid])
                    if bool(has_leading_space[next_tid]) and not bool(is_boundary[curr_tid]):
                        byte_count += 1
                    baseline_bits = -float(lp_row[next_tid]) / math.log(2.0)
                    local["positions"] += 1
                    local["bytes"] += byte_count
                    local["baseline_bits"] += baseline_bits

                    ngram_pred = ngram.predict()
                    repeat_pred = repeat.predict()
                    if repeat_pred is not None:
                        local["repeat_emitted"] += 1
                    if ngram_pred is None:
                        prefix_state.update_with_target(next_tid)
                        continue
                    local["ngram_emitted"] += 1

                    safe_prob = sparse_mixed_token_prob_from_row(
                        lp_row,
                        next_tid,
                        ngram_pred.ids,
                        ngram_pred.probs,
                        safe_alpha,
                    )
                    local["safe_delta_bits"] += -math.log2(safe_prob) - baseline_bits
                    local["safe_rows"] += 1

                    backbone_top1_token_id = int(np.argmax(lp_row))
                    lp_row64 = lp_row.astype(np.float64, copy=False)
                    backbone_top1_prob = math.exp(float(lp_row64[backbone_top1_token_id]))
                    backbone_entropy_bits = float(
                        -(np.exp(lp_row64) * lp_row64).sum() / math.log(2.0)
                    )
                    metadata = ngram_pred.metadata or {}
                    repeat_metadata = repeat_pred.metadata if repeat_pred is not None else {}
                    feature_vector = build_live_probe_feature_vector(
                        ngram_prediction_token_id=int(metadata.get("prediction_token_id", -1)),
                        ngram_support=int(ngram_pred.support),
                        ngram_prediction_prob=float(metadata.get("prediction_prob", 0.0)),
                        ngram_top2_prediction_prob=float(metadata.get("top2_prediction_prob", 0.0)),
                        ngram_unique_continuations=int(metadata.get("unique_continuations", 0)),
                        ngram_continuation_entropy_bits=float(ngram_pred.entropy),
                        lz_triggered=1 if repeat_pred is not None else 0,
                        lz_match_len=int(repeat_metadata.get("match_len", 0)),
                        lz_support=0 if repeat_pred is None else int(repeat_pred.support),
                        lz_prediction_prob=float(repeat_metadata.get("prediction_prob", 0.0)),
                        backbone_top1_token_id=backbone_top1_token_id,
                        backbone_entropy_bits=backbone_entropy_bits,
                        backbone_top1_prob=backbone_top1_prob,
                        prefix_nonbos_tokens=prefix_state.prefix_nonbos_tokens(),
                        regime_repeat_fraction=prefix_state.regime_repeat_fraction(),
                        out=feature_buffer,
                    )
                    accepted = bool(artifact.accepts(feature_vector))
                    if artifact.uses_two_level_uplift:
                        floor_alpha = min(float(artifact.deployment_alpha_lo), float(alpha))
                        boosted = accepted and float(alpha) > floor_alpha + 1e-12
                        probe_alpha = float(alpha) if boosted else float(floor_alpha)
                    else:
                        boosted = accepted
                        probe_alpha = float(alpha) * float(artifact.safe_alpha_scale) if accepted else 0.0
                    probe_prob = sparse_mixed_token_prob_from_row(
                        lp_row,
                        next_tid,
                        ngram_pred.ids,
                        ngram_pred.probs,
                        probe_alpha,
                    )
                    local["probe_delta_bits"] += -math.log2(probe_prob) - baseline_bits
                    local["probe_rows"] += 1
                    local["probe_alpha_sum"] += probe_alpha
                    if boosted:
                        local["probe_boosted"] += 1
                    prefix_state.update_with_target(next_tid)
            block_cursor = block_indices[-1] + 1

    local["elapsed_seconds_local"] = time.perf_counter() - t_start
    if dist.is_available() and dist.is_initialized():
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    merged = _merge_stats(gathered)
    total_bytes = int(merged["bytes"])
    baseline_bpb = float(merged["baseline_bits"]) / max(1, total_bytes)
    safe_delta_bpb = float(merged["safe_delta_bits"]) / max(1, total_bytes)
    probe_delta_bpb = float(merged["probe_delta_bits"]) / max(1, total_bytes)
    probe_rows = int(merged["probe_rows"])
    probe_mean_alpha = float(merged["probe_alpha_sum"]) / max(1, probe_rows)
    return ShallowBlueEvalSummary(
        docs=int(merged["docs"]),
        scored_positions=int(merged["positions"]),
        total_bytes=total_bytes,
        baseline_bpb=baseline_bpb,
        ngram_emitted=int(merged["ngram_emitted"]),
        repeat_emitted=int(merged["repeat_emitted"]),
        safe_delta_bpb=safe_delta_bpb,
        safe_mixed_bpb=baseline_bpb + safe_delta_bpb,
        safe_bits_saved=-float(merged["safe_delta_bits"]),
        probe_delta_bpb=probe_delta_bpb,
        probe_mixed_bpb=baseline_bpb + probe_delta_bpb,
        probe_bits_saved=-float(merged["probe_delta_bits"]),
        probe_mean_alpha=probe_mean_alpha,
        probe_alpha_rows=probe_rows,
        probe_boosted_rows=int(merged["probe_boosted"]),
        elapsed_seconds=float(merged["elapsed_seconds_local"]),
    )
