#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import math
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_NGRAM_SRC = SCRIPT_DIR / "online_ngram_state.c"
ONLINE_NGRAM_LIB = SCRIPT_DIR / "libonline_ngram_state.so"

WHITESPACE_BYTE_IDS = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 36}
EDGE_PUNCT = ".,:;!?()[]{}<>\"'`"


def normalize_word(text: str, mode: str) -> str:
    text = text.strip()
    if mode == "lower":
        return text.lower()
    if mode == "identity":
        return text
    if mode == "strip_punct_lower":
        return text.strip(EDGE_PUNCT).lower()
    raise ValueError(f"Unknown word normalization mode: {mode}")


def apply_boost(
    llm_true_probs: np.ndarray,
    llm_hint_probs: np.ndarray,
    hit_mask: np.ndarray,
    gate_mask: np.ndarray,
    boost: float | np.ndarray,
) -> np.ndarray:
    boosted = llm_true_probs.astype(np.float64, copy=True)
    if not gate_mask.any():
        return boosted

    if np.isscalar(boost):
        scale = math.exp(float(boost))
        hit_gate = gate_mask & hit_mask
        miss_gate = gate_mask & ~hit_mask
        boosted[hit_gate] = (scale * llm_true_probs[hit_gate]) / (
            1.0 - llm_true_probs[hit_gate] + scale * llm_true_probs[hit_gate]
        )
        boosted[miss_gate] = llm_true_probs[miss_gate] / (
            1.0 - llm_hint_probs[miss_gate] + scale * llm_hint_probs[miss_gate]
        )
        return boosted

    boost_arr = boost.astype(np.float64, copy=False)
    scale = np.ones(llm_true_probs.shape, dtype=np.float64)
    scale[gate_mask] = np.exp(boost_arr[gate_mask])
    denom = 1.0 - llm_hint_probs + scale * llm_hint_probs
    boosted = llm_true_probs / denom
    hit_gate = gate_mask & hit_mask
    boosted[hit_gate] *= scale[hit_gate]
    return boosted


def expected_gain(top_prob: np.ndarray, llm_hint_prob: np.ndarray, boost: float) -> np.ndarray:
    q = np.clip(llm_hint_prob.astype(np.float64, copy=False), 1e-12, 1.0 - 1e-12)
    p = np.clip(top_prob.astype(np.float64, copy=False), 0.0, 1.0)
    log_norm = np.log1p(q * (math.exp(boost) - 1.0))
    return (p * boost - log_norm).astype(np.float32)


def compute_best_agreement_chunk(
    *,
    llm_chunk: np.ndarray,
    true_targets: np.ndarray,
    token_top_prob: np.ndarray,
    token_top_token: np.ndarray,
    token_hint_probs: np.ndarray,
    within_top_prob: np.ndarray,
    within_top_token: np.ndarray,
    within_valid: np.ndarray,
    within_hint_probs: np.ndarray,
    word_top_prob: np.ndarray,
    word_top_token: np.ndarray,
    word_hint_probs: np.ndarray,
    token_threshold: float,
    token_boost: float,
    within_tau: float,
    within_boost: float,
    word_tau: float,
    word_boost: float,
    agree_add_boost: float,
) -> np.ndarray:
    token_hit = token_top_token == true_targets
    token_gate = token_top_prob >= token_threshold
    token_exp_gain = expected_gain(token_top_prob, token_hint_probs, token_boost)

    within_hit = within_top_token == true_targets
    within_gate = within_valid & (within_top_prob >= within_tau)
    within_exp_gain = expected_gain(within_top_prob, within_hint_probs, within_boost)

    word_hit = word_top_token == true_targets
    word_gate = word_top_prob >= word_tau
    word_exp_gain = expected_gain(word_top_prob, word_hint_probs, word_boost)

    within_pick = within_gate & (~token_gate | (within_exp_gain > token_exp_gain))
    token_pick_tw = token_gate & ~within_pick
    tw_gate = token_pick_tw | within_pick

    word_pick = word_gate & ((~tw_gate) | (token_pick_tw & (word_exp_gain > token_exp_gain)) | (within_pick & (word_exp_gain > within_exp_gain)))
    token_pick = token_pick_tw & ~word_pick
    within_pick_final = within_pick & ~word_pick
    chosen_gate = token_pick | within_pick_final | word_pick

    chosen_hint_probs = np.zeros(llm_chunk.shape, dtype=np.float64)
    chosen_hint_probs[token_pick] = token_hint_probs[token_pick]
    chosen_hint_probs[within_pick_final] = within_hint_probs[within_pick_final]
    chosen_hint_probs[word_pick] = word_hint_probs[word_pick]

    chosen_hit = np.zeros(llm_chunk.shape, dtype=np.bool_)
    chosen_hit[token_pick] = token_hit[token_pick]
    chosen_hit[within_pick_final] = within_hit[within_pick_final]
    chosen_hit[word_pick] = word_hit[word_pick]

    chosen_boost = np.zeros(llm_chunk.shape, dtype=np.float64)
    chosen_boost[token_pick] = token_boost
    chosen_boost[within_pick_final] = within_boost
    chosen_boost[word_pick] = word_boost

    selected_token = np.zeros(llm_chunk.shape, dtype=np.uint16)
    selected_token[token_pick] = token_top_token[token_pick]
    selected_token[within_pick_final] = within_top_token[within_pick_final]
    selected_token[word_pick] = word_top_token[word_pick]

    agree_count = np.zeros(llm_chunk.shape, dtype=np.uint8)
    agree_count += (token_gate & (token_top_token == selected_token)).astype(np.uint8)
    agree_count += (within_gate & (within_top_token == selected_token)).astype(np.uint8)
    agree_count += (word_gate & (word_top_token == selected_token)).astype(np.uint8)
    agree_any = chosen_gate & (agree_count >= 2)

    agree_boost = chosen_boost.copy()
    agree_boost[agree_any] += agree_add_boost
    return apply_boost(llm_chunk, chosen_hint_probs, chosen_hit, chosen_gate, agree_boost)


def dist_max_float(value: float, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def suggest_table_bits(expected_entries: int, load_factor: float) -> int:
    expected_entries = max(int(expected_entries), 1)
    size = 1
    while size * load_factor < expected_entries:
        size <<= 1
    return max(size.bit_length() - 1, 10)


def loss_to_bpb(total_loss: float, total_bytes: float) -> float:
    return total_loss / (total_bytes * math.log(2.0))


def loss_to_nats_per_byte(total_loss: float, total_bytes: float) -> float:
    return total_loss / total_bytes


def build_chunk_windows(total_targets: int, seq_len: int, stride: int, chunk_tokens: int) -> list[list[int]]:
    window_starts = [
        ws
        for ws in range(0, total_targets, stride)
        if min(ws + seq_len, total_targets) - ws >= stride or ws == 0
    ]
    full_num_chunks = (total_targets + chunk_tokens - 1) // chunk_tokens
    chunk_windows: list[list[int]] = [[] for _ in range(full_num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_targets)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // chunk_tokens, full_num_chunks - 1)
        chunk_windows[ci].append(ws)
    return chunk_windows


def ensure_online_ngram_lib(log0) -> ctypes.CDLL:
    needs_build = (not ONLINE_NGRAM_LIB.exists()) or (
        ONLINE_NGRAM_SRC.stat().st_mtime_ns > ONLINE_NGRAM_LIB.stat().st_mtime_ns
    )
    if needs_build:
        log0(f"building_native_ngram_helper src={ONLINE_NGRAM_SRC.name}")
        subprocess.run(
            [
                "gcc",
                "-O3",
                "-march=native",
                "-shared",
                "-fPIC",
                "-o",
                str(ONLINE_NGRAM_LIB),
                str(ONLINE_NGRAM_SRC),
            ],
            check=True,
        )
    lib = ctypes.CDLL(str(ONLINE_NGRAM_LIB))
    lib.online_ngram_state_create.restype = ctypes.c_void_p
    lib.online_ngram_state_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.online_ngram_state_destroy.restype = None
    lib.online_ngram_state_destroy.argtypes = [ctypes.c_void_p]
    lib.online_ngram_state_seed_prefix_token.restype = None
    lib.online_ngram_state_seed_prefix_token.argtypes = [ctypes.c_void_p, ctypes.c_uint16]
    lib.online_ngram_state_process_chunk.restype = ctypes.c_int
    lib.online_ngram_state_process_chunk.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    return lib


class OnlineNgramState:
    def __init__(
        self,
        *,
        lib: ctypes.CDLL,
        token_ctx_len: int,
        token_table_bits: int,
        within_table_bits: int,
        starts_new_word_lut: np.ndarray,
        boundary_lut: np.ndarray,
        seed_prefix_token: int,
    ) -> None:
        self.lib = lib
        self.state = lib.online_ngram_state_create(token_ctx_len, token_table_bits, within_table_bits)
        if not self.state:
            raise RuntimeError(
                "Failed to allocate native online ngram state. "
                f"token_table_bits={token_table_bits} within_table_bits={within_table_bits}"
            )
        self.starts_new_word_lut = np.ascontiguousarray(starts_new_word_lut.astype(np.uint8, copy=False))
        self.boundary_lut = np.ascontiguousarray(boundary_lut.astype(np.uint8, copy=False))
        self.lib.online_ngram_state_seed_prefix_token(self.state, ctypes.c_uint16(int(seed_prefix_token)))

    def close(self) -> None:
        if self.state:
            self.lib.online_ngram_state_destroy(self.state)
            self.state = None

    def __del__(self) -> None:
        self.close()

    def process_chunk(
        self,
        chunk_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(chunk_tokens.size)
        token_top_token = np.zeros(n, dtype=np.uint16)
        token_top_prob = np.zeros(n, dtype=np.float32)
        within_top_token = np.zeros(n, dtype=np.uint16)
        within_top_prob = np.zeros(n, dtype=np.float32)
        within_valid = np.zeros(n, dtype=np.uint8)
        rc = self.lib.online_ngram_state_process_chunk(
            self.state,
            chunk_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            self.starts_new_word_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.boundary_lut.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            token_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            token_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            within_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            within_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            within_valid.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"Native online ngram chunk processing failed rc={rc}")
        return token_top_token, token_top_prob, within_top_token, within_top_prob, within_valid.astype(bool)


class WordStartState:
    def __init__(
        self,
        *,
        sp: spm.SentencePieceProcessor,
        order: int,
        normalize_mode: str,
    ) -> None:
        self.sp = sp
        self.ctx_w = max(order - 1, 0)
        self.normalize_mode = normalize_mode
        self.prev_word_ids: deque[int] = deque(maxlen=self.ctx_w)
        self.current_word_tokens: list[int] = []
        self.word_to_id: dict[str, int] = {}
        self.next_word_id = 1
        self.ctx_total: dict[tuple[int, ...], int] = {}
        self.pair_count: dict[tuple[tuple[int, ...], int], int] = {}
        self.ctx_best_token: dict[tuple[int, ...], int] = {}
        self.ctx_best_count: dict[tuple[int, ...], int] = {}

    def _flush_current_word(self) -> None:
        if not self.current_word_tokens:
            return
        text = normalize_word(
            self.sp.decode(self.current_word_tokens),
            self.normalize_mode,
        )
        if text:
            word_id = self.word_to_id.get(text)
            if word_id is None:
                word_id = self.next_word_id
                self.word_to_id[text] = word_id
                self.next_word_id += 1
            if self.ctx_w > 0:
                self.prev_word_ids.append(word_id)
        self.current_word_tokens = []

    def process_chunk(
        self,
        chunk_tokens: np.ndarray,
        *,
        starts_new_word_lut: np.ndarray,
        boundary_lut: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        top_token = np.zeros(chunk_tokens.size, dtype=np.uint16)
        top_prob = np.zeros(chunk_tokens.size, dtype=np.float32)
        for i, tok_u16 in enumerate(chunk_tokens):
            tok = int(tok_u16)
            is_boundary = bool(boundary_lut[tok])
            is_word_start = bool(starts_new_word_lut[tok]) or not self.current_word_tokens
            if is_boundary:
                self._flush_current_word()
                continue
            if bool(starts_new_word_lut[tok]):
                self._flush_current_word()

            ctx_key: tuple[int, ...] | None = None
            if is_word_start and len(self.prev_word_ids) >= self.ctx_w:
                ctx_key = tuple(self.prev_word_ids) if self.ctx_w > 0 else ()
                total = self.ctx_total.get(ctx_key, 0)
                if total > 0:
                    top_token[i] = np.uint16(self.ctx_best_token[ctx_key])
                    top_prob[i] = np.float32(self.ctx_best_count[ctx_key] / total)

            if is_word_start:
                if ctx_key is not None:
                    pair_key = (ctx_key, tok)
                    pair = self.pair_count.get(pair_key, 0) + 1
                    self.pair_count[pair_key] = pair
                    total = self.ctx_total.get(ctx_key, 0) + 1
                    self.ctx_total[ctx_key] = total
                    best_count = self.ctx_best_count.get(ctx_key, 0)
                    if pair > best_count:
                        self.ctx_best_count[ctx_key] = pair
                        self.ctx_best_token[ctx_key] = tok
                self.current_word_tokens = [tok]
            else:
                self.current_word_tokens.append(tok)
        return top_token, top_prob


def build_piece_luts(
    *,
    tokenizer_path: str,
    vocab_size: int,
) -> tuple[spm.SentencePieceProcessor, np.ndarray, np.ndarray]:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    pieces = [sp.id_to_piece(i) for i in range(sp.vocab_size())]
    starts_new_word_lut = np.zeros(vocab_size, dtype=np.uint8)
    for i, piece in enumerate(pieces):
        starts_new_word_lut[i] = 1 if piece.startswith("▁") else 0
    boundary_lut = np.zeros(vocab_size, dtype=np.uint8)
    bos_id = sp.bos_id()
    if bos_id >= 0 and bos_id < vocab_size:
        boundary_lut[bos_id] = 1
    for tok in range(min(sp.vocab_size(), vocab_size)):
        if sp.is_byte(tok) and tok in WHITESPACE_BYTE_IDS:
            boundary_lut[tok] = 1
    return sp, starts_new_word_lut, boundary_lut


def compile_logits_fn(model: torch.nn.Module, *, seq_len: int, device: torch.device, log0):
    if os.environ.get("EVAL_COMPILE", "0") != "1":
        log0("eval-pass-online: using eager logits path")
        return model.forward_logits
    log0("eval-pass-online: compiling logits path")
    compiled = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    dummy = torch.zeros(1, seq_len, dtype=torch.int64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        _ = compiled(dummy)
    del dummy
    log0("eval-pass-online: compile warmup done")
    return compiled


def partition_windows(windows: list[int], rank: int, world_size: int) -> list[int]:
    start = (len(windows) * rank) // world_size
    end = (len(windows) * (rank + 1)) // world_size
    return windows[start:end]


def eval_val_sliding_online_best_agree(
    *,
    args,
    base_model: torch.nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
    log0=print,
) -> tuple[float, float, dict[str, float]]:
    startup_t0 = time.perf_counter()
    seq_len = eval_seq_len or args.train_seq_len
    chunk_tokens = int(os.environ.get("CHUNK_TOKENS", "131072"))
    token_order = int(os.environ.get("TOKEN_ORDER", "16"))
    token_threshold = float(os.environ.get("TOKEN_THRESHOLD", "0.800"))
    token_boost = float(os.environ.get("TOKEN_BOOST", "2.625"))
    within_tau = float(os.environ.get("WITHIN_TAU", "0.450"))
    within_boost = float(os.environ.get("WITHIN_BOOST", "0.750"))
    word_order = int(os.environ.get("WORD_ORDER", "4"))
    word_normalize = os.environ.get("WORD_NORMALIZE", "strip_punct_lower")
    word_tau = float(os.environ.get("WORD_TAU", "0.650"))
    word_boost = float(os.environ.get("WORD_BOOST", "0.750"))
    agree_add_boost = float(os.environ.get("AGREE_ADD_BOOST", "0.500"))

    total_targets = val_tokens.numel() - 1
    tokens_np = val_tokens.cpu().numpy().astype(np.uint16, copy=False)
    sp, starts_new_word_lut, boundary_lut = build_piece_luts(
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
    )
    token_table_bits = int(
        os.environ.get(
            "TOKEN_TABLE_BITS",
            str(suggest_table_bits(total_targets, load_factor=0.55)),
        )
    )
    within_table_bits = int(
        os.environ.get(
            "WITHIN_TABLE_BITS",
            str(suggest_table_bits(max(total_targets // 2, 1), load_factor=0.60)),
        )
    )
    online_lib = ensure_online_ngram_lib(log0)
    ngram_state = OnlineNgramState(
        lib=online_lib,
        token_ctx_len=max(token_order - 1, 0),
        token_table_bits=token_table_bits,
        within_table_bits=within_table_bits,
        starts_new_word_lut=starts_new_word_lut,
        boundary_lut=boundary_lut,
        seed_prefix_token=int(tokens_np[0]),
    )
    word_state = WordStartState(
        sp=sp,
        order=word_order,
        normalize_mode=word_normalize,
    )

    compiled_logits = compile_logits_fn(base_model, seq_len=seq_len, device=device, log0=log0 if rank == 0 else (lambda *_: None))
    startup_s = time.perf_counter() - startup_t0
    startup_max_s = dist_max_float(startup_s, device, world_size)
    if rank == 0:
        log0(
            f"online_best_agree:start total_targets={total_targets} seq_len={seq_len} stride={stride} "
            f"chunk_tokens={chunk_tokens} batch_seqs={batch_seqs} token_order={token_order} "
            f"word_order={word_order} startup_max={startup_max_s:.2f}s"
        )

    chunk_windows = build_chunk_windows(total_targets, seq_len, stride, chunk_tokens)

    llm_loss_sum = 0.0
    best_agree_loss_sum = 0.0
    byte_sum = 0.0
    token_count = 0.0
    state_time_s = 0.0
    input_time_s = 0.0
    forward_time_s = 0.0
    blend_time_s = 0.0
    loop_t0 = time.perf_counter()

    try:
        with torch.inference_mode():
            for ci, windows in enumerate(chunk_windows):
                if not windows:
                    continue
                chunk_t0 = ci * chunk_tokens
                chunk_t1 = min((ci + 1) * chunk_tokens, total_targets)
                chunk_target_tokens = np.ascontiguousarray(tokens_np[chunk_t0 + 1 : chunk_t1 + 1], dtype=np.uint16)

                t_state0 = time.perf_counter()
                token_top_token, token_top_prob, within_top_token, within_top_prob, within_valid = ngram_state.process_chunk(
                    chunk_target_tokens
                )
                word_top_token, word_top_prob = word_state.process_chunk(
                    chunk_target_tokens,
                    starts_new_word_lut=starts_new_word_lut,
                    boundary_lut=boundary_lut,
                )
                state_time_s += time.perf_counter() - t_state0

                my_windows = partition_windows(windows, rank, world_size)
                for bi in range(0, len(my_windows), batch_seqs):
                    batch_ws = my_windows[bi : bi + batch_seqs]
                    if not batch_ws:
                        continue
                    t_input0 = time.perf_counter()
                    bsz = len(batch_ws)
                    x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    token_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    within_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    word_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                    wlens: list[int] = []
                    score_starts: list[int] = []

                    for i, ws in enumerate(batch_ws):
                        end = min(ws + seq_len, total_targets)
                        wlen = end - ws
                        wlens.append(wlen)
                        local = val_tokens[ws : end + 1].to(device=device, dtype=torch.int64)
                        x_batch[i, :wlen] = local[:-1]
                        y_batch[i, :wlen] = local[1:]
                        s = 0 if ws == 0 else max(wlen - stride, 0)
                        score_starts.append(s)
                        if wlen - s <= 0:
                            continue
                        c0 = ws + s - chunk_t0
                        c1 = ws + wlen - chunk_t0
                        token_batch[i, s:wlen] = torch.from_numpy(
                            np.asarray(token_top_token[c0:c1], dtype=np.int64)
                        ).to(device=device, dtype=torch.int64)
                        within_batch[i, s:wlen] = torch.from_numpy(
                            np.asarray(within_top_token[c0:c1], dtype=np.int64)
                        ).to(device=device, dtype=torch.int64)
                        word_batch[i, s:wlen] = torch.from_numpy(
                            np.asarray(word_top_token[c0:c1], dtype=np.int64)
                        ).to(device=device, dtype=torch.int64)
                    input_time_s += time.perf_counter() - t_input0

                    t_forward0 = time.perf_counter()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                            torch.compiler.cudagraph_mark_step_begin()
                        logits = compiled_logits(x_batch)
                    logits_f = logits.float()
                    log_norm = torch.logsumexp(logits_f, dim=-1)
                    true_logits = logits_f.gather(-1, y_batch.unsqueeze(-1)).squeeze(-1)
                    token_logits = logits_f.gather(-1, token_batch.unsqueeze(-1)).squeeze(-1)
                    within_logits = logits_f.gather(-1, within_batch.unsqueeze(-1)).squeeze(-1)
                    word_logits = logits_f.gather(-1, word_batch.unsqueeze(-1)).squeeze(-1)
                    true_probs = (true_logits - log_norm).exp()
                    token_hint = (token_logits - log_norm).exp()
                    within_hint = (within_logits - log_norm).exp()
                    word_hint = (word_logits - log_norm).exp()
                    forward_time_s += time.perf_counter() - t_forward0

                    t_blend0 = time.perf_counter()
                    for i, ws in enumerate(batch_ws):
                        wlen = wlens[i]
                        s = score_starts[i]
                        if wlen - s <= 0:
                            continue
                        c0 = ws + s - chunk_t0
                        c1 = ws + wlen - chunk_t0
                        llm_chunk = true_probs[i, s:wlen].detach().cpu().numpy().astype(np.float64, copy=False)
                        token_hint_chunk = token_hint[i, s:wlen].detach().cpu().numpy().astype(np.float32, copy=False)
                        within_hint_chunk = within_hint[i, s:wlen].detach().cpu().numpy().astype(np.float32, copy=False)
                        word_hint_chunk = word_hint[i, s:wlen].detach().cpu().numpy().astype(np.float32, copy=False)
                        best_agree_chunk = compute_best_agreement_chunk(
                            llm_chunk=llm_chunk,
                            true_targets=chunk_target_tokens[c0:c1],
                            token_top_prob=np.asarray(token_top_prob[c0:c1], dtype=np.float32),
                            token_top_token=np.asarray(token_top_token[c0:c1], dtype=np.uint16),
                            token_hint_probs=token_hint_chunk,
                            within_top_prob=np.asarray(within_top_prob[c0:c1], dtype=np.float32),
                            within_top_token=np.asarray(within_top_token[c0:c1], dtype=np.uint16),
                            within_valid=np.asarray(within_valid[c0:c1], dtype=np.bool_),
                            within_hint_probs=within_hint_chunk,
                            word_top_prob=np.asarray(word_top_prob[c0:c1], dtype=np.float32),
                            word_top_token=np.asarray(word_top_token[c0:c1], dtype=np.uint16),
                            word_hint_probs=word_hint_chunk,
                            token_threshold=token_threshold,
                            token_boost=token_boost,
                            within_tau=within_tau,
                            within_boost=within_boost,
                            word_tau=word_tau,
                            word_boost=word_boost,
                            agree_add_boost=agree_add_boost,
                        )
                        llm_loss_sum += float((-np.log(np.clip(llm_chunk, 1e-12, 1.0))).sum())
                        best_agree_loss_sum += float((-np.log(np.clip(best_agree_chunk, 1e-12, 1.0))).sum())
                        token_count += float(c1 - c0)
                        tgt = y_batch[i, s:wlen]
                        prev = x_batch[i, s:wlen]
                        tb = base_bytes_lut[tgt].to(torch.float64)
                        tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                        byte_sum += float(tb.sum().item())
                    blend_time_s += time.perf_counter() - t_blend0
    finally:
        ngram_state.close()

    llm_loss_t = torch.tensor([llm_loss_sum], dtype=torch.float64, device=device)
    best_loss_t = torch.tensor([best_agree_loss_sum], dtype=torch.float64, device=device)
    byte_sum_t = torch.tensor([byte_sum], dtype=torch.float64, device=device)
    token_count_t = torch.tensor([token_count], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(llm_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(best_loss_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count_t, op=dist.ReduceOp.SUM)

    state_max_s = dist_max_float(state_time_s, device, world_size)
    input_max_s = dist_max_float(input_time_s, device, world_size)
    forward_max_s = dist_max_float(forward_time_s, device, world_size)
    blend_max_s = dist_max_float(blend_time_s, device, world_size)
    loop_total_max_s = dist_max_float(time.perf_counter() - loop_t0, device, world_size)

    llm_total_loss = float(llm_loss_t.item())
    best_total_loss = float(best_loss_t.item())
    total_bytes = float(byte_sum_t.item())
    total_token_count = float(token_count_t.item())
    llm_bpb = loss_to_bpb(llm_total_loss, total_bytes)
    best_agree_bpb = loss_to_bpb(best_total_loss, total_bytes)

    timings = {
        "llm_bpb": llm_bpb,
        "best_agree_bpb": best_agree_bpb,
        "gain_bpb": llm_bpb - best_agree_bpb,
        "startup_max_s": startup_max_s,
        "loop_total_max_s": loop_total_max_s,
        "state_max_s": state_max_s,
        "input_max_s": input_max_s,
        "forward_max_s": forward_max_s,
        "blend_max_s": blend_max_s,
        "llm_nats_per_byte": loss_to_nats_per_byte(llm_total_loss, total_bytes),
        "best_agree_nats_per_byte": loss_to_nats_per_byte(best_total_loss, total_bytes),
        "gain_nats_per_byte": loss_to_nats_per_byte(llm_total_loss, total_bytes)
        - loss_to_nats_per_byte(best_total_loss, total_bytes),
    }
    if rank == 0:
        log0(
            f"online_best_agree:done llm_bpb={llm_bpb:.8f} best_agree_bpb={best_agree_bpb:.8f} "
            f"gain_bpb={llm_bpb - best_agree_bpb:.8f} startup_max={startup_max_s:.2f}s "
            f"loop_total_max={loop_total_max_s:.2f}s state_max={state_max_s:.2f}s "
            f"input_max={input_max_s:.2f}s forward_max={forward_max_s:.2f}s blend_max={blend_max_s:.2f}s"
        )
    return best_total_loss / max(total_token_count, 1.0), best_agree_bpb, timings
