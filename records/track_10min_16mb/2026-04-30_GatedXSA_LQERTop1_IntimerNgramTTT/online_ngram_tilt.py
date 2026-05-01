"""
Vendored online n-gram tilt helpers from PR #1145 (AnirudhRahul, valerio-endorsed).

Provides causal, normalized, prefix-only n-gram experts that propose at most one
hinted token per scored position. Caller obtains q_t = p(h_t | x) from the model
(post-TTT-adapt logits) and applies multiplicative-boost-with-renorm:

    p'(a)   = exp(beta * 1[a == h_t]) * p(a) / Z_t
    Z_t     = 1 - q_t + exp(beta) * q_t = 1 + q_t * (exp(beta) - 1)
    -log p'(y_realized) = -log p(y) - beta * 1[y == h_t] + log Z_t
                        = ptl - beta * is_hit + log1p(q_t * (exp(beta) - 1))

Compliance:
- C1 causal: hint h_t computed from strict prefix (tokens 0..t-1 only)
- C2 normalized over Sigma: closed-form Z_t over full vocab softmax
- C3 score-before-update: hints precomputed in single L->R pass; loss uses prefix-only
- C4 single pass: process_chunk advances state monotonically

Compatible with both #1934/#1855 base architectures via Hyperparameter env-var gates.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch


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


def suggest_table_bits(expected_entries: int, load_factor: float) -> int:
    if expected_entries <= 0:
        return 16
    target = max(int(expected_entries / max(load_factor, 1e-6)), 1)
    bits = max(int(math.ceil(math.log2(target))), 12)
    return min(bits, 28)


def ensure_online_ngram_lib(log0=print) -> ctypes.CDLL:
    needs_build = (not ONLINE_NGRAM_LIB.exists()) or (
        ONLINE_NGRAM_SRC.stat().st_mtime_ns > ONLINE_NGRAM_LIB.stat().st_mtime_ns
    )
    if needs_build:
        log0(f"ngram_tilt:building_native_helper src={ONLINE_NGRAM_SRC.name}")
        subprocess.run(
            [
                "gcc", "-O3", "-march=native", "-shared", "-fPIC",
                "-o", str(ONLINE_NGRAM_LIB),
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
    lib.online_ngram_state_process_chunk_token_only.restype = ctypes.c_int
    lib.online_ngram_state_process_chunk_token_only.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
    ]
    return lib


class OnlineNgramState:
    def __init__(
        self, *, lib, token_ctx_len, token_table_bits, within_table_bits,
        starts_new_word_lut, boundary_lut, seed_prefix_token,
    ):
        self.lib = lib
        self.state = lib.online_ngram_state_create(token_ctx_len, token_table_bits, within_table_bits)
        if not self.state:
            raise RuntimeError(
                f"Native ngram state alloc failed token_table_bits={token_table_bits} within_table_bits={within_table_bits}"
            )
        self.starts_new_word_lut = np.ascontiguousarray(starts_new_word_lut.astype(np.uint8, copy=False))
        self.boundary_lut = np.ascontiguousarray(boundary_lut.astype(np.uint8, copy=False))
        self.lib.online_ngram_state_seed_prefix_token(self.state, ctypes.c_uint16(int(seed_prefix_token)))

    def close(self):
        if self.state:
            self.lib.online_ngram_state_destroy(self.state)
            self.state = None

    def __del__(self):
        self.close()

    def process_chunk(self, chunk_tokens):
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
            raise RuntimeError(f"Native ngram process_chunk failed rc={rc}")
        return token_top_token, token_top_prob, within_top_token, within_top_prob, within_valid.astype(bool)

    def process_chunk_token_only(self, chunk_tokens):
        chunk_tokens = np.ascontiguousarray(chunk_tokens.astype(np.uint16, copy=False))
        n = int(chunk_tokens.size)
        token_top_token = np.zeros(n, dtype=np.uint16)
        token_top_prob = np.zeros(n, dtype=np.float32)
        rc = self.lib.online_ngram_state_process_chunk_token_only(
            self.state,
            chunk_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_int64(n),
            token_top_token.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            token_top_prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"Native ngram token-only process_chunk failed rc={rc}")
        return token_top_token, token_top_prob


class WordStartState:
    def __init__(self, *, sp, order, normalize_mode):
        self.sp = sp
        self.ctx_w = max(order - 1, 0)
        self.normalize_mode = normalize_mode
        self.prev_word_ids: deque = deque(maxlen=self.ctx_w)
        self.current_word_tokens: list = []
        self.word_to_id: dict = {}
        self.next_word_id = 1
        self.ctx_total: dict = {}
        self.pair_count: dict = {}
        self.ctx_best_token: dict = {}
        self.ctx_best_count: dict = {}

    def _flush_current_word(self):
        if not self.current_word_tokens:
            return
        text = normalize_word(self.sp.decode(self.current_word_tokens), self.normalize_mode)
        if text:
            wid = self.word_to_id.get(text)
            if wid is None:
                wid = self.next_word_id
                self.word_to_id[text] = wid
                self.next_word_id += 1
            if self.ctx_w > 0:
                self.prev_word_ids.append(wid)
        self.current_word_tokens = []

    def process_chunk(self, chunk_tokens, *, starts_new_word_lut, boundary_lut):
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
            ctx_key = None
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


def build_piece_luts(*, tokenizer_path, vocab_size):
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


def build_hints_for_targets(
    *, target_token_ids_np, tokenizer_path, vocab_size, log0=print,
    token_order=16, token_threshold=0.800, token_boost=2.625,
    within_tau=999.0, within_boost=0.0,
    word_order=4, word_normalize="strip_punct_lower",
    word_tau=999.0, word_boost=0.0,
    agree_add_boost=0.0,
):
    """Single L->R pass. Returns dict with hint_ids, gate_mask, boost_per_pos.

    target_token_ids_np: np.uint16 array of realized targets (length = total_targets).
    Output arrays are aligned to target_token_ids_np indexing.

    For each scored position t we pick at most one hint h_t:
      - prefer the expert with highest expected gain = p_top * boost - log1p(p_top * (exp(boost)-1))
      - if multiple experts agree on the same h_t, additive boost agree_add_boost
      - gate (don't tilt) when no expert clears its threshold

    The realized loss formula used by the caller:
      ptl' = ptl - beta * 1[y == h_t] + log1p(q_t * (exp(beta) - 1))   when gate_mask == True
      ptl' = ptl                                                        when gate_mask == False
    """
    sp, starts_new_word_lut, boundary_lut = build_piece_luts(
        tokenizer_path=tokenizer_path, vocab_size=vocab_size
    )
    total = int(target_token_ids_np.size)
    if total == 0:
        return {
            "hint_ids":   np.zeros(0, dtype=np.int64),
            "gate_mask":  np.zeros(0, dtype=bool),
            "boost":      np.zeros(0, dtype=np.float32),
            "sp":         sp,
            "starts_new_word_lut": starts_new_word_lut,
            "boundary_lut": boundary_lut,
        }

    token_table_bits = suggest_table_bits(total, load_factor=0.55)
    within_table_bits = suggest_table_bits(max(total // 2, 1), load_factor=0.60)
    token_only = (
        float(within_boost) == 0.0
        and float(word_boost) == 0.0
    )
    online_lib = ensure_online_ngram_lib(log0)
    ngram_state = OnlineNgramState(
        lib=online_lib,
        token_ctx_len=max(token_order - 1, 0),
        token_table_bits=token_table_bits,
        within_table_bits=within_table_bits,
        starts_new_word_lut=starts_new_word_lut,
        boundary_lut=boundary_lut,
        seed_prefix_token=int(target_token_ids_np[0]),
    )
    if token_only:
        token_top_tok, token_top_prob = ngram_state.process_chunk_token_only(target_token_ids_np)
        token_gate = token_top_prob >= np.float32(token_threshold)
        hint_ids = np.where(token_gate, token_top_tok.astype(np.int64), 0).astype(np.int64)
        boost = np.where(token_gate, np.float32(token_boost), np.float32(0.0)).astype(np.float32)
        log0(
            f"ngram_tilt:hints total={total} gated={int(token_gate.sum())} "
            f"token_gate={int(token_gate.sum())} within_gate=0 word_gate=0 agree2plus=0"
        )
        return {
            "hint_ids":   hint_ids,
            "gate_mask":  token_gate,
            "boost":      boost,
            "sp":         sp,
            "starts_new_word_lut": starts_new_word_lut,
            "boundary_lut": boundary_lut,
        }
    word_state = WordStartState(sp=sp, order=word_order, normalize_mode=word_normalize)

    token_top_tok, token_top_prob, within_top_tok, within_top_prob, within_valid = (
        ngram_state.process_chunk(target_token_ids_np)
    )
    word_top_tok, word_top_prob = word_state.process_chunk(
        target_token_ids_np,
        starts_new_word_lut=starts_new_word_lut,
        boundary_lut=boundary_lut,
    )

    def _expected_gain(p_top, boost):
        # E[ -log p'(y) under -log p(y)] when y ~ p
        # = p_top * boost - log1p(p_top * (exp(boost) - 1))
        # Maximizing this over experts => pick the most informative hint.
        log_norm = np.log1p(p_top * (math.exp(boost) - 1.0))
        return p_top * boost - log_norm

    token_gate = token_top_prob >= np.float32(token_threshold)
    within_gate = within_valid & (within_top_prob >= np.float32(within_tau))
    word_gate = word_top_prob >= np.float32(word_tau)

    token_gain = np.where(token_gate, _expected_gain(token_top_prob.astype(np.float64), token_boost), -np.inf)
    within_gain = np.where(within_gate, _expected_gain(within_top_prob.astype(np.float64), within_boost), -np.inf)
    word_gain = np.where(word_gate, _expected_gain(word_top_prob.astype(np.float64), word_boost), -np.inf)

    stack = np.stack([token_gain, within_gain, word_gain], axis=1)
    best_idx = np.argmax(stack, axis=1)
    best_gain = np.max(stack, axis=1)
    any_gate = best_gain > -np.inf

    hint_ids = np.zeros(total, dtype=np.int64)
    boost = np.zeros(total, dtype=np.float32)
    base_boost_per_expert = np.array([token_boost, within_boost, word_boost], dtype=np.float32)
    hint_per_expert = np.stack([
        token_top_tok.astype(np.int64),
        within_top_tok.astype(np.int64),
        word_top_tok.astype(np.int64),
    ], axis=1)

    rows = np.arange(total)
    hint_ids[any_gate] = hint_per_expert[rows[any_gate], best_idx[any_gate]]
    boost[any_gate] = base_boost_per_expert[best_idx[any_gate]]

    # Agreement bonus: if 2+ experts agree on the same hint as best, add agree_add_boost
    gate_mask_each = np.stack([token_gate, within_gate, word_gate], axis=1)
    expert_hints = hint_per_expert.copy()
    expert_hints[~gate_mask_each] = -1
    agreements = (expert_hints == hint_ids[:, None]).sum(axis=1)
    agreement_extra = np.where(agreements >= 2, np.float32(agree_add_boost), np.float32(0.0))
    boost = (boost + agreement_extra).astype(np.float32)

    log0(
        f"ngram_tilt:hints total={total} gated={int(any_gate.sum())} "
        f"token_gate={int(token_gate.sum())} within_gate={int(within_gate.sum())} word_gate={int(word_gate.sum())} "
        f"agree2plus={int((agreements >= 2).sum())}"
    )

    return {
        "hint_ids":   hint_ids,
        "gate_mask":  any_gate,
        "boost":      boost,
        "sp":         sp,
        "starts_new_word_lut": starts_new_word_lut,
        "boundary_lut": boundary_lut,
    }


def apply_tilt_to_ptl_torch(
    ptl: torch.Tensor,
    log_q_hint: torch.Tensor,
    target_ids: torch.Tensor,
    hint_ids: torch.Tensor,
    gate_mask: torch.Tensor,
    boost: torch.Tensor,
):
    """Closed-form tilt applied to per-token NLL.

    All tensors same shape [..., L].
        ptl_tilted = ptl - beta * 1[y == h] + log1p(q * (exp(beta) - 1))   if gate else ptl
    """
    boost64 = boost.to(torch.float64)
    q = log_q_hint.to(torch.float64).clamp_(max=0.0).exp()
    is_hit = (target_ids == hint_ids).to(torch.float64)
    log_Z = torch.log1p(q * (torch.expm1(boost64)))
    ptl_tilted = ptl.to(torch.float64) - boost64 * is_hit + log_Z
    return torch.where(gate_mask, ptl_tilted, ptl.to(torch.float64)).to(ptl.dtype)


def apply_tilt_to_ptl_torch_fast(
    ptl: torch.Tensor,
    log_q_hint: torch.Tensor,
    target_ids: torch.Tensor,
    hint_ids: torch.Tensor,
    gate_mask: torch.Tensor,
    boost: torch.Tensor,
):
    """fp32 variant of apply_tilt — cast removed where safe.

    BPB downstream accumulator is fp64, so per-token tilt computation in
    fp32 has no impact on final precision. Saves ~10-15s per eval pass on
    H100 (avoids fp64 ALU + double memory traffic).
    """
    boost32 = boost.to(torch.float32)
    q = log_q_hint.to(torch.float32).clamp_(max=0.0).exp()
    is_hit = (target_ids == hint_ids).to(torch.float32)
    log_Z = torch.log1p(q * (torch.expm1(boost32)))
    ptl_f32 = ptl.to(torch.float32)
    ptl_tilted = ptl_f32 - boost32 * is_hit + log_Z
    return torch.where(gate_mask, ptl_tilted, ptl_f32).to(ptl.dtype)
