"""Training script for the SP1024 linear-attention hybrid submission.

This file is a single-process entry point launched via ``torchrun``.  It builds
a small hybrid language model that alternates softmax attention with a gated
linear-time mixer, trains it with a Muon/AdamW mix for ~10 minutes on 8xH100s,
runs short late-stage quantization-aware training, then performs a GPTQ-style
post-training quantization pass to integer weights and brotli-compresses the
result so the artifact stays below 16,000,000 bytes.

The expected command line is::

    torchrun --standalone --nproc_per_node=8 train_gpt.py

All tunables are read from environment variables so sweeps can override a
single knob without editing the file.

The script is capped at 1500 lines (the challenge rule for reference entry
scripts).  Keep it concise.
"""
from __future__ import annotations

import bisect
import bz2
import copy
import glob
import io
import math
import os
import pickle
import random
import re
import struct
import sys
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import brotli  # type: ignore
    _HAVE_BROTLI = True
except Exception:  # pragma: no cover
    _HAVE_BROTLI = False


# =============================================================================
# Hyperparameters
# =============================================================================

class Hyperparameters:
    """All tunables exposed through environment variables."""

    # ---- Data ----
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    # ---- Cadence ----
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 300))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 50))

    # ---- Budget ----
    iterations = int(os.environ.get("ITERATIONS", 2400))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 60))
    warmdown_start = int(os.environ.get("WARMDOWN_START", 1900))  # step to begin linear cool-down
    warmdown_end = int(os.environ.get("WARMDOWN_END", 2400))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 595.0))
    max_quant_wallclock_seconds = float(os.environ.get("MAX_QUANT_WALLCLOCK_SECONDS", 540.0))

    # ---- Model shape ----
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 2))
    mlp_mult_num = int(os.environ.get("MLP_MULT_NUM", 5))  # MLP hidden = (num/den) * model_dim
    mlp_mult_den = int(os.environ.get("MLP_MULT_DEN", 2))
    rope_frac = float(os.environ.get("ROPE_FRAC", 0.5))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    swa_window = int(os.environ.get("SWA_WINDOW", 512))
    swa_layers = int(os.environ.get("SWA_LAYERS", 8))   # first N layers use sliding window
    mixer_layers_str = os.environ.get("MIXER_LAYERS", "1,3,5,7,9")  # which layers use the linear mixer
    mixer_chunk = int(os.environ.get("MIXER_CHUNK", 64))
    mixer_state_dim = int(os.environ.get("MIXER_STATE_DIM", 64))
    mixer_heads = int(os.environ.get("MIXER_HEADS", 4))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))

    # ---- Optimiser ----
    embed_lr = float(os.environ.get("EMBED_LR", 0.35))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.028))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.032))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_ns_steps = int(os.environ.get("MUON_NS_STEPS", 5))
    muon_weight_decay = float(os.environ.get("MUON_WD", 0.05))
    adam_weight_decay = float(os.environ.get("ADAM_WD", 0.0))
    adam_beta1 = float(os.environ.get("ADAM_BETA1", 0.9))
    adam_beta2 = float(os.environ.get("ADAM_BETA2", 0.98))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 4.0))

    # ---- EMA / SWA ----
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ema_interval = int(os.environ.get("EMA_INTERVAL", 32))
    swa_start = int(os.environ.get("SWA_START", 2100))
    swa_interval = int(os.environ.get("SWA_INTERVAL", 75))

    # ---- Quant-aware training ----
    qat_start = int(os.environ.get("QAT_START", 1950))
    qat_bits_matrix = int(os.environ.get("QAT_BITS_MATRIX", 6))

    # ---- Post-training quantisation ----
    gptq_matrix_bits = int(os.environ.get("GPTQ_MATRIX_BITS", 6))
    gptq_embed_bits = int(os.environ.get("GPTQ_EMBED_BITS", 7))
    gptq_hessian_rows = int(os.environ.get("GPTQ_HESSIAN_ROWS", 16))
    gptq_sigma_attn = float(os.environ.get("GPTQ_SIGMA_ATTN", 3.1))
    gptq_sigma_mlp = float(os.environ.get("GPTQ_SIGMA_MLP", 3.0))
    gptq_sigma_mixer = float(os.environ.get("GPTQ_SIGMA_MIXER", 3.2))
    gptq_sigma_embed = float(os.environ.get("GPTQ_SIGMA_EMBED", 3.4))
    keep_float_numel = int(os.environ.get("KEEP_FLOAT_NUMEL", 65_536))

    # ---- Artifact cap ----
    artifact_cap_bytes = int(os.environ.get("ARTIFACT_CAP_BYTES", 16_000_000))

    # ---- Sliding-window eval ----
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_ctx = int(os.environ.get("EVAL_CTX", 1024))


# =============================================================================
# Logging helpers
# =============================================================================

def _is_rank_zero() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def log0(msg: str) -> None:
    if _is_rank_zero():
        print(msg, flush=True)


# =============================================================================
# Data loading
# =============================================================================

def load_data_shard(path: Path) -> Tensor:
    """Load a single ``.bin`` shard written by the challenge data pipeline.

    The shards start with a small header and then pack the tokens as uint16.
    """
    with open(path, "rb") as fh:
        header = fh.read(2 * 256)
        header_words = np.frombuffer(header, dtype=np.int32)
        magic = int(header_words[0])
        version = int(header_words[1])
        if magic != 20240520 or version != 1:
            raise RuntimeError(f"bad shard header at {path}: {magic},{version}")
        num_tokens = int(header_words[2])
        buf = fh.read(num_tokens * 2)
        if len(buf) != num_tokens * 2:
            raise RuntimeError(f"short read in {path}")
    return torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).astype(np.int32)).to(torch.int64)


class ShardReader:
    """Streams tokens from a glob of shards in deterministic rank-interleaved order."""

    def __init__(self, pattern: str, rank: int, world_size: int, seq_len: int, batch_seqs: int):
        self.paths = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.paths:
            raise FileNotFoundError(f"no shards matched {pattern}")
        self.rank = rank
        self.world_size = world_size
        self.seq_len = seq_len
        self.batch_seqs = batch_seqs
        self.chunk = seq_len * batch_seqs * world_size
        self.reset()

    def reset(self) -> None:
        self.shard_idx = 0
        self.cursor = 0
        self._refill()

    def _refill(self) -> None:
        while True:
            self.tokens = load_data_shard(self.paths[self.shard_idx])
            self.cursor = 0
            self.shard_idx = (self.shard_idx + 1) % len(self.paths)
            if self.tokens.numel() >= self.chunk + 1:
                return

    def next_batch(self, device: torch.device) -> tuple[Tensor, Tensor]:
        if self.cursor + self.chunk + 1 > self.tokens.numel():
            self._refill()
        c = self.cursor
        chunk = self.tokens[c : c + self.chunk + 1]
        self.cursor += self.chunk
        start = self.rank * self.seq_len * self.batch_seqs
        local = chunk[start : start + self.seq_len * self.batch_seqs + 1]
        x = local[:-1].reshape(self.batch_seqs, self.seq_len).to(device, non_blocking=True)
        y = local[1:].reshape(self.batch_seqs, self.seq_len).to(device, non_blocking=True)
        return x, y


# =============================================================================
# Tokenizer LUTs (tokenizer-agnostic bits-per-byte)
# =============================================================================

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab = int(sp.vocab_size())
    table = max(sp_vocab, vocab_size)
    base = np.zeros((table,), dtype=np.int16)
    has_space = np.zeros((table,), dtype=np.bool_)
    boundary = np.ones((table,), dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        boundary[tid] = False
        if sp.is_byte(tid):
            base[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(boundary, dtype=torch.bool, device=device),
    )


# =============================================================================
# Model
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dt)


def half_rope_cos_sin(seq_len: int, rope_dim: int, base: float, device: torch.device) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device).float() / rope_dim))
    t = torch.arange(seq_len, device=device).float()
    ang = torch.outer(t, freqs)
    return ang.cos(), ang.sin()


def apply_half_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    # x: [B, H, T, D]
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x1 = x_rot[..., 0::2]
    x2 = x_rot[..., 1::2]
    cos_ = cos[None, None, : x.size(-2), :]
    sin_ = sin[None, None, : x.size(-2), :]
    r1 = x1 * cos_ - x2 * sin_
    r2 = x1 * sin_ + x2 * cos_
    rotated = torch.stack((r1, r2), dim=-1).flatten(-2)
    return torch.cat((rotated, x_pass), dim=-1)


class Attention(nn.Module):
    """Grouped-query attention with half-RoPE and optional sliding window."""

    def __init__(self, cfg: Hyperparameters, use_swa: bool):
        super().__init__()
        self.cfg = cfg
        self.use_swa = use_swa
        self.head_dim = cfg.model_dim // cfg.num_heads
        self.nq = cfg.num_heads
        self.nk = cfg.num_kv_heads
        self.rep = self.nq // self.nk
        self.rope_dim = int(self.head_dim * cfg.rope_frac)
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1
        self.q_proj = nn.Linear(cfg.model_dim, self.nq * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.model_dim, self.nk * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.model_dim, self.nk * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.nq * self.head_dim, cfg.model_dim, bias=False)
        self.qk_gain = nn.Parameter(torch.tensor(cfg.qk_gain_init))
        # half-RoPE tables are created lazily on first forward
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None
        # Initialise linear projections to a small scale.
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.o_proj.weight)

    def _rope(self, device: torch.device, seq_len: int) -> tuple[Tensor, Tensor]:
        if self._cos is None or self._cos.size(0) < seq_len or self._cos.device != device:
            cos, sin = half_rope_cos_sin(max(seq_len, self.cfg.train_seq_len), self.rope_dim, self.cfg.rope_base, device)
            self._cos, self._sin = cos, sin
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nq, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nk, self.head_dim).transpose(1, 2)
        # Normalise Q/K then scale by a learned gain (stabilises training, keeps softmax temperature meaningful).
        q = F.rms_norm(q, (self.head_dim,), eps=1e-6) * self.qk_gain
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)
        cos, sin = self._rope(x.device, T)
        q = apply_half_rope(q, cos, sin, self.rope_dim)
        k = apply_half_rope(k, cos, sin, self.rope_dim)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        if self.use_swa:
            attn_mask = _sliding_window_mask(T, self.cfg.swa_window, x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nq * self.head_dim)
        return self.o_proj(out)


def _sliding_window_mask(T: int, window: int, device: torch.device) -> Tensor:
    i = torch.arange(T, device=device)
    j = torch.arange(T, device=device)
    allowed = (j[None, :] <= i[:, None]) & (j[None, :] > i[:, None] - window)
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)
    mask[allowed] = True
    # SDPA expects a boolean mask that is ``True`` where attention is allowed.
    return mask


class GatedMixer(nn.Module):
    """Chunked gated linear-time mixer.

    Within each chunk we perform full softmax attention; state is carried across
    chunks as a per-head low-rank matrix S in R^{H x Dk x Dv}, updated by a
    decayed outer product.  This gives O(T * chunk) wall-cost with full
    intra-chunk expressivity.
    """

    def __init__(self, cfg: Hyperparameters):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.mixer_heads
        self.dk = cfg.mixer_state_dim
        self.dv = cfg.model_dim // self.h
        self.chunk = cfg.mixer_chunk
        self.q_proj = nn.Linear(cfg.model_dim, self.h * self.dk, bias=False)
        self.k_proj = nn.Linear(cfg.model_dim, self.h * self.dk, bias=False)
        self.v_proj = nn.Linear(cfg.model_dim, self.h * self.dv, bias=False)
        self.g_proj = nn.Linear(cfg.model_dim, self.h, bias=True)
        self.o_proj = nn.Linear(self.h * self.dv, cfg.model_dim, bias=False)
        self.qk_gain = nn.Parameter(torch.tensor(cfg.qk_gain_init))
        for m in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.g_proj.weight)
        nn.init.constant_(self.g_proj.bias, 3.0)  # high gate -> initially mostly local

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        C = self.chunk
        if T % C != 0:
            # Pad on the right; trimmed at the end.
            pad = (C - T % C) % C
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.size(1)
        q = self.q_proj(x).view(B, T_pad, self.h, self.dk).transpose(1, 2)  # [B,H,T,Dk]
        k = self.k_proj(x).view(B, T_pad, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(x).view(B, T_pad, self.h, self.dv).transpose(1, 2)  # [B,H,T,Dv]
        g = torch.sigmoid(self.g_proj(x)).transpose(1, 2)  # [B,H,T] in (0,1) - retention factor
        q = F.rms_norm(q, (self.dk,), eps=1e-6) * self.qk_gain
        k = F.rms_norm(k, (self.dk,), eps=1e-6)
        n_chunks = T_pad // C
        q = q.view(B, self.h, n_chunks, C, self.dk)
        k = k.view(B, self.h, n_chunks, C, self.dk)
        v = v.view(B, self.h, n_chunks, C, self.dv)
        g = g.view(B, self.h, n_chunks, C)
        # ---- Intra-chunk softmax ----
        # causal per chunk
        mask = torch.ones(C, C, device=x.device, dtype=torch.bool).tril()
        scores = torch.einsum("bhntd,bhnsd->bhnts", q, k) / math.sqrt(self.dk)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        y_in = torch.einsum("bhnts,bhnsd->bhntd", attn, v)  # [B,H,N,C,Dv]
        # ---- Cross-chunk recurrent state ----
        # S[n] = S[n-1] * gamma[n] + sum_t (k_t ⊗ v_t),  gamma[n] = prod_t g_t
        # For simplicity we compress each chunk into a single additive update:
        kv = torch.einsum("bhntd,bhnte->bhnde", k, v)  # [B,H,N,Dk,Dv] per-chunk outer product sum
        gamma = g.log().sum(dim=-1).exp()  # [B,H,N] product of gate values in chunk
        # Prefix-scan chunk states
        state = torch.zeros(B, self.h, self.dk, self.dv, device=x.device, dtype=kv.dtype)
        ys_cross = []
        for n in range(n_chunks):
            # output uses state BEFORE applying this chunk's updates (keeps causality)
            y_cross = torch.einsum("bhtd,bhde->bhte", q[:, :, n], state)  # [B,H,C,Dv]
            ys_cross.append(y_cross)
            state = state * gamma[:, :, n, None, None] + kv[:, :, n]
        y_cross = torch.stack(ys_cross, dim=2)  # [B,H,N,C,Dv]
        y = y_in + y_cross
        y = y.reshape(B, self.h, T_pad, self.dv).transpose(1, 2).contiguous().view(B, T_pad, self.h * self.dv)
        y = self.o_proj(y)
        if T_pad != T:
            y = y[:, :T, :]
        return y


class MLP(nn.Module):
    def __init__(self, cfg: Hyperparameters):
        super().__init__()
        hid = cfg.model_dim * cfg.mlp_mult_num // cfg.mlp_mult_den
        hid = (hid + 7) // 8 * 8
        self.fc1 = nn.Linear(cfg.model_dim, 2 * hid, bias=False)  # [gate | value]
        self.fc2 = nn.Linear(hid, cfg.model_dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc2.weight)
        self.hid = hid

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: Hyperparameters, layer_idx: int, mixer_idx_set: set[int]):
        super().__init__()
        self.pre_mixer_norm = RMSNorm(cfg.model_dim)
        if layer_idx in mixer_idx_set:
            self.mixer: nn.Module = GatedMixer(cfg)
            self.kind = "mixer"
        else:
            use_swa = layer_idx < cfg.swa_layers
            self.mixer = Attention(cfg, use_swa=use_swa)
            self.kind = "attn"
        self.pre_mlp_norm = RMSNorm(cfg.model_dim)
        self.mlp = MLP(cfg)
        # Small residual scales to stabilise the zero-init output paths.
        self.mix_scale = nn.Parameter(torch.full((cfg.model_dim,), 0.5))
        self.mlp_scale = nn.Parameter(torch.full((cfg.model_dim,), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mix_scale * self.mixer(self.pre_mixer_norm(x))
        x = x + self.mlp_scale * self.mlp(self.pre_mlp_norm(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: Hyperparameters):
        super().__init__()
        self.cfg = cfg
        mixer_idxs = {int(s) for s in cfg.mixer_layers_str.split(",") if s.strip()}
        # Keep mixer usage within valid layer range.
        mixer_idxs = {i for i in mixer_idxs if 0 <= i < cfg.num_layers}
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        nn.init.normal_(self.embed.weight, std=cfg.tied_embed_init_std)
        self.blocks = nn.ModuleList([Block(cfg, i, mixer_idxs) for i in range(cfg.num_layers)])
        self.norm_out = RMSNorm(cfg.model_dim)
        if cfg.tie_embeddings:
            self.head_weight: Tensor | None = None  # uses self.embed.weight
            self.head_bias = None
        else:
            self.head = nn.Linear(cfg.model_dim, cfg.vocab_size, bias=False)
            nn.init.zeros_(self.head.weight)
        self.head_scale = nn.Parameter(torch.tensor(cfg.logit_softcap))
        self.mixer_idxs = mixer_idxs

    def head_project(self, h: Tensor) -> Tensor:
        if self.cfg.tie_embeddings:
            w = self.embed.weight
            logits = F.linear(h, w)
        else:
            logits = self.head(h)
        # softcap stabilises logits; cap scales with a learned head gain.
        cap = self.cfg.logit_softcap
        logits = cap * torch.tanh(logits / cap)
        return logits

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)
        logits = self.head_project(h)
        if y is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        return loss


# =============================================================================
# Newton–Schulz orthogonaliser for Muon
# =============================================================================

@torch.no_grad()
def newton_schulz5(g: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalise a matrix via 5-step Newton–Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    if x.size(0) > x.size(1):
        x = x.mT
        flip = True
    else:
        flip = False
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_ = x @ x.mT
        b_ = b * a_ + c * a_ @ a_
        x = a * x + b_ @ x
    if flip:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimiser — orthogonalised momentum SGD for 2D matrices.

    Each param group is expected to contain matrices of the same shape so we
    can flatten updates across ranks.  1D parameters should be handled by
    a separate AdamW.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nes = group["nesterov"]
            ns = group["ns_steps"]
            wd = group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            world = dist.get_world_size() if distributed else 1
            rank = dist.get_rank() if distributed else 0
            for i, p in enumerate(params):
                if (i % world) == rank:
                    g = p.grad
                    if wd != 0:
                        g = g.add(p, alpha=wd)
                    st = self.state[p]
                    buf = st.get("mom")
                    if buf is None:
                        buf = torch.zeros_like(p)
                        st["mom"] = buf
                    buf.mul_(mom).add_(g)
                    g2 = g.add(buf, alpha=mom) if nes else buf.clone()
                    g2 = newton_schulz5(g2, steps=ns)
                    # Scale correction from the reference Muon implementation.
                    g2 *= max(1.0, g2.size(0) / g2.size(1)) ** 0.5
                    flat[cur : cur + p.numel()] = g2.reshape(-1).to(torch.bfloat16)
                cur += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                upd = flat[cur : cur + p.numel()].view_as(p).to(p.dtype)
                p.add_(upd, alpha=-lr)
                cur += p.numel()
        return loss


# =============================================================================
# Optimiser assembly
# =============================================================================

def build_optimisers(model: LM, cfg: Hyperparameters) -> tuple[list, Callable[[int], dict]]:
    muon_matrices: list[nn.Parameter] = []
    adam_embed: list[nn.Parameter] = []
    adam_scalar: list[nn.Parameter] = []
    adam_head: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name:
            adam_embed.append(p)
        elif p.ndim >= 2:
            muon_matrices.append(p)
        elif "head_scale" in name:
            adam_head.append(p)
        else:
            adam_scalar.append(p)
    muon = Muon(muon_matrices, lr=cfg.matrix_lr, momentum=cfg.muon_momentum,
                ns_steps=cfg.muon_ns_steps, weight_decay=cfg.muon_weight_decay)
    adam = torch.optim.AdamW(
        [
            {"params": adam_embed, "lr": cfg.tied_embed_lr if cfg.tie_embeddings else cfg.embed_lr,
             "betas": (cfg.adam_beta1, cfg.adam_beta2), "weight_decay": cfg.adam_weight_decay},
            {"params": adam_scalar, "lr": cfg.scalar_lr,
             "betas": (cfg.adam_beta1, cfg.adam_beta2), "weight_decay": cfg.adam_weight_decay},
            {"params": adam_head, "lr": cfg.head_lr,
             "betas": (cfg.adam_beta1, cfg.adam_beta2), "weight_decay": cfg.adam_weight_decay},
        ]
    )

    base_lrs = {
        "muon": [g["lr"] for g in muon.param_groups],
        "adam": [g["lr"] for g in adam.param_groups],
    }

    def set_lr(step: int) -> dict:
        mult = lr_schedule(step, cfg)
        for bg, g in zip(base_lrs["muon"], muon.param_groups):
            g["lr"] = bg * mult
        for bg, g in zip(base_lrs["adam"], adam.param_groups):
            g["lr"] = bg * mult
        return {"lr_mul": mult}

    return [muon, adam], set_lr


def lr_schedule(step: int, cfg: Hyperparameters) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / max(cfg.warmup_steps, 1)
    if step < cfg.warmdown_start:
        return 1.0
    if step < cfg.warmdown_end:
        # Linear cool-down to 0.
        frac = (step - cfg.warmdown_start) / max(cfg.warmdown_end - cfg.warmdown_start, 1)
        return 1.0 - frac
    return 0.0


# =============================================================================
# EMA / SWA shadow model
# =============================================================================

class ParamEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def copy_into(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))


class SWA:
    def __init__(self):
        self.count = 0
        self.avg: dict[str, Tensor] | None = None

    @torch.no_grad()
    def absorb(self, model: nn.Module) -> None:
        sd = model.state_dict()
        if self.avg is None:
            self.avg = {k: v.detach().clone().float() for k, v in sd.items()}
            self.count = 1
            return
        self.count += 1
        for k, v in sd.items():
            if k in self.avg:
                self.avg[k].mul_((self.count - 1) / self.count).add_(v.detach().float(), alpha=1.0 / self.count)

    @torch.no_grad()
    def copy_into(self, model: nn.Module) -> None:
        if self.avg is None:
            return
        sd = model.state_dict()
        for k, v in self.avg.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))


# =============================================================================
# Quant-aware training (fake-quant round-trip in forward)
# =============================================================================

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor, bits: int):
        q_max = (1 << (bits - 1)) - 1
        with torch.no_grad():
            scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / q_max
            q = torch.round(w / scale).clamp(-q_max - 1, q_max)
            r = q * scale
        return r

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def enable_qat_hooks(model: LM, bits: int) -> list[Callable[[], None]]:
    """Patch Linear forward methods so their weight is fake-quantised before matmul.

    Returns a list of callables that undo the patching.
    """
    undo: list[Callable[[], None]] = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and mod.weight.numel() > 0:
            orig_forward = mod.forward

            def make_forward(m=mod, o=orig_forward, b=bits):
                def fwd(x):
                    w_q = FakeQuantSTE.apply(m.weight, b)
                    return F.linear(x, w_q, m.bias)
                return fwd

            mod.forward = make_forward()
            undo.append(lambda m=mod, o=orig_forward: setattr(m, "forward", o))
    return undo


# =============================================================================
# GPTQ-flavoured post-training quantiser
# =============================================================================

def _role_of(name: str) -> str:
    if "embed" in name:
        return "embed"
    if "mlp.fc" in name:
        return "mlp"
    if "o_proj" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
        # Distinguish attention vs mixer projections by parent class at call-site.
        return "attn"
    if "g_proj" in name:
        return "attn"
    return "other"


def _sigma_for(name: str, cfg: Hyperparameters) -> float:
    # Heuristic: if the parameter lives inside a GatedMixer module it will
    # typically have slightly heavier tails (due to the product-of-gates path),
    # but our gating is bounded so a single role map suffices for naming.
    if "embed" in name:
        return cfg.gptq_sigma_embed
    if "mlp.fc" in name:
        return cfg.gptq_sigma_mlp
    if "mixer" in name and ("q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "g_proj" in name):
        return cfg.gptq_sigma_mixer
    return cfg.gptq_sigma_attn


@torch.no_grad()
def collect_hessian(model: LM, tokens: Tensor, cfg: Hyperparameters, device: torch.device) -> dict[str, Tensor]:
    """Collect a diagonal Hessian estimate (input variance per column) per Linear layer.

    Registers forward hooks that accumulate E[x^2] on the input activation; used
    later to re-scale weights before rounding so that columns which matter
    more on the calibration set get a finer grid.
    """
    hessians: dict[str, Tensor] = {}
    handles = []

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            key = name
            def mk_hook(k=key):
                def hook(m, inp, out):
                    x = inp[0].detach().float()
                    # Accumulate diag(X^T X) / N along last dim.
                    x2 = (x * x).flatten(0, -2).mean(dim=0)
                    if k in hessians:
                        hessians[k] = 0.9 * hessians[k] + 0.1 * x2
                    else:
                        hessians[k] = x2
                return hook
            handles.append(mod.register_forward_hook(mk_hook()))

    model.eval()
    B = 2
    seq = cfg.train_seq_len
    # Use a few short rollouts from the validation tokens.
    rows = cfg.gptq_hessian_rows
    for i in range(rows):
        s = (i * seq) % max(tokens.numel() - seq - 1, 1)
        chunk = tokens[s : s + seq].to(device).long().unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _ = model(chunk)
    for h in handles:
        h.remove()
    return hessians


@torch.no_grad()
def gptq_quantise_tensor(w: Tensor, bits: int, sigma: float, hess_diag: Tensor | None) -> tuple[Tensor, Tensor]:
    """Per-row symmetric integer quantisation with Hessian-aware column rescale.

    Returns (int_weights_int8_storage, scales_fp16).  ``bits`` may be 4..8.
    """
    w = w.float()
    if hess_diag is not None and hess_diag.numel() == w.size(1):
        # Use input-activation RMS as column importance.  Shrinking unused
        # columns pulls outliers away from the quantisation grid boundary.
        imp = hess_diag.clamp_min(1e-8).sqrt()
        imp = imp / imp.mean().clamp_min(1e-8)
        w = w * imp[None, :]
        scales_col = imp
    else:
        scales_col = torch.ones(w.size(1), device=w.device)

    q_max = (1 << (bits - 1)) - 1
    std = w.std(dim=-1, keepdim=True).clamp_min(1e-8)
    amax = w.abs().amax(dim=-1, keepdim=True)
    clip = torch.minimum(amax, sigma * std)
    scale = clip.clamp_min(1e-8) / q_max
    q = torch.round(w / scale).clamp(-q_max - 1, q_max)
    # Undo the column scaling: effective dequant = q * scale / scales_col[None,:].
    # We fold that into a per-row scale and per-column multiplier kept as fp16.
    return q.to(torch.int16), scale.squeeze(-1).to(torch.float16), scales_col.to(torch.float16)


def _pack_ints(q: Tensor, bits: int) -> bytes:
    """Pack signed integer tensor into a tight little-endian bit stream."""
    q = q.detach().cpu().numpy().astype(np.int32)
    unsigned = (q + (1 << (bits - 1))).astype(np.uint64) & ((1 << bits) - 1)
    flat = unsigned.reshape(-1)
    out = bytearray((flat.size * bits + 7) // 8)
    acc = 0
    nbits = 0
    idx = 0
    for v in flat:
        acc |= int(v) << nbits
        nbits += bits
        while nbits >= 8:
            out[idx] = acc & 0xFF
            acc >>= 8
            nbits -= 8
            idx += 1
    if nbits:
        out[idx] = acc & 0xFF
    return bytes(out)


def _unpack_ints(buf: bytes, bits: int, numel: int, shape: tuple) -> Tensor:
    arr = np.zeros(numel, dtype=np.int64)
    acc = 0
    nbits = 0
    bi = 0
    mask = (1 << bits) - 1
    for i in range(numel):
        while nbits < bits and bi < len(buf):
            acc |= buf[bi] << nbits
            nbits += 8
            bi += 1
        v = acc & mask
        acc >>= bits
        nbits -= bits
        arr[i] = int(v) - (1 << (bits - 1))
    return torch.from_numpy(arr.reshape(shape))


# =============================================================================
# Serialisation
# =============================================================================

SUBMISSION_MAGIC = b"PGF1"


def quantise_state_dict(model: LM, cfg: Hyperparameters, hessians: dict[str, Tensor]) -> dict:
    out: dict[str, dict] = {}
    for name, p in model.state_dict().items():
        t = p.detach()
        if t.dtype in (torch.int16, torch.int32, torch.int64, torch.bool):
            out[name] = {"kind": "raw", "dtype": str(t.dtype), "shape": tuple(t.shape),
                         "data": t.cpu().numpy().tobytes()}
            continue
        numel = t.numel()
        if t.ndim < 2 or numel <= cfg.keep_float_numel:
            # Keep small tensors in float16 (controls & scales).
            h = t.float().cpu().to(torch.float16)
            out[name] = {"kind": "fp16", "shape": tuple(t.shape),
                         "data": h.numpy().tobytes()}
            continue
        bits = cfg.gptq_embed_bits if "embed" in name else cfg.gptq_matrix_bits
        sigma = _sigma_for(name, cfg)
        # Linear weights are stored as ``<name>.weight``; pick the matching hessian key.
        hk = name[: -len(".weight")] if name.endswith(".weight") else name
        hess = hessians.get(hk)
        q, scale_row, scale_col = gptq_quantise_tensor(t.float().cpu(), bits, sigma, hess)
        packed = _pack_ints(q, bits)
        out[name] = {
            "kind": "qint",
            "bits": bits,
            "shape": tuple(t.shape),
            "data": packed,
            "scale_row": scale_row.cpu().numpy().tobytes(),
            "scale_col": scale_col.cpu().numpy().tobytes(),
        }
    return out


def serialise(payload: dict) -> bytes:
    raw = pickle.dumps(payload, protocol=5)
    if _HAVE_BROTLI:
        comp = brotli.compress(raw, quality=11)
        return b"BR1" + comp
    else:
        comp = bz2.compress(raw, compresslevel=9)
        return b"BZ1" + comp


def deserialise(buf: bytes) -> dict:
    tag = buf[:3]
    body = buf[3:]
    if tag == b"BR1":
        raw = brotli.decompress(body)
    elif tag == b"BZ1":
        raw = bz2.decompress(body)
    else:
        raise RuntimeError(f"unknown tag {tag!r}")
    return pickle.loads(raw)


def reconstruct_state_dict(q_state: dict, cfg: Hyperparameters) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, entry in q_state.items():
        kind = entry["kind"]
        shape = tuple(entry["shape"])
        if kind == "fp16":
            data = np.frombuffer(entry["data"], dtype=np.float16).reshape(shape)
            out[name] = torch.from_numpy(data.copy()).float()
        elif kind == "raw":
            dt_str = entry["dtype"]
            np_dt = {
                "torch.int16": np.int16,
                "torch.int32": np.int32,
                "torch.int64": np.int64,
                "torch.bool": np.bool_,
            }[dt_str]
            data = np.frombuffer(entry["data"], dtype=np_dt).reshape(shape)
            out[name] = torch.from_numpy(data.copy())
        elif kind == "qint":
            bits = entry["bits"]
            numel = int(np.prod(shape))
            q = _unpack_ints(entry["data"], bits, numel, shape).float()
            sr = torch.from_numpy(np.frombuffer(entry["scale_row"], dtype=np.float16).copy()).float()
            sc = torch.from_numpy(np.frombuffer(entry["scale_col"], dtype=np.float16).copy()).float()
            w = q * sr.view(-1, 1) / sc.view(1, -1).clamp_min(1e-8)
            out[name] = w
        else:
            raise RuntimeError(f"unknown kind {kind}")
    return out


# =============================================================================
# Sliding-window evaluation
# =============================================================================

@torch.no_grad()
def eval_sliding(model: LM, val_tokens: Tensor, base_lut: Tensor, ls_lut: Tensor,
                 bound_lut: Tensor, device: torch.device, cfg: Hyperparameters,
                 rank: int, world_size: int) -> tuple[float, float]:
    model.eval()
    T = val_tokens.numel() - 1
    stride = cfg.eval_stride
    ctx = cfg.eval_ctx
    # Distribute start-of-window positions round-robin across ranks.
    starts = list(range(0, T - ctx, stride))
    if not starts:
        starts = [0]
    mine = starts[rank::world_size]
    loss_sum = 0.0
    token_count = 0
    byte_count = 0
    for s in mine:
        ids = val_tokens[s : s + ctx + 1].to(device).long()
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        # Only score the last `stride` positions — every token is scored exactly once.
        score_start = ctx - stride if s > 0 else 0
        scored_y = y[0, score_start:]
        scored_lp = logp[0, score_start:, :]
        loss_sum += float((-scored_lp.gather(-1, scored_y.unsqueeze(-1)).squeeze(-1)).sum())
        token_count += int(scored_y.numel())
        prev = x[0, score_start:]
        tb = base_lut[scored_y].to(torch.int32)
        tb += (ls_lut[scored_y] & ~bound_lut[prev]).to(torch.int32)
        byte_count += int(tb.sum())

    # Also score tokens that fall outside any stride window (the final tail).
    loss_t = torch.tensor([loss_sum, float(token_count), float(byte_count)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    ls, tc, bc = loss_t.tolist()
    val_loss = ls / max(tc, 1.0)
    bpb = (val_loss / math.log(2.0)) * (tc / max(bc, 1.0))
    model.train()
    return float(val_loss), float(bpb)


# =============================================================================
# Main training loop
# =============================================================================

def main() -> None:
    cfg = Hyperparameters()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    dist_enabled = "RANK" in os.environ
    if dist_enabled:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device("cuda", rank % torch.cuda.device_count())
    else:
        rank = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log0(f"run_id={cfg.run_id} rank={rank}/{world} device={device}")
    log0(f"model dim={cfg.model_dim} layers={cfg.num_layers} vocab={cfg.vocab_size} "
         f"heads={cfg.num_heads}/{cfg.num_kv_heads} mlp={cfg.mlp_mult_num}/{cfg.mlp_mult_den} "
         f"mixer={cfg.mixer_layers_str} chunk={cfg.mixer_chunk}")

    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)
    base_lut, ls_lut, bound_lut = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    local_batch_tokens = cfg.train_batch_tokens // world
    local_batch_seqs = max(local_batch_tokens // cfg.train_seq_len, 1)

    reader = ShardReader(cfg.train_files, rank, world, cfg.train_seq_len, local_batch_seqs)
    val_tokens = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(cfg.val_files))]).contiguous()
    log0(f"val_tokens={val_tokens.numel()} train_batch_tokens={cfg.train_batch_tokens} "
         f"local_batch_seqs={local_batch_seqs}")

    model = LM(cfg).to(device)
    model = model.to(memory_format=torch.channels_last) if False else model  # no-op; placeholder
    param_count = sum(p.numel() for p in model.parameters())
    log0(f"param_count={param_count}")

    if dist_enabled:
        ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=True)
        forward_model = ddp_model
    else:
        forward_model = model

    optimisers, set_lr = build_optimisers(model, cfg)

    ema = ParamEMA(model, decay=cfg.ema_decay)
    swa = SWA()

    start = time.time()
    step = 0
    qat_active = False
    qat_undo: list[Callable[[], None]] = []
    last_log = start

    val_bpb_best = math.inf
    loss_acc = 0.0

    while step < cfg.iterations:
        elapsed = time.time() - start
        if elapsed > cfg.max_wallclock_seconds:
            log0(f"[train] wallclock cap reached at step {step}; stopping")
            break
        if (not qat_active) and step >= cfg.qat_start:
            qat_undo = enable_qat_hooks(model, cfg.qat_bits_matrix)
            qat_active = True
            log0(f"[train] QAT enabled at step {step} bits={cfg.qat_bits_matrix}")

        info = set_lr(step)
        x, y = reader.next_batch(device)
        for opt in optimisers:
            opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = forward_model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimisers:
            opt.step()
        loss_acc += float(loss.detach())

        if step % cfg.ema_interval == 0:
            ema.update(model)

        if step >= cfg.swa_start and ((step - cfg.swa_start) % cfg.swa_interval == 0):
            swa.absorb(model)

        if step % cfg.train_log_every == 0 and step > 0:
            now = time.time()
            avg = loss_acc / cfg.train_log_every
            loss_acc = 0.0
            log0(f"[train] step:{step}/{cfg.iterations} loss:{avg:.4f} "
                 f"lr_mul:{info['lr_mul']:.3f} t:{now-start:.1f}s dt:{(now-last_log)*1000/cfg.train_log_every:.1f}ms")
            last_log = now

        if cfg.val_loss_every > 0 and step > 0 and (step % cfg.val_loss_every == 0):
            vl, vb = eval_sliding(model, val_tokens, base_lut, ls_lut, bound_lut, device, cfg, rank, world)
            if vb < val_bpb_best:
                val_bpb_best = vb
            log0(f"step:{step}/{cfg.iterations} val_loss:{vl:.4f} val_bpb:{vb:.6f} best:{val_bpb_best:.6f} "
                 f"t:{time.time()-start:.1f}s")

        step += 1

    # Undo QAT hooks so post-processing runs on true weights.
    for u in qat_undo:
        u()

    # Merge SWA + EMA: SWA takes priority if we collected at least one sample.
    if swa.avg is not None:
        swa.copy_into(model)
        log0(f"[post] SWA merged ({swa.count} checkpoints)")
    else:
        ema.copy_into(model)
        log0("[post] EMA merged")

    # Pre-quant eval.
    vl, vb = eval_sliding(model, val_tokens, base_lut, ls_lut, bound_lut, device, cfg, rank, world)
    log0(f"pre_quant val_loss:{vl:.6f} val_bpb:{vb:.6f}")

    # Collect Hessians for GPTQ on rank 0 only; broadcast packed payload.
    if _is_rank_zero():
        hessians = collect_hessian(model, val_tokens, cfg, device)
        q_state = quantise_state_dict(model, cfg, hessians)
        payload = {
            "cfg": {
                "vocab_size": cfg.vocab_size,
                "num_layers": cfg.num_layers,
                "model_dim": cfg.model_dim,
                "num_heads": cfg.num_heads,
                "num_kv_heads": cfg.num_kv_heads,
                "mlp_mult_num": cfg.mlp_mult_num,
                "mlp_mult_den": cfg.mlp_mult_den,
                "rope_frac": cfg.rope_frac,
                "rope_base": cfg.rope_base,
                "logit_softcap": cfg.logit_softcap,
                "swa_window": cfg.swa_window,
                "swa_layers": cfg.swa_layers,
                "mixer_layers_str": cfg.mixer_layers_str,
                "mixer_chunk": cfg.mixer_chunk,
                "mixer_state_dim": cfg.mixer_state_dim,
                "mixer_heads": cfg.mixer_heads,
                "tie_embeddings": cfg.tie_embeddings,
                "qk_gain_init": cfg.qk_gain_init,
            },
            "state": q_state,
            "magic": SUBMISSION_MAGIC.decode("latin1"),
        }
        blob = serialise(payload)
        size = len(blob)
        log0(f"[quant] artifact_bytes={size} limit={cfg.artifact_cap_bytes}")
        if size > cfg.artifact_cap_bytes:
            log0("[quant] WARNING: artifact over cap; trying coarser embedding grid")
            # Fallback: drop embedding to 6 bits.
            cfg.gptq_embed_bits = max(6, cfg.gptq_embed_bits - 1)
            q_state = quantise_state_dict(model, cfg, hessians)
            payload["state"] = q_state
            blob = serialise(payload)
            log0(f"[quant] after fallback artifact_bytes={len(blob)}")
        # Round-trip: decode and evaluate.
        payload_rt = deserialise(blob)
        sd = reconstruct_state_dict(payload_rt["state"], cfg)
        # Load into a fresh model.
        reloaded = LM(cfg).to(device)
        missing = reloaded.load_state_dict(sd, strict=False)
        log0(f"[quant] reload_missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")
        vl_q, vb_q = eval_sliding(reloaded, val_tokens, base_lut, ls_lut, bound_lut, device, cfg, rank, world)
        total_bytes = len(blob) + _code_bytes()
        log0(f"final_artifact_only_bytes={len(blob)} code_bytes={_code_bytes()} total_bytes={total_bytes}")
        log0(f"final_int8_zlib_roundtrip val_loss:{vl_q:.4f} val_bpb:{vb_q:.4f} total_bytes:{total_bytes}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{vl_q:.8f} val_bpb:{vb_q:.8f}")

    if dist_enabled:
        dist.barrier()
        dist.destroy_process_group()


def _code_bytes() -> int:
    try:
        return os.path.getsize(__file__)
    except Exception:
        return 0


if __name__ == "__main__":
    main()
