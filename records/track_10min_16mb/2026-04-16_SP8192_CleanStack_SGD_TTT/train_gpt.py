"""
SP8192 + 3-Layer Depth Recurrence (L3-5) + Parallel Residuals (L7+)
+ QK-Gain 5.25 + Score-First SGD TTT + Full Hessian GPTQ SDClip + Brotli-11

Architecture:  11 transformer layers, 512d, 8 heads / 4 KV heads (GQA),
               MLP 4x (2048) with LeakyReLU(0.5)^2, U-Net skip connections,
               XSA on all 11 layers, SmearGate, partial RoPE (16/64 dims),
               LN Scale 1/sqrt(layer+1), tied embeddings, logit softcap=30.
               Depth recurrence on L3-5 (×2, activated at wallclock frac=0.35),
               parallel residuals from L7 (GPT-J style).
Training:      MuonEq-R (row-normalized Muon NS5 + WD=0.095), AdamW
               for embeddings/scalars. EMA(0.9965), warmdown_frac=0.72,
               MLR=0.022.
Quantization:  Full Hessian GPTQ SDClip: int6 matrices (k=12.85σ row std),
               int8 embeddings (k=20.0σ row std). Brotli-11 compression
               (fallback LZMA-9).
Eval:          Sliding-window stride=64 + score-first SGD TTT (3 epochs per
               32K-token chunk, lr=0.005, cosine decay, grad_clip=1.0).
Data:          SP8192 from kevclark/parameter-golf (8192-vocab SentencePiece BPE).

Building on our Apr-8 submission (val_bpb 1.1156) and the merged SOTA
#1493 (val_bpb 1.0810).  Key improvements over Apr-8:
  - SP8192 vocab  (+~0.015 BPB over SP1024)
  - MuonEq-R row normalisation
  - MLP 4x (was 3x)
  - SDClip GPTQ + int8 embeddings + Brotli-11
  - Score-first SGD TTT (replaces LoRA AdamW TTT)
  - Tuned hypers: EMA 0.9965, WD 0.095, MLR 0.022, warmdown 0.72
"""
from __future__ import annotations

import copy
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False

try:
    import brotli as _brotli_mod
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

# ===========================================================================
# Hyperparameters  (all overridable via env vars)
# ===========================================================================

class Hyperparameters:
    # Data
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH",
                                     "./data/tokenizers/fineweb_8192_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    # Training schedule
    val_batch_size       = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every       = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every      = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations           = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps         = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens   = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len        = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len         = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    warmdown_frac        = float(os.environ.get("WARMDOWN_FRAC", 0.72))
    enable_looping_at    = float(os.environ.get("ENABLE_LOOPING_AT", 0.35))
    loop_warmup_steps    = int(os.environ.get("LOOP_WARMUP_STEPS", 20))

    # Model architecture
    vocab_size           = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers           = int(os.environ.get("NUM_LAYERS", 11))
    model_dim            = int(os.environ.get("MODEL_DIM", 512))
    num_heads            = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads         = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult             = float(os.environ.get("MLP_MULT", 4.0))
    tie_embeddings       = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base            = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap        = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init         = float(os.environ.get("QK_GAIN_INIT", 5.25))
    tied_embed_init_std  = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Features
    xsa_last_n           = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims            = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale             = bool(int(os.environ.get("LN_SCALE", "1")))

    # Depth recurrence
    recur_start          = int(os.environ.get("RECUR_START", 3))
    recur_end            = int(os.environ.get("RECUR_END", 5))
    recur_passes         = int(os.environ.get("RECUR_PASSES", 2))

    # Parallel residuals (GPT-J style) from this layer onward
    parallel_resid_start = int(os.environ.get("PARALLEL_RESID_START", 7))

    # Optimizer
    embed_lr             = float(os.environ.get("EMBED_LR", 0.6))
    head_lr              = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr        = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr            = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr            = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum        = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps   = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_row_normalize   = bool(int(os.environ.get("MUON_ROW_NORMALIZE", "1")))
    beta1                = float(os.environ.get("BETA1", 0.9))
    beta2                = float(os.environ.get("BETA2", 0.95))
    adam_eps             = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm       = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd              = float(os.environ.get("MUON_WD", 0.095))
    adam_wd              = float(os.environ.get("ADAM_WD", 0.02))
    embed_wd             = float(os.environ.get("EMBED_WD", 0.085))

    # Weight averaging
    swa_enabled          = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every            = int(os.environ.get("SWA_EVERY", 50))
    late_qat_threshold   = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ema_decay            = float(os.environ.get("EMA_DECAY", 0.9965))
    eval_stride          = int(os.environ.get("EVAL_STRIDE", 64))

    # GPTQ
    gptq_block_size          = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", 64))
    gptq_reserve_seconds     = float(os.environ.get("GPTQ_RESERVE_SECONDS", 12.0))
    matrix_clip_sigmas       = float(os.environ.get("MATRIX_CLIP_SIGMAS", 12.85))
    embed_clip_sigmas        = float(os.environ.get("EMBED_CLIP_SIGMAS", 20.0))

    # TTT (test-time training)
    ttt_enabled          = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_chunk_tokens     = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_lr               = float(os.environ.get("TTT_LR", 0.0005))
    ttt_epochs           = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_momentum         = float(os.environ.get("TTT_MOMENTUM", 0.9))


# ===========================================================================
# Control tensor patterns  (kept at higher precision during quantization)
# ===========================================================================

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,"
        "q_gain,skip_weight,skip_weights,smear,par_gate,recur_gates",
    ).split(",") if p
)

# ===========================================================================
# Batched Newton-Schulz orthogonalisation
# ===========================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X

# ===========================================================================
# Parallel MuonEq-R: row-normalised Muon with parameter banking
# ===========================================================================

class Muon(torch.optim.Optimizer):
    """Parallel MuonEq-R: reduce-scatter → row-norm → NS5 → all-gather."""

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 row_normalize: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      nesterov=nesterov,
                                      weight_decay=weight_decay,
                                      row_normalize=row_normalize))
        self._built = False

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p, 'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self._built:
            self._build()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            row_norm = group.get("row_normalize", True)

            prev_ag_handle = None
            prev_m = None
            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue
                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                # MuonEq-R: row normalise before NS5
                if row_norm and g.ndim >= 2:
                    row_norms = g.reshape(g.shape[0], -1).norm(dim=1)
                    row_norms = row_norms.view(g.shape[0], *([1] * (g.ndim - 1))).clamp_min(1e-7)
                    g = g / row_norms

                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])
            if hasattr(self, '_rs_futures'):
                del self._rs_futures
        return loss

# ===========================================================================
# Tokenizer BPB helpers
# ===========================================================================

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    base_bytes = np.zeros(sz, dtype=np.int16)
    has_space = np.zeros(sz, dtype=np.bool_)
    is_boundary = np.ones(sz, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files for: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Val split too short for seq_len={seq_len}")
    return tokens[:usable + 1]

# ===========================================================================
# Data loading
# ===========================================================================

def load_data_shard(file):
    hdr_bytes = 256 * np.dtype("<i4").itemsize
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or int(hdr[0]) != 20240520 or int(hdr[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(hdr[2])
    if file.stat().st_size != hdr_bytes + n * np.dtype("<u2").itemsize:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens = np.fromfile(file, dtype="<u2", count=n, offset=hdr_bytes)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No train files for: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(left, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, accum):
        local = global_tokens // (self.world_size * accum)
        span = local + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        t = chunk[start:start + span].to(dtype=torch.int64)
        x, y = t[:-1].reshape(-1, seq_len), t[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ===========================================================================
# Transformer modules
# ===========================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) *
                       scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) \
               and p.dtype != torch.float32:
                p.data = p.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2,
                                                 dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                s = seq_len / self.train_seq_len
                new_base = self.base * (s ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (
                    torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        h = rope_dims // 2
        x1, x2 = x_rope[..., :h], x_rope[..., h:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x, q_w, k_w, v_w, out_w):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                try:
                    y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True)
                except TypeError:
                    reps = self.num_heads // self.num_kv_heads
                    kt = kt.repeat_interleave(reps, dim=1)
                    vt = vt.repeat_interleave(reps, dim=1)
                    y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            y = y.transpose(1, 2)

        if self.use_xsa:
            y = self._xsa_efficient(y, v)

        return F.linear(y.reshape(bsz, seqlen, dim), out_w.to(x.dtype))


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class MLP(nn.Module):
    def forward(self, x, up_w, down_w):
        return F.linear(F.leaky_relu(F.linear(x, up_w.to(x.dtype)), 0.5).square(),
                        down_w.to(x.dtype))


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 layer_idx=0, ln_scale=False, parallel_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP()
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel_residual = parallel_residual
        if parallel_residual:
            self.par_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed, q_w, k_w, v_w, out_w)

        if self.parallel_residual:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor, up_w, down_w)
            g = self.par_gate.to(dtype=x.dtype)
            combined = (g * self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
                        + (1 - g) * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out)
            return x_in + combined
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, xsa_last_n=0, rope_dims=0, ln_scale=False,
                 recur_start=3, recur_end=5, recur_passes=2,
                 parallel_resid_start=7):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.recur_start = recur_start
        self.recur_end = recur_end
        self.recur_passes = recur_passes
        self.loop_enabled = False  # activated mid-training at frac=0.35

        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear = SmearGate(model_dim)

        # U-Net skip connections
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim,
                                                     dtype=torch.float32))

        # Depth recurrence: per-layer blend gates (init=0 → identity at start)
        num_recur_layers = max(0, recur_end - recur_start + 1)
        self.recur_gates = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            for _ in range(num_recur_layers)
        ]) if num_recur_layers > 0 and recur_passes > 1 else nn.ParameterList()

        # Parameter banks
        self.qo_bank    = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank    = nn.Parameter(torch.empty(2 * num_layers, kv_dim,   model_dim))
        self.mlp_up_bank   = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale,
                  parallel_residual=(i >= parallel_resid_start))
            for i in range(num_layers)
        ])

        # Partial RoPE
        if rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024,
                                           rope_dims=rope_dims)

        # XSA on last xsa_last_n layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head = (None if tie_embeddings
                        else CastedLinear(model_dim, vocab_size, bias=False))
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i], gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i], gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif (module.weight.ndim == 2
                      and module.weight.shape[0] >= 64
                      and module.weight.shape[1] >= 64):
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _run_block(self, i, x, x0):
        n = self.num_layers
        return self.blocks[i](x, x0,
                               self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
                               self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i])

    def _forward_body(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            if (self.loop_enabled and self.recur_start <= i <= self.recur_end
                    and self.recur_passes > 1):
                recur_idx = i - self.recur_start
                gate = (self.recur_gates[recur_idx].to(dtype=x.dtype)
                        if recur_idx < len(self.recur_gates) else torch.zeros(1, device=x.device))
                for pass_idx in range(self.recur_passes):
                    x_new = self._run_block(i, x, x0)
                    x = gate * x_new + (1.0 - gate) * x if pass_idx > 0 else x_new
            else:
                x = self._run_block(i, x, x0)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            if (self.loop_enabled and self.recur_start <= bi <= self.recur_end
                    and self.recur_passes > 1):
                recur_idx = bi - self.recur_start
                gate = (self.recur_gates[recur_idx].to(dtype=x.dtype)
                        if recur_idx < len(self.recur_gates) else torch.zeros(1, device=x.device))
                for pass_idx in range(self.recur_passes):
                    x_new = self._run_block(bi, x, x0)
                    x = gate * x_new + (1.0 - gate) * x if pass_idx > 0 else x_new
            else:
                x = self._run_block(bi, x, x0)

        return self.final_norm(x)

    def forward(self, input_ids, target_ids):
        x = self._forward_body(input_ids)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids):
        x = self._forward_body(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# ===========================================================================
# Standard evaluation (non-overlapping, used during training)
# ===========================================================================

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
             eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_seqs = max((args.val_batch_size // (world_size * grad_accum_steps)) // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size
    loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt    = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt   = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be = min(bs + local_seqs, s1)
            raw_s, raw_e = bs * seq_len, be * seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            n = float(y.numel())
            loss_sum += bl.to(torch.float64) * n
            tok_cnt  += n
            prev, tgt = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int16)
            byte_cnt += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_cnt).item()
    bpt = vl / math.log(2.0)
    tpb = tok_cnt.item() / byte_cnt.item()
    model.train()
    return vl, float(bpt * tpb)

# ===========================================================================
# Sliding window evaluation
# ===========================================================================

def eval_val_sliding(args, model, rank, world_size, device,
                     val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                     stride=64, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum    += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    return val_loss, float(bpt * tpb)

# ===========================================================================
# Score-First SGD TTT Evaluation  (compliant with Issue #1017 Track B)
#
# Conditions satisfied:
#   1. Causality:    sliding window eval is strictly causal
#   2. Normalised:   standard softmax over full vocab, no biasing
#   3. Score-first:  chunk fully scored under no_grad BEFORE any SGD update
#   4. Single-pass:  each token scored exactly once, left-to-right
# ===========================================================================

def eval_val_ttt_sgd(args, base_model, rank, world_size, device,
                     val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                     eval_seq_len=None):
    seq_len       = eval_seq_len or args.train_seq_len
    chunk_tokens  = args.ttt_chunk_tokens
    ttt_lr        = args.ttt_lr
    ttt_epochs    = args.ttt_epochs
    ttt_momentum  = args.ttt_momentum

    total_tokens  = val_tokens.numel() - 1
    rank_start    = (total_tokens * rank) // world_size
    rank_end      = (total_tokens * (rank + 1)) // world_size
    rank_len      = rank_end - rank_start

    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
    optimizer = torch.optim.SGD(base_model.parameters(), lr=ttt_lr, momentum=ttt_momentum)

    num_chunks = max(1, (rank_len + chunk_tokens - 1) // chunk_tokens)

    for chunk_idx in range(num_chunks):
        chunk_start = rank_start + chunk_idx * chunk_tokens
        chunk_end   = min(chunk_start + chunk_tokens, rank_end)
        if chunk_end <= chunk_start:
            break

        # Context: up to seq_len tokens before chunk for long-range attention
        ctx_start = max(rank_start, chunk_start - max(0, seq_len - chunk_tokens))
        x_chunk = val_tokens[ctx_start:chunk_end].to(dtype=torch.int64, device=device).unsqueeze(0)
        y_chunk = val_tokens[ctx_start + 1:chunk_end + 1].to(dtype=torch.int64, device=device).unsqueeze(0)
        score_offset = chunk_start - ctx_start  # new tokens start here in the window

        # === PHASE 1: Score (strictly no model-state change) ===
        base_model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_chunk)
            scored_logits  = logits[0, score_offset:].float()
            scored_targets = y_chunk[0, score_offset:]
            nll = F.cross_entropy(scored_logits, scored_targets, reduction="none")
            loss_sum    += nll.to(torch.float64).sum()
            token_count += float(chunk_end - chunk_start)
            tgt  = scored_targets
            prev = x_chunk[0, score_offset:]
            tb   = base_bytes_lut[tgt].to(torch.float64)
            tb  += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.float64)
            byte_count += tb.sum()

        # === PHASE 2: Adapt (cosine-decayed SGD for ttt_epochs) ===
        cos_frac = 0.5 * (1.0 + math.cos(math.pi * chunk_idx / max(num_chunks, 1)))
        for pg in optimizer.param_groups:
            pg['lr'] = ttt_lr * cos_frac

        # Disable QAT during adapt: GPTQ weights are already quantized, enabling
        # fake-quant (CastedLinear._qat_enabled=True from training) would double-
        # quantize them and corrupt the gradients.
        _saved_qat = CastedLinear._qat_enabled
        CastedLinear._qat_enabled = False
        base_model.train()
        for _ in range(ttt_epochs):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                adapt_loss = base_model(x_chunk, y_chunk)
            adapt_loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            if dist.is_available() and dist.is_initialized():
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        CastedLinear._qat_enabled = _saved_qat

    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, float(bpt * tpb)

# ===========================================================================
# AR self-generated calibration data for GPTQ
# ===========================================================================

def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                   vocab_size=8192, temperature=0.8, batch_size=8, seed=42):
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i + 1])
    return all_tokens

# ===========================================================================
# GPTQ SDClip quantisation helpers
# ===========================================================================

def _sdclip_int6(t, k: float) -> tuple:
    """Per-row int6 quantisation with SDClip: scale = k * std(row) / 31."""
    t32 = t.float()
    if t32.ndim == 2:
        clip = (k * t32.std(dim=1)).clamp_min(1.0 / 31.0)
        s = (clip / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -31, 31).to(torch.int8)
        return q, s
    amax = t32.abs().max().item()
    s = torch.tensor(max(amax / 31.0, 1.0 / 31.0), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()), -31, 31).to(torch.int8)
    return q, s


def _sdclip_int8(t, k: float) -> tuple:
    """Per-row int8 quantisation with SDClip: scale = k * std(row) / 127."""
    t32 = t.float()
    if t32.ndim == 2:
        clip = (k * t32.std(dim=1)).clamp_min(1.0 / 127.0)
        s = (clip / 127.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / s.float()[:, None]), -127, 127).to(torch.int8)
        return q, s
    amax = t32.abs().max().item()
    s = torch.tensor(max(amax / 127.0, 1.0 / 127.0), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / s.float()), -127, 127).to(torch.int8)
    return q, s


def quantize_gptq_sdclip(weight, hessian, k: float, block_size: int = 128) -> tuple:
    """Full Hessian-aware GPTQ with SDClip clipping (int6, range ±31)."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _sdclip_int6(t32, k)

    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H.diagonal().add_(damp)
    perm     = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]

    Hinv = None
    for damp_scale in [1.0, 10.0, 100.0, 1000.0]:
        try:
            H_try = H.clone()
            if damp_scale > 1.0:
                H_try.diagonal().add_(damp * damp_scale)
            L = torch.linalg.cholesky(H_try)
            Hinv_full = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        return _sdclip_int6(t32, k)

    # SDClip scale: k * std(row) on the ORIGINAL weight
    clip = (k * t32.std(dim=1)).clamp_min(1.0 / 31.0)
    s  = (clip / 31.0).to(torch.float16)
    sf = s.float()

    Q = torch.zeros_like(W, dtype=torch.int8)
    W_work = W.clone()
    for i1 in range(0, cols, block_size):
        i2    = min(i1 + block_size, cols)
        count = i2 - i1
        W1    = W_work[:, i1:i2].clone()
        Q1    = torch.zeros(rows, count, dtype=torch.int8)
        Err1  = torch.zeros(rows, count)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for ci in range(count):
            w = W1[:, ci]
            d = Hinv1[ci, ci]
            q = torch.clamp(torch.round(w / sf), -31, 31).to(torch.int8)
            Q1[:, ci] = q
            err = (w - q.float() * sf) / d
            W1[:, ci:] -= err.unsqueeze(1) * Hinv1[ci, ci:].unsqueeze(0)
            Err1[:, ci] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Q = Q[:, inv_perm]
    return Q, s


# ===========================================================================
# Unbank / Rebank  state dicts
# ===========================================================================

def _unbank_state_dict(sd, num_layers):
    out = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"]   = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"]   = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"]   = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"]   = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"]      = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"]    = tensor[i]
        else:
            out[name] = tensor
    return out


def _rebank_state_dict(sd, num_layers, template_sd):
    n = num_layers
    qo   = [None] * (2 * n)
    kv   = [None] * (2 * n)
    up   = [None] * n
    down = [None] * n
    consumed = set()
    for i in range(n):
        for key, lst, idx in [
            (f"blocks.{i}.attn.c_q.weight",  qo,   i),
            (f"blocks.{i}.attn.proj.weight",  qo,   n + i),
            (f"blocks.{i}.attn.c_k.weight",  kv,   i),
            (f"blocks.{i}.attn.c_v.weight",  kv,   n + i),
            (f"blocks.{i}.mlp.fc.weight",    up,   i),
            (f"blocks.{i}.mlp.proj.weight",  down, i),
        ]:
            if key in sd:
                lst[idx] = sd[key]
                consumed.add(key)
    out = {
        "qo_bank":       torch.stack(qo).to(dtype=template_sd["qo_bank"].dtype),
        "kv_bank":       torch.stack(kv).to(dtype=template_sd["kv_bank"].dtype),
        "mlp_up_bank":   torch.stack(up).to(dtype=template_sd["mlp_up_bank"].dtype),
        "mlp_down_bank": torch.stack(down).to(dtype=template_sd["mlp_down_bank"].dtype),
    }
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out

# ===========================================================================
# Non-banked model for Hessian collection
# ===========================================================================

class _HAttn(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim,    dim,    bias=False)
        self.c_k   = CastedLinear(dim,    kv_dim, bias=False)
        self.c_v   = CastedLinear(dim,    kv_dim, bias=False)
        self.proj  = CastedLinear(dim,    dim,    bias=False)
        self.q_gain   = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary    = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa   = False

    def _xsa(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(-1, keepdim=True) * vn).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                try:
                    y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=True)
                except TypeError:
                    reps = self.num_heads // self.num_kv_heads
                    kt = kt.repeat_interleave(reps, dim=1)
                    vt = vt.repeat_interleave(reps, dim=1)
                    y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            else:
                y = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
            y = y.transpose(1, 2)
        if self.use_xsa:
            y = self._xsa(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


class _HMLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc   = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


class _HBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 layer_idx=0, ln_scale=False, parallel_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = _HAttn(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp  = _HMLP(dim, 4.0)           # always MLP 4x for Hessian model
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel_residual = parallel_residual
        if parallel_residual:
            self.par_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, x0):
        mix  = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed   = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed)
        if self.parallel_residual:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            g   = self.par_gate.to(dtype=x.dtype)
            out = g * self.attn_scale.to(x.dtype)[None, None, :] * attn_out \
                + (1 - g) * self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
            return x_in + out
        x_out  = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor)


class _HGPT(nn.Module):
    """Non-banked GPT for Hessian collection (no depth recurrence)."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init,
                 xsa_last_n=0, rope_dims=0, ln_scale=False, parallel_resid_start=7):
        super().__init__()
        self.tie_embeddings  = tie_embeddings
        self.logit_softcap   = logit_softcap
        self.num_layers      = num_layers
        self.tok_emb         = nn.Embedding(vocab_size, model_dim)
        self.smear           = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights    = nn.Parameter(torch.ones(self.num_skip_weights, model_dim,
                                                        dtype=torch.float32))
        self.blocks = nn.ModuleList([
            _HBlock(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                    layer_idx=i, ln_scale=ln_scale,
                    parallel_residual=(i >= parallel_resid_start))
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(model_dim // num_heads, base=rope_base,
                                           train_seq_len=1024, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self.final_norm = RMSNorm()
        self.lm_head = (None if tie_embeddings
                        else CastedLinear(model_dim, vocab_size, bias=False))

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0)
        x = self.final_norm(x)
        x_flat  = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# ===========================================================================
# Hessian collection + mixed quantisation
# ===========================================================================

def collect_hessians(hessian_model, token_seqs, device):
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            pname = name + ".weight"
            cols = module.weight.shape[1]
            hessians[pname] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def _make_hook(pn):
                def _hook(mod, inp, outp):
                    x = inp[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pn] += (x.T @ x).cpu()
                return _hook
            hooks.append(module.register_forward_hook(_make_hook(pname)))
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    n_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= n_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians


def mixed_quantize(state_dict, hessians, matrix_k: float, embed_k: float):
    """Quantise: int6 (SDClip, GPTQ Hessian) for attn/mlp, int8 (SDClip) for embeddings."""
    result = {}
    meta   = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65_536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name]   = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name]   = "passthrough_ctrl"
            continue
        # tok_emb → int8 SDClip
        if "tok_emb" in name or "lm_head" in name:
            q, s = _sdclip_int8(t, embed_k)
            result[name + ".q"] = q
            result[name + ".s"] = s
            meta[name] = {"type": "int8"}
        # attention / mlp matrices → int6 GPTQ SDClip
        elif ((".attn." in name or ".mlp." in name)
              and t.ndim == 2 and t.shape[0] >= 64 and t.shape[1] >= 64):
            H = hessians.get(name)
            q, s = quantize_gptq_sdclip(t, H, matrix_k)
            result[name + ".q"] = q
            result[name + ".s"] = s
            meta[name] = {"type": "int6"}
        else:
            result[name] = t.to(torch.float16)
            meta[name]   = "passthrough"
    return result, meta


def dequantize(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if isinstance(info, str) and info.startswith("passthrough"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q = result[name + ".q"]
        s = result[name + ".s"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], 1)).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# ===========================================================================
# Compression helpers
# ===========================================================================

_BROTLI_MAGIC = b"BROTLI"

def compress_weights(raw: bytes) -> bytes:
    if _HAS_BROTLI:
        return _BROTLI_MAGIC + _brotli_mod.compress(raw, quality=11)
    return lzma.compress(raw, preset=9)


def decompress_weights(data: bytes) -> bytes:
    if data[:6] == _BROTLI_MAGIC:
        import brotli
        return brotli.decompress(data[6:])
    return lzma.decompress(data)

# ===========================================================================
# Main training loop
# ===========================================================================

def _make_model(args):
    return GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        recur_start=args.recur_start, recur_end=args.recur_end,
        recur_passes=args.recur_passes, parallel_resid_start=args.parallel_resid_start,
    )


def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    # grad_accum keeps effective batch = 8-GPU-equivalent regardless of how many GPUs
    grad_accum_steps = max(1, 8 // world_size) if 8 % world_size == 0 else 1
    grad_scale       = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = (rank == 0)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg, console=True):
        if not master:
            return
        if console:
            print(msg, flush=True)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}  PyTorch {torch.__version__}  "
         f"FA3={_HAS_FA3}  BROTLI={_HAS_BROTLI}", console=False)
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                        check=False).stdout, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Hyperparameter log
    hparams = {k: v for k, v in vars(type(args)).items()
               if not k.startswith("_") and not callable(v)}
    hparams["distributed"] = distributed
    hparams["world_size"]  = world_size
    hparams["rank"]        = rank
    hparams["local_rank"]  = local_rank
    hparams["brotli"]      = _HAS_BROTLI
    hparams["fa3"]         = _HAS_FA3
    hparams["train_files"] = args.train_files
    hparams["val_files"]   = args.val_files
    log0("Hyperparameters:")
    for k, v in sorted(hparams.items()):
        log0(f"  {k}: {v}")

    sp     = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    sp_vs  = int(sp.vocab_size())
    if sp_vs != args.vocab_size:
        raise ValueError(f"Vocab mismatch: expected {args.vocab_size}, got {sp_vs}")

    val_seq_len = max(args.train_seq_len, args.eval_seq_len)
    val_tokens  = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_space_lut, is_boundary_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel() - 1}  FA3:{_HAS_FA3}  BROTLI:{_HAS_BROTLI}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    CastedLinear._qat_enabled = False
    base_model = _make_model(args).to(device).bfloat16()
    # Keep banks in fp32 (Muon needs fp32 for NS5 stability)
    for attr in ("qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"):
        getattr(base_model, attr).data = getattr(base_model, attr).data.float()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    par_layers = [i for i, b in enumerate(base_model.blocks) if b.parallel_residual]
    log0(f"model_params:{n_params}")
    log0(f"XSA:{xsa_layers}  parallel_resid:{par_layers}")
    log0(f"recur:[{args.recur_start}-{args.recur_end}]x{args.recur_passes}  "
         f"qk_gain:{args.qk_gain_init}")

    # ------------------------------------------------------------------
    # Optimisers
    # ------------------------------------------------------------------
    matrix_params = [base_model.qo_bank, base_model.kv_bank,
                     base_model.mlp_up_bank, base_model.mlp_down_bank]
    block_np = list(base_model.blocks.named_parameters())
    scalar_params = [p for n, p in block_np
                     if (p.ndim < 2 or any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS))]
    scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    for rg in base_model.recur_gates:
        scalar_params.append(rg)

    token_lr  = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]

    opt_tok    = torch.optim.AdamW(tok_groups, betas=(args.beta1, args.beta2),
                                   eps=args.adam_eps, weight_decay=args.embed_wd, fused=True)
    opt_muon   = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                      backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
                      row_normalize=args.muon_row_normalize)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    opt_head = None
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(opt_head)

    # Params that need all-reduce (not handled by Muon's reduce-scatter)
    replicated_params = list(tok_groups[0]["params"]) + scalar_params
    if base_model.lm_head is not None:
        replicated_params.append(base_model.lm_head.weight)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    ema_decay = args.ema_decay

    # ------------------------------------------------------------------
    # LR schedule
    # ------------------------------------------------------------------
    max_wall_ms = (1000.0 * args.max_wallclock_seconds
                   if args.max_wallclock_seconds > 0 else None)
    gptq_reserve_ms = 1000.0 * args.gptq_reserve_seconds
    effective_wall_ms = ((max_wall_ms - gptq_reserve_ms)
                         if max_wall_ms is not None else None)

    def lr_mul(elapsed_ms: float) -> float:
        if args.warmdown_frac <= 0 or effective_wall_ms is None:
            return 1.0
        wd_start = effective_wall_ms * (1.0 - args.warmdown_frac)
        if elapsed_ms < wd_start:
            return 1.0
        remaining = max(effective_wall_ms - elapsed_ms, 0.0)
        return remaining / max(effective_wall_ms * args.warmdown_frac, 1e-9)

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Initial warmup (primes compilation, state is restored after)
    # ------------------------------------------------------------------
    if args.warmup_steps > 0:
        log0(f"warmup: priming compilation for {args.warmup_steps} steps")
        init_sd  = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for _ in range(grad_accum_steps):
                x, y = DistributedTokenLoader(
                    args.train_files, rank, world_size, device
                ).next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            if distributed:
                for p in replicated_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for o in optimizers:
                o.step()
            zero_grad()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup_step: {ws + 1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt):
            o.load_state_dict(s)
        zero_grad()
        ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    # ------------------------------------------------------------------
    # Training data loader
    # ------------------------------------------------------------------
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    swa_state, swa_count = None, 0
    looping_enabled = False
    stop_step = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_ms = 0.0
    step = 0

    model.train()
    log0(f"training:start world:{world_size} accum:{grad_accum_steps} "
         f"batch:{args.train_batch_tokens} seq:{args.train_seq_len} "
         f"warmdown_frac:{args.warmdown_frac}")

    while True:
        last   = step == args.iterations or (stop_step is not None and step >= stop_step)
        do_val = last or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if do_val:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
            log0(f"{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{train_ms/1000:.1f}s")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{train_ms:.0f}ms "
                     f"step:{step}/{args.iterations}")
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale   = lr_mul(elapsed)

        # Late QAT
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold \
                and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        # Depth recurrence warmup: enable looping at target wallclock fraction
        if (not looping_enabled and args.recur_passes > 1 and effective_wall_ms is not None):
            elapsed_frac = elapsed / effective_wall_ms
            if elapsed_frac >= args.enable_looping_at:
                looping_enabled = True
                base_model.loop_enabled = True
                # Loop warmup: prime the NEW computation graph, then restore state
                log0(f"loop_warmup:enabled frac:{elapsed_frac:.3f} step:{step}")
                loop_save_sd  = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                loop_save_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
                loop_ema_save = {k: v.clone() for k, v in ema_state.items()}
                # Re-use the existing compiled_model; loop_enabled guard triggers recompile
                # on the first forward pass - no need for a separate torch.compile call.
                model.train()
                lu_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
                for lws in range(args.loop_warmup_steps):
                    zero_grad()
                    for _ in range(grad_accum_steps):
                        xl, yl = lu_loader.next_batch(
                            args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            wl = model(xl, yl)
                        (wl * grad_scale).backward()
                    if distributed:
                        for p in replicated_params:
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    for o in optimizers:
                        o.step()
                    zero_grad()
                    if (lws + 1) % 10 == 0 or lws + 1 == args.loop_warmup_steps:
                        log0(f"loop_warmup_step: {lws + 1}/{args.loop_warmup_steps}")
                # Restore checkpoint (warmup was just to prime compilation)
                base_model.load_state_dict(loop_save_sd, strict=True)
                base_model.loop_enabled = True
                for o, s in zip(optimizers, loop_save_opt):
                    o.load_state_dict(s)
                ema_state = loop_ema_save
                zero_grad()
                log0(f"layer_loop:enabled step:{step} frac:{elapsed_frac:.3f} "
                     f"encoder:[0,1,2,{args.recur_start},{args.recur_start+1},"
                     f"{args.recur_end},{args.recur_start},{args.recur_start+1}] "
                     f"decoder:[{args.recur_end},{args.recur_start},{args.recur_start+1},"
                     f"{args.recur_end},{args.recur_end+1},{args.recur_end+2},"
                     f"{args.recur_end+3},{args.recur_end+4},{args.recur_end+5}]")
                # Preserve elapsed time before resetting interval timer
                train_ms += 1000.0 * (time.perf_counter() - t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()

        # Gradient step
        zero_grad()
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len,
                                           grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = (min(step / args.muon_momentum_warmup_steps, 1.0)
                if args.muon_momentum_warmup_steps > 0 else 1.0)
        muon_mom = ((1 - frac) * args.muon_momentum_warmup_start
                    + frac * args.muon_momentum)
        for g in opt_muon.param_groups:
            g["momentum"] = muon_mom

        # LR scheduling
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # Overlapped Muon + Adam steps
        opt_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        opt_tok.step()
        opt_scalar.step()
        if opt_head is not None:
            opt_head.step()
        opt_muon.step()
        zero_grad()

        # EMA update
        with torch.no_grad():
            for n, t in base_model.state_dict().items():
                ema_state[n].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)

        # Tight SWA
        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items():
                    swa_state[n] += t.detach().cpu()
                swa_count += 1

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms/1000:.1f}s avg:{approx_ms/step:.2f}ms/step "
                 f"tok/s:{int(args.train_batch_tokens * 1000 / (approx_ms / step))}"
                 f"{' QAT' if CastedLinear._qat_enabled else ''}")

        reached = max_wall_ms is not None and approx_ms >= max_wall_ms - gptq_reserve_ms
        if distributed and max_wall_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_step is None and reached:
            stop_step = step

    log0(f"peak memory allocated:{torch.cuda.max_memory_allocated()//1024//1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    # ------------------------------------------------------------------
    # Apply EMA weights
    # ------------------------------------------------------------------
    log0("ema:applying EMA weights")
    current_sd = base_model.state_dict()
    avg_state  = {n: t.to(dtype=current_sd[n].dtype) for n, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # Optional: blend in SWA
    if swa_state is not None and swa_count > 0:
        swa_avg = {n: (t / swa_count).to(device=avg_state[n].device, dtype=current_sd[n].dtype)
                   for n, t in swa_state.items()}
        for n in avg_state:
            if n in swa_avg:
                avg_state[n] = 0.5 * avg_state[n] + 0.5 * swa_avg[n]
        base_model.load_state_dict(avg_state, strict=True)
        log0(f"swa:applied swa_count={swa_count}")

    # ------------------------------------------------------------------
    # Pre-GPTQ diagnostic eval
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_vl, diag_vb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps,
                                 val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
    torch.cuda.synchronize()
    log0(f"pre-quantization post-ema val_loss:{diag_vl:.8f} val_bpb:{diag_vb:.8f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_diag):.0f}ms")

    # ------------------------------------------------------------------
    # GPTQ: Hessian collection + quantisation
    # ------------------------------------------------------------------
    full_sd   = base_model.state_dict()
    sd_cpu    = {k: v.detach().cpu() for k, v in full_sd.items()}
    unbanked  = _unbank_state_dict(sd_cpu, args.num_layers)

    log0("GPTQ:building non-banked model for Hessian collection...")
    hm = _HGPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        parallel_resid_start=args.parallel_resid_start,
    ).to(device).bfloat16()
    for m in hm.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hm)
    missing, unexpected = hm.load_state_dict(
        {k: v.to(device) for k, v in unbanked.items() if k in hm.state_dict()}, strict=False)
    log0(f"GPTQ:hessian model loaded  missing={len(missing)} unexpected={len(unexpected)}")

    log0(f"GPTQ:collecting {args.gptq_calibration_batches} Hessians "
         f"from AR-generated calibration data...")
    t_gen = time.perf_counter()
    ar_tokens = generate_autoregressive_calib(
        base_model, device, num_seqs=args.gptq_calibration_batches,
        seq_len=args.train_seq_len, vocab_size=args.vocab_size,
        temperature=0.8, batch_size=8, seed=args.seed)
    log0(f"GPTQ:generated {len(ar_tokens)} seqs in {time.perf_counter()-t_gen:.1f}s")

    log0("GPTQ:collecting Hessians...")
    t_h = time.perf_counter()
    hessians = collect_hessians(hm, ar_tokens, device)
    log0(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t_h:.1f}s")
    del ar_tokens, hm
    torch.cuda.empty_cache()

    log0(f"GPTQ:quantising (matrix_k={args.matrix_clip_sigmas} embed_k={args.embed_clip_sigmas})")
    quant_result, quant_meta = mixed_quantize(
        unbanked, hessians, args.matrix_clip_sigmas, args.embed_clip_sigmas)

    # Report what was quantised
    int6_names = [n for n, m in quant_meta.items() if isinstance(m, dict) and m["type"] == "int6"]
    int8_names = [n for n, m in quant_meta.items() if isinstance(m, dict) and m["type"] == "int8"]
    log0(f"Quantized weights:")
    log0(f"  gptq (int6): {', '.join(int6_names[:6])}{'...' if len(int6_names) > 6 else ''}")
    log0(f"  gptq (int8): {', '.join(int8_names)}")
    log0(f"  passthrough: "
         f"{sum(1 for m in quant_meta.values() if isinstance(m,str) and 'passthrough' in m)} tensors")

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, buf)
    quant_raw  = buf.getvalue()
    quant_blob = compress_weights(quant_raw)

    model_bytes = len(quant_blob)
    code_bytes  = len(code.encode("utf-8"))
    log0(f"Serialized model quantized+{'brotli' if _HAS_BROTLI else 'lzma'}: {model_bytes} bytes")
    log0(f"Code size: {code_bytes} bytes")
    total_bytes = model_bytes + code_bytes
    log0(f"Total submission size quantized+compressed: {total_bytes} bytes  "
         f"{'PASS' if total_bytes <= 16_000_000 else 'FAIL: OVER 16MB!'}")

    if master:
        with open("final_model.ptz", "wb") as f:
            f.write(quant_blob)

    # ------------------------------------------------------------------
    # Roundtrip + full evaluation
    # ------------------------------------------------------------------
    if distributed:
        dist.barrier()

    with open("final_model.ptz", "rb") as f:
        blob_disk = f.read()
    raw_disk = decompress_weights(blob_disk)
    quant_state = torch.load(io.BytesIO(raw_disk), map_location="cpu", weights_only=False)

    deq_unbanked = dequantize(quant_state["w"], quant_state["m"], unbanked)
    deq_state    = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    eval_model = _make_model(args).to(device).bfloat16()
    eval_model.loop_enabled = looping_enabled
    for attr in ("qo_bank", "kv_bank", "mlp_up_bank", "mlp_down_bank"):
        getattr(eval_model, attr).data = getattr(eval_model, attr).data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # Standard roundtrip
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_vl, q_vb = eval_val(args, eval_model, rank, world_size, device, grad_accum_steps,
                           val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                           eval_seq_len=args.eval_seq_len)
    torch.cuda.synchronize()
    log0(f"quantized val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")

    # Sliding window
    if args.eval_stride > 0 and args.eval_stride < args.eval_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_vl, sw_vb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
            stride=args.eval_stride, eval_seq_len=args.eval_seq_len)
        torch.cuda.synchronize()
        log0(f"quantized_sliding_window val_loss:{sw_vl:.8f} val_bpb:{sw_vb:.8f} "
             f"eval_time:{1000.0*(time.perf_counter()-t_slide):.0f}ms")

    # Score-First SGD TTT
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        log0(f"ttt:start chunks=~{val_tokens.numel() // args.ttt_chunk_tokens} "
             f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs}")
        ttt_vl, ttt_vb = eval_val_ttt_sgd(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
            eval_seq_len=args.eval_seq_len)
        torch.cuda.synchronize()
        log0(f"quantized_ttt val_loss:{ttt_vl:.8f} val_bpb:{ttt_vb:.8f} "
             f"eval_time:{1000.0*(time.perf_counter()-t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
