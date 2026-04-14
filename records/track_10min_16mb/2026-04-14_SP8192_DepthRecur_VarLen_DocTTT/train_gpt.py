"""
SP8192 + Depth Recurrence + VarLen Attention + Doc-LoRA TTT + Fused MLP
+ Parallel Residuals + BigramHash + XSA + AdamW-TTT

Architecture:  11 transformer layers, 512d, 8 heads / 4 KV heads (GQA),
               MLP 3x (1536) with LeakyReLU(0.5)^2, U-Net skip connections,
               BigramHash 3072x112, XSA on all 11 layers, SmearGate,
               Partial RoPE (16/64 dims), LN Scale 1/sqrt(layer+1),
               Value Embedding VE128 (layers 9-10), tied embeddings,
               Depth recurrence on layers 3-5 (2 passes),
               Parallel residuals on layers 7+,
               QK-Gain 5.25 for improved attention scaling.
Training:      Parallel Muon with Parameter Banking, EMA(0.997) + Tight SWA,
               Muon momentum 0.97, warmdown_frac=0.75,
               Late QAT at LR scale < 0.15, WD=0.095.
Quantization:  Full Hessian GPTQ int6 with AR self-generated calibration,
               selective +/-1 pruning, LZMA preset=9.
Eval:          Document-aware VarLen sliding eval with score-first
               AdamW LoRA TTT (chunk=64, rank=8, per-document).
"""
from __future__ import annotations

import copy
import glob
import io
import lzma
import math
import os
import random
import struct
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
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _HAS_FA_VARLEN = True
except ImportError:
    _HAS_FA_VARLEN = False

# ===========================================================================
# Hyperparameters (all overridable via env vars)
# ===========================================================================

class Hyperparameters:
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    iterations      = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps    = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len   = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len    = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    warmdown_frac   = float(os.environ.get("WARMDOWN_FRAC", 0.75))

    # Model
    vocab_size      = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers      = int(os.environ.get("NUM_LAYERS", 11))
    model_dim       = int(os.environ.get("MODEL_DIM", 512))
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult        = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init    = float(os.environ.get("QK_GAIN_INIT", 5.25))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Features
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 3072))
    bigram_dim      = int(os.environ.get("BIGRAM_DIM", 112))
    xsa_last_n      = int(os.environ.get("XSA_LAST_N", 11))
    rope_dims       = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale        = bool(int(os.environ.get("LN_SCALE", "1")))
    ve_enabled      = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim          = int(os.environ.get("VE_DIM", 128))
    ve_layers       = os.environ.get("VE_LAYERS", "9,10")

    # Depth recurrence
    recur_start     = int(os.environ.get("RECUR_START", 3))
    recur_end       = int(os.environ.get("RECUR_END", 5))
    recur_passes    = int(os.environ.get("RECUR_PASSES", 2))

    # Parallel residuals (GPT-J style) from this layer onward
    parallel_resid_start = int(os.environ.get("PARALLEL_RESID_START", 7))

    # Optimizer
    embed_lr        = float(os.environ.get("EMBED_LR", 0.6))
    head_lr         = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR", 0.035))
    matrix_lr       = float(os.environ.get("MATRIX_LR", 0.022))
    scalar_lr       = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM", 0.97))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1           = float(os.environ.get("BETA1", 0.9))
    beta2           = float(os.environ.get("BETA2", 0.95))
    adam_eps         = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm  = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd         = float(os.environ.get("MUON_WD", 0.095))
    adam_wd         = float(os.environ.get("ADAM_WD", 0.04))

    # Weight averaging
    swa_enabled     = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every       = int(os.environ.get("SWA_EVERY", 50))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    eval_stride     = int(os.environ.get("EVAL_STRIDE", 64))

    # GPTQ
    gptq_block_size = int(os.environ.get("GPTQ_BLOCK_SIZE", 128))

    # TTT (test-time training)
    ttt_enabled     = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_chunk_size  = int(os.environ.get("TTT_CHUNK_SIZE", 64))
    ttt_lora_rank   = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lr          = float(os.environ.get("TTT_LR", 1e-3))
    ttt_beta1       = float(os.environ.get("TTT_BETA1", 0.9))
    ttt_beta2       = float(os.environ.get("TTT_BETA2", 0.999))
    ttt_target_layers = os.environ.get("TTT_TARGET_LAYERS", "0,1,2,3,4,5,6,7,8,9,10")

# ===========================================================================
# Control tensor patterns  (kept at higher precision during quantization)
# ===========================================================================

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,smear,ve_layer_scales,ve_shared.scale,"
        "par_gate,recur_gate",
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16

# ===========================================================================
# Batched Newton-Schulz orthogonalization
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
# Parallel Muon optimizer with parameter banking
# ===========================================================================

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay))
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
        raise FileNotFoundError(f"No files for: {pattern}")
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
    if tokens.size != n:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
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
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                s = seq_len / self.train_seq_len
                new_base = self.base * (s ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
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

    def forward(self, x, q_w, k_w, v_w, out_w, v_embed=None, v0=None):
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]

        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
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

        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v

class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev

class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class MLP(nn.Module):
    """LeakyReLU(0.5)^2 MLP with optional fused forward."""
    def __init__(self, dim, mlp_mult):
        super().__init__()
    def forward(self, x, up_w, down_w):
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False, parallel_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel_residual = parallel_residual
        if parallel_residual:
            self.par_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                v_embed=None, v0=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out, raw_v = self.attn(normed, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)

        if self.parallel_residual:
            # GPT-J style: attention and MLP computed in parallel from the same input
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor, up_w, down_w)
            g = self.par_gate.to(dtype=x.dtype)
            combined = g * self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out + \
                       (1 - g) * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            x_out = x_in + combined
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_out) * self.ln_scale_factor, up_w, down_w)
        return x_out, raw_v


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, bigram_vocab_size=0, bigram_dim=128,
                 xsa_last_n=0, rope_dims=0, ln_scale=False,
                 ve_enabled=False, ve_dim=128, ve_layers="9,10",
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
        head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * head_dim
        mlp_dim = int(mlp_mult * model_dim)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)

        # U-Net skip connections
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Depth recurrence: learnable blending gate per recur layer
        num_recur_layers = max(0, recur_end - recur_start + 1)
        if num_recur_layers > 0 and recur_passes > 1:
            self.recur_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
                for _ in range(num_recur_layers)
            ])
        else:
            self.recur_gates = nn.ParameterList()

        # Parameter banks
        self.qo_bank = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                  qk_gain_init, layer_idx=i, ln_scale=ln_scale,
                  parallel_residual=(i >= parallel_resid_start))
            for i in range(num_layers)
        ])

        # Partial RoPE
        if rope_dims > 0:
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)

        # XSA on all layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True

        # Value Embedding
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        kv_dim_ve = kv_dim
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
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
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if ve_cache is not None and 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve'] if ve_cache is not None else self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _run_block(self, i, x, x0, input_ids, ve_cache, v0):
        n = self.num_layers
        ve = self._get_ve(i, input_ids, ve_cache)
        x_new, raw_v = self.blocks[i](x, x0,
            self.qo_bank[i], self.kv_bank[i], self.kv_bank[n + i],
            self.qo_bank[n + i], self.mlp_up_bank[i], self.mlp_down_bank[i],
            v_embed=ve, v0=v0)
        return x_new, raw_v

    def _forward_body(self, input_ids):
        """Shared forward body returning final hidden states."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips = []
        ve_cache = {}

        for i in range(self.num_encoder_layers):
            # Depth recurrence: repeat selected layers
            if self.recur_start <= i <= self.recur_end and self.recur_passes > 1:
                recur_idx = i - self.recur_start
                gate = self.recur_gates[recur_idx].to(dtype=x.dtype) if recur_idx < len(self.recur_gates) else 0.5
                for p in range(self.recur_passes):
                    x_new, raw_v = self._run_block(i, x, x0, input_ids, ve_cache, v0)
                    if p > 0:
                        # Gated blending: avoid overwriting with hard substitution
                        x = gate * x_new + (1 - gate) * x
                    else:
                        x = x_new
                    if v0 is None and raw_v is not None:
                        v0 = raw_v
            else:
                x, raw_v = self._run_block(i, x, x0, input_ids, ve_cache, v0)
                if v0 is None and raw_v is not None:
                    v0 = raw_v
            skips.append(x)

        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            # Depth recurrence in decoder layers too
            if self.recur_start <= bi <= self.recur_end and self.recur_passes > 1:
                recur_idx = bi - self.recur_start
                gate = self.recur_gates[recur_idx].to(dtype=x.dtype) if recur_idx < len(self.recur_gates) else 0.5
                for p in range(self.recur_passes):
                    x_new, _ = self._run_block(bi, x, x0, input_ids, ve_cache, v0)
                    if p > 0:
                        x = gate * x_new + (1 - gate) * x
                    else:
                        x = x_new
            else:
                x, _ = self._run_block(bi, x, x0, input_ids, ve_cache, v0)

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
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_seqs = max(local_batch_tokens // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s0, s1, local_seqs):
            be = min(bs + local_seqs, s1)
            raw_s, raw_e = bs * seq_len, be * seq_len + 1
            local = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                bl = model(x, y).detach()
            n = float(y.numel())
            loss_sum += bl.to(torch.float64) * n
            tok_cnt += n
            prev, tgt = x.reshape(-1), y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int16)
            byte_cnt += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_cnt
    bpt = vl.item() / math.log(2.0)
    tpb = tok_cnt.item() / byte_cnt.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)

# ===========================================================================
# Sliding window evaluation (no TTT)
# ===========================================================================

def eval_val_sliding(args, base_model, rank, world_size, device,
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
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
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
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bpt * tpb

# ===========================================================================
# Document-aware Score-First AdamW LoRA TTT Eval
#
# Original implementation. Key properties:
#   1. Documents identified by BOS/EOS markers in the token stream
#   2. Each document gets a fresh LoRA adapter (A, B matrices)
#   3. Each chunk is SCORED first under torch.no_grad() (score-first)
#   4. Then LoRA is updated with AdamW on that chunk's loss
#   5. Single left-to-right pass, each token scored exactly once
#   6. LoRA is discarded at document boundary
#   7. AdamW optimizer (not SGD) for per-parameter adaptive lr
# ===========================================================================

class LoRAAdapter:
    """Minimal LoRA: delta_W = B @ A where A: [rank, in], B: [out, rank]."""
    __slots__ = ('A', 'B', 'opt_state_A', 'opt_state_B', 'step')

    def __init__(self, in_dim, out_dim, rank, device, dtype=torch.bfloat16):
        # Kaiming init for A, zero init for B (so initial delta_W = 0)
        self.A = torch.randn(rank, in_dim, device=device, dtype=dtype) * (1.0 / math.sqrt(in_dim))
        self.B = torch.zeros(out_dim, rank, device=device, dtype=dtype)
        # AdamW state
        self.opt_state_A = {
            'm': torch.zeros_like(self.A),
            'v': torch.zeros_like(self.A),
        }
        self.opt_state_B = {
            'm': torch.zeros_like(self.B),
            'v': torch.zeros_like(self.B),
        }
        self.step = 0

    def get_delta(self):
        return self.B @ self.A

    def adamw_step(self, grad_A, grad_B, lr, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0):
        self.step += 1
        bc1 = 1.0 - beta1 ** self.step
        bc2 = 1.0 - beta2 ** self.step

        for param, grad, state in [(self.A, grad_A, self.opt_state_A),
                                   (self.B, grad_B, self.opt_state_B)]:
            # Weight decay (decoupled)
            if wd > 0:
                param.mul_(1.0 - lr * wd)
            # Momentum
            state['m'].mul_(beta1).add_(grad, alpha=1 - beta1)
            state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            # Bias-corrected
            m_hat = state['m'] / bc1
            v_hat = state['v'] / bc2
            param.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


def _find_document_boundaries(tokens, vocab_size):
    """Find document boundaries based on token value wrapping / BOS detection.
    Returns list of (start, end) index pairs."""
    # Simple heuristic: treat token 0 or 1 as BOS markers, or detect large
    # discontinuities. For FineWeb with SP tokenizer, documents are packed
    # contiguously. We split on token_id == 1 (BOS).
    boundaries = [0]
    t = tokens.cpu().numpy() if tokens.is_cuda else tokens.numpy()
    for i in range(1, len(t)):
        if t[i] == 1:  # BOS token
            boundaries.append(i)
    boundaries.append(len(t))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
            if boundaries[i + 1] - boundaries[i] > 1]


def eval_val_doc_ttt(args, base_model, rank, world_size, device,
                     val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                     eval_seq_len=None):
    """Score-first document-aware AdamW LoRA TTT evaluation.

    For each document:
      1. Initialize fresh LoRA adapters
      2. Process document in fixed-size chunks
      3. For each chunk: score first (no_grad), then adapt LoRA with AdamW
      4. Discard LoRA at document end

    Conditions satisfied (Issue #1017):
      - Condition 1: Strict causal dependence (standard causal masking)
      - Condition 2: Full normalized softmax distribution
      - Condition 3: Score before update (chunk scored under no_grad BEFORE adapt)
      - Condition 4: Single left-to-right pass, each token scored once
    """
    seq_len = eval_seq_len or args.train_seq_len
    chunk_size = args.ttt_chunk_size
    lora_rank = args.ttt_lora_rank
    ttt_lr = args.ttt_lr
    target_layers = [int(x) for x in args.ttt_target_layers.split(",") if x.strip()]
    model_dim = args.model_dim

    total_tokens = val_tokens.numel() - 1
    # Split validation tokens across ranks
    rank_start = (total_tokens * rank) // world_size
    rank_end = (total_tokens * (rank + 1)) // world_size
    my_tokens = val_tokens[rank_start:rank_end + 1]

    # Find document boundaries in our shard
    doc_ranges = _find_document_boundaries(my_tokens, args.vocab_size)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()

    # Cache the original output projection weights for targeted layers
    n = args.num_layers
    orig_out_weights = {}
    for li in target_layers:
        if li < n:
            orig_out_weights[li] = base_model.qo_bank[n + li].detach().clone()

    for doc_start, doc_end in doc_ranges:
        doc_len = doc_end - doc_start
        if doc_len < 2:
            continue

        # Fresh LoRA per document for each target layer
        lora_adapters = {}
        for li in target_layers:
            if li < n:
                lora_adapters[li] = LoRAAdapter(model_dim, model_dim, lora_rank, device)

        doc_tokens = my_tokens[doc_start:doc_end].to(dtype=torch.int64, device=device)

        # Process in chunks
        pos = 0
        while pos < doc_len - 1:
            end = min(pos + chunk_size, doc_len - 1)
            actual_chunk = end - pos
            if actual_chunk < 1:
                break

            # Build chunk input with context window
            ctx_start = max(0, pos - (seq_len - chunk_size))
            x_chunk = doc_tokens[ctx_start:end].unsqueeze(0)
            y_chunk = doc_tokens[ctx_start + 1:end + 1].unsqueeze(0)
            chunk_len = x_chunk.shape[1]

            # === PHASE 1: SCORE the chunk (no_grad, score-first) ===
            with torch.no_grad():
                # Temporarily apply current LoRA deltas to output projections
                for li, adapter in lora_adapters.items():
                    delta = adapter.get_delta()
                    base_model.qo_bank.data[n + li] = orig_out_weights[li] + delta.to(orig_out_weights[li].dtype)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_chunk)

                # Score only the new positions (not context prefix)
                score_start = pos - ctx_start
                scored_logits = logits[0, score_start:].float()
                scored_targets = y_chunk[0, score_start:]

                nll = F.cross_entropy(scored_logits, scored_targets, reduction="none")
                loss_sum += nll.to(torch.float64).sum()
                token_count += float(actual_chunk)

                # Byte counting
                tgt = scored_targets
                prev = x_chunk[0, score_start:]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            # === PHASE 2: ADAPT LoRA on the already-scored chunk ===
            if actual_chunk >= 2:  # Need at least 2 tokens for meaningful gradient
                # Use the scored chunk for adaptation
                for li, adapter in lora_adapters.items():
                    adapter.A.requires_grad_(True)
                    adapter.B.requires_grad_(True)

                # Forward with LoRA (differentiable)
                # We apply LoRA by modifying the output projection temporarily
                # and running a small forward pass just for the gradient
                for li, adapter in lora_adapters.items():
                    delta = adapter.B @ adapter.A
                    base_model.qo_bank.data[n + li] = orig_out_weights[li] + delta.to(orig_out_weights[li].dtype)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    adapt_logits = base_model.forward_logits(x_chunk)

                adapt_loss = F.cross_entropy(
                    adapt_logits[0, score_start:].float(),
                    y_chunk[0, score_start:],
                    reduction="mean")
                adapt_loss.backward()

                # Extract gradients via the bank and update LoRA
                for li, adapter in lora_adapters.items():
                    # The gradient flows through qo_bank to B@A
                    # We need to compute grad_A and grad_B from the chain rule
                    bank_grad = base_model.qo_bank.grad
                    if bank_grad is not None:
                        out_grad = bank_grad[n + li].float()
                        # delta_W = B @ A, so d_loss/dB = out_grad @ A^T, d_loss/dA = B^T @ out_grad
                        grad_B = out_grad @ adapter.A.float().T
                        grad_A = adapter.B.float().T @ out_grad
                        adapter.adamw_step(
                            grad_A.to(adapter.A.dtype),
                            grad_B.to(adapter.B.dtype),
                            lr=ttt_lr,
                            beta1=args.ttt_beta1,
                            beta2=args.ttt_beta2,
                        )

                # Zero grads
                if base_model.qo_bank.grad is not None:
                    base_model.qo_bank.grad = None
                for p in base_model.parameters():
                    if p.grad is not None:
                        p.grad = None

                for li, adapter in lora_adapters.items():
                    adapter.A = adapter.A.detach()
                    adapter.B = adapter.B.detach()

            pos = end

        # Restore original weights after each document
        for li in target_layers:
            if li < n:
                base_model.qo_bank.data[n + li] = orig_out_weights[li].clone()

    # All-reduce across ranks
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, token_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bpt * tpb

# ===========================================================================
# AR self-generated calibration data
# ===========================================================================

def generate_autoregressive_calib(model, device, num_seqs=64, seq_len=2048,
                                  vocab_size=1024, temperature=0.8, batch_size=8, seed=42):
    model.eval()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    all_tokens = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, num_seqs, batch_size):
            bs = min(batch_size, num_seqs - batch_start)
            tokens = torch.randint(0, vocab_size, (bs, 1), device=device, generator=rng)
            for _ in range(seq_len - 1):
                logits = model.forward_logits(tokens)
                next_logit = logits[:, -1, :]
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1, generator=rng)
                tokens = torch.cat([tokens, next_tok], dim=1)
            for i in range(bs):
                all_tokens.append(tokens[i:i+1])
    return all_tokens

# ===========================================================================
# Int6 quantization helpers
# ===========================================================================

def quantize_int6_per_row(t, clip_range=31):
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float('inf')
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale

def quantize_int6_gptq(weight, hessian, clip_range=31, block_size=128):
    """Full GPTQ: Hessian-aware int6 quantization with Cholesky error compensation."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return quantize_int6_per_row(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    W[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = None
    for damp_scale in [1.0, 10.0, 100.0, 1000.0]:
        try:
            H_try = H.clone()
            if damp_scale > 1.0:
                extra = damp * damp_scale
                H_try[torch.arange(cols), torch.arange(cols)] += extra
            L = torch.linalg.cholesky(H_try)
            Hinv_full = torch.cholesky_inverse(L)
            Hinv = torch.linalg.cholesky(Hinv_full, upper=True)
            break
        except torch._C._LinAlgError:
            continue
    if Hinv is None:
        return quantize_int6_per_row(t32, clip_range)

    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            count = i2 - i1
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, count, dtype=torch.int8)
            Err1 = torch.zeros(rows, count)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for ci in range(count):
                w = W1[:, ci]
                d = Hinv1[ci, ci]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, ci] = q
                err = (w - q.float() * sf) / d
                W1[:, ci:] -= err.unsqueeze(1) * Hinv1[ci, ci:].unsqueeze(0)
                Err1[:, ci] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]
    return best_q, best_scale

# ===========================================================================
# Unbank / Rebank state dicts
# ===========================================================================

def _unbank_state_dict(sd, num_layers):
    out = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"] = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"] = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"] = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"] = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"] = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"] = tensor[i]
        else:
            out[name] = tensor
    return out

def _rebank_state_dict(sd, num_layers, template_sd):
    out = {}
    n = num_layers
    qo = [None] * (2 * n)
    kv = [None] * (2 * n)
    up = [None] * n
    down = [None] * n
    consumed = set()
    for i in range(n):
        for k, lst, idx in [
            (f"blocks.{i}.attn.c_q.weight", qo, i),
            (f"blocks.{i}.attn.proj.weight", qo, n + i),
            (f"blocks.{i}.attn.c_k.weight", kv, i),
            (f"blocks.{i}.attn.c_v.weight", kv, n + i),
            (f"blocks.{i}.mlp.fc.weight", up, i),
            (f"blocks.{i}.mlp.proj.weight", down, i),
        ]:
            if k in sd:
                lst[idx] = sd[k]
                consumed.add(k)
    out["qo_bank"] = torch.stack(qo).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"] = torch.stack(kv).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"] = torch.stack(up).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down).to(dtype=template_sd["mlp_down_bank"].dtype)
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
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
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

    def forward(self, x, v_embed=None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
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
        return self.proj(y.reshape(bsz, seqlen, dim))

class _HMLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, int(mlp_mult * dim), bias=False)
        self.proj = CastedLinear(int(mlp_mult * dim), dim, bias=False)
    def forward(self, x):
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class _HBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False, parallel_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = _HAttn(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = _HMLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel_residual = parallel_residual
        if parallel_residual:
            self.par_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, x0, v_embed=None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed, v_embed=v_embed)
        if self.parallel_residual:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            g = self.par_gate.to(dtype=x.dtype) if hasattr(self, 'par_gate') else 0.5
            combined = g * self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out + \
                       (1 - g) * self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
            x_out = x_in + combined
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(
                self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out

class _HGPT(nn.Module):
    """Non-banked GPT for Hessian collection."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, logit_softcap, rope_base, qk_gain_init,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0, rope_dims=0,
                 ln_scale=False, ve_enabled=False, ve_dim=128, ve_layers="9,10",
                 parallel_resid_start=7):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            _HBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                    qk_gain_init, layer_idx=i, ln_scale=ln_scale,
                    parallel_residual=(i >= parallel_resid_start))
            for i in range(num_layers)
        ])
        if rope_dims > 0:
            head_dim = model_dim // num_heads
            for block in self.blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            self.ve_shared = ValueEmbedding(vocab_size, ve_dim, kv_dim)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared = None
            self.ve_layer_scales = nn.ParameterList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

    def _get_ve(self, layer_idx, input_ids, ve_cache):
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_idx = self.ve_layer_indices.index(layer_idx)
        return ve_cache['ve'] * self.ve_layer_scales[ve_idx].to(dtype=ve_cache['ve'].dtype)

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips = []
        ve_cache = {}
        for i in range(self.num_encoder_layers):
            ve = self._get_ve(i, input_ids, ve_cache)
            x = self.blocks[i](x, x0, v_embed=ve)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            ve = self._get_ve(bi, input_ids, ve_cache)
            x = self.blocks[bi](x, x0, v_embed=ve)
        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x_flat, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

# ===========================================================================
# Hessian collection + mixed quantization
# ===========================================================================

def collect_hessians_from_tokens(hessian_model, token_seqs, device):
    hessians = {}
    hooks = []
    for name, module in hessian_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    hessian_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for seq in token_seqs:
            x = seq[:, :-1].to(device)
            y = seq[:, 1:].to(device)
            hessian_model(x, y)
    for h in hooks:
        h.remove()
    num_batches = len(token_seqs)
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
        hessians[name] = H
    return hessians

def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"

def mixed_quantize_int6(state_dict, int6_cats, hessians=None):
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if cat in int6_cats and t.ndim >= 1:
            H = hessians.get(name) if hessians else None
            if H is not None:
                q, s = quantize_int6_gptq(t, hessian=H, clip_range=31)
            else:
                q, s = quantize_int6_per_row(t, clip_range=31)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            t32 = t.float()
            if t32.ndim == 2:
                pct_q = 99.99984 / 100.0
                clip_abs = torch.quantile(t32.abs(), pct_q, dim=1)
                clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
                s = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
                q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127).to(torch.int8)
                result[name + ".q"] = q.contiguous()
                result[name + ".scale"] = s.to(torch.float16).contiguous()
            else:
                amax = t32.abs().max().item()
                s = torch.tensor(amax / 127.0 if amax > 0 else 1.0, dtype=torch.float32)
                q = torch.clamp(torch.round(t32 / s), -127, 127).to(torch.int8)
                result[name + ".q"] = q.contiguous()
                result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

# ===========================================================================
# Main training loop
# ===========================================================================

def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
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
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}  PyTorch {torch.__version__}  FA3={_HAS_FA3}  FA_VARLEN={_HAS_FA_VARLEN}",
         console=False)
    log0(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout, console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_space_lut, is_boundary_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val tokens:{val_tokens.numel()-1}  FA3:{_HAS_FA3}  FA_VARLEN:{_HAS_FA_VARLEN}")

    # ---- Model ----
    CastedLinear._qat_enabled = False
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        recur_start=args.recur_start, recur_end=args.recur_end,
        recur_passes=args.recur_passes,
        parallel_resid_start=args.parallel_resid_start,
    ).to(device).bfloat16()

    base_model.qo_bank.data = base_model.qo_bank.data.float()
    base_model.kv_bank.data = base_model.kv_bank.data.float()
    base_model.mlp_up_bank.data = base_model.mlp_up_bank.data.float()
    base_model.mlp_down_bank.data = base_model.mlp_down_bank.data.float()
    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = compiled_model

    # ---- Optimizers ----
    matrix_params = [base_model.qo_bank, base_model.kv_bank,
                     base_model.mlp_up_bank, base_model.mlp_down_bank]
    block_np = list(base_model.blocks.named_parameters())
    scalar_params = [p for n, p in block_np
                     if p.ndim < 2 or any(c in n for c in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    # Depth recurrence gates
    for rg in base_model.recur_gates:
        scalar_params.append(rg)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(base_model.bigram.proj.weight)
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(base_model.ve_shared.proj.weight)
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)

    opt_tok = torch.optim.AdamW(tok_params, betas=(args.beta1, args.beta2),
                                eps=args.adam_eps, weight_decay=args.adam_wd, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
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

    replicated_params = list(opt_tok.param_groups[0]["params"])
    for pg in opt_tok.param_groups[1:]:
        replicated_params.extend(pg["params"])
    replicated_params.extend(scalar_params)
    if base_model.lm_head is not None:
        replicated_params.append(base_model.lm_head.weight)

    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    par_layers = [i for i, b in enumerate(base_model.blocks) if b.parallel_residual]
    log0(f"params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
         f"heads:{args.num_heads}/{args.num_kv_heads} mlp:{args.mlp_mult}")
    log0(f"XSA:{xsa_layers} bigram:{args.bigram_vocab_size}x{args.bigram_dim} "
         f"rope_dims:{args.rope_dims} ve:{args.ve_layers if args.ve_enabled else 'off'}")
    log0(f"recur:[{args.recur_start}-{args.recur_end}]x{args.recur_passes} "
         f"parallel_resid:{par_layers} qk_gain:{args.qk_gain_init}")
    log0(f"world:{world_size} accum:{grad_accum_steps} batch:{args.train_batch_tokens} "
         f"seq:{args.train_seq_len} warmdown_frac:{args.warmdown_frac}")
    log0(f"vocab:{args.vocab_size} muon_lr:{args.matrix_lr} muon_wd:{args.muon_wd} "
         f"muon_mom:{args.muon_momentum}")
    if args.ttt_enabled:
        log0(f"TTT: chunk={args.ttt_chunk_size} rank={args.ttt_lora_rank} "
             f"lr={args.ttt_lr} layers={args.ttt_target_layers}")

    # ---- EMA ----
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
    ema_decay = 0.997

    # ---- Data + training ----
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    max_wall_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def zero_grad():
        for o in optimizers:
            o.zero_grad(set_to_none=True)

    def lr_mul(step, elapsed_ms):
        """Warmdown schedule based on wall-clock fraction."""
        if args.warmdown_frac <= 0:
            return 1.0
        if max_wall_ms is None:
            return 1.0
        # Compute fraction of total time that is warmdown
        warmdown_ms = max_wall_ms * args.warmdown_frac
        warmdown_start_ms = max_wall_ms - warmdown_ms
        if elapsed_ms < warmdown_start_ms:
            return 1.0
        remaining = max(max_wall_ms - elapsed_ms, 0.0)
        return remaining / max(warmdown_ms, 1e-9)

    # ---- Warmup (compile priming) ----
    if args.warmup_steps > 0:
        init_sd = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
        init_opt = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            for ms in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for o in optimizers:
                o.step()
            zero_grad()
            if ws + 1 == args.warmup_steps or (ws + 1) % 10 == 0:
                log0(f"warmup {ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_sd, strict=True)
        for o, s in zip(optimizers, init_opt, strict=True):
            o.load_state_dict(s)
        zero_grad()
        ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ---- SWA state ----
    swa_state = None
    swa_count = 0

    # ---- Training loop ----
    train_ms, stop_step = 0.0, None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last = step == args.iterations or (stop_step is not None and step >= stop_step)
        do_val = last or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"time:{train_ms:.0f}ms avg:{train_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last:
            if stop_step is not None and step < args.iterations:
                log0(f"wallclock_stop step:{step}/{args.iterations} time:{train_ms:.0f}ms")
            break

        elapsed = train_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed)

        # Late QAT
        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        zero_grad()
        train_loss = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups:
            g["momentum"] = muon_mom

        # LR scheduling
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # === 3-phase overlapped optimizer step ===
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
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"time:{approx_ms:.0f}ms avg:{approx_ms/step:.2f}ms scale:{scale:.4f}"
                 f"{' QAT' if CastedLinear._qat_enabled else ''}")

        reached = max_wall_ms is not None and approx_ms >= max_wall_ms
        if distributed and max_wall_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_step is None and reached:
            stop_step = step

    log0(f"peak_mem alloc:{torch.cuda.max_memory_allocated()//1024//1024}MiB "
         f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    # ---- Apply EMA weights ----
    log0("ema:applying EMA weights")
    current_sd = base_model.state_dict()
    avg_state = {n: t.to(dtype=current_sd[n].dtype) for n, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)

    # ---- Diagnostic eval ----
    torch.cuda.synchronize()
    t_diag = time.perf_counter()
    diag_vl, diag_vb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps,
                                val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut)
    torch.cuda.synchronize()
    log0(f"DIAGNOSTIC post_ema val_loss:{diag_vl:.4f} val_bpb:{diag_vb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_diag):.0f}ms")

    # ---- Unbank for GPTQ ----
    full_sd = base_model.state_dict()
    export_sd = {k: v for k, v in full_sd.items()}
    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked_sd = _unbank_state_dict(sd_cpu, args.num_layers)

    # ---- Build non-banked model for Hessian collection ----
    log0("gptq:building non-banked model for Hessian collection...")
    hm = _HGPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        parallel_resid_start=args.parallel_resid_start,
    ).to(device).bfloat16()
    for m in hm.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(hm)
    hm.load_state_dict({k: v.to(device) for k, v in unbanked_sd.items()
                        if k in hm.state_dict()}, strict=False)

    # ---- AR self-generated calibration ----
    log0("gptq:generating AR calibration data (64 seqs x 2048 tokens, temp=0.8)...")
    base_model.load_state_dict(export_sd, strict=False)
    t_gen = time.perf_counter()
    ar_tokens = generate_autoregressive_calib(
        base_model, device, num_seqs=64, seq_len=args.train_seq_len,
        vocab_size=args.vocab_size, temperature=0.8, batch_size=8, seed=args.seed)
    log0(f"gptq:generated {len(ar_tokens)} sequences in {time.perf_counter()-t_gen:.1f}s")

    log0("gptq:collecting hessians from AR data...")
    hessians = collect_hessians_from_tokens(hm, ar_tokens, device)
    log0(f"gptq:collected hessians for {len(hessians)} layers")
    del ar_tokens, hm
    torch.cuda.empty_cache()

    # ---- GPTQ quantization ----
    quant_result, quant_meta = mixed_quantize_int6(unbanked_sd, {"mlp", "attn"}, hessians=hessians)

    # ---- Selective +/-1 pruning ----
    target_mb = float(os.environ.get("TARGET_MB", "15.9"))
    code_bytes_est = len(code.encode("utf-8"))
    ones_info = []
    for name, info in quant_meta.items():
        if not (isinstance(info, dict) and info.get("type") == "int6"):
            continue
        qk, sk = name + ".q", name + ".scale"
        if qk not in quant_result or sk not in quant_result:
            continue
        q, s = quant_result[qk], quant_result[sk]
        if s.ndim > 0:
            ones_mask = (q.abs() == 1)
            if ones_mask.any():
                row_idx = torch.arange(q.shape[0]).unsqueeze(1).expand_as(q)[ones_mask]
                flat_idx = torch.arange(q.numel()).reshape(q.shape)[ones_mask]
                errors = s.float()[row_idx].pow(2)
                for fi, err in zip(flat_idx.tolist(), errors.tolist()):
                    ones_info.append((qk, fi, err))

    if ones_info:
        ones_info.sort(key=lambda x: x[2])
        def _try_prune(n):
            tmp = {k: v.clone() for k, v in quant_result.items()}
            for i in range(min(n, len(ones_info))):
                tmp[ones_info[i][0]].view(-1)[ones_info[i][1]] = 0
            buf = io.BytesIO()
            torch.save({"w": tmp, "m": quant_meta}, buf)
            return len(lzma.compress(buf.getvalue(), preset=9)) + code_bytes_est, tmp

        no_sz, _ = _try_prune(0)
        target_bytes = int(target_mb * 1024 * 1024)
        log0(f"selective_prune: {len(ones_info)} +/-1 candidates, "
             f"unpruned={no_sz/(1024*1024):.2f}MB target={target_mb}MB")
        if no_sz <= target_bytes:
            log0("selective_prune: already fits, no pruning needed")
        else:
            full_sz, _ = _try_prune(len(ones_info))
            log0(f"selective_prune: full prune={full_sz/(1024*1024):.2f}MB")
            if full_sz > target_bytes:
                log0("selective_prune: even full prune not enough, applying all")
                _, quant_result = _try_prune(len(ones_info))
            else:
                lo, hi = 0, len(ones_info)
                while lo < hi:
                    mid = (lo + hi) // 2
                    sz, _ = _try_prune(mid)
                    if sz <= target_bytes:
                        hi = mid
                    else:
                        lo = mid + 1
                log0(f"selective_prune: pruning {lo}/{len(ones_info)} "
                     f"({100*lo/len(ones_info):.1f}%) to fit {target_mb}MB")
                _, quant_result = _try_prune(lo)

    # ---- Compress ----
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)

    if master:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        model_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {model_bytes} bytes")
        log0(f"Code: {code_bytes} bytes")
        log0(f"TOTAL ARTIFACT: {model_bytes + code_bytes} bytes "
             f"{'PASS' if model_bytes + code_bytes <= 16_000_000 else 'OVER 16MB!'}")

    # ---- Roundtrip verification ----
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu",
                             weights_only=False)
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_sd)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
        recur_start=args.recur_start, recur_end=args.recur_end,
        recur_passes=args.recur_passes,
        parallel_resid_start=args.parallel_resid_start,
    ).to(device).bfloat16()
    eval_model.qo_bank.data = eval_model.qo_bank.data.float()
    eval_model.kv_bank.data = eval_model.kv_bank.data.float()
    eval_model.mlp_up_bank.data = eval_model.mlp_up_bank.data.float()
    eval_model.mlp_down_bank.data = eval_model.mlp_down_bank.data.float()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # Standard roundtrip eval
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_vl, q_vb = eval_val(args, eval_model, rank, world_size, device, grad_accum_steps,
                          val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
                          eval_seq_len=effective_eval_seq_len)
    torch.cuda.synchronize()
    log0(f"final_int6_roundtrip val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"final_int6_roundtrip_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")

    # Sliding window eval
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_vl, sw_vb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len)
        torch.cuda.synchronize()
        log0(f"final_int6_sliding_window val_loss:{sw_vl:.4f} val_bpb:{sw_vb:.4f} "
             f"stride:{args.eval_stride} eval_time:{1000.0*(time.perf_counter()-t_slide):.0f}ms")
        log0(f"final_int6_sliding_window_exact val_loss:{sw_vl:.8f} val_bpb:{sw_vb:.8f}")

    # Document-aware TTT eval
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_vl, ttt_vb = eval_val_doc_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_space_lut, is_boundary_lut,
            eval_seq_len=effective_eval_seq_len)
        torch.cuda.synchronize()
        log0(f"final_doc_ttt val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f} "
             f"chunk:{args.ttt_chunk_size} rank:{args.ttt_lora_rank} "
             f"eval_time:{1000.0*(time.perf_counter()-t_ttt):.0f}ms")
        log0(f"final_doc_ttt_exact val_loss:{ttt_vl:.8f} val_bpb:{ttt_vb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
