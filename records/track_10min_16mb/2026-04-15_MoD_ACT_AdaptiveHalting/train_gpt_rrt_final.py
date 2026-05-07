"""
train_gpt_rrt_final.py
======================
Relaxed Recursive Transformer (RRT-LoRA) for OpenAI Parameter Golf.

Novel contribution: Per-step LoRA adapters on depth-recurrence layers.
Each recurrence pass applies a tiny learned delta (rank=4) to Q and V,
allowing passes to specialize while sharing base weights. LoRA B matrices
initialize to zero — identical to baseline at step 0 — and warm up via
alpha schedule starting at recur_activate_frac * iterations.

Full SOTA stack:
  - SP8192 tokenizer
  - 11L x 512d x 8H/4KV, MLP 4x, LeakyReLU(0.5)^2
  - Partial RoPE (16/64 head dims)
  - Depth recurrence on layers 3,4,5 (3 steps, RRT-LoRA rank=4)
  - Parallel residuals (GPT-J style) on layers 7+
  - QK-Gain 5.25 (learnable per-head query scaling)
  - MuonEq-R optimizer (row-normalized Muon, arXiv:2603.28254)
  - EMA decay 0.9965
  - GPTQ SDClip: int6 matrices (k=12.85), int8 embeddings (k=20.0)
  - Bit-packed int6 storage (saves ~25% vs int8 storage)
  - Brotli-11 model compression
  - Legal score-first TTT: SGD lr=0.005, momentum=0.9, 3 epochs/32K chunk
  - LZMA code compression
  - WD=0.095, MLR=0.022, warmdown=72%, EMA=0.9965

Usage (8xH100):
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\
  python3 data/cached_challenge_fineweb.py --variant sp8192

  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_final.py
"""

from __future__ import annotations

import glob
import io
import lzma
import math
import os
import random
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import brotli
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

# ============================================================
# HYPERPARAMETERS
# ============================================================

class Hyperparameters:
    # Paths
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    # Eval / logging
    val_batch_size      = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every      = int(os.environ.get("VAL_LOSS_EVERY",   1000))
    train_log_every     = int(os.environ.get("TRAIN_LOG_EVERY",  200))

    # Schedule
    iterations              = int(os.environ.get("ITERATIONS",           20_000))
    warmdown_frac           = float(os.environ.get("WARMDOWN_FRAC",      0.72))
    warmup_steps            = int(os.environ.get("WARMUP_STEPS",         20))
    train_batch_tokens      = int(os.environ.get("TRAIN_BATCH_TOKENS",   524_288))
    train_seq_len           = int(os.environ.get("TRAIN_SEQ_LEN",        1024))
    max_wallclock_seconds   = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model
    vocab_size          = int(os.environ.get("VOCAB_SIZE",        8192))
    num_layers          = int(os.environ.get("NUM_LAYERS",        11))
    num_kv_heads        = int(os.environ.get("NUM_KV_HEADS",      4))
    model_dim           = int(os.environ.get("MODEL_DIM",         512))
    num_heads           = int(os.environ.get("NUM_HEADS",         8))
    mlp_mult            = int(os.environ.get("MLP_MULT",          4))
    tie_embeddings      = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base           = float(os.environ.get("ROPE_BASE",       10000.0))
    rope_partial_dim    = int(os.environ.get("ROPE_PARTIAL_DIM",  16))
    logit_softcap       = float(os.environ.get("LOGIT_SOFTCAP",   30.0))
    qk_gain_init        = float(os.environ.get("QK_GAIN_INIT",    5.25))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # RRT-LoRA recurrence
    recur_layers        = [int(x) for x in os.environ.get("RECUR_LAYERS", "3,4,5").split(",")]
    recur_steps         = int(os.environ.get("RECUR_STEPS",       3))
    recur_lora_rank     = int(os.environ.get("RECUR_LORA_RANK",   4))
    recur_activate_frac = float(os.environ.get("RECUR_ACTIVATE_FRAC", 0.35))
    lora_warmup_steps   = int(os.environ.get("LORA_WARMUP_STEPS", 500))

    # Architecture
    parallel_resid_from = int(os.environ.get("PARALLEL_RESID_FROM", 7))

    # TTT
    ttt_enabled         = bool(int(os.environ.get("TTT_ENABLED",     1)))
    ttt_lr              = float(os.environ.get("TTT_LR",             0.005))
    ttt_epochs          = int(os.environ.get("TTT_EPOCHS",           3))
    ttt_chunk_tokens    = int(os.environ.get("TTT_CHUNK_TOKENS",     32_768))
    ttt_freeze_layers   = int(os.environ.get("TTT_FREEZE_LAYERS",    2))
    ttt_momentum        = float(os.environ.get("TTT_MOMENTUM",       0.9))
    ttt_grad_clip       = float(os.environ.get("TTT_GRAD_CLIP",      1.0))

    # EMA
    ema_decay           = float(os.environ.get("EMA_DECAY",         0.9965))

    # Optimizer
    matrix_lr               = float(os.environ.get("MATRIX_LR",              0.022))
    scalar_lr               = float(os.environ.get("SCALAR_LR",              0.04))
    embed_lr                = float(os.environ.get("EMBED_LR",               0.6))
    tied_embed_lr           = float(os.environ.get("TIED_EMBED_LR",          0.05))
    head_lr                 = float(os.environ.get("HEAD_LR",                0.008))
    weight_decay            = float(os.environ.get("WEIGHT_DECAY",           0.095))
    muon_momentum           = float(os.environ.get("MUON_MOMENTUM",          0.95))
    muon_backend_steps      = int(os.environ.get("MUON_BACKEND_STEPS",       5))
    muon_mom_warmup_start   = float(os.environ.get("MUON_MOM_WARMUP_START",  0.85))
    muon_mom_warmup_steps   = int(os.environ.get("MUON_MOM_WARMUP_STEPS",    500))
    beta1                   = float(os.environ.get("BETA1",                  0.9))
    beta2                   = float(os.environ.get("BETA2",                  0.95))
    adam_eps                = float(os.environ.get("ADAM_EPS",               1e-8))
    grad_clip_norm          = float(os.environ.get("GRAD_CLIP_NORM",         1.0))

    @property
    def warmdown_iters(self) -> int:
        return int(self.iterations * self.warmdown_frac)


# ============================================================
# MUONEQ-R OPTIMIZER
# Row-normalized Muon — better spectral conditioning for NS5
# arXiv:2603.28254
# ============================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X


class MuonEqR(torch.optim.Optimizer):
    """Row-normalized Muon with Newton-Schulz-5 orthogonalization."""

    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int = 5, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank_d = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total = sum(p.numel() for p in params)
            updates = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0

            for i, p in enumerate(params):
                if i % world_size == rank_d:
                    g = p.grad.clone()
                    state = self.state.setdefault(p, {})
                    buf = state.setdefault("buf", torch.zeros_like(g))
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                    # Row-normalize before orthogonalization (MuonEq-R)
                    if g.ndim == 2:
                        g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-7)

                    g = zeropower_via_newtonschulz5(g, steps=steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                p.add_(updates[curr: curr + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                curr += p.numel()

        return loss


# ============================================================
# TOKENIZER-AGNOSTIC BPB EVALUATION
# ============================================================

def build_sentencepiece_luts(sp, vocab_size: int, device):
    sp_vocab = int(sp.vocab_size())
    n = max(sp_vocab, vocab_size)
    base_bytes   = np.zeros(n, dtype=np.int16)
    has_space    = np.zeros(n, dtype=np.bool_)
    is_boundary  = np.ones(n,  dtype=np.bool_)

    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode())

    return (torch.tensor(base_bytes,  dtype=torch.int16, device=device),
            torch.tensor(has_space,   dtype=torch.bool,  device=device),
            torch.tensor(is_boundary, dtype=torch.bool,  device=device))


def compute_bpb(logits_loss: float, n_tokens: int, n_bytes: int) -> float:
    return (logits_loss / math.log(2.0)) * (n_tokens / max(1, n_bytes))


# ============================================================
# DATA LOADING
# ============================================================

def load_data_shard(file: Path) -> Tensor:
    hdr = np.fromfile(file, dtype="<i4", count=256)
    if hdr.size != 256 or hdr[0] != 20240520 or hdr[1] != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(hdr[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=n, offset=1024).astype(np.uint16))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1].contiguous()


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(pattern)
        self.fi, self.pos = 0, 0
        self.buf = load_data_shard(self.files[0])

    def _next(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.buf = load_data_shard(self.files[self.fi])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        out, rem = [], n
        while rem > 0:
            avail = self.buf.numel() - self.pos
            if avail <= 0:
                self._next()
                continue
            k = min(rem, avail)
            out.append(self.buf[self.pos: self.pos + k])
            self.pos += k
            rem -= k
        return out[0] if len(out) == 1 else torch.cat(out)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum: int):
        local = global_tokens // (self.world_size * grad_accum)
        span  = local + 1
        chunk = self.stream.take(span * self.world_size)
        s = self.rank * span
        loc = chunk[s: s + span].to(torch.int64)
        return (loc[:-1].reshape(-1, seq_len).to(self.device, non_blocking=True),
                loc[ 1:].reshape(-1, seq_len).to(self.device, non_blocking=True))


# ============================================================
# QUANTIZATION: INT6 BIT-PACKED + INT8 FOR EMBEDDINGS
# Bit-packing: 4 int6 values → 3 bytes (saves 25% vs int8 storage)
# ============================================================

CTRL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain",
                  "skip_weight", "lora_", "layerscale")
FP16_THRESH = 65_536


def _sdclip_int6(t: Tensor, k: float = 12.85) -> Tuple[Tensor, Tensor]:
    f = t.float()
    if f.ndim == 2:
        ca = (k * f.std(dim=1)).clamp(min=1e-6)
        cl = torch.clamp(f, -ca[:, None], ca[:, None])
        sc = (ca / 31.0).clamp(min=1.0 / 31.0)
        q  = torch.clamp(torch.round(cl / sc[:, None]), -31, 31).to(torch.int8)
        return q.contiguous(), sc.to(torch.float16).contiguous()
    ca = float((k * f.std()).clamp(min=1e-6).item())
    sc = torch.tensor(ca / 31.0, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f, -ca, ca) / sc), -31, 31).to(torch.int8)
    return q.contiguous(), sc


def _sdclip_int8(t: Tensor, k: float = 20.0) -> Tuple[Tensor, Tensor]:
    f = t.float()
    if f.ndim == 2:
        ca = (k * f.std(dim=1)).clamp(min=1e-6)
        cl = torch.clamp(f, -ca[:, None], ca[:, None])
        sc = (ca / 127.0).clamp(min=1.0 / 127.0)
        q  = torch.clamp(torch.round(cl / sc[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), sc.to(torch.float16).contiguous()
    ca = float((k * f.std()).clamp(min=1e-6).item())
    sc = torch.tensor(ca / 127.0, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f, -ca, ca) / sc), -127, 127).to(torch.int8)
    return q.contiguous(), sc


def _pack6(q: Tensor) -> bytes:
    flat = (q.cpu().numpy().astype(np.int8).flatten() & 0x3F)
    n4 = (len(flat) // 4) * 4
    out = []
    for i in range(0, n4, 4):
        a, b, c, d = flat[i], flat[i+1], flat[i+2], flat[i+3]
        out += [a | ((b & 3) << 6),
                ((b >> 2) & 0xF) | ((c & 0xF) << 4),
                ((c >> 4) & 3) | ((d & 0x3F) << 2)]
    return bytes(out)


def _unpack6(data: bytes, numel: int) -> Tensor:
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(numel, dtype=np.int8)
    o = 0
    for i in range(0, len(arr) - 2, 3):
        if o + 4 > numel:
            break
        b0, b1, b2 = int(arr[i]), int(arr[i+1]), int(arr[i+2])
        vals = [b0 & 0x3F,
                ((b0 >> 6) & 3) | ((b1 & 0xF) << 2),
                ((b1 >> 4) & 0xF) | ((b2 & 3) << 4),
                (b2 >> 2) & 0x3F]
        for v in vals:
            if o < numel:
                out[o] = v if v < 32 else v - 64
                o += 1
    return torch.from_numpy(out)


def quantize_state_dict(sd: Dict) -> Dict:
    qb, sc, dt, pt = {}, {}, {}, {}
    for name, t in sd.items():
        cpu = t.detach().cpu().contiguous()
        is_fp = cpu.is_floating_point()
        if not is_fp or cpu.numel() <= FP16_THRESH or any(p in name for p in CTRL_PATTERNS):
            pt[name] = cpu.to(torch.float16) if is_fp else cpu
            continue
        dtype_str = str(cpu.dtype).replace("torch.", "")
        if "tok_emb" in name or "lm_head" in name:
            q, s = _sdclip_int8(cpu, k=20.0)
            qb[name] = q.numpy().tobytes()
        else:
            q, s = _sdclip_int6(cpu, k=12.85)
            qb[name] = _pack6(q)
        sc[name] = s
        dt[name] = dtype_str
    return {"qb": qb, "sc": sc, "dt": dt, "pt": pt}


def dequantize_state_dict(obj: Dict) -> Dict:
    out = {}
    for name, data in obj["qb"].items():
        dtype = getattr(torch, obj["dt"][name])
        s = obj["sc"][name].to(torch.float32)
        if s.ndim > 0:
            nr = s.shape[0]
            if "tok_emb" in name or "lm_head" in name:
                q = torch.frombuffer(bytearray(data), dtype=torch.int8).float()
                nc = q.numel() // nr
                out[name] = (q.view(nr, nc) * s[:, None]).to(dtype)
            else:
                total = (len(data) * 4) // 3
                nc = total // nr
                q = _unpack6(data, nr * nc).float().view(nr, nc)
                out[name] = (q * s[:, None]).to(dtype)
        else:
            total = (len(data) * 4) // 3
            q = _unpack6(data, total).float()
            out[name] = (q * float(s.item())).to(dtype)
    for name, t in obj["pt"].items():
        out[name] = t
    return out


def compress_artifact(model: nn.Module, code: str) -> Tuple[int, int, int]:
    sd = quantize_state_dict(model.state_dict())
    buf = io.BytesIO()
    torch.save(sd, buf)
    model_compressed = brotli.compress(buf.getvalue(), quality=11)
    code_compressed  = lzma.compress(code.encode(), preset=9)
    return len(code_compressed), len(model_compressed), len(code_compressed) + len(model_compressed)


# ============================================================
# MODEL COMPONENTS
# ============================================================

class RMSNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Rotary(nn.Module):
    """Partial RoPE: only first `partial_dim` head dimensions get rotated."""

    def __init__(self, head_dim: int, partial_dim: int, base: float = 10000.0):
        super().__init__()
        self.partial_dim = partial_dim
        inv = 1.0 / (base ** (torch.arange(0, partial_dim, 2, dtype=torch.float32) / partial_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cache: Optional[Tuple] = None

    def forward(self, seq: int, device, dtype):
        if self._cache is None or self._cache[0] != seq or self._cache[1] != device:
            t = torch.arange(seq, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq, device, f.cos()[None, None], f.sin()[None, None])
        return self._cache[2].to(dtype), self._cache[3].to(dtype)


def apply_rope_partial(x: Tensor, cos: Tensor, sin: Tensor, pd: int) -> Tensor:
    h = pd // 2
    xr, xp = x[..., :pd], x[..., pd:]
    x1, x2 = xr[..., :h], xr[..., h:]
    return torch.cat([torch.cat([x1*cos + x2*sin, -x2*cos + x1*sin], dim=-1), xp], dim=-1)


class LoRAAdapter(nn.Module):
    """Zero-initialized LoRA adapter. Output = alpha * B(A(x))."""

    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)   # zero init → no effect at start

    def forward(self, x: Tensor, alpha: float) -> Tensor:
        return alpha * self.B(self.A(x))


class RRTAttention(nn.Module):
    """
    CausalSelfAttention with optional per-step LoRA on Q and V.
    is_recur=True adds `recur_steps` LoRA pairs.
    """

    def __init__(self, dim: int, args: Hyperparameters, is_recur: bool):
        super().__init__()
        assert dim % args.num_heads == 0
        self.nh  = args.num_heads
        self.nkv = args.num_kv_heads
        self.hd  = dim // args.num_heads
        kd = self.nkv * self.hd

        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kd,  bias=False)
        self.c_v = CastedLinear(dim, kd,  bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((self.nh,), args.qk_gain_init))
        self.rotary = Rotary(self.hd, args.rope_partial_dim, args.rope_base)
        self.pd = args.rope_partial_dim

        if is_recur:
            self.lora_q = nn.ModuleList([LoRAAdapter(dim, dim, args.recur_lora_rank)
                                          for _ in range(args.recur_steps)])
            self.lora_v = nn.ModuleList([LoRAAdapter(dim, kd,  args.recur_lora_rank)
                                          for _ in range(args.recur_steps)])
        else:
            self.lora_q = self.lora_v = None

    def forward(self, x: Tensor, step: int = -1, alpha: float = 1.0) -> Tensor:
        B, T, D = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        if self.lora_q is not None and 0 <= step < len(self.lora_q):
            q = q + self.lora_q[step](x, alpha)
            v = v + self.lora_v[step](x, alpha)

        q = q.view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = k.view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.hd).transpose(1, 2)

        q = F.rms_norm(q, (self.hd,))
        k = F.rms_norm(k, (self.hd,))

        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope_partial(q, cos, sin, self.pd // 2)
        k = apply_rope_partial(k, cos, sin, self.pd // 2)

        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            enable_gqa=(self.nkv != self.nh))
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    """LeakyReLU(0.5)² MLP — matches SOTA stack."""

    def __init__(self, dim: int, mult: int):
        super().__init__()
        self.fc   = CastedLinear(dim, mult * dim, bias=False)
        self.proj = CastedLinear(mult * dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.leaky_relu(self.fc(x), 0.5).square())


class Block(nn.Module):
    """
    Transformer block with:
    - Parallel residuals (GPT-J style) when parallel=True
    - Layerwise attn/mlp scales
    - U-Net resid_mix gating
    - RRT attention with optional LoRA
    """

    def __init__(self, dim: int, args: Hyperparameters, is_recur: bool, parallel: bool):
        super().__init__()
        self.parallel = parallel
        self.norm1 = RMSNorm()
        self.attn  = RRTAttention(dim, args, is_recur)
        if not parallel:
            self.norm2 = RMSNorm()
        self.mlp  = MLP(dim, args.mlp_mult)
        self.as_  = nn.Parameter(torch.ones(dim))   # attn_scale
        self.ms_  = nn.Parameter(torch.ones(dim))   # mlp_scale
        self.rm_  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor, step: int = -1, alpha: float = 1.0) -> Tensor:
        rm = self.rm_.to(x.dtype)
        x  = rm[0][None, None] * x + rm[1][None, None] * x0

        if self.parallel:
            n = self.norm1(x)
            x = x + self.as_.to(x.dtype)[None, None] * self.attn(n, step, alpha)
            x = x + self.ms_.to(x.dtype)[None, None] * self.mlp(n)
        else:
            x = x + self.as_.to(x.dtype)[None, None] * self.attn(self.norm1(x), step, alpha)
            x = x + self.ms_.to(x.dtype)[None, None] * self.mlp(self.norm2(x))
        return x


# ============================================================
# MAIN MODEL: RELAXED RECURSIVE TRANSFORMER
# ============================================================

class RRTModel(nn.Module):
    """
    11-layer U-Net GPT with:
    - Depth recurrence on layers {3,4,5} with per-step LoRA (rank=4)
    - Parallel residuals on layers 7-10
    - QK-Gain 5.25, partial RoPE, LeakyReLU(0.5)²
    - LoRA alpha ramps 0→1 over lora_warmup_steps after recurrence activates
    """

    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.recur_set  = set(args.recur_layers)
        self.recur_steps = args.recur_steps
        self.lora_alpha  = 0.0          # controlled externally during training

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        enc = args.num_layers // 2
        dec = args.num_layers - enc
        self.num_enc = enc
        self.num_dec = dec
        self.skip_w  = nn.Parameter(torch.ones(min(enc, dec), args.model_dim))

        self.blocks = nn.ModuleList([
            Block(args.model_dim, args,
                  is_recur=(i in self.recur_set),
                  parallel=(i >= args.parallel_resid_from))
            for i in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()

        if args.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(args.model_dim, args.vocab_size, bias=False)
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, 0.0, self.args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def set_lora_alpha(self, alpha: float):
        self.lora_alpha = float(alpha)

    def _run(self, idx: int, x: Tensor, x0: Tensor) -> Tensor:
        if idx in self.recur_set:
            for s in range(self.recur_steps):
                x = self.blocks[idx](x, x0, step=s, alpha=self.lora_alpha)
        else:
            x = self.blocks[idx](x, x0)
        return x

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(ids), (self.args.model_dim,))
        x0, skips = x, []

        for i in range(self.num_enc):
            x = self._run(i, x, x0)
            skips.append(x)

        for j in range(self.num_dec):
            if skips:
                x = x + self.skip_w[j].to(x.dtype)[None, None] * skips.pop()
            x = self._run(self.num_enc + j, x, x0)

        x = self.final_norm(x).reshape(-1, self.args.model_dim)
        w = self.tok_emb.weight if self.args.tie_embeddings else self.lm_head.weight
        logits = self.args.logit_softcap * torch.tanh(F.linear(x, w.to(x.dtype)) / self.args.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            self.shadow[k].mul_(self.decay).add_(p.data.float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.named_parameters():
            p.data.copy_(self.shadow[k].to(p.dtype))


# ============================================================
# EVALUATION — SLIDING WINDOW (no TTT)
# ============================================================

@torch.no_grad()
def eval_sliding(args: Hyperparameters, model: nn.Module,
                  val_tokens: Tensor, device,
                  bb_lut: Tensor, sp_lut: Tensor, ib_lut: Tensor,
                  rank: int, world_size: int, grad_accum: int) -> Tuple[float, float]:
    model.eval()
    local_seqs = max(1, args.val_batch_size // (world_size * grad_accum * args.train_seq_len))
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    s0 = (total_seqs * rank) // world_size
    s1 = (total_seqs * (rank + 1)) // world_size

    tot_loss = torch.zeros((), device=device, dtype=torch.float64)
    tot_tok  = torch.zeros((), device=device, dtype=torch.float64)
    tot_byte = torch.zeros((), device=device, dtype=torch.float64)

    for bs in range(s0, s1, local_seqs):
        be  = min(bs + local_seqs, s1)
        raw = val_tokens[bs * args.train_seq_len: be * args.train_seq_len + 1].to(device, torch.int64)
        x   = raw[:-1].reshape(-1, args.train_seq_len)
        y   = raw[ 1:].reshape(-1, args.train_seq_len)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, y)
        nt = float(y.numel())
        tot_loss += loss.double() * nt
        tot_tok  += nt
        tb = bb_lut[y.reshape(-1)].to(torch.int16)
        tb += (sp_lut[y.reshape(-1)] & ~ib_lut[x.reshape(-1)]).to(torch.int16)
        tot_byte += tb.double().sum()

    if dist.is_available() and dist.is_initialized():
        for t in (tot_loss, tot_tok, tot_byte):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    model.train()
    vl  = float(tot_loss / tot_tok)
    bpb = (vl / math.log(2.0)) * float(tot_tok / tot_byte)
    return vl, bpb


# ============================================================
# LEGAL SCORE-FIRST TEST-TIME TRAINING
# Per Issue #1017 (Track B compliance):
#   1. Causality    — sliding window, strictly causal
#   2. Normalized   — standard softmax, no logit biasing
#   3. Score-first  — chunk scored under no_grad BEFORE any update
#   4. Single pass  — each token scored exactly once
# ============================================================

def eval_with_ttt(args: Hyperparameters, model: nn.Module,
                   val_tokens: Tensor, device,
                   bb_lut: Tensor, sp_lut: Tensor, ib_lut: Tensor,
                   rank: int, world_size: int) -> Tuple[float, float]:
    model.eval()
    orig = {k: v.clone() for k, v in model.state_dict().items()}

    # Freeze first ttt_freeze_layers blocks, train everything else
    frozen = set()
    for i in range(args.ttt_freeze_layers):
        for n, _ in model.named_parameters():
            if f"blocks.{i}." in n:
                frozen.add(n)

    ttt_params = [p for n, p in model.named_parameters()
                   if n not in frozen and p.requires_grad]
    opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    SL  = args.train_seq_len
    tot_loss = torch.zeros((), device=device, dtype=torch.float64)
    tot_tok  = torch.zeros((), device=device, dtype=torch.float64)
    tot_byte = torch.zeros((), device=device, dtype=torch.float64)

    n_full   = (val_tokens.numel() - 1) // args.ttt_chunk_tokens
    total_chunks = n_full

    for ci in range(total_chunks):
        # Shard chunks across ranks for distributed TTT
        if ci % world_size != rank:
            continue

        start = ci * args.ttt_chunk_tokens
        end   = start + args.ttt_chunk_tokens + 1
        chunk = val_tokens[start: min(end, val_tokens.numel())].to(device, torch.int64)
        if chunk.numel() < SL + 1:
            break

        usable = ((chunk.numel() - 1) // SL) * SL
        x = chunk[:usable    ].reshape(-1, SL)
        y = chunk[1:usable + 1].reshape(-1, SL)

        # ── STEP 1: SCORE (official graded prediction, no grad) ──
        with torch.inference_mode():
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            nt = float(y.numel())
            tot_loss += loss.double() * nt
            tot_tok  += nt
            tb  = bb_lut[y.reshape(-1)].to(torch.int16)
            tb += (sp_lut[y.reshape(-1)] & ~ib_lut[x.reshape(-1)]).to(torch.int16)
            tot_byte += tb.double().sum()

        # ── STEP 2: TRAIN on already-scored tokens ──
        model.train()
        n_seqs  = x.shape[0]
        for epoch in range(args.ttt_epochs):
            # Cosine LR decay across epochs
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * (epoch + 1) / args.ttt_epochs))
            for pg in opt.param_groups:
                pg["lr"] = args.ttt_lr * lr_scale

            perm = torch.randperm(n_seqs, device=device)
            for bs in range(0, n_seqs, 16):
                bi  = perm[bs: bs + 16]
                xb, yb = x[bi], y[bi]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", torch.bfloat16):
                    l = model(xb, yb)
                l.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                opt.step()

        model.eval()

    # All-reduce scores across ranks
    if dist.is_available() and dist.is_initialized():
        for t in (tot_loss, tot_tok, tot_byte):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Restore weights — TTT is eval-only
    model.load_state_dict(orig)
    model.train()

    vl  = float(tot_loss / tot_tok.clamp(min=1))
    bpb = (vl / math.log(2.0)) * float(tot_tok / tot_byte.clamp(min=1))
    return vl, bpb


# ============================================================
# TRAINING UTILITIES
# ============================================================

CTRL = ("attn_scale", "mlp_scale", "resid_mix", "q_gain",
         "skip_weight", "lora_", "layerscale", "as_", "ms_", "rm_")


def restore_fp32(model: nn.Module):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (p.ndim < 2 or any(c in n for c in CTRL)) and p.dtype != torch.float32:
                p.data = p.data.float()


def build_param_groups(model: nn.Module, args: Hyperparameters):
    """
    Three-way split:
      - 2D weight matrices (excl. embeddings/head) → MuonEqR
      - Embeddings + lm_head                        → AdamW (high LR)
      - Scalars / biases / control params            → AdamW (low LR)
    """
    muon, embed, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_emb  = "tok_emb" in name or "lm_head" in name
        is_ctrl = any(c in name for c in CTRL)
        if is_emb:
            embed.append(p)
        elif p.ndim >= 2 and not is_ctrl:
            muon.append(p)
        else:
            scalar.append(p)
    return muon, embed, scalar


# ============================================================
# MAIN
# ============================================================

def main():
    code = Path(__file__).read_text()
    args = Hyperparameters()

    # ── Distributed setup ──
    distributed = "RANK" in os.environ
    rank        = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 0 and 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum  = max(1, 8 // world_size)
    grad_scale  = 1.0 / grad_accum

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group("nccl", device_id=device)
        dist.barrier()

    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_flash_sdp, enable_math_sdp,
                                      enable_mem_efficient_sdp, enable_cudnn_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # ── Model ──
    model = RRTModel(args).to(device)
    restore_fp32(model)

    if master:
        total  = sum(p.numel() for p in model.parameters())
        lora_n = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        print(f"RRTModel | params={total:,} | lora={lora_n:,}")
        print(f"  recur={args.recur_layers} steps={args.recur_steps} rank={args.recur_lora_rank}")
        print(f"  parallel_resid_from={args.parallel_resid_from} qk_gain={args.qk_gain_init}")

    ema   = EMA(model, args.ema_decay)
    model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if distributed else model

    # ── Optimizers ──
    muon_p, embed_p, scalar_p = build_param_groups(raw, args)
    opt_muon = MuonEqR(muon_p, lr=args.matrix_lr,
                        momentum=args.muon_momentum,
                        backend_steps=args.muon_backend_steps)
    emb_lr   = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_adam = torch.optim.AdamW(
        [{"params": embed_p,  "lr": emb_lr},
         {"params": scalar_p, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.weight_decay, fused=True)

    # ── Data & tokenizer ──
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    bb_lut, sp_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens   = load_validation_tokens(args.val_files, args.train_seq_len)
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Schedule helpers ──
    WD   = args.warmdown_iters
    WU   = args.warmup_steps
    ITERS = args.iterations
    recur_start = int(args.recur_activate_frac * ITERS)

    def lr_scale(step: int) -> float:
        if step < WU:
            return step / max(1, WU)
        if step >= ITERS - WD:
            return max(0.0, (ITERS - step) / max(1, WD))
        return 1.0

    # ── Training loop ──
    t0 = time.time()
    loss_acc = 0.0
    model.train()

    for step in range(ITERS):
        if time.time() - t0 > args.max_wallclock_seconds:
            if master:
                print(f"Wallclock {args.max_wallclock_seconds}s hit at step {step}")
            break

        # LoRA alpha schedule
        if step < recur_start:
            alpha = 0.0
        else:
            alpha = min(1.0, (step - recur_start) / max(1, args.lora_warmup_steps))
        raw.set_lora_alpha(alpha)

        # LR
        lrs = lr_scale(step)
        for g in opt_muon.param_groups:
            g["lr"] = args.matrix_lr * lrs
        adam_base = [emb_lr, args.scalar_lr]
        for i, g in enumerate(opt_adam.param_groups):
            g["lr"] = adam_base[i] * lrs

        # Muon momentum warmup
        if step < args.muon_mom_warmup_steps:
            m = args.muon_mom_warmup_start + (
                (args.muon_momentum - args.muon_mom_warmup_start)
                * step / args.muon_mom_warmup_steps)
            for g in opt_muon.param_groups:
                g["momentum"] = m

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            step_loss += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        opt_muon.step()
        opt_adam.step()
        ema.update(raw)
        loss_acc += step_loss

        if master and step % args.train_log_every == 0:
            avg = loss_acc / args.train_log_every if step else step_loss
            loss_acc = 0.0
            print(f"step={step:5d} loss={avg:.4f} lr={lrs:.4f} α={alpha:.3f} t={time.time()-t0:.0f}s")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0:
            ema.copy_to(raw)
            vl, vb = eval_sliding(args, raw, val_tokens, device,
                                   bb_lut, sp_lut, ib_lut, rank, world_size, grad_accum)
            if master:
                print(f"  [sliding] val_loss={vl:.4f} val_bpb={vb:.4f}")
            restore_fp32(raw)
            model.train()

    # ── Final evaluation ──
    ema.copy_to(raw)

    if master:
        print("\n── Final sliding eval ──")
    vl_s, vb_s = eval_sliding(args, raw, val_tokens, device,
                                bb_lut, sp_lut, ib_lut, rank, world_size, grad_accum)

    vl_t, vb_t = vl_s, vb_s
    if args.ttt_enabled:
        if master:
            print("── Final TTT eval ──")
        vl_t, vb_t = eval_with_ttt(args, raw, val_tokens, device,
                                     bb_lut, sp_lut, ib_lut, rank, world_size)

    if master:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"  Sliding  val_bpb = {vb_s:.4f}")
        if args.ttt_enabled:
            print(f"  TTT      val_bpb = {vb_t:.4f}  (gain: {vb_s - vb_t:.4f})")
        print(f"{'='*60}")
        cb, mb, tb_sz = compress_artifact(raw, code)
        print(f"Artifact: code={cb:,}B  model={mb:,}B  total={tb_sz:,}B")
        print(f"Under 16MB: {tb_sz < 16_000_000}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
