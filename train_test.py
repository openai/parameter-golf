"""
K-Splanifold Multi-Spine Transformer
══════════════════════════════════════════════════════════════
Novel approach: Replace transformer MLP with K-Splanifold structured
manifold evaluation (Adams, 2024). Instead of learned weight matrices,
each MLP block evaluates a cubic Hermite spine-deviation manifold.

Architecture:
  SP8192 · 16-20L × 512d · 8H/4KV (GQA)
  MultiSpineMLP: 8 independent K-Splanifolds in 64-dim subspaces
  Each spine: learned projection t, Hermite interp Psi(t) + Delta(t,delta)
  Control matrices stored as low-rank U@V (rank=16), int8+LZMA artifact
  Full-rank attention (CastedLinear + Muon)
  U-Net skip connections · Logit softcap=30 · Tied embeddings
  EMA 0.9965 (delayed 30%) · AdamW for splanifold controls
  Sliding window eval stride=16 · 50k steps non-record track

Key properties vs standard MLP:
  - Structured cubic interpolation instead of arbitrary learned map
  - ~365k params/layer vs ~2.1M for 4x MLP (5.7x fewer)
  - Lower entropy-coded bits/param (3.71 vs 5.94, paper Table 4)
  - Allows more layers within 16MB artifact budget
  - First application of K-Splanifolds to language modeling

Reference: K-Splanifolds: Advancing General Purpose Regression with
Linear-Time Parametric Spline Manifolds (Adams, 2024)
"""
from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import lzma
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ══════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path,           "fineweb_train_*.bin")
    val_files      = os.path.join(data_path,           "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",         str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",       42))

    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY",   1000))
    train_log_every  = int(os.environ.get("TRAIN_LOG_EVERY",  200))
    # Sliding window eval — stride=16 matches competition evaluator
    sliding_stride   = int(os.environ.get("SLIDING_STRIDE",   16))

    iterations       = int(os.environ.get("ITERATIONS",       50000))
    warmdown_iters   = int(os.environ.get("WARMDOWN_ITERS",   10000))  # 20% warmdown
    warmup_steps     = int(os.environ.get("WARMUP_STEPS",     20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len    = int(os.environ.get("TRAIN_SEQ_LEN",    1024))
    max_wallclock_s  = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))  # 0 = no cap

    # Architecture
    vocab_size    = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers    = int(os.environ.get("NUM_LAYERS",    16))
    num_kv_heads  = int(os.environ.get("NUM_KV_HEADS",  4))
    model_dim     = int(os.environ.get("MODEL_DIM",     512))
    num_heads     = int(os.environ.get("NUM_HEADS",     8))
    rope_base     = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qk_gain_init  = float(os.environ.get("QK_GAIN_INIT",  1.5))
    tied_embed_std = float(os.environ.get("TIED_EMBED_STD", 0.005))

    # K-Splanifold MLP hyperparameters
    # num_spines: number of independent splanifold heads (like attn heads)
    # spine_rank: rank of low-rank control matrix factorizations
    # delta_max: deviation warp radius (controls extrapolation behavior)
    num_spines   = int(os.environ.get("NUM_SPINES",   8))
    spine_rank   = int(os.environ.get("SPINE_RANK",   16))
    delta_max    = float(os.environ.get("DELTA_MAX",  2.0))
    eps_sigma    = float(os.environ.get("EPS_SIGMA",  0.02))  # smooth weight reg

    # EMA
    ema_decay      = float(os.environ.get("EMA_DECAY",      0.9965))
    ema_start_frac = float(os.environ.get("EMA_START_FRAC", 0.30))

    # Optimiser
    matrix_lr      = float(os.environ.get("MATRIX_LR",      0.04))
    spine_lr       = float(os.environ.get("SPINE_LR",       0.008))
    embed_lr       = float(os.environ.get("EMBED_LR",       0.05))
    scalar_lr      = float(os.environ.get("SCALAR_LR",      0.04))
    muon_momentum  = float(os.environ.get("MUON_MOMENTUM",  0.99))
    muon_ns_steps  = int(os.environ.get("MUON_NS_STEPS",    5))
    muon_mom_warmup_start = float(os.environ.get("MUON_MOM_WARMUP_START", 0.92))
    muon_mom_warmup_steps = int(os.environ.get("MUON_MOM_WARMUP_STEPS",   1500))
    beta1          = float(os.environ.get("BETA1",           0.9))
    beta2          = float(os.environ.get("BETA2",           0.95))
    adam_eps       = float(os.environ.get("ADAM_EPS",        1e-8))
    weight_decay   = float(os.environ.get("WEIGHT_DECAY",   0.01))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


# ══════════════════════════════════════════════════════════════
# MUON OPTIMIZER
# ══════════════════════════════════════════════════════════════
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a*X + (b*A + c*A@A) @ X
    return X.T if G.size(0) > G.size(1) else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, ns_steps=5, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      ns_steps=ns_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        don = dist.is_available() and dist.is_initialized()
        ws  = dist.get_world_size() if don else 1
        rk  = dist.get_rank()       if don else 0
        for group in self.param_groups:
            ps   = group["params"]; lr = group["lr"]
            mom  = group["momentum"]; ns = group["ns_steps"]
            nest = group["nesterov"]
            total = sum(p.numel() for p in ps)
            flat  = torch.zeros(total, device=ps[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(ps):
                if i % ws == rk and p.grad is not None:
                    g = p.grad.float()
                    st = self.state[p]
                    if "buf" not in st: st["buf"] = torch.zeros_like(g)
                    st["buf"].mul_(mom).add_(g)
                    g = g.add(st["buf"], alpha=mom) if nest else st["buf"].clone()
                    g = zeropower_via_newtonschulz5(g, ns)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[cur:cur+p.numel()] = g.reshape(-1)
                cur += p.numel()
            if don: dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            cur = 0
            for p in ps:
                p.add_(flat[cur:cur+p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                cur += p.numel()
        return loss


# ══════════════════════════════════════════════════════════════
# TOKENISER UTILITIES
# ══════════════════════════════════════════════════════════════
def build_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size()); sz = max(sv, vocab_size)
    bb = np.zeros((sz,), dtype=np.int16)
    hl = np.zeros((sz,), dtype=np.bool_)
    ib = np.ones ((sz,), dtype=np.bool_)
    for t in range(sv):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        piece = sp.id_to_piece(t)
        if piece.startswith("▁"): hl[t] = True; piece = piece[1:]
        bb[t] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hl, dtype=torch.bool,  device=device),
            torch.tensor(ib, dtype=torch.bool,  device=device))


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
def load_shard(file: Path) -> Tensor:
    hb = 256 * np.dtype("<i4").itemsize
    hd = np.fromfile(file, dtype="<i4", count=256)
    if hd.size != 256 or int(hd[0]) != 20240520 or int(hd[1]) != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(hd[2])
    if file.stat().st_size != hb + n*2: raise ValueError(f"Size mismatch: {file}")
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=n, offset=hb).astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(pattern)
        self.fi = 0; self.pos = 0; self.tokens = load_shard(self.files[0])
    def _adv(self):
        self.fi = (self.fi+1) % len(self.files)
        self.tokens = load_shard(self.files[self.fi]); self.pos = 0
    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            av = self.tokens.numel() - self.pos
            if av <= 0: self._adv(); continue
            k = min(rem, av)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DDPLoader:
    def __init__(self, pattern, rank, ws, device):
        self.rank = rank; self.ws = ws; self.dev = device
        self.stream = TokenStream(pattern)
    def next_batch(self, total, seq_len, accum):
        local = total // (self.ws * accum); span = local + 1
        chunk = self.stream.take(span * self.ws); s = self.rank * span
        loc = chunk[s:s+span].to(torch.int64)
        x = loc[:-1].reshape(-1, seq_len); y = loc[1:].reshape(-1, seq_len)
        return x.to(self.dev, non_blocking=True), y.to(self.dev, non_blocking=True)


def load_val(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(pattern)
    tokens = torch.cat([load_shard(f) for f in files]).contiguous()
    u = ((tokens.numel()-1) // seq_len) * seq_len
    return tokens[:u+1]


# ══════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._cached_seq = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None
    def forward(self, seq, device, dtype):
        if self._cos is None or self._cached_seq != seq or self._cos.device != device:
            t = torch.arange(seq, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None]; self._sin = f.sin()[None, None]
            self._cached_seq = seq
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rope(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1*cos + x2*sin, x1*(-sin) + x2*cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        D, H, Hkv = args.model_dim, args.num_heads, args.num_kv_heads
        self.H = H; self.Hkv = Hkv; self.hD = D // H
        self.kv_groups = H // Hkv
        kv_dim = Hkv * self.hD
        self.c_q   = CastedLinear(D, D, bias=False)
        self.c_k   = CastedLinear(D, kv_dim, bias=False)
        self.c_v   = CastedLinear(D, kv_dim, bias=False)
        self.proj  = CastedLinear(D, D, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((H,), args.qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.hD, base=args.rope_base)
    def forward(self, x):
        B, T, D = x.shape; H, Hkv, hD = self.H, self.Hkv, self.hD
        q = self.c_q(x).reshape(B, T, H,   hD).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, Hkv, hD).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, Hkv, hD).transpose(1, 2)
        q = F.rms_norm(q, (hD,)); k = F.rms_norm(k, (hD,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)
            v = v.repeat_interleave(self.kv_groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


# ══════════════════════════════════════════════════════════════
# K-SPLANIFOLD MULTI-SPINE MLP
# ══════════════════════════════════════════════════════════════
class KSplanifoldSpine(nn.Module):
    """
    Single K-Splanifold spine in a head_dim subspace.

    Maps input r in [0,1]^head_dim to output in R^head_dim via:
        t    = sigmoid(r @ spine_proj)          spine coordinate in (0,1)
        delta = r - t * ones / head_dim         zero-mean deviation
        delta_tilde = warp(delta)               bounded deviation
        w    = smooth_weights(r)                tangent blending weights
        Psi  = hermite(P0, P1, V0(w), V1(w), t) spine curve
        Delta = hermite(E0@delta_t, E1@delta_t,
                        Ep0@delta_t, Ep1@delta_t, t) transverse displacement
        out  = Psi + Delta

    Control matrices stored as low-rank: E = U @ V, U:(D,R), V:(R,D)
    This gives O(D*R) storage vs O(D^2) for full rank.

    Reference: K-Splanifolds (Adams, 2024), equations (1-15)
    """
    def __init__(self, head_dim: int, rank: int, delta_max: float, eps_sigma: float):
        super().__init__()
        D, R = head_dim, rank
        self.D = D; self.R = R
        self.delta_max = delta_max
        self.eps_sigma = eps_sigma

        # Spine direction: learned projection to scalar t
        self.spine_proj = nn.Parameter(torch.randn(D) / math.sqrt(D))

        # Anchor points P0, P1 on the spine
        self.P0 = nn.Parameter(torch.zeros(D))
        self.P1 = nn.Parameter(torch.randn(D) * 0.02)

        # Spine tangent contribution matrices P'0, P'1: (D, D) as U@V
        self.Pp0_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.Pp0_V = nn.Parameter(torch.zeros(R, D))
        self.Pp1_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.Pp1_V = nn.Parameter(torch.zeros(R, D))

        # Transverse basis matrices E0, E1: (D, D) as U@V
        self.E0_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.E0_V = nn.Parameter(torch.eye(D)[:R] if R <= D else torch.randn(R, D) / math.sqrt(D))
        self.E1_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.E1_V = nn.Parameter(torch.eye(D)[:R] if R <= D else torch.randn(R, D) / math.sqrt(D))

        # Transverse basis derivative matrices E'0, E'1: (D, D) as U@V
        self.Ep0_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.Ep0_V = nn.Parameter(torch.zeros(R, D))
        self.Ep1_U = nn.Parameter(torch.randn(D, R) / math.sqrt(R))
        self.Ep1_V = nn.Parameter(torch.zeros(R, D))

    def _hermite(self, A: Tensor, B: Tensor, dA: Tensor, dB: Tensor, t: Tensor) -> Tensor:
        """
        Cubic Hermite interpolation.
        A, B:   (N, D) endpoint values
        dA, dB: (N, D) endpoint derivatives
        t:      (N,)   parameter in [0,1]
        Returns (N, D)
        Reference: equation (1) in paper
        """
        t = t.unsqueeze(-1)  # (N, 1)
        t2 = t * t; t3 = t2 * t
        h00 =  2*t3 - 3*t2 + 1
        h01 = -2*t3 + 3*t2
        h10 = t3 - 2*t2 + t
        h11 = t3 - t2
        return h00*A + h01*B + h10*dA + h11*dB

    def _smooth_weights(self, u: Tensor) -> Tensor:
        """
        Smooth tangent-blending weights (equation 7).
        u: (N, D) in [0,1]^D
        Returns w: (N, D), sum(w, dim=-1) = 1
        """
        sigma = u.sum(dim=-1, keepdim=True)  # (N, 1)
        eps2  = self.eps_sigma ** 2
        w = (sigma * u + eps2 / self.D) / (sigma**2 + eps2)
        return w  # (N, D)

    def _warp(self, delta: Tensor) -> Tensor:
        """
        Deviation warp for bounded transverse magnitude (equation 5).
        delta: (N, D) zero-mean deviation
        Returns delta_tilde: (N, D) bounded deviation
        """
        dn = delta.norm(dim=-1, keepdim=True)  # (N, 1)
        return delta / (1.0 + (dn / self.delta_max).pow(2)).sqrt()

    def _lr_apply(self, U: Tensor, V: Tensor, x: Tensor) -> Tensor:
        """
        Apply low-rank matrix (U@V) to x in row-vector convention.
        U: (D, R), V: (R, D), x: (N, D)
        x @ (U@V)^T = x @ V^T @ U^T
        (N,D)@(D,R)=(N,R);  (N,R)@(R,D)=(N,D). Never materializes (D,D).
        """
        return (x @ V.T) @ U.T

    def forward(self, r: Tensor) -> Tensor:
        """
        r: (N, D) inputs in [0,1]^D
        Returns: (N, D) splanifold output
        """
        N, D = r.shape

        # 1. Spine coordinate t = sigmoid(r @ spine_proj) in (0, 1)
        t = torch.sigmoid(r @ self.spine_proj)  # (N,)

        # 2. Deviation from spine direction
        # The "spine direction" in input space corresponds to spine_proj
        # Project out the spine component to get deviation
        sp_norm = F.normalize(self.spine_proj.unsqueeze(0), dim=-1)  # (1, D)
        proj_onto_spine = (r @ sp_norm.T) * sp_norm  # (N, D)
        delta = r - proj_onto_spine  # (N, D) deviation perpendicular to spine

        # 3. Deviation warp for bounded extrapolation
        delta_tilde = self._warp(delta)  # (N, D)

        # 4. Smooth tangent weights for blending per-dim spine tangents
        w = self._smooth_weights(r)  # (N, D)

        # 5. Spine anchor derivatives (blended tangent contributions)
        # V0(u) = P'0 @ w(u), applied as low-rank: w @ (U@V)^T
        dV0 = self._lr_apply(self.Pp0_U, self.Pp0_V, w)  # (N, D)
        dV1 = self._lr_apply(self.Pp1_U, self.Pp1_V, w)  # (N, D)

        # 6. Spine curve Psi(t)
        P0 = self.P0.unsqueeze(0).expand(N, -1)  # (N, D)
        P1 = self.P1.unsqueeze(0).expand(N, -1)
        psi = self._hermite(P0, P1, dV0, dV1, t)  # (N, D)

        # 7. Transverse displacement Delta(t, delta_tilde)
        # D_i = E_i @ delta_tilde (endpoint displacements)
        D0 = self._lr_apply(self.E0_U, self.E0_V, delta_tilde)   # (N, D)
        D1 = self._lr_apply(self.E1_U, self.E1_V, delta_tilde)
        # T_i = E'_i @ delta_tilde (endpoint displacement derivatives)
        T0 = self._lr_apply(self.Ep0_U, self.Ep0_V, delta_tilde)  # (N, D)
        T1 = self._lr_apply(self.Ep1_U, self.Ep1_V, delta_tilde)
        delta_out = self._hermite(D0, D1, T0, T1, t)              # (N, D)

        # 8. Splanifold output: spine + transverse
        return psi + delta_out  # (N, D)
class FusedMultiSpineMLP(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        D = args.model_dim
        S = args.num_spines
        assert D % S == 0

        self.S = S
        self.D = D
        self.H = D // S   # head_dim
        R = args.spine_rank

        self.delta_max = args.delta_max
        self.eps_sigma = args.eps_sigma

        self.input_norm = RMSNorm()

        # ===== spine params (stacked) =====
        self.spine_proj = nn.Parameter(torch.randn(S, self.H) / math.sqrt(self.H))

        self.P0 = nn.Parameter(torch.zeros(S, self.H))
        self.P1 = nn.Parameter(torch.randn(S, self.H) * 0.02)

        # low-rank stacks: (S, D, R) and (S, R, D)
        def lr():
            return (
                nn.Parameter(torch.randn(S, self.H, R) / math.sqrt(R)),
                nn.Parameter(torch.zeros(S, R, self.H))
            )

        self.Pp0_U, self.Pp0_V = lr()
        self.Pp1_U, self.Pp1_V = lr()

        self.E0_U, self.E0_V = lr()
        self.E1_U, self.E1_V = lr()

        self.Ep0_U, self.Ep0_V = lr()
        self.Ep1_U, self.Ep1_V = lr()

        self.out_proj = CastedLinear(D, D, bias=False)
        self.out_proj._zero_init = True

        self.gate = nn.Parameter(torch.zeros(1))

    def lr_apply(self, x, U, V):
        # x: (N, S, D)
        # U: (S, D, R), V: (S, R, D)
        tmp = torch.einsum("nsd,srd->nsr", x, V)
        out = torch.einsum("nsr,sdr->nsd", tmp, U)
        return out

    def hermite(self, A, B, dA, dB, t):
        # all shapes (N, S, D), t: (N, S)
        t = t.unsqueeze(-1)
        t2 = t * t
        t3 = t2 * t
        h00 =  2*t3 - 3*t2 + 1
        h01 = -2*t3 + 3*t2
        h10 = t3 - 2*t2 + t
        h11 = t3 - t2
        return h00*A + h01*B + h10*dA + h11*dB

    def forward(self, x):
        B, T, D = x.shape
        N = B * T

        x_norm = self.input_norm(x)
        r = (torch.tanh(x_norm) + 1) / 2
        r = r.view(N, self.S, self.H)

        # ===== spine coordinate =====
        t = torch.sigmoid(torch.einsum("nsd,sd->ns", r, self.spine_proj))

        # ===== deviation =====
        sp = F.normalize(self.spine_proj, dim=-1)  # (S, H)
        proj = torch.einsum("nsd,sd->ns", r, sp)
        proj = proj.unsqueeze(-1) * sp.unsqueeze(0)
        delta = r - proj

        # ===== warp =====
        dn = delta.norm(dim=-1, keepdim=True)
        delta_tilde = delta / torch.sqrt(1 + (dn / self.delta_max) ** 2)

        # ===== smooth weights =====
        sigma = r.sum(dim=-1, keepdim=True)
        eps2 = self.eps_sigma ** 2
        w = (sigma * r + eps2 / self.H) / (sigma**2 + eps2)

        # ===== spine derivatives =====
        dV0 = self.lr_apply(w, self.Pp0_U, self.Pp0_V)
        dV1 = self.lr_apply(w, self.Pp1_U, self.Pp1_V)

        # ===== spine curve =====
        P0 = self.P0.unsqueeze(0)
        P1 = self.P1.unsqueeze(0)
        psi = self.hermite(P0, P1, dV0, dV1, t)

        # ===== transverse =====
        D0 = self.lr_apply(delta_tilde, self.E0_U, self.E0_V)
        D1 = self.lr_apply(delta_tilde, self.E1_U, self.E1_V)
        T0 = self.lr_apply(delta_tilde, self.Ep0_U, self.Ep0_V)
        T1 = self.lr_apply(delta_tilde, self.Ep1_U, self.Ep1_V)

        delta_out = self.hermite(D0, D1, T0, T1, t)

        out = psi + delta_out  # (N, S, H)

        # reshape back
        out = out.reshape(B, T, D)
        proj = self.out_proj(out)

        return x + torch.sigmoid(self.gate) * proj

class MultiSpineMLP(nn.Module):
    """
    K-Splanifold Multi-Spine MLP replacing standard transformer MLP.

    Uses num_spines independent splanifold heads, each operating in a
    head_dim=model_dim//num_spines subspace. Outputs concatenated and
    projected back to model_dim via a learned linear.

    Input normalization: r = (tanh(rmsnorm(x)) + 1) / 2 maps to [0,1]^D
    Gated residual output: x + sigmoid(gate) * proj(concat(spine_outs))

    Parameter count per layer (D=512, num_spines=8, rank=16):
      - 8 spines × ~12k params = ~96k
      - output projection: 512×512 = 262k
      - gate: 1
      - Total: ~358k vs ~2.1M for 4x standard MLP

    This allows ~5.7x more layers within the same parameter budget,
    or dramatically better compression at matched depth.
    """
    def __init__(self, args: Hyperparameters):
        super().__init__()
        D = args.model_dim
        S = args.num_spines
        assert D % S == 0, f"model_dim {D} must be divisible by num_spines {S}"
        self.head_dim = D // S
        self.num_spines = S
        self.D = D

        # Input normalization before splanifold
        self.input_norm = RMSNorm()

        # Independent splanifold per head
        self.spines = nn.ModuleList([
            KSplanifoldSpine(
                head_dim=self.head_dim,
                rank=args.spine_rank,
                delta_max=args.delta_max,
                eps_sigma=args.eps_sigma,
            )
            for _ in range(S)
        ])

        # Output projection: concat(spine_outputs) -> D
        self.out_proj = CastedLinear(D, D, bias=False)
        self.out_proj._zero_init = True  # zero init for stable residual start

        # Learned gate for residual (sigmoid gate = starts at 0.5)
        # Zero-init so gate starts near 0.5 but early gradients shrink it
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        N = B * T

        # Normalize and map to [0,1]^D
        x_norm = self.input_norm(x)           # (B, T, D)
        x_tanh = torch.tanh(x_norm)           # (B, T, D) in (-1, 1)
        r = (x_tanh + 1.0) / 2.0             # (B, T, D) in (0, 1)
        r_flat = r.reshape(N, D)              # (N, D)

        # Split into head chunks and evaluate each spine
        spine_outs = []
        for i, spine in enumerate(self.spines):
            start = i * self.head_dim
            end   = start + self.head_dim
            r_head = r_flat[:, start:end]     # (N, head_dim)
            out_head = spine(r_head)          # (N, head_dim)
            spine_outs.append(out_head)

        # Concatenate and project
        concat = torch.cat(spine_outs, dim=-1)  # (N, D)
        proj   = self.out_proj(concat.reshape(B, T, D))  # (B, T, D)

        # Gated residual
        gate = torch.sigmoid(self.gate)
        return x + gate * proj


# ══════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK + GPT
# ══════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, args: Hyperparameters, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.attn = CausalSelfAttention(args)
        self.mlp = FusedMultiSpineMLP(args)
        self.attn_scale = nn.Parameter(torch.ones(args.model_dim, dtype=torch.float32))
        # resid_mix blends current hidden state with input embedding (U-Net style)
        self.resid_mix = nn.Parameter(
            torch.stack([torch.ones(args.model_dim), torch.zeros(args.model_dim)]).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x   = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        # MultiSpineMLP has its own internal RMSNorm before splanifold evaluation.
        # Pass raw x so the residual connection inside MultiSpineMLP is correct.
        x   = self.mlp(x)
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.tok_emb  = nn.Embedding(args.vocab_size, args.model_dim)
        self.head_cor = nn.Parameter(
            torch.randn(args.vocab_size, args.model_dim) * args.tied_embed_std)
        self.blocks   = nn.ModuleList([Block(args, i) for i in range(args.num_layers)])
        self.norm_out = RMSNorm()
        self.logit_softcap = args.logit_softcap

        # U-Net skip gates: encoder half saves activations, decoder half adds them
        self.num_enc = args.num_layers // 2
        self.num_dec = args.num_layers - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        self.skip_ws = nn.Parameter(
            torch.ones(self.num_skip, args.model_dim, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=self.args.tied_embed_std)
        for m in self.modules():
            if isinstance(m, CastedLinear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)
            elif isinstance(m, CastedLinear):
                nn.init.normal_(m.weight, std=0.02)

    def set_attn_fp32(self):
        """Keep attention CastedLinear weights in fp32."""
        for m in self.modules():
            if isinstance(m, CastedLinear):
                m.float()

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        B, T = x.shape
        h  = self.tok_emb(x)                     # (B, T, D)
        h  = F.rms_norm(h, (h.size(-1),))        # normalize embedding
        x0 = h                                    # input injection anchor

        # Encoder half — save activations for U-Net skips
        enc_acts: list[Tensor | None] = [None] * self.num_enc
        for i in range(self.num_enc):
            h = self.blocks[i](h, x0)
            enc_acts[i] = h

        # Decoder half — add mirrored encoder activations
        for i in range(self.num_dec):
            skip_i = self.num_enc - 1 - i
            if 0 <= skip_i < self.num_skip and enc_acts[skip_i] is not None:
                sw = self.skip_ws[skip_i].to(dtype=h.dtype)[None, None, :]
                h  = h + sw * enc_acts[skip_i]
            h = self.blocks[self.num_enc + i](h, x0)

        h = self.norm_out(h)
        W = self.tok_emb.weight.to(h.dtype) + self.head_cor.to(h.dtype)
        logits = h @ W.T
        sc = self.logit_softcap
        logits = sc * torch.tanh(logits / sc)

        if y is None: return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


# ══════════════════════════════════════════════════════════════
# EMA
# ══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model: GPT, decay: float):
        self.model = model; self.decay = decay
        self.shadow: dict[str, Tensor] = {}
    def init(self):
        with torch.no_grad():
            self.shadow = {n: p.detach().clone().float()
                           for n, p in self.model.named_parameters()}
    def update(self):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n].lerp_(p.detach().float(), 1.0 - self.decay)
    def apply(self):
        bk = {}
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    bk[n] = p.data.clone()
                    p.data.copy_(self.shadow[n].to(p.dtype))
        return bk
    def restore(self, bk):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in bk: p.data.copy_(bk[n])


# ══════════════════════════════════════════════════════════════
# QUANTIZATION — int8 + LZMA
# ══════════════════════════════════════════════════════════════
CTRL_PATTERNS = ("q_gain", "skip_ws", "gate", "spine_proj", "P0", "P1",
                 "attn_scale", "resid_mix", "head_cor")

def _is_ctrl(name):
    return any(p in name for p in CTRL_PATTERNS)

def quantize_tensor(t: Tensor):
    """Per-row int8 for 2D, per-tensor for 1D."""
    t32 = t.float()
    if t32.ndim == 2:
        clip = torch.quantile(t32.abs(), 0.9999, dim=1).clamp_min(1e-8)
        scale = (clip / 127.0).clamp_min(1.0/127.0)
        q = (t32.clamp(-clip[:, None], clip[:, None]) / scale[:, None]
             ).round().clamp(-127, 127).to(torch.int8)
        return q.cpu().contiguous(), scale.to(torch.float16).cpu().contiguous()
    clip = float(torch.quantile(t32.abs().flatten(), 0.9999).item())
    scale = max(clip / 127.0, 1e-8 / 127.0)
    q = (t32.clamp(-clip, clip) / scale).round().clamp(-127, 127).to(torch.int8)
    return q.cpu().contiguous(), torch.tensor([scale], dtype=torch.float32).cpu()

def dequantize_tensor(q: Tensor, scale: Tensor, dtype) -> Tensor:
    q32 = q.float()
    if q32.ndim == 2:
        s = scale.float().to(q.device)
        return (q32 * s[:, None]).to(dtype)
    return (q32 * float(scale[0].item())).to(dtype)

def build_artifact(model: GPT, path: str):
    import pickle
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    quant, scales, passthr, dtypes = {}, {}, {}, {}
    for name, t in state.items():
        if not t.is_floating_point() or t.numel() < 512 or _is_ctrl(name):
            passthr[name] = t.to(torch.float16).contiguous()
            continue
        q, s = quantize_tensor(t)
        quant[name] = q; scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    obj = {"quant": quant, "scales": scales, "pass": passthr, "dtypes": dtypes,
           "fmt": "ksplan_int8_v1"}
    buf = io.BytesIO(); pickle.dump(obj, buf, protocol=4)
    blob = lzma.compress(buf.getvalue(), preset=9)
    with open(path, "wb") as f: f.write(blob)
    MB = os.path.getsize(path) / 1e6
    print(f"[artifact] {path}  {MB:.3f} MB {'✓' if MB <= 16 else '⚠ OVER'}", flush=True)
    return MB

def load_artifact(path: str, args: Hyperparameters, device):
    import pickle
    with open(path, "rb") as f: blob = f.read()
    obj = pickle.loads(lzma.decompress(blob))
    state = {}
    for name, q in obj["quant"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        state[name] = dequantize_tensor(q, obj["scales"][name], dtype)
    for name, t in obj["pass"].items():
        orig = obj.get("orig_dtypes", {}).get(name)
        state[name] = t.to(getattr(torch, orig) if orig else torch.float32)
    model = GPT(args).to(device)
    model.load_state_dict(state, strict=False)
    return model

@torch.no_grad()
def eval_fast(args, model, val_tokens, device, max_batches=32):
    model.eval()
    losses = []
    seq_len = args.train_seq_len

    # sample a few chunks instead of full sweep
    for _ in range(max_batches):
        i = torch.randint(0, val_tokens.numel() - seq_len - 1, (1,)).item()
        chunk = val_tokens[i:i+seq_len+1].to(device)

        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
# ══════════════════════════════════════════════════════════════
# SLIDING WINDOW EVALUATION (competition standard, stride=16)
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_sliding(args, model, val_tokens, bb, hl, ib, rank, ws, device):
    """
    Sliding window BPB evaluation matching competition evaluator.
    stride=16: each window advances 16 tokens, only new tokens are scored.
    This is the number that goes on the leaderboard.
    """
    T = min(args.train_seq_len, val_tokens.numel() // 2)
    stride = args.sliding_stride
    N = val_tokens.numel() - 1
    starts = list(range(0, N - T, stride))
    local_starts = starts[rank::ws]

    nll = torch.zeros((), dtype=torch.float64, device=device)
    tok = torch.zeros((), dtype=torch.float64, device=device)
    byt = torch.zeros((), dtype=torch.float64, device=device)
    model.eval()

    for s in local_starts:
        win = val_tokens[s:s+T+1].to(device, torch.int64)
        x, y = win[:-1].unsqueeze(0), win[1:].unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        # Only score the last `stride` tokens (novel tokens not seen in prior window)
        c0 = 0 if s == 0 else max(0, T - stride)
        lg = logits[:, c0:]; yt = y[:, c0:]; xt = x[:, c0:]
        nll += F.cross_entropy(
            lg.reshape(-1, args.vocab_size), yt.reshape(-1), reduction="sum").double()
        tok += yt.numel()
        tgt = yt.reshape(-1); prev = xt.reshape(-1)
        tb = bb[tgt].to(torch.int32) + (hl[tgt] & ~ib[prev]).to(torch.int32)
        byt += tb.double().sum()

    stats = torch.stack([nll, tok, byt])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    vl  = (stats[0] / stats[1]).item()
    bpb = (vl / math.log(2.0)) * (stats[1] / stats[2]).item()
    model.train()
    return vl, bpb


# ══════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════
def lr_scale(step: int, args: Hyperparameters) -> float:
    wu = args.warmup_steps
    if step < wu: return step / max(1, wu)
    wd_start = max(args.iterations - args.warmdown_iters, wu)
    if step < wd_start: return 1.0
    return max(0.0, (args.iterations - step) / max(1, args.warmdown_iters))


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def restore_fp32_controls(model: nn.Module):
    """Keep small/control parameters in fp32 for optimizer stability."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (param.ndim < 2 or _is_ctrl(name)) and param.dtype != torch.float32:
                param.data = param.data.float()


def train(args: Hyperparameters):
    # ── distributed setup ──────────────────────────────────────
    ddp = "RANK" in os.environ and dist.is_available()
    if ddp:
        dist.init_process_group("nccl"); rank = dist.get_rank(); ws = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        rank, ws = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    master = (rank == 0)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(f"logfile: {logfile}", flush=True)

    def log(msg):
        if not master: return
        print(msg, flush=True)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    # ── tokenizer + data ───────────────────────────────────────
    sp = spm.SentencePieceProcessor(); sp.Load(args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Tokenizer vocab {sp.vocab_size()} != VOCAB_SIZE {args.vocab_size}")
    bb, hl, ib = build_luts(sp, args.vocab_size, device)
    loader     = DDPLoader(args.train_files, rank, ws, device)
    val_tokens = load_val(args.val_files, args.train_seq_len).to(device)

    # ── model ──────────────────────────────────────────────────
    model = GPT(args).to(device).bfloat16()

    if device.type == "cuda":
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    model.set_attn_fp32()
    restore_fp32_controls(model)
    if ddp: model = DDP(model, device_ids=[device.index], broadcast_buffers=False)
    raw = model.module if ddp else model
    ema = EMA(raw, args.ema_decay)

    n_params = sum(p.numel() for p in raw.parameters())
    n_spine  = sum(p.numel() for n, p in raw.named_parameters() if "spines" in n)
    n_attn   = sum(p.numel() for n, p in raw.named_parameters()
                   if any(x in n for x in ["c_q", "c_k", "c_v", "proj"]))
    log(f"params: total={n_params:,}  spine={n_spine:,}  attn={n_attn:,}")
    log(f"layers={args.num_layers}  dim={args.model_dim}  "
        f"spines={args.num_spines}  spine_rank={args.spine_rank}")

    # ── optimizer split ────────────────────────────────────────
    # Muon: full-rank 2D attention weight matrices
    # AdamW: splanifold control matrices (low-rank, not suitable for Muon)
    # Adam: embeddings, scalars, control params
    attn_2d, spine_params, embed_params, scalar_params = [], [], [], []
    for name, p in raw.named_parameters():
        if any(x in name for x in ["c_q", "c_k", "c_v"]) and p.ndim == 2:
            attn_2d.append(p)
        elif any(k in name for k in [
            "spine_proj", "P0", "P1",
            "Pp0", "Pp1",
            "E0", "E1",
            "Ep0", "Ep1"
        ]):
            spine_params.append(p)
        elif "tok_emb" in name or "head_cor" in name:
            embed_params.append(p)
        else:
            scalar_params.append(p)
    print(f"spine params: {len(spine_params)}")
    opts = [
        Muon(attn_2d, lr=args.matrix_lr, momentum=args.muon_momentum,
             ns_steps=args.muon_ns_steps),
        torch.optim.AdamW(spine_params, lr=args.spine_lr,
                          betas=(args.beta1, args.beta2), eps=args.adam_eps,
                          weight_decay=args.weight_decay, fused=True),
        torch.optim.Adam(embed_params, lr=args.embed_lr,
                         betas=(args.beta1, args.beta2), eps=args.adam_eps,
                         fused=True),
        torch.optim.Adam(scalar_params, lr=args.scalar_lr,
                         betas=(args.beta1, args.beta2), eps=args.adam_eps,
                         fused=True),
    ]
    for opt in opts:
        for g in opt.param_groups: g["_base"] = g["lr"]

    accum = max(1, 8 // ws)
    ema_start_step = int(args.iterations * args.ema_start_frac)
    ema_on = False
    best_bpb = float("inf")
    log_acc  = 0.0
    t0 = time.perf_counter()

    log(f"training: {args.iterations} steps  accum={accum}  "
        f"ema_start={ema_start_step}  warmdown={args.warmdown_iters}")

    # ── training loop ──────────────────────────────────────────
    for step in range(args.iterations + 1):
        elapsed = time.perf_counter() - t0

        # Wallclock cap (0 = disabled for non-record track)
        if args.max_wallclock_s > 0 and elapsed >= args.max_wallclock_s:
            log(f"wallclock cap at step={step}"); break

        # EMA start
        if step == ema_start_step and not ema_on:
            ema.init(); ema_on = True
            log(f"EMA started (decay={args.ema_decay})")

        # Validation
        last_step = (step == args.iterations)
        if step % args.val_loss_every == 0 and step > 0 or last_step:
            bk = ema.apply() if ema_on else None

            if last_step:
                # only do expensive eval once
                _, bpb = eval_sliding(args, raw, val_tokens, bb, hl, ib, rank, ws, device)
                log(f"[FINAL] BPB={bpb:.4f}")
            else:
                val_loss = eval_fast(args, raw, val_tokens, device)
                log(f"[{step:6d}] val_loss={val_loss:.4f}")

            if bk: ema.restore(bk)

        # LR schedule
        s = lr_scale(step, args)
        frac = min(step / max(1, args.muon_mom_warmup_steps), 1.0)
        muon_mom = (1-frac)*args.muon_mom_warmup_start + frac*args.muon_momentum
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = g["_base"] * s
                if "momentum" in g: g["momentum"] = muon_mom

        # Forward / backward
        model.train(); sl = 0.0
        for _ in range(accum):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, accum)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(x, y) / accum
            loss.backward(); sl += loss.item()
        nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip_norm)
        for opt in opts: opt.step(); opt.zero_grad(set_to_none=True)
        restore_fp32_controls(raw)
        if ema_on: ema.update()
        log_acc += sl

        if step % args.train_log_every == 0 and master:
            log(f"[{step:6d}] loss={log_acc/max(1,args.train_log_every):.4f}  "
                f"lr={s*args.matrix_lr:.2e}  t={elapsed:.0f}s")
            log_acc = 0.0

    # ── final artifact ─────────────────────────────────────────
    if master:
        log(f"\nDone. Best BPB={best_bpb:.4f}")
        if ema_on: bk = ema.apply()
        # Final sliding eval
        _, sw = eval_sliding(args, raw, val_tokens, bb, hl, ib, 0, 1, device)
        log(f"Final sliding BPB (EMA): {sw:.4f}")
        # Build artifact
        mb = build_artifact(raw, f"artifact_{args.run_id}.ksplan")
        log(f"Artifact size: {mb:.2f} MB")
        if ema_on: ema.restore(bk)

    if ddp: dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train(Hyperparameters())


# ══════════════════════════════════════════════════════════════
# RUN COMMANDS
# ══════════════════════════════════════════════════════════════
#
# ── 1. SMOKE TEST (laptop CPU, SP1024) ────────────────────────
#
#   pip install sentencepiece lzma
#   python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
#
#   DATA_PATH=./data/datasets/fineweb10B_sp1024                 \
#   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model     \
#   VOCAB_SIZE=1024  ITERATIONS=60  TRAIN_BATCH_TOKENS=4096     \
#   TRAIN_SEQ_LEN=256  VAL_LOSS_EVERY=20  SLIDING_STRIDE=16     \
#   NUM_LAYERS=4  NUM_SPINES=4  SPINE_RANK=8                    \
#   python train_gpt_splanifold.py
#
#   Expected: loss trending down from ~6.9. Artifact built under 2 MB.
#
#
# ── 2. 1×H100 VALIDATION RUN ($7, ~2h) ───────────────────────
#
#   SEED=42  ITERATIONS=5000  TRAIN_BATCH_TOKENS=524288          \
#   NUM_LAYERS=16  NUM_SPINES=8  SPINE_RANK=16                  \
#   VAL_LOSS_EVERY=500  SLIDING_STRIDE=16                        \
#   GRAD_ACCUM_STEPS=1                                           \
#   python train_gpt_splanifold.py
#
#   Key checkpoints:
#   - Step 1000: BPB should be < 3.0 (splanifold is learning)
#   - Step 3000: BPB should be < 2.0 (converging)
#   - Step 5000: BPB should be < 1.5 (good trajectory for 50k)
#   If BPB > 3.0 at step 1000, the spine structure is too constrained.
#
#
# ── 3. SIZE VALIDATION (5 steps only) ────────────────────────
#
#   ITERATIONS=5  VAL_LOSS_EVERY=999999                          \
#   python train_gpt_splanifold.py
#   # Check: "artifact X.XX MB ✓" — must be under 16.0 MB
#
#
# ── 4. FULL NON-RECORD RUN (8×H100, ~2h, $55) ────────────────
#
#   SEED=42  ITERATIONS=50000  TRAIN_BATCH_TOKENS=524288         \
#   NUM_LAYERS=16  NUM_SPINES=8  SPINE_RANK=16                  \
#   SLIDING_STRIDE=16  EMA_DECAY=0.9965  EMA_START_FRAC=0.3     \
#   torchrun --standalone --nproc_per_node=8 train_gpt_splanifold.py
#
#
# ── 5. RANK SWEEP (ablation study for submission writeup) ─────
#
#   for rank in 4 8 16 32; do
#     SPINE_RANK=$rank ITERATIONS=5000 SEED=42                   \
#     python train_gpt_splanifold.py 2>&1 | grep BPB
#   done
#   # This produces the ablation table for your RESULTS.md
#
#
# ── 6. LAYER SWEEP (find optimal depth) ──────────────────────
#
#   for layers in 12 16 20; do
#     NUM_LAYERS=$layers ITERATIONS=5000 SEED=42                 \
#     python train_gpt_splanifold.py 2>&1 | grep -E "BPB|MB"
#   done
#   # Check both BPB and artifact size for each depth