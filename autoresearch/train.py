"""
Parameter Golf autoresearch training script.
This is the ONLY file the agent modifies.
Launch: ./run.sh
"""

from __future__ import annotations

import copy
import io
import math
import os
import random
import re
import time
import zlib
from pathlib import Path

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from prepare import (
    TIME_BUDGET, TRAIN_SEQ_LEN, VOCAB_SIZE, NUM_GPUS,
    DATA_PATH, TOKENIZER_PATH,
    load_validation_tokens, eval_val, build_sentencepiece_luts,
    DistributedTokenLoader, measure_model_size,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────

SEED = 1337
NUM_LAYERS = 11
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3
TIE_EMBEDDINGS = True
ROPE_BASE = 10000.0
ROPE_DIM = 16                    # number of head dims to apply rotary to (0=full, 16=partial like SOTA)
LOGIT_SOFTCAP = 8.0
QK_GAIN_INIT = 1.5

TRAIN_BATCH_TOKENS = 98_304
ITERATIONS = 20000
WARMDOWN_ITERS = 7000
WARMUP_STEPS = 20
VAL_LOSS_EVERY = 500
TRAIN_LOG_EVERY = 100

EMBED_LR = 0.3
HEAD_LR = 0.008
TIED_EMBED_LR = 0.03
TIED_EMBED_INIT_STD = 0.005
MATRIX_LR = 0.10
SCALAR_LR = 0.08
MUON_MOMENTUM = 0.90
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.85
MUON_MOMENTUM_WARMUP_STEPS = 250
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
GRAD_CLIP_NORM = 0.0

# ── SOTA Technique Flags ─────────────────────────────────────────────────────
ACTIVATION = "silu"           # "silu" | "relu_sq" | "leaky_relu_sq"
SMEAR_GATE_ENABLED = True     # blend each token with previous token's embedding
ORTHOGONAL_INIT = True        # orthogonal init for weight matrices (SOTA uses this)
SWA_ENABLED = False           # SWA consistently degrades by 0.008 with warmdown schedule
SWA_START_FRAC = 0.10
SWA_EVERY = 30
INT6_QUANT = True             # use int6 for MLP+attention (vs int8 for all)
QUANT_BITS = 6                # quantization bits for MLP+attention (5 or 6)
BIGRAM_HASH_ENABLED = False   # BigramHash tested: +0.006 worse on MLP3x, even with 2048 vocab
BIGRAM_VOCAB_SIZE = 2048
BIGRAM_DIM = 128

# ── Overparameterized Training ────────────────────────────────────────────────
# The 16MB limit is on the FINAL ARTIFACT, not the model during training.
# Train a larger model, then compress/prune/distill to fit in 16MB.
LORA_ENABLED = False          # LoRA adapters during training (merged at end, zero artifact cost)
LORA_RANK = 64                # LoRA rank (higher = more capacity during training)
LORA_ALPHA = 1.0              # LoRA scaling factor (alpha / rank)
LORA_TARGETS = "all"          # "all" | "attn" | "mlp" — which layers get LoRA
SVD_COMPRESS = False           # Post-training SVD compression of weight matrices
SVD_RANK_FRAC = 0.5           # Keep this fraction of singular values (0.5 = 50% compression)
PRUNING_ENABLED = False        # Structured pruning after training
PRUNING_RATIO = 0.3           # Remove this fraction of neurons (by magnitude)
PRUNING_FINETUNE_FRAC = 0.15  # Fraction of total time budget for post-pruning finetuning
TRAIN_DIM_MULT = 1.0          # Train at wider dim (e.g. 1.5x), compress after. 1.0 = no overparameterization

# ── Advanced Compression Methods ────────────────────────────────────────────
# The 16MB limit is on the FINAL ARTIFACT. These methods offer better
# compression than uniform int6/int8 by exploiting structure in weight matrices.
COMPRESS_METHOD = "nonuniform" # "int6" | "factored_int4" | "nonuniform" | "fft_int4"
COMPRESS_ALGO = "zstd"        # "zlib" | "zstd" — zstd-22 saves ~8% over zlib-9
# Factored int4: W ≈ B @ A via SVD, B and A stored at int4. Product ≈ int8.
FACTORED_RANK = 384           # rank of factorization (lower = more compression, less quality)
# Non-uniform quantization: important layers get more bits, unimportant fewer
NONUNIFORM_IMPORTANT_BITS = 6
NONUNIFORM_UNIMPORTANT_BITS = 5
NONUNIFORM_IMPORTANT_FRAC = 0.0  # all layers at int5 (11L MLP3x needs aggressive compression)
# Hadamard rotation before quantization: makes weight distributions uniform, better compression
HADAMARD_ROTATE = False
# FFT int4: FFT per row → quantize all coefficients at int4 → iFFT reconstructs ≈ int8
FFT_KEEP_FRAC = 1.0           # 1.0 = keep all freqs, <1.0 = sparse (drop weak frequencies)

# ── Embedding Geometry Flags ──────────────────────────────────────────────────
EMBED_MODE = "baseline"       # "baseline" | "sphere" | "rff" | "product" | "wavelet_init"
WAVELET_LEVELS = 4            # for wavelet_init mode: keep first 2^N Haar components
RFF_GAMMA = 0.1               # for rff mode: frequency scaling
CLUSTER_INIT = False          # use GPT-2 32-cluster-aware initialization

# Optional: path to pretrained embedding init [vocab_size, model_dim]
INIT_EMBEDDINGS_PATH = os.environ.get("INIT_EMBEDDINGS", "/media/Datacenter_storage/winston_001a/openai/parameter-golf/gpt2_pca_embeddings_512.pt")

# Geometry distillation: similarity-preserving loss from GPT-2 embedding structure
GEO_DISTILL_ENABLED = False
GEO_DISTILL_LAMBDA = 0.01    # weight for geometry loss (tune: 0.01 - 0.1)
GEO_DISTILL_KNN_ONLY = False  # only match 10-nearest-neighbor similarities (cheaper, more local)
GEO_DISTILL_WARMUP_FRAC = 1.0 # only apply geo loss during first X fraction of training (1.0 = always)
GEO_DISTILL_PATH = "/media/Datacenter_storage/winston_001a/openai/parameter-golf/gpt2_embedding_geometry.pt"

# Teacher adapter (legacy, mostly disproven — kept for reference)
TEACHER_ADAPTER_ENABLED = False
TEACHER_ADAPTER_DIM = 64
TEACHER_EMB_PATH = "/media/Datacenter_storage/winston_001a/openai/parameter-golf/gpt2_pca_embeddings_512.pt"
FREEZE_ADAPTER_PATH = os.environ.get("FREEZE_ADAPTER_PATH", "")
SAVE_ADAPTER_PATH = os.environ.get("SAVE_ADAPTER_PATH", "")

# ── Muon Optimizer ───────────────────────────────────────────────────────────

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            wd = group.get("weight_decay", 0.0)
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ── Model ────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim, rope_dim=0, base=10000.0):
        super().__init__()
        # rope_dim=0 means full rotary; >0 means only first rope_dim dims get rotary
        self.rope_dim = rope_dim if rope_dim > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dim=0):
    if rope_dim > 0 and rope_dim < x.size(-1):
        # Partial RoPE: only rotate first rope_dim dims
        x_rot = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        half = rope_dim // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dim = ROPE_DIM
        self.rotary = Rotary(self.head_dim, rope_dim=ROPE_DIM, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, rope_dim=self.rope_dim)
        k = apply_rotary_emb(k, cos, sin, rope_dim=self.rope_dim)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        if ACTIVATION == "relu_sq":
            return self.proj(torch.relu(self.fc(x)).square())
        if ACTIVATION == "leaky_relu_sq":
            return self.proj(F.leaky_relu(self.fc(x), 0.5).square())
        return self.proj(F.silu(self.fc(x)))


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table."""
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


# ── Geometric Embedding Modes ────────────────────────────────────────────────

class SphereEmbedding(nn.Module):
    """Embed on sphere: L2 normalize + learnable radius."""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.radius = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, ids):
        x = self.tok_emb(ids)
        x = F.normalize(x.float(), dim=-1)
        return x * self.radius.to(dtype=x.dtype)


class ProductEmbedding(nn.Module):
    """Half spherical (L2 norm), half hyperbolic (Poincaré ball)."""
    def __init__(self, vocab_size, dim, curvature=1.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.split = dim // 2
        self.c = curvature
        self.radius = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def _expmap0(self, v):
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.clamp(v.norm(dim=-1, keepdim=True), min=1e-6)
        return torch.tanh(sqrt_c * v_norm / 2) * v / (sqrt_c * v_norm)

    def forward(self, ids):
        x = self.tok_emb(ids).float()
        x_s = F.normalize(x[..., :self.split], dim=-1) * self.radius.to(dtype=x.dtype)
        x_h = self._expmap0(x[..., self.split:])
        x = torch.cat([x_s, x_h], dim=-1)
        return F.rms_norm(x, (x.size(-1),))


class RFFEmbedding(nn.Module):
    """Random Fourier Features — frozen random cos projection, zero storage cost."""
    def __init__(self, vocab_size, dim, gamma=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        gen = torch.Generator()
        gen.manual_seed(42)
        self.register_buffer('W', torch.randn(dim, dim, generator=gen) * gamma)
        self.register_buffer('b', torch.rand(dim, generator=gen) * 2 * math.pi)
        self.mix = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, ids):
        x = self.tok_emb(ids).float()
        rff = math.sqrt(2.0 / x.size(-1)) * torch.cos(x @ self.W + self.b)
        alpha = torch.sigmoid(self.mix)
        x = (1 - alpha) * x + alpha * rff
        return F.rms_norm(x, (x.size(-1),))


def build_haar_matrix(n):
    """Build n×n Haar wavelet matrix (orthogonal, deterministic, zero storage)."""
    H = torch.zeros(n, n)
    H[0] = 1.0 / math.sqrt(n)
    idx = 1
    for level in range(int(math.log2(n))):
        block_size = n >> (level + 1)
        scale = 1.0 / math.sqrt(block_size * 2)
        for k in range(1 << level):
            start = k * block_size * 2
            H[idx, start:start + block_size] = scale
            H[idx, start + block_size:start + 2 * block_size] = -scale
            idx += 1
    return H


def wavelet_init_embeddings(pca_emb, n_levels, cluster_labels=None):
    """Decompose PCA embeddings with Haar wavelets, keep coarse components.

    If cluster_labels is provided, reorder tokens by cluster before transform
    so Haar basis aligns with natural cluster boundaries (literature: GPT-2
    tokens cluster by orthographic features like leading space, and Haar
    wavelets capture more structure when similar items are adjacent).
    """
    n = pca_emb.shape[0]  # 1024
    if cluster_labels is not None:
        order = torch.argsort(cluster_labels)
        pca_emb = pca_emb[order]
    H = build_haar_matrix(n)  # [1024, 1024]
    C = H @ pca_emb  # wavelet coefficients [1024, 512]
    keep = min(2 ** n_levels, n)
    C[keep:] = 0
    result = H.T @ C  # reconstruct from coarse components
    if cluster_labels is not None:
        inv_order = torch.argsort(order)
        result = result[inv_order]
    return result


# ── Int6 mixed quantization ──────────────────────────────────────────────────

FP16_KEEP_NAME_PATTERNS = ("tok_emb",)


def _compress_bytes(data: bytes) -> bytes:
    """Compress bytes using the selected algorithm."""
    if COMPRESS_ALGO == "zstd" and HAS_ZSTD:
        cctx = zstandard.ZstdCompressor(level=22)
        return cctx.compress(data)
    return zlib.compress(data, level=9)

GPTQ_LITE = False             # search over clip percentiles — saves 0.0006 BPB but adds 1.4MB (over limit)

def quantize_intN_per_row(t, bits=6):
    """Quantize tensor to intN range [-(2^(bits-1)), 2^(bits-1)-1] per row.
    With GPTQ_LITE: search clip percentiles to minimize MSE per row."""
    max_val = (1 << (bits - 1)) - 1  # 31 for 6-bit, 15 for 5-bit
    min_val = -(1 << (bits - 1))     # -32 for 6-bit, -16 for 5-bit
    t32 = t.float()
    if t32.ndim == 2:
        if GPTQ_LITE:
            # Search over clip percentiles to find min-MSE per row
            row_max = t32.abs().amax(dim=1)  # [rows]
            clip_fracs = torch.tensor([0.90, 0.95, 0.98, 1.0], device=t32.device)
            best_q = None
            best_scale = None
            best_mse = torch.full((t32.shape[0],), float('inf'), device=t32.device)
            for frac in clip_fracs:
                clip_max = row_max * frac
                sc = (clip_max / max_val).clamp_min(1e-12).to(torch.float16)
                sc = sc.clamp_min(torch.finfo(torch.float16).tiny)
                q_cand = torch.clamp(torch.round(t32 / sc.float()[:, None]), min_val, max_val).to(torch.int8)
                recon = q_cand.float() * sc.float()[:, None]
                mse = (t32 - recon).square().mean(dim=1)
                improved = mse < best_mse
                if best_q is None:
                    best_q = q_cand
                    best_scale = sc
                    best_mse = mse
                else:
                    best_q[improved] = q_cand[improved]
                    best_scale[improved] = sc[improved]
                    best_mse[improved] = mse[improved]
            return best_q, best_scale
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / max_val).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), min_val, max_val).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / max_val, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), min_val, max_val).to(torch.int8)
    return q, scale


def _classify_param(name):
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_float_tensor_int8(t):
    """Standard int8 per-row quantization."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), 0.9999984, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def mixed_quantize_int6(state_dict, int6_cats={"mlp", "attn"}):
    """Int6 for MLP+attention, int8 for other large tensors, FP16 for embeddings."""
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pattern in name for pattern in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(dtype=torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_intN_per_row(t, bits=QUANT_BITS)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{QUANT_BITS}"}
        else:
            q, s = quantize_float_tensor_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result, meta, template_sd):
    """Reverse mixed int6 quantization for roundtrip validation."""
    out = {}
    for name, orig in template_sd.items():
        info = meta[name]
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


def measure_model_size_int6(state_dict):
    """Estimate int6+zlib compressed model size in bytes."""
    result, meta = mixed_quantize_int6(state_dict)
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    compressed = _compress_bytes(buf.getvalue())
    return len(compressed)


# ── Advanced Compression Functions ──────────────────────────────────────────

def _layer_importance(state_dict):
    """Score each transformer block by total weight Frobenius norm."""
    scores = {}
    for name, t in state_dict.items():
        if not t.is_floating_point() or t.ndim != 2:
            continue
        m = re.search(r'blocks\.(\d+)', name)
        if m:
            idx = int(m.group(1))
            scores[idx] = scores.get(idx, 0.0) + t.float().norm().item()
    return scores


def factored_int4_compress(state_dict, rank=128):
    """W ≈ B @ A via SVD. B[out,r] and A[r,in] quantized to int4. Product ≈ int8.
    This halves storage for large matrices while maintaining ~int8 effective precision."""
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pat in name for pat in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough_fp16"
            continue
        cat = _classify_param(name)
        if t.ndim == 2 and cat in ("mlp", "attn") and min(t.shape) > rank:
            t32 = t.float()
            U, S, Vt = torch.linalg.svd(t32, full_matrices=False)
            r = min(rank, min(t32.shape))
            sqrtS = S[:r].sqrt()
            B = U[:, :r] * sqrtS[None, :]   # [out, r]
            A = Vt[:r, :] * sqrtS[:, None]   # [r, in]
            qB, sB = quantize_intN_per_row(B, bits=4)
            qA, sA = quantize_intN_per_row(A, bits=4)
            result[name + ".B"] = qB
            result[name + ".A"] = qA
            result[name + ".sB"] = sB
            result[name + ".sA"] = sA
            meta[name] = {"type": "factored_int4", "rank": r}
        else:
            q, s = quantize_float_tensor_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def _hadamard(n: int):
    """Build normalized Hadamard matrix of size n (must be power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be power of 2"
    H = torch.ones(1, 1)
    while H.size(0) < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0) / math.sqrt(2)
    return H

def _block_hadamard_rotate(W):
    """Block-diagonal Hadamard rotation along column dimension. Self-inverse."""
    out_dim, in_dim = W.shape
    block = in_dim & (-in_dim)  # largest power-of-2 divisor
    H = _hadamard(block).to(W.dtype).to(W.device)
    if block == in_dim:
        return W @ H
    num_blocks = in_dim // block
    W_blocks = W.view(out_dim, num_blocks, block)
    W_rot = torch.einsum("onb,bk->onk", W_blocks, H)
    return W_rot.reshape(out_dim, in_dim)


def nonuniform_compress(state_dict, important_bits=8, unimportant_bits=4, important_frac=0.4):
    """Allocate more bits to important layers (by weight norm), fewer to unimportant ones."""
    scores = _layer_importance(state_dict)
    important_layers = set()
    if scores and important_frac > 0:
        sorted_scores = sorted(scores.values(), reverse=True)
        n_important = max(1, int(len(scores) * important_frac))
        threshold = sorted_scores[min(n_important - 1, len(sorted_scores) - 1)]
        important_layers = {idx for idx, s in scores.items() if s >= threshold}
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pat in name for pat in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough_fp16"
            continue
        m = re.search(r'blocks\.(\d+)', name)
        layer_idx = int(m.group(1)) if m else -1
        bits = important_bits if layer_idx in important_layers else unimportant_bits
        cat = _classify_param(name)
        if cat in ("mlp", "attn") and t.ndim >= 1:
            t_q = t
            hadamard = HADAMARD_ROTATE and t.ndim == 2
            if hadamard:
                t_q = _block_hadamard_rotate(t.float())
            q, s = quantize_intN_per_row(t_q, bits=bits)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{bits}", "hadamard": hadamard}
        else:
            q, s = quantize_float_tensor_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def fft_int4_compress(state_dict, keep_frac=1.0):
    """FFT per row → quantize real/imag at int4 → iFFT reconstructs ≈ int8.
    Frequency-domain quantization errors distribute evenly in spatial domain."""
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        if any(pat in name for pat in FP16_KEEP_NAME_PATTERNS):
            result[name] = t.to(torch.float16)
            meta[name] = "passthrough_fp16"
            continue
        cat = _classify_param(name)
        if t.ndim == 2 and cat in ("mlp", "attn"):
            t32 = t.float()
            F_c = torch.fft.rfft(t32, dim=1)  # [out, in//2+1] complex
            n_freq = F_c.shape[1]
            if keep_frac < 1.0:
                n_keep = max(1, int(n_freq * keep_frac))
                mags = F_c.abs()
                _, topk_idx = mags.topk(n_keep, dim=1)
                mask = torch.zeros_like(F_c, dtype=torch.bool)
                mask.scatter_(1, topk_idx, True)
                F_c = F_c * mask
            qr, sr = quantize_intN_per_row(F_c.real, bits=4)
            qi, si = quantize_intN_per_row(F_c.imag, bits=4)
            result[name + ".qr"] = qr
            result[name + ".qi"] = qi
            result[name + ".sr"] = sr
            result[name + ".si"] = si
            result[name + ".shape"] = torch.tensor(list(t32.shape), dtype=torch.int32)
            meta[name] = {"type": "fft_int4", "n_freq": n_freq}
        else:
            q, s = quantize_float_tensor_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta


def measure_model_size_advanced(state_dict):
    """Measure compressed model size using the selected COMPRESS_METHOD."""
    if COMPRESS_METHOD == "factored_int4":
        result, meta = factored_int4_compress(state_dict, rank=FACTORED_RANK)
    elif COMPRESS_METHOD == "nonuniform":
        result, meta = nonuniform_compress(
            state_dict, NONUNIFORM_IMPORTANT_BITS,
            NONUNIFORM_UNIMPORTANT_BITS, NONUNIFORM_IMPORTANT_FRAC)
    elif COMPRESS_METHOD == "fft_int4":
        result, meta = fft_int4_compress(state_dict, keep_frac=FFT_KEEP_FRAC)
    else:
        # Default: int6 or int8
        if INT6_QUANT:
            return measure_model_size_int6(state_dict)
        return measure_model_size(state_dict)
    buf = io.BytesIO()
    torch.save({"w": result, "m": meta}, buf)
    compressed = _compress_bytes(buf.getvalue())
    return len(compressed)


# ── LoRA: Low-Rank Adaptation ─────────────────────────────────────────────────
# Adds trainable low-rank matrices A,B to linear layers during training.
# At end: merge W += (alpha/rank)*B@A into base weights, discard A,B. Zero artifact cost.

class LoRALinear(nn.Module):
    """Wraps a CastedLinear with LoRA adapters. Merged at end for zero overhead."""
    def __init__(self, base_linear, rank, alpha=1.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = alpha / rank
        out_features = base_linear.weight.shape[0]
        in_features = base_linear.weight.shape[1]
        # A: [rank, in_features], B: [out_features, rank]
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = self.base(x)
        # LoRA path: x @ A^T @ B^T * scale
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    def merge_and_return_base(self):
        """Merge LoRA weights into base and return the base linear layer."""
        with torch.no_grad():
            self.base.weight.add_((self.scale * self.lora_B @ self.lora_A).to(self.base.weight.dtype))
        return self.base


def apply_lora(model, rank, alpha, targets="all"):
    """Wrap linear layers in LoRA adapters. Returns list of LoRA params for optimizer."""
    lora_params = []
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, CastedLinear):
            continue
        if getattr(module, '_zero_init', False):
            continue  # skip zero-init projection layers
        if targets == "attn" and ".mlp." in name:
            continue
        if targets == "mlp" and ".attn." in name:
            continue
        # Don't LoRA the lm_head or embedding-related layers
        if "lm_head" in name or "tok_emb" in name or "bigram" in name:
            continue
        lora = LoRALinear(module, rank, alpha)
        lora_params.extend([lora.lora_A, lora.lora_B])
        replacements.append((name, lora))
    # Apply replacements
    for name, lora in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lora)
    return lora_params


def merge_lora(model):
    """Merge all LoRA adapters back into base weights. Call before saving."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            replacements.append((name, module.merge_and_return_base()))
    for name, base in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], base)


# ── Post-training SVD Compression ─────────────────────────────────────────────

def svd_compress_model(model, rank_frac=0.5):
    """Replace large weight matrices W with low-rank approximation U@V.
    This reduces model size at the cost of some accuracy."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, (CastedLinear, nn.Linear)):
                continue
            W = module.weight.data.float()
            if W.ndim != 2 or min(W.shape) < 64:
                continue
            # Skip embeddings and small layers
            if "tok_emb" in name or "lm_head" in name:
                continue
            k = max(1, int(min(W.shape) * rank_frac))
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            # Reconstruct with top-k singular values
            W_approx = (U[:, :k] * S[:k].unsqueeze(0)) @ Vt[:k, :]
            module.weight.data.copy_(W_approx.to(module.weight.dtype))


# ── Structured Pruning ────────────────────────────────────────────────────────

def prune_model_structured(model, ratio=0.3):
    """Remove lowest-magnitude neurons from MLP layers.
    Zeros out entire rows/columns to enable structured sparsity."""
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, MLP):
                continue
            # Prune by MLP hidden neuron magnitude
            fc_weight = module.fc.weight.data.float()  # [hidden, dim]
            neuron_norms = fc_weight.norm(dim=1)  # [hidden]
            n_prune = int(neuron_norms.shape[0] * ratio)
            if n_prune == 0:
                continue
            _, indices = neuron_norms.topk(n_prune, largest=False)
            # Zero out pruned neurons in both fc and proj
            module.fc.weight.data[indices] = 0
            module.proj.weight.data[:, indices] = 0


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class TeacherAdapter(nn.Module):
    """Frozen teacher embeddings + trainable bottleneck adapter.
    Provides persistent GPT-2 knowledge injection at every forward pass."""
    def __init__(self, vocab_size, teacher_dim, model_dim, bottleneck_dim, teacher_emb_path):
        super().__init__()
        # Frozen teacher embedding table (loaded from GPT-2 PCA)
        self.teacher_emb = nn.Embedding(vocab_size, teacher_dim)
        teacher_weights = torch.load(teacher_emb_path, map_location="cpu", weights_only=True)
        self.teacher_emb.weight.data.copy_(teacher_weights)
        self.teacher_emb.weight.requires_grad = False  # frozen
        # Trainable bottleneck: teacher_dim → bottleneck → model_dim
        self.down = CastedLinear(teacher_dim, bottleneck_dim, bias=False)
        self.up = CastedLinear(bottleneck_dim, model_dim, bias=False)
        self.up._zero_init = True  # start as identity residual (no effect initially)
        # Learnable gate to control adapter influence
        self.gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, token_ids):
        # token_ids: [B, T]
        t = self.teacher_emb(token_ids)  # [B, T, teacher_dim] — frozen
        h = F.silu(self.down(t))  # [B, T, bottleneck] — nonlinear adapter
        out = self.up(h)  # [B, T, model_dim]
        return torch.sigmoid(self.gate).to(dtype=out.dtype) * out


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tie_embeddings = TIE_EMBEDDINGS
        # Embedding mode selection
        if EMBED_MODE == "sphere":
            self._emb_module = SphereEmbedding(VOCAB_SIZE, MODEL_DIM)
            self.tok_emb = self._emb_module.tok_emb
        elif EMBED_MODE == "product":
            self._emb_module = ProductEmbedding(VOCAB_SIZE, MODEL_DIM)
            self.tok_emb = self._emb_module.tok_emb
        elif EMBED_MODE == "rff":
            self._emb_module = RFFEmbedding(VOCAB_SIZE, MODEL_DIM, gamma=RFF_GAMMA)
            self.tok_emb = self._emb_module.tok_emb
        else:
            self._emb_module = None
            self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        # BigramHash embedding
        self.bigram = BigramHashEmbedding(BIGRAM_VOCAB_SIZE, BIGRAM_DIM, MODEL_DIM) if BIGRAM_HASH_ENABLED else None
        # SmearGate
        self.smear = SmearGate(MODEL_DIM) if SMEAR_GATE_ENABLED else None
        # Teacher adapter (legacy)
        self.teacher_adapter = None
        if TEACHER_ADAPTER_ENABLED and os.path.exists(TEACHER_EMB_PATH):
            self.teacher_adapter = TeacherAdapter(
                VOCAB_SIZE, MODEL_DIM, MODEL_DIM, TEACHER_ADAPTER_DIM, TEACHER_EMB_PATH
            )
        self.num_encoder_layers = NUM_LAYERS // 2
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, MODEL_DIM, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT)
            for _ in range(NUM_LAYERS)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if TIE_EMBEDDINGS else CastedLinear(MODEL_DIM, VOCAB_SIZE, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=TIED_EMBED_INIT_STD)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif ORTHOGONAL_INIT and module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward(self, input_ids, target_ids):
        if self._emb_module is not None:
            x = self._emb_module(input_ids)
        else:
            x = self.tok_emb(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.teacher_adapter is not None:
            x = x + self.teacher_adapter(input_ids)
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ── Training ─────────────────────────────────────────────────────────────────

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


def main():
    global zeropower_via_newtonschulz5

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 1
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg):
        if master_process:
            print(msg)

    # Seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Tokenizer + validation
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    val_tokens = load_validation_tokens(TRAIN_SEQ_LEN)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE, device)

    # Model
    base_model = GPT().to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Optional pretrained embedding init (with wavelet/cluster modes)
    if INIT_EMBEDDINGS_PATH and os.path.exists(INIT_EMBEDDINGS_PATH):
        init_emb = torch.load(INIT_EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
        assert init_emb.shape == base_model.tok_emb.weight.shape, \
            f"Shape mismatch: {init_emb.shape} vs {base_model.tok_emb.weight.shape}"
        if EMBED_MODE == "wavelet_init":
            # Load cluster labels for cluster-ordered wavelet (better Haar alignment)
            wv_labels = None
            if os.path.exists(GEO_DISTILL_PATH):
                geo_wv = torch.load(GEO_DISTILL_PATH, map_location="cpu", weights_only=True)
                if "cluster_labels_32" in geo_wv:
                    wv_labels = geo_wv["cluster_labels_32"].long()
                    log0(f"Wavelet init: using cluster-ordered Haar (32 clusters)")
            init_emb = wavelet_init_embeddings(init_emb.float(), WAVELET_LEVELS, cluster_labels=wv_labels)
            log0(f"Wavelet init: {WAVELET_LEVELS} levels, keeping {min(2**WAVELET_LEVELS, init_emb.shape[0])} coarse components")
        if CLUSTER_INIT and os.path.exists(GEO_DISTILL_PATH):
            geo_data = torch.load(GEO_DISTILL_PATH, map_location="cpu", weights_only=True)
            labels = geo_data["cluster_labels_32"].long()
            centroids = geo_data["cluster_centroids_32"].float()
            # Init as centroid + small noise
            init_emb = centroids[labels] + 0.01 * torch.randn_like(init_emb)
            log0(f"Cluster init: 32 clusters, centroid-based initialization")
        with torch.no_grad():
            base_model.tok_emb.weight.copy_(init_emb.to(dtype=base_model.tok_emb.weight.dtype))
        log0(f"Loaded pretrained embeddings from {INIT_EMBEDDINGS_PATH}")

    # Phase 2: Load pre-trained adapter and freeze it (simulates submission)
    if FREEZE_ADAPTER_PATH and base_model.teacher_adapter is not None:
        adapter_state = torch.load(FREEZE_ADAPTER_PATH, map_location="cpu", weights_only=True)
        base_model.teacher_adapter.load_state_dict(adapter_state, strict=False)
        # Freeze ALL adapter params (both teacher_emb which is already frozen, and down/up/gate)
        for p in base_model.teacher_adapter.parameters():
            p.requires_grad = False
        log0(f"Phase 2: Loaded and froze pre-trained adapter from {FREEZE_ADAPTER_PATH}")
        log0(f"  Adapter gate value: {base_model.teacher_adapter.gate.item():.4f} "
             f"(sigmoid={torch.sigmoid(base_model.teacher_adapter.gate).item():.4f})")

    # Load teacher geometry for distillation
    teacher_sim = None
    teacher_knn_indices = None
    if GEO_DISTILL_ENABLED and os.path.exists(GEO_DISTILL_PATH):
        geo_data = torch.load(GEO_DISTILL_PATH, map_location=device, weights_only=True)
        teacher_sim = geo_data["sim_matrix"].to(device=device, dtype=torch.float32)  # [1024, 1024]
        if GEO_DISTILL_KNN_ONLY and "knn_indices_10" in geo_data:
            teacher_knn_indices = geo_data["knn_indices_10"].to(device=device, dtype=torch.long)
            log0(f"Loaded KNN indices for local geometry distillation")
        log0(f"Loaded teacher geometry from {GEO_DISTILL_PATH}, sim_matrix shape: {teacher_sim.shape}")
    elif GEO_DISTILL_ENABLED:
        log0(f"WARNING: GEO_DISTILL_ENABLED but {GEO_DISTILL_PATH} not found, skipping")

    # Apply LoRA adapters (adds trainable low-rank matrices, merged before saving)
    lora_params = []
    if LORA_ENABLED:
        lora_params = apply_lora(base_model, LORA_RANK, LORA_ALPHA, LORA_TARGETS)
        n_lora = sum(p.numel() for p in lora_params)
        log0(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}, "
             f"extra_params={n_lora} ({n_lora/1e6:.1f}M) — will be merged before saving")

    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model
    n_params = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log0(f"model_params: {n_params} ({n_params/1e6:.1f}M), trainable: {n_trainable} ({n_trainable/1e6:.1f}M)")

    # Optimizers
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # SmearGate: gate is a scalar param
    if base_model.smear is not None:
        scalar_params.append(base_model.smear.gate)
    # Geometry embedding extra params (radius, mix, etc.)
    if hasattr(base_model, '_emb_module') and base_model._emb_module is not None:
        for name, p in base_model._emb_module.named_parameters():
            if p is not base_model.tok_emb.weight and p.requires_grad:
                scalar_params.append(p)

    token_lr = TIED_EMBED_LR if TIE_EMBEDDINGS else EMBED_LR
    tok_params_list = [base_model.tok_emb.weight]
    optimizer_tok = torch.optim.Adam(
        [{"params": tok_params_list, "lr": token_lr, "base_lr": token_lr}],
        betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=MATRIX_LR, momentum=MUON_MOMENTUM, backend_steps=MUON_BACKEND_STEPS, weight_decay=0.04)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = MATRIX_LR
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": SCALAR_LR, "base_lr": SCALAR_LR}],
        betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    # LoRA optimizer (Adam on low-rank adapter matrices)
    if lora_params:
        optimizer_lora = torch.optim.Adam(
            [{"params": lora_params, "lr": MATRIX_LR, "base_lr": MATRIX_LR}],
            betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True,
        )
        optimizers.append(optimizer_lora)
    # BigramHash optimizer (matrix params for embed+proj, scalar for scale)
    if base_model.bigram is not None:
        bigram = base_model.bigram
        bigram_groups = []
        bigram_matrix = [p for p in [bigram.embed.weight] if p.requires_grad]
        if bigram.proj is not None:
            bigram_matrix.append(bigram.proj.weight)
        if bigram_matrix:
            bigram_groups.append({"params": bigram_matrix, "lr": EMBED_LR, "base_lr": EMBED_LR})
        bigram_groups.append({"params": [bigram.scale], "lr": SCALAR_LR, "base_lr": SCALAR_LR})
        optimizer_bigram = torch.optim.Adam(bigram_groups, betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True)
        optimizers.append(optimizer_bigram)
    # Teacher adapter optimizer (only if adapter exists and has trainable params)
    if base_model.teacher_adapter is not None:
        adapter = base_model.teacher_adapter
        adapter_trainable = [p for p in [adapter.down.weight, adapter.up.weight, adapter.gate] if p.requires_grad]
        if adapter_trainable:
            # Phase 1: adapter is trainable
            adapter_matrix_params = [p for p in [adapter.down.weight, adapter.up.weight] if p.requires_grad]
            adapter_scalar_params = [p for p in [adapter.gate] if p.requires_grad]
            opt_groups = []
            if adapter_matrix_params:
                opt_groups.append({"params": adapter_matrix_params, "lr": EMBED_LR, "base_lr": EMBED_LR})
            if adapter_scalar_params:
                opt_groups.append({"params": adapter_scalar_params, "lr": SCALAR_LR, "base_lr": SCALAR_LR})
            if opt_groups:
                optimizer_adapter = torch.optim.Adam(opt_groups, betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True)
                optimizers.append(optimizer_adapter)
            n_adapter_params = sum(p.numel() for p in adapter_trainable)
            log0(f"teacher_adapter: bottleneck={TEACHER_ADAPTER_DIM}, trainable_params={n_adapter_params}, "
                 f"frozen_emb_params={adapter.teacher_emb.weight.numel()}")
        else:
            # Phase 2: adapter is fully frozen
            log0(f"teacher_adapter: FROZEN (Phase 2 submission mode), "
                 f"total_params={sum(p.numel() for p in adapter.parameters())}")
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": HEAD_LR, "base_lr": HEAD_LR}],
            betas=(BETA1, BETA2), eps=ADAM_EPS, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    train_loader = DistributedTokenLoader(rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * TIME_BUDGET

    def lr_mul(step, elapsed_ms):
        if WARMDOWN_ITERS <= 0:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = WARMDOWN_ITERS * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup
    if WARMUP_STEPS > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(WARMUP_STEPS):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(TRAIN_BATCH_TOKENS, TRAIN_SEQ_LEN, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(rank, world_size, device)
        log0(f"warmup: {WARMUP_STEPS} steps done")

    # SWA state
    swa_state = None
    swa_count = 0

    # Training loop
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == ITERATIONS or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (VAL_LOSS_EVERY > 0 and step % VAL_LOSS_EVERY == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{ITERATIONS} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < ITERATIONS:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{ITERATIONS}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(TRAIN_BATCH_TOKENS, TRAIN_SEQ_LEN, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            # Geometry distillation loss (with KNN-only and warmup fraction options)
            if teacher_sim is not None and GEO_DISTILL_LAMBDA > 0:
                # Check warmup fraction: only apply during first X% of training
                geo_active = True
                if GEO_DISTILL_WARMUP_FRAC < 1.0:
                    elapsed_ms_now = training_time_ms + 1000.0 * (time.perf_counter() - t0)
                    geo_active = elapsed_ms_now < max_wallclock_ms * GEO_DISTILL_WARMUP_FRAC
                if geo_active:
                    emb_w = base_model.tok_emb.weight.float()  # [V, D]
                    emb_norm = F.normalize(emb_w, dim=1)
                    student_sim = emb_norm @ emb_norm.T  # [V, V]
                    if GEO_DISTILL_KNN_ONLY and teacher_knn_indices is not None:
                        # Only match 10-nearest-neighbor similarities
                        knn_mask = torch.zeros_like(teacher_sim, dtype=torch.bool)
                        rows = torch.arange(teacher_sim.size(0), device=device).unsqueeze(1).expand_as(teacher_knn_indices)
                        knn_mask[rows, teacher_knn_indices] = True
                        geo_loss = F.mse_loss(student_sim[knn_mask], teacher_sim[knn_mask])
                    else:
                        geo_loss = F.mse_loss(student_sim, teacher_sim)
                    loss = loss + GEO_DISTILL_LAMBDA * geo_loss
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0) if MUON_MOMENTUM_WARMUP_STEPS > 0 else 1.0
        muon_momentum = (1 - frac) * MUON_MOMENTUM_WARMUP_START + frac * MUON_MOMENTUM
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), GRAD_CLIP_NORM)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # SWA: collect checkpoints during warmdown
        if SWA_ENABLED and scale < SWA_START_FRAC and step % SWA_EVERY == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        if TRAIN_LOG_EVERY > 0 and (step <= 10 or step % TRAIN_LOG_EVERY == 0):
            log0(f"step:{step}/{ITERATIONS} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = approx_training_time_ms >= max_wallclock_ms
        if distributed:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Apply SWA if collected
    if SWA_ENABLED and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        current_state = base_model.state_dict()
        avg_state = {
            name: (tensor / swa_count).to(dtype=current_state[name].dtype)
            for name, tensor in swa_state.items()
        }
        base_model.load_state_dict(avg_state, strict=True)
        # Re-evaluate after SWA
        val_loss, val_bpb = eval_val(
            model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"swa:post_eval val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # ── Post-training compression ──────────────────────────────────────────────
    # Merge LoRA → prune → SVD compress → then measure size

    if LORA_ENABLED:
        merge_lora(base_model)
        log0(f"LoRA merged into base weights")

    if PRUNING_ENABLED:
        prune_model_structured(base_model, ratio=PRUNING_RATIO)
        log0(f"Structured pruning: removed {PRUNING_RATIO*100:.0f}% of MLP neurons")
        # Optional: finetune after pruning (uses remaining time budget)
        if PRUNING_FINETUNE_FRAC > 0:
            ft_steps = max(1, int(step * PRUNING_FINETUNE_FRAC))
            log0(f"Post-pruning finetune: {ft_steps} steps")
            model.train()
            for ft_step in range(ft_steps):
                zero_grad_all()
                x, y = train_loader.next_batch(TRAIN_BATCH_TOKENS, TRAIN_SEQ_LEN, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                (loss * grad_scale).backward()
                for opt in optimizers:
                    for group in opt.param_groups:
                        group["lr"] = group["base_lr"] * 0.01  # very low LR for finetuning
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
            # Re-evaluate
            val_loss, val_bpb = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"post_prune_finetune val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    if SVD_COMPRESS:
        svd_compress_model(base_model, rank_frac=SVD_RANK_FRAC)
        log0(f"SVD compression: rank_frac={SVD_RANK_FRAC}")
        # Re-evaluate after SVD
        val_loss, val_bpb = eval_val(
            model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        log0(f"post_svd val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

    # ── Compression roundtrip test: compress → decompress → re-evaluate ──────
    # This measures the TRUE quality after compression, not just the size.
    # NOTE: all ranks must participate because eval_val uses DDP collectives.
    if COMPRESS_METHOD != "int6":
        log0(f"compress_roundtrip: testing {COMPRESS_METHOD} reconstruction quality...")
        orig_sd = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        if COMPRESS_METHOD == "factored_int4":
            result_q, meta_q = factored_int4_compress(orig_sd, rank=FACTORED_RANK)
            # Reconstruct: for factored entries, dequantize B,A and compute B@A
            recon_sd = {}
            for name in orig_sd:
                info = meta_q[name]
                if isinstance(info, dict) and info.get("type") == "factored_int4":
                    qB = result_q[name + ".B"].float()
                    sB = result_q[name + ".sB"].float()
                    qA = result_q[name + ".A"].float()
                    sA = result_q[name + ".sA"].float()
                    if sB.ndim > 0:
                        B = qB * sB.view(qB.shape[0], *([1] * (qB.ndim - 1)))
                    else:
                        B = qB * float(sB.item())
                    if sA.ndim > 0:
                        A = qA * sA.view(qA.shape[0], *([1] * (qA.ndim - 1)))
                    else:
                        A = qA * float(sA.item())
                    recon_sd[name] = (B @ A).to(orig_sd[name].dtype)
                elif isinstance(info, dict) and info["type"] == "int8":
                    q = result_q[name + ".q"].float()
                    s = result_q[name + ".scale"].float()
                    if s.ndim > 0:
                        recon_sd[name] = (q * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_sd[name].dtype)
                    else:
                        recon_sd[name] = (q * float(s.item())).to(orig_sd[name].dtype)
                else:
                    recon_sd[name] = result_q[name].to(orig_sd[name].dtype) if name in result_q else orig_sd[name]
            base_model.load_state_dict({k: v.to(device) for k, v in recon_sd.items()}, strict=True)
            val_loss_recon, val_bpb_recon = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"compress_roundtrip: {COMPRESS_METHOD} rank={FACTORED_RANK} "
                 f"val_bpb_before={val_bpb:.4f} val_bpb_after={val_bpb_recon:.4f} "
                 f"delta={val_bpb_recon - val_bpb:+.4f}")
            # Restore original weights for correct final summary
            base_model.load_state_dict({k: v.to(device) for k, v in orig_sd.items()}, strict=True)
        elif COMPRESS_METHOD == "nonuniform":
            result_q, meta_q = nonuniform_compress(
                orig_sd, NONUNIFORM_IMPORTANT_BITS, NONUNIFORM_UNIMPORTANT_BITS, NONUNIFORM_IMPORTANT_FRAC)
            recon_sd = {}
            for name in orig_sd:
                info = meta_q[name]
                if isinstance(info, dict):
                    q = result_q[name + ".q"].float()
                    s = result_q[name + ".scale"].float()
                    if s.ndim > 0:
                        W = q * s.view(q.shape[0], *([1] * (q.ndim - 1)))
                    else:
                        W = q * float(s.item())
                    if info.get("hadamard"):
                        W = _block_hadamard_rotate(W)  # inverse rotation (self-inverse)
                    recon_sd[name] = W.to(orig_sd[name].dtype)
                else:
                    recon_sd[name] = result_q[name].to(orig_sd[name].dtype) if name in result_q else orig_sd[name]
            base_model.load_state_dict({k: v.to(device) for k, v in recon_sd.items()}, strict=True)
            val_loss_recon, val_bpb_recon = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"compress_roundtrip: {COMPRESS_METHOD} "
                 f"val_bpb_before={val_bpb:.4f} val_bpb_after={val_bpb_recon:.4f} "
                 f"delta={val_bpb_recon - val_bpb:+.4f}")
            base_model.load_state_dict({k: v.to(device) for k, v in orig_sd.items()}, strict=True)
        elif COMPRESS_METHOD == "fft_int4":
            result_q, meta_q = fft_int4_compress(orig_sd, keep_frac=FFT_KEEP_FRAC)
            recon_sd = {}
            for name in orig_sd:
                info = meta_q[name]
                if isinstance(info, dict) and info.get("type") == "fft_int4":
                    qr = result_q[name + ".qr"].float()
                    sr = result_q[name + ".sr"].float()
                    qi = result_q[name + ".qi"].float()
                    si = result_q[name + ".si"].float()
                    orig_shape = result_q[name + ".shape"].tolist()
                    # Dequantize real and imaginary parts
                    if sr.ndim > 0:
                        real = qr * sr.view(qr.shape[0], *([1] * (qr.ndim - 1)))
                    else:
                        real = qr * float(sr.item())
                    if si.ndim > 0:
                        imag = qi * si.view(qi.shape[0], *([1] * (qi.ndim - 1)))
                    else:
                        imag = qi * float(si.item())
                    F_c = torch.complex(real, imag)
                    recon = torch.fft.irfft(F_c, n=orig_shape[1], dim=1)
                    recon_sd[name] = recon.to(orig_sd[name].dtype)
                elif isinstance(info, dict) and info.get("type") == "int8":
                    q = result_q[name + ".q"].float()
                    s = result_q[name + ".scale"].float()
                    if s.ndim > 0:
                        recon_sd[name] = (q * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_sd[name].dtype)
                    else:
                        recon_sd[name] = (q * float(s.item())).to(orig_sd[name].dtype)
                else:
                    recon_sd[name] = result_q[name].to(orig_sd[name].dtype) if name in result_q else orig_sd[name]
            base_model.load_state_dict({k: v.to(device) for k, v in recon_sd.items()}, strict=True)
            val_loss_recon, val_bpb_recon = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"compress_roundtrip: {COMPRESS_METHOD} keep_frac={FFT_KEEP_FRAC} "
                 f"val_bpb_before={val_bpb:.4f} val_bpb_after={val_bpb_recon:.4f} "
                 f"delta={val_bpb_recon - val_bpb:+.4f}")
            base_model.load_state_dict({k: v.to(device) for k, v in orig_sd.items()}, strict=True)

    # Summary
    log0(f"peak_vram_mb: {torch.cuda.max_memory_allocated() // 1024 // 1024}")

    if master_process:
        model_bytes = measure_model_size_advanced(base_model.state_dict())
        code_bytes = len(Path(__file__).read_text().encode("utf-8"))
        total_bytes = model_bytes + code_bytes
        log0(f"---")
        log0(f"val_bpb:          {val_bpb:.6f}")
        log0(f"val_loss:         {val_loss:.6f}")
        log0(f"training_seconds: {training_time_ms / 1000:.1f}")
        log0(f"peak_vram_mb:     {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f}")
        log0(f"total_tokens_M:   {step * TRAIN_BATCH_TOKENS / 1e6:.1f}")
        log0(f"num_steps:        {step}")
        log0(f"num_params_M:     {n_params / 1e6:.1f}")
        log0(f"model_bytes:      {model_bytes}")
        log0(f"code_bytes:       {code_bytes}")
        log0(f"total_bytes:      {total_bytes}")
        log0(f"under_16mb:       {total_bytes <= 16_000_000}")
        log0(f"num_layers:       {NUM_LAYERS}")

        # Phase 1: Save trained adapter weights for later freeze
        if SAVE_ADAPTER_PATH and base_model.teacher_adapter is not None:
            adapter_state = {
                k: v.cpu() for k, v in base_model.teacher_adapter.state_dict().items()
            }
            torch.save(adapter_state, SAVE_ADAPTER_PATH)
            log0(f"Saved adapter weights to {SAVE_ADAPTER_PATH}")
            log0(f"  gate={base_model.teacher_adapter.gate.item():.4f} "
                 f"(sigmoid={torch.sigmoid(base_model.teacher_adapter.gate).item():.4f})")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
