#!/usr/bin/env -S python3 -u
"""
Unified Retrodiction v4096 — configurable layers/dim + compile warmup reset.

Usage:
  python train_retro_v4096.py --layers=9 --dim=448 --train_budget_secs=600
  torchrun --nproc_per_node=8 train_retro_v4096.py --layers=11 --dim=512 --train_budget_secs=540
"""
from __future__ import annotations

import argparse
import glob
import lzma
import math
import os
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict
from contextlib import nullcontext

import numpy as np
import sentencepiece as spm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# FlashAttention 3 (optional, falls back to SDPA)
try:
    from flash_attn import flash_attn_func
    HAS_FA3 = True
except ImportError:
    HAS_FA3 = False

# ==============================================================================
# CONFIG
# ==============================================================================
COMPUTE_DTYPE = torch.bfloat16

DATA_DIR = "/workspace/data_v4096_full"
TOKENIZER_PATH = "/workspace/bpe_v4096.model"

# These get overridden by args in main()
VOCAB_SIZE = 4096
NUM_LAYERS = 9           # overridden by --layers
MODEL_DIM = 448          # overridden by --dim
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 3
BIGRAM_DIM = 128         # auto-scaled: dim // 4

ROPE_BASE = 10000.0
QK_GAIN_INIT = 1.5
TIED_EMBED_INIT_STD = 0.005
LOGIT_SOFTCAP = 30.0
SEQ_LEN = 1024

XSA_LAST_N = 4           # auto-scaled: min(4, layers)
BIGRAM_BUCKETS = 2048
ROPE_DIMS = 0            # 0=full RoPE, 16=partial (Top 1 uses 16)

# Optimizer
TIED_EMBED_LR = 0.035
MATRIX_LR = 0.025
SCALAR_LR = 0.025
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
MUON_MOMENTUM = 0.99
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.92
MUON_MOMENTUM_WARMUP_STEPS = 1500
WEIGHT_DECAY = 0.04
GRAD_CLIP = 0.3

# EMA
EMA_DECAY = 0.997

# Retrodiction
RETRO_ALPHA = 0.3

SEED = 1337

# ==============================================================================
# DISTRIBUTED HELPERS
# ==============================================================================
def is_distributed():
    return dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main():
    return get_rank() == 0

def print_main(*args, **kwargs):
    if is_main():
        try:
            print(*args, **kwargs, flush=True)
        except (BrokenPipeError, OSError):
            pass  # SSH pipe broken — don't crash rank 0

# ==============================================================================
# MATH HELPERS
# ==============================================================================
def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def zeropower_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7,
                            row_normalize: bool = False) -> torch.Tensor:
    """Newton-Schulz iteration to approximate the matrix sign function (Muon optimizer).
    row_normalize=True enables MuonEq-R (arXiv:2603.28254)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    if row_normalize:
        # MuonEq-R: row-normalize before NS to balance row scales
        row_norms = x.norm(dim=-1, keepdim=True).clamp(min=eps)
        x = x / row_norms
    x = x / (x.norm() + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


# ==============================================================================
# ROTARY POSITION EMBEDDING
# ==============================================================================
class RotaryEmbedding(nn.Module):
    """Standard rotary position embedding."""
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cos/sin cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, head_dim//2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, rope_dims: int = 0) -> torch.Tensor:
        """x: [B, H, T, D]. rope_dims>0: only apply RoPE to first rope_dims dims."""
        seq_len = x.shape[2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)

        if rope_dims > 0 and rope_dims < x.shape[-1]:
            # Partial RoPE: only rotate first rope_dims, pass through the rest
            x_rope = x[..., :rope_dims]
            x_pass = x[..., rope_dims:]
            r1 = x_rope[..., :rope_dims // 2]
            r2 = x_rope[..., rope_dims // 2:]
            cos_r = cos[..., :rope_dims // 2]
            sin_r = sin[..., :rope_dims // 2]
            x_rope = torch.cat([r1 * cos_r - r2 * sin_r, r2 * cos_r + r1 * sin_r], dim=-1)
            return torch.cat([x_rope, x_pass], dim=-1)
        else:
            # Full RoPE (original)
            x1 = x[..., :self.head_dim // 2]
            x2 = x[..., self.head_dim // 2:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ==============================================================================
# DATA LOADING
# ==============================================================================
def load_data_shard(path):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern, rank=0, world_size=1):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        # For DDP: each rank skips to its portion of files
        self.rank = rank
        self.world_size = world_size
        self.epoch = 1
        self.file_idx = rank  # stagger file start per rank
        self.tokens = load_data_shard(self.files[self.file_idx % len(self.files)])
        self.pos = 0

    def next_file(self):
        self.file_idx += self.world_size  # each rank advances by world_size
        if self.file_idx >= len(self.files):
            self.file_idx = self.rank
            self.epoch += 1
            print_main(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx % len(self.files)])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern, device, rank=0, world_size=1):
        self.stream = TokenStream(pattern, rank=rank, world_size=world_size)
        self.device = device

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return (torch.tensor(x, dtype=torch.long, device=self.device),
                torch.tensor(y, dtype=torch.long, device=self.device))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================
class CastedLinear(nn.Module):
    """Linear with float32 weights, cast to input dtype at forward time."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Initialize using default nn.Linear init, but store as float32 Parameter
        temp = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = nn.Parameter(temp.weight.data.float())  # [out, in]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class RMSNormNoWeight(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x)


class DualModeAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float, use_xsa: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_groups = num_heads // num_kv_heads  # GQA group size
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.ones(num_heads, dtype=torch.float32) * qk_gain_init)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=SEQ_LEN + 64)
        self.scale = self.head_dim ** -0.5
        self.use_xsa = use_xsa

    def _xsa(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Subtract self-value projection (XSA)."""
        bsz, seqlen, dim = y.shape
        hd = self.head_dim
        nkv = self.num_kv_heads
        group = self.num_groups
        # y: [B, T, nh*hd] -> [B, T, nkv, group, hd]
        y_g = y.reshape(bsz, seqlen, nkv, group, hd)
        # v: [B, nkv, T, hd] -> [B, T, nkv, hd]
        v_t = v.transpose(1, 2)
        vn = v_t / (v_t.norm(dim=-1, keepdim=True) + 1e-8)
        vn = vn.unsqueeze(3)  # [B, T, nkv, 1, hd]
        # Project y onto v direction and subtract
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(bsz, seqlen, dim)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm + RoPE
        q = self.rope(rms_norm(q).to(COMPUTE_DTYPE), rope_dims=ROPE_DIMS)
        k = self.rope(rms_norm(k).to(COMPUTE_DTYPE), rope_dims=ROPE_DIMS)
        # Q gain
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        if USE_FA3:
            # FA3 expects [B, T, H, D] and handles GQA natively
            q_fa = q.transpose(1, 2)   # [B, T, nh, hd]
            k_fa = k.transpose(1, 2)   # [B, T, nkv, hd]
            v_fa = v.transpose(1, 2)   # [B, T, nkv, hd]
            y = flash_attn_func(q_fa, k_fa, v_fa, causal=is_causal,
                                softmax_scale=self.scale)  # [B, T, nh, hd]
            y = y.reshape(bsz, seqlen, dim)
        else:
            # GQA: expand K/V to match Q heads
            if self.num_kv_heads < self.num_heads:
                k = k.repeat_interleave(self.num_groups, dim=1)
                v_expanded = v.repeat_interleave(self.num_groups, dim=1)
            else:
                v_expanded = v
            # PyTorch SDPA (FA2 backend)
            y = F.scaled_dot_product_attention(q, k, v_expanded, is_causal=is_causal,
                                               scale=self.scale)
            y = y.transpose(1, 2).reshape(bsz, seqlen, dim)

        # XSA
        if self.use_xsa:
            y = self._xsa(y, v)
        return self.proj(y)


class MLP(nn.Module):
    """LeakyReLU(0.5)^2 MLP."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = torch.where(h >= 0, h, 0.5 * h)
        return self.proj(h * h)


class BigramHashEmbedding(nn.Module):
    """Learned bigram hash embeddings."""
    def __init__(self, buckets: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.buckets = buckets
        self.embed = nn.Embedding(buckets, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.int()
        mod = self.buckets - 1
        shifted = torch.cat([torch.full((t.shape[0], 1), mod, dtype=torch.int32,
                                        device=t.device),
                             t[:, :-1]], dim=1)
        hashed = (36313 * t + 27191 * shifted) % mod
        return hashed.long()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(h.dtype)


class SmearGate(nn.Module):
    """Learned blending with previous token."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate.to(x.dtype)).unsqueeze(0).unsqueeze(0)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float,
                 layer_idx: int = 0, use_xsa: bool = False):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = DualModeAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                      use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        # LN Scale: 1/sqrt(layer+1) — matches MLX version exactly
        self.ln_scale = 1.0 / math.sqrt(layer_idx + 1)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([
            torch.ones(dim, dtype=torch.float32),
            torch.zeros(dim, dtype=torch.float32),
        ]))

    def forward(self, x: torch.Tensor, x0: torch.Tensor,
                is_causal: bool = True) -> torch.Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0].unsqueeze(0).unsqueeze(0) * x + mix[1].unsqueeze(0).unsqueeze(0) * x0
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale, is_causal=is_causal)
        x = x + self.attn_scale.to(x.dtype).unsqueeze(0).unsqueeze(0) * attn_out
        x = x + self.mlp_scale.to(x.dtype).unsqueeze(0).unsqueeze(0) * self.mlp(
            self.mlp_norm(x) * self.ln_scale)
        return x


class GPTv2(nn.Module):
    """Standard 11L GPT with U-net skip connections. No depth recurrence."""
    def __init__(self):
        super().__init__()
        self.logit_softcap = LOGIT_SOFTCAP
        self.tok_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        nn.init.normal_(self.tok_emb.weight, std=TIED_EMBED_INIT_STD)
        self.tok_emb.weight.data = self.tok_emb.weight.data.to(COMPUTE_DTYPE)

        self.bigram = BigramHashEmbedding(BIGRAM_BUCKETS, BIGRAM_DIM, MODEL_DIM)
        self.smear = SmearGate(MODEL_DIM)

        # U-net: 5 encoder layers + 6 decoder layers = 11 total
        self.num_encoder_layers = NUM_LAYERS // 2   # 5
        self.num_decoder_layers = NUM_LAYERS - self.num_encoder_layers  # 6
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)  # 5
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, MODEL_DIM,
                                                    dtype=torch.float32))

        # 11 unique layers (standard, no weight sharing)
        self.blocks = nn.ModuleList()
        for i in range(NUM_LAYERS):
            use_xsa = i >= (NUM_LAYERS - XSA_LAST_N)  # last 4 layers (7, 8, 9, 10)
            self.blocks.append(
                Block(MODEL_DIM, NUM_HEADS, NUM_KV_HEADS, MLP_MULT, ROPE_BASE, QK_GAIN_INIT,
                      layer_idx=i, use_xsa=use_xsa)
            )
        self.final_norm = RMSNormNoWeight()

        # Zero out output projections (matches MLX init)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)

    def softcap(self, logits: torch.Tensor) -> torch.Tensor:
        c = self.logit_softcap
        return c * torch.tanh(logits / c)

    def forward_hidden(self, input_ids: torch.Tensor,
                       is_causal: bool = True) -> torch.Tensor:
        x = self.tok_emb(input_ids).to(COMPUTE_DTYPE)
        x = x + self.bigram(input_ids).to(COMPUTE_DTYPE)
        x = rms_norm(x)
        x = self.smear(x)
        x0 = x

        # Encoder: layers 0..4, save skips
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, is_causal=is_causal)
            skips.append(x)

        # Decoder: layers 5..10, add U-net skip connections
        for i in range(self.num_decoder_layers):
            if skips:
                skip_w = self.skip_weights[i]
                x = x + skip_w.to(x.dtype).unsqueeze(0).unsqueeze(0) * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, is_causal=is_causal)

        return self.final_norm(x)

    def forward(self, input_ids: torch.Tensor,
                target_ids: torch.Tensor | None = None,
                is_causal: bool = True,
                retrodiction: bool = False) -> torch.Tensor:
        """
        If target_ids is None: return hidden states [B, T, D].
        If target_ids is given: return scalar loss.
        """
        if target_ids is None:
            return self.forward_hidden(input_ids, is_causal=is_causal)

        # Standard AR loss: h[t] -> token[t+1]
        h = self.forward_hidden(input_ids, is_causal=is_causal)
        h_flat = h.reshape(-1, MODEL_DIM)
        y = target_ids.reshape(-1)
        logits = self.softcap(F.linear(h_flat, self.tok_emb.weight.to(h_flat.dtype)))
        forward_loss = F.cross_entropy(logits.float(), y, reduction="mean")

        if not retrodiction:
            return forward_loss

        # Retrodiction: backward AR loss
        # .flip() in compiled graph causes NaN, so caller must pass pre-flipped tensors
        # via x_rev, y_rev arguments (see training loop)
        return forward_loss  # retrodiction handled in training loop, not here

    def loss_fn(self, input_ids: torch.Tensor, target_ids: torch.Tensor,
                is_causal: bool = True) -> torch.Tensor:
        """Convenience method for eval (forward-only loss)."""
        return self.forward(input_ids, target_ids, is_causal=is_causal, retrodiction=False)


# ==============================================================================
# OPTIMIZER (Muon + AdamW split)
# ==============================================================================
CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
                    "gate", "scale", "ln_scale")


class Muon:
    """Muon optimizer: Newton-Schulz orthogonalization for matrix parameters."""
    def __init__(self, params_dict: dict[str, nn.Parameter], row_normalize: bool = False):
        self.keys = list(params_dict.keys())
        self.buffers = {k: torch.zeros_like(p.data) for k, p in params_dict.items()}
        self.row_normalize = row_normalize

    @torch.no_grad()
    def step(self, params_dict: dict[str, nn.Parameter],
             grads_dict: dict[str, torch.Tensor],
             step: int, lr_mul: float):
        t = min(step / max(MUON_MOMENTUM_WARMUP_STEPS, 1), 1.0)
        momentum = (1.0 - t) * MUON_MOMENTUM_WARMUP_START + t * MUON_MOMENTUM
        lr = MATRIX_LR * lr_mul
        for k in self.keys:
            if k not in grads_dict:
                continue
            p = params_dict[k]
            g = grads_dict[k]
            g_norm = g.norm()
            if g_norm > GRAD_CLIP:
                g = g * (GRAD_CLIP / (g_norm + 1e-8))
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_BACKEND_STEPS,
                                                row_normalize=self.row_normalize)
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            p.data.mul_(1 - lr * WEIGHT_DECAY).add_(g_ortho.to(p.dtype), alpha=-lr * scale)


class SplitOptimizers:
    """Split parameters into Muon (matrix), AdamW (embedding), AdamW (scalar)."""
    def __init__(self, model: nn.Module, muon_eq_r: bool = False):
        self.embed_key = "tok_emb.weight"
        self.matrix_params = {}
        self.scalar_params = {}
        self.embed_param = None

        for name, p in model.named_parameters():
            if name == self.embed_key:
                self.embed_param = p
            elif p.ndim == 2 and not any(pat in name for pat in CONTROL_PATTERNS):
                self.matrix_params[name] = p
            else:
                self.scalar_params[name] = p

        self.muon = Muon(self.matrix_params, row_normalize=muon_eq_r)

        # AdamW for embedding
        self.adam_embed = torch.optim.AdamW(
            [self.embed_param], lr=TIED_EMBED_LR,
            betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=0.0
        )
        # AdamW for scalar params
        self.adam_scalar = torch.optim.AdamW(
            list(self.scalar_params.values()), lr=SCALAR_LR,
            betas=(BETA1, BETA2), eps=ADAM_EPS, weight_decay=0.0
        )

        print_main(f"  Muon params: {len(self.matrix_params)} tensors")
        print_main(f"  Scalar params: {len(self.scalar_params)} tensors")
        print_main(f"  Embed param: {self.embed_key}")

    def step(self, model: nn.Module, step: int, lr_mul: float, async_muon: bool = False):
        """Update all param groups. If async_muon=True, run Muon on separate stream."""
        # Gather gradients
        grads_dict = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads_dict[name] = p.grad

        # Muon step
        muon_grads = {k: grads_dict[k] for k in self.matrix_params if k in grads_dict}
        if async_muon:
            if not hasattr(self, '_muon_stream'):
                self._muon_stream = torch.cuda.Stream()
            with torch.cuda.stream(self._muon_stream):
                self.muon.step(self.matrix_params, muon_grads, step=step, lr_mul=lr_mul)
        else:
            self.muon.step(self.matrix_params, muon_grads, step=step, lr_mul=lr_mul)

        # AdamW for embedding (fast, always on default stream)
        for pg in self.adam_embed.param_groups:
            pg["lr"] = TIED_EMBED_LR * lr_mul
        self.adam_embed.step()

        # AdamW for scalars (fast, always on default stream)
        for pg in self.adam_scalar.param_groups:
            pg["lr"] = SCALAR_LR * lr_mul
        self.adam_scalar.step()

    def sync_muon(self):
        """Wait for async Muon to finish (call before next backward)."""
        if hasattr(self, '_muon_stream'):
            self._muon_stream.synchronize()

    def zero_grad(self):
        self.adam_embed.zero_grad(set_to_none=True)
        self.adam_scalar.zero_grad(set_to_none=True)
        for p in self.matrix_params.values():
            p.grad = None


# ==============================================================================
# SENTENCEPIECE BPB
# ==============================================================================
def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def compute_bpb(total_nll, total_tokens, total_bytes):
    avg_loss = total_nll / total_tokens
    bpt = avg_loss / math.log(2.0)
    return bpt * (total_tokens / total_bytes)


# ==============================================================================
# LR SCHEDULE
# ==============================================================================
def lr_schedule(step, total_steps, warmdown_iters):
    warmdown_start = max(total_steps - warmdown_iters, 0)
    if warmdown_start <= step < total_steps:
        return max((total_steps - step) / max(warmdown_iters, 1), 0.0)
    return 1.0


# ==============================================================================
# INT6 QUANTIZATION + LZMA SAVE
# ==============================================================================
def pack_int6(values: np.ndarray) -> np.ndarray:
    """Pack 6-bit signed integers into bytes. 4 values -> 3 bytes."""
    # values: int8 array in [-31, 31], shift to unsigned [0, 62]
    unsigned = (values.astype(np.int16) + 31).astype(np.uint8)  # [0, 62], fits in 6 bits
    n = len(unsigned)
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        unsigned = np.concatenate([unsigned, np.zeros(pad, dtype=np.uint8)])
    n_padded = len(unsigned)
    # Pack 4 values (24 bits) into 3 bytes
    a = unsigned[0::4]
    b = unsigned[1::4]
    c = unsigned[2::4]
    d = unsigned[3::4]
    byte0 = (a << 2) | (b >> 4)
    byte1 = ((b & 0x0F) << 4) | (c >> 2)
    byte2 = ((c & 0x03) << 6) | d
    packed = np.stack([byte0, byte1, byte2], axis=1).flatten()
    return packed, n  # return original length for unpadding


def unpack_int6(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack 6-bit values from bytes."""
    byte0 = packed[0::3]
    byte1 = packed[1::3]
    byte2 = packed[2::3]
    a = byte0 >> 2
    b = ((byte0 & 0x03) << 4) | (byte1 >> 4)
    c = ((byte1 & 0x0F) << 2) | (byte2 >> 6)
    d = byte2 & 0x3F
    unsigned = np.stack([a, b, c, d], axis=1).flatten()[:n_values]
    return unsigned.astype(np.int16) - 31  # back to signed [-31, 31]


def quantize_int6_perrow(tensor_f32: np.ndarray):
    """
    Per-row int6 quantization (range [-31, 31]) with independent scale per row.
    For 1-D tensors, falls back to per-tensor.
    Returns: (flat_quantized int8, scales float32 array, original shape)
    """
    arr = tensor_f32.astype(np.float32)
    shape = arr.shape
    if arr.ndim >= 2:
        rows = arr.reshape(shape[0], -1)           # [R, C]
        absmax = np.abs(rows).max(axis=1)          # [R]
        absmax = np.where(absmax < 1e-12, 1.0, absmax)
        scales = (absmax / 31.0).astype(np.float32)  # [R]
        quantized = np.clip(
            np.round(rows / scales[:, None]), -31, 31
        ).astype(np.int8).flatten()
    else:
        absmax = float(np.abs(arr).max())
        scale = np.float32(absmax / 31.0 if absmax > 1e-12 else 1.0)
        scales = np.array([scale], dtype=np.float32)
        quantized = np.atleast_1d(
            np.clip(np.round(arr / scale), -31, 31).astype(np.int8))
    return quantized, scales, shape


def save_model_int6_lzma(state_dict: dict, path: str):
    """Save model with per-row int6 quantization as int8 + lzma compression.
    Uses int8 storage (not 6-bit packing) because lzma compresses int8 much better
    than packed 6-bit (int8 values in [-31,31] have low entropy → high lzma ratio)."""
    quantized_data = {}
    for k, v in state_dict.items():
        arr = v.cpu().float().numpy()
        q, scales, shape = quantize_int6_perrow(arr)
        # Store as int8 (NOT 6-bit packed) — lzma compresses this much better
        quantized_data[k] = {"q": q, "s": scales, "sh": shape}
    compressed = lzma.compress(pickle.dumps(quantized_data), preset=9)
    with open(path, "wb") as f:
        f.write(compressed)
    return len(compressed)


def load_model_int6_lzma(path: str):
    """Load per-row int6 quantized + lzma compressed model."""
    with open(path, "rb") as f:
        data = pickle.loads(lzma.decompress(f.read()))
    state_dict = {}
    for k, v in data.items():
        q = v["q"]  # int8 array
        shape = v["sh"]
        scales = v["s"]                              # float32 array
        if len(scales) > 1:                          # per-row
            rows = q.reshape(shape[0], -1).astype(np.float32)
            rows *= scales[:, None]
            arr = rows.reshape(shape)
        else:                                        # per-tensor (1D)
            arr = q.astype(np.float32) * float(scales[0])
            arr = arr.reshape(shape)
        state_dict[k] = torch.tensor(arr)
    return state_dict


# ==============================================================================
# MIXED QUANTIZATION (Gemma-style Q4/Q5/Q6)
# ==============================================================================
def quantize_perrow(tensor_f32: np.ndarray, bits: int):
    """Per-row quantization to N bits. bits=4→[-8,7], bits=5→[-16,15], bits=6→[-31,31]."""
    max_val = 2 ** (bits - 1) - 1
    arr = tensor_f32.astype(np.float32)
    shape = arr.shape
    if arr.ndim >= 2:
        rows = arr.reshape(shape[0], -1)
        absmax = np.abs(rows).max(axis=1)
        absmax = np.where(absmax < 1e-12, 1.0, absmax)
        scales = (absmax / max_val).astype(np.float32)
        quantized = np.clip(np.round(rows / scales[:, None]),
                            -(max_val + 1), max_val).astype(np.int8).flatten()
    else:
        absmax = float(np.abs(arr).max())
        scale = np.float32(absmax / max_val if absmax > 1e-12 else 1.0)
        scales = np.array([scale], dtype=np.float32)
        quantized = np.atleast_1d(
            np.clip(np.round(arr / scale), -(max_val + 1), max_val).astype(np.int8))
    return quantized, scales, shape


# Gemma-style: sensitive layers → higher bits, big/insensitive → lower bits
QUANT_STRATEGIES = {
    "aggressive": {
        # Q4: Q, K, embedding, gate/up, bigram (big, less sensitive)
        # Q5: output proj, MLP down (medium sensitive)
        # Q6: V (most sensitive)
        "attn.c_q": 4, "attn.c_k": 4, "attn.c_v": 6, "attn.proj": 5,
        "mlp.fc": 4, "mlp.proj": 5,
        "tok_emb": 4, "bigram": 4, "smear": 6, "skip": 6,
        "_default": 6,
    },
    "smart": {
        # Q4: Q, K, gate/up (biggest tensors)
        # Q5: everything else
        "attn.c_q": 4, "attn.c_k": 4, "attn.c_v": 5, "attn.proj": 5,
        "mlp.fc": 4, "mlp.proj": 5,
        "tok_emb": 5, "bigram": 4, "smear": 6, "skip": 6,
        "_default": 6,
    },
}


def save_model_mixed_quant(state_dict: dict, path: str, strategy_name: str = "aggressive"):
    """Save with mixed Q4/Q5/Q6 quantization + lzma compression."""
    strategy = QUANT_STRATEGIES[strategy_name]
    quantized_data = {}
    for k, v in state_dict.items():
        arr = v.cpu().float().numpy()
        # Determine bits for this weight
        bits = strategy.get("_default", 6)
        for pattern, b in strategy.items():
            if pattern != "_default" and pattern in k:
                bits = b
                break
        q, scales, shape = quantize_perrow(arr, bits)
        quantized_data[k] = {"q": q, "s": scales, "sh": shape, "bits": bits}
    compressed = lzma.compress(pickle.dumps(quantized_data), preset=9)
    with open(path, "wb") as f:
        f.write(compressed)
    return len(compressed)


def load_model_mixed_quant(path: str):
    """Load mixed quantized + lzma compressed model."""
    with open(path, "rb") as f:
        data = pickle.loads(lzma.decompress(f.read()))
    state_dict = {}
    for k, v in data.items():
        q = v["q"]
        shape = v["sh"]
        scales = v["s"]
        bits = v.get("bits", 6)
        max_val = 2 ** (bits - 1) - 1
        if len(scales) > 1:
            rows = q.reshape(shape[0], -1).astype(np.float32)
            rows *= scales[:, None]
            arr = rows.reshape(shape)
        else:
            arr = q.astype(np.float32) * float(scales[0])
            arr = arr.reshape(shape)
        state_dict[k] = torch.tensor(arr)
    return state_dict


# ==============================================================================
# EMA
# ==============================================================================
class EMAState:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v, alpha=1 - d)

    def apply(self, model: nn.Module):
        """Temporarily load EMA weights into model. Returns original state."""
        orig = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return orig

    def restore(self, model: nn.Module, orig: dict):
        """Restore original weights after EMA eval."""
        model.load_state_dict(orig, strict=False)


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Retrodiction v4096 — PyTorch H100")
    parser.add_argument("--layers", type=int, default=9, help="Number of layers (5/7/9/11)")
    parser.add_argument("--dim", type=int, default=448, help="Model dimension")
    parser.add_argument("--vocab_size", type=int, default=0,
                        help="Vocab size (0=auto from tokenizer)")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--xsa_all", action="store_true",
                        help="Apply XSA to ALL layers (Top 1 does this), default=last 4 only")
    parser.add_argument("--bigram_buckets", type=int, default=0,
                        help="BigramHash buckets (0=default 2048, Top 1 uses 3072)")
    parser.add_argument("--bigram_dim_override", type=int, default=0,
                        help="BigramHash embed dim (0=auto dim//4, Top 1 uses 112)")
    parser.add_argument("--no_retro", action="store_true", help="Disable retrodiction")
    parser.add_argument("--retro_fadeout", type=float, default=1.0,
                        help="Fraction of training to use retro (0.5=first half retro, second half AR)")
    parser.add_argument("--retro_late_start", type=float, default=0.0,
                        help="Start retro only after this fraction of training (0.8=last 20 pct only)")
    parser.add_argument("--retro_mode", type=str, default="merged",
                        choices=["merged", "alternating", "partial"],
                        help="Retro implementation: "
                             "merged=50/50 merged batch (2x overhead, default), "
                             "alternating=odd/even steps (1.0x overhead), "
                             "partial=85/15 mixed batch (1.15x overhead)")
    parser.add_argument("--retro_partial_frac", type=float, default=0.15,
                        help="Fraction of batch used for reverse in 'partial' mode (default 0.15)")
    parser.add_argument("--retro_alt_ratio", type=int, default=2,
                        help="In alternating mode: do reverse every N steps (2=50/50, 10=90/10)")
    parser.add_argument("--cdm", action="store_true",
                        help="Pure CDM mode: bidirectional attention + variable mask training")
    parser.add_argument("--shared_cdm", action="store_true",
                        help="Shared-weight mode: alternate AR loss + CDM loss each step")
    parser.add_argument("--cdm_weight", type=float, default=0.3,
                        help="Weight for CDM loss in shared mode (default 0.3)")
    parser.add_argument("--cdm_mask_low", type=float, default=0.15,
                        help="CDM min mask rate (default 0.15)")
    parser.add_argument("--cdm_mask_high", type=float, default=0.50,
                        help="CDM max mask rate (default 0.50)")
    # === New optimizations (from PR #1334 analysis) ===
    parser.add_argument("--ortho_init", action="store_true",
                        help="Orthogonal init for Q/K/V/MLP-up (Top 1 uses this)")
    parser.add_argument("--rope_dims", type=int, default=0,
                        help="Partial RoPE: only apply to first N dims (0=full, Top 1 uses 16)")
    parser.add_argument("--qk_gain", type=float, default=0.0,
                        help="QK-Gain init value (0=use default 1.5, Top 1 uses 4.0, PR#1334 uses 5.0)")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Override weight decay (0=use default 0.04, PR#1334 uses 0.090)")
    parser.add_argument("--matrix_lr", type=float, default=0.0,
                        help="Override matrix LR (0=use default 0.025, PR#1334 uses 0.022)")
    parser.add_argument("--muon_eq_r", action="store_true",
                        help="MuonEq-R: row-normalize gradient before Newton-Schulz")
    parser.add_argument("--ema_from_start", action="store_true",
                        help="Start EMA from step 1 (Top 1 does this), default=80pct of training")
    parser.add_argument("--no_fa3", action="store_true",
                        help="Disable FlashAttention 3, use PyTorch SDPA instead")
    parser.add_argument("--split_logits", action="store_true",
                        help="Use split F.linear for retro (v1 style) instead of merged")
    parser.add_argument("--parallel_muon", action="store_true",
                        help="Run Muon optimizer on separate CUDA stream (overlap with next fwd)")
    parser.add_argument("--compile_warmup", type=int, default=20,
                        help="Compile warmup steps (0=skip warmup)")
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (1 for 8xH100 = 524K tokens/step)")
    parser.add_argument("--microbatch_tokens", type=int, default=65536,
                        help="Tokens per microbatch per GPU (65536 x 8 GPUs = 524K)")
    parser.add_argument("--warmdown", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--val_tokens", type=int, default=1_000_000)
    parser.add_argument("--save_path", type=str, default="golf_11L_retro.npz")
    parser.add_argument("--save_int6_path", type=str, default="golf_11L_retro_int6.lzma")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_11L",
                        help="Directory for periodic checkpoints")
    parser.add_argument("--checkpoint_every", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--test_steps", type=int, default=0,
                        help="If > 0, run N test steps on 1 GPU then exit")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (for debugging)")
    parser.add_argument("--train_budget_secs", type=int, default=0,
                        help="Wall-clock budget for training in seconds (0=disabled). "
                             "Use 540 for competition (9 min train + 1 min save+eval).")
    args = parser.parse_args()

    # Override globals from args
    global NUM_LAYERS, MODEL_DIM, BIGRAM_DIM, XSA_LAST_N, SEQ_LEN, VOCAB_SIZE
    NUM_LAYERS = args.layers
    MODEL_DIM = args.dim
    BIGRAM_DIM = args.bigram_dim_override if args.bigram_dim_override > 0 else max(64, args.dim // 4)
    XSA_LAST_N = args.layers if args.xsa_all else min(4, args.layers)
    SEQ_LEN = args.seq_len

    global BIGRAM_BUCKETS
    if args.bigram_buckets > 0:
        BIGRAM_BUCKETS = args.bigram_buckets
    if args.vocab_size > 0:
        VOCAB_SIZE = args.vocab_size

    # New optimization overrides
    global QK_GAIN_INIT, WEIGHT_DECAY, MATRIX_LR, ROPE_DIMS
    ROPE_DIMS = args.rope_dims
    if args.qk_gain > 0:
        QK_GAIN_INIT = args.qk_gain
    if args.wd > 0:
        WEIGHT_DECAY = args.wd
    if args.matrix_lr > 0:
        MATRIX_LR = args.matrix_lr

    global USE_FA3
    USE_FA3 = HAS_FA3 and not args.no_fa3
    if USE_FA3:
        print_main("Using FlashAttention 3")
    else:
        print_main(f"Using PyTorch SDPA (FA3 available: {HAS_FA3})")

    data_dir = args.data_dir
    tokenizer_path = args.tokenizer_path

    # ---- Distributed setup ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=5))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = get_rank()

    # ---- Test mode ----
    if args.test_steps > 0:
        args.steps = args.test_steps
        args.val_every = args.test_steps
        args.grad_accum = 1
        args.microbatch_tokens = 4096  # small for quick test
        print_main(f"=== TEST MODE: {args.test_steps} steps on 1 GPU ===")

    effective_batch = args.grad_accum * args.microbatch_tokens * world_size
    OFFICIAL_BATCH = 524_288
    if effective_batch != OFFICIAL_BATCH and args.test_steps == 0:
        print_main(f"WARNING: effective_batch={effective_batch:,} != official {OFFICIAL_BATCH:,}")
        print_main(f"   grad_accum={args.grad_accum} x microbatch={args.microbatch_tokens} x {world_size} GPUs")

    print_main("=" * 70)
    if args.no_retro:
        retro_tag = "Retro OFF"
    else:
        retro_tag = f"Retro {args.retro_mode.upper()}"
        if args.retro_mode == "partial":
            retro_tag += f" ({args.retro_partial_frac:.0%})"
        if args.retro_fadeout < 1.0:
            retro_tag += f" fadeout@{args.retro_fadeout:.0%}"
    print_main(f"Unified v4096 | {NUM_LAYERS}L d={MODEL_DIM} MLP={MLP_MULT}x "
               f"seq={SEQ_LEN} | {retro_tag} | steps={args.steps}")
    print_main(f"NUM_HEADS={NUM_HEADS} head_dim={MODEL_DIM//NUM_HEADS} "
               f"NUM_KV_HEADS={NUM_KV_HEADS} BIGRAM_DIM={BIGRAM_DIM}")
    print_main(f"Encoder={NUM_LAYERS//2} Decoder={NUM_LAYERS - NUM_LAYERS//2} "
               f"(U-net skip connections)")
    print_main(f"Retro alpha={RETRO_ALPHA} | XSA last {XSA_LAST_N} layers | "
               f"LeakyReLU^2 | BigramHash({BIGRAM_BUCKETS}) | EMA({EMA_DECAY})")
    print_main(f"Device: {device} | World size: {world_size} | "
               f"Grad accum: {args.grad_accum}")
    print_main(f"Effective batch: {effective_batch:,} tok/step")
    print_main("=" * 70)

    # ---- Tokenizer & BPB LUTs ----
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, VOCAB_SIZE))

    # ---- Validation data ----
    val_tokens = load_validation_tokens(f"{data_dir}/fineweb_val_*.bin", SEQ_LEN)
    if args.val_tokens > 0 and args.val_tokens < val_tokens.size:
        usable = (args.val_tokens // SEQ_LEN) * SEQ_LEN
        val_short = val_tokens[:usable + 1]
    else:
        val_short = val_tokens
    print_main(f"Val tokens: {val_tokens.size - 1:,} (eval on {val_short.size - 1:,})")

    # ---- Model ----
    # Same seed for model init (all ranks get same weights)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = GPTv2().to(device)

    # Orthogonal init (Top 1 uses this for Q/K/V/MLP-up)
    if args.ortho_init:
        for name, p in model.named_parameters():
            if p.ndim == 2 and p.shape[0] >= 64 and p.shape[1] >= 64:
                if 'proj' in name or 'mlp.proj' in name:
                    continue  # skip output projections (already zeros)
                if 'tok_emb' in name:
                    continue  # skip embedding
                nn.init.orthogonal_(p)
                print_main(f"  ortho_init: {name} {list(p.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print_main(f"Model params: {n_params:,} ({n_params/1e6:.1f}M)")
    print_main(f"Estimated size BF16: {n_params*2/1e6:.1f}MB | int6: {n_params*0.75/1e6:.1f}MB")

    # ---- DDP FIRST, then compile (order matters for 8xH100) ----
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DDP) else model

    if not args.no_compile and device.type == "cuda":
        print_main("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")
        print_main("  Compilation requested (will happen on first forward pass)")

    # Unwrap to raw model for optimizer / EMA / saving
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    # ---- Per-rank random seed (different masking/augmentation per GPU) ----
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)

    # ---- Optimizer ----
    opt = SplitOptimizers(raw_model, muon_eq_r=args.muon_eq_r)

    # ---- Data loader ----
    train_loader = TokenLoader(f"{data_dir}/fineweb_train_*.bin",
                               device=device, rank=rank, world_size=world_size)

    # ---- Compile warmup reset (NOT counted in training budget) ----
    if not args.no_compile and device.type == "cuda" and args.compile_warmup > 0:
        print_main(f"Compile warmup: {args.compile_warmup} steps (not counted in budget)...")
        t_warmup = time.perf_counter()
        # Save initial state
        initial_model_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
        initial_muon_bufs = {k: v.clone() for k, v in opt.muon.buffers.items()}
        initial_adam_embed_state = {k: v if not isinstance(v, torch.Tensor) else v.clone()
                                    for k, v in opt.adam_embed.state_dict().items()}
        initial_adam_scalar_state = {k: v if not isinstance(v, torch.Tensor) else v.clone()
                                     for k, v in opt.adam_scalar.state_dict().items()}
        # Run warmup steps to trigger JIT compilation.
        # CRITICAL: must walk the EXACT same code path as training loop,
        # otherwise first real step triggers re-compile (~3-4 min wasted).
        for _ws in range(args.compile_warmup):
            opt.zero_grad()
            x, y = train_loader.next_batch(args.microbatch_tokens, SEQ_LEN)
            if not args.no_retro and args.retro_mode == "merged":
                # Warmup: merged-batch path
                x_rev = x.flip(dims=[1])
                y_rev = y.flip(dims=[1])
                x_both = torch.cat([x, x_rev], dim=0)
                y_both = torch.cat([y, y_rev], dim=0)
                with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                    h = model(x_both)
                    h_fwd, h_bwd = h.chunk(2, dim=0)
                    w = raw_model.tok_emb.weight.to(h.dtype)
                    if args.split_logits:
                        logits_fwd = raw_model.softcap(F.linear(h_fwd.reshape(-1, MODEL_DIM), w))
                        logits_bwd = raw_model.softcap(F.linear(h_bwd.reshape(-1, MODEL_DIM), w))
                        fwd_loss = F.cross_entropy(logits_fwd.float(), y.reshape(-1))
                        bwd_loss = F.cross_entropy(logits_bwd.float(), y_rev.reshape(-1))
                    else:
                        logits_both = raw_model.softcap(F.linear(h.reshape(-1, MODEL_DIM), w))
                        BT = y.numel()
                        per_tok_loss = F.cross_entropy(
                            logits_both.float(), y_both.reshape(-1), reduction='none')
                        fwd_loss = per_tok_loss[:BT].mean()
                        bwd_loss = per_tok_loss[BT:].mean()
                    loss = fwd_loss + RETRO_ALPHA * bwd_loss
            elif not args.no_retro and args.retro_mode == "partial":
                # Warmup: partial batch path
                B = x.shape[0]
                n_rev = max(int(B * args.retro_partial_frac), 1)
                x_rev_part = x[:n_rev].flip(dims=[1])
                y_rev_part = y[:n_rev].flip(dims=[1])
                x_all = torch.cat([x, x_rev_part], dim=0)
                y_all = torch.cat([y, y_rev_part], dim=0)
                with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                    h = model(x_all)
                    w = raw_model.tok_emb.weight.to(h.dtype)
                    logits_all = raw_model.softcap(F.linear(h.reshape(-1, MODEL_DIM), w))
                    per_tok_loss = F.cross_entropy(
                        logits_all.float(), y_all.reshape(-1), reduction='none')
                    BT_fwd = y.numel()
                    fwd_loss = per_tok_loss[:BT_fwd].mean()
                    bwd_loss = per_tok_loss[BT_fwd:].mean()
                    loss = fwd_loss + RETRO_ALPHA * bwd_loss
            elif args.cdm:
                # CDM warmup: bidirectional + mask
                B, T = x.shape
                mask = torch.rand(B, T, device=x.device) < 0.3
                x_m = x.clone()
                n_m = mask.sum().item()
                if n_m > 0:
                    x_m[mask] = torch.randint(0, VOCAB_SIZE, (int(n_m),), device=x.device)
                with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                    h = raw_model.forward_hidden(x_m, is_causal=False)
                    logits = raw_model.softcap(
                        F.linear(h.reshape(-1, MODEL_DIM),
                                 raw_model.tok_emb.weight.to(h.dtype)))
                    loss = F.cross_entropy(logits.float(), x.reshape(-1))
            else:
                with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                    loss = model(x, target_ids=y, retrodiction=False)
            loss.backward()
            opt.step(raw_model, step=0, lr_mul=1.0)
        # Reset everything back to initial state
        raw_model.load_state_dict(initial_model_state)
        for k, v in initial_muon_bufs.items():
            opt.muon.buffers[k] = v
        opt.adam_embed.load_state_dict(initial_adam_embed_state)
        opt.adam_scalar.load_state_dict(initial_adam_scalar_state)
        # Reset data loader position
        train_loader.stream.file_idx = train_loader.stream.rank
        train_loader.stream.tokens = load_data_shard(
            train_loader.stream.files[train_loader.stream.file_idx % len(train_loader.stream.files)])
        train_loader.stream.pos = 0
        train_loader.stream.epoch = 1
        warmup_elapsed = time.perf_counter() - t_warmup
        print_main(f"Compile warmup done in {warmup_elapsed:.1f}s, weights reset.")

    # ---- Resume checkpoint ----
    start_step = 0
    ema_state = None
    ema_start_step = 0 if args.ema_from_start else int(args.steps * 0.8)

    if args.resume and os.path.exists(args.resume):
        print_main(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        if "ema" in ckpt and ckpt["ema"] is not None:
            ema_state = EMAState(raw_model)
            ema_state.shadow = ckpt["ema"]
        print_main(f"  Resumed at step {start_step}")

    print_main(f"EMA starts step {ema_start_step}")

    # ---- Checkpoint directory ----
    if is_main() and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Context manager for DDP no_sync on non-final accumulation steps
    def ddp_no_sync_ctx(is_last_micro):
        # Only use no_sync when grad_accum > 1 AND not last microstep
        # With grad_accum=1 (8xH100), never use no_sync (must always sync)
        if isinstance(model, DDP) and not is_last_micro and args.grad_accum > 1:
            return model.no_sync()
        return nullcontext()

    # ---- Validation ----
    @torch.no_grad()
    def eval_val(vtokens):
        raw_model.eval()
        batch_seqs = max(args.microbatch_tokens // SEQ_LEN, 1)
        total_seqs = (vtokens.size - 1) // SEQ_LEN
        total_nll = 0.0
        total_tok = 0
        total_bytes = 0.0
        for s in range(0, total_seqs, batch_seqs):
            e = min(s + batch_seqs, total_seqs)
            chunk = vtokens[s * SEQ_LEN:(e * SEQ_LEN) + 1]
            x_np = chunk[:-1].reshape(-1, SEQ_LEN)
            y_np = chunk[1:].reshape(-1, SEQ_LEN)
            x = torch.tensor(x_np, dtype=torch.long, device=device)
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            ct = float(y.numel())
            with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                bl = raw_model.loss_fn(x, y, is_causal=True).float()
            total_nll += bl.item() * ct
            prev_ids = x_np.reshape(-1)
            tgt_ids = y_np.reshape(-1)
            bytes_np = base_bytes_lut[tgt_ids].astype(np.float64)
            bytes_np += (has_leading_space_lut[tgt_ids] &
                         ~is_boundary_token_lut[prev_ids]).astype(np.float64)
            total_tok += int(ct)
            total_bytes += bytes_np.sum()
        raw_model.train()
        return compute_bpb(total_nll, total_tok, total_bytes)

    # ---- Training loop ----
    t0 = time.perf_counter()
    t_budget_start = t0
    WARMDOWN_BUDGET_SECS = 80
    best_bpb = float("inf")

    if args.train_budget_secs > 0:
        print_main(f"Time budget: {args.train_budget_secs}s "
                   f"(warmdown in last {WARMDOWN_BUDGET_SECS}s)")

    print_main("Starting training...")

    for step in range(start_step, args.steps + 1):
        is_last = (step == args.steps)

        # ---- Time budget check ----
        if args.train_budget_secs > 0 and not is_last:
            _elapsed_bgt = time.perf_counter() - t_budget_start
            if _elapsed_bgt >= args.train_budget_secs:
                is_last = True
                print_main(f"  Budget {args.train_budget_secs}s reached at step {step} "
                           f"({_elapsed_bgt:.0f}s elapsed) — triggering final eval+save")

        # ---- Validation (rank 0 only to avoid desync) ----
        if is_last or (args.val_every > 0 and step % args.val_every == 0):
            use_ema = ema_state is not None
            if use_ema and is_main():
                orig = ema_state.apply(raw_model)

            val_bpb = eval_val(val_short) if is_main() else best_bpb
            marker = " *BEST*" if val_bpb < best_bpb else ""
            best_bpb = min(best_bpb, val_bpb)
            elapsed = time.perf_counter() - t0
            tokens_seen = step * effective_batch
            ema_tag = " [EMA]" if use_ema else ""
            print_main(f"step:{step}/{args.steps} val_bpb:{val_bpb:.4f}{marker}{ema_tag} "
                       f"tokens:{tokens_seen / 1e6:.0f}M elapsed:{elapsed:.0f}s")

            if use_ema and is_main():
                ema_state.restore(raw_model, orig)

            if is_last:
                if ema_state is not None and is_main():
                    ema_state.apply(raw_model)
                break

        # ---- LR schedule ----
        lrm = lr_schedule(step, args.steps, args.warmdown)
        # Time-based warmdown override
        if args.train_budget_secs > 0:
            _elapsed_bgt = time.perf_counter() - t_budget_start
            _remaining_bgt = args.train_budget_secs - _elapsed_bgt
            if 0 < _remaining_bgt < WARMDOWN_BUDGET_SECS:
                lrm = min(lrm, _remaining_bgt / WARMDOWN_BUDGET_SECS)

        # ---- Sync parallel Muon before next backward ----
        if args.parallel_muon:
            opt.sync_muon()

        # ---- Gradient accumulation ----
        opt.zero_grad()
        train_loss_accum = 0.0

        # Retro fadeout: decide ONCE per step whether to use retro.
        # When fadeout triggers, switch to pure AR path (different shape [B,T] vs [2B,T]).
        # This WILL cause a one-time re-compile (~3-4 min) at the fadeout boundary,
        # but avoids the 1.64x throughput penalty of running merged batch with alpha=0.
        retro_fraction = getattr(args, 'retro_fadeout', 1.0)
        if args.train_budget_secs > 0:
            _frac_done = (time.perf_counter() - t_budget_start) / args.train_budget_secs
        else:
            _frac_done = step / max(args.steps, 1)
        retro_late = getattr(args, 'retro_late_start', 0.0)
        use_retro_now = (not args.no_retro) and (_frac_done < retro_fraction) and (_frac_done >= retro_late)

        for micro_step in range(args.grad_accum):
            is_last_micro = (micro_step == args.grad_accum - 1)
            x, y = train_loader.next_batch(args.microbatch_tokens, SEQ_LEN)

            with ddp_no_sync_ctx(is_last_micro):
                if use_retro_now and args.retro_mode == "merged":
                    # === MODE: merged (original) ===
                    # 50/50 merged batch — overhead 2.0x, strongest retro signal
                    x_rev = x.flip(dims=[1])
                    y_rev = y.flip(dims=[1])
                    x_both = torch.cat([x, x_rev], dim=0)       # [2B, T]
                    y_both = torch.cat([y, y_rev], dim=0)        # [2B, T]
                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        h = model(x_both)
                        h_fwd, h_bwd = h.chunk(2, dim=0)
                        w = raw_model.tok_emb.weight.to(h.dtype)
                        if args.split_logits:
                            logits_fwd = raw_model.softcap(F.linear(h_fwd.reshape(-1, MODEL_DIM), w))
                            logits_bwd = raw_model.softcap(F.linear(h_bwd.reshape(-1, MODEL_DIM), w))
                            fwd_loss = F.cross_entropy(logits_fwd.float(), y.reshape(-1))
                            bwd_loss = F.cross_entropy(logits_bwd.float(), y_rev.reshape(-1))
                        else:
                            logits_both = raw_model.softcap(F.linear(h.reshape(-1, MODEL_DIM), w))
                            per_tok_loss = F.cross_entropy(
                                logits_both.float(), y_both.reshape(-1), reduction='none')
                            BT = y.numel()
                            fwd_loss = per_tok_loss[:BT].mean()
                            bwd_loss = per_tok_loss[BT:].mean()
                        loss = (fwd_loss + RETRO_ALPHA * bwd_loss) / args.grad_accum
                    loss.backward()
                    train_loss_accum += loss.item() * args.grad_accum

                elif use_retro_now and args.retro_mode == "alternating":
                    # === MODE A: alternating ===
                    # Every N steps do reverse, rest forward (N = retro_alt_ratio)
                    # Overhead 1.0x — same speed as no retro, no shape change
                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        if step % args.retro_alt_ratio != 0:
                            # Forward step (majority)
                            loss = model(x, target_ids=y, retrodiction=False) / args.grad_accum
                        else:
                            # Reverse step (every N-th)
                            x_rev = x.flip(dims=[1])
                            y_rev = y.flip(dims=[1])
                            loss = model(x_rev, target_ids=y_rev, retrodiction=False) * RETRO_ALPHA / args.grad_accum
                    loss.backward()
                    train_loss_accum += loss.item() * args.grad_accum

                elif use_retro_now and args.retro_mode == "partial":
                    # === MODE C: partial (85/15 mixed batch) ===
                    # Only retro_partial_frac of sequences are reversed
                    # Overhead ~1.15x (batch is 1.15× bigger, not 2×)
                    B = x.shape[0]  # number of sequences in this microbatch
                    n_rev = max(int(B * args.retro_partial_frac), 1)
                    x_rev_part = x[:n_rev].flip(dims=[1])
                    y_rev_part = y[:n_rev].flip(dims=[1])
                    x_all = torch.cat([x, x_rev_part], dim=0)    # [B + n_rev, T]
                    y_all = torch.cat([y, y_rev_part], dim=0)
                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        h = model(x_all)
                        w = raw_model.tok_emb.weight.to(h.dtype)
                        logits_all = raw_model.softcap(F.linear(h.reshape(-1, MODEL_DIM), w))
                        per_tok_loss = F.cross_entropy(
                            logits_all.float(), y_all.reshape(-1), reduction='none')
                        BT_fwd = y.numel()
                        BT_bwd = y_rev_part.numel()
                        fwd_loss = per_tok_loss[:BT_fwd].mean()
                        bwd_loss = per_tok_loss[BT_fwd:].mean()
                        loss = (fwd_loss + RETRO_ALPHA * bwd_loss) / args.grad_accum
                    loss.backward()
                    train_loss_accum += loss.item() * args.grad_accum
                elif args.shared_cdm:
                    # Shared-weight: BATCH SPLITTING — half AR, half CDM per step
                    # Smooth gradient (no oscillation), ~85% of pure AR speed
                    B, T = x.shape
                    half = max(B // 2, 1)

                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        # AR half (causal)
                        ar_loss = raw_model.loss_fn(x[:half], y[:half], is_causal=True)

                        # CDM half (bidirectional + mask)
                        x_cdm = x[half:].clone()
                        B_cdm = x_cdm.shape[0]
                        mask_rate = np.random.uniform(args.cdm_mask_low, args.cdm_mask_high)
                        mask = torch.rand(B_cdm, T, device=x.device) < mask_rate
                        n_masked = mask.sum().item()
                        if n_masked > 0:
                            x_cdm[mask] = torch.randint(0, VOCAB_SIZE, (int(n_masked),), device=x.device)
                        h_cdm = raw_model.forward_hidden(x_cdm, is_causal=False)
                        logits_cdm = raw_model.softcap(
                            F.linear(h_cdm.reshape(-1, MODEL_DIM),
                                     raw_model.tok_emb.weight.to(h_cdm.dtype)))
                        per_tok_cdm = F.cross_entropy(logits_cdm.float(), x[half:].reshape(-1), reduction='none')
                        mask_flat = mask.reshape(-1).float()
                        cdm_loss = (per_tok_cdm * mask_flat).sum() / (mask_flat.sum() + 1e-8)

                        loss = (ar_loss + args.cdm_weight * cdm_loss) / args.grad_accum
                    loss.backward()
                    train_loss_accum += loss.item() * args.grad_accum
                elif args.cdm:
                    # Pure CDM: bidirectional attention + variable mask
                    B, T = x.shape
                    mask_rate = np.random.uniform(args.cdm_mask_low, args.cdm_mask_high)
                    mask = torch.rand(B, T, device=x.device) < mask_rate
                    x_masked = x.clone()
                    n_masked = mask.sum().item()
                    if n_masked > 0:
                        x_masked[mask] = torch.randint(0, VOCAB_SIZE, (int(n_masked),), device=x.device)
                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        h = raw_model.forward_hidden(x_masked, is_causal=False)
                        logits = raw_model.softcap(
                            F.linear(h.reshape(-1, MODEL_DIM),
                                     raw_model.tok_emb.weight.to(h.dtype)))
                        # Loss only on masked positions
                        per_tok = F.cross_entropy(logits.float(), x.reshape(-1), reduction='none')
                        mask_flat = mask.reshape(-1).float()
                        cdm_loss = (per_tok * mask_flat).sum() / (mask_flat.sum() + 1e-8) / args.grad_accum
                    cdm_loss.backward()
                    train_loss_accum += cdm_loss.item() * args.grad_accum
                else:
                    with torch.amp.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        fwd_loss = model(x, target_ids=y, retrodiction=False) / args.grad_accum
                    fwd_loss.backward()
                    train_loss_accum += fwd_loss.item() * args.grad_accum

        # ---- Optimizer step ----
        opt.step(raw_model, step=step, lr_mul=lrm, async_muon=args.parallel_muon)

        # ---- EMA ----
        if step == ema_start_step:
            ema_state = EMAState(raw_model)
            print_main(f"  EMA started at step {step}")
        elif ema_state is not None:
            ema_state.update(raw_model)

        # ---- Logging ----
        if step % 100 == 0 and step > 0:
            elapsed = time.perf_counter() - t0
            tps = (step - start_step) * effective_batch / elapsed
            if args.cdm:
                retro_tag = " CDM"
            elif not args.no_retro:
                retro_tag = f" retro:{'ON' if use_retro_now else 'OFF'}"
            else:
                retro_tag = ""
            print_main(f"  step:{step} train_loss:{train_loss_accum:.4f} "
                       f"lr_mul:{lrm:.4f} tok/s:{tps:.0f}{retro_tag}")

        # ---- Checkpoint ----
        if (is_main() and args.checkpoint_dir and args.checkpoint_every > 0
                and step > 0 and step % args.checkpoint_every == 0):
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
            ckpt = {
                "step": step,
                "model": raw_model.state_dict(),
                "ema": ema_state.shadow if ema_state else None,
            }
            torch.save(ckpt, ckpt_path)
            print_main(f"  Checkpoint saved: {ckpt_path}")

    # ---- Save final model ----
    if is_main():
        sd = raw_model.state_dict()

        # Save BF16 .pt (for TTT + SLOT eval, no quantization loss)
        bf16_path = args.save_path.replace(".npz", "_bf16.pt")
        torch.save(sd, bf16_path)
        bf16_size = os.path.getsize(bf16_path)
        print_main(f"\nSaved BF16 .pt to {bf16_path} ({bf16_size/1e6:.1f}MB)")

        # Save as npz
        np_weights = {}
        for k, v in sd.items():
            t = v.cpu()
            if t.dtype == torch.bfloat16:
                np_weights[k] = t.float().numpy()
            else:
                np_weights[k] = t.numpy()
        np.savez(args.save_path, **np_weights)
        print_main(f"Saved NPZ to {args.save_path}")

        # Save int6 + lzma
        nbytes = save_model_int6_lzma(sd, args.save_int6_path)
        print_main(f"Saved int6+lzma to {args.save_int6_path} ({nbytes/1e6:.1f}MB)")

        # Save mixed quantization variants (Gemma-style)
        for strat in ["aggressive", "smart"]:
            mixed_path = args.save_int6_path.replace("_int6.lzma", f"_mixed_{strat}.lzma")
            nbytes_m = save_model_mixed_quant(sd, mixed_path, strategy_name=strat)
            fits = "✅" if nbytes_m <= 16 * 1024 * 1024 else "❌"
            print_main(f"Saved mixed-{strat} to {mixed_path} ({nbytes_m/1e6:.1f}MB) {fits}")

    # ---- Summary ----
    elapsed = time.perf_counter() - t0
    total_tokens = (args.steps - start_step) * effective_batch
    print_main("=" * 70)
    print_main(f"FINAL val_bpb: {best_bpb:.4f}")
    print_main(f"Total tokens: {total_tokens / 1e9:.3f}B in {elapsed:.0f}s")
    print_main(f"Model: {NUM_LAYERS}L d={MODEL_DIM} MLP={MLP_MULT}x | {n_params:,} params")
    print_main(f"Throughput: {total_tokens / elapsed:.0f} tok/s")
    print_main("=" * 70)

    # ---- Cleanup ----
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
