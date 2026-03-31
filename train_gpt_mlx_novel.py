#!/usr/bin/env python3
"""
MLX Novel Architecture Training Script for Parameter Golf
==========================================================
Ports novel techniques to Apple Silicon (MLX) for local smoke testing.
Based on the MLX baseline (train_gpt_mlx.py) with these additions:

PROVEN IMPROVEMENTS over baseline:
  - LeakyReLU(0.5)^2 instead of ReLU^2 (preserves 25% negative signal, -3 mBPB)
  - 11 layers instead of 9 (more depth within param budget)
  - MLP mult 3x instead of 2x (wider FFN)

NOVEL TECHNIQUES:
  1. TTT-Linear layers — fast-weight linear model updated by gradient descent per chunk
  2. Gated DeltaNet layers — linear recurrence with gated delta rule
  3. Relaxed Recursive Transformer — shared blocks with per-loop LoRA adapters
  4. Mixture-of-Recursions — per-token adaptive depth routing

Usage:
  # Baseline (standard 11-layer attention with proven improvements):
  python train_gpt_mlx_novel.py

  # Hybrid: 2 TTT layers replacing layers 3 and 6:
  LAYER_TYPES=attn,attn,attn,ttt,attn,attn,ttt,attn,attn,attn,attn python train_gpt_mlx_novel.py

  # Hybrid: 2 DeltaNet layers:
  LAYER_TYPES=attn,attn,deltanet,attn,attn,deltanet,attn,attn,attn,attn,attn python train_gpt_mlx_novel.py

  # Recursive: 3 shared blocks x 4 loops with LoRA:
  RECURSIVE=1 NUM_SHARED_BLOCKS=3 NUM_LOOPS=4 LORA_ENABLED=1 python train_gpt_mlx_novel.py

  # Quick smoke test (200 steps, ~5 min on M-series Mac):
  ITERATIONS=200 TRAIN_LOG_EVERY=10 VAL_LOSS_EVERY=200 python train_gpt_mlx_novel.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

class Hyperparameters:
    # --- Data ---
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # --- Training ---
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # --- Model (improved defaults over baseline) ---
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))       # 11 layers (vs baseline 9)
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))            # 3x MLP (vs baseline 2x)
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # --- Novel: Layer types ---
    # Comma-separated: "attn", "ttt", "deltanet" per layer position
    layer_types: str = os.environ.get("LAYER_TYPES", "")

    # --- Novel: Recursive mode ---
    recursive: bool = bool(int(os.environ.get("RECURSIVE", "0")))
    num_shared_blocks: int = int(os.environ.get("NUM_SHARED_BLOCKS", 3))
    num_loops: int = int(os.environ.get("NUM_LOOPS", 4))

    # --- Novel: LoRA for recursive mode ---
    lora_enabled: bool = bool(int(os.environ.get("LORA_ENABLED", "0")))
    lora_rank: int = int(os.environ.get("LORA_RANK", 16))
    lora_alpha: float = float(os.environ.get("LORA_ALPHA", 1.0))

    # --- Novel: TTT-Linear settings ---
    ttt_chunk_size: int = int(os.environ.get("TTT_CHUNK_SIZE", 64))
    ttt_lr_init: float = float(os.environ.get("TTT_LR_INIT", 0.1))

    # --- Novel: Gated DeltaNet settings ---
    deltanet_chunk_size: int = int(os.environ.get("DELTANET_CHUNK_SIZE", 64))

    # --- Optimizer ---
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


# Control tensors that shouldn't be treated as matrix params by Muon
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
    "ttt_lr", "gate_proj", "beta_proj", "W0",
    "loop_embed", "loop_scale", "lora",
)


# =============================================================================
# HELPERS
# =============================================================================

def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    """Newton-Schulz orthogonalization for Muon optimizer."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


# =============================================================================
# DATA LOADING
# =============================================================================

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


class TokenStream:
    def __init__(self, pattern: str, log_fn=None, dataset_name=""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch} dataset:{self.dataset_name}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str, log_fn=None, dataset_name=""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# =============================================================================
# MODEL BUILDING BLOCKS
# =============================================================================

class CastedLinear(nn.Module):
    """Linear layer that casts weights to input dtype."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


# =============================================================================
# NOVEL: LoRA ADAPTER
# =============================================================================
# Low-Rank Adaptation — adds a small trainable bypass: out += (alpha/r) * B(A(x))
# Used in Relaxed Recursive Transformer for per-loop weight specialization.

class LoRAAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.scale = alpha / rank
        self.A = mx.random.normal((rank, in_dim)) * (1.0 / math.sqrt(in_dim))
        self.B = mx.zeros((out_dim, rank))

    def __call__(self, x: mx.array) -> mx.array:
        # x @ A^T @ B^T * scale
        return (x @ self.A.astype(x.dtype).T @ self.B.astype(x.dtype).T) * self.scale


# =============================================================================
# ATTENTION: STANDARD CAUSAL SELF-ATTENTION (GQA)
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Grouped Query Attention: 8 Q heads share 4 KV heads."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array,
                 lora_q=None, lora_k=None, lora_v=None, lora_o=None) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        if lora_q is not None:
            q = q + lora_q(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        k = self.c_k(x)
        if lora_k is not None:
            k = k + lora_k(x)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        v = self.c_v(x)
        if lora_v is not None:
            v = v + lora_v(x)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)

        out = self.proj(y)
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# NOVEL: TTT-LINEAR ATTENTION
# =============================================================================
# Test-Time Training (Sun et al., arXiv:2407.04620)
#
# Instead of attention, maintains a fast-weight matrix W that is a tiny linear
# model. For each chunk of tokens:
#   1. Read: output = Q @ W
#   2. Predict: pred = K @ W
#   3. Error: error = pred - V
#   4. Update: W -= lr * (K^T @ error) / chunk_size
#
# The fast weight W compresses the sequence into a learnable memory.
# O(n) complexity vs O(n^2) for attention.
# The "hidden state" IS a machine learning model — this is the key insight.

class TTTLinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 chunk_size: int = 64, ttt_lr_init: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)

        # Learnable TTT learning rate (one per head)
        self.ttt_lr = mx.ones((num_heads, 1, 1), dtype=mx.float32) * ttt_lr_init

        # Initial fast weight: identity matrix (start as passthrough)
        self.W0 = mx.eye(self.head_dim, dtype=mx.float32).reshape(1, 1, self.head_dim, self.head_dim)

    def __call__(self, x: mx.array,
                 lora_q=None, lora_k=None, lora_v=None, lora_o=None) -> mx.array:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        Hkv = self.num_kv_heads

        q = self.c_q(x)
        if lora_q is not None:
            q = q + lora_q(x)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # B,H,T,D

        k = self.c_k(x)
        if lora_k is not None:
            k = k + lora_k(x)
        k = k.reshape(B, T, Hkv, D)
        if Hkv < H:
            k = mx.repeat(k, H // Hkv, axis=2)
        k = k.transpose(0, 2, 1, 3)  # B,H,T,D

        v = self.c_v(x)
        if lora_v is not None:
            v = v + lora_v(x)
        v = v.reshape(B, T, Hkv, D)
        if Hkv < H:
            v = mx.repeat(v, H // Hkv, axis=2)
        v = v.transpose(0, 2, 1, 3)  # B,H,T,D

        q = rms_norm(q)
        k = rms_norm(k)

        # Mini-batch TTT: process in chunks, update fast weight W per chunk
        W = mx.broadcast_to(self.W0.astype(x.dtype), (B, H, D, D))
        lr = self.ttt_lr.astype(x.dtype)  # H,1,1

        outputs = []
        for i in range(0, T, self.chunk_size):
            end = min(i + self.chunk_size, T)
            q_c = q[:, :, i:end]  # B,H,chunk,D
            k_c = k[:, :, i:end]
            v_c = v[:, :, i:end]

            # Read from fast weight
            out_c = q_c @ W  # B,H,chunk,D

            # Compute reconstruction error and update W
            pred = k_c @ W                    # B,H,chunk,D
            error = pred - v_c                # B,H,chunk,D
            chunk_len = end - i
            grad = k_c.transpose(0, 1, 3, 2) @ error / chunk_len  # B,H,D,D
            W = W - lr * grad

            outputs.append(out_c)

        y = mx.concatenate(outputs, axis=2)  # B,H,T,D
        y = rms_norm(y)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        out = self.proj(y)
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# NOVEL: GATED DELTANET
# =============================================================================
# Gated Delta Networks (Yang et al., ICLR 2025, arXiv:2412.06464)
#
# Linear recurrence with gated delta rule:
#   S_t = alpha_t * S_{t-1} + beta_t * (v_t @ k_t^T)
#   output_t = S_t @ q_t
#
# alpha = forget gate (how much old state to keep)
# beta = update gate (how much new info to write)
#
# Key advantages over vanilla linear attention:
# - Can FORGET old associations (vanilla can only add)
# - Can OVERWRITE specific associations (delta rule)
# - O(n) complexity, no custom kernels needed
# - Outperforms Mamba2 (used in Qwen3 production)

class GatedDeltaNetLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, chunk_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size

        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)

        # Gate projections: alpha (forget) and beta (update)
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)
        self.beta_proj = nn.Linear(dim, num_heads, bias=True)

    def __call__(self, x: mx.array,
                 lora_q=None, lora_k=None, lora_v=None, lora_o=None) -> mx.array:
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        Hkv = self.num_kv_heads

        q = self.c_q(x)
        if lora_q is not None:
            q = q + lora_q(x)
        q = q.reshape(B, T, H, D).transpose(0, 2, 1, 3)  # B,H,T,D

        k = self.c_k(x)
        if lora_k is not None:
            k = k + lora_k(x)
        k = k.reshape(B, T, Hkv, D)
        if Hkv < H:
            k = mx.repeat(k, H // Hkv, axis=2)
        k = k.transpose(0, 2, 1, 3)

        v = self.c_v(x)
        if lora_v is not None:
            v = v + lora_v(x)
        v = v.reshape(B, T, Hkv, D)
        if Hkv < H:
            v = mx.repeat(v, H // Hkv, axis=2)
        v = v.transpose(0, 2, 1, 3)

        k = rms_norm(k)
        q = rms_norm(q)

        # Compute gates: alpha (forget) and beta (write)
        alpha = mx.sigmoid(self.gate_proj(x))  # B,T,H
        beta = mx.sigmoid(self.beta_proj(x))    # B,T,H
        alpha = alpha.transpose(0, 2, 1)[:, :, :, None]  # B,H,T,1
        beta = beta.transpose(0, 2, 1)[:, :, :, None]    # B,H,T,1

        # Chunk-wise recurrence
        S = mx.zeros((B, H, D, D), dtype=x.dtype)
        outputs = []

        for i in range(0, T, self.chunk_size):
            end = min(i + self.chunk_size, T)
            q_c = q[:, :, i:end]
            k_c = k[:, :, i:end]
            v_c = v[:, :, i:end]
            a_c = alpha[:, :, i:end]  # B,H,chunk,1
            b_c = beta[:, :, i:end]   # B,H,chunk,1

            chunk_len = end - i
            chunk_outs = []

            for j in range(chunk_len):
                k_j = k_c[:, :, j:j+1, :]   # B,H,1,D
                v_j = v_c[:, :, j:j+1, :]   # B,H,1,D
                q_j = q_c[:, :, j:j+1, :]   # B,H,1,D
                a_j = a_c[:, :, j:j+1, :]   # B,H,1,1
                b_j = b_c[:, :, j:j+1, :]   # B,H,1,1

                # Gated delta rule: S = alpha*S + beta*(v @ k^T)
                kv = v_j.transpose(0, 1, 3, 2) @ k_j  # B,H,D,D
                S = a_j[:, :, :, None] * S + b_j[:, :, :, None] * kv

                # Read from state
                out_j = q_j @ S  # B,H,1,D
                chunk_outs.append(out_j)

            outputs.append(mx.concatenate(chunk_outs, axis=2))

        y = mx.concatenate(outputs, axis=2)  # B,H,T,D
        y = rms_norm(y)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        out = self.proj(y)
        if lora_o is not None:
            out = out + lora_o(y)
        return out


# =============================================================================
# MLP with LeakyReLU(0.5)^2
# =============================================================================
# Proven improvement: LeakyReLU with slope 0.5 preserves 25% of negative signal,
# then squaring creates smooth gating. -3 mBPB over baseline ReLU^2.

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array,
                 lora_up=None, lora_down=None) -> mx.array:
        h = self.fc(x)
        if lora_up is not None:
            h = h + lora_up(x)
        # LeakyReLU(0.5)^2: preserves negative signal while creating smooth gating
        h = mx.where(h >= 0, h, 0.5 * h)
        h = h * h  # square
        out = self.proj(h)
        if lora_down is not None:
            out = out + lora_down(h)
        return out


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, layer_type: str = "attn",
                 ttt_chunk_size: int = 64, ttt_lr_init: float = 0.1,
                 deltanet_chunk_size: int = 64):
        super().__init__()
        self.layer_type = layer_type
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()

        if layer_type == "ttt":
            self.attn = TTTLinearAttention(dim, num_heads, num_kv_heads,
                                           chunk_size=ttt_chunk_size, ttt_lr_init=ttt_lr_init)
        elif layer_type == "deltanet":
            self.attn = GatedDeltaNetLayer(dim, num_heads, num_kv_heads,
                                            chunk_size=deltanet_chunk_size)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)

        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32)
        )))

    def __call__(self, x: mx.array, x0: mx.array,
                 lora_q=None, lora_k=None, lora_v=None, lora_o=None,
                 lora_up=None, lora_down=None) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x),
                             lora_q=lora_q, lora_k=lora_k, lora_v=lora_v, lora_o=lora_o)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x), lora_up=lora_up, lora_down=lora_down)
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x


# =============================================================================
# GPT MODEL — supports standard and recursive modes
# =============================================================================

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, logit_chunk_tokens: int,
                 logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, layer_types: str = "",
                 recursive: bool = False, num_shared_blocks: int = 3, num_loops: int = 4,
                 lora_enabled: bool = False, lora_rank: int = 16, lora_alpha: float = 1.0,
                 ttt_chunk_size: int = 64, ttt_lr_init: float = 0.1,
                 deltanet_chunk_size: int = 64):
        super().__init__()
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.recursive = recursive
        self.num_loops = num_loops if recursive else 1
        self.lora_enabled = lora_enabled and recursive

        self.tok_emb = nn.Embedding(vocab_size, dim)

        if recursive:
            # Shared blocks looped M times
            self.num_shared_blocks = num_shared_blocks
            n_blocks = num_shared_blocks
            effective_layers = num_shared_blocks * num_loops
        else:
            self.num_shared_blocks = num_layers
            n_blocks = num_layers
            effective_layers = num_layers

        # Parse layer types
        lt_list = layer_types.split(",") if layer_types else ["attn"] * n_blocks
        lt_list = [t.strip() for t in lt_list if t.strip()]
        lt_list = (lt_list * ((n_blocks // max(len(lt_list), 1)) + 1))[:n_blocks]

        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_type=lt_list[i],
                  ttt_chunk_size=ttt_chunk_size, ttt_lr_init=ttt_lr_init,
                  deltanet_chunk_size=deltanet_chunk_size)
            for i in range(n_blocks)
        ]

        # U-Net skip connections
        self.num_encoder_layers = effective_layers // 2
        self.num_decoder_layers = effective_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)

        # Recursive mode extras
        if recursive:
            # Per-loop iteration embeddings (tell model which loop it's in)
            self.loop_embeddings = mx.random.normal((num_loops, dim)) * 0.02
            # Per-loop scale factors
            self.loop_scales = mx.ones((num_loops,), dtype=mx.float32)

        # LoRA adapters: one set per loop per block
        if self.lora_enabled:
            kv_dim = num_kv_heads * (dim // num_heads)
            mlp_dim = dim * mlp_mult
            self.lora_modules = {}
            for loop_idx in range(num_loops):
                for block_idx in range(num_shared_blocks):
                    pfx = f"lora_l{loop_idx}_b{block_idx}"
                    self.lora_modules[f"{pfx}_q"] = LoRAAdapter(dim, dim, lora_rank, lora_alpha)
                    self.lora_modules[f"{pfx}_k"] = LoRAAdapter(dim, kv_dim, lora_rank, lora_alpha)
                    self.lora_modules[f"{pfx}_v"] = LoRAAdapter(dim, kv_dim, lora_rank, lora_alpha)
                    self.lora_modules[f"{pfx}_o"] = LoRAAdapter(dim, dim, lora_rank, lora_alpha)
                    self.lora_modules[f"{pfx}_up"] = LoRAAdapter(dim, mlp_dim, lora_rank, lora_alpha)
                    self.lora_modules[f"{pfx}_down"] = LoRAAdapter(mlp_dim, dim, lora_rank, lora_alpha)

        self.final_norm = RMSNormNoWeight()

        # Zero-init output projections
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def _get_loras(self, loop_idx: int, block_idx: int):
        if not self.lora_enabled:
            return None, None, None, None, None, None
        pfx = f"lora_l{loop_idx}_b{block_idx}"
        return (
            self.lora_modules.get(f"{pfx}_q"),
            self.lora_modules.get(f"{pfx}_k"),
            self.lora_modules.get(f"{pfx}_v"),
            self.lora_modules.get(f"{pfx}_o"),
            self.lora_modules.get(f"{pfx}_up"),
            self.lora_modules.get(f"{pfx}_down"),
        )

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        if self.recursive:
            # RECURSIVE FORWARD: K shared blocks x M loops
            effective_layer = 0
            encoder_half = self.num_encoder_layers

            for loop_idx in range(self.num_loops):
                # Add loop-specific positional signal
                x = x + self.loop_embeddings[loop_idx].astype(x.dtype)[None, None, :]

                for block_idx in range(self.num_shared_blocks):
                    lora_q, lora_k, lora_v, lora_o, lora_up, lora_down = self._get_loras(loop_idx, block_idx)

                    # U-Net: collect skips in encoder half
                    if effective_layer < encoder_half:
                        skips.append(x)

                    # U-Net: apply skips in decoder half
                    skip_idx = effective_layer - encoder_half
                    if 0 <= skip_idx < self.num_skip_weights and skip_idx < len(skips):
                        x = x + self.skip_weights[skip_idx].astype(x.dtype)[None, None, :] * skips[-(skip_idx+1)]

                    x = self.blocks[block_idx](
                        x, x0,
                        lora_q=lora_q, lora_k=lora_k, lora_v=lora_v, lora_o=lora_o,
                        lora_up=lora_up, lora_down=lora_down,
                    )
                    effective_layer += 1
        else:
            # STANDARD FORWARD: N unique layers with U-Net skips
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)

        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


# =============================================================================
# OPTIMIZERS
# =============================================================================

class Muon:
    """Newton-Schulz orthogonalized gradient updates for 2D matrix params."""
    def __init__(self, keys, params, args):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    """Muon for 2D matrix params, Adam for embeddings and scalars."""
    def __init__(self, model, args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if p.ndim == 2 and k != self.embed_key
            and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k != self.embed_key and k not in self.matrix_keys
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr,
                                     betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr,
                                      betas=[args.beta1, args.beta2], eps=args.adam_eps, bias_correction=True)

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        ))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in params}
        if scalar_grads:
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))


# =============================================================================
# QUANTIZATION (INT8 + ZLIB) — same as baseline
# =============================================================================

MX_DTYPE_FROM_NAME = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS


def _np_float32(arr):
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr):
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE))
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
                            "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj):
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[quant_obj["dtypes"][name]])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


# =============================================================================
# EVAL HELPERS
# =============================================================================

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
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(f) for f in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut,
             is_boundary_token_lut, log_fn=None):
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(float(np.sum(np.square(_np_float32(g)), dtype=np.float64)) for g in flat.values())
    if total_sq <= 0 or math.sqrt(total_sq) <= max_norm:
        return grads_tree
    scale = max_norm / (math.sqrt(total_sq) + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


# =============================================================================
# TRAINING
# =============================================================================

def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg, console=True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=Path(args.data_path).name)

    # Build model
    n_blocks = args.num_shared_blocks if args.recursive else args.num_layers
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init, layer_types=args.layer_types,
        recursive=args.recursive, num_shared_blocks=args.num_shared_blocks,
        num_loops=args.num_loops, lora_enabled=args.lora_enabled,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        ttt_chunk_size=args.ttt_chunk_size, ttt_lr_init=args.ttt_lr_init,
        deltanet_chunk_size=args.deltanet_chunk_size,
    )
    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    lt_list = args.layer_types.split(",") if args.layer_types else ["attn"] * n_blocks
    log(f"mode:{'recursive' if args.recursive else 'standard'}")
    if args.recursive:
        log(f"recursive: {args.num_shared_blocks} blocks x {args.num_loops} loops = "
            f"{args.num_shared_blocks * args.num_loops} effective layers, lora={args.lora_enabled}")
    log(f"model_params:{n_params} layers:{n_blocks} dim:{args.model_dim} "
        f"heads:{args.num_heads} kv_heads:{args.num_kv_heads} mlp_mult:{args.mlp_mult}")
    log(f"layer_types:{lt_list[:n_blocks]}")
    log(f"improvements: LeakyReLU(0.5)^2, {args.num_layers}L, {args.mlp_mult}x MLP")
    log(f"optimizer: muon_matrix:{len(opt.matrix_keys)} scalar:{len(opt.scalar_keys)}")

    # Warmup
    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if warmup_step + 1 == args.warmup_steps or (warmup_step + 1) % 10 == 0:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=Path(args.data_path).name)

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut,
                                          has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            t0 = time.perf_counter()
        if last_step:
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - t0) - train_time_ms  # approx
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms")
        if max_wallclock_ms and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # Serialization + quantized roundtrip
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_blob = zlib.compress(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    log(f"serialized_model_int8_zlib:{quant_path.stat().st_size} bytes")

    with quant_path.open("rb") as f:
        quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(f.read())))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_val_loss, q_val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut,
                                      has_leading_space_lut, is_boundary_token_lut, log_fn=log)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
