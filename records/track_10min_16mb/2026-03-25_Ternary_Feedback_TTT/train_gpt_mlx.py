#!/usr/bin/env python3
"""
Ternary Reasoner — MLX local testing version.

This is the Apple Silicon (MLX) mirror of train_gpt.py for local development
and smoke-testing on Mac. It implements the same Ternary Reasoner architecture:
  - Ternary weights {-1, 0, +1} with STE
  - U-Net encoder-decoder with skip connections
  - Iterative Hadamard-gated backward correction
  - Recurrent capsule bank
  - XSA, VRL, Partial RoPE, BigramHash, LeakyReLU², LN Scale Damping
  - EMA weight averaging
  - Ternary packing (base-3) + LZMA compression

NOT intended as a competition submission (that's the PyTorch version).
This is for rapid local iteration before burning cloud GPU time.

Author: Aki Gogikar (OneNewAI)
"""
from __future__ import annotations

import copy
import glob
import json
import lzma
import math
import os
import pickle
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

# MLX < 0.22 on CPU only supports float32 matmul; use bfloat16 on Apple Silicon GPU
_mlx_ver = tuple(int(x) for x in mx.__version__.split(".")[:2])
COMPUTE_DTYPE = mx.bfloat16 if _mlx_ver >= (0, 22) else mx.float32

# ---------------------------------------------------------------------------
# Env-var helper
# ---------------------------------------------------------------------------
def _e(name, default, typ=str):
    v = os.environ.get(name)
    if v is None:
        return default
    if typ is bool:
        return v not in ("0", "false", "False", "")
    return typ(v)

# ---------------------------------------------------------------------------
# Hyperparameters — same env vars as the PyTorch version
# ---------------------------------------------------------------------------
class Hyperparameters:
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", f"mlx_{uuid.uuid4().hex[:8]}")
    seed = _e("SEED", 1337, int)
    out_dir = _e("OUT_DIR", "logs")

    # Training loop
    iterations = _e("ITERATIONS", 10000, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 0, int)
    val_batch_size = _e("VAL_BATCH_SIZE", 65536, int)
    max_val_tokens = _e("MAX_VAL_TOKENS", 0, int)  # 0=no limit (use all val data)
    train_log_every = _e("TRAIN_LOG_EVERY", 100, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 32768, int)
    grad_accum_steps = _e("GRAD_ACCUM_STEPS", 4, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 1024, int)
    warmup_steps = _e("WARMUP_STEPS", 5, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.5, float)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 600.0, float)
    mlx_max_microbatch_tokens = _e("MLX_MAX_MICROBATCH_TOKENS", 8192, int)
    mlx_eager_eval = _e("MLX_EAGER_EVAL", 1, bool)

    # Model
    vocab_size = _e("VOCAB_SIZE", 8192, int)
    num_layers = _e("NUM_LAYERS", 12, int)
    model_dim = _e("MODEL_DIM", 768, int)
    num_heads = _e("NUM_HEADS", 8, int)
    num_kv_heads = _e("NUM_KV_HEADS", 4, int)
    mlp_mult = _e("MLP_MULT", 4, int)
    embed_dim = _e("EMBED_DIM", 254, int)
    tie_embeddings = _e("TIE_EMBEDDINGS", 1, bool)
    tied_embed_init_std = _e("TIED_EMBED_INIT_STD", 0.005, float)
    logit_softcap = _e("LOGIT_SOFTCAP", 30.0, float)
    rope_base = _e("ROPE_BASE", 5000.0, float)
    qk_gain_init = _e("QK_GAIN_INIT", 2.25, float)
    activation_type = _e("ACTIVATION", "lrelu2")
    leaky_relu_slope = _e("LEAKY_RELU_SLOPE", 0.5, float)
    bitnet_group_size = _e("BITNET_GROUP_SIZE", 128, int)

    # Feedback
    feedback_enabled = _e("FEEDBACK_ENABLED", 1, bool)
    feedback_dim = _e("FEEDBACK_DIM", 64, int)
    feedback_sketch_tokens = _e("FEEDBACK_SKETCH_TOKENS", 4, int)
    feedback_passes = _e("FEEDBACK_PASSES", 1, int)
    eval_feedback_passes = _e("EVAL_FEEDBACK_PASSES", 2, int)

    # Capsule
    capsule_enabled = _e("CAPSULE_ENABLED", 1, bool)
    capsule_num = _e("CAPSULE_NUM", 16, int)
    capsule_dim = _e("CAPSULE_DIM", 64, int)

    # Koopman dynamics in capsule space
    koopman_enabled = _e("KOOPMAN_ENABLED", 1, bool)
    koopman_rank = _e("KOOPMAN_RANK", 4, int)
    koopman_diag_init = _e("KOOPMAN_DIAG_INIT", 0.9, float)  # critical damping
    koopman_consistency_weight = _e("KOOPMAN_CONSISTENCY_WEIGHT", 0.005, float)

    # Adaptive halting (eval only)
    adaptive_halt_enabled = _e("ADAPTIVE_HALT_ENABLED", 1, bool)
    adaptive_halt_threshold = _e("ADAPTIVE_HALT_THRESHOLD", 0.05, float)
    max_eval_passes = _e("MAX_EVAL_PASSES", 3, int)

    # Cross-window capsule carry (eval only)
    capsule_carry_enabled = _e("CAPSULE_CARRY_ENABLED", 1, bool)
    capsule_carry_decay = _e("CAPSULE_CARRY_DECAY", 0.8, float)

    # Features
    bigram_hash_enabled = _e("BIGRAM_HASH_ENABLED", 1, bool)
    bigram_hash_buckets = _e("BIGRAM_HASH_BUCKETS", 4096, int)
    bigram_hash_dim = _e("BIGRAM_HASH_DIM", 128, int)
    engram_num_heads = _e("ENGRAM_NUM_HEADS", 4, int)
    engram_num_orders = _e("ENGRAM_NUM_ORDERS", 2, int)  # 2 = bigram + trigram
    engram_inject_layer = _e("ENGRAM_INJECT_LAYER", 1, int)  # -1 = input only
    vrl_enabled = _e("VRL_ENABLED", 1, bool)
    vrl_start_layer = _e("VRL_START_LAYER", 10, int)
    ln_scale_damping = _e("LN_SCALE_DAMPING", 1, bool)
    partial_rope_dims = _e("PARTIAL_ROPE_DIMS", 16, int)
    xsa_start_layer = _e("XSA_START_LAYER", 8, int)
    ema_enabled = _e("EMA_ENABLED", 1, bool)
    ema_decay = _e("EMA_DECAY", 0.997, float)
    ema_start_fraction = _e("EMA_START_FRACTION", 0.5, float)
    gptq_lite_enabled = _e("GPTQ_LITE_ENABLED", 1, bool)
    gptq_lite_percentiles = _e("GPTQ_LITE_PERCENTILES", 5, int)
    turbo_quant_export = _e("TURBO_QUANT_EXPORT", 1, bool)  # Hadamard rotation at export for lower MSE
    turbo_quant_train = _e("TURBO_QUANT_TRAIN", 0, bool)   # Hadamard rotation during training STE (adds overhead)

    # Optimizer
    matrix_lr = _e("MATRIX_LR", 0.025, float)
    scalar_lr = _e("SCALAR_LR", 0.025, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.035, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_momentum_warmup_start = _e("MUON_MOMENTUM_WARMUP_START", 0.85, float)
    muon_momentum_warmup_steps = _e("MUON_MOMENTUM_WARMUP_STEPS", 1500, int)
    beta1 = _e("BETA1", 0.9, float)
    beta2 = _e("BETA2", 0.95, float)
    adam_eps = _e("ADAM_EPS", 1e-8, float)
    grad_clip_norm = _e("GRAD_CLIP_NORM", 0.3, float)

    @property
    def train_files(self):
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self):
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self):
        return self.train_batch_tokens // self.grad_accum_steps

# CTP: parameter names routed to scalar Adam (not Muon)
CTP = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weights",
    "vocab_bias", "add_gate", "mul_gate", "recurrent_gate", "vrl_alpha", "gate",
    "koopman",  # Koopman dynamics params go to scalar Adam for stability
)

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def rms_norm(x, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)

def zeropower_newtonschulz5(g, steps, eps=1e-7):
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

# ---------------------------------------------------------------------------
# TurboQuant-inspired Hadamard rotation for ternary quantization
# ---------------------------------------------------------------------------
_HADAMARD_CACHE = {}

def _build_hadamard(n):
    """Build normalized orthogonal Hadamard matrix H_n (H @ H^T = I).
    TurboQuant (Zandieh et al. 2025): random rotation before scalar quantization
    achieves near-optimal MSE. Hadamard is a deterministic rotation that distributes
    outliers across all coordinates, reducing quantization distortion at zero param cost.
    """
    if n in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[n]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    if n == 1:
        h = mx.array([[1.0]], dtype=mx.float32)
    else:
        h_half = _build_hadamard(n // 2)
        # Unnormalized: top = [H, H], bot = [H, -H]
        top = mx.concatenate([h_half, h_half], axis=1)
        bot = mx.concatenate([h_half, -h_half], axis=1)
        h = mx.concatenate([top, bot], axis=0)
    # Normalize so H @ H^T = I
    h = h / math.sqrt(n)
    _HADAMARD_CACHE[n] = h
    return h

# ---------------------------------------------------------------------------
# Ternary STE quantization (for training — MLX version)
# ---------------------------------------------------------------------------
def ternary_ste(w, group_size=128, turbo=False):
    """Ternary quantization with Straight-Through Estimator.

    When turbo=True, applies Hadamard rotation before quantization (TurboQuant).
    This provably reduces MSE by distributing weight outliers across coordinates.
    The Hadamard matrix is deterministic and self-inverse, so:
    - No extra parameters to store
    - Dequant = quantize in rotated space, then inverse-rotate back
    """
    shape = w.shape
    w_f = w.astype(mx.float32)
    # Pad if needed
    flat = w_f.reshape(-1)
    pad_len = (group_size - flat.shape[0] % group_size) % group_size
    if pad_len > 0:
        flat = mx.concatenate([flat, mx.zeros((pad_len,), dtype=mx.float32)])
    grouped = flat.reshape(-1, group_size)

    if turbo and (group_size & (group_size - 1)) == 0:
        # TurboQuant: rotate before quantization
        H = _build_hadamard(group_size).astype(grouped.dtype)
        grouped = grouped @ H  # Hadamard rotation

    scale = mx.mean(mx.abs(grouped), axis=-1, keepdims=True)
    scale = mx.maximum(scale, mx.array(1e-8))
    normalized = grouped / scale
    # Quantize to {-1, 0, +1} — round and clamp
    q = mx.clip(mx.round(normalized), -1, 1)
    dequantized = q * scale

    if turbo and (group_size & (group_size - 1)) == 0:
        # Inverse rotation (H is self-inverse for normalized Hadamard)
        dequantized = dequantized @ H

    # STE: forward uses quantized, backward uses original
    w_ternary = dequantized.reshape(-1)
    if pad_len > 0:
        w_ternary = w_ternary[:w_f.size]
    w_ternary = w_ternary.reshape(shape)
    # STE trick: w + stop_gradient(w_ternary - w) = w_ternary in forward, grad flows to w
    return w + mx.stop_gradient(w_ternary - w)

# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------
# Module-level flag set by GPT.__init__ from args.turbo_quant_train
_TURBO_QUANT = False

class TernaryLinear(nn.Module):
    """Linear layer with ternary STE quantization during forward."""
    def __init__(self, in_dim, out_dim, group_size=128):
        super().__init__()
        self.weight = mx.random.normal((out_dim, in_dim), dtype=mx.float32) * 0.02
        self.group_size = group_size
        self._zero_init = False

    def __call__(self, x):
        w = ternary_ste(self.weight, self.group_size, turbo=_TURBO_QUANT)
        return x @ w.astype(x.dtype).T


class NormedTernaryLinear(TernaryLinear):
    """Ternary linear with RMSNorm on input."""
    def __call__(self, x):
        return super().__call__(rms_norm(x))


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = mx.random.normal((num_embeddings, embedding_dim), dtype=mx.float32) * 0.02

    def __call__(self, ids):
        return self.weight[ids]


class EmbedProj(nn.Module):
    """Simple linear projection (no ternary) for embedding dimension adapter."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = mx.random.normal((out_dim, in_dim), dtype=mx.float32) * 0.02

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class EngramHash(nn.Module):
    """Engram-inspired multi-head, multi-order n-gram memory with context gating.

    Key ideas from DeepSeek Engram paper:
    1. Multi-head hashing: K heads per n-gram order reduce collision rate
    2. Context-aware gating: sigmoid gate from hidden state suppresses noisy lookups
    3. Multi-order: bigrams + trigrams capture different context scales
    4. Injection at internal layers (not just input) for richer context gating

    Per-head dim = hash_dim // (num_orders * num_heads).
    Total table params = num_orders * num_heads * buckets_per_head * per_head_dim.
    """
    # Distinct large primes for each (order, head) hash function
    _PRIMES = [92821, 131071, 174763, 216091, 262147, 314159, 393241, 462841,
               524287, 611953, 700001, 786433, 873781, 967229, 1048573, 1153381]

    def __init__(self, num_buckets, hash_dim, model_dim, group_size=128,
                 num_heads=4, num_orders=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_orders = num_orders  # 1=bigram only, 2=bigram+trigram
        self.head_dim = hash_dim // (num_orders * num_heads)
        assert self.head_dim > 0, f"hash_dim={hash_dim} too small for {num_orders}×{num_heads} heads"
        self.buckets_per_head = num_buckets

        # Embedding tables: one per (order, head)
        self.tables = []
        for _ in range(num_orders * num_heads):
            self.tables.append(
                mx.random.normal((num_buckets, self.head_dim), dtype=mx.float32) * 0.02
            )

        # Projection from concatenated hash_dim to model_dim
        self.proj = TernaryLinear(hash_dim, model_dim, group_size=group_size)

        # Context-aware gating (Engram Section 2.3)
        # Gate key projection: retrieved memory -> gate space
        self.gate_k = EmbedProj(hash_dim, model_dim)
        # Gate is: sigmoid(RMSNorm(hidden) · RMSNorm(gate_k(memory)) / sqrt(d))
        self.gate_scale = model_dim ** -0.5

    def _hash_ngram(self, input_ids, order, head_idx):
        """Compute hash indices for n-gram of given order using head-specific primes."""
        B, T = input_ids.shape
        prime_idx = order * self.num_heads + head_idx
        p = self._PRIMES[prime_idx % len(self._PRIMES)]

        if order == 0:  # bigram: (t-1, t)
            prev = input_ids[:, :-1]
            curr = input_ids[:, 1:]
            h = (prev.astype(mx.int64) * p + curr.astype(mx.int64)) % self.buckets_per_head
            h = mx.concatenate([mx.zeros((B, 1), dtype=mx.int32), h.astype(mx.int32)], axis=1)
        elif order == 1:  # trigram: (t-2, t-1, t)
            pp = input_ids[:, :-2]
            prev = input_ids[:, 1:-1]
            curr = input_ids[:, 2:]
            h = (pp.astype(mx.int64) * (p * p) + prev.astype(mx.int64) * p
                 + curr.astype(mx.int64)) % self.buckets_per_head
            h = mx.concatenate([mx.zeros((B, 2), dtype=mx.int32), h.astype(mx.int32)], axis=1)
        else:
            raise ValueError(f"Unsupported n-gram order {order+2}")
        return h

    def retrieve(self, input_ids):
        """Retrieve and concatenate multi-head, multi-order n-gram embeddings."""
        parts = []
        for order in range(self.num_orders):
            for head in range(self.num_heads):
                idx = self._hash_ngram(input_ids, order, head)
                table_idx = order * self.num_heads + head
                parts.append(self.tables[table_idx][idx])  # (B, T, head_dim)
        return mx.concatenate(parts, axis=-1)  # (B, T, hash_dim)

    def __call__(self, input_ids, hidden=None):
        """Retrieve n-gram memory, optionally gated by hidden state.

        Args:
            input_ids: (B, T) token IDs
            hidden: (B, T, model_dim) hidden state for context gating. If None, ungated.
        Returns:
            (B, T, model_dim) memory injection signal
        """
        memory = self.retrieve(input_ids)  # (B, T, hash_dim)

        if hidden is not None:
            # Context-aware gating (Engram paper Eq. 3-4)
            k = self.gate_k(memory)  # (B, T, model_dim)
            # Normalized dot-product gate
            h_norm = rms_norm(hidden)
            k_norm = rms_norm(k)
            gate = mx.sigmoid(mx.sum(h_norm * k_norm, axis=-1, keepdims=True) * self.gate_scale)
            projected = self.proj(memory)  # (B, T, model_dim)
            return gate * projected
        else:
            # Ungated (input layer, no hidden state yet)
            return self.proj(memory)


class FeedbackPooler(nn.Module):
    """Compress decoder output into a low-dim semantic sketch."""
    def __init__(self, model_dim, feedback_dim, num_tokens):
        super().__init__()
        self.num_tokens = max(1, num_tokens)
        self.proj = EmbedProj(model_dim, feedback_dim)

    def __call__(self, x):
        # x: (B, T, D) -> pool to (B, num_tokens, D) -> project
        B, T, D = x.shape
        # Simple chunked mean pooling
        chunk_size = max(1, T // self.num_tokens)
        pooled_list = []
        for i in range(self.num_tokens):
            start = i * chunk_size
            end = min(start + chunk_size, T)
            if start >= T:
                pooled_list.append(mx.zeros((B, 1, D), dtype=x.dtype))
            else:
                pooled_list.append(mx.mean(x[:, start:end, :], axis=1, keepdims=True))
        pooled = mx.concatenate(pooled_list, axis=1)  # (B, num_tokens, D)
        return self.proj(rms_norm(pooled))


class FeedbackAdapter(nn.Module):
    """Hadamard-gated backward semantic correction.

    Two channels:
      - additive: x += gate_a * proj_a(sketch)
      - multiplicative: x *= 1 + gate_m * tanh(proj_m(sketch))
    Both gates are zero-initialized -> identity at init.
    """
    def __init__(self, model_dim, feedback_dim):
        super().__init__()
        self.read = EmbedProj(feedback_dim, model_dim * 2)
        self.add_gate = mx.zeros((model_dim,), dtype=mx.float32)
        self.mul_gate = mx.zeros((model_dim,), dtype=mx.float32)

    def __call__(self, x, sketch):
        if sketch is None:
            return x
        context = mx.mean(sketch, axis=1)  # (B, feedback_dim)
        projected = self.read(context)  # (B, model_dim * 2)
        projected = mx.expand_dims(projected, axis=1)  # (B, 1, model_dim * 2)
        add_term = projected[:, :, :x.shape[-1]]
        mul_term = projected[:, :, x.shape[-1]:]
        gate_a = mx.tanh(self.add_gate).astype(x.dtype)
        gate_m = mx.tanh(self.mul_gate).astype(x.dtype)
        return x * (1.0 + gate_m * mx.tanh(mul_term)) + gate_a * add_term


class KoopmanDynamics(nn.Module):
    """Diagonal + low-rank stable linear dynamics in capsule space.

    Predicts next-pass capsule state from current state:
        c_pred = D ⊙ c + U(V^T c)
        c_new  = α ⊙ c_observed + (1-α) ⊙ c_pred

    First-principles design:
        - D initialized at 0.9 (critical damping, ρ(D)=0.9 < 1)
        - UV initialized small (spectral perturbation << 1-ρ(D))
        - α at sigmoid(0)=0.5 (maximum-entropy prior between prediction and observation)
        - Stability guaranteed at init: ρ(D + UV^T) ≤ 0.9 + ε
    """
    def __init__(self, capsule_dim, rank=4, diag_init=0.9):
        super().__init__()
        self.diag = mx.full((capsule_dim,), diag_init, dtype=mx.float32)
        init_scale = 0.01 / max(rank ** 0.5, 1.0)
        self.U = mx.random.normal((capsule_dim, rank), dtype=mx.float32) * init_scale
        self.V = mx.random.normal((capsule_dim, rank), dtype=mx.float32) * init_scale
        self.alpha = mx.zeros((capsule_dim,), dtype=mx.float32)  # sigmoid -> 0.5

    def predict(self, c):
        """Predict next-pass capsule state. c: (B, N, capsule_dim)"""
        # Diagonal evolution
        c_diag = self.diag.astype(c.dtype) * c  # (B, N, D)
        # Low-rank coupling: U @ (V^T @ c^T)^T = (c @ V) @ U^T
        c_lowrank = (c @ self.V.astype(c.dtype)) @ self.U.astype(c.dtype).T  # (B, N, D)
        return c_diag + c_lowrank

    def blend(self, c_observed, c_prev):
        """Blend observed capsules with predicted evolution from previous state."""
        c_pred = self.predict(c_prev)
        alpha = mx.sigmoid(self.alpha).astype(c_observed.dtype)
        return alpha * c_observed + (1.0 - alpha) * c_pred, c_pred


class CapsuleBank(nn.Module):
    """Structured semantic state carriers with Koopman-driven recurrent dynamics.

    Upgrade from simple gated blending to predictive latent dynamics:
    - Koopman module predicts where capsule state should evolve
    - Blend prediction with fresh observation from current pass
    - Returns c_pred for consistency loss (auxiliary training signal)
    """
    def __init__(self, model_dim, capsule_num, capsule_dim,
                 koopman_enabled=True, koopman_rank=4, koopman_diag_init=0.9):
        super().__init__()
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.prototypes = mx.random.normal((capsule_num, capsule_dim), dtype=mx.float32) * 0.02
        self.read_proj = EmbedProj(model_dim, capsule_dim)
        self.write_proj = EmbedProj(capsule_dim, model_dim)
        self.recurrent_gate = mx.zeros((capsule_dim,), dtype=mx.float32)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)
        # Koopman dynamics (first-principles innovation)
        self.koopman = None
        if koopman_enabled:
            self.koopman = KoopmanDynamics(capsule_dim, rank=koopman_rank,
                                           diag_init=koopman_diag_init)

    def __call__(self, x, prev_capsules=None):
        """Returns (corrected_x, capsule_state, c_pred_for_loss).
        c_pred is None on first pass or when Koopman is disabled."""
        B, T, D = x.shape
        x_proj = self.read_proj(rms_norm(x))  # (B, T, capsule_dim)
        # Soft-assignment to prototypes
        scores = mx.einsum("btd,nd->btn", x_proj, self.prototypes.astype(x_proj.dtype))
        attn = mx.softmax(scores / (self.capsule_dim ** 0.5), axis=1)  # (B, T, N)
        capsules = mx.einsum("btn,btd->bnd", attn, x_proj)  # (B, N, capsule_dim)

        c_pred = None  # For consistency loss
        if prev_capsules is not None:
            if self.koopman is not None:
                # Koopman-driven update: predict + blend
                capsules, c_pred = self.koopman.blend(capsules, prev_capsules)
            else:
                # Fallback: simple gated blending
                rg = mx.sigmoid(self.recurrent_gate).astype(capsules.dtype)
                capsules = rg * capsules + (1.0 - rg) * prev_capsules

        # Write back
        readout = mx.einsum("btn,bnd->btd", attn, capsules)
        correction = self.write_proj(readout)
        g = mx.tanh(self.gate).astype(x.dtype)
        return x + g * correction, capsules, c_pred


def apply_rotary_emb(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], axis=-1)


def build_rope_cache(seq_len, dim, base=10000.0, dtype=mx.float32):
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(seq_len, dtype=mx.float32)
    freqs = mx.outer(t, inv_freq)
    cos = mx.cos(freqs).astype(dtype)
    sin = mx.sin(freqs).astype(dtype)
    return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 group_size=128, partial_rope_dims=0, vrl_enabled=False, xsa=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.rope_dims = partial_rope_dims if partial_rope_dims > 0 else self.head_dim
        self.vrl_enabled = vrl_enabled
        self.xsa = xsa
        self.rope_base = rope_base

        self.c_qkv = TernaryLinear(dim, self.q_size + 2 * self.kv_size, group_size=group_size)
        self.proj = NormedTernaryLinear(dim, dim, group_size=group_size)
        self.proj._zero_init = True
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.q_gain = mx.full((num_heads,), qk_gain_init, dtype=mx.float32)
        if vrl_enabled:
            self.vrl_alpha = mx.array(0.5, dtype=mx.float32)

    def __call__(self, x, v0=None):
        B, T, D = x.shape
        qkv = self.c_qkv(x)
        q = qkv[:, :, :self.q_size].reshape(B, T, self.num_heads, self.head_dim)
        k = qkv[:, :, self.q_size:self.q_size + self.kv_size].reshape(B, T, self.num_kv_heads, self.head_dim)
        v = qkv[:, :, self.q_size + self.kv_size:].reshape(B, T, self.num_kv_heads, self.head_dim)

        q = rms_norm(q)
        k = rms_norm(k)

        # RoPE (partial or full)
        cos, sin = build_rope_cache(T, self.rope_dims, self.rope_base, dtype=q.dtype)
        # cos, sin: (T, rope_dims//2) -> need (1, T, 1, rope_dims//2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        if self.rope_dims < self.head_dim:
            q_rot = apply_rotary_emb(q[..., :self.rope_dims], cos, sin)
            k_rot = apply_rotary_emb(k[..., :self.rope_dims], cos, sin)
            q = mx.concatenate([q_rot, q[..., self.rope_dims:]], axis=-1)
            k = mx.concatenate([k_rot, k[..., self.rope_dims:]], axis=-1)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        q = q * self.q_gain.astype(q.dtype)[None, None, :, None]

        # VRL: blend current values with first-layer values
        if self.vrl_enabled and v0 is not None:
            alpha = mx.sigmoid(self.vrl_alpha).astype(v.dtype)
            v = alpha * v + (1.0 - alpha) * v0

        # Transpose for attention: (B, H, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)

        scale = self.head_dim ** -0.5
        # Build causal mask explicitly (MLX < 0.22 doesn't support mask="causal")
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(q.dtype)
        y = mx.fast.scaled_dot_product_attention(q, k, v_t, scale=scale, mask=causal_mask)

        # XSA: subtract self-value
        if self.xsa:
            kv_rep = self.num_heads // self.num_kv_heads
            if kv_rep > 1:
                v_expanded = mx.repeat(v_t, kv_rep, axis=1)
            else:
                v_expanded = v_t
            y = y - v_expanded

        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(y), v


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult, group_size=128, activation="lrelu2", leaky_relu_slope=0.5):
        super().__init__()
        hidden = dim * mlp_mult
        self.activation = activation
        self.leaky_relu_slope = leaky_relu_slope
        self.fc = TernaryLinear(dim, hidden, group_size=group_size)
        self.proj = NormedTernaryLinear(hidden, dim, group_size=group_size)
        self.proj._zero_init = True
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, x):
        h = self.fc(x)
        if self.activation == "lrelu2":
            h = mx.where(h >= 0, h, h * self.leaky_relu_slope)
            h = h * h
        elif self.activation == "relu2":
            h = mx.maximum(h, 0)
            h = h * h
        else:
            h = mx.maximum(h, 0)
        return self.proj(h)


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 group_size=128, activation="lrelu2", leaky_relu_slope=0.5,
                 partial_rope_dims=0, vrl_enabled=False, ln_scale_factor=1.0, xsa=False):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        group_size=group_size, partial_rope_dims=partial_rope_dims,
                                        vrl_enabled=vrl_enabled, xsa=xsa)
        self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation,
                       leaky_relu_slope=leaky_relu_slope)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack([np.ones(dim, dtype=np.float32),
                                            np.zeros(dim, dtype=np.float32)]))

    def __call__(self, x, x0, v0=None):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = rms_norm(x) * self.ln_scale_factor
        attn_out, v_out = self.attn(normed, v0=v0)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(rms_norm(x) * self.ln_scale_factor)
        return x, v_out


class GPT(nn.Module):
    """Ternary Reasoner — MLX version."""
    def __init__(self, args: Hyperparameters):
        super().__init__()
        global _TURBO_QUANT
        _TURBO_QUANT = args.turbo_quant_train
        dim = args.model_dim
        self.args = args
        self.feedback_enabled = args.feedback_enabled
        self.feedback_passes = args.feedback_passes
        self._train_feedback_passes = args.feedback_passes

        # Embedding
        self.tok_emb = Embedding(args.vocab_size, args.embed_dim if args.embed_dim > 0 else dim)
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * args.tied_embed_init_std
        self.embed_proj = EmbedProj(args.embed_dim, dim) if args.embed_dim > 0 and args.embed_dim != dim else None
        self.embed_proj_rev = EmbedProj(dim, args.embed_dim) if args.embed_dim > 0 and args.embed_dim != dim else None

        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.vocab_bias = mx.zeros((args.vocab_size,), dtype=mx.float32)
        self.logit_softcap = args.logit_softcap

        # Blocks
        self.blocks = []
        for i in range(args.num_layers):
            layer_vrl = args.vrl_enabled and i >= args.vrl_start_layer
            ln_sf = 1.0 / (i + 1) ** 0.5 if args.ln_scale_damping else 1.0
            layer_xsa = args.xsa_start_layer >= 0 and i >= args.xsa_start_layer
            self.blocks.append(Block(
                dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                args.rope_base, args.qk_gain_init, group_size=args.bitnet_group_size,
                activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
                partial_rope_dims=args.partial_rope_dims, vrl_enabled=layer_vrl,
                ln_scale_factor=ln_sf, xsa=layer_xsa,
            ))

        # Engram-style multi-head n-gram memory (replaces simple BigramHash)
        self.engram = None
        self.engram_inject_layer = args.engram_inject_layer
        if args.bigram_hash_enabled:
            self.engram = EngramHash(
                args.bigram_hash_buckets, args.bigram_hash_dim, dim,
                group_size=args.bitnet_group_size,
                num_heads=args.engram_num_heads,
                num_orders=args.engram_num_orders,
            )

        # Capsule bank
        self.capsule_bank = None
        if args.capsule_enabled:
            self.capsule_bank = CapsuleBank(
                dim, args.capsule_num, args.capsule_dim,
                koopman_enabled=args.koopman_enabled,
                koopman_rank=args.koopman_rank,
                koopman_diag_init=args.koopman_diag_init,
            )

        # Feedback
        self.feedback_pooler = None
        self.feedback_adapters = None
        if self.feedback_enabled:
            self.feedback_pooler = FeedbackPooler(dim, args.feedback_dim, args.feedback_sketch_tokens)
            self.feedback_adapters = [
                FeedbackAdapter(dim, args.feedback_dim)
                for _ in range(self.num_decoder_layers)
            ]

    def set_feedback_passes(self, num_passes):
        """Switch feedback pass count (for train vs eval)."""
        self.feedback_passes = num_passes

    def softcap(self, logits):
        c = self.logit_softcap
        x_sc = mx.clip(logits / c, -2.0, 2.0)
        x2 = x_sc * x_sc
        return c * mx.clip(x_sc * (1.0 - x2 / 3.0 + x2 * x2 / 15.0), -1.0, 1.0)

    def _apply_embedding(self, input_ids):
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        if self.engram is not None and self.engram_inject_layer < 0:
            # Input-only injection (ungated, no hidden state yet)
            x = x + self.engram(input_ids, hidden=None).astype(x.dtype)
        x = rms_norm(x)
        return x, x

    def _decoder_pass(self, x, x0, skips, sketch, v0):
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if i < self.num_skip_weights:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips[-(i + 1)]
            x, _ = self.blocks[bi](x, x0, v0=v0)
            if self.feedback_adapters is not None and sketch is not None:
                x = self.feedback_adapters[i](x, sketch)
        return x

    def __call__(self, input_ids, carry_capsules=None):
        """Core KoopCaps-HRM forward: encode → [correct]^N → decode.

        Returns: (hidden, capsule_state, consistency_losses)
        - hidden: final hidden states for logit projection
        - capsule_state: for cross-window carry
        - consistency_losses: list of (c_pred, c_actual) for aux loss
        """
        x, x0 = self._apply_embedding(input_ids)
        skips = []
        v0 = None

        # Encoder pass (runs once)
        for i in range(self.num_encoder_layers):
            # Engram injection at internal layer (context-gated)
            if (self.engram is not None
                    and self.engram_inject_layer >= 0
                    and i == self.engram_inject_layer):
                engram_out = self.engram(input_ids, hidden=x)
                x = x + engram_out.astype(x.dtype)
            x, v_out = self.blocks[i](x, x0, v0=v0)
            if v0 is None and v_out is not None:
                v0 = mx.stop_gradient(v_out)
            skips.append(x)

        # Capsule init — use carry_capsules for cross-window persistence
        # Average carry across batch dim and broadcast to match current batch
        capsule_state = None
        if carry_capsules is not None:
            B_curr = x.shape[0]
            # Average carry state across old batch → (1, N, D), broadcast to new batch
            carry_avg = mx.mean(carry_capsules, axis=0, keepdims=True)  # (1, N, D)
            capsule_state = mx.broadcast_to(carry_avg, (B_curr, carry_avg.shape[1], carry_avg.shape[2]))
        if self.capsule_bank is not None:
            x, capsule_state, _ = self.capsule_bank(x, prev_capsules=capsule_state)

        encoded = x

        # Iterative correction loop with Koopman dynamics + adaptive halting
        sketch = None
        consistency_losses = []  # (c_pred, c_actual) pairs for aux loss
        prev_capsule_state = None

        num_passes = self.feedback_passes
        for correction_pass in range(num_passes + 1):
            if correction_pass > 0 and self.feedback_enabled and self.feedback_pooler is not None:
                sketch = self.feedback_pooler(rms_norm(x))
            else:
                sketch = None

            if self.capsule_bank is not None and correction_pass > 0:
                prev_capsule_state = capsule_state
                encoded, capsule_state, c_pred = self.capsule_bank(
                    encoded, prev_capsules=capsule_state
                )
                # Collect consistency loss pair
                if c_pred is not None:
                    consistency_losses.append((c_pred, mx.stop_gradient(capsule_state)))

                # Adaptive halting (eval only): check capsule convergence
                # Only at pass >= 2 — always run blind (0) + first feedback (1)
                if (not self.training and self.args.adaptive_halt_enabled
                        and prev_capsule_state is not None
                        and correction_pass >= 2):
                    # Relative capsule change
                    delta = mx.sqrt(mx.mean((capsule_state - prev_capsule_state) ** 2))
                    norm = mx.sqrt(mx.mean(capsule_state ** 2)) + 1e-8
                    relative_delta = delta / norm
                    # Halt if converged (materialize to check)
                    mx.eval(relative_delta)
                    if float(relative_delta.item()) < self.args.adaptive_halt_threshold:
                        # Capsule has converged — skip remaining passes
                        break

            x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0)

            if not self.feedback_enabled or self.feedback_pooler is None:
                break

        return rms_norm(x), capsule_state, consistency_losses

    def loss(self, input_ids, target_ids, carry_capsules=None):
        x, capsule_state, consistency_losses = self(input_ids, carry_capsules=carry_capsules)
        x = x.reshape(-1, x.shape[-1])
        y = target_ids.reshape(-1)
        # Tied embeddings
        w = self.tok_emb.weight.astype(x.dtype)
        if self.embed_proj_rev is not None:
            x = x @ self.embed_proj_rev.weight.astype(x.dtype).T
        logits = x @ w.T + self.vocab_bias.astype(x.dtype)
        logits = self.softcap(logits)
        ce_loss = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        # Capsule consistency auxiliary loss (Koopman dynamics training signal)
        if consistency_losses and self.args.koopman_consistency_weight > 0:
            consist_sum = mx.array(0.0, dtype=mx.float32)
            for c_pred, c_actual in consistency_losses:
                consist_sum = consist_sum + mx.mean((c_pred - c_actual) ** 2)
            consist_loss = consist_sum / len(consistency_losses)
            ce_loss = ce_loss + self.args.koopman_consistency_weight * consist_loss

        return ce_loss


# ---------------------------------------------------------------------------
# EMA Helper
# ---------------------------------------------------------------------------
class EMAHelper:
    def __init__(self, model, decay=0.997):
        self.decay = decay
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        self._ever_updated = False  # Track whether EMA has ever been updated

    def update(self, model):
        d = self.decay
        self._ever_updated = True
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = d * self.shadow[k] + (1.0 - d) * v

    def apply(self, model):
        if not self._ever_updated:
            return  # Don't overwrite with un-updated shadow (stale init weights)
        self.original = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
        model.update(tree_unflatten(list(self.shadow.items())))

    def restore(self, model):
        if hasattr(self, 'original'):
            model.update(tree_unflatten(list(self.original.items())))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class Muon:
    def __init__(self, keys, params, args):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params, grads, step, lr_mul):
        if self.args.muon_momentum_warmup_steps > 0:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            if k not in grads:
                continue
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model, args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if p.ndim == 2
            and not any(pat in k for pat in CTP)
            and "feedback" not in k
            and "capsule" not in k
            and k != self.embed_key
            and "embed_proj" not in k
            and "engram.tables" not in k
        ]
        # Koopman diagonal gets its own lower LR (stability-critical)
        self.koopman_diag_keys = [
            k for k in params if "koopman.diag" in k
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k not in self.matrix_keys and k != self.embed_key
            and k not in self.koopman_diag_keys
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr,
                                     betas=[args.beta1, args.beta2], eps=args.adam_eps)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr,
                                      betas=[args.beta1, args.beta2], eps=args.adam_eps)
        # Koopman diagonal: lower LR to protect spectral stability
        self.adam_koopman_diag = optim.Adam(learning_rate=0.01,
                                            betas=[args.beta1, args.beta2], eps=args.adam_eps)

    def step(self, model, grads_tree, step, lr_mul):
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        if self.embed_key in grads:
            updated.update(self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            ))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_params = {k: params[k] for k in self.scalar_keys if k in params}
        if scalar_grads:
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        # Koopman diagonal with lower LR
        self.adam_koopman_diag.learning_rate = 0.01 * lr_mul
        kd_grads = {k: grads[k] for k in self.koopman_diag_keys if k in grads}
        kd_params = {k: params[k] for k in self.koopman_diag_keys if k in params}
        if kd_grads:
            updated.update(self.adam_koopman_diag.apply_gradients(kd_grads, kd_params))

        model.update(tree_unflatten(list(updated.items())))

        # Stability constraint: clamp Koopman diagonal to (-0.999, 0.999)
        # This is a hard constraint from dynamical systems theory: ρ(A) must be < 1
        if (model.capsule_bank is not None
                and model.capsule_bank.koopman is not None):
            clamped = mx.clip(model.capsule_bank.koopman.diag, -0.999, 0.999)
            model.capsule_bank.koopman.diag = clamped


# ---------------------------------------------------------------------------
# Ternary serialization (base-3 packing + LZMA)
# ---------------------------------------------------------------------------
def pack_ternary_base3(q_np):
    """Pack ternary values {-1,0,+1} as base-3: 5 trits per byte."""
    flat = (q_np.astype(np.int8).ravel() + 1).astype(np.uint8)  # map to {0,1,2}
    n = flat.shape[0]
    pad = (5 - n % 5) % 5
    if pad > 0:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    grouped = flat.reshape(-1, 5)
    packed = (grouped[:, 0].astype(np.uint16)
              + grouped[:, 1].astype(np.uint16) * 3
              + grouped[:, 2].astype(np.uint16) * 9
              + grouped[:, 3].astype(np.uint16) * 27
              + grouped[:, 4].astype(np.uint16) * 81)
    return packed.astype(np.uint8), n


def _gptq_lite_ternary(arr_np, gs, num_percentiles=5):
    """GPTQ-lite: per-row clip percentile search before ternary quantization.
    
    For each row, try clipping at different percentiles and pick the one
    that minimizes reconstruction MSE. This gives better ternary approximation.
    """
    rows, cols = arr_np.shape
    pad_cols = (gs - cols % gs) % gs
    if pad_cols > 0:
        arr_padded = np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
    else:
        arr_padded = arr_np.copy()
    
    best_q = np.zeros_like(arr_padded, dtype=np.int8).reshape(-1, gs)
    best_scale = np.zeros((arr_padded.size // gs, 1), dtype=np.float32)
    best_mse = np.full(rows, np.inf, dtype=np.float32)
    
    # Percentiles to try: 100% (no clip), 99%, 97%, 95%, 90%
    percentiles = np.linspace(100.0, 90.0, num_percentiles)
    
    for pct in percentiles:
        clipped = arr_padded.copy()
        for r in range(rows):
            if pct < 100.0:
                threshold = np.percentile(np.abs(arr_padded[r]), pct)
                clipped[r] = np.clip(clipped[r], -threshold, threshold)
        
        grouped = clipped.reshape(-1, gs)
        scale = np.mean(np.abs(grouped), axis=-1, keepdims=True)
        scale = np.maximum(scale, 1e-8)
        q = np.clip(np.round(grouped / scale), -1, 1).astype(np.int8)
        
        # Reconstruct and measure per-row MSE
        recon = (q.astype(np.float32) * scale).reshape(rows, -1)
        mse = np.mean((arr_padded - recon) ** 2, axis=1)
        
        # Update best per row
        for r in range(rows):
            if mse[r] < best_mse[r]:
                best_mse[r] = mse[r]
                row_groups = arr_padded.shape[1] // gs
                best_q[r * row_groups:(r + 1) * row_groups] = q[r * row_groups:(r + 1) * row_groups]
                best_scale[r * row_groups:(r + 1) * row_groups] = scale[r * row_groups:(r + 1) * row_groups]
    
    return best_q, best_scale, pad_cols


def _build_hadamard_np(n):
    """Build normalized orthogonal Hadamard matrix in numpy. H @ H.T = I."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    h_half = _build_hadamard_np(n // 2)
    top = np.concatenate([h_half, h_half], axis=1)
    bot = np.concatenate([h_half, -h_half], axis=1)
    h = np.concatenate([top, bot], axis=0)
    return h / np.sqrt(n)

_HADAMARD_NP_CACHE = {}
def _get_hadamard_np(n):
    if n not in _HADAMARD_NP_CACHE:
        _HADAMARD_NP_CACHE[n] = _build_hadamard_np(n)
    return _HADAMARD_NP_CACHE[n]


def _turbo_ternary_quantize(arr_np, gs):
    """TurboQuant-style ternary quantization: Hadamard rotate → quantize → store rotated.

    At inference, the load path inverse-rotates (H is self-inverse) to recover dense weights.
    Storage is still ternary (base-3 packed), but quantization MSE is provably lower because
    Hadamard distributes outliers uniformly across coordinates.
    """
    rows, cols = arr_np.shape
    pad_cols = (gs - cols % gs) % gs
    if pad_cols > 0:
        arr_np = np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
    grouped = arr_np.reshape(-1, gs)

    # TurboQuant: Hadamard rotation before scalar quantization
    H = _get_hadamard_np(gs)
    grouped_rotated = grouped @ H

    # Quantize in rotated space
    scale = np.mean(np.abs(grouped_rotated), axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(grouped_rotated / scale), -1, 1).astype(np.int8)

    return q, scale, pad_cols


def quantize_for_export(model, args):
    """Quantize model to ternary + float, serialize to LZMA blob."""
    params = dict(tree_flatten(model.parameters()))
    # Force-evaluate all parameters before numpy conversion to avoid lazy graph recomputation
    mx.eval(*params.values())
    use_turbo = args.turbo_quant_export
    q_sd = {}
    for name, arr in params.items():
        arr_np = np.array(arr.astype(mx.float32), dtype=np.float32)
        if arr.ndim == 2 and arr.size > 1024 and "embed" not in name and "engram.tables" not in name:
            gs = args.bitnet_group_size
            rows, cols = arr_np.shape
            turbo_used = False

            if use_turbo and (gs & (gs - 1)) == 0:
                # TurboQuant: Hadamard rotation for near-optimal ternary quantization
                q, scale, pad_cols = _turbo_ternary_quantize(arr_np, gs)
                turbo_used = True
            elif args.gptq_lite_enabled and args.gptq_lite_percentiles > 1:
                # GPTQ-lite: clip percentile search for better ternary approximation
                q, scale, pad_cols = _gptq_lite_ternary(
                    arr_np, gs, num_percentiles=args.gptq_lite_percentiles
                )
            else:
                # Plain ternary quantization
                pad_cols = (gs - cols % gs) % gs
                if pad_cols > 0:
                    arr_np = np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
                grouped = arr_np.reshape(-1, gs)
                scale = np.mean(np.abs(grouped), axis=-1, keepdims=True)
                scale = np.maximum(scale, 1e-8)
                q = np.clip(np.round(grouped / scale), -1, 1).astype(np.int8)

            packed, n_trits = pack_ternary_base3(q)
            scale_f16 = scale.astype(np.float16)
            entry = {"type": "ternary", "packed": packed, "scale": scale_f16,
                     "shape": (rows, cols), "n_trits": n_trits, "pad_cols": pad_cols}
            if turbo_used:
                entry["turbo"] = True  # load path must inverse-Hadamard after dequant
            q_sd[name] = entry
        else:
            q_sd[name] = {"type": "float", "data": arr_np.astype(np.float16)}
    blob = pickle.dumps(q_sd, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = lzma.compress(blob, preset=9)
    return compressed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=256 * 4)
    return tokens.astype(np.int32)


class TokenStream:
    def __init__(self, pattern, log_fn=None):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn:
                self.log_fn(f"WARNING: starting epoch:{self.epoch}")
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, self.tokens.size - self.pos)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern, log_fn=None):
        self.stream = TokenStream(pattern, log_fn=log_fn)

    def next_batch(self, batch_tokens, seq_len):
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def build_luts(sp, vocab_size):
    table_size = max(sp.vocab_size(), vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_leading_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    for tid in range(sp.vocab_size()):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return base_bytes, has_leading_space, is_boundary


def eval_val(model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut, log_fn=None):
    """Validation with cross-window capsule carry.

    When capsule_carry_enabled, capsule state persists across sequential
    validation batches with exponential decay. This gives the model structured
    long-range memory during eval at zero parameter cost.
    """
    model.eval()  # Enable adaptive halting (self.training = False)
    seq_len = args.train_seq_len
    batch_seqs = max(1, args.val_batch_size // seq_len)
    total_seqs = (val_tokens.size - 1) // seq_len
    total_loss = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    # Cross-window capsule carry state
    carry_capsules = None
    decay = args.capsule_carry_decay if args.capsule_carry_enabled else 0.0
    use_carry = args.capsule_carry_enabled and args.capsule_enabled

    for start in range(0, total_seqs, batch_seqs):
        end = min(start + batch_seqs, total_seqs)
        raw_s = start * seq_len
        raw_e = end * seq_len + 1
        chunk = val_tokens[raw_s:raw_e]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)

        if use_carry:
            # Forward pass to get capsule state for carry
            hidden, capsule_state, _ = model(x, carry_capsules=carry_capsules)
            # Compute loss from hidden
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            y_flat = y.reshape(-1)
            w = model.tok_emb.weight.astype(hidden_flat.dtype)
            if model.embed_proj_rev is not None:
                hidden_flat = hidden_flat @ model.embed_proj_rev.weight.astype(hidden_flat.dtype).T
            logits = hidden_flat @ w.T + model.vocab_bias.astype(hidden_flat.dtype)
            logits = model.softcap(logits)
            loss = nn.losses.cross_entropy(logits.astype(mx.float32), y_flat, reduction="mean")
            mx.eval(loss)
            # Update carry with exponential decay — average across batch for size-independence
            if capsule_state is not None:
                # Reduce to (1, N, D) for batch-independent carry
                cs_avg = mx.mean(capsule_state, axis=0, keepdims=True)
                if carry_capsules is not None:
                    carry_capsules = mx.stop_gradient(
                        decay * carry_capsules + (1.0 - decay) * cs_avg
                    )
                else:
                    carry_capsules = mx.stop_gradient(cs_avg)
                mx.eval(carry_capsules)
        else:
            loss = model.loss(x, y).astype(mx.float32)
            mx.eval(loss)

        n = float(y.size)
        total_loss += float(loss.item()) * n
        prev_ids = x_np.ravel()
        tgt_ids = y_np.ravel()
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).astype(np.int16)
        total_tokens += n
        total_bytes += float(bytes_np.sum())
    model.train()  # Restore training mode
    val_loss = total_loss / total_tokens
    bpt = val_loss / math.log(2.0)
    val_bpb = bpt * (total_tokens / total_bytes)
    return val_loss, val_bpb


def load_validation_tokens(pattern, seq_len, max_tokens=0):
    files = sorted(glob.glob(pattern))
    all_tokens = []
    total = 0
    for f in files:
        shard = load_data_shard(f)
        all_tokens.append(shard)
        total += shard.size
        if max_tokens > 0 and total >= max_tokens:
            break
    tokens = np.concatenate(all_tokens)
    if max_tokens > 0 and tokens.size > max_tokens:
        tokens = tokens[:max_tokens]
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------
def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(float(mx.sum(g * g).item()) for g in flat.values())
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


# ---------------------------------------------------------------------------
# Chunked loss + grad for memory management
# ---------------------------------------------------------------------------
def token_chunks(total_tokens, seq_len, max_chunk):
    usable = (total_tokens // seq_len) * seq_len
    chunk = max((max_chunk // seq_len) * seq_len, seq_len)
    chunks = []
    remaining = usable
    while remaining > 0:
        c = min(remaining, chunk)
        chunks.append(c)
        remaining -= c
    return chunks


def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"

    def log(msg, console=True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log(f"Ternary Reasoner — MLX local testing")
    log(f"Python {sys.version}")
    log(f"MLX {mx.__version__}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, max_tokens=args.max_val_tokens)
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = build_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log)

    # Model
    model = GPT(args)
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
        f"heads:{args.num_heads} seq_len:{args.train_seq_len}")
    log(f"feedback:{args.feedback_enabled} passes:{args.feedback_passes} "
        f"capsule:{args.capsule_enabled} engram:{args.bigram_hash_enabled}({args.engram_num_heads}h{args.engram_num_orders}o@L{args.engram_inject_layer}) "
        f"vrl:{args.vrl_enabled} xsa_start:{args.xsa_start_layer} "
        f"partial_rope:{args.partial_rope_dims} lrelu2:{args.activation_type}")
    if args.koopman_enabled and args.capsule_enabled:
        log(f"koopman:rank={args.koopman_rank} diag_init={args.koopman_diag_init} "
            f"consist_w={args.koopman_consistency_weight} "
            f"halt:{args.adaptive_halt_enabled}@{args.adaptive_halt_threshold} "
            f"carry:{args.capsule_carry_enabled}@{args.capsule_carry_decay}")
    log(f"turbo_quant: train={args.turbo_quant_train} export={args.turbo_quant_export}")

    opt = SplitOptimizers(model, args)

    # Compiled functions
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    # EMA
    ema = EMAHelper(model, args.ema_decay) if args.ema_enabled else None

    # LR schedule
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_fraction <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = int(args.iterations * (1.0 - args.warmdown_fraction))
            if step >= warmdown_start:
                return max((args.iterations - step) / max(args.iterations * args.warmdown_fraction, 1), 0.0)
            return 1.0
        warmdown_ms = max_wallclock_ms * args.warmdown_fraction
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Training loop
    train_time_ms = 0.0
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Switch to eval-time feedback passes
            if args.feedback_enabled:
                model.set_feedback_passes(args.eval_feedback_passes)
            val_loss, val_bpb = eval_val(model, args, val_tokens, base_bytes_lut,
                                         has_leading_space_lut, is_boundary_lut, log_fn=log)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms")
            # Restore training-time feedback passes
            if args.feedback_enabled:
                model.set_feedback_passes(model._train_feedback_passes)
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        step_t0 = time.perf_counter()
        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps

        for _ in range(args.grad_accum_steps):
            chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len,
                                       args.mlx_max_microbatch_tokens)
            total_chunk_tokens = float(sum(chunk_sizes))
            for ct in chunk_sizes:
                x, y = train_loader.next_batch(ct, args.train_seq_len)
                loss, grads = compiled_loss_and_grad(x, y)
                cs = float(y.size) / total_chunk_tokens * grad_scale
                train_loss = train_loss + loss.astype(mx.float32) * cs
                accum = accumulate_flat_grads(accum, grads, cs)
                if args.mlx_eager_eval:
                    mx.eval(train_loss, accum)

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        opt.step(model, grads, step=step, lr_mul=scale)
        # Force evaluation of all parameters (mx.eval(model.state) is a no-op in MLX ≥ 0.22)
        mx.eval(*[v for _, v in tree_flatten(model.parameters())])

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        step += 1

        # EMA update
        if ema is not None:
            progress = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / args.iterations
            if progress >= args.ema_start_fraction:
                ema.update(model)

        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            log(f"step:{step}/{args.iterations} loss:{float(train_loss.item()):.4f} "
                f"t:{approx_ms:.0f}ms step:{step_ms:.0f}ms")

        if max_wallclock_ms and stop_after_step is None:
            approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            if approx_ms >= max_wallclock_ms:
                stop_after_step = step

    # Apply EMA before serialization (only if it was actually updated)
    if ema is not None:
        if ema._ever_updated:
            ema.apply(model)
            log("EMA shadow weights applied")
        else:
            log("EMA shadow was never updated (short run) — using trained weights directly")

    # Serialize
    blob = quantize_for_export(model, args)
    artifact_path = out_dir / f"{args.run_id}_model.ternary.ptz"
    with artifact_path.open("wb") as f:
        f.write(blob)
    code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
    total = len(blob) + code_bytes
    log(f"artifact:{len(blob)/1e6:.2f}MB code:{code_bytes} total:{total}/{16000000} "
        f"({'FITS' if total <= 16000000 else 'OVER'}")

    log(f"Done. Artifact saved to {artifact_path}")


if __name__ == "__main__":
    main()
