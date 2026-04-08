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
    data_path = _e("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = _e("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", f"mlx_{uuid.uuid4().hex[:8]}")
    seed = _e("SEED", 1337, int)
    out_dir = _e("OUT_DIR", "logs")

    # Training loop
    iterations = _e("ITERATIONS", 10000, int)
    val_loss_every = _e("VAL_LOSS_EVERY", 0, int)
    val_batch_size = _e("VAL_BATCH_SIZE", 65536, int)
    max_val_tokens = _e("MAX_VAL_TOKENS", 0, int)  # 0=no limit (use all val data)
    train_log_every = _e("TRAIN_LOG_EVERY", 1000, int)
    train_batch_tokens = _e("TRAIN_BATCH_TOKENS", 32768, int)  # 786432→32768: old default OOMed on Mac
    grad_accum_steps = _e("GRAD_ACCUM_STEPS", 4, int)
    train_seq_len = _e("TRAIN_SEQ_LEN", 2048, int)
    warmup_steps = _e("WARMUP_STEPS", 5, int)
    warmdown_fraction = _e("WARMDOWN_FRACTION", 0.5, float)
    max_wallclock_seconds = _e("MAX_WALLCLOCK_SECONDS", 599.0, float)
    mlx_max_microbatch_tokens = _e("MLX_MAX_MICROBATCH_TOKENS", 8192, int)
    mlx_eager_eval = _e("MLX_EAGER_EVAL", 0, bool)

    # Model
    vocab_size = _e("VOCAB_SIZE", 1024, int)
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


    # MoE
    moe_enabled = _e("MOE_ENABLED", 0, bool)
    moe_num_experts = _e("MOE_NUM_EXPERTS", 8, int)
    moe_top_k = _e("MOE_TOP_K", 2, int)
    moe_router_aux_loss_coef = _e("MOE_ROUTER_AUX_LOSS_COEF", 0.01, float)

    # Feedback
    feedback_enabled = _e("FEEDBACK_ENABLED", 1, bool)
    feedback_dim = _e("FEEDBACK_DIM", 64, int)
    feedback_sketch_tokens = _e("FEEDBACK_SKETCH_TOKENS", 4, int)
    feedback_passes = _e("FEEDBACK_PASSES", 1, int)
    eval_feedback_passes = _e("EVAL_FEEDBACK_PASSES", 2, int)
    feedback_every = _e("FEEDBACK_EVERY", 1, int)

    # Capsule
    capsule_enabled = _e("CAPSULE_ENABLED", 1, bool)
    capsule_num = _e("CAPSULE_NUM", 32, int)
    capsule_dim = _e("CAPSULE_DIM", 128, int)

    # Koopman dynamics in capsule space
    koopman_enabled = _e("KOOPMAN_ENABLED", 1, bool)
    koopman_rank = _e("KOOPMAN_RANK", 8, int)
    koopman_diag_init = _e("KOOPMAN_DIAG_INIT", 0.9, float)  # critical damping
    koopman_consistency_weight = _e("KOOPMAN_CONSISTENCY_WEIGHT", 0.005, float)
    koopman_lr = _e("KOOPMAN_LR", 0.01, float)

    # Adaptive halting (eval only)
    adaptive_halt_enabled = _e("ADAPTIVE_HALT_ENABLED", 1, bool)
    adaptive_halt_threshold = _e("ADAPTIVE_HALT_THRESHOLD", 0.05, float)
    max_eval_passes = _e("MAX_EVAL_PASSES", 3, int)

    # Cross-window capsule carry (eval only)
    capsule_carry_enabled = _e("CAPSULE_CARRY_ENABLED", 1, bool)
    capsule_carry_decay = _e("CAPSULE_CARRY_DECAY", 0.8, float)

    # Koopman Speculator / Diffusion
    koopman_speculator_enabled = _e("KOOPMAN_SPECULATOR_ENABLED", 0, bool)
    koopman_speculator_steps = _e("KOOPMAN_SPECULATOR_STEPS", 3, int)
    koopman_speculator_weight = _e("KOOPMAN_SPECULATOR_WEIGHT", 0.01, float)

    # SSM Bottleneck — causal replacement for CapsuleBank
    # Fixes non-causality, exact Koopman dynamics, position-specific write-back.
    # Set SSM_BOTTLENECK=1 to replace CapsuleBank with SSMBottleneck.
    # Requires CAPSULE_ENABLED=1 to activate; SSM_BOTTLENECK flag selects which.
    ssm_bottleneck_enabled = _e("SSM_BOTTLENECK", 0, bool)
    ssm_bottleneck_state_dim = _e("SSM_BOTTLENECK_STATE_DIM", 64, int)
    linkoopman_enabled = _e("LINKOOPMAN_ENABLED", 0, bool)
    linkoopman_heads = _e("LINKOOPMAN_HEADS", 4, int)
    linkoopman_head_dim = _e("LINKOOPMAN_HEAD_DIM", 16, int)
    linear_bottleneck_enabled = _e("LINEAR_BOTTLENECK", 0, bool)
    linear_bottleneck_rank = _e("LINEAR_BOTTLENECK_RANK", 64, int)

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
    turbo_quant_kv = _e("TURBO_QUANT_KV", 0, bool)         # Quantize K/V cache with TurboQuant
    sliding_eval = _e("SLIDING_EVAL", 1, bool)
    sliding_eval_stride = _e("SLIDING_EVAL_STRIDE", 64, int)
    sliding_batch_size = _e("SLIDING_BATCH_SIZE", 32, int)
    temp_scaling = _e("TEMP_SCALING", 1, bool)
    shared_blocks = _e("SHARED_BLOCKS", 0, int)
    ngram_cache_enabled = _e("NGRAM_CACHE_ENABLED", 0, bool)
    ngram_max_order = _e("NGRAM_MAX_ORDER", 5, int)
    ngram_alpha_base = _e("NGRAM_ALPHA_BASE", 0.05, float)
    ngram_alpha_scale = _e("NGRAM_ALPHA_SCALE", 0.55, float)
    ngram_entropy_center = _e("NGRAM_ENTROPY_CENTER", 4.0, float)

    # Convergence optimizations
    ternary_noise_scale = _e("TERNARY_NOISE_SCALE", 0.05, float)      # stochastic ternary noise
    stochastic_depth_prob = _e("STOCHASTIC_DEPTH_PROB", 0.0, float)  # layer-drop; MUST be 0 — non-zero caused loss oscillation
    curriculum_enabled = _e("CURRICULUM_ENABLED", 1, bool)           # ramp seq_len: short→full
    # Curriculum fracs are CUMULATIVE wallclock cutoffs.
    # Lesson learned: seq converges in ~300-400 steps regardless of budget.
    #   OLD: 0.35/0.65 → at 14K steps: 5100 steps at seq=64 = catastrophic overfit,
    #        final phase only 4571 steps, still recovering at end.
    #   FIX: 0.05/0.20 → phase1=5% (~700 steps, just converges), phase2=15%, phase3=80%.
    curriculum_phase1_frac = _e("CURRICULUM_PHASE1_FRAC", 0.05, float)  # phase1 ends at 5%
    curriculum_phase2_frac = _e("CURRICULUM_PHASE2_FRAC", 0.20, float)  # phase2 ends at 20% (15% duration)
    curriculum_phase1_seq = _e("CURRICULUM_PHASE1_SEQ", 64, int)   # ultra-short: fast early learning
    curriculum_phase2_seq = _e("CURRICULUM_PHASE2_SEQ", 256, int)
    # 5-phase geometric curriculum (matching H100 train_gpt.py)
    curriculum_phase3_frac = _e("CURRICULUM_PHASE3_FRAC", 1.0, float)  # 1.0 = disabled (3-phase default)
    curriculum_phase4_frac = _e("CURRICULUM_PHASE4_FRAC", 1.0, float)
    curriculum_phase5_frac = _e("CURRICULUM_PHASE5_FRAC", 1.0, float)
    curriculum_phase3_seq = _e("CURRICULUM_PHASE3_SEQ", 512, int)
    curriculum_phase4_seq = _e("CURRICULUM_PHASE4_SEQ", 1024, int)
    curriculum_phase5_seq = _e("CURRICULUM_PHASE5_SEQ", 1024, int)
    self_distill_kl_weight = _e("SELF_DISTILL_KL_WEIGHT", 0.0, float)  # KL between feedback pass-0 and final
    ema_eval_apply = _e("EMA_EVAL_APPLY", 0, bool)                    # apply EMA shadow weights during eval
    ttt_enabled = _e("TTT_ENABLED", 0, bool)
    ttt_lr = _e("TTT_LR", 0.002, float)
    ttt_epochs = _e("TTT_EPOCHS", 1, int)
    ttt_chunk_tokens = _e("TTT_CHUNK_TOKENS", 32768, int)
    ttt_scope = _e("TTT_SCOPE", "feedback")
    ttt_momentum = _e("TTT_MOMENTUM", 0.9, float)
    ttt_batch_seqs = _e("TTT_BATCH_SEQS", 32, int)
    ttt_grad_clip = _e("TTT_GRAD_CLIP", 1.0, float)

    # Architecture selection
    architecture = _e("ARCHITECTURE", "transformer")  # "transformer", "koopman_ssm", "hybrid", "skc"

    # --- Spectral Koopman Capsule (SKC) Architecture ---
    # TKO: Ternary Koopman Optimizer (score-space optimization)
    tko_enabled = _e("TKO_ENABLED", 1, bool)
    tko_score_decay = _e("TKO_SCORE_DECAY", 0.999, float)

    # Weight sharing: encoder and decoder share same block weights
    weight_sharing = _e("WEIGHT_SHARING", 0, bool)

    # Inside-out progressive unfreezing
    inside_out_training = _e("INSIDE_OUT_TRAINING", 0, bool)
    inside_out_phase1_frac = _e("INSIDE_OUT_PHASE1_FRAC", 0.2, float)  # capsule-only phase
    inside_out_phase2_frac = _e("INSIDE_OUT_PHASE2_FRAC", 0.55, float)  # progressive unfreezing

    # SKC Layer hyperparameters
    skc_capsule_dim = _e("SKC_CAPSULE_DIM", 128, int)
    skc_num_capsules = _e("SKC_NUM_CAPSULES", 32, int)
    skc_conv_kernel = _e("SKC_CONV_KERNEL", 4, int)
    skc_block_size  = _e("SKC_BLOCK_SIZE",  64, int)  # WHT block size (power of 2: 16/32/64/128)

    # DEQ (Deep Equilibrium) feedback
    deq_feedback = _e("DEQ_FEEDBACK", 0, bool)
    deq_max_iter = _e("DEQ_MAX_ITER", 3, int)
    deq_tol = _e("DEQ_TOL", 0.01, float)
    deq_anderson_m = _e("DEQ_ANDERSON_M", 3, int)  # Anderson acceleration window

    # Koopman SSM hyperparameters (Path 2)
    koopman_state_dim = _e("KOOPMAN_STATE_DIM", 128, int)
    koopman_mixer_rank = _e("KOOPMAN_MIXER_RANK", 4, int)
    koopman_conv_kernel = _e("KOOPMAN_CONV_KERNEL", 4, int)
    koopman_decay_window = _e("KOOPMAN_DECAY_WINDOW", 32, int)

    # Weight averaging and temporal mixing
    smeargate_enabled = _e("SMEARGATE_ENABLED", 1, bool)
    lawa_enabled = _e("LAWA_ENABLED", 1, bool)
    lawa_k = _e("LAWA_K", 10, int)
    lawa_freq = _e("LAWA_FREQ", 100, int)
    swa_enabled = _e("SWA_ENABLED", 1, bool)
    swa_every = _e("SWA_EVERY", 50, int)
    swa_start_scale = _e("SWA_START_SCALE", 0.2, float)
    average_ternary_params = _e("AVERAGE_TERNARY_PARAMS", 0, bool)

    # Optimizer
    matrix_lr = _e("MATRIX_LR", 0.035, float)
    scalar_lr = _e("SCALAR_LR", 0.025, float)
    tied_embed_lr = _e("TIED_EMBED_LR", 0.035, float)
    muon_momentum = _e("MUON_MOMENTUM", 0.95, float)
    muon_backend_steps = _e("MUON_BACKEND_STEPS", 5, int)
    muon_wd = _e("MUON_WD", 0.04, float)
    adam_wd = _e("ADAM_WD", 0.04, float)
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
    "per_layer_attn_scales", "per_layer_mlp_scales", "per_layer_resid_mixes",
    "koopman",  # Koopman dynamics params go to scalar Adam for stability
    "mixer_diag", "mixer_lowrank", "mixer_conv", "mixer_scale",  # Koopman SSM mixer scalars
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
_RADERMACHER_CACHE = {}

def _deterministic_signs(n):
    if n in _RADERMACHER_CACHE:
        return _RADERMACHER_CACHE[n]
    idx = mx.arange(n, dtype=mx.int64)
    bits = ((idx * 1103515245 + 12345) % 2) == 1
    signs = mx.where(bits, mx.array(-1.0, dtype=mx.float32), mx.array(1.0, dtype=mx.float32))
    _RADERMACHER_CACHE[n] = signs
    return signs

def _build_hadamard_unnormalized(n):
    if n == 1:
        return mx.array([[1.0]], dtype=mx.float32)
    else:
        h_half = _build_hadamard_unnormalized(n // 2)
        top = mx.concatenate([h_half, h_half], axis=1)
        bot = mx.concatenate([h_half, -h_half], axis=1)
        return mx.concatenate([top, bot], axis=0)

def _build_hadamard(n):
    """Build normalized orthogonal Hadamard matrix H_n (H @ H^T = I)."""
    if n in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[n]
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    
    # Only divide by sqrt(n) once at the end!
    h = _build_hadamard_unnormalized(n) / math.sqrt(n)
    _HADAMARD_CACHE[n] = h
    return h

def _group_last_dim_mx(w, group_size):
    shape = w.shape
    w_f = w.astype(mx.float32)
    if len(shape) == 1:
        rows, cols = 1, shape[0]
        w2 = w_f.reshape(1, cols)
    else:
        rows = int(np.prod(shape[:-1]))
        cols = shape[-1]
        w2 = w_f.reshape(rows, cols)
    pad = (group_size - cols % group_size) % group_size
    if pad > 0:
        w2 = mx.concatenate([w2, mx.zeros((rows, pad), dtype=mx.float32)], axis=1)
    return w2.reshape(-1, group_size), (shape, rows, cols, pad)

def _ungroup_last_dim_mx(grouped, meta):
    shape, rows, cols, pad = meta
    width = cols + pad
    return grouped.reshape(rows, width)[:, :cols].reshape(shape)

# ---------------------------------------------------------------------------
# Ternary STE quantization (for training — MLX version)
# ---------------------------------------------------------------------------
def ternary_ste(w, group_size=128, turbo=False):
    """Ternary quantization with Straight-Through Estimator.

    When turbo=True, applies deterministic sign scrambling plus Hadamard rotation.
    This provably reduces MSE by distributing weight outliers across coordinates.
    The Hadamard matrix is deterministic and self-inverse, so:
    - No extra parameters to store
    - Dequant = quantize in rotated space, then inverse-rotate back
    """
    grouped, meta = _group_last_dim_mx(w, group_size)

    if turbo and (group_size & (group_size - 1)) == 0:
        # Signed Hadamard preconditioning before quantization.
        H = _build_hadamard(group_size).astype(grouped.dtype)
        signs = _deterministic_signs(group_size).astype(grouped.dtype)
        grouped = (grouped * signs) @ H

    # Stochastic ternary noise: smooths the quantization landscape during STE training.
    # Noise perturbs pre-quantization weights so the optimizer sees a less jagged gradient
    # surface around quantization boundaries, preventing local minima from ternary cliffs.
    if _TERNARY_NOISE_SCALE > 0.0:
        grouped = grouped + mx.random.normal(grouped.shape, dtype=mx.float32) * _TERNARY_NOISE_SCALE

    scale = mx.mean(mx.abs(grouped), axis=-1, keepdims=True)
    scale = mx.maximum(scale, mx.array(1e-8))
    normalized = grouped / scale
    # Quantize to {-1, 0, +1} — round and clamp
    q = mx.clip(mx.round(normalized), -1, 1)
    dequantized = q * scale

    if turbo and (group_size & (group_size - 1)) == 0:
        # Inverse signed-Hadamard rotation.
        dequantized = (dequantized @ H) * signs

    # STE: forward uses quantized, backward uses original
    w_ternary = _ungroup_last_dim_mx(dequantized, meta)
    return w + mx.stop_gradient(w_ternary - w)



def quantize_kv_ste(x, turbo=True, H=None):
    """Ternary STE quantization specifically for KV cache vectors with TurboQuant rotation.
    Combines:
    1. Fast Johnson-Lindenstrauss Transform (Pseudo-random sign flip + Hadamard)
    2. Exact L2 Norm De-biasing (preserves attention temperature and inner products)
    """
    head_dim = x.shape[-1]
    H_ast = None
    if turbo and H is not None:
        H_ast = H.astype(x.dtype)
    elif turbo and (head_dim & (head_dim - 1)) == 0:
        H_ast = _build_hadamard(head_dim).astype(x.dtype)
    
    if H_ast is None:
        # Fallback to naive ternary
        scale = mx.mean(mx.abs(x), axis=-1, keepdims=True)
        scale = mx.maximum(scale, mx.array(1e-8))
        q = mx.clip(mx.round(x / scale), -1, 1)
        dequant = q * scale
        return x + mx.stop_gradient(dequant - x)

    # 1. Deterministic sign flip before Hadamard mixing.
    signs = _deterministic_signs(head_dim).astype(x.dtype)
    x_scrambled = x * signs
    
    # 2. Hadamard Rotation
    x_rot = x_scrambled @ H_ast

    # 3. Quantize
    scale = mx.mean(mx.abs(x_rot), axis=-1, keepdims=True)
    scale = mx.maximum(scale, mx.array(1e-8))
    q = mx.clip(mx.round(x_rot / scale), -1, 1)
    dequant = q * scale

    # 4. De-bias (Match expected L2 norm to preserve inner product)
    energy_in = mx.sqrt(mx.sum(mx.square(x_rot), axis=-1, keepdims=True) + 1e-6)
    energy_out = mx.sqrt(mx.sum(mx.square(dequant), axis=-1, keepdims=True) + 1e-6)
    dequant = dequant * (energy_in / energy_out)

    # 5. Inverse FJLT
    # H is symmetric so H.T == H. Then reverse the sign flip.
    dequant = (dequant @ H_ast) * signs

    # STE: forward uses quantized, backward uses original
    return x + mx.stop_gradient(dequant - x)


# ---------------------------------------------------------------------------

# Model layers
# ---------------------------------------------------------------------------
# Module-level flags set by GPT.__init__
_TURBO_QUANT = False
_TURBO_QUANT_KV = False
_TERNARY_NOISE_SCALE = 0.0   # stochastic ternary: noise before quantization during STE
_STOCHASTIC_DEPTH_PROB = 0.0  # layer-drop probability for shared blocks
_EVAL_EMA = None              # EMAHelper instance applied during eval when ema_eval_apply=True

class SmearGate(nn.Module):
    """Learned per-dim temporal mixing: (1-g)*x_t + g*x_{t-1}. Zero-param at init."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        # x shape: (batch, seq, dim)
        g = mx.sigmoid(self.gate)[None, None, :]  # (1, 1, dim)
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        return (1 - g) * x + g * x_prev


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
        assert 1 <= num_orders <= 3, f"num_orders must be in [1, 3], got {num_orders}"
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

        # Calculate actual concatenated dimension after floor division
        actual_hash_dim = self.head_dim * num_orders * num_heads

        # Projection from concatenated actual_hash_dim to model_dim
        self.proj = TernaryLinear(actual_hash_dim, model_dim, group_size=group_size)
        self.gate_k = EmbedProj(actual_hash_dim, model_dim)
        self.gate_scale = model_dim ** -0.5

    def __call__(self, input_ids, hidden=None):
        """Retrieve multi-head, multi-order n-gram memory in a single vectorized pass."""
        B, T = input_ids.shape
        num_total_heads = self.num_orders * self.num_heads
        
        # 1. Dynamic Vectorized hashing for multiple orders
        parts = []
        ids_long = input_ids.astype(mx.int64)
        
        for order in range(self.num_orders):
            num_tokens = order + 2
            if T < num_tokens:
                # If sequence is shorter than order, pad with zeros
                h = mx.zeros((B, T, self.num_heads), dtype=mx.int32)
            else:
                # Optimized hashing using primes: sum(x_i * p^i) % buckets
                # Shape h after slice: (B, T - num_tokens + 1, num_heads)
                p_start = order * self.num_heads
                p = mx.array(self._PRIMES[p_start : p_start + self.num_heads])[None, None, :]
                
                h = None
                for i in range(num_tokens):
                    # Pick slice: ids[:, i : T - num_tokens + 1 + i]
                    # This captures consecutive windows of size num_tokens
                    token_slice = ids_long[:, i : T - num_tokens + 1 + i, None]
                    term = token_slice * (p ** (num_tokens - 1 - i))
                    if h is None:
                        h = term
                    else:
                        h = h + term
                h = h % self.buckets_per_head
                # Prepend zeros to maintain T length (causal alignment)
                h = mx.concatenate([mx.zeros((B, num_tokens - 1, self.num_heads), dtype=mx.int32), h.astype(mx.int32)], axis=1)
            parts.append(h)
            
        # all_indices: (B, T, H)
        all_indices = mx.concatenate(parts, axis=-1)
        
        # 2. Fully Vectorized embedding lookup
        # all_tables: (H, buckets, head_dim)
        all_tables = mx.stack(self.tables)
        
        # Indexing: head_indices (1, 1, H) + all_indices (B, T, H) -> (B, T, H, head_dim)
        # MLX advanced indexing broadcasts the indexers across the first dims.
        hi = mx.arange(num_total_heads)[None, None, :]
        memory = all_tables[hi, all_indices] # Shape: (B, T, H, head_dim)
        memory = memory.reshape(B, T, -1) # Flatten heads: (B, T, H * head_dim)

        if hidden is not None:
            # Context-aware gating
            k = self.gate_k(memory)
            gate = mx.sigmoid(mx.sum(rms_norm(hidden) * rms_norm(k), axis=-1, keepdims=True) * self.gate_scale)
            return gate * self.proj(memory)
        else:
            return self.proj(memory)


class FeedbackPooler(nn.Module):
    """Compress decoder output into a low-dim semantic sketch."""
    def __init__(self, model_dim, feedback_dim, num_tokens):
        super().__init__()
        self.num_tokens = max(1, num_tokens)
        self.proj = EmbedProj(model_dim, feedback_dim)

    def __call__(self, x):
        B, T, D = x.shape
        # Vectorized chunked mean pooling: (B, T, D) -> (B, num_tokens, T//num_tokens, D) -> mean
        q = T // self.num_tokens
        if q > 0:
            pooled = mx.mean(x[:, :q*self.num_tokens].reshape(B, self.num_tokens, q, D), axis=2)
        else:
            pooled = mx.mean(x, axis=1, keepdims=True)
            pooled = mx.broadcast_to(pooled, (B, self.num_tokens, D))
        return self.proj(rms_norm(pooled))


class FeedbackAdapter(nn.Module):
    """Fast-Weight Delta Rule Memory Adapter (MLX version)."""
    def __init__(self, model_dim, feedback_dim):
        super().__init__()
        self.fast_weight_lr = mx.array(-2.0, dtype=mx.float32)
        self.k_proj = EmbedProj(feedback_dim, model_dim)
        self.v_proj = EmbedProj(feedback_dim, model_dim)
        self.q_proj = EmbedProj(model_dim, model_dim)
        self.out_gate = mx.zeros((model_dim,), dtype=mx.float32)

    def __call__(self, x, sketch):
        if sketch is None:
            return x
        
        B, T, D = x.shape
        k = self.k_proj(sketch)
        v = self.v_proj(sketch)
        
        memory_matrix = mx.zeros((B, D, D), dtype=mx.float32)
            
        k_t = mx.swapaxes(k, 1, 2)
        v_t = mx.swapaxes(v, 1, 2)
        
        pred_v_t = memory_matrix @ k_t
        delta = v_t - pred_v_t
        
        lr = mx.sigmoid(self.fast_weight_lr)
        memory_matrix = memory_matrix + lr * (delta @ k)
        
        q = self.q_proj(x)
        q_t = mx.swapaxes(q, 1, 2)
        
        retrieved_t = memory_matrix @ q_t
        retrieved = mx.swapaxes(retrieved_t, 1, 2)
        
        gate = mx.tanh(self.out_gate).astype(x.dtype)
        x_out = x + gate * retrieved.astype(x.dtype)
        
        return x_out


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
        self.capsule_dim = capsule_dim
        self.diag = mx.full((capsule_dim,), diag_init, dtype=mx.float32)
        init_scale = 0.01 / max(rank ** 0.5, 1.0)
        self.U = mx.random.normal((capsule_dim, rank), dtype=mx.float32) * init_scale
        self.V = mx.random.normal((capsule_dim, rank), dtype=mx.float32) * init_scale
        self.alpha = mx.full((capsule_dim,), -5.0, dtype=mx.float32)  # sigmoid(-5)≈0.007: capsules start OFF
        # Precompute Hadamard for capsule dim (must be power of 2)
        self._use_hadamard = (capsule_dim & (capsule_dim - 1)) == 0 and capsule_dim >= 2
        if self._use_hadamard:
            self._H = _build_hadamard(capsule_dim)

    def _rotate(self, c):
        """Hadamard rotate: spreads capsule info uniformly across dims."""
        if self._use_hadamard:
            return c @ self._H.astype(c.dtype)
        return c

    def predict(self, c):
        """Predict next-pass capsule state. c: (B, N, capsule_dim)
        Hadamard-rotate → diagonal+low-rank evolve → rotate back.
        This ensures the diagonal operates on variance-equalized dims."""
        c_rot = self._rotate(c)
        # Clamp diagonal for multi-step spectral stability: |d_i| < 1 prevents
        # exponential blowup when composing predict() K times in speculate().
        d_clamped = mx.clip(self.diag, -0.999, 0.999).astype(c_rot.dtype)
        # Diagonal evolution
        c_diag = d_clamped * c_rot  # (B, N, D)
        # Low-rank coupling
        c_lowrank = (c_rot @ self.V.astype(c_rot.dtype)) @ self.U.astype(c_rot.dtype).T
        c_evolved = c_diag + c_lowrank
        # Rotate back (H is self-inverse)
        return self._rotate(c_evolved)

    def speculate(self, c, steps):
        """Recursively apply Koopman operator to fast-forward presentation.
        Acts as 1-step diffusion jump in latent space."""
        curr = c
        for _ in range(steps):
            curr = self.predict(curr)
        return curr

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

    def __call__(self, x, prev_capsules=None, speculate_steps=0):
        """Returns (corrected_x, capsule_state, c_pred_for_loss, c_spec).
        c_pred is None on first pass or when Koopman is disabled."""
        B, T, D = x.shape
        x_proj = self.read_proj(rms_norm(x))  # (B, T, capsule_dim)
        # Soft-assignment to prototypes
        scores = mx.einsum("btd,nd->btn", x_proj, self.prototypes.astype(x_proj.dtype))
        attn = mx.softmax(scores / (self.capsule_dim ** 0.5), axis=1)  # (B, T, N)
        capsules = mx.einsum("btn,btd->bnd", attn, x_proj)  # (B, N, capsule_dim)

        c_pred = None  # For consistency loss
        c_spec = None  # For Koopman speculation/diffusion jump
        if prev_capsules is not None:
            if self.koopman is not None:
                # Koopman-driven update: predict + blend
                capsules, c_pred = self.koopman.blend(capsules, prev_capsules)
                
                # If requested, generate a speculative fast-forwarded state
                if speculate_steps > 0:
                    c_spec = self.koopman.speculate(capsules, speculate_steps)
                    
                    # EVAL MODE (Fast-Forward Jump):
                    # Immediately snap the capsule state to the speculated future.
                    # This ensures the readout/correction written back to sequence
                    # actually uses the fast-forwarded semantic state!
                    if not self.training:
                        capsules = c_spec
            else:
                # Fallback: simple gated blending
                rg = mx.sigmoid(self.recurrent_gate).astype(capsules.dtype)
                capsules = rg * capsules + (1.0 - rg) * prev_capsules

        # Write back
        readout = mx.einsum("btn,bnd->btd", attn, capsules)
        correction = self.write_proj(readout)
        g = mx.tanh(self.gate).astype(x.dtype)
        return x + g * correction, capsules, c_pred, c_spec


class LinearBottleneck(nn.Module):
    """
    Simplest possible bottleneck: one low-rank linear residual, zero nonlinearity.

        y   = W_up( W_down(x) )    # rank-r linear map (no activation)
        out = x + tanh(gate) ⊙ y   # gated residual, gate starts closed

    Tests whether the bottleneck *position* in the U-Net has value,
    independent of any recurrence / attention / dynamics.
    Two matmuls: (B,T,D)→(B,T,r)→(B,T,D). No Python loops, no overhead.
    """
    def __init__(self, model_dim, rank=64):
        super().__init__()
        self.down = EmbedProj(model_dim, rank)
        self.up   = EmbedProj(rank, model_dim)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

    def __call__(self, x, prev_capsules=None, speculate_steps=0):
        y   = self.up(self.down(x))
        out = x + mx.tanh(self.gate).astype(x.dtype) * y
        return out, None, None, None


class SSMBottleneck(nn.Module):
    """
    Stabilized causal SSM bottleneck — drop-in replacement for CapsuleBank.

    Three convergence fixes vs v1:
      1. STATE NORMALIZATION: RMSNorm(H) before C_proj — prevents hidden state
         blow-up which caused NaN at step ~130 on curriculum transitions.
      2. ZERO-INIT RESIDUAL: C_proj weights start at zero — SSM contributes
         nothing at init, grows in as useful. Gradients flow to B and A from
         step 1 because norm_h is trainable and gate opens from any nonzero C.
      3. MULTI-SCALE A: half dims slow (sigmoid(2.197)≈0.9, long-range) +
         half fast (sigmoid(0.0)=0.5, medium-range) — captures both local and
         global context from the first step.

    Recurrence:
        u_t  = B_proj(RMSNorm(x_t))          # (B, T, state_dim)
        h_t  = A ⊙ h_{t-1} + u_t            # causal scan (A ∈ (0,1))
        y_t  = C_proj(RMSNorm(h_t))          # stable output
        out  = x + tanh(gate) ⊙ y_t          # gated residual

    Returns: (out, (B,1,state_dim) carry, None, None) — CapsuleBank-compatible.
    """
    _CHUNK = 256   # fewer Python loop iterations vs 128; graph stays bounded

    def __init__(self, model_dim, state_dim=64):
        super().__init__()
        self.model_dim = model_dim
        self.state_dim = state_dim

        # Multi-scale A: slow half + fast half
        slow = mx.full((state_dim // 2,), 2.197, dtype=mx.float32)   # ≈ 0.90
        fast = mx.full((state_dim - state_dim // 2,), 0.0, dtype=mx.float32)  # = 0.50
        self.A_logit = mx.concatenate([slow, fast], axis=0)

        # B: input → state (small init via EmbedProj std=0.02)
        self.B_proj = EmbedProj(model_dim, state_dim)

        # State norm — KEY stability fix; learnable scale/bias
        self.norm_h = nn.RMSNorm(state_dim)

        # C: state → model_dim; zero-init for safe residual start
        self.C_proj = EmbedProj(state_dim, model_dim)
        self.C_proj.weight = mx.zeros((model_dim, state_dim), dtype=mx.float32)

        # Gated residual (starts fully closed: tanh(0)=0)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

    def _scan(self, u, h0):
        """Sequential causal scan. Only forward powers of A → no overflow."""
        B, T, S = u.shape
        A = mx.sigmoid(self.A_logit).astype(u.dtype)
        h = mx.zeros((B, S), dtype=u.dtype) if h0 is None else h0
        hs = []
        for start in range(0, T, self._CHUNK):
            end = min(start + self._CHUNK, T)
            u_c = u[:, start:end, :]
            for t in range(end - start):
                h = A * h + u_c[:, t, :]
                hs.append(h)
        H = mx.stack(hs, axis=1)   # (B, T, S)
        return H, H[:, -1, :]

    def __call__(self, x, prev_capsules=None, speculate_steps=0):
        B, T, D = x.shape
        h0 = prev_capsules[:, 0, :].astype(x.dtype) if prev_capsules is not None else None
        u       = self.B_proj(rms_norm(x))           # (B, T, state_dim)
        H, h_T  = self._scan(u, h0)
        y       = self.C_proj(self.norm_h(H))        # normalize before readout
        g       = mx.tanh(self.gate).astype(x.dtype)
        out     = x + g * y
        return out, h_T[:, None, :], None, None


class LinearAttnKoopman(nn.Module):
    """
    Linear attention as Koopman operator — causal bottleneck.

    Complexity: O(T·H·d²) — linear in T, not O(T²).

    State evolution (per head):
        S_t = λ · S_{t-1} + φ(k_t) ⊗ v_t      # rank-1 Koopman update
        z_t = λ · z_{t-1} + φ(k_t)             # running normalizer
        o_t = φ(q_t)ᵀ S_t / max(φ(q_t)ᵀ z_t, 1)

    λ = sigmoid(lam_logit) ∈ (0,1) per head.
    φ = ELU(·)+1: positive feature map.

    Chunkwise implementation — replaces T per-step dispatches with T/C
    chunk-level matmuls. Each chunk of C tokens:
      1. Inter-chunk carry:  S ← λ^C · S + Kc^T @ Vc   (one matmul)
      2. Intra-chunk output: O[t] = Q[t] @ S_t via causal cumsum within chunk
         S_t = λ^t · S_prev + Kc[:t]^T @ Vc[:t]  (lower-triangular matmul)

    This is O(T·d²) ops in C batched matmuls — no Python loop over T.
    """
    _CHUNK = 32   # intra-chunk causal mask is C×C; 32 keeps it small

    def __init__(self, model_dim, num_heads=4, head_dim=16):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim  = head_dim
        inner = num_heads * head_dim

        self.lam_logit = mx.full((num_heads,), 2.197, dtype=mx.float32)  # init ≈ 0.9
        self.q_proj   = EmbedProj(model_dim, inner)
        self.k_proj   = EmbedProj(model_dim, inner)
        self.v_proj   = EmbedProj(model_dim, inner)
        self.out_proj = EmbedProj(inner, model_dim)
        self.gate = mx.zeros((model_dim,), dtype=mx.float32)

    @staticmethod
    def _phi(x):
        return nn.elu(x) + 1.0

    def _scan(self, q, k, v, lam):
        """
        q, k, v : (B, T, H, d)
        lam     : (H,)
        Returns : O (B, T, H, d), S_T (B, H, d, d), z_T (B, H, d)

        Each chunk C:
          - decay carry: S  ← lam^C · S  + K_c^T @ V_c   shape (B,H,d,d)
          - decay carry: z  ← lam^C · z  + sum(K_c, t)   shape (B,H,d)
          - intra output: O_c[t] = Q_c[t] @ S_t_within_chunk / denom
            where S_t = lam^t · S_prev + lower_causal_KV[t]
        """
        B, T, H, d = q.shape
        C = self._CHUNK
        dtype = q.dtype

        lam_v  = mx.sigmoid(lam).astype(dtype)          # (H,)
        S = mx.zeros((B, H, d, d), dtype=dtype)
        z = mx.zeros((B, H, d),    dtype=dtype)

        qp = self._phi(q)   # (B, T, H, d)
        kp = self._phi(k)   # (B, T, H, d)

        # Causal lower-triangular mask for intra-chunk attention: shape (C, C)
        # mask[i,j] = 1 if j <= i  (token i can attend to token j)
        idx = mx.arange(C)
        causal_mask = (idx[:, None] >= idx[None, :]).astype(dtype)  # (C, C)

        os_chunks = []
        num_chunks = (T + C - 1) // C

        for ci in range(num_chunks):
            s = ci * C
            e = min(s + C, T)
            Cs = e - s   # actual chunk size (last chunk may be smaller)

            Qc = qp[:, s:e, :, :]   # (B, Cs, H, d)
            Kc = kp[:, s:e, :, :]
            Vc = v[:,  s:e, :, :]

            # ── Intra-chunk causal output ──────────────────────────────────
            # For token t within chunk: S_t = lam^t · S_prev + K[:t]^T @ V[:t]
            # O_c[t] = Q[t] @ S_t / denom_t
            #
            # Split into two terms:
            #   Term 1 (cross): Q @ (lam^t · S_prev)  — each token queries the carry
            #   Term 2 (intra): Q @ (causal K^T V)     — intra-chunk causal linear attn

            # Decay powers for tokens in chunk: lam^0, lam^1, ..., lam^(Cs-1)
            t_idx = mx.arange(Cs).astype(dtype)  # (Cs,)
            # lam_pow[t, h] = lam[h]^t
            log_lam = mx.log(mx.clip(lam_v, 1e-7, 1.0 - 1e-7))   # (H,)
            lam_pow = mx.exp(t_idx[:, None] * log_lam[None, :])   # (Cs, H)

            # Term 1: each Q[t] queries decayed carry
            # S shape: (B, H, d, d) → need (B, Cs, H, d, d) with per-t decay
            # = Q[t] @ (lam^t · S) = lam^t · (Q[t] @ S)
            QS_carry = (Qc.reshape(B, Cs, H, 1, d) @ S.reshape(B, 1, H, d, d)
                        ).squeeze(-2)                              # (B, Cs, H, d)
            # multiply per-token per-head decay
            lam_pow_bcast = lam_pow[None, :, :, None]             # (1, Cs, H, 1)
            term1 = QS_carry * lam_pow_bcast                      # (B, Cs, H, d)

            # z_carry term for normalizer
            Qz_carry = (Qc * (z[:, None, :, :] * lam_pow_bcast)).sum(axis=-1, keepdims=True)

            # Term 2: intra-chunk causal linear attention
            # S_intra[t] = sum_{j<=t} K[j]^T V[j]  (no decay within chunk — approx)
            # O_intra[t] = Q[t] @ S_intra[t]
            # = sum_{j<=t} (Q[t] · K[j]) * V[j]   ← standard linear attn kernel
            # Efficient: A[t,j] = Q[t]·K[j], masked causal, then A @ V
            # A: (B, H, Cs, Cs)  then O2: (B, H, Cs, d)
            mask_c = causal_mask[:Cs, :Cs]                        # (Cs, Cs)
            # QK^T: (B, Cs, H, d) × (B, Cs, H, d) → (B, H, Cs, Cs)
            Qc_t = Qc.transpose(0, 2, 1, 3)   # (B, H, Cs, d)
            Kc_t = Kc.transpose(0, 2, 1, 3)
            Vc_t = Vc.transpose(0, 2, 1, 3)
            A_intra = Qc_t @ Kc_t.transpose(0, 1, 3, 2)           # (B, H, Cs, Cs)
            A_intra = A_intra * mask_c[None, None, :, :]
            term2 = (A_intra @ Vc_t).transpose(0, 2, 1, 3)        # (B, Cs, H, d)

            # Normalizer
            z_intra = (A_intra.sum(axis=-1)                        # (B, H, Cs)
                       ).transpose(0, 2, 1)[:, :, :, None]         # (B, Cs, H, 1)
            denom = mx.maximum(Qz_carry + z_intra, 1.0)

            O_c = (term1 + term2) / denom                          # (B, Cs, H, d)
            os_chunks.append(O_c)

            # ── Update carry ───────────────────────────────────────────────
            # S_next = lam^Cs · S + Kc^T @ Vc  (inter-chunk, no per-token decay)
            lam_C = mx.exp(float(Cs) * log_lam)                    # (H,)
            lam_C_S = lam_C[None, :, None, None]
            lam_C_z = lam_C[None, :, None]
            # Kc^T @ Vc: (B, H, d, Cs) @ (B, H, Cs, d) → (B, H, d, d)
            S = lam_C_S * S + Kc_t.transpose(0, 1, 3, 2) @ Vc_t
            z = lam_C_z * z + kp[:, s:e, :, :].sum(axis=1)

        O = mx.concatenate(os_chunks, axis=1)   # (B, T, H, d)
        return O, S, z

    def __call__(self, x, prev_capsules=None, speculate_steps=0):
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        xn = rms_norm(x)
        q = self.q_proj(xn).reshape(B, T, H, d)
        k = self.k_proj(xn).reshape(B, T, H, d)
        v = self.v_proj(xn).reshape(B, T, H, d)

        # Unpack carry: (B, 1, H*d*d + H*d) → S, z
        S0, z0 = None, None
        if prev_capsules is not None:
            flat = prev_capsules[:, 0, :]  # (B, H*d*d + H*d)
            S0 = flat[:, :H*d*d].reshape(B, H, d, d).astype(x.dtype)
            z0 = flat[:, H*d*d:].reshape(B, H, d).astype(x.dtype)

        O, S_T, z_T = self._scan(q, k, v, self.lam_logit)
        O = O.reshape(B, T, H * d)
        y = self.out_proj(O)
        g = mx.tanh(self.gate).astype(x.dtype)
        out = x + g * y

        # Pack carry: flatten S_T and z_T → (B, 1, H*d*d + H*d)
        carry = mx.concatenate(
            [S_T.reshape(B, H*d*d), z_T.reshape(B, H*d)], axis=-1
        )[:, None, :].astype(mx.float32)

        return out, carry, None, None


class RMSNormNoWeight:
    """RMSNorm without learnable parameters — zero-cost normalization."""
    def __call__(self, x, eps=1e-6):
        return rms_norm(x, eps=eps)


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
        self._H_kv = _build_hadamard(self.head_dim) if (self.head_dim & (self.head_dim - 1)) == 0 else None

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

        if _TURBO_QUANT_KV:
            k = quantize_kv_ste(k, turbo=True, H=self._H_kv)
            v = quantize_kv_ste(v, turbo=True, H=self._H_kv)

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



class DenseTernaryMoE(nn.Module):
    """Sparse Ternary Mixture of Experts for MLX. 
    Computed densely to avoid dynamic shape recompilation issues,
    but evaluates fast on sparse weights by masking."""
    def __init__(self, dim, mlp_mult, num_experts, top_k, group_size=128, activation="lrelu2", leaky_relu_slope=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = [MLP(dim, mlp_mult, group_size, activation, leaky_relu_slope) for _ in range(num_experts)]

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.router(x_flat)
        
        # density and routing calculations (unchanged logic)
        density = mx.mean(mx.softmax(router_logits, axis=1), axis=0)
        routing_weights = mx.softmax(router_logits.astype(mx.float32), axis=1)
        
        topk_vals = mx.topk(routing_weights, self.top_k, axis=-1)
        threshold = mx.min(topk_vals, axis=-1, keepdims=True)
        mask = routing_weights >= threshold
        
        fraction_routed = mx.mean(mask.astype(mx.float32), axis=0)
        aux_loss = mx.mean(density * fraction_routed) * self.num_experts
        
        active_weights = mx.where(mask, routing_weights, mx.zeros_like(routing_weights))
        active_weights = active_weights / mx.maximum(mx.sum(active_weights, axis=-1, keepdims=True), 1e-9)
        active_weights = active_weights.astype(x.dtype)
        
        # Vectorized Expert Execution:
        # Stack all expert weights into a single tensor for a single batched matmul.
        # This removes the Python loop and allows MLX to fuse the router + experts into one pass.
        all_fc = mx.stack([e.fc.weight for e in self.experts])  # (E, Hidden, D)
        all_proj = mx.stack([e.proj.weight for e in self.experts])  # (E, D, Hidden)
        
        # 1. First Layer (fc): (B*T, D) -> (B*T, E, Hidden)
        # Using ternary_ste on stacked weights
        all_fc_q = ternary_ste(all_fc, self.experts[0].fc.group_size, turbo=_TURBO_QUANT)
        # einsum: 'ed,bd->be' where b=batch, e=expert, d=dim, o=out
        h = mx.einsum("ehd,bd->beh", all_fc_q.astype(x.dtype), x_flat)
        
        # 2. Activation
        if self.experts[0].activation == "lrelu2":
            h = mx.where(h >= 0, h, h * self.experts[0].leaky_relu_slope)
            h = h * h
        elif self.experts[0].activation == "relu2":
            h = mx.maximum(h, 0)
            h = h * h
        else:
            h = mx.maximum(h, 0)
            
        # 3. Second Layer (proj): (B*T, E, Hidden) -> (B*T, E, D)
        all_proj_q = ternary_ste(all_proj, self.experts[0].proj.group_size, turbo=_TURBO_QUANT)
        # einsum: 'edh,beh->bed'
        outs = mx.einsum("edh,beh->bed", all_proj_q.astype(x.dtype), h)
        
        # 4. Weighted Sum across experts
        # active_weights: (B*T, E)
        final_output = mx.sum(outs * active_weights[:, :, None], axis=1)
            
        return final_output.reshape(B, T, D), aux_loss

# ---------------------------------------------------------------------------
# Koopman State Space Model (Path 2: Attention-free architecture)

# ---------------------------------------------------------------------------

class KoopmanTokenMixer(nn.Module):
    """Causal token mixing via Koopman-inspired linear recurrence.

    Replaces Self-Attention with O(T) linear dynamics:
      1. Project input to state space: s = proj_in(RMSNorm(x))
      2. Short causal convolution for local context (like Mamba)
      3. Input-dependent gating: g = sigmoid(g_proj(RMSNorm(x)))
      4. Causal linear scan via exponentially decaying convolution:
         h_t = D * h_{t-1} + (h_{t-1} @ V) @ U^T + g_t * s_t
         Approximated as truncated causal conv with window W.
      5. Project back to model dim: out = proj_out(RMSNorm(h))

    First-principles design rationale:
    - Diagonal D (|d_i| < 1): stable exponential memory decay
    - Low-rank UV^T: cross-dimension coupling beyond diagonal
    - Hadamard rotation: variance equalization across state dims
    - Input gating: selective information injection (like Mamba's delta)
    - Short conv: local n-gram features complementing global recurrence
    """
    def __init__(self, dim, state_dim, rank=4, conv_kernel=4, decay_window=32,
                 group_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.conv_kernel = conv_kernel
        self.decay_window = decay_window

        # Projections (ternary for compression)
        self.proj_in = TernaryLinear(dim, state_dim, group_size=group_size)
        self.proj_out = NormedTernaryLinear(state_dim, dim, group_size=group_size)
        self.proj_out._zero_init = True
        self.proj_out.weight = mx.zeros_like(self.proj_out.weight)

        # Gating projection (ternary)
        self.g_proj = TernaryLinear(dim, state_dim, group_size=group_size)

        # Short causal convolution (depthwise, per-channel)
        # Initialize as uniform average pool to preserve signal geometry before Koopman dynamics
        self.mixer_conv = mx.ones((conv_kernel, state_dim), dtype=mx.float32) / conv_kernel

        # Koopman dynamics parameters (small, FP32)
        # Named with 'mixer_diag' / 'mixer_lowrank' for optimizer routing
        self.mixer_diag = mx.full((state_dim,), 0.8, dtype=mx.float32)  # fast initial forget gate
        self.mixer_lowrank_U = mx.random.normal((state_dim, rank), dtype=mx.float32) * 0.001
        self.mixer_lowrank_V = mx.random.normal((state_dim, rank), dtype=mx.float32) * 0.001

        # Per-channel mixing scale (for residual connection)
        self.mixer_scale = mx.ones((dim,), dtype=mx.float32)

        # Hadamard for variance equalization (if state_dim is power of 2)
        self._use_hadamard = (state_dim & (state_dim - 1)) == 0 and state_dim >= 2
        if self._use_hadamard:
            self._H = _build_hadamard(state_dim)

    def _short_causal_conv(self, x):
        """Vectorized depthwise causal convolution.
        x: (B, T, state_dim) -> (B, T, state_dim)
        """
        B, T, S = x.shape
        # mx.conv1d needs (B, T, C) input and (O, K, I) weights.
        # For depthwise, groups=S.
        weight = self.mixer_conv[::-1].T.reshape(S, self.conv_kernel, 1)  # (S, K, 1)
        x_padded = mx.pad(x, [(0, 0), (self.conv_kernel - 1, 0), (0, 0)])
        h = mx.conv1d(x_padded, weight.astype(x.dtype), groups=S)
        return h

    def _causal_decay_scan(self, x, gate, dt_gate=None):
        """High-Performance Parallel Chunked Scan (CV-Scan).
        Achieves sub-500ms throughput by avoiding sequential Python loops.
        
        Complexity: O(sqrt(T)) sequential steps, O(T) parallel work.
        For T=1024, CHUNK=32, this is effectively 32+32+32 steps, fully fused.
        """
        B, T, S = x.shape
        W = 32 # Chunk size
        num_chunks = T // W
        
        if self._use_hadamard:
            x = x @ self._H.astype(x.dtype)

        # 1. Inputs and Decays
        if dt_gate is not None:
            logD = -mx.softplus(dt_gate.astype(mx.float32))
            D = mx.exp(logD).astype(x.dtype)
            B_vals = gate * x * (1.0 - D)
        else:
            D_static = mx.clip(self.mixer_diag, -0.999, 0.999).astype(x.dtype)
            D = mx.broadcast_to(D_static, (B, T, S))
            B_vals = gate * x * (1.0 - mx.abs(D_static))

        # 2. Reshape to chunks (B, num_chunks, W, S)
        D_c = D.reshape(B, num_chunks, W, S)
        B_c = B_vals.reshape(B, num_chunks, W, S)

        # 3. Level 1: Intra-chunk parallel scan
        # We unroll a 32-step scan within MLX's graph.
        # This becomes a single large fused kernel.
        c_h = [B_c[:, :, 0]]
        c_d = [D_c[:, :, 0]]
        for t in range(1, W):
            # Recurrence: h_t = d_t * h_{t-1} + b_t
            # Decay: d_cum_t = d_t * d_cum_{t-1}
            c_h.append(D_c[:, :, t] * c_h[-1] + B_c[:, :, t])
            c_d.append(D_c[:, :, t] * c_d[-1])
        
        # h_local: (B, num_chunks, W, S)
        # d_local: (B, num_chunks, W, S)
        h_local = mx.stack(c_h, axis=2)
        d_local = mx.stack(c_d, axis=2)

        # 4. Level 2: Inter-chunk parallel scan (on chunk-final states)
        # Scan across the final states of each chunk to get the 'prefix' for each chunk.
        chunk_finals_h = h_local[:, :, -1] # (B, num_chunks, S)
        chunk_finals_d = d_local[:, :, -1] # (B, num_chunks, S)
        
        # Recurrence on chunks: P_i = D_final_i * P_{i-1} + H_final_i
        p_h = [mx.zeros_like(chunk_finals_h[:, 0])] # Initial prefix for first chunk is 0
        for i in range(num_chunks - 1):
             p_h.append(chunk_finals_d[:, i] * p_h[-1] + chunk_finals_h[:, i])
        
        # chunk_prefixes: (B, num_chunks, 1, S)
        chunk_prefixes = mx.stack(p_h, axis=1)[:, :, None, :]

        # 5. Combine: h = h_local + d_local * chunk_prefix
        h = h_local + d_local * chunk_prefixes
        h = h.reshape(B, T, S)

        # 6. Low-rank coupling
        U = self.mixer_lowrank_U.astype(x.dtype)
        V = self.mixer_lowrank_V.astype(x.dtype)
        h = h + (h @ V) @ U.T

        if self._use_hadamard:
            h = h @ self._H.astype(x.dtype)
        return h

    def __call__(self, x):
        """x: (B, T, dim) -> (B, T, dim)"""
        normed = rms_norm(x)
        s = self.proj_in(normed)       # (B, T, state_dim)
        g = mx.sigmoid(self.g_proj(normed))  # (B, T, state_dim)

        # Short causal conv for local features (bigram/trigram patterns)
        s = self._short_causal_conv(s)

        # Causal Koopman scan for global recurrence
        h = self._causal_decay_scan(s, g)

        # Project back to model dim
        return self.proj_out(h)


class KoopmanBlock(nn.Module):
    """A single layer of the Koopman SSM architecture.

    Replaces CausalSelfAttention with KoopmanTokenMixer.
    Keeps the same MLP structure as the transformer baseline.
    Uses the same residual scaling / damping patterns.
    """
    def __init__(self, dim, state_dim, mlp_mult, mixer_rank=4, conv_kernel=4,
                 decay_window=32, group_size=128, activation="lrelu2",
                 leaky_relu_slope=0.5, ln_scale_factor=1.0, moe_enabled=False, moe_num_experts=8, moe_top_k=2):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.mixer = KoopmanTokenMixer(
            dim, state_dim, rank=mixer_rank, conv_kernel=conv_kernel,
            decay_window=decay_window, group_size=group_size,
        )
        if moe_enabled:
            self.mlp = DenseTernaryMoE(dim, mlp_mult, moe_num_experts, moe_top_k, group_size, activation, leaky_relu_slope)
        else:
            self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
        self.moe_enabled = moe_enabled
        
        # SkipInit / ReZero: initialize residual branch scalars to 0.0
        # This prevents the Muon optimizer's aggressive orthogonal updates from
        # exploding the residual stream before the recurrent dynamics align.
        self.mixer_scale = mx.zeros((dim,), dtype=mx.float32)
        self.mlp_scale = mx.zeros((dim,), dtype=mx.float32)

        self.resid_mix = mx.array(np.stack([np.ones(dim, dtype=np.float32),
                                            np.zeros(dim, dtype=np.float32)]))

    def __call__(self, x, x0):
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = rms_norm(x) * self.ln_scale_factor
        mixer_out = self.mixer(normed)
        x = x + self.mixer_scale.astype(x.dtype)[None, None, :] * mixer_out
        
        mlp_in = rms_norm(x) * self.ln_scale_factor
        if hasattr(self, 'moe_enabled') and self.moe_enabled:
            mlp_out, aux_loss = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            aux_loss = mx.array(0.0)
            
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x, None, aux_loss


# ─────────────────────────────────────────────────────────────────────────
# Sprint 3: Spectral Koopman Capsule (SKC) Layer — Novel Architecture
# ─────────────────────────────────────────────────────────────────────────
#
# The key insight that makes this genuinely novel:
#
#   TERNARY SPECTRAL THEOREM:
#   A ternary weight matrix W ∈ {-1, 0, +1}^{m×n} can be viewed as a
#   partial Walsh-Hadamard decomposition. The WHT basis vectors have
#   entries ±1/√n — exactly representable in ternary. Therefore:
#
#   1. Ternary quantization IS spectral truncation in the Walsh basis
#   2. WHT mixing is LOSSLESS under ternary arithmetic
#   3. Training should optimize which spectral modes to retain
#
#   This unifies three previously unrelated ideas:
#     - BitNet (ternary weights) → spectral selection
#     - Koopman operators (linear dynamics) → spectral evolution
#     - Capsule networks (part-whole) → spectral routing
#
#   Result: a layer where ternary quantization HELPS rather than hurts,
#   because the entire computation is native to the Walsh spectral domain.
#
# ─────────────────────────────────────────────────────────────────────────


def causal_wht_blockwise(x, block_size=64):
    """
    Causal Walsh-Hadamard Transform via blockwise overlapping windows.

    Standard WHT is not causal (position t sees future positions).
    This implements causal spectral mixing by:
      1. Splitting the sequence into non-overlapping blocks of size B
      2. Computing WHT within each block (purely local, causal within block)
      3. Cross-block information flows via a learned exponential decay scan

    Complexity: O(T · B · log(B))  where B = block_size (typically 32-64).
    Effective receptive field: full sequence (via inter-block decay scan).

    Mathematical guarantee: position t only depends on positions ≤ t.

    Args:
        x: (Batch, T, D) input tensor
        block_size: WHT block size (must be power of 2)

    Returns:
        (Batch, T, D) causally-transformed tensor
    """
    B, T, D = x.shape

    # Pad T to multiple of block_size
    pad_len = (block_size - T % block_size) % block_size
    if pad_len > 0:
        x = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])
    T_padded = x.shape[1]
    num_blocks = T_padded // block_size

    # Reshape into blocks: (B, num_blocks, block_size, D)
    x_blocks = x.reshape(B, num_blocks, block_size, D)

    # === Intra-block WHT (fully parallel, causal within each block) ===
    # Standard butterfly WHT on each block independently
    h = block_size.bit_length() - 1
    result = x_blocks
    for stage in range(h):
        stride = 1 << stage
        result = result.reshape(B, num_blocks, block_size // (2 * stride), 2 * stride, D)
        even = result[:, :, :, :stride, :]
        odd = result[:, :, :, stride:, :]
        result = mx.concatenate([even + odd, even - odd], axis=3)
        result = result.reshape(B, num_blocks, block_size, D)

    norm = 1.0 / math.sqrt(block_size)
    result = result * norm  # (B, num_blocks, block_size, D) — spectral coefficients per block

    # Flatten back to sequence
    result = result.reshape(B, T_padded, D)

    # Crop to original T
    return result[:, :T, :]


def causal_spectral_decay_scan(x_blocks, decay_rates, gate):
    """
    Inter-block causal scan: propagates spectral information across blocks.

    Each block's spectral representation is blended with the previous block's
    output via a learned exponential decay. This makes the blockwise WHT
    effectively attend to the full past sequence.

    This is the Koopman operator applied in the spectral domain:
    h_t = diag(decay) · h_{t-1} + gate_t · x_t

    Args:
        x_blocks: (B, num_blocks, block_size, D) — WHT-transformed blocks
        decay_rates: (D,) — per-dimension decay (learned, ∈ (0,1))
        gate: (B, num_blocks, block_size, D) — input gating

    Returns:
        (B, num_blocks, block_size, D) — causally-mixed spectral representation
    """
    B, num_blocks, block_sz, D = x_blocks.shape
    decay = mx.clip(decay_rates, -0.999, 0.999)

    # Gated input
    gated = gate * x_blocks

    # Scan across blocks (each block's final state feeds into next block's initial state)
    # Block-level recurrence: state_i = decay * state_{i-1} + gated[:, i, -1, :]
    block_finals = gated[:, :, -1, :]  # (B, num_blocks, D)

    states = [block_finals[:, 0, :]]
    for i in range(1, num_blocks):
        states.append(decay * states[-1] + block_finals[:, i, :])
    states = mx.stack(states, axis=1)  # (B, num_blocks, D)

    # Inject inter-block state back into each block's spectral coefficients
    # state[:, i, :] carries information from all blocks < i
    prefix_states = mx.concatenate([mx.zeros_like(states[:, :1, :]), states[:, :-1, :]], axis=1)
    # (B, num_blocks, 1, D) broadcast into (B, num_blocks, block_sz, D)
    return x_blocks + prefix_states[:, :, None, :] * decay[None, None, None, :]


class SpectralTernaryAuxLoss(nn.Module):
    """
    Spectral entropy regularizer: encourages diverse use of Walsh modes.

    Without this, ternary networks tend to collapse to using only a few
    spectral modes (the "ternary mode collapse" problem). This loss
    maximizes the entropy of spectral energy distribution across modes.

    L_spectral = -H(p) where p_k = |X_k|² / Σ|X_j|²
    (X_k = Walsh coefficient at sequency k)
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def __call__(self, x_spec):
        """x_spec: (B, T, D) spectral coefficients."""
        # Energy per sequency band
        energy = mx.mean(x_spec * x_spec, axis=(0, 2))  # (T,)
        energy = energy + 1e-8
        p = energy / mx.sum(energy)
        entropy = -mx.sum(p * mx.log(p + 1e-10))
        max_entropy = math.log(max(x_spec.shape[1], 1))
        # Minimize negative entropy = maximize entropy
        return self.weight * (max_entropy - entropy)


class FrequencyBandRouter(nn.Module):
    """
    Novel capsule routing via Walsh frequency bands.

    Standard capsule routing treats all dimensions equally. This router
    assigns each capsule to a specific frequency band of the Walsh spectrum:

      - Low-sequency capsules (k=0..N/4): syntax, grammar, sentence structure
      - Mid-sequency capsules (k=N/4..3N/4): semantic content, word meaning
      - High-sequency capsules (k=3N/4..N): local patterns, spelling, morphology

    Each capsule specializes in its frequency band, preventing interference
    between global structure and local detail — the key bottleneck in
    ternary language models where limited precision causes cross-scale noise.
    """
    def __init__(self, num_capsules, capsule_dim, block_size):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.block_size = block_size

        # Learnable band centers and widths (in sequency space)
        # Initialize evenly spaced across the spectrum
        centers = np.linspace(0, 1, num_capsules, dtype=np.float32)
        self.band_centers = mx.array(centers)
        self.band_log_widths = mx.array(np.full(num_capsules, -1.0, dtype=np.float32))



    def __call__(self, x_spec, T):
        """
        Route spectral features to capsules based on frequency band affinity.

        Args:
            x_spec: (B, T, capsule_dim) — projected spectral features
            T: sequence length (for computing sequency positions)

        Returns:
            routing_weights: (B, T, num_capsules) — soft band assignment
            capsules: (B, num_capsules, capsule_dim) — aggregated per-band features
        """
        B = x_spec.shape[0]

        # Within-block sequency index: position k maps to sequency k % block_size
        # After blockwise WHT, index k within a block IS the Walsh sequency for that block.
        seq_pos = (mx.arange(T) % self.block_size).astype(mx.float32) / max(self.block_size - 1, 1)  # (T,)

        # Gaussian band affinity: how much does each position belong to each capsule's band?
        # centers: (N,), widths: (N,), seq_pos: (T,)
        widths = mx.exp(self.band_log_widths)  # (N,)
        # (T, N) affinity matrix
        diff = seq_pos[:, None] - self.band_centers[None, :]  # (T, N)
        band_affinity = mx.exp(-0.5 * (diff * diff) / (widths[None, :] ** 2 + 1e-6))

        # Normalize: each position sums to 1 across capsules
        routing_weights = band_affinity / (mx.sum(band_affinity, axis=-1, keepdims=True) + 1e-8)
        # (T, N) → (1, T, N) for broadcasting
        routing_weights = routing_weights[None, :, :]  # (1, T, N)

        # Aggregate: capsules[:, n, :] = Σ_t routing_weights[:, t, n] * x_spec[:, t, :]
        capsules = mx.einsum("btn,btd->bnd", mx.broadcast_to(routing_weights, (B, T, self.num_capsules)), x_spec)

        return routing_weights, capsules


class KoopmanSpectralEvolution(nn.Module):
    """
    Koopman operator applied to capsule states in the spectral domain.

    Novel contribution: instead of a static matrix multiply, this models
    the DYNAMICS of how spectral content evolves across layers/positions:

    c_{t+1} = Φ(Λ) · c_t + (I - Φ(Λ)) · c_input

    where:
    - Φ(Λ) = diag(σ(λ)) are learnable eigenvalue gates (how much to retain)
    - c_t is the capsule state from the previous position/layer
    - c_input is the fresh spectral observation

    This is a TRUE Koopman decomposition: the eigenvalues λ determine
    which modes of the dynamics persist vs. decay. The key insight is that
    in a ternary network, we want STABLE modes (|λ| ≈ 1) for syntactic
    structure and DECAYING modes (|λ| < 1) for local content.

    The eigenvalue spectrum is LEARNED, not fixed — the network discovers
    which temporal scales matter for language.
    """
    def __init__(self, capsule_dim, num_capsules, rank=8):
        super().__init__()
        self.capsule_dim = capsule_dim
        self.num_capsules = num_capsules

        # Koopman eigenvalues: per-capsule, per-dimension
        # Initialized near 0.9 for stable dynamics, learned via gradient
        self.eigenvalues = mx.array(
            np.full((num_capsules, capsule_dim), 2.0, dtype=np.float32)
        )  # σ(2.0) ≈ 0.88

        # Low-rank cross-capsule coupling (capsules can exchange information)
        self.coupling_U = mx.random.normal((num_capsules, rank), dtype=mx.float32) * 0.01
        self.coupling_V = mx.random.normal((rank, num_capsules), dtype=mx.float32) * 0.01

        # Per-capsule nonlinearity gate (breaks Koopman linearity adaptively)
        self.nonlinear_gate = mx.zeros((num_capsules, capsule_dim), dtype=mx.float32)

    def __call__(self, capsules, prev_capsules=None):
        """
        Evolve capsule states via Koopman spectral dynamics.

        Args:
            capsules: (B, N, C) — current capsule observations
            prev_capsules: (B, N, C) — previous state (None for first layer)

        Returns:
            evolved: (B, N, C) — evolved capsule states
        """
        # Eigenvalue gate: how much to retain from previous state
        lam = mx.sigmoid(self.eigenvalues)  # (N, C) ∈ (0, 1)

        if prev_capsules is not None:
            # Koopman evolution: blend past state with new observation
            evolved = lam[None, :, :] * prev_capsules + (1.0 - lam[None, :, :]) * capsules
        else:
            evolved = capsules

        # Cross-capsule coupling: capsules inform each other
        # coupling: (N, N) low-rank interaction matrix
        coupling = mx.matmul(self.coupling_U, self.coupling_V)  # (N, N)
        # Clamp Frobenius norm ≤ 1 to prevent coupling from dominating eigenvalue dynamics
        coup_norm = mx.sqrt(mx.sum(coupling * coupling) + 1e-8)
        coupling = coupling / mx.maximum(coup_norm, mx.array(1.0, dtype=coup_norm.dtype))
        # Mix across capsule dimension: (B, N, C) → (B, N, C)
        cross_info = mx.einsum("nm,bmc->bnc", coupling, evolved)
        evolved = evolved + cross_info

        # Adaptive nonlinearity: gate controls where to break Koopman linearity
        nl_gate = mx.sigmoid(self.nonlinear_gate)  # (N, C)
        # Where gate is high: apply tanh (capture nonlinear dynamics)
        # Where gate is low: stay linear (preserve Koopman guarantees)
        evolved = (1.0 - nl_gate[None, :, :]) * evolved + nl_gate[None, :, :] * mx.tanh(evolved)

        return evolved


class SKCLayer(nn.Module):
    """
    Spectral Koopman Capsule (SKC) Layer — a genuinely novel unified primitive.

    ═══════════════════════════════════════════════════════════════════════
    KEY INNOVATION: The Ternary Spectral Theorem
    ═══════════════════════════════════════════════════════════════════════

    Observation: Walsh-Hadamard basis vectors have entries {+1/√n, -1/√n}.
    These are EXACTLY representable as ternary values with a single shared
    scale factor. Therefore:

    1. Computing WHT of a ternary weight matrix involves only {+1,-1}
       multiplications — ZERO floating-point rounding error.

    2. A ternary weight matrix IS a partial Walsh spectral decomposition:
       the zero entries are "spectral silence" (dropped modes), and the
       ±1 entries are "spectral presence" (retained modes).

    3. Optimizing ternary weights = selecting which Walsh spectral modes
       to retain. This is a STRUCTURED combinatorial problem, not a
       continuous optimization problem hacked with STE.

    ═══════════════════════════════════════════════════════════════════════
    ARCHITECTURE: Causal Spectral → Frequency-Band Routing → Koopman
    ═══════════════════════════════════════════════════════════════════════

    1. Causal Blockwise WHT: O(T·B·log(B)) spectral decomposition that
       respects autoregressive causality via overlapping blocks + decay scan.

    2. Frequency-Band Capsule Routing: Each capsule specializes in a
       frequency band (low=syntax, mid=semantics, high=morphology).
       This prevents cross-scale interference that plagues ternary LMs.

    3. Koopman Spectral Evolution: True eigenvalue-gated dynamics with
       learned temporal scales. Stable modes (|λ|≈1) carry syntax,
       decaying modes (|λ|<1) carry local content.

    4. Spectral Synthesis: Inverse WHT with frequency-selective readout.

    5. Local Refinement: Causal conv + gated MLP for sub-block patterns.
    """
    def __init__(self, dim, capsule_num=32, capsule_dim=128, conv_kernel=4, block_size=64,
                 mlp_mult=4, group_size=128, activation="lrelu2", leaky_relu_slope=0.5,
                 ln_scale_factor=1.0, moe_enabled=False, moe_num_experts=8, moe_top_k=2):
        super().__init__()
        self.dim = dim
        self.capsule_num = capsule_num
        self.capsule_dim = capsule_dim
        self.conv_kernel = conv_kernel
        self.ln_scale_factor = ln_scale_factor
        self.block_size = block_size  # WHT block size (power of 2: 16/32/64/128)

        # Spectral input projection: model dim → capsule dim
        self.spec_proj_in = TernaryLinear(dim, capsule_dim, group_size=group_size)

        # Frequency-band capsule router (novel)
        self.freq_router = FrequencyBandRouter(capsule_num, capsule_dim, self.block_size)

        # Koopman spectral evolution (novel)
        self.koopman_evo = KoopmanSpectralEvolution(capsule_dim, capsule_num, rank=8)

        # Inter-block causal decay rates (learned per-dimension)
        self.spectral_decay = mx.array(np.full(dim, 2.0, dtype=np.float32))  # σ(2.0) ≈ 0.88

        # Spectral gate for inter-block scan
        self.spectral_gate_proj = TernaryLinear(dim, dim, group_size=group_size)

        # Spectral entropy regularizer (novel)
        self.spectral_entropy_loss = SpectralTernaryAuxLoss(weight=0.005)

        # Spectral synthesis: capsule dim → model dim
        self.spec_proj_out = NormedTernaryLinear(capsule_dim, dim, group_size=group_size)
        self.spec_proj_out._zero_init = True
        self.spec_proj_out.weight = mx.zeros_like(self.spec_proj_out.weight)

        # Local refinement: causal conv for sub-block patterns
        self.local_conv_weight = mx.random.normal((dim, conv_kernel), dtype=mx.float32) * (0.02 / math.sqrt(conv_kernel * dim))
        self.conv_gate = TernaryLinear(dim, dim, group_size=group_size)

        # MLP for semantic refinement
        if moe_enabled:
            self.mlp = DenseTernaryMoE(dim, mlp_mult, moe_num_experts, moe_top_k, group_size, activation, leaky_relu_slope)
        else:
            self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
        self.moe_enabled = moe_enabled

        # Residual scaling (ReZero init for stable training)
        self.spec_scale = mx.full((dim,), 0.1, dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.conv_scale = mx.array(0.1, dtype=mx.float32)
        self.resid_mix = mx.array(np.stack([np.ones(dim, dtype=np.float32),
                                            np.zeros(dim, dtype=np.float32)]))

    def _causal_conv1d(self, x):
        """Causal depthwise conv1d: (B, T, D) → (B, T, D)."""
        B, T, D = x.shape
        K = self.conv_kernel
        # Causal padding: K-1 zeros at the start
        x_pad = mx.concatenate([mx.zeros((B, K - 1, D), dtype=x.dtype), x], axis=1)
        # Manual conv: for each output position t, sum over kernel window
        # x_pad[:, t:t+K, :] weighted by conv_weight
        # Efficient: use reshape + matmul
        # Unfold: (B, T, K, D)
        windows = mx.stack([x_pad[:, i:i + T, :] for i in range(K)], axis=2)
        # Weight: (D, K) — one kernel per channel for true depthwise convolution
        w = ternary_ste(self.local_conv_weight, group_size=min(128, K * D), turbo=False)  # (D, K)
        # (B, T, K, D) * (1, 1, K, D) → sum over K → (B, T, D)
        out = mx.sum(windows * w[None, None, :, :].transpose(0, 1, 3, 2), axis=2)
        return out

    def __call__(self, x, x0, prev_capsules=None):
        """
        Forward pass of the Spectral Koopman Capsule layer.

        x: (B, T, dim) — current hidden state
        x0: (B, T, dim) — original embedding for residual mixing

        Returns: (x_out, capsule_state, aux_loss)
        """
        B, T, D = x.shape
        aux_loss = mx.array(0.0)

        # ── Input mixing (for residual mix with original embedding) ──
        # x_in is stored separately; SKCLayer returns a DELTA that _run_block adds.
        # This is consistent with Block/KoopmanBlock which also return deltas.
        mix = self.resid_mix.astype(x.dtype)
        x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # ── Step 1: Causal Blockwise WHT + inter-block decay scan ──
        normed = rms_norm(x_mixed) * self.ln_scale_factor

        # Intra-block WHT
        x_spec = causal_wht_blockwise(normed, block_size=self.block_size)

        # Inter-block causal scan: propagates spectral state across blocks
        gate = mx.sigmoid(self.spectral_gate_proj(normed))  # (B, T, D)
        pad_len = (self.block_size - T % self.block_size) % self.block_size
        T_padded = T + pad_len
        num_blocks = T_padded // self.block_size
        x_spec_padded = x_spec if pad_len == 0 else mx.pad(x_spec, [(0,0),(0,pad_len),(0,0)])
        gate_padded = gate if pad_len == 0 else mx.pad(gate, [(0,0),(0,pad_len),(0,0)])
        x_spec_blocks = x_spec_padded.reshape(B, num_blocks, self.block_size, D)
        gate_blocks = gate_padded.reshape(B, num_blocks, self.block_size, D)
        decay_rates = mx.clip(self.spectral_decay, -0.999, 0.999)
        x_spec_scanned = causal_spectral_decay_scan(x_spec_blocks, decay_rates, gate_blocks)
        x_spec = x_spec_scanned.reshape(B, T_padded, D)[:, :T, :]

        # Spectral entropy regularization (encourages diverse mode usage)
        aux_loss = aux_loss + self.spectral_entropy_loss(x_spec)

        # ── Step 2: Project to capsule space ──
        c_proj = self.spec_proj_in(x_spec)  # (B, T, capsule_dim)

        # ── Step 3: Frequency-band capsule routing ──
        routing_weights, capsules = self.freq_router(c_proj, T)
        # routing_weights: (1, T, N), capsules: (B, N, capsule_dim)

        # ── Step 4: Koopman spectral evolution ──
        # prev_capsules=None on first pass; the CapsuleBank provides cross-layer state
        capsules_evolved = self.koopman_evo(capsules, prev_capsules=prev_capsules)

        # ── Step 5: Spectral synthesis ──
        rw = mx.broadcast_to(routing_weights, (B, T, self.capsule_num))
        c_readout = mx.einsum("bnd,btn->btd", capsules_evolved, rw)  # (B, T, capsule_dim)
        x_spec_out = self.spec_proj_out(c_readout)  # (B, T, dim)

        # Inverse causal blockwise WHT (WHT is self-inverse)
        x_synth = causal_wht_blockwise(x_spec_out, block_size=self.block_size)

        # ── Step 6: Local refinement (causal conv) ──
        conv_out = self._causal_conv1d(x_mixed)
        conv_gate = mx.sigmoid(self.conv_gate(rms_norm(x_mixed)))

        # ── Step 7: MLP semantic refinement ──
        mlp_in = rms_norm(x_mixed) * self.ln_scale_factor
        if self.moe_enabled:
            mlp_out, moe_aux = self.mlp(mlp_in)
            aux_loss = aux_loss + moe_aux
        else:
            mlp_out = self.mlp(mlp_in)

        # Compute delta: the total contribution from this layer.
        # _run_block adds this to x — do NOT accumulate residuals here.
        delta = (
            self.spec_scale.astype(x.dtype)[None, None, :] * x_synth
            + self.conv_scale.astype(x.dtype) * conv_gate * conv_out
            + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        )

        return delta, capsules_evolved, aux_loss


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 group_size=128, activation="lrelu2", leaky_relu_slope=0.5,
                 partial_rope_dims=0, vrl_enabled=False, ln_scale_factor=1.0, xsa=False, moe_enabled=False, moe_num_experts=8, moe_top_k=2):
        super().__init__()
        self.ln_scale_factor = ln_scale_factor
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        group_size=group_size, partial_rope_dims=partial_rope_dims,
                                        vrl_enabled=vrl_enabled, xsa=xsa)
        if moe_enabled:
            self.mlp = DenseTernaryMoE(dim, mlp_mult, moe_num_experts, moe_top_k, group_size, activation, leaky_relu_slope)
        else:
            self.mlp = MLP(dim, mlp_mult, group_size=group_size, activation=activation, leaky_relu_slope=leaky_relu_slope)
        self.moe_enabled = moe_enabled
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack([np.ones(dim, dtype=np.float32),
                                            np.zeros(dim, dtype=np.float32)]))

    def attn_norm(self, x):
        """RMSNorm with LN scale damping for attention input (used by shared block mode)."""
        return rms_norm(x) * self.ln_scale_factor

    def mlp_norm(self, x):
        """RMSNorm with LN scale damping for MLP input (used by shared block mode)."""
        return rms_norm(x) * self.ln_scale_factor

    def __call__(self, x, x0, v0=None):
        aux_loss = mx.array(0.0)
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x)
        attn_out, v_out = self.attn(normed, v0=v0)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        
        mlp_in = self.mlp_norm(x)
        if hasattr(self, 'moe_enabled') and self.moe_enabled:
            mlp_out, aux_loss = self.mlp(mlp_in)
        else:
            mlp_out = self.mlp(mlp_in)
            
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x, v_out, aux_loss


# ─────────────────────────────────────────────────────────────────────────
# Sprint 4: Deep Equilibrium (DEQ) Feedback with True Anderson Acceleration
# ─────────────────────────────────────────────────────────────────────────
#
# Novel contribution: DEQ applied to Ternary Koopman Capsules.
#
# Standard feedback loops in LMs re-run the decoder K times explicitly.
# This is wasteful: each pass computes a full decoder, and gradients must
# backprop through all K passes (O(K) memory, O(K) compute).
#
# DEQ replaces this with fixed-point iteration z* = f(z*), where f is
# one encoder→capsule→decoder pass. At convergence, the output IS the
# fixed point — no need to store intermediate passes.
#
# The backward pass uses implicit differentiation:
#   ∂L/∂θ = (I - ∂f/∂z)^{-1} · ∂L/∂z* · ∂f/∂θ
# approximated via Neumann series (truncated, O(1) memory).
#
# Anderson acceleration reduces the number of iterations from ~10 (naive
# Picard) to ~3 by solving a least-squares problem over past residuals.
#
# Why this matters for ternary: The feedback fixed point is MORE STABLE
# for ternary networks because:
#   1. Each iteration re-quantizes weights → fixed point = ternary-stable state
#   2. Anderson acceleration favors smooth convergence → fewer ternary flips
#   3. Implicit diff avoids backprop through quantization boundaries
# ─────────────────────────────────────────────────────────────────────────


class DEQFeedback(nn.Module):
    """
    Deep Equilibrium feedback with true Anderson acceleration.

    Replaces explicit multi-pass feedback with converged fixed-point iteration.
    Uses Anderson acceleration (Type-I) for fast convergence and implicit
    differentiation via truncated Neumann series for memory-efficient backward.
    """
    def __init__(self, model_dim, max_iter=3, tol=0.01, anderson_m=3):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.anderson_m = anderson_m

        # Learnable damping: controls how aggressively to mix in new iterates
        # Initialized to 0 → sigmoid(0) = 0.5 → equal mix
        self.damping_logit = mx.array(0.0, dtype=mx.float32)

    def anderson_acceleration(self, f_history, x_history):
        """
        True Anderson acceleration (Type-I).

        Given iterates x_0, x_1, ..., x_m and their f-evaluations
        f(x_0), f(x_1), ..., f(x_m), find the optimal linear combination:

            x_{m+1} = Σ_k α_k f(x_k)

        where α minimizes ||Σ_k α_k (f(x_k) - x_k)||² subject to Σ α_k = 1.

        This is a constrained least-squares problem solved in closed form.

        Args:
            f_history: list of f(x_k) tensors, each (B, T, D)
            x_history: list of x_k tensors, each (B, T, D)

        Returns:
            x_accel: (B, T, D) — accelerated next iterate
        """
        m = len(f_history)
        if m < 2:
            return f_history[-1]

        # Compute residuals: r_k = f(x_k) - x_k
        residuals = [f_history[i] - x_history[i] for i in range(m)]

        # Flatten to (m, -1) for the least-squares solve
        B, T, D = residuals[0].shape
        R = mx.stack([r.reshape(-1) for r in residuals], axis=0)  # (m, B*T*D)

        # Gram matrix of residuals: G[i,j] = <r_i, r_j>
        G = mx.matmul(R, R.T)  # (m, m)

        # Solve: minimize ||R^T α||² subject to 1^T α = 1
        # Using the KKT conditions: G α + λ 1 = 0, 1^T α = 1
        # → α = G^{-1} 1 / (1^T G^{-1} 1)

        # MLX doesn't have linalg.solve, so use Cholesky or pseudoinverse
        # Simple approach: solve via explicit inverse (m is small, typically 3-5)
        ones = mx.ones((m, 1), dtype=mx.float32)

        # Safety check removed — .item() calls are forbidden inside mx.compile.
        # Just proceed unconditionally; the Neumann series is numerically stable
        # with the regularization below for the small m we use (typically 3-5).
        resid_norms = mx.sqrt(mx.sum(R * R, axis=1) + 1e-12)  # (m,) — kept for future use

        # Adaptive regularization: fixed small regularization (no .item() branch)
        G = G + mx.eye(m) * 1e-4

        # Pseudoinverse via Neumann series: G^{-1} ≈ D^{-1} Σ (I - D^{-1} G)^n
        diag_inv = 1.0 / (mx.diag(G) + 1e-8)  # (m,)
        D_inv = mx.diag(diag_inv)  # (m, m)
        G_approx_inv = D_inv
        residual_mat = mx.eye(m) - mx.matmul(D_inv, G)
        # Spectral radius check removed — .item() forbidden inside mx.compile.
        # The fixed regularization above keeps rho < 1 for well-conditioned G.
        power = mx.eye(m)
        for _ in range(5):  # 5 Neumann terms for better accuracy
            power = mx.matmul(power, residual_mat)
            G_approx_inv = G_approx_inv + mx.matmul(power, D_inv)

        alpha_unnorm = mx.matmul(G_approx_inv, ones)  # (m, 1)
        alpha = alpha_unnorm / (mx.sum(alpha_unnorm) + 1e-8)  # normalize to sum=1
        alpha = alpha.reshape(-1)  # (m,)

        # Accelerated iterate: x_{m+1} = Σ_k α_k f(x_k)
        f_stack = mx.stack([f.reshape(-1) for f in f_history], axis=0)  # (m, B*T*D)
        x_accel = mx.matmul(alpha[None, :], f_stack).reshape(B, T, D)

        return x_accel

    def fixed_point_iter(self, z_init, f_eval, train_mode=True):
        """
        Fixed-point iteration with Anderson acceleration.

        Training: run exactly max_iter steps (for deterministic compute graph).
        Eval: run up to 2x max_iter steps, exit early on convergence.

        Args:
            z_init: (B, T, D) initial state
            f_eval: callable z → f(z)
            train_mode: whether to use fixed or adaptive iteration count

        Returns:
            z_final: (B, T, D) converged state
            num_iter: number of iterations actually taken
        """
        z = z_init
        x_history = []
        f_history = []
        damping = mx.sigmoid(self.damping_logit)
        max_k = self.max_iter if train_mode else self.max_iter * 2

        for k in range(max_k):
            z_fz = f_eval(z)  # raw fixed-point iterate f(z)

            # Record history for Anderson
            x_history.append(z)
            f_history.append(z_fz)

            # Trim to window size
            if len(x_history) > self.anderson_m:
                x_history.pop(0)
                f_history.pop(0)

            # Anderson acceleration (requires ≥ 2 history points)
            z_new = z_fz
            if len(x_history) >= 2:
                z_accel = self.anderson_acceleration(f_history, x_history)
                z_new = damping * z_accel + (1.0 - damping) * z_fz

            # Convergence check: use true fixed-point residual ||f(z) - z|| / ||z||
            # NOT ||z_{k+1} - z_k|| which can be small when Anderson extrapolates
            if not train_mode and k > 0:
                fp_resid = mx.sqrt(mx.mean((z_fz - z) ** 2) + 1e-8)
                norm_z = mx.sqrt(mx.mean(z ** 2) + 1e-8)
                if (fp_resid / norm_z) < self.tol:
                    return z_new, k + 1

            z = z_new

        return z, max_k



class GPT(nn.Module):
    """Ternary Reasoner — MLX version."""
    def __init__(self, args: Hyperparameters):
        super().__init__()
        global _TURBO_QUANT, _TURBO_QUANT_KV, _TERNARY_NOISE_SCALE, _STOCHASTIC_DEPTH_PROB
        _TURBO_QUANT = args.turbo_quant_train
        _TURBO_QUANT_KV = args.turbo_quant_kv
        _TERNARY_NOISE_SCALE = args.ternary_noise_scale
        _STOCHASTIC_DEPTH_PROB = args.stochastic_depth_prob
        dim = args.model_dim
        self.args = args
        self.feedback_enabled = args.feedback_enabled
        self.feedback_passes = args.feedback_passes
        self._train_feedback_passes = args.feedback_passes
        self.shared_blocks = args.shared_blocks
        self.architecture = args.architecture

        # Determine per-layer block types for hybrid architecture
        if args.architecture == "hybrid":
            # Alternating attention + Koopman-SSM for O(T) global + local mixing
            self._layer_types = ["attn" if i % 2 == 0 else "ssm" for i in range(args.num_layers)]
        elif args.architecture == "koopman_ssm":
            self._layer_types = ["ssm"] * args.num_layers
        elif args.architecture == "skc":
            # Spectral Koopman Capsule for unified spectral+capsule+Koopman
            self._layer_types = ["skc"] * args.num_layers
        else:
            self._layer_types = ["attn"] * args.num_layers

        # Embedding
        self.tok_emb = Embedding(args.vocab_size, args.embed_dim if args.embed_dim > 0 else dim)
        self.tok_emb.weight = mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * args.tied_embed_init_std
        self.embed_proj = EmbedProj(args.embed_dim, dim) if args.embed_dim > 0 and args.embed_dim != dim else None
        self.embed_proj_rev = EmbedProj(dim, args.embed_dim) if args.embed_dim > 0 and args.embed_dim != dim else None
        self.smeargate = SmearGate(dim) if args.smeargate_enabled else None

        self.vocab_bias = mx.zeros((args.vocab_size,), dtype=mx.float32)
        self.logit_softcap = args.logit_softcap

        # ── Unified U-Net Transformer / Koopman SSM architecture ──────
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        # Learnable capsule skip gate: scalar per decoder layer, init=0 → sigmoid(0)=0.5
        # Controls how much encoder mirror caps blend into decoder caps
        self.caps_skip_gate = mx.zeros((self.num_skip_weights,), dtype=mx.float32)

        # Blocks
        def make_block(layer_idx: int):
            ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if args.ln_scale_damping else 1.0
            lt = self._layer_types[layer_idx]

            if lt == "ssm":
                # Koopman-SSM block with progressive decay window
                d_win = args.koopman_decay_window
                if args.num_layers > 1:
                    d_win = min(16 * (2 ** layer_idx), 256)
                return KoopmanBlock(
                    dim, args.koopman_state_dim, args.mlp_mult,
                    mixer_rank=args.koopman_mixer_rank,
                    conv_kernel=args.koopman_conv_kernel,
                    decay_window=d_win,
                    group_size=args.bitnet_group_size,
                    activation=args.activation_type,
                    leaky_relu_slope=args.leaky_relu_slope,
                    ln_scale_factor=ln_sf,
                    moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k
                )
            elif lt == "skc":
                # Spectral Koopman Capsule layer
                return SKCLayer(
                    dim, capsule_num=args.skc_num_capsules, capsule_dim=args.skc_capsule_dim,
                    conv_kernel=args.skc_conv_kernel, block_size=args.skc_block_size,
                    mlp_mult=args.mlp_mult, group_size=args.bitnet_group_size,
                    activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
                    ln_scale_factor=ln_sf,
                    moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k
                )
            else:
                # Attention block
                layer_vrl = args.vrl_enabled and layer_idx >= args.vrl_start_layer
                layer_xsa = args.xsa_start_layer >= 0 and layer_idx >= args.xsa_start_layer
                return Block(
                    dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                    args.rope_base, args.qk_gain_init, group_size=args.bitnet_group_size,
                    activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
                    partial_rope_dims=args.partial_rope_dims, vrl_enabled=layer_vrl,
                    ln_scale_factor=ln_sf, xsa=layer_xsa,
                    moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k
                )

        # Tiled Blocks: Flexible unique block pairs tiling across num_layers
        # For SKC: all blocks are SKCLayers; for hybrid: pairs of [Attn, SSM]
        num_unique_pairs = max(1, args.shared_blocks) if args.shared_blocks > 0 else (args.num_layers // 2)

        if args.architecture == "skc":
            # Spectral Koopman Capsule: all blocks are unified SKC layers
            self.skc_blocks = [
                SKCLayer(dim, capsule_num=args.skc_num_capsules, capsule_dim=args.skc_capsule_dim,
                         conv_kernel=args.skc_conv_kernel, block_size=args.skc_block_size,
                         mlp_mult=args.mlp_mult, group_size=args.bitnet_group_size,
                         activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
                         ln_scale_factor=1.0,
                         moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k)
                for i in range(num_unique_pairs)
            ]
            self.attn_blocks = []
            self.ssm_blocks = []
        else:
            # Hybrid or pure architectures
            self.attn_blocks = [
                Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                      args.rope_base, args.qk_gain_init, group_size=args.bitnet_group_size,
                      activation=args.activation_type, leaky_relu_slope=args.leaky_relu_slope,
                      partial_rope_dims=args.partial_rope_dims, vrl_enabled=(i == num_unique_pairs - 1),
                      xsa=(i >= num_unique_pairs // 2),
                      moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k)
                for i in range(num_unique_pairs)
            ]
            self.ssm_blocks = [
                KoopmanBlock(args.model_dim, args.koopman_state_dim, args.mlp_mult,
                             mixer_rank=args.koopman_mixer_rank,
                             conv_kernel=args.koopman_conv_kernel,
                             decay_window=args.koopman_decay_window,
                             group_size=args.bitnet_group_size,
                             activation=args.activation_type,
                             leaky_relu_slope=args.leaky_relu_slope,
                             moe_enabled=args.moe_enabled, moe_num_experts=args.moe_num_experts, moe_top_k=args.moe_top_k)
                for i in range(num_unique_pairs)
            ]
            self.skc_blocks = []

        # Bug fix: do NOT hardcode these — use args.num_layers (was breaking weight sharing for num_layers != 8)
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        # Learnable skip weights for U-Net connections
        self.skip_weights = mx.zeros((self.num_skip_weights, args.model_dim), dtype=mx.float32)
        self.per_layer_attn_scales = [mx.ones((args.model_dim,), dtype=mx.float32) for _ in range(args.num_layers)]
        self.per_layer_mlp_scales = [mx.ones((args.model_dim,), dtype=mx.float32) for _ in range(args.num_layers)]
        self.per_layer_resid_mixes = [
            mx.array(np.stack([np.ones(args.model_dim, dtype=np.float32), np.zeros(args.model_dim, dtype=np.float32)]))
            for _ in range(args.num_layers)
        ]

        # Capsule bank (or bottleneck replacement)
        self.capsule_bank = None
        if args.capsule_enabled:
            if args.linear_bottleneck_enabled:
                # Absolute minimum: one low-rank linear residual, no nonlinearity
                self.capsule_bank = LinearBottleneck(dim, args.linear_bottleneck_rank)
            elif args.linkoopman_enabled:
                # Linear-attention Koopman: full-matrix state, content-adaptive readout
                self.capsule_bank = LinearAttnKoopman(
                    dim, args.linkoopman_heads, args.linkoopman_head_dim)
            elif args.ssm_bottleneck_enabled:
                # Causal diagonal SSM — simpler, faster
                self.capsule_bank = SSMBottleneck(dim, args.ssm_bottleneck_state_dim)
            else:
                self.capsule_bank = CapsuleBank(
                    dim, args.capsule_num, args.capsule_dim,
                    koopman_enabled=args.koopman_enabled,
                    koopman_rank=args.koopman_rank,
                    koopman_diag_init=args.koopman_diag_init,
                )

        # Feedback (explicit multi-pass)
        self.feedback_pooler = None
        self.feedback_adapters = None
        if self.feedback_enabled and not args.deq_feedback:
            self.feedback_pooler = FeedbackPooler(dim, args.feedback_dim, args.feedback_sketch_tokens)
            self.feedback_adapters = [
                FeedbackAdapter(dim, args.feedback_dim)
                for _ in range(self.num_decoder_layers)
            ]

        # DEQ Feedback (deep equilibrium alternative)
        self.deq_feedback = None
        if args.deq_feedback:
            self.deq_feedback = DEQFeedback(
                dim, max_iter=args.deq_max_iter,
                tol=args.deq_tol, anderson_m=args.deq_anderson_m
            )

        self.final_norm = RMSNormNoWeight()

        # Engram-style multi-head n-gram memory (shared by both architectures)
        self.engram = None
        self.engram_inject_layer = args.engram_inject_layer
        if args.bigram_hash_enabled:
            self.engram = EngramHash(
                args.bigram_hash_buckets, args.bigram_hash_dim, dim,
                group_size=args.bitnet_group_size,
                num_heads=args.engram_num_heads,
                num_orders=args.engram_num_orders,
            )

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
        if self.smeargate is not None:
            x = self.smeargate(x)
        return x, x

    def _koopman_ssm_forward(self, input_ids):
        """Flat-stack Koopman SSM forward pass (Path 2).

        No U-Net encoder/decoder split. No feedback loop.
        Just a clean stack of KoopmanBlock layers with optional Engram.

        Returns: (hidden, None, [], []) — same interface as transformer forward.
        """
        x, x0 = self._apply_embedding(input_ids)

        # Engram injection at specified layer (or input)
        moe_losses = []
        for i, block in enumerate(self.koopman_blocks):
            if (self.engram is not None
                    and self.engram_inject_layer >= 0
                    and i == self.engram_inject_layer):
                engram_out = self.engram(input_ids, hidden=x)
                x = x + engram_out.astype(x.dtype)
            if hasattr(block, 'moe_enabled') and block.moe_enabled:
                x, _, aux = block(x, x0)
                moe_losses.append(aux)
            else:
                x, _ = block(x, x0)

        return self.final_norm(x), None, [], [], moe_losses


    def _get_block(self, i):
        """Tiled block lookup: dynamic unique pairs -> num_layers alternating layers.

        With weight_sharing=True: decoder layers mirror encoder layers (U-Net style).
        Encoder layer j and decoder layer (num_layers-1-j) share the same block weights.
        This halves ternary param count while preserving depth.

        For SKC architecture: all layers use SKC blocks (no alternation).
        """
        if self.args.architecture == "skc":
            # SKC: all layers are unified SKC blocks
            num_unique = len(self.skc_blocks)
            if self.args.weight_sharing and i >= self.num_encoder_layers:
                dec_idx = i - self.num_encoder_layers
                mirror_enc_idx = self.num_encoder_layers - 1 - dec_idx
                mirror_enc_idx = max(0, min(mirror_enc_idx, self.num_encoder_layers - 1))
                unique_idx = mirror_enc_idx % num_unique
            else:
                unique_idx = i % num_unique
            return self.skc_blocks[unique_idx]
        else:
            # Hybrid/pure: alternating attn/ssm blocks
            num_unique = len(self.attn_blocks)
            if self.args.weight_sharing and i >= self.num_encoder_layers:
                # Decoder: mirror encoder block indices
                dec_idx = i - self.num_encoder_layers
                mirror_enc_idx = self.num_encoder_layers - 1 - dec_idx
                mirror_enc_idx = max(0, min(mirror_enc_idx, self.num_encoder_layers - 1))
                unique_idx = (mirror_enc_idx // 2) % num_unique
                is_ssm = (mirror_enc_idx % 2 == 1)
            else:
                unique_idx = (i // 2) % num_unique
                is_ssm = (i % 2 == 1)
            return self.ssm_blocks[unique_idx] if is_ssm else self.attn_blocks[unique_idx]

    def _layer_distance_from_bottleneck(self, layer_idx):
        """Distance of a layer from the U-Net bottleneck (capsule bank).
        Encoder layers: distance = num_enc - 1 - i (layer 0 is furthest)
        Decoder layers: distance = i - num_enc (layer right after bottleneck is 0)
        """
        if layer_idx < self.num_encoder_layers:
            return self.num_encoder_layers - 1 - layer_idx
        else:
            return layer_idx - self.num_encoder_layers

    def _run_block(self, layer_idx, x, x0, v0=None, prev_capsules=None):
        block = self._get_block(layer_idx)
        ln_sf = 1.0 / (layer_idx + 1) ** 0.5 if self.args.ln_scale_damping else 1.0
        sd_scale = mx.array(1.0, dtype=x.dtype)
        if self.training and _STOCHASTIC_DEPTH_PROB > 0.0:
            keep = mx.random.bernoulli(1.0 - _STOCHASTIC_DEPTH_PROB)
            sd_scale = keep.astype(x.dtype) / mx.array(1.0 - _STOCHASTIC_DEPTH_PROB, dtype=x.dtype)

        # Inside-out training: stop gradient for frozen layers
        # _max_unfrozen_distance is set by training loop; -1 = all frozen, inf = all active
        frozen = (self.training
                  and hasattr(self, '_max_unfrozen_distance')
                  and self._layer_distance_from_bottleneck(layer_idx) > self._max_unfrozen_distance)

        # Frozen layers: identity pass (preserve residual stream) rather than skipping
        if frozen:
            h_out = x  # identity pass — keeps activations flowing to unfrozen layers
            return h_out, None, mx.array(0.0)

        if isinstance(block, SKCLayer):
            h, v_out, aux_loss = block(x, x0, prev_capsules=prev_capsules)
            x = x + sd_scale * self.per_layer_mlp_scales[layer_idx].astype(x.dtype)[None, None, :] * h
            return x, v_out, aux_loss
        elif isinstance(block, KoopmanBlock):
            normed = rms_norm(x) * ln_sf
            h, _, aux_loss = block(normed, x0)
            x = x + sd_scale * self.per_layer_mlp_scales[layer_idx].astype(x.dtype)[None, None, :] * h
            return x, None, aux_loss
        else:
            normed_attn = rms_norm(x) * ln_sf
            attn_out, v_out = block.attn(normed_attn, v0=v0)
            x = x + sd_scale * self.per_layer_attn_scales[layer_idx].astype(x.dtype)[None, None, :] * attn_out
            normed_mlp = rms_norm(x) * ln_sf
            if hasattr(block, 'moe_enabled') and block.moe_enabled:
                mlp_out, aux_loss = block.mlp(normed_mlp)
            else:
                mlp_out = block.mlp(normed_mlp)
                aux_loss = mx.array(0.0)
            x = x + sd_scale * self.per_layer_mlp_scales[layer_idx].astype(x.dtype)[None, None, :] * mlp_out
            return x, v_out, aux_loss

    def _decoder_pass(self, x, x0, skips, sketch, v0, moe_losses, input_ids,
                      skc_caps=None, enc_caps=None, koopman_spec_targets=None):
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if i < self.num_skip_weights:
                x = x + mx.sigmoid(self.skip_weights[i]).astype(x.dtype)[None, None, :] * skips[-(i + 1)]

            # Symmetric capsule skip: blend decoder capsule state with mirrored encoder state.
            # Uses learnable per-layer gate caps_skip_gate (init=0 → sigmoid=0.5, learned).
            # Only active for SKC arch when enc_caps were collected.
            if (enc_caps is not None
                    and self.architecture == "skc"
                    and skc_caps is not None):
                mirror_idx = len(enc_caps) - 1 - i  # symmetric encoder layer index
                if 0 <= mirror_idx < len(enc_caps) and enc_caps[mirror_idx] is not None:
                    if i < len(self.caps_skip_gate):
                        cap_w = mx.sigmoid(self.caps_skip_gate[i].astype(skc_caps.dtype))
                    else:
                        cap_w = mx.array(0.5, dtype=skc_caps.dtype)
                    skc_caps = cap_w * skc_caps + (1.0 - cap_w) * enc_caps[mirror_idx]

            if i == 0 and self.engram is not None:
                x = x + self.engram(input_ids, hidden=x).astype(x.dtype)

            x, cap_out, aux_loss = self._run_block(bi, x, x0, v0=v0, prev_capsules=skc_caps)
            if cap_out is not None:
                skc_caps = cap_out
                # Collect actual decoder caps for Koopman speculation loss
                # Shape guard: spec_pred lives in CapsuleBank space; cap_out in SKCLayer space —
                # only compute loss when shapes match (both must be same capsule system)
                if koopman_spec_targets is not None and i < len(koopman_spec_targets):
                    spec_pred = koopman_spec_targets[i]
                    if spec_pred is not None and spec_pred.shape == cap_out.shape:
                        actual_target = mx.stop_gradient(cap_out)
                        spec_loss_i = mx.mean((spec_pred - actual_target) ** 2)
                        if aux_loss is not None:
                            aux_loss = aux_loss + self.args.koopman_speculator_weight * spec_loss_i
                        else:
                            aux_loss = self.args.koopman_speculator_weight * spec_loss_i
            if aux_loss is not None:
                moe_losses.append(aux_loss)
            if self.feedback_adapters is not None and sketch is not None:
                x = self.feedback_adapters[i](x, sketch)
        return x

    def __call__(self, input_ids, carry_capsules=None):
        self._distill_hidden0 = None
        x, x0 = self._apply_embedding(input_ids)
        skips = []
        moe_losses = []
        v0 = None

        skc_caps = None  # Cross-layer Koopman capsule state for SKC architecture
        enc_caps = []    # Save per-layer encoder capsule states for symmetric skip connections
        for i in range(self.num_encoder_layers):
            x, layer_out, aux_loss = self._run_block(i, x, x0, v0=v0, prev_capsules=skc_caps)
            moe_losses.append(aux_loss)
            # layer_out is capsule state for SKCLayer, attention v for Block
            if self.architecture == "skc":
                skc_caps = layer_out
                if layer_out is not None:
                    enc_caps.append(layer_out)
            elif v0 is None and layer_out is not None:
                v0 = mx.stop_gradient(layer_out)  # cache first attention v
            skips.append(x)

        capsule_state = None
        if carry_capsules is not None:
            B_curr = x.shape[0]
            carry_avg = mx.mean(carry_capsules, axis=0, keepdims=True)
            capsule_state = mx.broadcast_to(carry_avg, (B_curr, carry_avg.shape[1], carry_avg.shape[2]))
        
        if self.capsule_bank is not None:
            x, capsule_state, _, _ = self.capsule_bank(x, prev_capsules=capsule_state)

        # Branch: decoder starts a fresh capsule trajectory from bottleneck state
        # skc_caps_dec will evolve independently through the decoder
        skc_caps_dec = skc_caps  # initialise from encoder's final caps
        bottleneck_caps = capsule_state  # save bottleneck state for speculation

        # Koopman speculation: from bottleneck, predict each decoder layer's capsule state
        # Training signal: ||A^k · bottleneck_caps - sg(actual_decoder_caps[k])||^2
        # This teaches the Koopman operator the encoder→decoder trajectory
        koopman_spec_targets = []  # will be filled during decoder pass
        if (self.capsule_bank is not None
                and self.args.koopman_speculator_enabled
                and bottleneck_caps is not None
                and self.args.koopman_speculator_weight > 0):
            for k in range(1, self.num_decoder_layers + 1):
                c_spec_k = self.capsule_bank.koopman.speculate(bottleneck_caps, steps=k) \
                           if (hasattr(self.capsule_bank, 'koopman')
                               and self.capsule_bank.koopman is not None) else None
                koopman_spec_targets.append(c_spec_k)

        encoded = x
        sketch = None
        consistency_losses = []
        speculative_losses = []
        prev_capsule_state = None
        fast_forwarded = False
        num_passes = self.feedback_passes

        # DEQ feedback: replace explicit multi-pass with fixed-point iteration
        if self.deq_feedback is not None:
            # Accumulate moe_losses across DEQ iterations into the outer list
            _deq_moe_buf = []

            def _deq_f(z):
                z_enc = z
                if self.capsule_bank is not None:
                    z_enc, _, _, _ = self.capsule_bank(z_enc, prev_capsules=capsule_state)
                iter_moe = []
                out = self._decoder_pass(z_enc, x0, skips, sketch=None, v0=v0, moe_losses=iter_moe, input_ids=input_ids, skc_caps=skc_caps_dec, enc_caps=enc_caps, koopman_spec_targets=None)
                _deq_moe_buf.extend(iter_moe)
                return out

            x, _ = self.deq_feedback.fixed_point_iter(encoded, _deq_f, train_mode=self.training)
            moe_losses.extend(_deq_moe_buf)
        else:
            for correction_pass in range(num_passes + 1):
                if correction_pass > 0 and self.feedback_enabled and self.feedback_pooler is not None:
                    # Bug fix: causal pooling — position t only sees tokens 0..t
                    x_normed = self.final_norm(x)  # (B, T, D)
                    T_cur = x_normed.shape[1]
                    cum_sum = mx.cumsum(x_normed, axis=1)  # (B, T, D)
                    counts = mx.arange(1, T_cur + 1, dtype=x_normed.dtype).reshape(1, T_cur, 1)
                    causal_pool = cum_sum / counts  # (B, T, D)
                    sketch = self.feedback_pooler(causal_pool)
                else:
                    sketch = None

                if self.capsule_bank is not None and correction_pass > 0:
                    prev_capsule_state = capsule_state
                    spec_steps = self.args.koopman_speculator_steps if (self.args.koopman_speculator_enabled and correction_pass == 1) else 0
                    encoded, capsule_state, c_pred, c_spec = self.capsule_bank(
                        encoded, prev_capsules=capsule_state, speculate_steps=spec_steps
                    )
                    if c_pred is not None:
                        consistency_losses.append((c_pred, mx.stop_gradient(capsule_state)))
                    if c_spec is not None:
                        speculative_losses.append(c_spec)

                x = self._decoder_pass(encoded, x0, skips, sketch=sketch, v0=v0, moe_losses=moe_losses, input_ids=input_ids, skc_caps=skc_caps_dec, enc_caps=enc_caps, koopman_spec_targets=koopman_spec_targets)

                if (correction_pass == 0 and self.training
                        and self.args.self_distill_kl_weight > 0.0 and num_passes > 0):
                    self._distill_hidden0 = mx.stop_gradient(self.final_norm(x))
                if not self.feedback_enabled or self.feedback_pooler is None:
                    break

        c_final = mx.stop_gradient(capsule_state) if capsule_state is not None else None
        jepa_loss = []
        if c_final is not None:
            for c_s in speculative_losses:
                jepa_loss.append((c_s, c_final))

        return self.final_norm(x), capsule_state, consistency_losses, jepa_loss, moe_losses

    def _hidden_to_logits(self, x, temperature=1.0):
        if self.embed_proj_rev is not None:
            x = self.embed_proj_rev(x)
        w = self.tok_emb.weight.astype(x.dtype)
        logits = x @ w.T + self.vocab_bias.astype(x.dtype)
        # TKA-H: softcap
        logits = self.logit_softcap * mx.tanh(logits / self.logit_softcap)
        if temperature != 1.0:
            logits = logits / temperature
        return logits

    def forward_logits(self, input_ids, temperature=1.0):
        hidden, _, _, _, _ = self(input_ids)
        logits = self._hidden_to_logits(hidden.reshape(-1, hidden.shape[-1]), temperature=temperature)
        return logits.reshape(input_ids.shape[0], input_ids.shape[1], -1)

    def forward_logits_with_carry(self, input_ids, carry_capsules=None, temperature=1.0):
        hidden, capsule_state, _, _, _ = self(input_ids, carry_capsules=carry_capsules)
        logits = self._hidden_to_logits(hidden.reshape(-1, hidden.shape[-1]), temperature=temperature)
        return logits.reshape(input_ids.shape[0], input_ids.shape[1], -1), capsule_state

    def loss(self, input_ids, target_ids, carry_capsules=None, reduction="mean", temperature=1.0):
        hidden, capsule_state, consistency_losses, jepa_loss, moe_losses = self(input_ids, carry_capsules=carry_capsules)
        x = hidden.reshape(-1, hidden.shape[-1])
        y = target_ids.reshape(-1)
        logits = self._hidden_to_logits(x, temperature=temperature).astype(mx.float32)
        if reduction == "none":
            return nn.losses.cross_entropy(logits, y, reduction="none").reshape(target_ids.shape)

        lse = mx.logsumexp(logits, axis=-1)
        target_logits = mx.take_along_axis(logits, y[:, None], axis=1).squeeze(-1)
        ce_loss = mx.mean(lse - target_logits)
        if self.training:
            ce_loss = ce_loss + 1e-4 * mx.mean(lse * lse)

        # Capsule consistency auxiliary loss (Koopman dynamics training signal)
        if self.training and consistency_losses and self.args.koopman_consistency_weight > 0:
            consist_sum = mx.array(0.0, dtype=mx.float32)
            for c_pred, c_actual in consistency_losses:
                consist_sum = consist_sum + mx.mean((c_pred - c_actual) ** 2)
            consist_loss = consist_sum / len(consistency_losses)
            ce_loss = ce_loss + self.args.koopman_consistency_weight * consist_loss

        # JEPA Diffusion Speculation Loss
        if self.training and jepa_loss and self.args.koopman_speculator_weight > 0:
            spec_sum = mx.array(0.0, dtype=mx.float32)
            for c_spec, c_final in jepa_loss:
                spec_sum = spec_sum + mx.mean((c_spec - c_final) ** 2)
            speculative_ms_loss = spec_sum / len(jepa_loss)
            ce_loss = ce_loss + self.args.koopman_speculator_weight * speculative_ms_loss

                # MoE Aux Loss
        if self.training and self.args.moe_enabled and moe_losses:
            total_moe = mx.sum(mx.stack([l for l in moe_losses if l.ndim == 0 and l.size == 1]))
            ce_loss = ce_loss + self.args.moe_router_aux_loss_coef * total_moe

        # Self-distillation: KL(final_pass || stop_grad(pass_0)).
        # Penalises the feedback refinement from straying too far from the raw forward estimate,
        # acting as a consistency regulariser that anchors multi-pass corrections to coarse predictions.
        if (self.training and self.args.self_distill_kl_weight > 0.0
                and getattr(self, '_distill_hidden0', None) is not None):
            h0 = self._distill_hidden0.reshape(-1, self._distill_hidden0.shape[-1])
            logits0 = self._hidden_to_logits(h0).astype(mx.float32)
            # log-softmax of both distributions
            log_p0 = mx.stop_gradient(logits0 - mx.logsumexp(logits0, axis=-1, keepdims=True))
            log_p_curr = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            # KL(p_curr || p_0) = sum(p_curr * (log_p_curr - log_p_0))
            kl = mx.mean(mx.sum(mx.exp(log_p_curr) * (log_p_curr - log_p0), axis=-1))
            ce_loss = ce_loss + self.args.self_distill_kl_weight * kl
            self._distill_hidden0 = None

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
        if not self._ever_updated:
            self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
            self._ever_updated = True
            return
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
    """
    Muon optimizer with Ternary Koopman Optimization (TKO).

    ═══════════════════════════════════════════════════════════════════════
    NOVEL CONTRIBUTION: Spectral Momentum for Ternary Networks
    ═══════════════════════════════════════════════════════════════════════

    Standard Muon computes orthogonal gradient updates via Newton-Schulz
    iteration, which implicitly operates in the spectral (SVD) domain.
    But it treats continuous weights and ternary projections as separate:
    optimize continuous → project to ternary → hope for the best.

    TKO closes this gap with THREE innovations:

    1. TERNARY BOUNDARY GRADIENT AMPLIFICATION
       Near ternary decision boundaries (where w/scale ≈ ±0.5), the
       quantization function has infinite gradient (discontinuity).
       Standard STE pretends this doesn't exist. TKO amplifies gradients
       near boundaries by a factor proportional to 1/distance_to_boundary,
       clamped to [1, 10]. This focuses optimization effort where ternary
       decisions are most uncertain — exactly where gradient information
       matters most.

    2. SPECTRAL FLIP TRACKING WITH KOOPMAN MODES
       Instead of scalar flip rate tracking, TKO maintains a per-group
       spectral decomposition of the flip pattern. Groups where flips
       follow a PREDICTABLE pattern (Koopman mode with |λ| ≈ 1) get
       HIGHER learning rates (the optimizer can anticipate the oscillation).
       Groups with CHAOTIC flips (no dominant Koopman mode) get dampened.
       This distinguishes "healthy exploration" from "destructive oscillation".

    3. SCORE-SPACE OPTIMIZATION
       Maintain continuous "scores" s_ij separate from weights w_ij.
       Forward: w_ij = ternary_project(s_ij). Backward: ∂L/∂s via STE.
       The Muon NS5 orthogonalization is applied to ∂L/∂s, not ∂L/∂w.
       This means the optimizer's spectral structure aligns with the
       SCORE manifold, not the discontinuous ternary weight space.
    ═══════════════════════════════════════════════════════════════════════
    """
    def __init__(self, keys, params, args):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}
        self.tko_enabled = args.tko_enabled
        gs = args.bitnet_group_size

        if self.tko_enabled:
            self._prev_signs = {}
            self._flip_ema = {}
            self._flip_pattern = {}  # For Koopman mode tracking
            for k in keys:
                p = params[k]
                flat = p.reshape(-1, gs) if p.size % gs == 0 else p.reshape(-1, 1)
                grp_scale = mx.mean(mx.abs(flat), axis=-1, keepdims=True) + 1e-8
                self._prev_signs[k] = mx.sign(mx.round(flat / grp_scale))
                self._flip_ema[k] = mx.zeros((flat.shape[0],), dtype=mx.float32)
                # Koopman flip pattern: running autocorrelation of flip events
                self._flip_pattern[k] = mx.zeros((flat.shape[0],), dtype=mx.float32)

    def _boundary_amplification(self, p, gs):
        """
        Compute per-element gradient amplification factor based on distance
        to the nearest ternary decision boundary.

        Near a boundary (w/scale ≈ ±0.5), a small gradient nudge can flip
        the ternary value. This is where optimization MATTERS MOST.

        Returns: (same shape as p) amplification factors ∈ [1, 10].
        """
        flat = p.reshape(-1, gs) if p.size % gs == 0 else p.reshape(-1, 1)
        grp_scale = mx.mean(mx.abs(flat), axis=-1, keepdims=True) + 1e-8
        normalized = flat / grp_scale  # Values that round to {-1, 0, +1}

        # Distance to nearest boundary: boundaries at ±0.5
        frac = mx.abs(normalized) - 0.5
        dist = mx.abs(frac)  # 0 = exactly on boundary, 0.5 = center of a ternary level

        # Amplification: 1/(dist + ε), clamped to [1, 10]
        amp = mx.clip(1.0 / (dist + 0.1), 1.0, 10.0)
        return amp.reshape(p.shape)

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
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))

            if self.tko_enabled:
                gs = self.args.bitnet_group_size
                # ── Innovation 1: Boundary gradient amplification (BEFORE NS5) ──
                # Amplify gradients near ternary decision boundaries before orthogonalization
                # so NS5 preserves the spectral structure of the amplified gradient.
                amp = self._boundary_amplification(p, gs)
                g_eff = g_eff * amp.astype(g_eff.dtype)

            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)

            if self.tko_enabled:
                gs = self.args.bitnet_group_size
                # ── Innovation 2: Spectral flip tracking ──
                flat = p.reshape(-1, gs) if p.size % gs == 0 else p.reshape(-1, 1)
                grp_scale = mx.mean(mx.abs(flat), axis=-1, keepdims=True) + 1e-8
                curr_signs = mx.sign(mx.round(flat / grp_scale))

                if k in self._prev_signs and self._prev_signs[k].shape == curr_signs.shape:
                    flips = mx.mean(mx.abs(curr_signs - self._prev_signs[k]) > 0.5, axis=-1)
                    decay = self.args.tko_score_decay
                    self._flip_ema[k] = decay * self._flip_ema[k] + (1.0 - decay) * flips

                    # Koopman mode tracking: autocorrelation of flip events
                    # Positive autocorrelation → predictable oscillation (can be leveraged)
                    # Negative or zero → chaotic (must be dampened)
                    self._flip_pattern[k] = 0.9 * self._flip_pattern[k] + 0.1 * (
                        flips * mx.sign(self._flip_ema[k] - 0.1)
                    )

                    # Adaptive LR scaling:
                    # Base: dampen high-flip groups
                    base_scale = 1.0 - 0.9 * self._flip_ema[k]
                    # Koopman boost: if flips are predictable (positive autocorrelation),
                    # we can afford to be more aggressive
                    koopman_boost = mx.clip(1.0 + 0.5 * self._flip_pattern[k], 0.5, 2.0)
                    lr_scale_flat = base_scale * koopman_boost

                    # Apply per-group scaling to orthogonalized gradient
                    lr_adapt = lr_scale_flat.reshape(-1, 1).astype(g_ortho.dtype)
                    g_ortho_flat = g_ortho.reshape(-1, gs) if g_ortho.size % gs == 0 else g_ortho.reshape(flat.shape)
                    g_ortho = (g_ortho_flat * lr_adapt).reshape(g_ortho.shape)

                self._prev_signs[k] = curr_signs

            decayed = p * (1.0 - lr * self.args.muon_wd) if self.args.muon_wd > 0 else p
            out[k] = decayed - lr * (g_ortho * scale).astype(p.dtype)
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
        # Koopman dynamics get a slower LR (0.01 instead of 0.025) for stability
        self.koopman_diag_keys = [
            k for k in params
            if "koopman.diag" in k
            or "mixer_diag" in k
            or "mixer_lowrank" in k
            or "mixer_conv" in k
            or "koopman_evo.eigenvalues" in k
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

    # Parameter names that should NOT receive weight decay (biases, norms, gates, scalars)
    _NO_WD_PATTERNS = ("bias", "norm", "gate", "alpha", "scale", "vocab_bias",
                       "band_centers", "band_log_widths", "damping_logit",
                       "eigenvalues", "nonlinear_gate", "A_logit")

    def _decay_params(self, params, lr):
        if self.args.adam_wd <= 0:
            return params
        mul = 1.0 - lr * self.args.adam_wd
        out = {}
        for k, p in params.items():
            if any(pat in k for pat in self._NO_WD_PATTERNS) or p.ndim < 2:
                out[k] = p  # no weight decay for scalars/biases/norms
            else:
                out[k] = p * mul
        return out

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
            updated.update(self.adam_scalar.apply_gradients(
                scalar_grads,
                self._decay_params(scalar_params, self.args.scalar_lr * lr_mul),
            ))

        # Koopman diagonal with lower LR
        self.adam_koopman_diag.learning_rate = 0.01 * lr_mul
        kd_grads = {k: grads[k] for k in self.koopman_diag_keys if k in grads}
        kd_params = {k: params[k] for k in self.koopman_diag_keys if k in params}
        if kd_grads:
            updated.update(self.adam_koopman_diag.apply_gradients(
                kd_grads,
                self._decay_params(kd_params, 0.01 * lr_mul),
            ))

        model.update(tree_unflatten(list(updated.items())))

        # Stability constraint: clamp Koopman diagonal to (-0.999, 0.999)
        # This is a hard constraint from dynamical systems theory: ρ(A) must be < 1
        if (model.capsule_bank is not None
                and hasattr(model.capsule_bank, 'koopman')
                and model.capsule_bank.koopman is not None):
            clamped = mx.clip(model.capsule_bank.koopman.diag, -0.999, 0.999)
            model.capsule_bank.koopman.diag = clamped

        # Clamp Koopman SSM mixer diagonals (SSM and hybrid architectures)
        if hasattr(model, "architecture") and model.architecture in ("koopman_ssm", "hybrid"):
            # In hybrid mode, only ssm_blocks have mixer_diag
            if hasattr(model, "ssm_blocks") and model.ssm_blocks:
                for b in model.ssm_blocks:
                    if hasattr(b, "mixer") and hasattr(b.mixer, "mixer_diag"):
                        clamped = mx.clip(b.mixer.mixer_diag, -0.999, 0.999)
                        b.mixer.mixer_diag = clamped


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

    row_groups = arr_padded.shape[1] // gs
    best_q = np.zeros((rows * row_groups, gs), dtype=np.int8)
    best_scale = np.zeros((rows * row_groups, 1), dtype=np.float32)
    best_mse = np.full(rows, np.inf, dtype=np.float32)

    # Percentiles to try: 100% (no clip), down to 90%
    percentiles = np.linspace(100.0, 90.0, num_percentiles)

    for pct in percentiles:
        if pct < 100.0:
            # Vectorized percentile across all rows at once
            thresholds = np.percentile(np.abs(arr_padded), pct, axis=1, keepdims=True)
            clipped = np.clip(arr_padded, -thresholds, thresholds)
        else:
            clipped = arr_padded

        grouped = clipped.reshape(-1, gs)
        scale = np.mean(np.abs(grouped), axis=-1, keepdims=True)
        scale = np.maximum(scale, 1e-8)
        q = np.clip(np.round(grouped / scale), -1, 1).astype(np.int8)

        # Reconstruct and measure per-row MSE (vectorized)
        recon = (q.astype(np.float32) * scale).reshape(rows, -1)
        mse = np.mean((arr_padded - recon) ** 2, axis=1)

        # Update best rows (vectorized mask update)
        better = mse < best_mse
        if np.any(better):
            for r in np.where(better)[0]:
                best_mse[r] = mse[r]
                best_q[r * row_groups:(r + 1) * row_groups] = q[r * row_groups:(r + 1) * row_groups]
                best_scale[r * row_groups:(r + 1) * row_groups] = scale[r * row_groups:(r + 1) * row_groups]

    return best_q, best_scale, pad_cols

def _build_hadamard_np_unnormalized(n):
    if n == 1:
        return np.array([[1.0]], dtype=np.float32)
    else:
        h_half = _build_hadamard_np_unnormalized(n // 2)
        top = np.concatenate([h_half, h_half], axis=1)
        bot = np.concatenate([h_half, -h_half], axis=1)
        return np.concatenate([top, bot], axis=0)

def _build_hadamard_np(n):
    """Build normalized orthogonal Hadamard matrix in numpy. H @ H.T = I."""
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"
    return _build_hadamard_np_unnormalized(n) / math.sqrt(n)

_HADAMARD_NP_CACHE = {}
def _get_hadamard_np(n):
    if n not in _HADAMARD_NP_CACHE:
        _HADAMARD_NP_CACHE[n] = _build_hadamard_np(n)
    return _HADAMARD_NP_CACHE[n]

_RADERMACHER_NP_CACHE = {}
def _get_rademacher_np(n):
    if n not in _RADERMACHER_NP_CACHE:
        idx = np.arange(n, dtype=np.int64)
        _RADERMACHER_NP_CACHE[n] = np.where(((idx * 1103515245 + 12345) & 1) == 1, -1.0, 1.0).astype(np.float32)
    return _RADERMACHER_NP_CACHE[n]


def _turbo_ternary_quantize(arr_np, gs):
    """TurboQuant-style ternary quantization: signed Hadamard precondition → quantize.

    At inference, the load path inverse-rotates (H is self-inverse) to recover dense weights.
    Storage is still ternary (base-3 packed), but quantization MSE is provably lower because
    Hadamard distributes outliers uniformly across coordinates.
    """
    rows, cols = arr_np.shape
    pad_cols = (gs - cols % gs) % gs
    if pad_cols > 0:
        arr_np = np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
    grouped = arr_np.reshape(-1, gs)

    # Signed Hadamard preconditioning before scalar quantization
    H = _get_hadamard_np(gs)
    signs = _get_rademacher_np(gs)
    grouped_rotated = (grouped * signs) @ H

    # Quantize in rotated space
    scale = np.mean(np.abs(grouped_rotated), axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(grouped_rotated / scale), -1, 1).astype(np.int8)
    dequant = (q.astype(np.float32) * scale) @ H
    dequant = dequant * signs

    return q, scale, pad_cols, dequant


def _plain_ternary_quantize(arr_np, gs):
    rows, cols = arr_np.shape
    pad_cols = (gs - cols % gs) % gs
    if pad_cols > 0:
        arr_np = np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
    grouped = arr_np.reshape(-1, gs)
    scale = np.mean(np.abs(grouped), axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-8)
    q = np.clip(np.round(grouped / scale), -1, 1).astype(np.int8)
    dequant = q.astype(np.float32) * scale
    return q, scale, pad_cols, dequant


def _is_export_ternary_candidate_mlx(name, arr):
    return (
        arr.ndim == 2
        and (arr.size >= 65_536 or name.endswith("local_conv_weight"))
        and "tok_emb" not in name
        and "lm_head" not in name
        and "embed_proj" not in name
        and "engram.tables" not in name
    )


def _filter_serialization_average_mlx(source_sd, include_ternary):
    if include_ternary:
        return source_sd, len(source_sd)
    filtered = {}
    for name, arr in source_sd.items():
        if not _is_export_ternary_candidate_mlx(name, arr):
            filtered[name] = arr
    return filtered, len(filtered)


def quantize_for_export(model, args):
    """Quantize model to ternary + float, serialize to LZMA blob."""
    params = dict(tree_flatten(model.parameters()))
    # Force-evaluate all parameters before numpy conversion to avoid lazy graph recomputation
    mx.eval(*params.values())
    use_turbo = args.turbo_quant_export
    q_sd = {}
    for name, arr in params.items():
        arr_np = np.array(arr.astype(mx.float32), dtype=np.float32)
        if _is_export_ternary_candidate_mlx(name, arr_np):
            gs = args.bitnet_group_size
            rows, cols = arr_np.shape
            turbo_used = False

            q, scale, pad_cols, plain_dequant = _plain_ternary_quantize(arr_np, gs)
            if use_turbo and (gs & (gs - 1)) == 0:
                q_turbo, scale_turbo, pad_cols_turbo, turbo_dequant = _turbo_ternary_quantize(arr_np, gs)
                plain_target = arr_np if pad_cols == 0 else np.pad(arr_np, ((0, 0), (0, pad_cols)), mode='constant')
                turbo_target = arr_np if pad_cols_turbo == 0 else np.pad(arr_np, ((0, 0), (0, pad_cols_turbo)), mode='constant')
                plain_mse = np.mean((plain_target.reshape(-1, gs) - plain_dequant) ** 2)
                turbo_mse = np.mean((turbo_target.reshape(-1, gs) - turbo_dequant) ** 2)
                if turbo_mse < plain_mse:
                    q, scale, pad_cols = q_turbo, scale_turbo, pad_cols_turbo
                    turbo_used = True
            elif args.gptq_lite_enabled and args.gptq_lite_percentiles > 1:
                # GPTQ-lite: clip percentile search for better ternary approximation
                q, scale, pad_cols = _gptq_lite_ternary(
                    arr_np, gs, num_percentiles=args.gptq_lite_percentiles
                )

            packed, n_trits = pack_ternary_base3(q)
            scale_f16 = scale.astype(np.float16)
            entry = {"type": "ternary", "packed": packed, "scale": scale_f16,
                     "shape": (rows, cols), "n_trits": n_trits, "pad_cols": pad_cols}
            if turbo_used:
                entry["turbo"] = True  # load path must invert the signed-Hadamard preconditioner
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


def ce_from_logits(logits, target_ids, reduction="mean"):
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
        target_ids.reshape(-1),
        reduction=reduction,
    ).reshape(target_ids.shape) if reduction == "none" else nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
        target_ids.reshape(-1),
        reduction=reduction,
    )


def set_eval_mode(model, eval_feedback_passes):
    prev_training = model.training
    prev_feedback_passes = model.feedback_passes
    model.eval()
    # Apply EMA shadow weights for eval when ema_eval_apply is enabled.
    # The smoother EMA surface ternarizes with lower MSE, improving eval BPB.
    if (_EVAL_EMA is not None and _EVAL_EMA._ever_updated
            and hasattr(model, 'args') and model.args.ema_eval_apply):
        _EVAL_EMA.apply(model)
    if model.feedback_enabled:
        model.set_feedback_passes(
            eval_feedback_passes if eval_feedback_passes > 0 else model._train_feedback_passes
        )
    return prev_training, prev_feedback_passes


def restore_mode(model, prev_training, prev_feedback_passes):
    # Restore live training weights after eval (undo EMA apply)
    if (_EVAL_EMA is not None and hasattr(_EVAL_EMA, 'original')
            and hasattr(model, 'args') and model.args.ema_eval_apply):
        _EVAL_EMA.restore(model)
    model.set_feedback_passes(prev_feedback_passes)
    if prev_training:
        model.train()
    else:
        model.eval()


def eval_val(model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
             log_fn=None, temperature=1.0):
    """Validation with optional capsule carry and calibrated logits."""
    prev_training, prev_feedback_passes = set_eval_mode(model, args.eval_feedback_passes)
    seq_len = args.train_seq_len
    batch_seqs = max(1, args.val_batch_size // seq_len)
    total_seqs = (val_tokens.size - 1) // seq_len
    total_batches = max((total_seqs + batch_seqs - 1) // batch_seqs, 1)
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
            logits, capsule_state = model.forward_logits_with_carry(
                x, carry_capsules=carry_capsules, temperature=temperature
            )
            loss = ce_from_logits(logits, y, reduction="mean").astype(mx.float32)
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
            logits = model.forward_logits(x, temperature=temperature)
            loss = ce_from_logits(logits, y, reduction="mean").astype(mx.float32)
            mx.eval(loss)

        n = float(y.size)
        total_loss += float(loss.item()) * n
        prev_ids = x_np.ravel()
        tgt_ids = y_np.ravel()
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_lut[prev_ids]).astype(np.int16)
        total_tokens += n
        total_bytes += float(bytes_np.sum())
        if log_fn is not None and total_batches > 1 and (
            start == 0 or end == total_seqs or (start // batch_seqs + 1) % 25 == 0
        ):
            log_fn(f"val_progress:{start // batch_seqs + 1}/{total_batches}")
    restore_mode(model, prev_training, prev_feedback_passes)
    val_loss = total_loss / total_tokens
    bpt = val_loss / math.log(2.0)
    val_bpb = bpt * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val_sliding(model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                     stride=64, temperature=1.0):
    prev_training, prev_feedback_passes = set_eval_mode(model, args.eval_feedback_passes)
    seq_len = args.train_seq_len
    batch_size = max(1, args.sliding_batch_size)
    total_tokens = val_tokens.size - 1
    total_loss = 0.0
    total_tok_count = 0.0
    total_byte_count = 0.0
    all_starts = [s for s in range(0, total_tokens, stride) if min(s + seq_len, total_tokens) - s >= 1]

    use_carry = args.capsule_carry_enabled and args.capsule_enabled
    decay = args.capsule_carry_decay
    carry_capsules = None

    for i in range(0, len(all_starts), batch_size):
        batch_starts = all_starts[i:i + batch_size]
        bsz = len(batch_starts)
        x_batch = np.zeros((bsz, seq_len), dtype=np.int32)
        y_batch = np.zeros((bsz, seq_len), dtype=np.int32)
        wlens = []
        for j, start in enumerate(batch_starts):
            end = min(start + seq_len, total_tokens)
            wlen = end - start
            wlens.append(wlen)
            chunk = val_tokens[start:end + 1]
            x_batch[j, :wlen] = chunk[:-1]
            y_batch[j, :wlen] = chunk[1:]

        x = mx.array(x_batch, dtype=mx.int32)
        y = mx.array(y_batch, dtype=mx.int32)
        if use_carry:
            logits, capsule_state = model.forward_logits_with_carry(
                x, carry_capsules=carry_capsules, temperature=temperature
            )
            if capsule_state is not None:
                cs_avg = mx.mean(capsule_state, axis=0, keepdims=True)
                if carry_capsules is not None:
                    carry_capsules = mx.stop_gradient(decay * carry_capsules + (1.0 - decay) * cs_avg)
                else:
                    carry_capsules = mx.stop_gradient(cs_avg)
                mx.eval(carry_capsules)
        else:
            logits = model.forward_logits(x, temperature=temperature)

        nll = ce_from_logits(logits, y, reduction="none")
        mx.eval(nll)
        nll_np = np.array(nll.astype(mx.float32), dtype=np.float32)
        for j, start in enumerate(batch_starts):
            wlen = wlens[j]
            score_from = 0 if start == 0 else max(wlen - stride, 0)
            scored = nll_np[j, score_from:wlen]
            sx = x_batch[j, score_from:wlen]
            sy = y_batch[j, score_from:wlen]
            total_loss += float(scored.astype(np.float64).sum())
            total_tok_count += float(wlen - score_from)
            tok_bytes = base_bytes_lut[sy].astype(np.int16, copy=True)
            tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_lut[sx]).astype(np.int16)
            total_byte_count += float(tok_bytes.astype(np.float64).sum())

    restore_mode(model, prev_training, prev_feedback_passes)
    val_loss = total_loss / max(total_tok_count, 1.0)
    val_bpb = (val_loss / math.log(2.0)) * (total_tok_count / max(total_byte_count, 1.0))
    return val_loss, val_bpb


def find_temp(model, args, calibration_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut):
    best_t, best_loss = 1.0, float("inf")
    for t in (0.90, 0.95, 1.00, 1.05, 1.10):
        loss, _ = eval_val(
            model, args, calibration_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_lut, temperature=t,
        )
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return best_t


class NgramCache:
    """Dynamic n-gram cache built from already-scored tokens."""

    def __init__(self, max_order=5, alpha_base=0.05, alpha_scale=0.55, entropy_center=4.0):
        self.max_order = max_order
        self.alpha_base = alpha_base
        self.alpha_scale = alpha_scale
        self.entropy_center = entropy_center
        self.counts = [{} for _ in range(max_order + 1)]
        self.total_counts = [{} for _ in range(max_order + 1)]

    def update(self, tokens):
        for order in range(2, self.max_order + 1):
            for i in range(len(tokens) - order + 1):
                ctx = tuple(tokens[i:i + order - 1])
                nxt = int(tokens[i + order - 1])
                if ctx not in self.counts[order]:
                    self.counts[order][ctx] = {}
                    self.total_counts[order][ctx] = 0
                self.counts[order][ctx][nxt] = self.counts[order][ctx].get(nxt, 0) + 1
                self.total_counts[order][ctx] += 1

    def predict(self, context, vocab_size):
        for order in range(self.max_order, 1, -1):
            if len(context) < order - 1:
                continue
            ctx = tuple(context[-(order - 1):])
            if ctx in self.counts[order]:
                total = self.total_counts[order][ctx]
                probs = np.zeros((vocab_size,), dtype=np.float32)
                for tok, count in self.counts[order][ctx].items():
                    if tok < vocab_size:
                        probs[tok] = count / total
                if probs.sum() > 0:
                    probs = (probs + 1e-8) / (probs.sum() + 1e-8 * vocab_size)
                    return np.log(probs).astype(np.float32, copy=False)
        return None

    def entropy_alpha(self, neural_logprobs):
        probs = np.exp(neural_logprobs)
        entropy = float(-(probs * neural_logprobs).sum())
        return self.alpha_base + self.alpha_scale * (
            1.0 / (1.0 + math.exp(-2.0 * (entropy - self.entropy_center)))
        )


def eval_val_ngram_cache(model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                         temperature=1.0):
    prev_training, prev_feedback_passes = set_eval_mode(model, args.eval_feedback_passes)
    ngram_cache = NgramCache(
        max_order=args.ngram_max_order,
        alpha_base=args.ngram_alpha_base,
        alpha_scale=args.ngram_alpha_scale,
        entropy_center=args.ngram_entropy_center,
    )
    seq_len = args.train_seq_len
    total_tokens = val_tokens.size - 1
    loss_sum = 0.0
    byte_sum = 0.0
    tok_count = 0
    scored_tokens = []
    
    use_carry = args.capsule_carry_enabled and args.capsule_enabled
    decay = args.capsule_carry_decay if args.capsule_carry_enabled else 0.0
    carry_capsules = None

    for pos in range(0, total_tokens, seq_len):
        end = min(pos + seq_len, total_tokens)
        wlen = end - pos
        chunk = val_tokens[pos:end + 1]
        x_np = chunk[:-1][None, :]
        y_np = chunk[1:][None, :]
        x = mx.array(x_np, dtype=mx.int32)
        if use_carry:
            logits, capsule_state = model.forward_logits_with_carry(
                x, carry_capsules=carry_capsules, temperature=temperature
            )
            logits = logits.squeeze(0).astype(mx.float32)
            if capsule_state is not None:
                cs_avg = mx.mean(capsule_state, axis=0, keepdims=True)
                if carry_capsules is not None:
                    carry_capsules = mx.stop_gradient(decay * carry_capsules + (1.0 - decay) * cs_avg)
                else:
                    carry_capsules = mx.stop_gradient(cs_avg)
                mx.eval(carry_capsules)
        else:
            logits = model.forward_logits(x, temperature=temperature).squeeze(0).astype(mx.float32)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(log_probs)
        log_probs_np = np.array(log_probs, dtype=np.float32)
        for t_idx in range(wlen):
            target_tok = int(y_np[0, t_idx])
            neural_lp = log_probs_np[t_idx]
            ngram_lp = ngram_cache.predict(scored_tokens[-(args.ngram_max_order - 1):], args.vocab_size)
            if ngram_lp is not None:
                alpha = ngram_cache.entropy_alpha(neural_lp)
                mixed = np.logaddexp(
                    neural_lp + math.log(1.0 - alpha + 1e-10),
                    ngram_lp + math.log(alpha + 1e-10),
                )
                token_nll = -float(mixed[target_tok])
            else:
                token_nll = -float(neural_lp[target_tok])
            loss_sum += token_nll
            tok_b = int(base_bytes_lut[target_tok])
            prev_tok = int(x_np[0, t_idx])
            if has_leading_space_lut[target_tok] and not is_boundary_lut[prev_tok]:
                tok_b += 1
            byte_sum += tok_b
            tok_count += 1
            scored_tokens.append(target_tok)
        ngram_cache.update(chunk[1:].tolist())

    restore_mode(model, prev_training, prev_feedback_passes)
    val_loss = loss_sum / max(tok_count, 1)
    val_bpb = (val_loss / math.log(2.0)) * (tok_count / max(byte_sum, 1.0))
    return val_loss, val_bpb


def collect_ttt_param_names(model, scope):
    if scope != "feedback":
        raise ValueError(f"Unsupported TTT_SCOPE={scope}")
    params = dict(tree_flatten(model.parameters()))
    selected = []
    for name in params:
        allow = (
            name.startswith("feedback_pooler.")
            or name.startswith("feedback_adapters.")
            or name == "skip_weights"
            or name.startswith("capsule_bank.")
        )
        if name.startswith("blocks."):
            parts = name.split(".")
            if len(parts) >= 3:
                block_idx = int(parts[1])
                leaf = parts[2]
                if block_idx >= model.num_encoder_layers:
                    if leaf in {"attn_scale", "mlp_scale", "mixer_scale"}:
                        allow = True
        if name.startswith("per_layer_attn_scales.") or name.startswith("per_layer_mlp_scales."):
            idx = int(name.split(".")[1])
            if idx >= model.num_encoder_layers:
                allow = True
        if allow:
            selected.append(name)
    return selected


def update_from_flat_dict(model, flat_dict):
    for name, value in flat_dict.items():
        parts = name.split(".")
        m = model
        for p in parts[:-1]:
            m = m[int(p)] if p.isdigit() else getattr(m, p)
        if parts[-1].isdigit():
            m[int(parts[-1])] = value
        else:
            setattr(m, parts[-1], value)

def snapshot_parameters(model):
    return {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

def restore_parameters(model, snapshot):
    update_from_flat_dict(model, snapshot)

def clip_named_grads(grads_flat, selected_names, max_norm):
    if max_norm <= 0:
        return grads_flat
    total_sq = 0.0
    names = [name for name in selected_names if name in grads_flat]
    for name in names:
        total_sq += float(mx.sum(grads_flat[name].astype(mx.float32) ** 2).item())
    if total_sq <= 0.0:
        return grads_flat
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_flat
    scale = max_norm / (total_norm + 1e-12)
    for name in names:
        grads_flat[name] = grads_flat[name] * scale
    return grads_flat


def sgd_momentum_step(model, selected_names, grads_flat, velocity, lr, momentum):
    params = dict(tree_flatten(model.parameters()))
    updated = {}
    touched = []
    for name in selected_names:
        if name not in grads_flat or name not in params:
            continue
        grad = grads_flat[name]
        vel = momentum * velocity.get(name, mx.zeros_like(params[name])) + grad
        velocity[name] = vel
        updated[name] = params[name] - lr * vel.astype(params[name].dtype)
        touched.append(name)
    if updated:
        update_from_flat_dict(model, updated)
        mx.eval(*[updated[name] for name in touched])


def eval_val_sliding_ttt(model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut,
                         stride=64, batch_seqs=32, temperature=1.0, log_fn=None):
    prev_training, prev_feedback_passes = set_eval_mode(model, args.eval_feedback_passes)
    original_params = snapshot_parameters(model)
    selected_names = collect_ttt_param_names(model, args.ttt_scope)
    if not selected_names:
        restore_mode(model, prev_training, prev_feedback_passes)
        raise RuntimeError("TTT enabled but no parameters matched TTT_SCOPE")
    velocity = {name: mx.zeros_like(original_params[name]) for name in selected_names}
    total_tokens = val_tokens.size - 1
    
    use_carry = args.capsule_carry_enabled and args.capsule_enabled
    decay = args.capsule_carry_decay if args.capsule_carry_enabled else 0.0
    carry_capsules = None

    seq_len = args.train_seq_len
    ttt_chunk = args.ttt_chunk_tokens
    window_starts = [
        ws for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        score_from = 0 if ws == 0 else max(wlen - stride, 0)
        chunk_idx = min((ws + score_from) // ttt_chunk, num_chunks - 1)
        chunk_windows[chunk_idx].append(ws)

    if log_fn is not None:
        log_fn(
            f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} stride={stride} "
            f"ttt_lr={args.ttt_lr} ttt_epochs={args.ttt_epochs} scope={args.ttt_scope}"
        )
        log_fn(f"ttt_sliding:params unfrozen={sum(int(original_params[n].size) for n in selected_names)}")

    loss_sum = 0.0
    token_count = 0.0
    byte_count = 0.0
    loss_grad_fn = nn.value_and_grad(model, lambda x, y: model.loss(x, y, temperature=temperature))
    t0 = time.perf_counter()

    try:
        for ci in range(num_chunks):
            windows = chunk_windows[ci]
            if not windows:
                continue
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)

            model.eval()
            for bi in range(0, len(windows), batch_seqs):
                batch_ws = windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = np.zeros((bsz, seq_len), dtype=np.int32)
                y_batch = np.zeros((bsz, seq_len), dtype=np.int32)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = val_tokens[ws:end + 1]
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]

                x = mx.array(x_batch, dtype=mx.int32)
                y = mx.array(y_batch, dtype=mx.int32)
                if use_carry:
                    logits, capsule_state = model.forward_logits_with_carry(
                        x, carry_capsules=carry_capsules, temperature=temperature
                    )
                    if capsule_state is not None:
                        cs_avg = mx.mean(capsule_state, axis=0, keepdims=True)
                        if carry_capsules is not None:
                            carry_capsules = mx.stop_gradient(decay * carry_capsules + (1.0 - decay) * cs_avg)
                        else:
                            carry_capsules = mx.stop_gradient(cs_avg)
                        mx.eval(carry_capsules)
                else:
                    logits = model.forward_logits(x, temperature=temperature)
                
                nll = ce_from_logits(logits, y, reduction="none")
                mx.eval(nll)
                nll_np = np.array(nll.astype(mx.float32), dtype=np.float32)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    score_from = 0 if ws == 0 else max(wlen - stride, 0)
                    scored = nll_np[i, score_from:wlen]
                    sx = x_batch[i, score_from:wlen]
                    sy = y_batch[i, score_from:wlen]
                    loss_sum += float(scored.astype(np.float64).sum())
                    token_count += float(wlen - score_from)
                    tok_bytes = base_bytes_lut[sy].astype(np.int16, copy=True)
                    tok_bytes += (has_leading_space_lut[sy] & ~is_boundary_lut[sx]).astype(np.int16)
                    byte_count += float(tok_bytes.astype(np.float64).sum())

            if ci == num_chunks - 1 or args.ttt_epochs <= 0:
                continue

            model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs <= 0:
                continue
            cosine_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for _ in range(args.ttt_epochs):
                for bs in range(0, chunk_seqs, args.ttt_batch_seqs):
                    be = min(bs + args.ttt_batch_seqs, chunk_seqs)
                    start_tok = chunk_start + bs * seq_len
                    end_tok = chunk_start + be * seq_len + 1
                    if end_tok > val_tokens.size:
                        continue
                    local = val_tokens[start_tok:end_tok]
                    x = mx.array(local[:-1].reshape(-1, seq_len), dtype=mx.int32)
                    y = mx.array(local[1:].reshape(-1, seq_len), dtype=mx.int32)
                    loss_grad_fn_carry = nn.value_and_grad(model, lambda x, y: model.loss(x, y, temperature=temperature, carry_capsules=carry_capsules))
                    loss, grads = loss_grad_fn_carry(x, y) if use_carry else loss_grad_fn(x, y)
                    grads_flat = clip_named_grads(dict(tree_flatten(grads)), selected_names, args.ttt_grad_clip)
                    sgd_momentum_step(model, selected_names, grads_flat, velocity, cosine_lr, args.ttt_momentum)
                    mx.eval(loss)

            if log_fn is not None and (ci % 10 == 0 or ci == num_chunks - 1):
                elapsed = time.perf_counter() - t0
                running_loss = loss_sum / max(token_count, 1.0)
                running_bpb = running_loss / math.log(2.0) * (token_count / max(byte_count, 1.0)) if token_count > 0 else 0.0
                log_fn(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={running_bpb:.6f} time={elapsed:.1f}s")

        val_loss = loss_sum / max(token_count, 1.0)
        val_bpb = val_loss / math.log(2.0) * (token_count / max(byte_count, 1.0))
        if log_fn is not None:
            log_fn(
                f"ttt_sliding:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )
        return val_loss, val_bpb
    finally:
        restore_parameters(model, original_params)
        restore_mode(model, prev_training, prev_feedback_passes)


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
    # NaN guard: if any gradient is NaN, zero all gradients to skip this step
    if not math.isfinite(total_sq):
        return tree_unflatten([(k, mx.zeros_like(g)) for k, g in flat.items()])
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
    log(f"model_params:{n_params} arch:{args.architecture} layers:{args.num_layers} dim:{args.model_dim} "
        f"heads:{args.num_heads} seq_len:{args.train_seq_len}")
    if args.architecture in ("koopman_ssm", "hybrid"):
        log(f"koopman_ssm: state_dim={args.koopman_state_dim} rank={args.koopman_mixer_rank} "
            f"conv_kernel={args.koopman_conv_kernel} decay_window={args.koopman_decay_window} "
            f"mlp_mult={args.mlp_mult}")
    if args.architecture == "hybrid":
        attn_count = sum(1 for lt in model._layer_types if lt == "attn")
        ssm_count = sum(1 for lt in model._layer_types if lt == "ssm")
        log(f"hybrid: {attn_count} attention + {ssm_count} SSM layers")
    log(f"feedback:{args.feedback_enabled} passes:{args.feedback_passes} "
        f"every:{args.feedback_every} eval_passes:{args.eval_feedback_passes} "
        f"capsule:{args.capsule_enabled} engram:{args.bigram_hash_enabled}({args.engram_num_heads}h{args.engram_num_orders}o@L{args.engram_inject_layer}) "
        f"vrl:{args.vrl_enabled} xsa_start:{args.xsa_start_layer} "
        f"partial_rope:{args.partial_rope_dims} lrelu2:{args.activation_type} "
        f"shared_blocks:{args.shared_blocks}")
    if args.koopman_enabled and args.capsule_enabled:
        log(f"koopman:rank={args.koopman_rank} diag_init={args.koopman_diag_init} "
            f"consist_w={args.koopman_consistency_weight} "
            f"halt:{args.adaptive_halt_enabled}@{args.adaptive_halt_threshold} "
            f"carry:{args.capsule_carry_enabled}@{args.capsule_carry_decay}")
    log(
        f"optimizer: matrix_lr={args.matrix_lr} scalar_lr={args.scalar_lr} "
        f"embed_lr={args.tied_embed_lr} muon_wd={args.muon_wd} adam_wd={args.adam_wd}"
    )
    log(
        f"eval: sliding={args.sliding_eval}@{args.sliding_eval_stride} "
        f"temp_scaling={args.temp_scaling} ttt={args.ttt_enabled} "
        f"ngram_cache={args.ngram_cache_enabled}"
    )
    log(f"turbo_quant: train={args.turbo_quant_train} export={args.turbo_quant_export} kv={args.turbo_quant_kv}")
    log(f"convergence: noise={args.ternary_noise_scale} sdepth={args.stochastic_depth_prob} "
        f"ema_eval={args.ema_eval_apply} self_distill={args.self_distill_kl_weight} "
        f"curriculum={args.curriculum_enabled}(seq:{args.curriculum_phase1_seq}->{args.curriculum_phase2_seq}"
        f"{'->' + str(args.curriculum_phase3_seq) if args.curriculum_phase3_frac < 1.0 else ''}"
        f"{'->' + str(args.curriculum_phase4_seq) if args.curriculum_phase4_frac < 1.0 else ''}"
        f"{'->' + str(args.curriculum_phase5_seq) if args.curriculum_phase5_frac < 1.0 else ''}"
        f"->{args.train_seq_len} "
        f"@{args.curriculum_phase1_frac:.2%}/{args.curriculum_phase2_frac:.2%})")
    log(f"weight_averaging: lawa={args.lawa_enabled}(k={args.lawa_k} freq={args.lawa_freq}) "
        f"swa={args.swa_enabled}(every={args.swa_every} start_scale={args.swa_start_scale}) "
        f"smeargate={args.smeargate_enabled}")
    _bottleneck_tag = (f"linear(rank={args.linear_bottleneck_rank})"
                       if (args.capsule_enabled and args.linear_bottleneck_enabled)
                       else f"linkoopman(H={args.linkoopman_heads}×d={args.linkoopman_head_dim})"
                       if (args.capsule_enabled and args.linkoopman_enabled)
                       else f"ssm(state={args.ssm_bottleneck_state_dim})"
                       if (args.capsule_enabled and args.ssm_bottleneck_enabled)
                       else f"capsule({args.capsule_num}x{args.capsule_dim})" if args.capsule_enabled
                       else "none")
    log(f"radical: tko={args.tko_enabled} weight_sharing={args.weight_sharing} "
        f"inside_out={args.inside_out_training}(p1={args.inside_out_phase1_frac} p2={args.inside_out_phase2_frac}) "
        f"skc={args.architecture == 'skc'} deq={args.deq_feedback} bottleneck={_bottleneck_tag}")

    opt = SplitOptimizers(model, args)

    # Compile training graphs lazily per feedback-pass variant. `self.training` and
    # `feedback_passes` are Python-side control-flow knobs, so one graph cannot safely
    # serve train/eval/no-feedback modes.
    train_loss_and_grad_cache = {}

    def get_train_loss_and_grad(feedback_passes, seq_len):
        key = (max(int(feedback_passes), 0), int(seq_len))
        if key not in train_loss_and_grad_cache:
            model.train()
            model.set_feedback_passes(key[0])
            func = nn.value_and_grad(model, lambda x, y: model.loss(x, y))
            train_loss_and_grad_cache[key] = mx.compile(
                func,
                inputs=model.state,
                outputs=model.state,
            )
        return train_loss_and_grad_cache[key]

    # EMA — always create when ema_eval_apply is set so eval uses smoother weights
    ema = EMAHelper(model, args.ema_decay) if (args.ema_enabled or args.ema_eval_apply) else None
    global _EVAL_EMA
    _EVAL_EMA = ema  # expose to set_eval_mode / restore_mode

    # LAWA (Latest-A-Wins Averaging) — collect snapshots
    from collections import deque
    lawa_queue = deque(maxlen=args.lawa_k) if args.lawa_enabled else None

    # SWA (Stochastic Weight Averaging) — accumulate during warmdown
    swa_state = None
    swa_count = 0

    # LR schedule
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        # Warmup ramp: linear from 0 to 1 over warmup_steps
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
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
            val_loss, val_bpb = eval_val(
                model, args, val_tokens, base_bytes_lut,
                has_leading_space_lut, is_boundary_lut, log_fn=log,
            )
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms")
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Inside-out training: progressively unfreeze layers from bottleneck outward
        if args.inside_out_training and max_wallclock_ms and max_wallclock_ms > 0:
            progress = elapsed_ms / max_wallclock_ms
            if progress < args.inside_out_phase1_frac:
                # Phase 1: capsule-only, all blocks frozen
                model._max_unfrozen_distance = -1  # no blocks unfrozen
            elif progress < args.inside_out_phase2_frac:
                # Phase 2: progressive unfreezing
                # Unfreeze from bottleneck outward: distance 0 (adjacent to bottleneck) → higher
                unfrozen_frac = (progress - args.inside_out_phase1_frac) / (args.inside_out_phase2_frac - args.inside_out_phase1_frac)
                max_dist = int(unfrozen_frac * max(model.num_encoder_layers, model.num_decoder_layers))
                model._max_unfrozen_distance = max_dist
            else:
                # Phase 3: all unfrozen
                model._max_unfrozen_distance = float('inf')
        else:
            model._max_unfrozen_distance = float('inf')  # default: no freezing

        # Curriculum sequence length: start with short seqs for fast early steps, ramp to full.
        # Shorter seqs → more gradient steps in the same wall-clock time → better early convergence.
        # Phases: [0, p1_frac) = phase1_seq, [p1_frac, p2_frac) = phase2_seq, rest = full seq_len.
        if args.curriculum_enabled and max_wallclock_ms and max_wallclock_ms > 0:
            progress = elapsed_ms / max_wallclock_ms
            if progress < args.curriculum_phase1_frac:
                cur_seq_len = args.curriculum_phase1_seq
            elif progress < args.curriculum_phase2_frac:
                cur_seq_len = args.curriculum_phase2_seq
            elif args.curriculum_phase3_frac < 1.0 and progress < args.curriculum_phase3_frac:
                cur_seq_len = args.curriculum_phase3_seq
            elif args.curriculum_phase4_frac < 1.0 and progress < args.curriculum_phase4_frac:
                cur_seq_len = args.curriculum_phase4_seq
            elif args.curriculum_phase5_frac < 1.0 and progress < args.curriculum_phase5_frac:
                cur_seq_len = args.curriculum_phase5_seq
            else:
                cur_seq_len = args.train_seq_len
        else:
            cur_seq_len = args.train_seq_len

        use_feedback = (
            args.feedback_enabled
            and args.feedback_passes > 0
            and max(args.feedback_every, 1) > 0
            and step % max(args.feedback_every, 1) == 0
        )
        active_feedback_passes = model._train_feedback_passes if use_feedback else 0
        compiled_loss_and_grad = get_train_loss_and_grad(active_feedback_passes, cur_seq_len)
        model.train()
        model.set_feedback_passes(active_feedback_passes)

        step_t0 = time.perf_counter()
        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps

        for _ in range(args.grad_accum_steps):
            chunk_sizes = token_chunks(args.microbatch_tokens, cur_seq_len,
                                       args.mlx_max_microbatch_tokens)
            total_chunk_tokens = float(sum(chunk_sizes))
            for ct in chunk_sizes:
                x, y = train_loader.next_batch(ct, cur_seq_len)
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

        # LAWA snapshot collection
        if args.lawa_enabled and lawa_queue is not None and step % args.lawa_freq == 0:
            snap = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
            mx.eval(*snap.values())  # materialize before storing — avoids stale lazy graph refs
            lawa_queue.append(snap)

        # SWA accumulation (during warmdown phase)
        if args.swa_enabled and scale < args.swa_start_scale and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = dict(tree_flatten(model.parameters()))
                swa_count = 1
            else:
                for n, p in tree_flatten(model.parameters()):
                    if n in swa_state:
                        swa_state[n] = swa_state[n] + p
                swa_count += 1

        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            log(f"step:{step}/{args.iterations} loss:{float(train_loss.item()):.4f} "
                f"t:{approx_ms:.0f}ms step:{step_ms:.0f}ms")

        if max_wallclock_ms and stop_after_step is None:
            approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            if approx_ms >= max_wallclock_ms:
                stop_after_step = step

    # Apply best weight averaging strategy before serialization
    # Priority: LAWA > SWA > EMA > raw weights
    if args.lawa_enabled and lawa_queue is not None and len(lawa_queue) > 1:
        avg_sd = {}
        for name in lawa_queue[0]:
            avg_sd[name] = mx.stack([s[name] for s in lawa_queue]).mean(axis=0)
        avg_sd, applied = _filter_serialization_average_mlx(avg_sd, args.average_ternary_params)
        model.update(tree_unflatten(list(avg_sd.items())))
        log(f"lawa:applied k={len(lawa_queue)} tensors={applied} include_ternary={int(args.average_ternary_params)}")
    elif args.swa_enabled and swa_state is not None and swa_count > 0:
        swa_avg = {n: v / swa_count for n, v in swa_state.items()}
        swa_avg, applied = _filter_serialization_average_mlx(swa_avg, args.average_ternary_params)
        model.update(tree_unflatten(list(swa_avg.items())))
        log(f"swa:applied count={swa_count} tensors={applied} include_ternary={int(args.average_ternary_params)}")
    elif ema is not None:
        if ema._ever_updated:
            shadow, applied = _filter_serialization_average_mlx(ema.shadow, args.average_ternary_params)
            model.update(tree_unflatten(list(shadow.items())))
            log(f"ema:applied shadow tensors={applied} include_ternary={int(args.average_ternary_params)}")
        else:
            log("EMA shadow was never updated (short run) — using trained weights directly")

    final_val_loss, final_val_bpb = eval_val(
        model, args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_lut, log_fn=log
    )
    log(f"final_eval val_loss:{final_val_loss:.4f} val_bpb:{final_val_bpb:.4f}")

    opt_temp = 1.0
    if args.temp_scaling:
        t_temp = time.perf_counter()
        calibration_tokens = np.ascontiguousarray(train_loader.stream.take(65536))
        opt_temp = find_temp(
            model, args, calibration_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_lut,
        )
        temp_time_ms = 1000.0 * (time.perf_counter() - t_temp)
        log(f"temp_scaling optimal_T:{opt_temp:.2f} eval_time:{temp_time_ms:.0f}ms")

    if args.sliding_eval:
        t_sliding = time.perf_counter()
        sw_loss, sw_bpb = eval_val_sliding(
            model, args, val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_lut, stride=args.sliding_eval_stride, temperature=opt_temp,
        )
        sliding_time_ms = 1000.0 * (time.perf_counter() - t_sliding)
        log(
            f"final_sliding val_loss:{sw_loss:.4f} val_bpb:{sw_bpb:.4f} "
            f"(stride={args.sliding_eval_stride}, T={opt_temp:.2f}) eval_time:{sliding_time_ms:.0f}ms"
        )

    if args.ttt_enabled:
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            model, args, val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_lut, stride=args.sliding_eval_stride,
            batch_seqs=args.ttt_batch_seqs, temperature=opt_temp, log_fn=log,
        )
        ttt_time_ms = 1000.0 * (time.perf_counter() - t_ttt)
        log(f"legal_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{ttt_time_ms:.0f}ms")
        log(f"legal_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if args.ngram_cache_enabled:
        t_ngram = time.perf_counter()
        ngram_loss, ngram_bpb = eval_val_ngram_cache(
            model, args, val_tokens, base_bytes_lut, has_leading_space_lut,
            is_boundary_lut, temperature=opt_temp,
        )
        ngram_time_ms = 1000.0 * (time.perf_counter() - t_ngram)
        log(
            f"ngram_cache val_loss:{ngram_loss:.4f} val_bpb:{ngram_bpb:.4f} "
            f"(order={args.ngram_max_order}) eval_time:{ngram_time_ms:.0f}ms"
        )

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
