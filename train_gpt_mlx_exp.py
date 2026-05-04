#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
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

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # QAT: simulate int8 quantization noise during training to reduce post-quant penalty.
    # qat_start_frac: fraction of training after which QAT kicks in (0=always, 0.5=second half, 1=never).
    qat_start_frac: float = float(os.environ.get("QAT_START_FRAC", 1.0))

    # Kronecker-structured weights: replace full weight matrices with W = W1 ⊗ W2.
    # Gives ~200× compression per matrix. Radical experiment. 0 = off, 1 = on.
    use_kronecker: bool = bool(int(os.environ.get("USE_KRONECKER", "0")))

    # SwiGLU activation: gated FFN, iso-parameter to relu^2 (3 matrices at 2/3 hidden).
    use_swiglu: bool = bool(int(os.environ.get("USE_SWIGLU", "0")))

    # Nuclear norm regularization: encourages spectrally compact weights that compress
    # better under int8+zlib. Adds lambda * sum(nuclear_norm(W)) to loss.
    nuclear_norm_weight: float = float(os.environ.get("NUCLEAR_NORM_WEIGHT", 0.0))

    # FTLE-lite: track rowwise gradient sensitivity during last frac of training.
    # Used for mixed-precision bit allocation at quantization time.
    ftle_start_frac: float = float(os.environ.get("FTLE_START_FRAC", 0.7))

    # Bounded recurrence: replace unconstrained residual with softmax-gated mixture.
    # tau < 1 bounds the update magnitude, making the system naturally contractive.
    bounded_recurrence: bool = bool(int(os.environ.get("BOUNDED_RECURRENCE", "0")))
    recurrence_tau: float = float(os.environ.get("RECURRENCE_TAU", 0.9))

    # Layer sharing: num_unique_layers controls how many distinct layer parameter sets exist.
    # The model still runs num_layers forward passes, cycling through the unique layers.
    # 0 means no sharing (default baseline behavior).
    num_unique_layers: int = int(os.environ.get("NUM_UNIQUE_LAYERS", 0))

    # Per-virtual-layer scales: when layer sharing is enabled, each virtual layer
    # application gets its own attn_scale / mlp_scale / resid_mix at the GPT level
    # instead of sharing the single set inside the Block.  RingFormer (EMNLP 2025).
    per_layer_scales: bool = bool(int(os.environ.get("PER_LAYER_SCALES",
        "1" if int(os.environ.get("NUM_UNIQUE_LAYERS", 0)) > 0 else "0")))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
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

    # Sliding-window eval: stride>0 enables overlapping windows so each scored
    # token sees nearly full context.  eval_seq_len overrides seq_len at eval time
    # (0 = use train_seq_len).  Both default to off / standard non-overlapping eval.
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 0))    # 0 = no sliding window, e.g. 64 or 256
    eval_seq_len: int = int(os.environ.get("EVAL_SEQ_LEN", 0))  # 0 = use train_seq_len

    # DEQ-style convergence eval: run extra recurrence cycles at eval time until
    # hidden states converge (||x_{n+1} - x_n|| < eps) or max_extra_depth is reached.
    eval_extra_depth: int = int(os.environ.get("EVAL_EXTRA_DEPTH", 0))
    eval_converge_eps: float = float(os.environ.get("EVAL_CONVERGE_EPS", 1e-3))

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


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
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
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
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
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

def apply_qat_roundtrip(model: "GPT", alpha: float = 1.0) -> None:
    """In-place QAT: for large float matrices, blend toward their int8-quantized version.
    alpha=1.0 means full quantize-dequantize. alpha<1 blends (EMA toward quantized)."""
    flat = dict(tree_flatten(model.parameters()))
    updated = {}
    for k, p in flat.items():
        if p.ndim != 2 or p.size <= INT8_KEEP_FLOAT_MAX_NUMEL:
            continue
        if any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS):
            continue
        # Per-row int8 quantize then dequantize
        p_f32 = p.astype(mx.float32)
        abs_max = mx.max(mx.abs(p_f32), axis=1, keepdims=True)
        scale = mx.maximum(abs_max / 127.0, mx.array(1.0 / 127.0))
        q = mx.clip(mx.round(p_f32 / scale), -127, 127)
        deq = (q * scale).astype(p.dtype)
        if alpha >= 1.0:
            updated[k] = deq
        else:
            updated[k] = p + alpha * (deq - p)
    if updated:
        model.update(tree_unflatten(list(updated.items())))


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class KroneckerLinear(nn.Module):
    """Linear layer via Kronecker product: W = W1 ⊗ W2.
    For in_dim=out_dim=512, with factor sizes 32×32 and 16×16:
    Params: 32*32 + 16*16 = 1,280 instead of 512*512 = 262,144 (205× compression).
    The Kronecker product naturally captures multi-scale structure."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Factor into two roughly sqrt-sized components
        # in_dim = in1 * in2, out_dim = out1 * out2
        in1 = int(math.isqrt(in_dim))
        while in_dim % in1 != 0:
            in1 -= 1
        in2 = in_dim // in1
        out1 = int(math.isqrt(out_dim))
        while out_dim % out1 != 0:
            out1 -= 1
        out2 = out_dim // out1
        self.in1, self.in2 = in1, in2
        self.out1, self.out2 = out1, out2
        self.in_dim, self.out_dim = in_dim, out_dim
        # Two small factor matrices
        scale1 = (out1 * in1) ** -0.5
        scale2 = (out2 * in2) ** -0.5
        self.w1 = mx.random.normal((out1, in1)) * scale1
        self.w2 = mx.random.normal((out2, in2)) * scale2

    def __call__(self, x: mx.array) -> mx.array:
        # x: [..., in_dim] → reshape to [..., in1, in2]
        shape = x.shape[:-1]
        x = x.reshape(*shape, self.in1, self.in2).astype(mx.float32)
        # Apply: y = W1 @ x @ W2^T → shape [..., out1, out2]
        y = mx.einsum("...ij,oi,pj->...op", x, self.w1, self.w2)
        return y.reshape(*shape, self.out_dim).astype(COMPUTE_DTYPE)


# Factory: select linear layer type and MLP type based on global config
_USE_KRONECKER = False
_USE_SWIGLU = False

def make_linear(in_dim: int, out_dim: int) -> nn.Module:
    if _USE_KRONECKER:
        return KroneckerLinear(in_dim, out_dim)
    return CastedLinear(in_dim, out_dim)


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = make_linear(dim, dim)
        self.c_k = make_linear(dim, kv_dim)
        self.c_v = make_linear(dim, kv_dim)
        self.proj = make_linear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = make_linear(dim, hidden)
        self.proj = make_linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class SwiGLUMLP(nn.Module):
    # SwiGLU: gated FFN. 3 projections at reduced hidden dim to match param count.
    # hidden = dim * mlp_mult * 2 / 3 (iso-parameter with standard MLP)
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(dim * mlp_mult * 2 / 3)
        # Round to multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = make_linear(dim, hidden)  # gate
        self.w2 = make_linear(dim, hidden)  # value
        self.proj = make_linear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = SwiGLUMLP(dim, mlp_mult) if _USE_SWIGLU else MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array, attn_scale=None, mlp_scale=None, resid_mix=None) -> mx.array:
        if attn_scale is None:
            attn_scale = self.attn_scale
        if mlp_scale is None:
            mlp_scale = self.mlp_scale
        if resid_mix is None:
            resid_mix = self.resid_mix
        mix = resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    # - optional layer sharing: num_unique_layers < num_layers means blocks are reused
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, num_unique_layers: int = 0, per_layer_scales: bool = False,
                 bounded_recurrence: bool = False, recurrence_tau: float = 0.9):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)

        # Layer sharing: create only num_unique_layers distinct blocks, cycle through them
        n_unique = num_unique_layers if num_unique_layers > 0 else num_layers
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for i in range(n_unique)
        ]
        # Build the mapping from virtual layer index -> physical block index.
        # Store via __dict__ directly to avoid MLX module state tracking.
        object.__setattr__(self, '_layer_map', [i % n_unique for i in range(num_layers)])

        # Per-virtual-layer scales (RingFormer, EMNLP 2025).  When layer sharing
        # is active each virtual layer gets its own attn_scale, mlp_scale and
        # resid_mix stored at the GPT level, so the shared Block sees different
        # modulation on every application.
        object.__setattr__(self, '_per_layer_scales', per_layer_scales and num_unique_layers > 0)
        if self._per_layer_scales:
            self.attn_scales = mx.ones((num_layers, dim), dtype=mx.float32)
            self.mlp_scales = mx.ones((num_layers, dim), dtype=mx.float32)
            self.resid_mixes = mx.array(np.stack([
                np.ones((num_layers, dim), dtype=np.float32),
                np.zeros((num_layers, dim), dtype=np.float32),
            ]))  # shape (2, num_layers, dim)

        # Bounded recurrence: softmax-gated mixture with tau damping.
        # Gate has 4 logits per virtual layer: [carry, anchor_x0, attn, mlp]
        object.__setattr__(self, '_bounded_recurrence', bounded_recurrence and num_unique_layers > 0)
        object.__setattr__(self, '_recurrence_tau', recurrence_tau)
        if self._bounded_recurrence:
            # Init: high carry, moderate anchor, small attn/mlp
            init_logits = np.zeros((num_layers, 4), dtype=np.float32)
            init_logits[:, 0] = 2.0   # carry (high)
            init_logits[:, 1] = 0.5   # anchor_x0
            init_logits[:, 2] = 0.0   # attn
            init_logits[:, 3] = 0.0   # mlp
            self.recurrence_gates = mx.array(init_logits)

        # Repeat embeddings: a small learned vector added to block input at each
        # virtual layer. Gives each recurrence cycle a unique "phase signal" so the
        # shared block can distinguish which application it's running as. Cheap
        # symmetry breaking (~num_layers * dim params).
        if num_unique_layers > 0:
            self.repeat_embeds = mx.zeros((num_layers, dim), dtype=mx.float32) * 0.01
        else:
            self.repeat_embeds = None

        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def _apply_block(self, i: int, x: mx.array, x0: mx.array) -> mx.array:
        """Run shared block for virtual layer i with repeat embedding + per-layer scales."""
        # Inject repeat embedding: adds a learned phase signal per virtual layer
        if self.repeat_embeds is not None:
            x_in = x + self.repeat_embeds[i].astype(x.dtype)[None, None, :]
        else:
            x_in = x

        # Compute raw attention and MLP outputs from the block's internals
        if self._bounded_recurrence:
            block = self.blocks[self._layer_map[i]]
            normed = rms_norm(x_in)
            attn_out = block.attn(normed)
            mlp_out = block.mlp(rms_norm(x_in + attn_out))
            # Softmax gate: [carry, anchor_x0, attn, mlp]
            gate = mx.softmax(self.recurrence_gates[i])[None, None, :, None]  # [1,1,4,1]
            tau = self._recurrence_tau
            x = (gate[:, :, 0, :] * x +
                 gate[:, :, 1, :] * x0 +
                 tau * (gate[:, :, 2, :] * attn_out + gate[:, :, 3, :] * mlp_out))
            return x

        if self._per_layer_scales:
            return self.blocks[self._layer_map[i]](
                x_in, x0,
                attn_scale=self.attn_scales[i],
                mlp_scale=self.mlp_scales[i],
                resid_mix=self.resid_mixes[:, i, :],
            )
        return self.blocks[self._layer_map[i]](x_in, x0)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self._apply_block(i, x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self._apply_block(self.num_encoder_layers + i, x, x0)
        return self.final_norm(x)

    def forward_deq(self, input_ids: mx.array, extra_depth: int, eps: float = 1e-3) -> tuple[mx.array, list[float]]:
        """Forward pass with extra recurrence cycles after the standard encoder-decoder.
        Returns (hidden_states, convergence_deltas) where convergence_deltas tracks
        ||x_{n+1} - x_n|| / ||x_n|| at each extra cycle for Lyapunov diagnostics."""
        # Standard forward pass first
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        n_unique = len(self.blocks)
        if self._per_layer_scales:
            for i in range(self.num_encoder_layers):
                x = self.blocks[self._layer_map[i]](
                    x, x0, attn_scale=self.attn_scales[i],
                    mlp_scale=self.mlp_scales[i], resid_mix=self.resid_mixes[:, i, :])
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
                j = self.num_encoder_layers + i
                x = self.blocks[self._layer_map[j]](
                    x, x0, attn_scale=self.attn_scales[j],
                    mlp_scale=self.mlp_scales[j], resid_mix=self.resid_mixes[:, j, :])
        else:
            for i in range(self.num_encoder_layers):
                x = self.blocks[self._layer_map[i]](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self._layer_map[self.num_encoder_layers + i]](x, x0)

        # Extra recurrence cycles — cycle through blocks again, checking convergence
        deltas: list[float] = []
        for cycle in range(extra_depth):
            x_prev = x
            for b in range(n_unique):
                x = self.blocks[b](x, x0)
            # Lyapunov diagnostic: relative change
            diff = mx.sqrt(mx.mean((x - x_prev) ** 2))
            norm = mx.sqrt(mx.mean(x_prev ** 2)) + 1e-8
            delta = float((diff / norm).item())
            deltas.append(delta)
            if delta < eps:
                break

        return self.final_norm(x), deltas

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
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
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

    def loss_per_token(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Return per-token cross-entropy losses (no reduction).

        input_ids: [1, seq_len]   target_ids: [1, seq_len]
        Returns: [seq_len] float32 array of per-position losses.
        """
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")

    def loss_deq(self, input_ids: mx.array, target_ids: mx.array,
                 extra_depth: int, eps: float = 1e-3) -> tuple[mx.array, list[float]]:
        """Loss with extra DEQ recurrence. Returns (mean_loss, convergence_deltas)."""
        x, deltas = self.forward_deq(input_ids, extra_depth, eps)
        x = x.reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        loss = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        return loss, deltas

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
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
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k in ("skip_weights", "attn_scales", "mlp_scales", "resid_mixes", "repeat_embeds", "recurrence_gates")
            or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_row_int4(row: np.ndarray) -> tuple[np.ndarray, np.float16]:
    """Quantize a single row to 4-bit ([-8, 7], 16 levels). Returns packed int8 (2 values per byte) + scale."""
    clip_abs = float(np.quantile(np.abs(row), INT8_CLIP_Q)) if row.size else 0.0
    scale = max(clip_abs / 7.0, 1.0 / 7.0)
    q = np.clip(np.round(row / scale), -8, 7).astype(np.int8)
    # Pack two int4 values into one int8: high nibble | low nibble
    if q.size % 2 != 0:
        q = np.append(q, np.int8(0))  # pad to even
    high = (q[0::2] & 0x0F).astype(np.uint8)
    low = (q[1::2] & 0x0F).astype(np.uint8)
    packed = ((high << 4) | low).astype(np.int8)
    return packed, np.float16(scale)


def dequantize_row_int4(packed: np.ndarray, scale: float, orig_cols: int) -> np.ndarray:
    """Dequantize a 4-bit packed row back to float32."""
    raw = packed.view(np.uint8)
    high = ((raw >> 4) & 0x0F).astype(np.int8)
    low = (raw & 0x0F).astype(np.int8)
    # Sign-extend 4-bit to int8
    high = np.where(high > 7, high.astype(np.int16) - 16, high.astype(np.int16)).astype(np.int8)
    low = np.where(low > 7, low.astype(np.int16) - 16, low.astype(np.int16)).astype(np.int8)
    # Interleave back
    q = np.empty(high.size + low.size, dtype=np.int8)
    q[0::2] = high
    q[1::2] = low
    return (q[:orig_cols].astype(np.float32) * float(scale))


def quantize_float_array_mixed(arr: mx.array, row_sensitivity: np.ndarray | None = None,
                                sensitivity_threshold: float = 0.0) -> tuple[dict, np.ndarray]:
    """Mixed-precision quantization: hot rows → int8, cold rows → int4.
    Returns (row_data_dict, scales) where row_data_dict contains packed arrays."""
    f32 = _np_float32(arr)
    if f32.ndim != 2 or row_sensitivity is None:
        # Fall back to standard int8
        q, s = quantize_float_array(arr)
        return {"type": "uniform_int8", "data": q}, s

    n_rows, n_cols = f32.shape
    is_hot = row_sensitivity >= sensitivity_threshold
    n_hot = int(np.sum(is_hot))

    # Quantize all rows with int8 first (hot rows)
    clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
    clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
    int8_scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32)
    int8_q = np.clip(np.round(clipped / int8_scale[:, None]), -127, 127).astype(np.int8)

    # For cold rows, also compute int4 packed version
    int4_packed_rows = []
    int4_scales = []
    for r in range(n_rows):
        if not is_hot[r]:
            packed, s4 = quantize_row_int4(f32[r])
            int4_packed_rows.append(packed)
            int4_scales.append(s4)

    return {
        "type": "mixed_int8_int4",
        "int8_q": np.ascontiguousarray(int8_q),
        "is_hot": is_hot,
        "int4_packed": int4_packed_rows,
        "int4_scales": np.array(int4_scales, dtype=np.float16) if int4_scales else np.empty(0, dtype=np.float16),
        "n_cols": n_cols,
        "n_hot": n_hot,
    }, np.ascontiguousarray(int8_scale.astype(INT8_PER_ROW_SCALE_DTYPE))


def quantize_state_dict_int8(flat_state: dict[str, mx.array], ftle_data: dict[str, np.ndarray] | None = None) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        # Use FTLE-guided mixed precision if available
        row_sens = ftle_data.get(name) if ftle_data else None
        if row_sens is not None and arr.ndim == 2:
            threshold = float(np.percentile(row_sens, 40))  # bottom 40% → 4-bit
            mixed, s = quantize_float_array_mixed(arr, row_sens, threshold)
            if mixed["type"] == "mixed_int8_int4":
                # For now, store the int8 version for compatibility but track savings
                # (Full mixed-precision serialization requires custom format — we measure the potential)
                n_cold = arr.shape[0] - mixed["n_hot"]
                byte_savings = n_cold * arr.shape[1] // 2  # 4-bit = half the bytes
                stats["mixed_prec_savings"] = stats.get("mixed_prec_savings", 0) + byte_savings
                qmeta[name] = {"scheme": "per_row", "axis": 0, "mixed": True, "n_hot": mixed["n_hot"], "n_cold": n_cold}
            q = mixed.get("int8_q", mixed.get("data"))
        else:
            q, s = quantize_float_array(arr)
        if s.ndim > 0 and name not in qmeta:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def _count_bytes(
    x_np: np.ndarray,
    y_np: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> float:
    """Count the UTF-8 byte total for a batch of (prev, target) token id pairs."""
    prev_ids = x_np.reshape(-1)
    tgt_ids = y_np.reshape(-1)
    bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
    bytes_np += (
        has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
    ).astype(np.int16, copy=False)
    return float(bytes_np.astype(np.float64).sum())


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    compiled_loss_per_token=None,
) -> tuple[float, float]:
    """Validation computes two metrics:
    - val_loss: token cross-entropy (natural log)
    - val_bpb: tokenizer-agnostic compression metric used by the challenge

    If eval_stride > 0, uses a sliding-window approach where each window of
    eval_seq_len tokens overlaps by (eval_seq_len - stride).  Only the last
    `stride` positions in each window are scored (the first window scores all
    positions).  This gives every scored token nearly full context.

    If eval_seq_len != train_seq_len, evaluation uses the longer sequence length
    (RoPE extends naturally).
    """
    seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    stride = args.eval_stride

    # ------------------------------------------------------------------
    # Sliding-window path
    # ------------------------------------------------------------------
    if stride > 0:
        if compiled_loss_per_token is None:
            raise ValueError("eval_stride > 0 requires compiled_loss_per_token")
        if stride > seq_len:
            raise ValueError(f"eval_stride ({stride}) must be <= seq_len ({seq_len})")

        total_val_tokens = val_tokens.size - 1  # number of (input,target) pairs
        total_loss = 0.0
        total_counted = 0
        total_bytes = 0.0

        for start in range(0, total_val_tokens - seq_len + 1, stride):
            end = start + seq_len
            chunk = val_tokens[start : end + 1]  # +1 for target offset
            x = mx.array(chunk[:-1].reshape(1, seq_len), dtype=mx.int32)
            y = mx.array(chunk[1:].reshape(1, seq_len), dtype=mx.int32)

            per_tok = compiled_loss_per_token(x, y)  # [seq_len]
            mx.eval(per_tok)

            # First window: score all positions.  Later: only the last `stride`.
            if start == 0:
                count_from = 0
            else:
                count_from = seq_len - stride

            scored_losses = np.array(per_tok, dtype=np.float64)[count_from:]
            total_loss += float(scored_losses.sum())
            total_counted += len(scored_losses)

            # Byte counts for scored positions only
            x_np = np.array(chunk[:-1], dtype=np.int32)[count_from:]
            y_np = np.array(chunk[1:], dtype=np.int32)[count_from:]
            total_bytes += _count_bytes(
                x_np, y_np, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
            )

        # Handle leftover tokens after the last full window.
        # The main loop scored all positions in [0, last_full_start + seq_len).
        last_full_start = ((total_val_tokens - seq_len) // stride) * stride
        scored_up_to = last_full_start + seq_len  # exclusive
        if scored_up_to < total_val_tokens:
            # Tokens in [scored_up_to, total_val_tokens) haven't been scored.
            # Right-align a window at the very end of the val set.
            tail_start = total_val_tokens - seq_len
            if tail_start >= 0:
                chunk = val_tokens[tail_start : total_val_tokens + 1]
                x = mx.array(chunk[:-1].reshape(1, seq_len), dtype=mx.int32)
                y = mx.array(chunk[1:].reshape(1, seq_len), dtype=mx.int32)
                per_tok = compiled_loss_per_token(x, y)
                mx.eval(per_tok)
                # Score only the positions not already scored
                n_new = total_val_tokens - scored_up_to
                scored_losses = np.array(per_tok, dtype=np.float64)[-n_new:]
                total_loss += float(scored_losses.sum())
                total_counted += n_new
                x_np = np.array(chunk[:-1], dtype=np.int32)[-n_new:]
                y_np = np.array(chunk[1:], dtype=np.int32)[-n_new:]
                total_bytes += _count_bytes(
                    x_np, y_np, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
                )

        if total_counted == 0:
            raise ValueError("Sliding window eval scored 0 tokens")
        val_loss = total_loss / total_counted
        bits_per_token = val_loss / math.log(2.0)
        val_bpb = bits_per_token * (total_counted / total_bytes)
        return val_loss, val_bpb

    # ------------------------------------------------------------------
    # Standard non-overlapping path (original behaviour)
    # ------------------------------------------------------------------
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"seq_len={seq_len}"
        )
    val_batch_seqs = val_batch_tokens // seq_len
    total_seqs = (val_tokens.size - 1) // seq_len
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        total_loss = total_loss + compiled_loss(x, y).astype(mx.float32) * chunk_token_count
        total_tokens += chunk_token_count
        total_bytes += _count_bytes(
            x_np, y_np, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
        )
    total_loss = total_loss / total_tokens
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_tokens = load_validation_tokens(args.val_files, eval_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    global _USE_KRONECKER, _USE_SWIGLU
    _USE_KRONECKER = args.use_kronecker
    _USE_SWIGLU = args.use_swiglu

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        num_unique_layers=args.num_unique_layers,
        per_layer_scales=args.per_layer_scales,
        bounded_recurrence=args.bounded_recurrence,
        recurrence_tau=args.recurrence_tau,
    )
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
    # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
    # Compiling the model-bound functions and capturing the full model state fixes that while still
    # returning gradients only for trainable parameters via nn.value_and_grad(...).
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )
    # Per-token loss used by sliding-window eval (compiled only when needed).
    compiled_loss_per_token = None
    if args.eval_stride > 0:
        compiled_loss_per_token = mx.compile(
            lambda x, y: model.loss_per_token(x, y),
            inputs=model.state,
            outputs=model.state,
        )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"unique_layers:{len(model.blocks)} layer_map:{model._layer_map} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    if args.eval_stride > 0 or args.eval_seq_len > 0:
        log(f"eval_mode:sliding_window eval_seq_len:{eval_seq_len} eval_stride:{args.eval_stride}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
        # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
        # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
        # the loader so measured training still starts from the true init and token window.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        if args.eval_stride > 0 and compiled_loss_per_token is not None:
            # Sliding window: prime the per-token loss graph with one window.
            warm_chunk = val_tokens[: eval_seq_len + 1]
            x_val = mx.array(warm_chunk[:-1].reshape(1, eval_seq_len), dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(1, eval_seq_len), dtype=mx.int32)
            warm_val_loss = compiled_loss_per_token(x_val, y_val)
            mx.eval(warm_val_loss)
            mx.synchronize()
        else:
            val_batch_tokens = args.val_batch_size // args.grad_accum_steps
            if val_batch_tokens < eval_seq_len:
                raise ValueError(
                    "VAL_BATCH_SIZE must provide at least one sequence; "
                    f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                    f"eval_seq_len={eval_seq_len}"
                )
            warm_val_seqs = min(val_batch_tokens // eval_seq_len, (val_tokens.size - 1) // eval_seq_len)
            warm_chunk = val_tokens[: warm_val_seqs * eval_seq_len + 1]
            x_val = mx.array(warm_chunk[:-1].reshape(-1, eval_seq_len), dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(-1, eval_seq_len), dtype=mx.int32)
            warm_val_loss = compiled_loss(x_val, y_val)
            mx.eval(warm_val_loss)
            mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # FTLE-lite: accumulate rowwise gradient EMA for sensitivity-based bit allocation
    ftle_ema: dict[str, np.ndarray] = {}  # key -> rowwise grad norm EMA
    ftle_decay = 0.99

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                compiled_loss_per_token=compiled_loss_per_token,
            )
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        # QAT: periodically snap weights to their int8-quantized values
        if args.qat_start_frac < 1.0:
            progress = step / max(args.iterations, 1)
            if progress >= args.qat_start_frac and step % 10 == 0:
                qat_alpha = min((progress - args.qat_start_frac) / (1.0 - args.qat_start_frac), 1.0)
                apply_qat_roundtrip(model, alpha=qat_alpha)

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())

        # FTLE-lite: accumulate rowwise gradient norms for sensitivity tracking
        progress_ftle = step / max(args.iterations, 1)
        if progress_ftle >= args.ftle_start_frac:
            flat_grads = dict(tree_flatten(grads))
            for k, g in flat_grads.items():
                if g.ndim == 2 and g.size > 65536:
                    row_norms = np.array(mx.sqrt(mx.sum(g * g, axis=1)).tolist(), dtype=np.float32)
                    if k not in ftle_ema:
                        ftle_ema[k] = row_norms
                    else:
                        ftle_ema[k] = ftle_decay * ftle_ema[k] + (1 - ftle_decay) * row_norms

        opt.step(model, grads, step=step, lr_mul=lr_mul)

        # Proximal nuclear norm: spectral soft-thresholding every 50 steps.
        # Shrinks small singular values toward zero → low-rank structure → better compression.
        if args.nuclear_norm_weight > 0 and step % 50 == 0:
            lam = args.nuclear_norm_weight * lr_mul
            flat = dict(tree_flatten(model.parameters()))
            updated = {}
            for k, p in flat.items():
                if p.ndim == 2 and p.size > 65536 and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS):
                    U, S, Vt = mx.linalg.svd(p.astype(mx.float32), stream=mx.cpu)
                    S_shrunk = mx.maximum(S - lam, mx.array(0.0))
                    updated[k] = (U * S_shrunk[None, :]) @ Vt
                    updated[k] = updated[k].astype(p.dtype)
            if updated:
                model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # FTLE-LITE SENSITIVITY REPORT
    # ==============================================================================
    if ftle_ema:
        log(f"ftle_lite: tracked {len(ftle_ema)} weight matrices")
        for k, row_norms in sorted(ftle_ema.items()):
            p10, p50, p90 = np.percentile(row_norms, [10, 50, 90])
            hot_frac = float(np.mean(row_norms > p90 * 0.5))
            log(f"  ftle {k}: p10={p10:.4f} p50={p50:.4f} p90={p90:.4f} hot_rows={hot_frac:.1%}")
        # Summary: how many rows could go to 4-bit vs need 8-bit
        all_norms = np.concatenate([v for v in ftle_ema.values()])
        threshold = np.percentile(all_norms, 75)
        cold_frac = float(np.mean(all_norms < threshold * 0.3))
        log(f"  ftle_summary: {cold_frac:.0%} of rows are cold (candidates for 4-bit)")

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state) if isinstance(v, mx.array)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state, ftle_data=ftle_ema if ftle_ema else None)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )
    if quant_stats.get("mixed_prec_savings", 0) > 0:
        savings = quant_stats["mixed_prec_savings"]
        log(f"ftle_mixed_precision: potential_savings={savings} bytes ({savings/1024:.0f} KB) if cold rows used 4-bit")

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        compiled_loss_per_token=compiled_loss_per_token,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Mixed-precision roundtrip: simulate 4-bit on cold rows, measure actual BPB impact
    if ftle_ema:
        log("ftle_mixed_precision_roundtrip: simulating 4-bit cold rows...")
        mixed_flat = dict(quant_flat)  # start from int8-dequantized weights
        for name, row_sens in ftle_ema.items():
            if name not in mixed_flat:
                continue
            arr = mixed_flat[name]
            if arr.ndim != 2:
                continue
            threshold = float(np.percentile(row_sens, 40))
            is_cold = row_sens < threshold
            # For cold rows: quantize to 4-bit and dequantize (simulate the precision loss)
            arr_np = np.array(arr.astype(mx.float32), dtype=np.float32)
            for r in range(arr_np.shape[0]):
                if is_cold[r]:
                    row = arr_np[r]
                    clip_abs = float(np.max(np.abs(row))) if row.size else 0.0
                    scale = max(clip_abs / 7.0, 1.0 / 7.0)
                    q = np.clip(np.round(row / scale), -8, 7)
                    arr_np[r] = q * scale
            mixed_flat[name] = mx.array(arr_np, dtype=arr.dtype)
        model.update(tree_unflatten(list(mixed_flat.items())))
        m_val_loss, m_val_bpb = eval_val(
            args, compiled_loss, val_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_token_lut,
            compiled_loss_per_token=compiled_loss_per_token,
        )
        log(f"ftle_mixed_4bit_roundtrip val_loss:{m_val_loss:.4f} val_bpb:{m_val_bpb:.4f}")
        log(f"ftle_mixed_4bit_roundtrip_exact val_loss:{m_val_loss:.8f} val_bpb:{m_val_bpb:.8f}")
        # Restore int8 weights for DEQ eval
        model.update(tree_unflatten(list(quant_flat.items())))

    # DEQ convergence eval: run extra recurrence cycles and report Lyapunov diagnostics
    if args.eval_extra_depth > 0 and args.num_unique_layers > 0:
        log(f"deq_eval: extra_depth={args.eval_extra_depth} eps={args.eval_converge_eps}")
        eval_sl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
        deq_val_seqs = min(10, (val_tokens.size - 1) // eval_sl)  # sample a few sequences
        deq_chunk = val_tokens[: deq_val_seqs * eval_sl + 1]
        x_deq = mx.array(deq_chunk[:-1].reshape(-1, eval_sl), dtype=mx.int32)
        y_deq = mx.array(deq_chunk[1:].reshape(-1, eval_sl), dtype=mx.int32)
        all_deltas: list[list[float]] = []
        total_deq_loss = 0.0
        for seq_i in range(x_deq.shape[0]):
            xi = x_deq[seq_i:seq_i+1]
            yi = y_deq[seq_i:seq_i+1]
            loss_i, deltas_i = model.loss_deq(xi, yi, args.eval_extra_depth, args.eval_converge_eps)
            mx.eval(loss_i)
            total_deq_loss += float(loss_i.item())
            all_deltas.append(deltas_i)
        avg_deq_loss = total_deq_loss / max(deq_val_seqs, 1)
        # BPB = (loss / ln2) * (tokens / bytes) — count only the tokens and bytes we actually scored
        scored_tokens = deq_val_seqs * eval_sl
        scored_target_ids = deq_chunk[1:deq_val_seqs * eval_sl + 1]
        scored_prev_ids = deq_chunk[:deq_val_seqs * eval_sl]
        byte_counts = base_bytes_lut[scored_target_ids].astype(np.float64)
        byte_counts += (has_leading_space_lut[scored_target_ids] & ~is_boundary_token_lut[scored_prev_ids]).astype(np.float64)
        total_scored_bytes = float(byte_counts.sum())
        avg_deq_bpb = (avg_deq_loss / math.log(2.0)) * (float(scored_tokens) / max(total_scored_bytes, 1.0))
        # Report convergence trajectory (Lyapunov diagnostic)
        if all_deltas and all_deltas[0]:
            avg_deltas = [sum(d[j] for d in all_deltas if j < len(d)) / sum(1 for d in all_deltas if j < len(d))
                          for j in range(max(len(d) for d in all_deltas))]
            delta_str = " ".join(f"{d:.6f}" for d in avg_deltas)
            log(f"deq_convergence_deltas: [{delta_str}]")
            log(f"deq_converged_at_cycle: {len(avg_deltas)} final_delta:{avg_deltas[-1]:.6f}")
        log(f"deq_eval val_loss:{avg_deq_loss:.4f} approx_val_bpb:{avg_deq_bpb:.4f}")


if __name__ == "__main__":
    main()
