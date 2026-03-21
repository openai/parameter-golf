#!/usr/bin/env python3
"""Starter script for local Parameter Golf research. Keep under 1500 lines."""
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
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    val_seq_len: int = int(os.environ.get("VAL_SEQ_LEN", os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024))))
    val_stride: int = int(os.environ.get("VAL_STRIDE", "0"))
    seq_len_schedule: str = os.environ.get("SEQ_LEN_SCHEDULE", "none")
    seq_len_min: int = int(os.environ.get("SEQ_LEN_MIN", os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024))))
    seq_len_ramp_steps: int = int(os.environ.get("SEQ_LEN_RAMP_STEPS", 0))
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
    depth_share_mode: str = os.environ.get("DEPTH_SHARE_MODE", "none")
    depth_unique_layers: int = int(os.environ.get("DEPTH_UNIQUE_LAYERS", 3))
    depth_step_scale: bool = bool(int(os.environ.get("DEPTH_STEP_SCALE", "1")))
    depth_share_heavy_only: bool = bool(int(os.environ.get("DEPTH_SHARE_HEAVY_ONLY", "0")))
    attnres_mode: str = os.environ.get("ATTNRES_MODE", "full")
    attnres_block_size: int = int(os.environ.get("ATTNRES_BLOCK_SIZE", 3))
    attnres_local_blend: float = float(os.environ.get("ATTNRES_LOCAL_BLEND", 0.5))
    attnres_sublayers: str = os.environ.get("ATTNRES_SUBLAYERS", "both")
    attnres_decoder_last_n: int = int(os.environ.get("ATTNRES_DECODER_LAST_N", 0))
    latent_mem_mode: str = os.environ.get("LATENT_MEM_MODE", "none")
    latent_mem_slots: int = int(os.environ.get("LATENT_MEM_SLOTS", 8))
    latent_mem_layers: int = int(os.environ.get("LATENT_MEM_LAYERS", 2))
    mod_keep: float = float(os.environ.get("MOD_KEEP", "1.0"))
    mod_core: int = int(os.environ.get("MOD_CORE", "1"))
    smeargate: bool = bool(int(os.environ.get("SMEARGATE", "0")))
    smeargate_init: float = float(os.environ.get("SMEARGATE_INIT", "-2.0"))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    mlp_lora_rank: int = int(os.environ.get("MLP_LORA_RANK", "0"))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_weight_decay: float = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")
    load_model_path: str = os.environ.get("LOAD_MODEL_PATH", "")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def train_seq_len_for_step(self, step: int) -> int:
        if self.seq_len_schedule == "none" or self.seq_len_ramp_steps <= 0 or self.seq_len_min >= self.train_seq_len:
            return self.train_seq_len
        if self.seq_len_schedule != "linear":
            raise ValueError(f"SEQ_LEN_SCHEDULE must be none or linear, got {self.seq_len_schedule!r}")
        t = min(step, self.seq_len_ramp_steps) / max(self.seq_len_ramp_steps, 1)
        return max(1, min(self.train_seq_len, int(round(self.seq_len_min + t * (self.train_seq_len - self.seq_len_min)))))

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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,attn_res_query,mlp_res_query,step_scale,step_scales,latent_mem,mlp_lora",
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

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
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
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
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
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class LatentSummaryMemory(nn.Module):
    def __init__(self, dim: int, slots: int):
        super().__init__()
        self.latent_mem_queries = mx.random.normal((slots, dim), dtype=mx.float32) * 0.02
        self.latent_mem_scale = mx.zeros((dim,), dtype=mx.float32)
        self.score_scale = dim ** -0.5

    def summarize(self, x: mx.array) -> mx.array:
        x_norm = rms_norm(x)
        q = self.latent_mem_queries.astype(x.dtype)
        scores = mx.sum(x_norm[:, :, None, :] * q[None, None, :, :], axis=-1) * self.score_scale
        weights = mx.softmax(scores.transpose(0, 2, 1), axis=-1)
        return weights @ x

    def read(self, x: mx.array, memory: mx.array) -> mx.array:
        scores = (rms_norm(x) @ rms_norm(memory).transpose(0, 2, 1)) * self.score_scale
        weights = mx.softmax(scores, axis=-1)
        ctx = weights @ memory
        return x + self.latent_mem_scale.astype(x.dtype)[None, None, :] * ctx


class BlockCore(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn.proj.weight = mx.zeros_like(self.attn.proj.weight)
        self.mlp.proj.weight = mx.zeros_like(self.mlp.proj.weight)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        share_heavy_only: bool = False,
    ):
        super().__init__()
        self.core = None if share_heavy_only else BlockCore(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))
        r = Hyperparameters.mlp_lora_rank
        self.mlp_lora_a = mx.random.normal((r, dim), dtype=mx.float32) * (1.0 / math.sqrt(max(dim, 1))) if r > 0 else None
        self.mlp_lora_b = mx.zeros((dim, r), dtype=mx.float32) if r > 0 else None
        self.mlp_lora_scale = 1.0 / max(r, 1)
        self.attn_res_query = mx.zeros((dim,), dtype=mx.float32)
        self.mlp_res_query = mx.zeros((dim,), dtype=mx.float32)
        self.attn_res_norm = RMSNormNoWeight()
        self.mlp_res_norm = RMSNormNoWeight()

    def _depth_mix(self, outputs: list, query: mx.array, norm) -> mx.array:
        """Softmax-weighted combination of prior layer outputs (Attention Residuals).
        Returns a weighted combination where weights come from softmax attention
        using the learned pseudo-query against RMSNorm'd layer outputs as keys."""
        q = query.astype(outputs[0].dtype)
        stacked = mx.stack(outputs, axis=0)  # (L, B, S, D)
        scores = mx.sum(norm(stacked) * q[None, None, None, :], axis=-1)
        weights = mx.softmax(scores, axis=0)[:, :, :, None]
        return mx.sum(weights * stacked, axis=0)

    def _blend_local(self, local_x: mx.array, mixed_x: mx.array, local_blend: float) -> mx.array:
        if local_blend <= 0.0:
            return mixed_x
        if local_blend >= 1.0:
            return local_x
        return local_x * local_blend + mixed_x * (1.0 - local_blend)

    def __call__(
        self,
        history_outputs: list,
        *,
        attnres_mode: str = "none",
        local_blend: float = 0.0,
        use_attnres_attn: bool = True,
        use_attnres_mlp: bool = True,
        step_scales: mx.array | None = None,
        core: BlockCore | None = None,
        mod_keep: float = 1.0,
    ) -> mx.array:
        core = self.core if core is None else core
        x = history_outputs[-1]  # standard residual base (previous layer output)
        attn_gate = mlp_gate = mx.array(1.0, dtype=x.dtype)
        if step_scales is not None:
            gates = step_scales.astype(x.dtype)
            attn_gate, mlp_gate = gates[0], gates[1]
        x0 = history_outputs[0]
        mix = self.resid_mix.astype(x.dtype)
        attn_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if attnres_mode != "none" and use_attnres_attn:
            attn_in = self._depth_mix(history_outputs, self.attn_res_query, self.attn_res_norm)
            attn_in = self._blend_local(x, attn_in, local_blend)
        attn_out = core.attn(self.attn_norm(attn_in))
        x = x + (attn_gate * self.attn_scale.astype(x.dtype))[None, None, :] * attn_out
        mlp_in = x
        if attnres_mode != "none" and use_attnres_mlp:
            mlp_in = self._depth_mix(history_outputs + [x], self.mlp_res_query, self.mlp_res_norm)
            mlp_in = self._blend_local(x, mlp_in, local_blend)
        mlp_x = self.mlp_norm(mlp_in)
        if mod_keep < 1.0:
            flat = mlp_x.reshape(-1, mlp_x.shape[-1]); k = max(1, int(round(mod_keep * float(flat.shape[0])))); idx = mx.stop_gradient(mx.argsort(mx.mean(mx.abs(flat.astype(mx.float32)), axis=-1))[-k:])
            sel = core.mlp(flat[idx]); mlp_out = mx.put_along_axis(mx.zeros_like(flat), mx.broadcast_to(idx[:, None], sel.shape), sel, axis=0).reshape(mlp_x.shape)
        else: mlp_out = core.mlp(mlp_x)
        if self.mlp_lora_a is not None:
            mlp_out = mlp_out + self.mlp_lora_scale * ((mlp_x @ self.mlp_lora_a.astype(x.dtype).T) @ self.mlp_lora_b.astype(x.dtype).T)
        x = x + (mlp_gate * self.mlp_scale.astype(x.dtype))[None, None, :] * mlp_out
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, depth_share_mode: str, depth_unique_layers: int, depth_step_scale: bool, depth_share_heavy_only: bool,
                 attnres_mode: str, attnres_block_size: int, attnres_local_blend: float,
                 attnres_sublayers: str, attnres_decoder_last_n: int,
                 latent_mem_mode: str, latent_mem_slots: int, latent_mem_layers: int, mod_keep: float, mod_core: int,
                 smeargate: bool, smeargate_init: float):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if depth_share_mode not in {"none", "cycle", "single", "encdec"}:
            raise ValueError(f"DEPTH_SHARE_MODE must be one of none, cycle, single, encdec; got {depth_share_mode!r}")
        if depth_unique_layers <= 0:
            raise ValueError(f"DEPTH_UNIQUE_LAYERS must be positive, got {depth_unique_layers}")
        if attnres_mode not in {"none", "full", "block", "hybrid"}:
            raise ValueError(f"ATTNRES_MODE must be one of none, full, block, hybrid; got {attnres_mode!r}")
        if attnres_block_size <= 0:
            raise ValueError(f"ATTNRES_BLOCK_SIZE must be positive, got {attnres_block_size}")
        if not 0.0 <= attnres_local_blend <= 1.0:
            raise ValueError(f"ATTNRES_LOCAL_BLEND must be in [0, 1], got {attnres_local_blend}")
        if attnres_sublayers not in {"both", "attn", "mlp"}:
            raise ValueError(f"ATTNRES_SUBLAYERS must be one of both, attn, mlp; got {attnres_sublayers!r}")
        if attnres_decoder_last_n < 0:
            raise ValueError(f"ATTNRES_DECODER_LAST_N must be non-negative, got {attnres_decoder_last_n}")
        if latent_mem_mode not in {"none", "summary", "recurrent", "segment"}:
            raise ValueError(f"LATENT_MEM_MODE must be one of none, summary, recurrent, segment; got {latent_mem_mode!r}")
        if latent_mem_slots <= 0:
            raise ValueError(f"LATENT_MEM_SLOTS must be positive, got {latent_mem_slots}")
        if latent_mem_layers < 0:
            raise ValueError(f"LATENT_MEM_LAYERS must be non-negative, got {latent_mem_layers}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.depth_share_mode = depth_share_mode
        self.depth_unique_layers = depth_unique_layers
        self.depth_step_scale = depth_step_scale
        self.depth_share_heavy_only = depth_share_heavy_only
        self.attnres_mode = attnres_mode
        self.attnres_block_size = attnres_block_size
        self.attnres_local_blend = attnres_local_blend
        self.attnres_sublayers = attnres_sublayers
        self.attnres_decoder_last_n = attnres_decoder_last_n
        self.latent_mem_mode = latent_mem_mode
        self.latent_mem_slots = latent_mem_slots
        self.latent_mem_layers = latent_mem_layers
        self.mod_keep = mod_keep
        self.mod_core = mod_core
        self.smeargate_gate = mx.ones((dim,), dtype=mx.float32) * smeargate_init if smeargate else None

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_layers_total = num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.layer_block_indices, num_unique_blocks = self._build_layer_block_indices()
        heads = (HEAD_SCHEDULE or [num_heads]) * num_unique_blocks if len(HEAD_SCHEDULE) <= 1 else HEAD_SCHEDULE
        kv_heads = (KV_HEAD_SCHEDULE or [num_kv_heads]) * num_unique_blocks if len(KV_HEAD_SCHEDULE) <= 1 else KV_HEAD_SCHEDULE
        if len(heads) != num_unique_blocks or len(kv_heads) != num_unique_blocks:
            raise ValueError(f"HEAD_SCHEDULE/KV_HEAD_SCHEDULE must have length 1 or {num_unique_blocks}")
        self.block_cores = (
            [BlockCore(dim, heads[i], kv_heads[i], mlp_mult, rope_base, qk_gain_init) for i in range(num_unique_blocks)]
            if self.depth_share_heavy_only else None
        )
        self.blocks = [Block(dim, num_heads if self.depth_share_heavy_only else heads[i], num_kv_heads if self.depth_share_heavy_only else kv_heads[i], mlp_mult, rope_base, qk_gain_init, share_heavy_only=self.depth_share_heavy_only)
                       for i in range(self.num_layers_total if self.depth_share_heavy_only else num_unique_blocks)]
        self.step_scales = (
            mx.ones((num_layers, 2), dtype=mx.float32)
            if self.depth_step_scale
            else None
        )
        self.latent_mem = LatentSummaryMemory(dim, latent_mem_slots) if self.latent_mem_mode != "none" and self.latent_mem_layers > 0 else None
        self.final_norm = RMSNormNoWeight()

        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def _build_layer_block_indices(self) -> tuple[list[int], int]:
        if self.depth_share_mode == "none":
            return list(range(self.num_layers_total)), self.num_layers_total
        if self.depth_share_mode == "single":
            return [0] * self.num_layers_total, 1
        if self.depth_share_mode == "cycle":
            num_unique = min(self.depth_unique_layers, self.num_layers_total)
            return [i % num_unique for i in range(self.num_layers_total)], num_unique

        enc_unique = min(self.depth_unique_layers, max(self.num_encoder_layers, 1))
        dec_unique = min(self.depth_unique_layers, max(self.num_decoder_layers, 1))
        indices: list[int] = []
        for i in range(self.num_encoder_layers):
            indices.append(i % enc_unique)
        offset = enc_unique
        for i in range(self.num_decoder_layers):
            indices.append(offset + (i % dec_unique))
        return indices, enc_unique + dec_unique

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array, latent_memory: mx.array | None = None) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        if self.smeargate_gate is not None:
            prev = mx.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
            gate = mx.sigmoid(self.smeargate_gate.astype(x.dtype))[None, None, :]
            x = x + gate * (prev - x)
        x = rms_norm(x)
        layer_outputs = [x]  # x0 is entry 0
        block_outputs = [x]
        latent_read = None

        def step_scales_for(layer_idx: int) -> mx.array | None:
            if self.step_scales is None:
                return None
            return self.step_scales[layer_idx]

        def block_for(layer_idx: int) -> Block:
            return self.blocks[layer_idx] if self.depth_share_heavy_only else self.blocks[self.layer_block_indices[layer_idx]]

        def core_for(layer_idx: int) -> BlockCore | None:
            return self.block_cores[self.layer_block_indices[layer_idx]] if self.block_cores is not None else None
        def mod_keep_for(layer_idx: int) -> float:
            return self.mod_keep if self.layer_block_indices[layer_idx] == self.mod_core else 1.0

        def history_for_mode() -> list[mx.array]:
            if self.attnres_mode == "block":
                return block_outputs + [layer_outputs[-1]]
            return layer_outputs

        def local_blend() -> float:
            return self.attnres_local_blend if self.attnres_mode == "hybrid" else 0.0

        def attnres_active(absolute_layer_idx: int) -> bool:
            if self.attnres_mode == "none":
                return False
            if self.attnres_decoder_last_n <= 0:
                return True
            if absolute_layer_idx < self.num_encoder_layers:
                return False
            decoder_idx = absolute_layer_idx - self.num_encoder_layers
            return decoder_idx >= self.num_decoder_layers - self.attnres_decoder_last_n

        def use_attnres_attn_for(absolute_layer_idx: int) -> bool:
            return attnres_active(absolute_layer_idx) and self.attnres_sublayers in {"both", "attn"}

        def use_attnres_mlp_for(absolute_layer_idx: int) -> bool:
            return attnres_active(absolute_layer_idx) and self.attnres_sublayers in {"both", "mlp"}

        for i in range(self.num_encoder_layers):
            block = block_for(i)
            x = block(
                history_for_mode(),
                attnres_mode=self.attnres_mode,
                local_blend=local_blend(),
                use_attnres_attn=use_attnres_attn_for(i),
                use_attnres_mlp=use_attnres_mlp_for(i),
                step_scales=step_scales_for(i),
                core=core_for(i),
                mod_keep=mod_keep_for(i),
            )
            layer_outputs.append(x)
            if self.attnres_mode == "block" and (
                (i + 1) % self.attnres_block_size == 0 or i + 1 == self.num_layers_total
            ):
                block_outputs.append(x)

        if self.latent_mem is not None:
            if self.latent_mem_mode == "summary":
                latent_read = self.latent_mem.summarize(layer_outputs[-1])
            elif self.latent_mem_mode in {"recurrent", "segment"} and latent_memory is not None:
                latent_read = mx.broadcast_to(latent_memory.astype(layer_outputs[-1].dtype), (x.shape[0],) + latent_memory.shape[1:])

        skips = list(layer_outputs[1:self.num_encoder_layers + 1])
        for i in range(self.num_decoder_layers):
            if skips:
                x = layer_outputs[-1] + self.skip_weights[i].astype(layer_outputs[-1].dtype)[None, None, :] * skips.pop()
                layer_outputs[-1] = x
            if latent_read is not None and i >= self.num_decoder_layers - self.latent_mem_layers:
                x = self.latent_mem.read(layer_outputs[-1], latent_read)
                layer_outputs[-1] = x
            absolute_layer_idx = self.num_encoder_layers + i
            block = block_for(absolute_layer_idx)
            x = block(
                history_for_mode(),
                attnres_mode=self.attnres_mode,
                local_blend=local_blend(),
                use_attnres_attn=use_attnres_attn_for(absolute_layer_idx),
                use_attnres_mlp=use_attnres_mlp_for(absolute_layer_idx),
                step_scales=step_scales_for(absolute_layer_idx),
                core=core_for(absolute_layer_idx),
                mod_keep=mod_keep_for(absolute_layer_idx),
            )
            layer_outputs.append(x)
            if self.attnres_mode == "block" and (
                (absolute_layer_idx + 1) % self.attnres_block_size == 0
                or absolute_layer_idx + 1 == self.num_layers_total
            ):
                block_outputs.append(x)

        return self.final_norm(layer_outputs[-1])

    def next_memory(self, input_ids: mx.array, latent_memory: mx.array | None = None) -> mx.array:
        if self.latent_mem is None or self.latent_mem_mode == "none":
            return mx.zeros((1, self.latent_mem_slots, self.tok_emb.weight.shape[1]), dtype=COMPUTE_DTYPE)
        if self.latent_mem_mode == "segment":
            x = self(input_ids, latent_memory=latent_memory)
            tail = x[-1:, -min(self.latent_mem_slots, x.shape[1]):, :]
            if tail.shape[1] == self.latent_mem_slots:
                return tail
            pad = mx.zeros((1, self.latent_mem_slots - tail.shape[1], tail.shape[2]), dtype=tail.dtype)
            return mx.concatenate([pad, tail], axis=1)
        if self.latent_mem_mode != "recurrent":
            return mx.zeros((1, self.latent_mem_slots, self.tok_emb.weight.shape[1]), dtype=COMPUTE_DTYPE)
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        summary = mx.mean(self.latent_mem.summarize(x), axis=0, keepdims=True)
        if latent_memory is None:
            return summary
        return 0.5 * latent_memory.astype(summary.dtype) + 0.5 * summary

    def loss(self, input_ids: mx.array, target_ids: mx.array, latent_memory: mx.array | None = None, reduction: str = "mean") -> mx.array:
        x = self(input_ids, latent_memory=latent_memory).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens > 0 and x.shape[0] > self.logit_chunk_tokens:
            loss_sum = mx.array(0.0, dtype=mx.float32); n = int(x.shape[0]); chunks = [] if reduction == "none" else None
            for s in range(0, n, self.logit_chunk_tokens):
                e = min(s + self.logit_chunk_tokens, n)
                ce = nn.losses.cross_entropy(self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T).astype(mx.float32), y[s:e], reduction="none")
                if chunks is not None: chunks.append(ce)
                else: loss_sum = loss_sum + mx.sum(ce)
            return mx.concatenate(chunks, axis=0) if chunks is not None else loss_sum / float(n)
        logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, y, reduction="none")
        return ce if reduction == "none" else mx.mean(ce)

class Muon:
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
            p_decay = p * (1.0 - lr * self.args.muon_weight_decay) if self.args.muon_weight_decay > 0 else p
            out[k] = p_decay - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # embeddings: Adam, 2D blocks: Muon, small/control tensors: Adam
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k != self.embed_key and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k != self.embed_key and (
                k in {"skip_weights", "step_scales"}
                or p.ndim < 2
                or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            )
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

QUANT_MATRIX_BITS = int(os.environ.get("QUANT_MATRIX_BITS", "8"))
if QUANT_MATRIX_BITS not in {8, 6, 4}:
    raise ValueError(f"QUANT_MATRIX_BITS must be 8, 6, or 4, got {QUANT_MATRIX_BITS}")
QUANT_MATRIX_BITS_OVERRIDES = tuple((pat, int(bits)) for spec in os.environ.get("QUANT_MATRIX_BITS_OVERRIDES", "").split(",") if ":" in spec for pat, bits in [spec.rsplit(":", 1)])
INT8_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("INT8_KEEP_FLOAT_MAX_NUMEL", "65536"))
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", "99.99984"))
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
HEAD_SCHEDULE = [int(x) for x in os.environ.get("HEAD_SCHEDULE", "").split(",") if x]
KV_HEAD_SCHEDULE = [int(x) for x in os.environ.get("KV_HEAD_SCHEDULE", "").split(",") if x]


def quant_label() -> str:
    base = "int8" if QUANT_MATRIX_BITS == 8 else f"m{QUANT_MATRIX_BITS}"
    return f"{base}mix" if QUANT_MATRIX_BITS_OVERRIDES else base


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))
def pack_nbits(values: np.ndarray, bits: int) -> tuple[np.ndarray, tuple[int, ...]]:
    q = np.asarray(values, dtype=np.int8, order="C")
    flat = q.reshape(-1).astype(np.uint8, copy=False) + (1 << (bits - 1))
    bits_u8 = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.ascontiguousarray(np.packbits(bits_u8, bitorder="little")), q.shape
def unpack_nbits(packed: np.ndarray, orig_shape: tuple[int, ...], bits: int) -> np.ndarray:
    needed = int(np.prod(orig_shape))
    bits_u8 = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[: needed * bits].reshape(needed, bits)
    flat = (bits_u8.astype(np.uint8) * (1 << np.arange(bits, dtype=np.uint8))).sum(axis=1, dtype=np.uint16)
    return np.ascontiguousarray((flat.astype(np.int16) - (1 << (bits - 1))).astype(np.int8).reshape(orig_shape))
def matrix_bits_for_name(name: str) -> int:
    return next((bits for pattern, bits in QUANT_MATRIX_BITS_OVERRIDES if pattern in name), QUANT_MATRIX_BITS)
def quantize_float_array(arr: mx.array, bits: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        qmax = 127 if bits == 8 else (1 << (bits - 1)) - 1
        scale = np.maximum(clip_abs / qmax, 1.0 / qmax).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -qmax, qmax).astype(np.int8, copy=False)
        meta: dict[str, object] = {"scheme": "per_row", "axis": 0, "bits": bits}
        if bits < 8:
            packed, orig_shape = pack_nbits(q, bits)
            meta["packed"] = True
            meta["orig_shape"] = orig_shape
            return packed, np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)), meta
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)), meta

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale, {"bits": 8}


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
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
        q, s, meta = quantize_float_array(arr, matrix_bits_for_name(name))
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": f"{quant_label()}_clean_per_row_v1",
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
        meta = qmeta.get(name, {})
        if meta.get("packed"):
            q_np = unpack_nbits(q, tuple(meta["orig_shape"]), int(meta["bits"]))
        else:
            q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if meta.get("scheme") == "per_row" or scale.ndim > 0:
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


def maybe_limit_validation_tokens(tokens: np.ndarray, seq_len: int, max_tokens: int) -> np.ndarray:
    if max_tokens <= 0:
        return tokens
    usable = min(((max_tokens // seq_len) * seq_len), tokens.size - 1)
    if usable <= 0:
        raise ValueError(f"VAL_MAX_TOKENS={max_tokens} is too small for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    compiled_next_memory,
    latent_memory: mx.array,
    seq_len: int,
) -> tuple[mx.array, dict, mx.array]:
    chunk_sizes = token_chunks(args.microbatch_tokens, seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, seq_len)
        loss, grads = compiled_loss_and_grad(x, y, latent_memory)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.latent_mem_mode in {"recurrent", "segment"}:
            latent_memory = compiled_next_memory(x, latent_memory)
    return loss_value, tree_unflatten(list(grad_accum.items())), latent_memory


def is_state_array_leaf(value: object) -> bool:
    return hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "nbytes")


def load_flat_state_npz(path: Path) -> dict[str, mx.array]:
    if not hasattr(mx, "load"):
        raise RuntimeError("mlx.core does not expose mx.load; cannot load raw .npz checkpoints safely")
    loaded = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mx.load({path}) to return a dict, got {type(loaded)!r}")
    return {str(k): v for k, v in loaded.items()}


def init_latent_memory(args: Hyperparameters) -> mx.array:
    return mx.zeros((1, max(args.latent_mem_slots, 1), args.model_dim), dtype=COMPUTE_DTYPE)

def eval_val(
    args: Hyperparameters,
    compiled_loss,
    compiled_loss_tokens,
    compiled_next_memory,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    # Validation reports cross-entropy and tokenizer-agnostic val_bpb.
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.val_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"VAL_SEQ_LEN={args.val_seq_len}"
        )
    if 0 < args.val_stride < args.val_seq_len:
        if args.latent_mem_mode != "none": raise ValueError("Sliding-window eval does not support latent memory modes yet")
        total_loss = total_tokens = total_bytes = 0.0; last_scored = -1; last_start = max(val_tokens.size - 1 - args.val_seq_len, 0)
        starts = list(range(0, last_start + 1, args.val_stride))
        if starts[-1] != last_start: starts.append(last_start)
        for start in starts:
            chunk = val_tokens[start : start + args.val_seq_len + 1]; x_np = chunk[:-1][None, :]; y_np = chunk[1:][None, :]
            losses = np.array(compiled_loss_tokens(mx.array(x_np, dtype=mx.int32), mx.array(y_np, dtype=mx.int32), init_latent_memory(args)), dtype=np.float32)
            score_from = max(last_scored + 1 - start, 0); last_scored = start + args.val_seq_len - 1
            prev_ids = x_np.reshape(-1)[score_from:]; tgt_ids = y_np.reshape(-1)[score_from:]; loss_np = losses[score_from:]
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True); bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16, copy=False)
            total_loss += float(loss_np.astype(np.float64).sum()); total_tokens += float(loss_np.size); total_bytes += float(bytes_np.astype(np.float64).sum())
        val_loss = total_loss / total_tokens; bits_per_token = val_loss / math.log(2.0); return val_loss, bits_per_token * (total_tokens / total_bytes)
    val_batch_seqs = val_batch_tokens // args.val_seq_len
    total_seqs = (val_tokens.size - 1) // args.val_seq_len
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    latent_memory = init_latent_memory(args)
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.val_seq_len
        raw_end = batch_seq_end * args.val_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.val_seq_len)
        y_np = chunk[1:].reshape(-1, args.val_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        total_loss = total_loss + compiled_loss(x, y, latent_memory).astype(mx.float32) * chunk_token_count
        if args.latent_mem_mode in {"recurrent", "segment"}:
            latent_memory = compiled_next_memory(x, latent_memory)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
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
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)
    if min(args.train_seq_len, args.val_seq_len, args.seq_len_min) <= 0:
        raise ValueError("TRAIN_SEQ_LEN, VAL_SEQ_LEN, and SEQ_LEN_MIN must be positive")
    if args.seq_len_min > args.train_seq_len:
        raise ValueError(f"SEQ_LEN_MIN={args.seq_len_min} must be <= TRAIN_SEQ_LEN={args.train_seq_len}")

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
    val_tokens = load_validation_tokens(args.val_files, args.val_seq_len)
    val_tokens = maybe_limit_validation_tokens(val_tokens, args.val_seq_len, args.val_max_tokens)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

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
        depth_share_mode=args.depth_share_mode,
        depth_unique_layers=args.depth_unique_layers,
        depth_step_scale=args.depth_step_scale,
        depth_share_heavy_only=args.depth_share_heavy_only,
        attnres_mode=args.attnres_mode,
        attnres_block_size=args.attnres_block_size,
        attnres_local_blend=args.attnres_local_blend,
        attnres_sublayers=args.attnres_sublayers,
        attnres_decoder_last_n=args.attnres_decoder_last_n,
        latent_mem_mode=args.latent_mem_mode,
        latent_mem_slots=args.latent_mem_slots,
        latent_mem_layers=args.latent_mem_layers,
        mod_keep=args.mod_keep,
        mod_core=args.mod_core,
        smeargate=args.smeargate,
        smeargate_init=args.smeargate_init,
    )
    if args.load_model_path:
        load_path = Path(args.load_model_path)
        if not load_path.exists():
            raise FileNotFoundError(f"LOAD_MODEL_PATH does not exist: {load_path}")
        model.update(tree_unflatten(list(load_flat_state_npz(load_path).items())))
    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y, m: model.loss(x, y, m), inputs=model.state, outputs=model.state)
    compiled_loss_tokens = mx.compile(lambda x, y, m: model.loss(x, y, m, reduction="none"), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y, m: model.loss(x, y, m)),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_next_memory = mx.compile(lambda x, m: model.next_memory(x, m), inputs=model.state, outputs=model.state)

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    if args.load_model_path:
        log(f"load_model_path:{args.load_model_path}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if args.val_max_tokens > 0:
        log(f"val_loader:limited_tokens:{val_tokens.size - 1}")
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
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"train_seq_len:{args.train_seq_len} val_seq_len:{args.val_seq_len} val_stride:{args.val_stride} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(
        f"seq_len_schedule:{args.seq_len_schedule} "
        f"seq_len_min:{args.seq_len_min} "
        f"seq_len_ramp_steps:{args.seq_len_ramp_steps}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"depth_share_mode:{args.depth_share_mode} "
        f"depth_unique_layers:{args.depth_unique_layers} "
        f"depth_step_scale:{int(args.depth_step_scale)} "
        f"depth_share_heavy_only:{int(args.depth_share_heavy_only)} "
        f"block_wrappers:{len(model.blocks)} unique_cores:{len(model.block_cores) if model.block_cores is not None else len(model.blocks)}"
    )
    if HEAD_SCHEDULE or KV_HEAD_SCHEDULE: log(f"head_schedule:{HEAD_SCHEDULE or [args.num_heads]} kv_head_schedule:{KV_HEAD_SCHEDULE or [args.num_kv_heads]}")
    log(
        f"attnres_mode:{args.attnres_mode} "
        f"attnres_block_size:{args.attnres_block_size} "
        f"attnres_local_blend:{args.attnres_local_blend:.3f} "
        f"attnres_sublayers:{args.attnres_sublayers} "
        f"attnres_decoder_last_n:{args.attnres_decoder_last_n}"
    )
    log(
        f"latent_mem_mode:{args.latent_mem_mode} "
        f"latent_mem_slots:{args.latent_mem_slots} "
        f"latent_mem_layers:{args.latent_mem_layers} "
        f"mod_keep:{args.mod_keep} mod_core:{args.mod_core} "
        f"smeargate:{int(args.smeargate)} smeargate_init:{args.smeargate_init}"
    )
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(
        f"quantization:label:{quant_label()} matrix_bits:{QUANT_MATRIX_BITS} "
        f"keep_float_max_numel:{INT8_KEEP_FLOAT_MAX_NUMEL} clip_percentile:{INT8_CLIP_PERCENTILE}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    sample_core = model.block_cores[0] if model.block_cores is not None else model.blocks[0].core
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{sample_core.attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            warm_memory = init_latent_memory(args)
            warm_seq_len = args.train_seq_len_for_step(warmup_step)
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads, warm_memory = loss_and_grad_chunked(
                    args, train_loader, compiled_loss_and_grad, compiled_next_memory, warm_memory, warm_seq_len
                )
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum, warm_memory)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.val_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"VAL_SEQ_LEN={args.val_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.val_seq_len, (val_tokens.size - 1) // args.val_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.val_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.val_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.val_seq_len), dtype=mx.int32)
        warm_memory = init_latent_memory(args)
        warm_val_loss = compiled_loss(x_val, y_val, warm_memory)
        warm_next_memory = compiled_next_memory(x_val, warm_memory)
        mx.eval(warm_val_loss, warm_next_memory)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    latent_memory = init_latent_memory(args)
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                compiled_loss_tokens,
                compiled_next_memory,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
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

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        step_seq_len = args.train_seq_len_for_step(step)
        for _ in range(args.grad_accum_steps):
            loss, grads, latent_memory = loss_and_grad_chunked(
                args, train_loader, compiled_loss_and_grad, compiled_next_memory, latent_memory, step_seq_len
            )
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f} "
                f"seq_len:{step_seq_len}"
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # Final serialization plus quantized roundtrip validation.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    raw_state_items = list(tree_flatten(model.state))
    flat_state = {k: v for k, v in raw_state_items if is_state_array_leaf(v)}
    dropped_state_keys = [k for k, v in raw_state_items if not is_state_array_leaf(v)]
    if dropped_state_keys:
        log(f"state_export:dropped_nonarray_keys:{','.join(dropped_state_keys)}")
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.{quant_label()}.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_{quant_label()}_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        compiled_loss_tokens,
        compiled_next_memory,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_{quant_label()}_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_{quant_label()}_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
