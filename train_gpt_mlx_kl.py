#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm
import zstandard as zstd

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16


class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3_500))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1_500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd: float = float(os.environ.get("MUON_WD", 0.04))
    adam_wd: float = float(os.environ.get("ADAM_WD", 0.04))

    bigram_hash_size: int = int(os.environ.get("BIGRAM_HASH_SIZE", 16_384))
    use_ortho_init: bool = bool(int(os.environ.get("USE_ORTHO_INIT", "1")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.995))
    ema_start_frac: float = float(os.environ.get("EMA_START_FRAC", 0.5))
    late_qat_threshold: float = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    learned_scales: bool = bool(int(os.environ.get("LEARNED_SCALES", "1")))
    eval_seq_len: int = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))

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
            if warmdown_start <= step < self.iterations:
                return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,log_qscale",
    ).split(",")
    if pattern
)
INT6_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT6_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_KEEP_FLOAT_STORE_DTYPE = np.float16
INT6_PER_ROW_SCALE_DTYPE = np.float16
MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


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


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
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


def fake_quant_int6(w: mx.array) -> mx.array:
    scale = mx.maximum(mx.max(mx.abs(w), axis=-1, keepdims=True) / 31.0, mx.array(1e-8, dtype=mx.float32))
    w_q = mx.clip(mx.round(w / scale), -32, 31) * scale
    return w + mx.stop_gradient(w_q - w)


def pack_int6_np(arr_int8: np.ndarray) -> np.ndarray:
    flat = (arr_int8.reshape(-1).astype(np.int16) + 32).astype(np.uint8)
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.uint8)])
    a, b, c, d = flat[0::4], flat[1::4], flat[2::4], flat[3::4]
    b0 = (a << 2) | (b >> 4)
    b1 = ((b & 0x0F) << 4) | (c >> 2)
    b2 = ((c & 0x03) << 6) | d
    return np.ascontiguousarray(np.stack([b0, b1, b2], axis=-1).reshape(-1))


def unpack_int6_np(packed: np.ndarray, numel: int) -> np.ndarray:
    packed = packed.reshape(-1, 3)
    b0, b1, b2 = packed[:, 0], packed[:, 1], packed[:, 2]
    a = b0 >> 2
    b = ((b0 & 0x03) << 4) | (b1 >> 4)
    c = ((b1 & 0x0F) << 2) | (b2 >> 6)
    d = b2 & 0x3F
    flat = np.stack([a, b, c, d], axis=-1).reshape(-1)[:numel]
    return (flat.astype(np.int16) - 32).astype(np.int8)


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


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        base = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = base.weight.astype(mx.float32)
        init_scale = mx.maximum(
            mx.max(mx.abs(self.weight), axis=-1, keepdims=True) / 31.0,
            mx.array(1e-5, dtype=mx.float32),
        )
        self.log_qscale = mx.log(init_scale)
        self.use_qat = False
        self.use_learned_scales = False

    def __call__(self, x: mx.array) -> mx.array:
        if self.use_qat:
            if self.use_learned_scales:
                s = mx.clip(mx.exp(self.log_qscale), 1e-5, 1.0)
                w_q = mx.clip(mx.round(self.weight / s), -32, 31) * s
                w = self.weight + mx.stop_gradient(w_q - self.weight)
            else:
                w = fake_quant_int6(self.weight)
        else:
            w = self.weight
        return x @ w.astype(x.dtype).T


class BigramHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, vocab_size: int):
        super().__init__()
        self.hash_size = hash_size
        self.table = nn.Embedding(hash_size, vocab_size)
        self.table.weight = (mx.random.normal(self.table.weight.shape, dtype=mx.float32) * 0.02).astype(mx.float32)

    def __call__(self, tokens: mx.array) -> mx.array:
        prev = tokens[:, :-1]
        curr = tokens[:, 1:]
        idx = mx.remainder(prev * 31337 + curr, self.hash_size)
        bias = self.table(idx)
        pad = mx.zeros((tokens.shape[0], 1, bias.shape[-1]), dtype=bias.dtype)
        return mx.concatenate([pad, bias], axis=1)


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
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


def ortho_init_matrix(weight: mx.array) -> mx.array:
    w = np.array(weight.astype(mx.float32), dtype=np.float64, copy=True)
    u, _, vt = np.linalg.svd(w, full_matrices=False)
    return mx.array((u @ vt).astype(np.float32, copy=False) * 0.5, dtype=mx.float32)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tied_embed_init_std: float,
        qk_gain_init: float,
        bigram_hash_size: int,
        use_ortho_init: bool,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.bigram_hash = BigramHashEmbedding(bigram_hash_size, vocab_size) if bigram_hash_size > 0 else None

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        if use_ortho_init:
            for block in self.blocks:
                block.attn.c_q.weight = ortho_init_matrix(block.attn.c_q.weight)
                block.attn.c_k.weight = ortho_init_matrix(block.attn.c_k.weight)
                block.attn.c_v.weight = ortho_init_matrix(block.attn.c_v.weight)
                block.mlp.fc.weight = ortho_init_matrix(block.mlp.fc.weight)
        for block in self.blocks:
            block.attn.proj.weight = mx.zeros_like(block.attn.proj.weight)
            block.mlp.proj.weight = mx.zeros_like(block.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def forward_logits(self, input_ids: mx.array) -> mx.array:
        hidden = self(input_ids)
        logits = hidden @ self.tok_emb.weight.astype(hidden.dtype).T
        logits = self.softcap(logits)
        if self.bigram_hash is not None:
            logits = logits + self.bigram_hash(input_ids).astype(logits.dtype)
        return logits

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        bigram_bias = None
        if self.bigram_hash is not None:
            bigram_bias = self.bigram_hash(input_ids).reshape(-1, self.vocab_size)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
            if bigram_bias is not None:
                logits = logits + bigram_bias.astype(logits.dtype)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            if bigram_bias is not None:
                logits = logits + bigram_bias[s:e].astype(logits.dtype)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


class EMABuffer:
    def __init__(self, flat_params: dict[str, mx.array], decay: float):
        self.decay = decay
        self.dtypes = {k: v.dtype for k, v in flat_params.items()}
        self.shadow = {k: _np_float32(v) for k, v in flat_params.items()}

    def update(self, flat_params: dict[str, mx.array]) -> None:
        d = self.decay
        for k, v in flat_params.items():
            self.shadow[k] = d * self.shadow[k] + (1.0 - d) * _np_float32(v)

    def as_mlx(self) -> dict[str, mx.array]:
        return {k: mx.array(v, dtype=self.dtypes[k]) for k, v in self.shadow.items()}


def all_casted_linears(model: GPT) -> list[CastedLinear]:
    layers: list[CastedLinear] = []
    for block in model.blocks:
        layers.extend(
            [
                block.attn.c_q,
                block.attn.c_k,
                block.attn.c_v,
                block.attn.proj,
                block.mlp.fc,
                block.mlp.proj,
            ]
        )
    return layers


def capture_qat_flags(model: GPT) -> list[tuple[bool, bool]]:
    return [(layer.use_qat, layer.use_learned_scales) for layer in all_casted_linears(model)]


def restore_qat_flags(model: GPT, flags: list[tuple[bool, bool]]) -> None:
    for layer, (use_qat, use_learned_scales) in zip(all_casted_linears(model), flags, strict=True):
        layer.use_qat = use_qat
        layer.use_learned_scales = use_learned_scales


def set_qat_mode(model: GPT, enabled: bool, use_learned_scales: bool) -> None:
    for layer in all_casted_linears(model):
        layer.use_qat = enabled
        layer.use_learned_scales = enabled and use_learned_scales


def flatten_params(model: GPT) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(model.parameters())}


def update_model_params(model: GPT, flat_params: dict[str, mx.array]) -> None:
    model.update(tree_unflatten(list(flat_params.items())))


def compile_model_fns(model: GPT):
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state,
        outputs=model.state,
    )
    return compiled_loss, compiled_loss_and_grad


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
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
            out[k] = out[k] - lr * self.args.muon_wd * p
        return out


MUON_WEIGHT_SUFFIXES = (
    ".attn.c_q.weight",
    ".attn.c_k.weight",
    ".attn.c_v.weight",
    ".attn.proj.weight",
    ".mlp.fc.weight",
    ".mlp.proj.weight",
)


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = flatten_params(model)
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k in params
            if k == "bigram_hash.table.weight" or any(k.endswith(suffix) for suffix in MUON_WEIGHT_SUFFIXES)
        ]
        self.scalar_keys = [k for k in params if k not in self.matrix_keys and k != self.embed_key]
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

    def _apply_adam_weight_decay(
        self,
        updated: dict[str, mx.array],
        params: dict[str, mx.array],
        keys: list[str],
        lr: float,
    ) -> None:
        if self.args.adam_wd <= 0.0:
            return
        for k in keys:
            updated[k] = updated[k] - lr * self.args.adam_wd * params[k]

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = flatten_params(model)
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        embed_lr = self.args.tied_embed_lr * lr_mul
        self.adam_embed.learning_rate = embed_lr
        embed_updated = self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        )
        self._apply_adam_weight_decay(embed_updated, params, [self.embed_key], embed_lr)
        updated.update(embed_updated)

        scalar_lr = self.args.scalar_lr * lr_mul
        self.adam_scalar.learning_rate = scalar_lr
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        scalar_updated = self.adam_scalar.apply_gradients(scalar_grads, scalar_params)
        self._apply_adam_weight_decay(scalar_updated, params, self.scalar_keys, scalar_lr)
        updated.update(scalar_updated)

        model.update(tree_unflatten(list(updated.items())))


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT6_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT6_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_state_dict_int6(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    numels: dict[str, int] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int6_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int6_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if arr.ndim == 2 and int(arr.size) > INT6_KEEP_FLOAT_MAX_NUMEL:
            f32 = _np_float32(arr)
            log_qscale_name = f"{name[:-7]}.log_qscale" if name.endswith(".weight") else ""
            if log_qscale_name and log_qscale_name in flat_state:
                scale = np.clip(np.exp(_np_float32(flat_state[log_qscale_name])), 1e-5, 1.0)
            else:
                scale = np.maximum(np.max(np.abs(f32), axis=-1, keepdims=True) / 31.0, 1e-5)
            q = np.clip(np.round(f32 / scale), -32, 31).astype(np.int8, copy=False)
            packed = pack_int6_np(q)
            quantized[name] = packed
            scales[name] = np.ascontiguousarray(scale.astype(INT6_PER_ROW_SCALE_DTYPE, copy=False))
            dtypes[name] = str(arr.dtype).split(".")[-1]
            shapes[name] = tuple(int(x) for x in f32.shape)
            numels[name] = int(f32.size)
            stats["num_float_tensors"] += 1
            stats["int6_payload_bytes"] += int(packed.nbytes + scales[name].nbytes)
            continue
        kept = keep_float_array(name, arr, passthrough_orig_dtypes)
        passthrough[name] = kept
        stats["int6_payload_bytes"] += int(kept.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int6_zstd_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "shapes": shapes,
        "numels": numels,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int6(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, packed in quant_obj["quantized"].items():
        dtype_name = quant_obj["dtypes"][name]
        shape = tuple(int(x) for x in quant_obj["shapes"][name])
        numel = int(quant_obj["numels"][name])
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        q_np = unpack_int6_np(np.asarray(packed, dtype=np.uint8), numel).reshape(shape)
        out_arr = q_np.astype(np.float32) * scale.reshape((shape[0],) + (1,) * (len(shape) - 1))
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
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
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss_fn,
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    del compiled_loss_fn
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
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
        batch_loss = model.loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        chunk_token_count = float(y.size)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val_sliding(
    args: Hyperparameters,
    compiled_loss_fn,
    model: GPT,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    del compiled_loss_fn
    seq_len = args.eval_seq_len
    stride = args.eval_stride
    total_tokens = val_tokens.size - 1
    if total_tokens < seq_len:
        raise ValueError(f"Validation split is too short for EVAL_SEQ_LEN={seq_len}")
    starts = list(range(0, total_tokens - seq_len + 1, stride))
    final_start = total_tokens - seq_len
    if starts[-1] != final_start:
        starts.append(final_start)
    total_loss_sum = 0.0
    total_scored_tokens = 0.0
    total_bytes = 0.0
    scored_until = 0
    for window_idx, start in enumerate(starts):
        end = start + seq_len
        chunk = val_tokens[start : end + 1]
        x = mx.array(chunk[:-1].reshape(1, seq_len), dtype=mx.int32)
        y = mx.array(chunk[1:].reshape(1, seq_len), dtype=mx.int32)
        logits = model.forward_logits(x)
        logits_flat = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
        targets_flat = y.reshape(-1)
        per_token_loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction="none")
        mx.eval(per_token_loss)
        per_token_loss_np = np.array(per_token_loss)
        score_start = 0 if window_idx == 0 else max(scored_until - start, 0)
        scored_losses = per_token_loss_np[score_start:]
        scored_targets = np.array(y.reshape(-1))[score_start:]
        scored_prevs = np.array(x.reshape(-1))[score_start:]
        bytes_np = base_bytes_lut[scored_targets].astype(np.int16)
        bytes_np += (has_leading_space_lut[scored_targets] & ~is_boundary_token_lut[scored_prevs]).astype(np.int16)
        total_loss_sum += float(np.sum(scored_losses))
        total_scored_tokens += float(len(scored_losses))
        total_bytes += float(np.sum(bytes_np.astype(np.float64)))
        scored_until = end
        if log_fn is not None and window_idx % 200 == 0:
            log_fn(f"sliding_progress:{start}/{total_tokens}")
    val_loss = total_loss_sum / total_scored_tokens
    val_bpb = (val_loss / math.log(2.0)) * (total_scored_tokens / total_bytes)
    return val_loss, val_bpb


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


def estimate_total_steps(args: Hyperparameters, step: int, elapsed_ms: float) -> int:
    if args.max_wallclock_seconds <= 0 or step <= 0 or elapsed_ms <= 0.0:
        return args.iterations
    step_ms = elapsed_ms / step
    wallclock_steps = max(1, int((1000.0 * args.max_wallclock_seconds) / max(step_ms, 1e-9)))
    return min(args.iterations, wallclock_steps)


def should_start_qat(args: Hyperparameters, step: int, lr_mul: float) -> bool:
    if args.max_wallclock_seconds <= 0:
        return (step / max(args.iterations, 1)) >= args.late_qat_threshold
    return lr_mul < args.late_qat_threshold


def with_eval_params(
    model: GPT,
    ema: EMABuffer | None,
    eval_fn: Callable[[], tuple[float, float]],
) -> tuple[float, float]:
    saved_params = flatten_params(model)
    saved_qat_flags = capture_qat_flags(model)
    try:
        if ema is not None:
            update_model_params(model, ema.as_mlx())
        set_qat_mode(model, False, False)
        return eval_fn()
    finally:
        restore_qat_flags(model, saved_qat_flags)
        update_model_params(model, saved_params)


def main() -> None:
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
        raise NotImplementedError("train_gpt_mlx_kl.py only supports tied embeddings")
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
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

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
        bigram_hash_size=args.bigram_hash_size,
        use_ortho_init=args.use_ortho_init,
    )
    opt = SplitOptimizers(model, args)
    compiled_loss, compiled_loss_and_grad = compile_model_fns(model)

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
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(
        f"features:bigram_hash:{args.bigram_hash_size} ortho_init:{int(args.use_ortho_init)} "
        f"ema_decay:{args.ema_decay} ema_start_frac:{args.ema_start_frac} "
        f"late_qat_threshold:{args.late_qat_threshold} learned_scales:{int(args.learned_scales)}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    do_final_eval = args.val_loss_every > 0 or args.max_wallclock_seconds > 0

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            mx.eval(warmup_loss, grads)
            mx.synchronize()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if do_final_eval:
            val_batch_tokens = args.val_batch_size // args.grad_accum_steps
            if val_batch_tokens < args.train_seq_len:
                raise ValueError(
                    "VAL_BATCH_SIZE must provide at least one sequence; "
                    f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                    f"TRAIN_SEQ_LEN={args.train_seq_len}"
                )
            warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
            warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
            x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
            warm_val_loss = compiled_loss(x_val, y_val)
            mx.eval(warm_val_loss)
            mx.synchronize()
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    ema: EMABuffer | None = None
    qat_active = False
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = with_eval_params(
                model,
                ema,
                lambda: eval_val(
                    args,
                    compiled_loss,
                    model,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                ),
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms"
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
        step += 1

        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        estimated_total_steps = estimate_total_steps(args, step, approx_train_time_ms)
        ema_start_step = max(1, int(math.ceil(args.ema_start_frac * estimated_total_steps)))
        if ema is None and step >= ema_start_step:
            ema = EMABuffer(flatten_params(model), decay=args.ema_decay)
            log(f"ema_started step:{step} est_total_steps:{estimated_total_steps}")
        elif ema is not None:
            ema.update(flatten_params(model))

        if not qat_active and should_start_qat(args, step, lr_mul):
            qat_active = True
            set_qat_mode(model, True, args.learned_scales)
            compiled_loss, compiled_loss_and_grad = compile_model_fns(model)
            log(f"qat_started step:{step} lr_mul:{lr_mul:.4f}")

        mx.synchronize()
        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        tok_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        tags = []
        if qat_active:
            tags.append("[QAT]")
        if ema is not None:
            tags.append("[EMA]")
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            suffix = f" {' '.join(tags)}" if tags else ""
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} step_ms:{step_ms:.0f} tok_s:{tok_s:.0f}{suffix}")
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    if ema is not None:
        update_model_params(model, ema.as_mlx())
        log("ema_applied_for_final_save")

    if do_final_eval:
        final_val_loss, final_val_bpb = with_eval_params(
            model,
            None,
            lambda: eval_val_sliding(
                args,
                compiled_loss,
                model,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            ),
        )
        log(f"final_prequant_sliding val_loss:{final_val_loss:.4f} val_bpb:{final_val_bpb:.4f}")
        log(f"final_prequant_sliding_exact val_loss:{final_val_loss:.8f} val_bpb:{final_val_bpb:.8f}")

    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int6(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zstd.ZstdCompressor(level=22).compress(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int6.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int6_payload_bytes"], 1)
    log(
        f"serialized_model_int6_zstd:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int6_payload_bytes']} raw_pickle:{len(quant_raw)} payload_ratio:{ratio:.2f}x)"
    )

    quant_flat = dequantize_state_dict_int6(pickle.loads(zstd.ZstdDecompressor().decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))
    if do_final_eval:
        q_val_loss, q_val_bpb = with_eval_params(
            model,
            None,
            lambda: eval_val_sliding(
                args,
                compiled_loss,
                model,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            ),
        )
        log(f"final_int6_zstd_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
        log(f"final_int6_zstd_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
