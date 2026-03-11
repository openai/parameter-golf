#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import time
import uuid
import zlib
from dataclasses import dataclass
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
# We support one shard format and one compute mode in this script.
COMPUTE_DTYPE = mx.bfloat16


# ==============================================================================
# MATH HELPERS
# ==============================================================================
def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    # train_gpt.py uses weightless RMSNorm in several places. This is the same normalization
    # (normalize each token vector by its RMS, then keep direction + learned projections elsewhere).
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
    header_bytes = 256 * np.dtype(np.int32).itemsize
    token_bytes = np.dtype(np.uint16).itemsize
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype=np.uint16, count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
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
    def __init__(self, pattern: str):
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
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
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - a learned scalar v_mix that reuses the first layer's values across layers
    # - causal masked SDPA
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.proj = CastedLinear(dim, dim)
        self.v_mix = mx.array(0.5, dtype=COMPUTE_DTYPE)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, v1: mx.array | None) -> tuple[mx.array, mx.array]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # First layer's value tensor is reused as a reference across layers.
        if v1 is None:
            v1 = v
        mix = self.v_mix.astype(v.dtype)
        v = (1.0 - mix) * v + mix * v1

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y), v1


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.resid_mix = mx.array([1.0, 0.0], dtype=COMPUTE_DTYPE)

    def __call__(self, x: mx.array, x0: mx.array, v1: mx.array | None) -> tuple[mx.array, mx.array]:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0] * x + mix[1] * x0
        attn_out, v1 = self.attn(self.attn_norm(x), v1)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, v1


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - optional tied embeddings for the LM head (default baseline setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 tie_embeddings: bool):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = mx.ones((self.num_decoder_layers,), dtype=COMPUTE_DTYPE)
        self.blocks = [Block(dim, num_heads, mlp_mult, rope_base) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()
        self.lm_head = None if tie_embeddings else CastedLinear(dim, vocab_size)

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        if self.lm_head is not None:
            self.lm_head.weight = mx.zeros_like(self.lm_head.weight)

        if self.tie_embeddings:
            self.tok_emb.weight = (
                mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
            ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        v1 = None
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x, v1 = self.blocks[i](x, x0, v1)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype) * skips.pop()
            x, v1 = self.blocks[self.num_encoder_layers + i](x, x0, v1)
        return self.final_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x)
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T if self.tie_embeddings else self.lm_head(x[s:e])
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Args:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp2048")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. train_batch_tokens is the optimizer-step budget; grad_accum_steps splits it
    # into smaller microbatches so Macs can fit long-context training in memory.
    iterations: int = int(os.environ.get("ITERATIONS", 100))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 25))
    val_tokens: int = int(os.environ.get("VAL_TOKENS", 8192))
    val_batch_tokens: int = int(os.environ.get("VAL_BATCH_TOKENS", 1024))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 65536))
    # Tuned locally on an Apple Silicon Mac for the baseline-like 11x512, seq=1024 setup:
    # 8 microsteps (microbatch 8 x 1024 = 8192 tokens) was faster than 16/32/64 and still fit.
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_max_seq_len: int = int(os.environ.get("TRAIN_MAX_SEQ_LEN", 1024))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 0))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 2048))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))

    # Optimizer. We preserve train_gpt.py's optimizer split idea (Muon for 2D block weights,
    # Adam for embeddings and scalar params), but tune per-group LRs for MLX stability by env.
    base_lr: float = float(os.environ.get("BASE_LR", 0.04))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    embed_lr: float = float(os.environ.get("EMBED_LR", 0.6))
    head_lr: float = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.01))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.01))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.01))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))

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

    def lr(self, step: int) -> float:
        # Keep the same simple schedule shape as train_gpt.py: optional warmup and warmdown that
        # scale the per-group base LRs. BASE_LR here only defines the multiplier schedule.
        lr = self.base_lr
        if self.warmup_steps and step < self.warmup_steps:
            lr *= (step + 1) / self.warmup_steps
        if self.warmdown_iters:
            s = max(self.iterations - self.warmdown_iters, 0)
            if s <= step < self.iterations:
                lr *= max((self.iterations - step) / self.warmdown_iters, 0.0)
        return lr

    def lr_mul(self, step: int) -> float:
        return self.lr(step) / self.base_lr


# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Args):
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
    # - embeddings: Adam with its own LR
    # - optional untied LM head: Adam
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.head_key = "lm_head.weight" if not args.tie_embeddings else None
        self.matrix_keys = [k for k, p in params.items() if k.startswith("blocks.") and p.ndim == 2]
        self.scalar_keys = [k for k, p in params.items() if k.startswith("blocks.") and p.ndim < 2] + ["skip_weights"]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr if args.tie_embeddings else args.embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_head = None
        if self.head_key is not None:
            self.adam_head = optim.Adam(
                learning_rate=args.head_lr,
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

        token_lr = self.args.tied_embed_lr if self.args.tie_embeddings else self.args.embed_lr
        self.adam_embed.learning_rate = token_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )
        if self.head_key is not None:
            assert self.adam_head is not None
            self.adam_head.learning_rate = self.args.head_lr * lr_mul
            updated.update(
                self.adam_head.apply_gradients(
                    {self.head_key: grads[self.head_key]},
                    {self.head_key: params[self.head_key]},
                )
            )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB, TRAIN_GPT-STYLE)
# ==============================================================================
# The baseline script reports both raw serialized size and an int8+zlib variant.
# We keep the same spirit here: symmetric per-tensor int8 quantization for every
# floating tensor, then zlib-compress the serialized payload.
MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def quantize_state_dict_int8_per_tensor(flat_params: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    orig_dtypes: dict[str, str] = {}
    nonfloat: dict[str, np.ndarray] = {}
    stats = {
        "param_count": 0,
        "num_tensors": 0,
        "num_float_tensors": 0,
        "num_nonfloat_tensors": 0,
        "baseline_tensor_bytes": 0,
        "int8_payload_bytes": 0,
    }
    for name, arr in flat_params.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if mx.issubdtype(arr.dtype, mx.floating):
            stats["num_float_tensors"] += 1
            # NumPy does not reliably ingest MLX bfloat16 buffers directly, so convert in MLX first.
            f32 = np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)
            max_abs = float(np.max(np.abs(f32))) if f32.size else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            q = np.clip(np.round(f32 / scale), -127, 127).astype(np.int8, copy=False)
            s = np.array(scale, dtype=np.float32)
            quantized[name] = np.ascontiguousarray(q)
            scales[name] = s
            orig_dtypes[name] = str(arr.dtype).split(".")[-1]
            stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
        else:
            stats["num_nonfloat_tensors"] += 1
            np_arr = np.array(arr)
            nonfloat[name] = np.ascontiguousarray(np_arr)
            stats["int8_payload_bytes"] += int(np_arr.nbytes)
    return {
        "__quant_format__": "int8_sym_per_tensor_v1",
        "quantized": quantized,
        "scales": scales,
        "orig_dtypes": orig_dtypes,
        "nonfloat": nonfloat,
    }, stats


def dequantize_state_dict_int8_per_tensor(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        scale = float(np.asarray(quant_obj["scales"][name], dtype=np.float32))
        dtype_name = quant_obj["orig_dtypes"][name]
        out[name] = mx.array(q_np.astype(np.float32) * scale, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["nonfloat"].items():
        out[name] = mx.array(arr)
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


def eval_val(
    args: Args,
    compiled_loss,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_steps = args.val_tokens // args.val_batch_tokens
    # Rebuild the validation stream for each eval so repeated checkpoints measure the same token window.
    val_loader = TokenLoader(args.val_files)
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for _ in range(val_steps):
        x, y = val_loader.next_batch(args.val_batch_tokens, args.train_max_seq_len)
        total_loss = total_loss + compiled_loss(x, y)
        x_np = np.array(x)
        y_np = np.array(y)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += float(y_np.size)
        total_bytes += float(bytes_np.astype(np.float64).sum())
    total_loss = total_loss / float(val_steps)
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def main() -> None:
    args = Args()
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    manifest_path = Path(args.data_path).resolve().parents[1] / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ds = next((x for x in manifest.get("datasets", []) if x.get("name") == Path(args.data_path).resolve().name), None)
        tok = next((x for x in manifest.get("tokenizers", []) if x.get("name") == ds.get("tokenizer_name")), None) if ds else None
        expected = Path((tok or {}).get("model_path") or (tok or {}).get("path") or "").name
        if expected and Path(args.tokenizer_path).name != expected:
            raise ValueError(f"{Path(args.data_path).name} expects tokenizer {expected}, got {Path(args.tokenizer_path).name}")

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files)

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        tie_embeddings=args.tie_embeddings,
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

    # Print config once so logs are self-describing. This mirrors the train_gpt style of writing enough
    # metadata into stdout that you can reconstruct a run later from a single log file.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"run_id:{args.run_id}")
    print(f"mlx_version:{mx.__version__}")
    print(f"train_loader:shards pattern={args.train_files}")
    print(f"val_loader:shards pattern={args.val_files}")
    print(f"tokenizer_path:{args.tokenizer_path}")
    print(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} seq_len:{args.train_max_seq_len} "
        f"tie_embeddings:{args.tie_embeddings}"
    )
    print(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_max_seq_len} "
        f"val_batch_tokens:{args.val_batch_tokens} val_tokens:{args.val_tokens}"
    )
    print(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr if args.tie_embeddings else args.embed_lr} "
        f"head_lr:{0.0 if args.tie_embeddings else args.head_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    print(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    print(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    print(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"lm_head:{'tied' if model.lm_head is None else model.lm_head.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    train_time_ms = 0.0
    t0 = time.perf_counter()
    for step in range(args.iterations + 1):
        last_step = step == args.iterations
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            print(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} train_time:{train_time_ms:.0f}ms "
                f"step_avg:{train_time_ms / max(step, 1):.2f}ms val_bpb:{val_bpb:.4f}"
            )
            t0 = time.perf_counter()
        if last_step:
            break

        lr = args.lr(step)
        lr_mul = args.lr_mul(step)
        step_t0 = time.perf_counter()

        accum: dict[str, mx.array] | None = None
        train_loss = None
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(args.microbatch_tokens, args.train_max_seq_len)
            loss, grads = compiled_loss_and_grad(x, y)
            flat = dict(tree_flatten(grads))
            if accum is None:
                accum = {k: g * grad_scale for k, g in flat.items()}
            else:
                for k, g in flat.items():
                    accum[k] = accum[k] + g * grad_scale
            train_loss = loss

        grads = tree_unflatten(list(accum.items()))
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        print(
            f"step:{step + 1}/{args.iterations} train_loss:{train_loss_value:.4f} lr:{lr:.2e} "
            f"step_ms:{step_ms:.0f} train_time:{approx_train_time_ms:.0f}ms "
            f"step_avg:{approx_train_time_ms / (step + 1):.2f}ms tok_s:{tok_s:.0f}"
        )

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_params = {k: v for k, v in tree_flatten(model.parameters())}
    mx.savez(str(out_path), **flat_params)
    print(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    quant_obj, quant_stats = quantize_state_dict_int8_per_tensor(flat_params)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    print(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    quant_flat = dequantize_state_dict_int8_per_tensor(pickle.loads(zlib.decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    print(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")


if __name__ == "__main__":
    main()
