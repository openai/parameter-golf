#!/usr/bin/env python3
from __future__ import annotations

import glob
import io
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
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
SHARD_HEADER_INTS = 256
COMPUTE_DTYPE = mx.bfloat16


# ==============================================================================
# TINY HELPERS
# ==============================================================================
# These exist only to keep the config and math code compact.
def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    # train_gpt.py uses weightless RMSNorm in several places. This is the same normalization
    # (normalize each token vector by its RMS, then keep direction + learned projections elsewhere).
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def fro_norm(x: mx.array) -> mx.array:
    return mx.sqrt(mx.sum(x * x))


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (fro_norm(x) + eps)
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
    header = np.fromfile(path, dtype=np.int32, count=SHARD_HEADER_INTS)
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype=np.uint16, count=num_tokens, offset=SHARD_HEADER_INTS * 4)
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
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
    # - tied embeddings for the LM head (default baseline setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, mlp_mult: int, max_seq_len: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = mx.ones((self.num_decoder_layers,), dtype=COMPUTE_DTYPE)
        self.blocks = [Block(dim, num_heads, mlp_mult, rope_base) for _ in range(num_layers)]
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
            logits = self.softcap(x @ self.tok_emb.weight.astype(x.dtype).T)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.softcap(x[s:e] @ self.tok_emb.weight.astype(x.dtype).T)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class Args:
    # Data / tokenizer.
    data_path: str = env_str("DATA_PATH", "./data/matched_10B_docs2m_seed1337/datasets/fineweb10B_sp2048")
    tokenizer_path: str = os.environ["TOKENIZER_PATH"]
    run_id: str = env_str("RUN_ID", str(uuid.uuid4()))
    seed: int = env_int("SEED", 1337)

    # Training loop. train_batch_tokens is the optimizer-step budget; grad_accum_steps splits it
    # into smaller microbatches so Macs can fit long-context training in memory.
    iterations: int = env_int("ITERATIONS", 100)
    val_loss_every: int = env_int("VAL_LOSS_EVERY", 25)
    val_tokens: int = env_int("VAL_TOKENS", 8192)
    val_batch_tokens: int = env_int("VAL_BATCH_TOKENS", 1024)
    train_batch_tokens: int = env_int("TRAIN_BATCH_TOKENS", 65536)
    # Tuned locally on an Apple Silicon Mac for the baseline-like 11x512, seq=1024 setup:
    # 8 microsteps (microbatch 8 x 1024 = 8192 tokens) was faster than 16/32/64 and still fit.
    grad_accum_steps: int = env_int("GRAD_ACCUM_STEPS", 8)
    train_max_seq_len: int = env_int("TRAIN_MAX_SEQ_LEN", 1024)
    warmup_steps: int = env_int("WARMUP_STEPS", 0)
    warmdown_iters: int = env_int("WARMDOWN_ITERS", 0)

    # Model (defaults match the current baseline setup).
    vocab_size: int = env_int("VOCAB_SIZE", 2048)
    num_layers: int = env_int("NUM_LAYERS", 11)
    model_dim: int = env_int("MODEL_DIM", 512)
    num_heads: int = env_int("NUM_HEADS", 8)
    mlp_mult: int = env_int("MLP_MULT", 4)
    tied_embed_init_std: float = env_float("TIED_EMBED_INIT_STD", 0.005)
    logit_chunk_tokens: int = env_int("LOGIT_CHUNK_TOKENS", 0)
    logit_softcap: float = env_float("LOGIT_SOFTCAP", 30.0)
    rope_base: float = env_float("ROPE_BASE", 10000.0)

    # Optimizer. We preserve train_gpt.py's optimizer split idea (Muon for 2D block weights,
    # Adam for embeddings and scalar params), but tune per-group LRs for MLX stability by env.
    base_lr: float = env_float("BASE_LR", 0.04)
    beta1: float = env_float("BETA1", 0.9)
    beta2: float = env_float("BETA2", 0.95)
    adam_eps: float = env_float("ADAM_EPS", 1e-8)
    tied_embed_lr: float = env_float("TIED_EMBED_LR", 0.01)
    matrix_lr: float = env_float("MATRIX_LR", 0.01)
    scalar_lr: float = env_float("SCALAR_LR", 0.01)
    muon_momentum: float = env_float("MUON_MOMENTUM", 0.95)
    muon_nesterov: bool = env_bool("MUON_NESTEROV", True)
    muon_backend_steps: int = env_int("MUON_BACKEND_STEPS", 5)
    muon_momentum_warmup_start: float = env_float("MUON_MOMENTUM_WARMUP_START", 0.85)
    muon_momentum_warmup_steps: int = env_int("MUON_MOMENTUM_WARMUP_STEPS", 500)

    out_dir: str = env_str("OUT_DIR", "logs")

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
            g_eff = g + momentum * buf if self.args.muon_nesterov else buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with its own LR (tied embedding matrix)
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [k for k, p in params.items() if k.startswith("blocks.") and p.ndim == 2]
        self.scalar_keys = [k for k, p in params.items() if k.startswith("blocks.") and p.ndim < 2] + ["skip_weights"]

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
        updated.update(self.adam_embed.apply_gradients({self.embed_key: grads[self.embed_key]}, {self.embed_key: params[self.embed_key]}))

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

    def state_for_eval(self):
        return [self.muon.buffers, self.adam_embed.state, self.adam_scalar.state]


# ==============================================================================
# TOKENIZER / BPB LOOKUP TABLES
# ==============================================================================
# Build the SentencePiece byte-accounting lookup once.
#
# BPB needs per-token byte counts, but SentencePiece pieces don't map 1:1 to bytes because pieces may
# encode a leading whitespace marker (▁). The baseline metric counts the actual reconstructed UTF-8 bytes,
# so we store three lookup tables:
# - base byte count for the token piece text (without the ▁ marker)
# - whether the token piece starts with ▁ (meaning it may contribute a space)
# - whether the previous token is a boundary token (if yes, we should not add that space)
args = Args()
sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
base_bytes_lut = np.zeros((args.vocab_size,), dtype=np.int16)
has_leading_space_lut = np.zeros((args.vocab_size,), dtype=np.bool_)
is_boundary_token_lut = np.ones((args.vocab_size,), dtype=np.bool_)
for token_id in range(sp.vocab_size()):
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


def bytes_per_token_np(input_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    # Count target token bytes with the same rule as train_gpt.py: a leading-space piece contributes
    # one extra byte iff the previous token is not a boundary/control token.
    prev = input_ids.reshape(-1)
    tgt = target_ids.reshape(-1)
    out = base_bytes_lut[tgt].astype(np.int16, copy=True)
    out += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.int16, copy=False)
    return out.reshape(target_ids.shape)


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


def _dtype_name(x: mx.array) -> str:
    return str(x.dtype).split(".")[-1]


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
            orig_dtypes[name] = _dtype_name(arr)
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


def serialize_quantized_int8_zlib(quant_obj: dict[str, object]) -> tuple[bytes, int]:
    raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(raw, level=9), len(raw)


def deserialize_quantized_int8_zlib(blob: bytes) -> dict[str, object]:
    return pickle.loads(zlib.decompress(blob))


# ==============================================================================
# TRAINING SETUP
# ==============================================================================

mx.random.seed(args.seed)
train_loader = TokenLoader(args.train_files)
val_loader = TokenLoader(args.val_files)

model = GPT(
    vocab_size=args.vocab_size,
    num_layers=args.num_layers,
    dim=args.model_dim,
    num_heads=args.num_heads,
    mlp_mult=args.mlp_mult,
    max_seq_len=args.train_max_seq_len,
    logit_chunk_tokens=args.logit_chunk_tokens,
    logit_softcap=args.logit_softcap,
    rope_base=args.rope_base,
    tied_embed_init_std=args.tied_embed_init_std,
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
    f"dim:{args.model_dim} heads:{args.num_heads} seq_len:{args.train_max_seq_len} tie_embeddings:True"
)
print(
    f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
    f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_max_seq_len} "
    f"val_batch_tokens:{args.val_batch_tokens} val_tokens:{args.val_tokens}"
)
print(
    f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
    f"embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
    f"muon_momentum:{args.muon_momentum} muon_nesterov:{args.muon_nesterov} muon_steps:{args.muon_backend_steps}"
)
print(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
print(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
print(
    f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
    f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
    f"skip_weights:{model.skip_weights.dtype}"
)


# ==============================================================================
# VALIDATION
# ==============================================================================
def eval_val() -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_steps = args.val_tokens // args.val_batch_tokens
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for _ in range(val_steps):
        x, y = val_loader.next_batch(args.val_batch_tokens, args.train_max_seq_len)
        total_loss = total_loss + compiled_loss(x, y)
        x_np = np.array(x)
        y_np = np.array(y)
        bytes_np = bytes_per_token_np(x_np, y_np)
        total_tokens += float(y_np.size)
        total_bytes += float(bytes_np.astype(np.float64).sum())
    total_loss = total_loss / float(val_steps)
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

train_time_ms = 0.0
t0 = time.perf_counter()
for step in range(args.iterations + 1):
    last_step = step == args.iterations
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        val_loss, val_bpb = eval_val()
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
    for micro_step in range(args.grad_accum_steps):
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
quant_blob, quant_serialized_bytes = serialize_quantized_int8_zlib(quant_obj)
quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
with quant_path.open("wb") as f:
    f.write(quant_blob)
quant_file_bytes = quant_path.stat().st_size
ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
print(
    f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
    f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
)

quant_flat = dequantize_state_dict_int8_per_tensor(deserialize_quantized_int8_zlib(quant_blob))
model.update(tree_unflatten(list(quant_flat.items())))
q_t0 = time.perf_counter()
q_val_loss, q_val_bpb = eval_val()
q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
print(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
