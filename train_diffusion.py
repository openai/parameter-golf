#!/usr/bin/env python3
"""
Simple MLX masked diffusion language model for Parameter Golf week 1.

This script intentionally stays close to `train_gpt_mlx.py` for:
- environment-variable configuration
- shard loading
- MLX compilation patterns
- Mac-friendly microbatching

It implements the smallest useful discrete diffusion baseline:
- bidirectional Transformer denoiser
- absorbing-mask corruption
- masked-token cross-entropy objective
- iterative unmasking samples for sanity checks
- synthetic repeated-pattern mode for quick overfit debugging
"""
from __future__ import annotations

import glob
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

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

    iterations: int = int(os.environ.get("ITERATIONS", 4_000))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 200))
    val_at_start: bool = bool(int(os.environ.get("VAL_AT_START", "1")))
    val_at_end: bool = bool(int(os.environ.get("VAL_AT_END", "1")))
    sample_every: int = int(os.environ.get("SAMPLE_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 65_536))
    val_batch_tokens: int = int(os.environ.get("VAL_BATCH_TOKENS", 65_536))
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 256))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 5))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 6))
    model_dim: int = int(os.environ.get("MODEL_DIM", 256))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.02))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    learning_rate: float = float(os.environ.get("LEARNING_RATE", 3e-4))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.0))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    num_diffusion_steps: int = int(os.environ.get("NUM_DIFFUSION_STEPS", 32))
    mask_schedule: str = os.environ.get("MASK_SCHEDULE", "cosine")
    min_mask_rate: float = float(os.environ.get("MIN_MASK_RATE", 0.0))
    max_mask_rate: float = float(os.environ.get("MAX_MASK_RATE", 1.0))
    mask_token_id: int = int(os.environ.get("MASK_TOKEN_ID", -1))
    sample_temperature: float = float(os.environ.get("SAMPLE_TEMPERATURE", 1.0))
    sample_prompt: str = os.environ.get("SAMPLE_PROMPT", "")
    sample_num_steps: int = int(os.environ.get("SAMPLE_NUM_STEPS", 0))

    train_shards: int = int(os.environ.get("TRAIN_SHARDS", 0))
    synthetic_data: bool = bool(int(os.environ.get("SYNTHETIC_DATA", "0")))
    synthetic_vocab_size: int = int(os.environ.get("SYNTHETIC_VOCAB_SIZE", 32))
    synthetic_pattern_len: int = int(os.environ.get("SYNTHETIC_PATTERN_LEN", 8))
    synthetic_train_tokens: int = int(os.environ.get("SYNTHETIC_TRAIN_TOKENS", 131_072))
    synthetic_val_tokens: int = int(os.environ.get("SYNTHETIC_VAL_TOKENS", 16_384))

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

    @property
    def sample_steps(self) -> int:
        return self.sample_num_steps if self.sample_num_steps > 0 else self.num_diffusion_steps


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


def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


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
        train_shards: int = 0,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        files = [Path(p) for p in sorted(glob.glob(pattern))]
        if train_shards > 0:
            files = files[:train_shards]
        self.files = files
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
                    f"WARNING: starting_epoch:{self.epoch} dataset:{self.dataset_name} train_shards:{len(self.files)}"
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
        train_shards: int = 0,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, train_shards=train_shards, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> np.ndarray:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable)
        return chunk.reshape(-1, seq_len)


class SyntheticLoader:
    def __init__(self, tokens: np.ndarray):
        self.tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        self.pos = 0

    def next_batch(self, batch_tokens: int, seq_len: int) -> np.ndarray:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        total = usable
        if self.pos + total > self.tokens.size:
            wrap = self.pos + total - self.tokens.size
            chunk = np.concatenate([self.tokens[self.pos :], self.tokens[:wrap]], axis=0)
            self.pos = wrap
        else:
            chunk = self.tokens[self.pos : self.pos + total]
            self.pos += total
        return chunk.reshape(-1, seq_len)


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("MODEL_DIM must be divisible by NUM_HEADS")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.q_proj = CastedLinear(dim, dim)
        self.k_proj = CastedLinear(dim, dim)
        self.v_proj = CastedLinear(dim, dim)
        self.out_proj = CastedLinear(dim, dim)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.q_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.out_proj(y)


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
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = BidirectionalSelfAttention(dim, num_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        mlp_mult: int,
        num_diffusion_steps: int,
        rope_base: float,
        tied_embed_init_std: float,
        logit_softcap: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError("LOGIT_SOFTCAP must be positive")
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.time_emb = nn.Embedding(num_diffusion_steps + 1, dim)
        self.blocks = [Block(dim, num_heads, mlp_mult, rope_base) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)
        self.time_emb.weight = (
            mx.random.normal(self.time_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def hidden(self, input_ids: mx.array, timesteps: mx.array) -> mx.array:
        x = self.tok_emb(input_ids).astype(COMPUTE_DTYPE)
        t = self.time_emb(timesteps).astype(COMPUTE_DTYPE)[:, None, :]
        x = rms_norm(x + t)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def logits(self, input_ids: mx.array, timesteps: mx.array) -> mx.array:
        h = self.hidden(input_ids, timesteps)
        logits = h @ self.tok_emb.weight.astype(h.dtype).T
        return self.softcap(logits)

    def loss(
        self,
        corrupted_ids: mx.array,
        target_ids: mx.array,
        timesteps: mx.array,
        loss_mask: mx.array,
    ) -> mx.array:
        logits = self.logits(corrupted_ids, timesteps).astype(mx.float32)
        losses = nn.losses.cross_entropy(logits, target_ids, reduction="none").astype(mx.float32)
        weights = loss_mask.astype(mx.float32)
        return mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1.0, dtype=mx.float32))


def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        g = np.asarray(grad.astype(mx.float32), dtype=np.float32)
        total_sq += float(np.sum(np.square(g), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def build_synthetic_tokens(args: Hyperparameters) -> tuple[np.ndarray, np.ndarray]:
    usable_vocab = min(args.synthetic_vocab_size, args.vocab_size - 1)
    if usable_vocab < 4:
        raise ValueError("SYNTHETIC_VOCAB_SIZE must leave room for a mask token")
    pattern = (np.arange(args.synthetic_pattern_len, dtype=np.int32) % usable_vocab) + 1
    train_repeats = (args.synthetic_train_tokens + pattern.size - 1) // pattern.size
    val_repeats = (args.synthetic_val_tokens + pattern.size - 1) // pattern.size
    train_tokens = np.tile(pattern, train_repeats)[: args.synthetic_train_tokens]
    val_tokens = np.tile(pattern, val_repeats)[: args.synthetic_val_tokens]
    return train_tokens, val_tokens


def load_validation_tokens(pattern: str, seq_len: int, max_tokens: int = 0) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    usable = (tokens.size // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(
            f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}; "
            f"got {tokens.size} tokens after VAL_MAX_TOKENS={max_tokens}"
        )
    return tokens[:usable]


def choose_mask_token_id(sp: spm.SentencePieceProcessor | None, args: Hyperparameters) -> int:
    if args.mask_token_id >= 0:
        if args.mask_token_id >= args.vocab_size:
            raise ValueError(f"MASK_TOKEN_ID={args.mask_token_id} must be < VOCAB_SIZE={args.vocab_size}")
        return args.mask_token_id
    if sp is None:
        return args.vocab_size - 1
    for token_id in (sp.pad_id(),):
        if 0 <= token_id < args.vocab_size and token_id != sp.unk_id():
            return int(token_id)
    for token_id in range(args.vocab_size - 1, -1, -1):
        if sp.is_unused(token_id) or sp.is_control(token_id):
            return token_id
    for token_id in (sp.eos_id(), sp.bos_id()):
        if 0 <= token_id < args.vocab_size and token_id != sp.unk_id():
            return int(token_id)
    return args.vocab_size - 1


def mask_rate_for_t(timesteps: np.ndarray, args: Hyperparameters) -> np.ndarray:
    frac = timesteps.astype(np.float32) / float(args.num_diffusion_steps)
    if args.mask_schedule == "linear":
        rate = frac
    elif args.mask_schedule == "cosine":
        rate = np.sin(0.5 * np.pi * frac) ** 2
    else:
        raise ValueError(f"Unknown MASK_SCHEDULE={args.mask_schedule}")
    return np.clip(rate, args.min_mask_rate, args.max_mask_rate)


def corrupt_batch_np(
    clean_ids: np.ndarray,
    args: Hyperparameters,
    rng: np.random.Generator,
    mask_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    batch, seq_len = clean_ids.shape
    timesteps = rng.integers(1, args.num_diffusion_steps + 1, size=(batch,), dtype=np.int32)
    mask_rates = mask_rate_for_t(timesteps, args)
    mask = rng.random((batch, seq_len), dtype=np.float32) < mask_rates[:, None]
    no_mask_rows = np.where(mask.sum(axis=1) == 0)[0]
    if no_mask_rows.size:
        cols = rng.integers(0, seq_len, size=(no_mask_rows.size,), dtype=np.int32)
        mask[no_mask_rows, cols] = True
    corrupted = clean_ids.copy()
    corrupted[mask] = mask_token_id
    return corrupted, timesteps, mask.astype(np.float32), float(mask.mean())


def make_eval_batches(tokens: np.ndarray, batch_tokens: int, seq_len: int) -> list[np.ndarray]:
    usable_batch = (batch_tokens // seq_len) * seq_len
    if usable_batch <= 0:
        raise ValueError(f"VAL_BATCH_TOKENS too small for TRAIN_SEQ_LEN={seq_len}")
    seqs_per_batch = usable_batch // seq_len
    total_seqs = tokens.size // seq_len
    return [tokens[i * seq_len : j * seq_len].reshape(-1, seq_len) for i in range(0, total_seqs, seqs_per_batch) for j in [min(i + seqs_per_batch, total_seqs)]]


def eval_val(
    model: DiffusionTransformer,
    compiled_loss,
    val_batches: list[np.ndarray],
    args: Hyperparameters,
    mask_token_id: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(12345)
    total_loss = 0.0
    total_masked = 0.0
    total_tokens = 0.0
    for batch in val_batches:
        corrupted, timesteps, loss_mask, _ = corrupt_batch_np(batch, args, rng, mask_token_id)
        x = mx.array(corrupted, dtype=mx.int32)
        y = mx.array(batch, dtype=mx.int32)
        t = mx.array(timesteps, dtype=mx.int32)
        m = mx.array(loss_mask, dtype=mx.float32)
        loss = compiled_loss(x, y, t, m).astype(mx.float32)
        mx.eval(loss)
        masked = float(loss_mask.sum())
        total_loss += float(loss.item()) * masked
        total_masked += masked
        total_tokens += float(loss_mask.size)
    return total_loss / max(total_masked, 1.0), total_masked / max(total_tokens, 1.0)


def decode_ids(ids: list[int], sp: spm.SentencePieceProcessor | None, synthetic: bool) -> str:
    if synthetic or sp is None:
        return " ".join(str(int(x)) for x in ids)
    return sp.decode(ids)


def sample_text(
    model: DiffusionTransformer,
    compiled_logits,
    args: Hyperparameters,
    sp: spm.SentencePieceProcessor | None,
    mask_token_id: int,
) -> str:
    rng = np.random.default_rng(args.seed + 999)
    tokens = np.full((1, args.train_seq_len), mask_token_id, dtype=np.int32)
    fixed = np.zeros((1, args.train_seq_len), dtype=bool)
    if args.sample_prompt and sp is not None and not args.synthetic_data:
        prompt_ids = np.array(sp.encode(args.sample_prompt), dtype=np.int32)[: args.train_seq_len]
        tokens[0, : prompt_ids.size] = prompt_ids
        fixed[0, : prompt_ids.size] = True
    current = mx.array(tokens, dtype=mx.int32)
    for step in range(args.sample_steps, 0, -1):
        t = mx.array([min(step, args.num_diffusion_steps)], dtype=mx.int32)
        logits = compiled_logits(current, t).astype(mx.float32) / args.sample_temperature
        sampled = mx.random.categorical(logits)
        mx.eval(sampled)
        sampled_np = np.asarray(sampled, dtype=np.int32)
        current_np = np.array(current, dtype=np.int32, copy=True)
        current_mask = (current_np == mask_token_id) & ~fixed
        if step == 1:
            reveal = current_mask
        else:
            p_now = mask_rate_for_t(np.array([step], dtype=np.int32), args)[0]
            p_next = mask_rate_for_t(np.array([step - 1], dtype=np.int32), args)[0]
            keep_prob = 0.0 if p_now <= 0 else float(np.clip(p_next / p_now, 0.0, 1.0))
            keep_masked = rng.random(current_mask.shape) < keep_prob
            reveal = current_mask & ~keep_masked
        current_np[reveal] = sampled_np[reveal]
        current = mx.array(current_np, dtype=mx.int32)
    ids = np.asarray(current, dtype=np.int32)[0].tolist()
    if args.synthetic_data:
        return decode_ids(ids, None, synthetic=True)
    if sp is not None:
        ids = [tok for tok in ids if tok != mask_token_id]
    return decode_ids(ids, sp, synthetic=False)


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader,
    rng: np.random.Generator,
    mask_token_id: int,
    compiled_loss_and_grad,
) -> tuple[mx.array, dict, float]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    masked_fraction_sum = 0.0
    for chunk_tokens in chunk_sizes:
        clean_np = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        corrupted_np, timesteps_np, loss_mask_np, masked_fraction = corrupt_batch_np(clean_np, args, rng, mask_token_id)
        x = mx.array(corrupted_np, dtype=mx.int32)
        y = mx.array(clean_np, dtype=mx.int32)
        t = mx.array(timesteps_np, dtype=mx.int32)
        m = mx.array(loss_mask_np, dtype=mx.float32)
        loss, grads = compiled_loss_and_grad(x, y, t, m)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        masked_fraction_sum += masked_fraction * scale
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items())), masked_fraction_sum


def main() -> None:
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}_diffusion.txt"
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

    mx.random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    sp: spm.SentencePieceProcessor | None = None
    dataset_name = "synthetic"
    if args.synthetic_data:
        train_tokens, val_tokens = build_synthetic_tokens(args)
        train_loader = SyntheticLoader(train_tokens)
        val_batches = make_eval_batches(val_tokens, args.val_batch_tokens, args.train_seq_len)
        args.vocab_size = max(args.vocab_size, args.synthetic_vocab_size)
    else:
        if not args.tokenizer_path.endswith(".model"):
            raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        dataset_name = Path(args.data_path).resolve().name
        train_loader = TokenLoader(args.train_files, train_shards=args.train_shards, log_fn=log, dataset_name=dataset_name)
        val_tokens = load_validation_tokens(args.val_files, args.train_seq_len, max_tokens=args.val_max_tokens)
        val_batches = make_eval_batches(val_tokens, args.val_batch_tokens, args.train_seq_len)

    mask_token_id = choose_mask_token_id(sp, args)
    model = DiffusionTransformer(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        num_diffusion_steps=args.num_diffusion_steps,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
    )
    optimizer = optim.Adam(
        learning_rate=args.learning_rate,
        betas=[args.beta1, args.beta2],
        eps=args.adam_eps,
        bias_correction=True,
    )

    compiled_loss = mx.compile(
        lambda x, y, t, m: model.loss(x, y, t, m),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_logits = mx.compile(
        lambda x, t: model.logits(x, t),
        inputs=model.state,
        outputs=model.state,
    )
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y, t, m: model.loss(x, y, t, m)),
        inputs=model.state,
        outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    train_shard_msg = args.train_shards if args.train_shards > 0 else "all"
    log(f"run_id:{args.run_id}")
    log(f"mode:{'synthetic' if args.synthetic_data else 'fineweb'} dataset:{dataset_name}")
    log(f"tokenizer_path:{args.tokenizer_path if sp is not None else 'synthetic'}")
    log(f"mask_token_id:{mask_token_id} mask_schedule:{args.mask_schedule} diffusion_steps:{args.num_diffusion_steps}")
    log(f"validation_tokens:{sum(batch.size for batch in val_batches)} val_max_tokens:{args.val_max_tokens}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} dim:{args.model_dim} "
        f"heads:{args.num_heads} seq_len:{args.train_seq_len}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} val_batch_tokens:{args.val_batch_tokens} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.1f} train_shards:{train_shard_msg}"
    )
    log(
        f"optimizer:adam lr:{args.learning_rate} wd:{args.weight_decay} "
        f"betas:({args.beta1},{args.beta2}) grad_clip_norm:{args.grad_clip_norm}"
    )

    if args.warmup_steps > 0:
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            for _ in range(args.grad_accum_steps):
                loss, grads, _ = loss_and_grad_chunked(args, train_loader, np_rng, mask_token_id, compiled_loss_and_grad)
                warmup_loss = warmup_loss + loss.astype(mx.float32) / args.grad_accum_steps
                accum = accumulate_flat_grads(accum, grads, 1.0 / args.grad_accum_steps)
            mx.eval(warmup_loss, accum)
            if args.warmup_steps <= 10 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        if args.synthetic_data:
            train_tokens, _ = build_synthetic_tokens(args)
            train_loader = SyntheticLoader(train_tokens)
        else:
            train_loader = TokenLoader(args.train_files, train_shards=args.train_shards, log_fn=log, dataset_name=dataset_name)

    t0 = time.perf_counter()
    step = 0
    stop_after_step: int | None = None
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_run_val = False
        if last_step and args.val_at_end:
            should_run_val = True
        elif step == 0 and args.val_at_start:
            should_run_val = True
        elif step > 0 and args.val_loss_every > 0 and step % args.val_loss_every == 0:
            should_run_val = True
        if should_run_val:
            val_loss, val_masked = eval_val(model, compiled_loss, val_batches, args, mask_token_id)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_masked_frac:{val_masked:.4f}")
        if last_step:
            break

        step_t0 = time.perf_counter()
        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        train_masked_fraction = 0.0
        for _ in range(args.grad_accum_steps):
            loss, grads, masked_fraction = loss_and_grad_chunked(
                args,
                train_loader,
                np_rng,
                mask_token_id,
                compiled_loss_and_grad,
            )
            train_loss = train_loss + loss.astype(mx.float32) / args.grad_accum_steps
            train_masked_fraction += masked_fraction / args.grad_accum_steps
            accum = accumulate_flat_grads(accum, grads, 1.0 / args.grad_accum_steps)
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)

        grads_tree = tree_unflatten(list(accum.items()))
        grads_tree = clip_grad_tree(grads_tree, args.grad_clip_norm)
        params = dict(tree_flatten(model.trainable_parameters()))
        grads = dict(tree_flatten(grads_tree))
        if args.weight_decay > 0:
            grads = {k: g + args.weight_decay * params[k] for k, g in grads.items()}
        updated = optimizer.apply_gradients(grads, params)
        model.update(tree_unflatten(list(updated.items())))
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        step += 1
        tok_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(
                f"step:{step}/{args.iterations} train_loss:{float(train_loss.item()):.4f} "
                f"masked_frac:{train_masked_fraction:.4f} tok_s:{tok_s:.0f}"
            )
        if args.sample_every > 0 and (step <= 3 or step % args.sample_every == 0):
            sample = sample_text(model, compiled_logits, args, sp, mask_token_id)
            log(f"sample_step:{step} text:{sample[:400]}")
        if max_wallclock_ms is not None and stop_after_step is None:
            elapsed_ms = 1000.0 * (time.perf_counter() - t0)
            if elapsed_ms >= max_wallclock_ms:
                stop_after_step = step

    out_path = out_dir / f"{args.run_id}_diffusion_mlx.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")


if __name__ == "__main__":
    main()
