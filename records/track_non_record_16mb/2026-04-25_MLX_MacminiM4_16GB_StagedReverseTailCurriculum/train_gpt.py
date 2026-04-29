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
import zlib
from collections.abc import Callable
from pathlib import Path
import numpy as np
_OPTIONAL_IMPORT_ERROR = None
try:
    import sentencepiece as spm; import mlx.core as mx; import mlx.nn as nn; import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
except ModuleNotFoundError as exc:
    _OPTIONAL_IMPORT_ERROR = exc
    class _Missing:
        class Module: pass
        def __getattr__(self, _: str): return self
        def __call__(self, *args, **kwargs): raise ModuleNotFoundError(f"Optional dependency missing: {_OPTIONAL_IMPORT_ERROR.name}") from _OPTIONAL_IMPORT_ERROR
    spm = mx = nn = optim = _Missing(); tree_flatten = tree_unflatten = _Missing()
COMPUTE_DTYPE = mx.bfloat16
REPO_ROOT = Path(__file__).resolve().parents[3]
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"))
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"))
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 8_192))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 6_144))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "0")))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 64))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    final_eval_reserve_seconds: float = float(os.environ.get("FINAL_EVAL_RESERVE_SECONDS", 72.0))
    final_eval_reserve_scale: float = float(os.environ.get("FINAL_EVAL_RESERVE_SCALE", 1.35))
    final_eval_estimate_batches: int = int(os.environ.get("FINAL_EVAL_ESTIMATE_BATCHES", 2))
    final_eval_serialization_seconds: float = float(os.environ.get("FINAL_EVAL_SERIALIZATION_SECONDS", 5.0))
    quant_aware_train_seconds: float = float(os.environ.get("QUANT_AWARE_TRAIN_SECONDS", 48.0))
    quant_aware_iters: int = int(os.environ.get("QUANT_AWARE_ITERS", 96))
    quant_aware_every: int = int(os.environ.get("QUANT_AWARE_EVERY", 24))
    quant_aware_embed_lr_mul: float = float(os.environ.get("QUANT_AWARE_EMBED_LR_MUL", 0.6))
    quant_aware_matrix_lr_mul: float = float(os.environ.get("QUANT_AWARE_MATRIX_LR_MUL", 0.35))
    quant_aware_scalar_lr_mul: float = float(os.environ.get("QUANT_AWARE_SCALAR_LR_MUL", 0.8))
    quant_aware_proj_start: float = float(os.environ.get("QUANT_AWARE_PROJ_START", 0.55))
    quant_aware_proj_step: float = float(os.environ.get("QUANT_AWARE_PROJ_STEP", 0.2))
    quant_aware_proj_end: float = float(os.environ.get("QUANT_AWARE_PROJ_END", 0.95))
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
    eval_stride: int = int(os.environ.get("EVAL_STRIDE", 64))
    eval_doc_isolated: bool = bool(int(os.environ.get("EVAL_DOC_ISOLATED", "1")))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 2.0))
    tail_recur_blocks: int = int(os.environ.get("TAIL_RECUR_BLOCKS", 2))
    tail_recur_ramp_start: float = float(os.environ.get("TAIL_RECUR_RAMP_START", 0.55))
    tail_recur_ramp_end: float = float(os.environ.get("TAIL_RECUR_RAMP_END", 0.9))
    tail_recur_min_gain: float = float(os.environ.get("TAIL_RECUR_MIN_GAIN", 0.35))
    tail_recur_stage_gap: float = float(os.environ.get("TAIL_RECUR_STAGE_GAP", 0.16))
    tail_recur_stage_span: float = float(os.environ.get("TAIL_RECUR_STAGE_SPAN", 0.12))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.02))
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
    @property
    def use_single_microbatch_path(self) -> bool:
        return self.grad_accum_steps == 1 and self.microbatch_tokens <= self.mlx_max_microbatch_tokens
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
CONTROL_TENSOR_NAME_PATTERNS = tuple(pattern for pattern in os.environ.get("CONTROL_TENSOR_NAME_PATTERNS", "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,tail_recur_gates,tail_carry_gates").split(",") if pattern)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(pattern for pattern in os.environ.get("INT8_KEEP_FLOAT_FP32_NAME_PATTERNS", ",".join(CONTROL_TENSOR_NAME_PATTERNS)).split(",") if pattern)
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
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)
    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T
class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
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
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
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
class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, tail_recur_blocks: int):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens; self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.logit_bias = mx.zeros((vocab_size,), dtype=mx.float32); self.logit_gain = mx.array(1.0, dtype=mx.float32); self.bigram_rank = int(os.environ.get("BIGRAM_RANK", 64))
        if self.bigram_rank > 0:
            self.bigram_in = nn.Embedding(vocab_size, self.bigram_rank); self.bigram_out = CastedLinear(self.bigram_rank, vocab_size); self.bigram_scale = mx.array(0.0, dtype=mx.float32)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        tail_recur_span = min(max(tail_recur_blocks, 0), self.num_decoder_layers)
        skip_count = max(min(self.num_encoder_layers, self.num_decoder_layers) - max(tail_recur_span - 1, 0), 0)
        self.skip_weights = mx.ones((skip_count, dim), dtype=mx.float32)
        self.decoder_skip_start = self.num_decoder_layers - int(self.skip_weights.shape[0])
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.tail_recur_start = num_layers - tail_recur_span
        self.final_norm = RMSNormNoWeight()
        self.tail_recur_gates = mx.zeros((tail_recur_span, dim), dtype=mx.float32) if tail_recur_span > 0 else None; self.tail_carry_gates = mx.zeros((tail_recur_span, dim), dtype=mx.float32) if tail_recur_span > 0 else None
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight); b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)
    def softcap(self, logits: mx.array) -> mx.array: return self.logit_softcap * mx.tanh(logits / self.logit_softcap)
    def project_logits(self, x: mx.array, prev_ids: mx.array | None = None) -> mx.array:
        logits = self.logit_gain.astype(x.dtype) * (x @ self.tok_emb.weight.astype(x.dtype).T)
        if self.bigram_rank > 0 and prev_ids is not None:
            logits = logits + self.bigram_scale.astype(x.dtype) * self.bigram_out(self.bigram_in(prev_ids.reshape(-1)).astype(x.dtype))
        return self.softcap(logits + self.logit_bias.astype(x.dtype))
    def __call__(self, input_ids: mx.array, tail_recur_gains: mx.array | None = None) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            skip_idx = i - self.decoder_skip_start
            if 0 <= skip_idx < int(self.skip_weights.shape[0]) and skips:
                x = x + self.skip_weights[skip_idx].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        if self.tail_recur_gates is not None:
            tail_anchor = x
            for block_idx in range(len(self.blocks) - 1, self.tail_recur_start - 1, -1):
                recur_idx = block_idx - self.tail_recur_start
                recur_gain = 1.0 if tail_recur_gains is None else tail_recur_gains[recur_idx].astype(x.dtype)
                recur_x = x + recur_gain * mx.tanh(self.tail_carry_gates[recur_idx]).astype(x.dtype)[None, None, :] * (tail_anchor - x); x = recur_x + recur_gain * mx.tanh(self.tail_recur_gates[recur_idx]).astype(x.dtype)[None, None, :] * (self.blocks[block_idx](recur_x, x0) - recur_x)
        return self.final_norm(x)
    def loss(self, input_ids: mx.array, target_ids: mx.array, tail_recur_gains: mx.array | None = None) -> mx.array:
        x = self(input_ids, tail_recur_gains).reshape(-1, self.tok_emb.weight.shape[1]); y = target_ids.reshape(-1); prev_ids = input_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.project_logits(x, prev_ids)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.project_logits(x[s:e], prev_ids[s:e])
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)
    def masked_loss(self, input_ids: mx.array, target_ids: mx.array, loss_mask: mx.array, tail_recur_gains: mx.array | None = None) -> mx.array:
        x = self(input_ids, tail_recur_gains).reshape(-1, self.tok_emb.weight.shape[1]); y = target_ids.reshape(-1); prev_ids = input_ids.reshape(-1)
        mask = loss_mask.reshape(-1).astype(mx.float32)
        denom = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.project_logits(x, prev_ids)
            logits_f = logits.astype(mx.float32)
            token_loss = mx.logsumexp(logits_f, axis=-1) - mx.take_along_axis(logits_f, y[:, None], axis=-1).reshape(-1)
            return mx.sum(token_loss.astype(mx.float32) * mask) / denom
        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.project_logits(x[s:e], prev_ids[s:e])
            logits_f = logits.astype(mx.float32)
            token_loss = mx.logsumexp(logits_f, axis=-1) - mx.take_along_axis(logits_f, y[s:e, None], axis=-1).reshape(-1)
            loss_sum = mx.sum(token_loss.astype(mx.float32) * mask[s:e])
        return loss_sum / denom
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
        return out
class SplitOptimizers:
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
            if k != self.embed_key and k not in self.matrix_keys and (k == "skip_weights" or p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS) or int(p.size) <= 65_536)
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
    def step(
        self,
        model: GPT,
        grads_tree: dict,
        step: int,
        lr_mul: float,
        embed_lr_mul: float = 1.0,
        matrix_lr_mul: float = 1.0,
        scalar_lr_mul: float = 1.0,
    ) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul * matrix_lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul * embed_lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul * scalar_lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))
MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_PER_ROW_OFFSET_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984; INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_PROJ_CLIP_PERCENTILE = float(os.environ.get("INT8_PROJ_CLIP_PERCENTILE", 99.9)); INT8_PROJ_CLIP_Q = INT8_PROJ_CLIP_PERCENTILE / 100.0
INT8_ROW_OFFSET_MIN_RATIO = float(os.environ.get("INT8_ROW_OFFSET_MIN_RATIO", 0.02))
INT8_FP16_TAIL_FULL_BLOCKS = int(os.environ.get("INT8_FP16_TAIL_FULL_BLOCKS", 0))
INT8_FP16_TAIL_PROJ_BLOCKS = int(os.environ.get("INT8_FP16_TAIL_PROJ_BLOCKS", 2))
PROJ_EMA_DECAY = float(os.environ.get("PROJ_EMA_DECAY", 0.94))
INT8_TRANSPOSE_SUFFIXES = tuple(suffix for suffix in os.environ.get("INT8_TRANSPOSE_SUFFIXES", "mlp.fc.weight").split(",") if suffix)
INT8_FP16_KEEP_NAMES = tuple(
    name
    for name in os.environ.get("INT8_FP16_KEEP_NAMES", "tok_emb.weight").split(",")
    if name
)
BLOCK_FP16_MATRIX_SUFFIXES = (
    "attn.c_q.weight",
    "attn.c_k.weight",
    "attn.c_v.weight",
    "attn.proj.weight",
    "mlp.fc.weight",
    "mlp.proj.weight",
)
BLOCK_FP16_PROJ_SUFFIXES = ("attn.proj.weight", "mlp.proj.weight")
def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)
def int8_clip_q(name: str) -> float:
    return INT8_PROJ_CLIP_Q if name.endswith(BLOCK_FP16_PROJ_SUFFIXES) else INT8_CLIP_Q
def should_transpose_quantize(name: str, arr_ndim: int) -> bool: return arr_ndim == 2 and any(name.endswith(suffix) for suffix in INT8_TRANSPOSE_SUFFIXES)
def build_int8_fp16_keep_names(num_layers: int, tail_recur_blocks: int) -> set[str]:
    keep = set(INT8_FP16_KEEP_NAMES)
    for block_idx in range(max(num_layers - INT8_FP16_TAIL_FULL_BLOCKS, 0), num_layers):
        prefix = f"blocks.{block_idx}."
        keep.update(prefix + suffix for suffix in BLOCK_FP16_MATRIX_SUFFIXES)
    for block_idx in range(max(num_layers - INT8_FP16_TAIL_PROJ_BLOCKS, 0), num_layers):
        prefix = f"blocks.{block_idx}."
        keep.update(prefix + suffix for suffix in BLOCK_FP16_PROJ_SUFFIXES)
    for block_idx in range(max(num_layers - tail_recur_blocks, 0), num_layers):
        prefix = f"blocks.{block_idx}."
        keep.update(prefix + suffix for suffix in BLOCK_FP16_MATRIX_SUFFIXES[:2])
    if num_layers > 0:
        prefix = f"blocks.{num_layers - 1}."
        keep.update(prefix + suffix for suffix in BLOCK_FP16_MATRIX_SUFFIXES[2:3])
    return keep
def should_keep_float_tensor(name: str, arr: mx.array, int8_fp16_keep_names: set[str]) -> bool:
    return name in int8_fp16_keep_names or int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL
def should_track_ema_tensor(name: str, arr: mx.array, int8_fp16_keep_names: set[str]) -> bool:
    return name.endswith(BLOCK_FP16_PROJ_SUFFIXES) or should_keep_float_tensor(name, arr, int8_fp16_keep_names)
def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))
def quantize_float_array(name: str, arr: mx.array) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, bool]:
    f32 = _np_float32(arr)
    transposed = should_transpose_quantize(name, f32.ndim)
    if transposed:
        f32 = np.ascontiguousarray(f32.T)
    clip_q = int8_clip_q(name)
    if f32.ndim == 2:
        row_offset = np.mean(f32, axis=1, dtype=np.float32) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        centered = f32 - row_offset[:, None]
        mean_abs = float(np.mean(np.abs(row_offset), dtype=np.float64)) if row_offset.size else 0.0
        resid_abs = float(np.mean(np.abs(centered), dtype=np.float64)) if centered.size else 0.0
        if mean_abs >= INT8_ROW_OFFSET_MIN_RATIO * max(resid_abs, 1e-12):
            clip_abs = np.quantile(np.abs(centered), clip_q, axis=1) if centered.size else np.empty((centered.shape[0],), dtype=np.float32)
            clipped = np.clip(centered, -clip_abs[:, None], clip_abs[:, None])
            scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
            q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
            return (
                np.ascontiguousarray(q),
                np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)),
                np.ascontiguousarray(row_offset.astype(INT8_PER_ROW_OFFSET_DTYPE, copy=False)),
                transposed,
            )
        clip_abs = np.quantile(np.abs(f32), clip_q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)), None, transposed
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), clip_q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale, None, transposed
def quantize_state_dict_int8(
    flat_state: dict[str, mx.array],
    int8_fp16_keep_names: set[str],
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    offsets: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    transposed_names: list[str] = []
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
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
        if should_keep_float_tensor(name, arr, int8_fp16_keep_names):
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        q, s, o, transposed = quantize_float_array(name, arr)
        quantized[name] = q
        scales[name] = s
        if o is not None:
            offsets[name] = o
        if transposed:
            transposed_names.append(name)
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes + (0 if o is None else o.nbytes))
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_offset_v3",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if offsets:
        obj["offsets"] = offsets
    if transposed_names:
        obj["transposed_names"] = tuple(transposed_names)
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats
def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    offsets = quant_obj.get("offsets", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    transposed_names = frozenset(quant_obj.get("transposed_names", ()))
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
            if name in offsets:
                out_arr = out_arr + np.asarray(offsets[name], dtype=np.float32).reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        if name in transposed_names:
            out_arr = np.ascontiguousarray(out_arr.T)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype]) if isinstance(orig_dtype, str) else mx.array(out_arr)
    return out
def roundtrip_tensor_like_final(
    name: str,
    arr: mx.array,
    int8_fp16_keep_names: set[str],
) -> mx.array:
    if not mx.issubdtype(arr.dtype, mx.floating):
        return arr
    if should_keep_float_tensor(name, arr, int8_fp16_keep_names):
        if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
            return mx.array(_np_float32(arr), dtype=arr.dtype)
        if arr.dtype in {mx.float32, mx.bfloat16}:
            return mx.array(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False), dtype=arr.dtype)
        return mx.array(np.array(arr, copy=True), dtype=arr.dtype)
    q, s, o, transposed = quantize_float_array(name, arr)
    scale = np.asarray(s, dtype=np.float32)
    out_arr = q.astype(np.float32) * (scale.reshape((q.shape[0],) + (1,) * (q.ndim - 1)) if scale.ndim > 0 else float(scale))
    if o is not None:
        out_arr = out_arr + np.asarray(o, dtype=np.float32).reshape((q.shape[0],) + (1,) * (q.ndim - 1))
    if transposed:
        out_arr = np.ascontiguousarray(out_arr.T)
    return mx.array(np.ascontiguousarray(out_arr), dtype=arr.dtype)
def blend_tensor_toward_final(
    name: str,
    arr: mx.array,
    int8_fp16_keep_names: set[str],
    mix: float,
) -> mx.array:
    target = roundtrip_tensor_like_final(name, arr, int8_fp16_keep_names)
    if mix >= 1.0 or not mx.issubdtype(arr.dtype, mx.floating) or should_keep_float_tensor(name, arr, int8_fp16_keep_names):
        return target
    return arr + (target - arr) * mix
def apply_final_roundtrip_to_state(model: GPT, int8_fp16_keep_names: set[str], mix: float = 1.0) -> None:
    model.update(
        tree_unflatten(
            [
                (name, blend_tensor_toward_final(name, arr, int8_fp16_keep_names, mix))
                for name, arr in tree_flatten(model.state)
            ]
        )
    )
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
        if piece.startswith("\x18"):
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
def build_validation_doc_spans(tokens: np.ndarray, bos_token_id: int) -> list[tuple[int, int]]:
    if bos_token_id < 0:
        return [(0, int(tokens.size))]
    starts = np.flatnonzero(tokens == bos_token_id).astype(np.int64, copy=False)
    if starts.size == 0 or int(starts[0]) != 0:
        starts = np.concatenate((np.array([0], dtype=np.int64), starts))
    spans: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        end = int(starts[idx + 1]) if idx + 1 < starts.size else int(tokens.size)
        start_i = int(start)
        if end - start_i >= 2:
            spans.append((start_i, end))
    return spans if spans else [(0, int(tokens.size))]
def count_doc_eval_windows(total_targets: int, seq_len: int, stride: int) -> int:
    if total_targets <= 0:
        return 0
    if stride <= 0 or stride >= seq_len:
        return max((total_targets + seq_len - 1) // seq_len, 1)
    remaining = max(total_targets - seq_len, 0)
    return 1 + (remaining + stride - 1) // stride
def fill_doc_window(doc_tokens: np.ndarray, seq_len: int, bos_token_id: int, score_end: int, score_tokens: int, x_row: np.ndarray, y_row: np.ndarray, mask_row: np.ndarray) -> None:
    raw_start = max(score_end - seq_len, 0)
    chunk = doc_tokens[raw_start : score_end + 1]
    pad = seq_len + 1 - int(chunk.size)
    if pad > 0:
        padded = np.empty((seq_len + 1,), dtype=np.int32)
        padded[:pad] = bos_token_id
        padded[pad:] = chunk
        x_row[:] = padded[:-1]
        y_row[:] = padded[1:]
    else:
        x_row[:] = chunk[:-1]
        y_row[:] = chunk[1:]
    mask_row.fill(0.0)
    mask_row[-score_tokens:] = 1.0
def eval_val_doc_isolated(args: Hyperparameters, compiled_masked_loss, val_tokens: np.ndarray, doc_spans: list[tuple[int, int]], bos_token_id: int, tail_recur_gains: mx.array, base_bytes_lut: np.ndarray, has_leading_space_lut: np.ndarray, is_boundary_token_lut: np.ndarray, log_fn: Callable[[str], None] | None = None) -> tuple[float, float]:
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_windows = sum(count_doc_eval_windows(end - start - 1, args.train_seq_len, args.eval_stride) for start, end in doc_spans)
    total_batches = max((total_windows + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    batch_idx = 0
    x_np = np.empty((val_batch_seqs, args.train_seq_len), dtype=np.int32)
    y_np = np.empty_like(x_np)
    mask_np = np.zeros((val_batch_seqs, args.train_seq_len), dtype=np.float32)
    pending = 0
    batch_token_count = 0.0
    batch_bytes = 0.0
    def flush_batch(num_rows: int, token_count: float, byte_count: float) -> tuple[float, float]:
        nonlocal batch_idx, total_loss_sum, total_tokens, total_bytes
        if num_rows <= 0:
            return 0.0, 0.0
        batch_idx += 1
        x = mx.array(x_np[:num_rows], dtype=mx.int32)
        y = mx.array(y_np[:num_rows], dtype=mx.int32)
        mask = mx.array(mask_np[:num_rows], dtype=mx.float32)
        batch_loss = compiled_masked_loss(x, y, mask, tail_recur_gains).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * token_count
        total_tokens += token_count
        total_bytes += byte_count
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
        return 0.0, 0.0
    for start, end in doc_spans:
        doc_tokens = val_tokens[start:end]
        total_doc_targets = int(doc_tokens.size - 1)
        if total_doc_targets <= 0:
            continue
        if args.eval_stride <= 0 or args.eval_stride >= args.train_seq_len:
            score_end = 0
            while score_end < total_doc_targets:
                next_score_end = min(score_end + args.train_seq_len, total_doc_targets)
                score_tokens = next_score_end - score_end
                fill_doc_window(
                    doc_tokens,
                    args.train_seq_len,
                    bos_token_id,
                    next_score_end,
                    score_tokens,
                    x_np[pending],
                    y_np[pending],
                    mask_np[pending],
                )
                prev_ids = x_np[pending, -score_tokens:]
                tgt_ids = y_np[pending, -score_tokens:]
                bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
                bytes_np += (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).astype(np.int16, copy=False)
                pending += 1
                batch_token_count += float(score_tokens)
                batch_bytes += float(bytes_np.astype(np.float64).sum())
                score_end = next_score_end
                if pending == val_batch_seqs:
                    batch_token_count, batch_bytes = flush_batch(pending, batch_token_count, batch_bytes)
                    pending = 0
            continue
        score_end = min(args.train_seq_len, total_doc_targets)
        prev_score_end = 0
        while True:
            score_tokens = score_end - prev_score_end
            fill_doc_window(
                doc_tokens,
                args.train_seq_len,
                bos_token_id,
                score_end,
                score_tokens,
                x_np[pending],
                y_np[pending],
                mask_np[pending],
            )
            prev_ids = x_np[pending, -score_tokens:]
            tgt_ids = y_np[pending, -score_tokens:]
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            pending += 1
            batch_token_count += float(score_tokens)
            batch_bytes += float(bytes_np.astype(np.float64).sum())
            prev_score_end = score_end
            if pending == val_batch_seqs:
                batch_token_count, batch_bytes = flush_batch(pending, batch_token_count, batch_bytes)
                pending = 0
            if score_end >= total_doc_targets:
                break
            score_end = min(score_end + args.eval_stride, total_doc_targets)
    if pending:
        flush_batch(pending, batch_token_count, batch_bytes)
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb
def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    tail_recur_gains: mx.array,
) -> tuple[mx.array, dict]:
    chunk_sizes = token_chunks(args.microbatch_tokens, args.train_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y, tail_recur_gains)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)
    return loss_value, tree_unflatten(list(grad_accum.items()))
def loss_and_grad_one_batch(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    tail_recur_gains: mx.array,
) -> tuple[mx.array, dict]:
    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len)
    return compiled_loss_and_grad(x, y, tail_recur_gains)
def eval_val(args: Hyperparameters, compiled_loss, compiled_masked_loss, val_tokens: np.ndarray, doc_spans: list[tuple[int, int]] | None, bos_token_id: int, tail_recur_gains: mx.array, base_bytes_lut: np.ndarray, has_leading_space_lut: np.ndarray, is_boundary_token_lut: np.ndarray, log_fn: Callable[[str], None] | None = None) -> tuple[float, float]:
    if args.eval_doc_isolated and doc_spans is not None and bos_token_id >= 0:
        return eval_val_doc_isolated(
            args,
            compiled_masked_loss,
            val_tokens,
            doc_spans,
            bos_token_id,
            tail_recur_gains,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log_fn,
        )
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    if args.eval_stride <= 0 or args.eval_stride >= args.train_seq_len:
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
            batch_loss = compiled_loss(x, y, tail_recur_gains).astype(mx.float32)
            mx.eval(batch_loss)
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
        bits_per_token = val_loss / math.log(2.0)
        val_bpb = bits_per_token * (total_tokens / total_bytes)
        return val_loss, val_bpb
    stride = args.eval_stride
    available_targets = val_tokens.size - 1
    usable_targets = args.train_seq_len + max(((available_targets - args.train_seq_len) // stride), 0) * stride
    total_windows = 1 + max((usable_targets - args.train_seq_len) // stride, 0)
    total_batches = max((total_windows + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, window_idx_start in enumerate(range(0, total_windows, val_batch_seqs), start=1):
        window_idx_end = min(window_idx_start + val_batch_seqs, total_windows)
        x_np = np.empty((window_idx_end - window_idx_start, args.train_seq_len), dtype=np.int32)
        y_np = np.empty_like(x_np)
        mask_np = np.zeros((window_idx_end - window_idx_start, args.train_seq_len), dtype=np.float32)
        batch_token_count = 0.0
        batch_bytes = 0.0
        for local_idx, window_idx in enumerate(range(window_idx_start, window_idx_end)):
            raw_start = window_idx * stride
            raw_end = raw_start + args.train_seq_len
            chunk = val_tokens[raw_start : raw_end + 1]
            x_np[local_idx] = chunk[:-1]
            y_np[local_idx] = chunk[1:]
            score_tokens = args.train_seq_len if window_idx == 0 else stride
            mask_np[local_idx, -score_tokens:] = 1.0
            prev_ids = x_np[local_idx, -score_tokens:]
            tgt_ids = y_np[local_idx, -score_tokens:]
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            batch_token_count += float(score_tokens)
            batch_bytes += float(bytes_np.astype(np.float64).sum())
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        mask = mx.array(mask_np, dtype=mx.float32)
        batch_loss = compiled_masked_loss(x, y, mask, tail_recur_gains).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * batch_token_count
        total_tokens += batch_token_count
        total_bytes += batch_bytes
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb
def estimate_eval_time_ms(args: Hyperparameters, compiled_loss, compiled_masked_loss, val_tokens: np.ndarray, doc_spans: list[tuple[int, int]] | None, bos_token_id: int, tail_recur_gains: mx.array) -> float:
    if args.eval_doc_isolated and doc_spans is not None and bos_token_id >= 0:
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        val_batch_seqs = val_batch_tokens // args.train_seq_len
        total_units = max(
            (
                sum(count_doc_eval_windows(end - start - 1, args.train_seq_len, args.eval_stride) for start, end in doc_spans)
                + val_batch_seqs
                - 1
            )
            // val_batch_seqs,
            1,
        )
        sample_units = min(max(args.final_eval_estimate_batches, 1), total_units)
        x_np = np.empty((val_batch_seqs, args.train_seq_len), dtype=np.int32)
        y_np = np.empty_like(x_np)
        mask_np = np.zeros((val_batch_seqs, args.train_seq_len), dtype=np.float32)
        start = time.perf_counter()
        pending = 0
        seen_units = 0
        for doc_start, doc_end in doc_spans:
            doc_tokens = val_tokens[doc_start:doc_end]
            total_doc_targets = int(doc_tokens.size - 1)
            if total_doc_targets <= 0:
                continue
            if args.eval_stride <= 0 or args.eval_stride >= args.train_seq_len:
                score_end = 0
                while score_end < total_doc_targets:
                    next_score_end = min(score_end + args.train_seq_len, total_doc_targets)
                    fill_doc_window(
                        doc_tokens,
                        args.train_seq_len,
                        bos_token_id,
                        next_score_end,
                        next_score_end - score_end,
                        x_np[pending],
                        y_np[pending],
                        mask_np[pending],
                    )
                    pending += 1
                    score_end = next_score_end
                    if pending == val_batch_seqs:
                        batch_loss = compiled_masked_loss(
                            mx.array(x_np, dtype=mx.int32),
                            mx.array(y_np, dtype=mx.int32),
                            mx.array(mask_np, dtype=mx.float32),
                            tail_recur_gains,
                        ).astype(mx.float32)
                        mx.eval(batch_loss)
                        seen_units += 1
                        pending = 0
                        if seen_units >= sample_units:
                            mx.synchronize()
                            sample_ms = 1000.0 * (time.perf_counter() - start)
                            return sample_ms * total_units / max(sample_units, 1)
                continue
            score_end = min(args.train_seq_len, total_doc_targets)
            prev_score_end = 0
            while True:
                fill_doc_window(
                    doc_tokens,
                    args.train_seq_len,
                    bos_token_id,
                    score_end,
                    score_end - prev_score_end,
                    x_np[pending],
                    y_np[pending],
                    mask_np[pending],
                )
                pending += 1
                prev_score_end = score_end
                if pending == val_batch_seqs:
                    batch_loss = compiled_masked_loss(
                        mx.array(x_np, dtype=mx.int32),
                        mx.array(y_np, dtype=mx.int32),
                        mx.array(mask_np, dtype=mx.float32),
                        tail_recur_gains,
                    ).astype(mx.float32)
                    mx.eval(batch_loss)
                    seen_units += 1
                    pending = 0
                    if seen_units >= sample_units:
                        mx.synchronize()
                        sample_ms = 1000.0 * (time.perf_counter() - start)
                        return sample_ms * total_units / max(sample_units, 1)
                if score_end >= total_doc_targets:
                    break
                score_end = min(score_end + args.eval_stride, total_doc_targets)
        if pending:
            batch_loss = compiled_masked_loss(
                mx.array(x_np[:pending], dtype=mx.int32),
                mx.array(y_np[:pending], dtype=mx.int32),
                mx.array(mask_np[:pending], dtype=mx.float32),
                tail_recur_gains,
            ).astype(mx.float32)
            mx.eval(batch_loss)
            seen_units += 1
        mx.synchronize()
        sample_ms = 1000.0 * (time.perf_counter() - start)
        return sample_ms * total_units / max(seen_units, 1)
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    if args.eval_stride <= 0 or args.eval_stride >= args.train_seq_len:
        total_units = max(((val_tokens.size - 1) // args.train_seq_len + val_batch_seqs - 1) // val_batch_seqs, 1)
        sample_units = min(max(args.final_eval_estimate_batches, 1), total_units)
        start = time.perf_counter()
        for batch_idx, batch_seq_start in enumerate(range(0, (val_tokens.size - 1) // args.train_seq_len, val_batch_seqs), start=1):
            batch_seq_end = min(batch_seq_start + val_batch_seqs, (val_tokens.size - 1) // args.train_seq_len)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            chunk = val_tokens[raw_start:raw_end]
            x = mx.array(chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
            y = mx.array(chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
            batch_loss = compiled_loss(x, y, tail_recur_gains).astype(mx.float32)
            mx.eval(batch_loss)
            if batch_idx >= sample_units:
                break
        mx.synchronize()
        sample_ms = 1000.0 * (time.perf_counter() - start)
        return sample_ms * total_units / max(sample_units, 1)
    stride = args.eval_stride
    available_targets = val_tokens.size - 1
    usable_targets = args.train_seq_len + max(((available_targets - args.train_seq_len) // stride), 0) * stride
    total_windows = 1 + max((usable_targets - args.train_seq_len) // stride, 0)
    total_units = max((total_windows + val_batch_seqs - 1) // val_batch_seqs, 1)
    sample_units = min(max(args.final_eval_estimate_batches, 1), total_units)
    start = time.perf_counter()
    for batch_idx, window_idx_start in enumerate(range(0, total_windows, val_batch_seqs), start=1):
        window_idx_end = min(window_idx_start + val_batch_seqs, total_windows)
        x_np = np.empty((window_idx_end - window_idx_start, args.train_seq_len), dtype=np.int32)
        y_np = np.empty_like(x_np)
        mask_np = np.zeros((window_idx_end - window_idx_start, args.train_seq_len), dtype=np.float32)
        for local_idx, window_idx in enumerate(range(window_idx_start, window_idx_end)):
            raw_start = window_idx * stride
            raw_end = raw_start + args.train_seq_len
            chunk = val_tokens[raw_start : raw_end + 1]
            x_np[local_idx] = chunk[:-1]
            y_np[local_idx] = chunk[1:]
            score_tokens = args.train_seq_len if window_idx == 0 else stride
            mask_np[local_idx, -score_tokens:] = 1.0
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        mask = mx.array(mask_np, dtype=mx.float32)
        batch_loss = compiled_masked_loss(x, y, mask, tail_recur_gains).astype(mx.float32)
        mx.eval(batch_loss)
        if batch_idx >= sample_units:
            break
    mx.synchronize()
    sample_ms = 1000.0 * (time.perf_counter() - start)
    return sample_ms * total_units / max(sample_units, 1)
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
def should_activate_quant_aware(args: Hyperparameters, step: int, elapsed_ms: float, max_wallclock_ms: float | None, reserved_final_ms: float) -> bool:
    if args.quant_aware_every <= 0:
        return False
    if max_wallclock_ms is None:
        return args.quant_aware_iters > 0 and step >= max(args.iterations - args.quant_aware_iters, 0)
    return elapsed_ms >= max(max_wallclock_ms - reserved_final_ms - 1000.0 * args.quant_aware_train_seconds, 0.0)
def quant_aware_lr_muls(args: Hyperparameters, quant_aware_active: bool) -> tuple[float, float, float]:
    if not quant_aware_active:
        return 1.0, 1.0, 1.0
    return args.quant_aware_embed_lr_mul, args.quant_aware_matrix_lr_mul, args.quant_aware_scalar_lr_mul
def tail_recur_schedule(args: Hyperparameters, step: int, active_blocks: int) -> mx.array:
    if active_blocks <= 0:
        return mx.zeros((0,), dtype=mx.float32)
    if args.tail_recur_ramp_end <= args.tail_recur_ramp_start:
        progress = 1.0 if step > 0 else 0.0
    else:
        progress = step / max(args.iterations, 1)
        progress = min(max((progress - args.tail_recur_ramp_start) / (args.tail_recur_ramp_end - args.tail_recur_ramp_start), 0.0), 1.0)
    if active_blocks == 1:
        return mx.ones((1,), dtype=mx.float32)
    gains = np.ones((active_blocks,), dtype=np.float32)
    for idx in range(active_blocks - 1):
        stage_start = idx * args.tail_recur_stage_gap
        stage_span = max(args.tail_recur_stage_span, 1e-6)
        stage_progress = min(max((progress - stage_start) / stage_span, 0.0), 1.0)
        gains[idx] = args.tail_recur_min_gain + (1.0 - args.tail_recur_min_gain) * stage_progress
    gains[-1] = 1.0
    return mx.array(gains, dtype=mx.float32)
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
    if _OPTIONAL_IMPORT_ERROR is not None:
        raise RuntimeError(f"Optional dependency missing for execution: {_OPTIONAL_IMPORT_ERROR.name}") from _OPTIONAL_IMPORT_ERROR
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
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bos_token_id = int(sp.bos_id())
    doc_spans = build_validation_doc_spans(val_tokens, bos_token_id) if bos_token_id >= 0 else None
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    model = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult, logit_chunk_tokens=args.logit_chunk_tokens, logit_softcap=args.logit_softcap, rope_base=args.rope_base, tied_embed_init_std=args.tied_embed_init_std, qk_gain_init=args.qk_gain_init, tail_recur_blocks=args.tail_recur_blocks)
    int8_fp16_keep_names = build_int8_fp16_keep_names(args.num_layers, args.tail_recur_blocks)
    opt = SplitOptimizers(model, args)
    tail_recur_eval_gains = mx.ones((args.tail_recur_blocks,), dtype=mx.float32)
    compiled_loss = mx.compile(lambda x, y, rg: model.loss(x, y, rg), inputs=model.state, outputs=model.state)
    compiled_masked_loss = mx.compile(lambda x, y, m, rg: model.masked_loss(x, y, m, rg), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(nn.value_and_grad(model, lambda x, y, rg: model.loss(x, y, rg)), inputs=model.state, outputs=model.state)
    if "MLX_EAGER_EVAL" not in os.environ:
        args.mlx_eager_eval = not args.use_single_microbatch_path
    elif args.use_single_microbatch_path and args.mlx_eager_eval:
        log("WARNING: disabling MLX_EAGER_EVAL on single_microbatch_path for throughput")
        args.mlx_eager_eval = False
    train_step_loss_and_grad = loss_and_grad_one_batch if args.use_single_microbatch_path else loss_and_grad_chunked
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(f"WARNING: train_loader:subset dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files} new epochs will arrive sooner than the full dataset")
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} val_batch_size:{args.val_batch_size} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} embed_lr:{args.tied_embed_lr} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}")
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    eval_mode = "doc_isolated_sliding" if args.eval_doc_isolated and doc_spans is not None and bos_token_id >= 0 else "flat_stream"
    log(f"eval_mode:{eval_mode} bos_token_id:{bos_token_id} val_docs:{0 if doc_spans is None else len(doc_spans)}")
    log(f"eval_stride:{args.eval_stride}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(f"dtypes tok_emb:{model.tok_emb.weight.dtype} linear_weight:{model.blocks[0].attn.c_q.weight.dtype} skip_weights:{model.skip_weights.dtype}")
    estimated_final_eval_ms = estimate_eval_time_ms(args, compiled_loss, compiled_masked_loss, val_tokens, doc_spans, bos_token_id, tail_recur_eval_gains)
    reserved_final_ms = max(
        1000.0 * args.final_eval_reserve_seconds,
        estimated_final_eval_ms * args.final_eval_reserve_scale + 1000.0 * args.final_eval_serialization_seconds,
    )
    log(f"final_eval_budget:estimate_ms:{estimated_final_eval_ms:.0f} reserve_ms:{reserved_final_ms:.0f} estimate_batches:{args.final_eval_estimate_batches}")
    log(f"quant_aware:train_seconds:{args.quant_aware_train_seconds:.1f} iters:{args.quant_aware_iters} every:{args.quant_aware_every}")
    log(f"quant_aware_lr_mul:embed:{args.quant_aware_embed_lr_mul} matrix:{args.quant_aware_matrix_lr_mul} scalar:{args.quant_aware_scalar_lr_mul}")
    log(f"quant_aware_proj_mix:start:{args.quant_aware_proj_start} step:{args.quant_aware_proj_step} end:{args.quant_aware_proj_end}")
    log(f"int8_transpose_suffixes:{','.join(INT8_TRANSPOSE_SUFFIXES) if INT8_TRANSPOSE_SUFFIXES else 'none'}")
    log(f"int8_fp16_keep:count:{len(int8_fp16_keep_names)} tail_full_blocks:{INT8_FP16_TAIL_FULL_BLOCKS} tail_proj_blocks:{INT8_FP16_TAIL_PROJ_BLOCKS}")
    log(
        f"tail_recur:blocks:{args.tail_recur_blocks} start:{model.tail_recur_start} "
        f"active:{0 if model.tail_recur_gates is None else model.tail_recur_gates.shape[0]}"
    )
    log(f"decoder_skip_alignment:start:{model.decoder_skip_start} count:{model.skip_weights.shape[0]} trim:{max((0 if model.tail_recur_gates is None else model.tail_recur_gates.shape[0]) - 1, 0)}")
    log("tail_recur_order:reverse"); log("tail_recur_carry:decoder_output_anchor")
    log(f"tail_recur_curriculum:min_gain:{args.tail_recur_min_gain} ramp_start:{args.tail_recur_ramp_start} ramp_end:{args.tail_recur_ramp_end}")
    log(f"tail_recur_staging:gap:{args.tail_recur_stage_gap} span:{args.tail_recur_stage_span}")
    log("tail_ema:decay:{:.2f} tracked_float_kept:all tracked_proj_suffixes:all".format(PROJ_EMA_DECAY))
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    last_quant_aware_step: int | None = None
    quant_aware_proj_mix = args.quant_aware_proj_start
    tracked_ema: dict[str, mx.array] | None = None
    tracked_ema_keep = 1.0 - PROJ_EMA_DECAY
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                compiled_masked_loss,
                val_tokens,
                doc_spans,
                bos_token_id,
                tail_recur_eval_gains,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
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
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        lr_mul = args.lr_mul(step, approx_train_time_ms)
        quant_aware_active = (
            last_quant_aware_step is not None
            or should_activate_quant_aware(args, step, approx_train_time_ms, max_wallclock_ms, reserved_final_ms)
        )
        embed_lr_mul, matrix_lr_mul, scalar_lr_mul = quant_aware_lr_muls(args, quant_aware_active)
        step_t0 = time.perf_counter()
        should_log_train = args.train_log_every > 0 and (
            step < 10 or (step + 1) % args.train_log_every == 0 or stop_after_step is not None
        )
        tail_recur_gains = tail_recur_schedule(args, step, args.tail_recur_blocks)
        if args.use_single_microbatch_path:
            train_loss, grads = train_step_loss_and_grad(args, train_loader, compiled_loss_and_grad, tail_recur_gains)
            if args.mlx_eager_eval:
                mx.eval(train_loss, grads)
            grads = clip_grad_tree(grads, args.grad_clip_norm)
        else:
            accum: dict[str, mx.array] | None = None
            train_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                loss, grads = train_step_loss_and_grad(args, train_loader, compiled_loss_and_grad, tail_recur_gains)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
                train_loss = train_loss + loss.astype(mx.float32) * grad_scale
                if args.mlx_eager_eval:
                    mx.eval(train_loss, accum)
            grads = tree_unflatten(list(accum.items()))
            grads = clip_grad_tree(grads, args.grad_clip_norm)
        opt.step(
            model,
            grads,
            step=step,
            lr_mul=lr_mul,
            embed_lr_mul=embed_lr_mul,
            matrix_lr_mul=matrix_lr_mul,
            scalar_lr_mul=scalar_lr_mul,
        )
        next_step = step + 1
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        did_quant_aware_roundtrip = False
        if should_activate_quant_aware(args, next_step, approx_train_time_ms, max_wallclock_ms, reserved_final_ms) and (
            last_quant_aware_step is None or next_step - last_quant_aware_step >= args.quant_aware_every
        ):
            apply_final_roundtrip_to_state(model, int8_fp16_keep_names, mix=quant_aware_proj_mix)
            last_quant_aware_step = next_step
            quant_aware_proj_mix = min(quant_aware_proj_mix + args.quant_aware_proj_step, args.quant_aware_proj_end)
            did_quant_aware_roundtrip = True
        if last_quant_aware_step is not None:
            flat_params = dict(tree_flatten(model.parameters()))
            if tracked_ema is None:
                tracked_ema = {
                    name: arr + mx.zeros_like(arr)
                    for name, arr in flat_params.items()
                    if mx.issubdtype(arr.dtype, mx.floating) and should_track_ema_tensor(name, arr, int8_fp16_keep_names)
                }
            else:
                for name in tracked_ema:
                    tracked_ema[name] = tracked_ema[name] + (flat_params[name] - tracked_ema[name]) * tracked_ema_keep
        mx.synchronize()
        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step = next_step
        if did_quant_aware_roundtrip:
            log(f"quant_aware_roundtrip:step:{step}")
        if should_log_train:
            train_loss_value = float(train_loss.item())
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
            )
        if (
            max_wallclock_ms is not None
            and stop_after_step is None
            and approx_train_time_ms >= max(max_wallclock_ms - reserved_final_ms, 0.0)
        ):
            stop_after_step = step
    if last_quant_aware_step is not None and last_quant_aware_step != step:
        apply_final_roundtrip_to_state(model, int8_fp16_keep_names)
        mx.synchronize()
        log(f"quant_aware_roundtrip:step:{step} final_pre_save")
    if tracked_ema:
        model.update(tree_unflatten(list(tracked_ema.items())))
        apply_final_roundtrip_to_state(model, int8_fp16_keep_names)
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state, int8_fp16_keep_names)
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
    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        compiled_loss,
        compiled_masked_loss,
        val_tokens,
        doc_spans,
        bos_token_id,
        tail_recur_eval_gains,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        log_fn=log,
    )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    total_train_tokens = step * args.train_batch_tokens
    if train_time_ms > 0.0:
        log(f"throughput:avg_tok_s:{total_train_tokens / (train_time_ms / 1000.0):.2f} total_tokens:{total_train_tokens}")
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
if __name__ == "__main__":
    main()
