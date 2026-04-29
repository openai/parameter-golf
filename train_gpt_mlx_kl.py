#!/usr/bin/env python3
"""KaiLean's Parameter Golf script — GPT training with int6 QAT, EMA, BigramHash,
EngramLite, SmearGate, XSA, complementary training, BackoffNgramMixer, and LoRA TTT."""
from __future__ import annotations
import glob, json, math, os, pickle, sys, time, uuid, copy
import zstandard
from collections import defaultdict
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

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 3500))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 11))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings: bool = True
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    eval_seq_len: int    = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride: int     = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs: int = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    bigram_hash_size: int = int(os.environ.get("BIGRAM_HASH_SIZE", 16384))
    qat_start_frac: float = float(os.environ.get("QAT_START_FRAC", 0.15))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.995))
    ema_start_frac: float = float(os.environ.get("EMA_START_FRAC", 0.5))
    use_ortho_init: bool = bool(int(os.environ.get("USE_ORTHO_INIT", "1")))
    use_swa: bool = bool(int(os.environ.get("USE_SWA", "0")))
    swa_decay: float = float(os.environ.get("SWA_DECAY", "0.4"))
    smear_enabled: bool    = bool(int(os.environ.get("USE_SMEARGATE", os.environ.get("SMEAR_ENABLED", "1"))))
    rope_dims: int         = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale_enabled: bool = bool(int(os.environ.get(
        "LN_SCALE_ENABLED", os.environ.get("USE_LN_SCALE", "1"))))
    xsa_last_n: int        = int(os.environ.get("XSA_LAST_N", 4))
    engram_lite_enabled: bool = bool(int(os.environ.get("ENGRAM_LITE_ENABLED", "0")))
    engram_hash_size: int = int(os.environ.get("ENGRAM_HASH_SIZE", "2048"))
    engram_embed_dim: int = int(os.environ.get("ENGRAM_EMBED_DIM", "128"))
    engram_n_heads: int = int(os.environ.get("ENGRAM_N_HEADS", "2"))
    skipgram_hash_size: int = int(os.environ.get("SKIPGRAM_HASH_SIZE", "0"))
    complement_alpha: float = float(os.environ.get("COMPLEMENT_ALPHA", "0.0"))
    ngram_mixer_enabled: bool = bool(int(os.environ.get("NGRAM_MIXER_ENABLED", "0")))
    ngram_alpha: float = float(os.environ.get("NGRAM_ALPHA", "0.25"))
    ngram_max_order: int = int(os.environ.get("NGRAM_MAX_ORDER", "4"))
    eval_mode: str = os.environ.get("EVAL_MODE", "sliding")
    ttt_enabled: bool = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_rank: int = int(os.environ.get("TTT_RANK", 4))
    ttt_lr: float = float(os.environ.get("TTT_LR", 0.001))
    ttt_steps: int = int(os.environ.get("TTT_STEPS", 2))
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
    muon_weight_decay: float = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adam_weight_decay: float = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.04))
    late_qat_threshold: float = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    use_gptq_lite: bool = bool(int(os.environ.get("USE_GPTQ_LITE", "1")))
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
            if warmdown_start <= step < self.iterations:
                return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
    "smear",
)

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
    return tokens.astype(np.int32, copy=False)

class TokenStream:
    def __init__(self, pattern: str, log_fn=None, dataset_name: str = ""):
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
    def __init__(self, pattern: str, log_fn=None, dataset_name: str = ""):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

class BigramHashEmbedding(nn.Module):
    def __init__(self, hash_size: int, dim: int):
        super().__init__()
        self.hash_size = hash_size
        self.table = nn.Embedding(hash_size, dim)
        self.table.weight = self.table.weight * 0.02

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens: (B, T) int32 → bigram embeddings: (B, T, dim)"""
        t_prev = tokens[:, :-1]
        t_curr = tokens[:, 1:]
        idx = mx.remainder(t_prev * 31337 + t_curr, self.hash_size)
        bigram_emb = self.table(idx)
        pad = mx.zeros((tokens.shape[0], 1, bigram_emb.shape[-1]), dtype=bigram_emb.dtype)
        return mx.concatenate([pad, bigram_emb], axis=1)

class EngramLiteEmbedding(nn.Module):
    def __init__(self, hash_size: int = 2048, embed_dim: int = 128,
                 output_dim: int = 1024, n_heads: int = 2,
                 orders: tuple = (2, 3)):
        super().__init__()
        self.hash_size = hash_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.orders = list(orders)
        _all_primes = [31337, 59999, 73721, 97531]
        if n_heads > len(_all_primes):
            raise ValueError(f"EngramLiteEmbedding: n_heads={n_heads} exceeds max supported ({len(_all_primes)})")
        self._primes = _all_primes[:n_heads]
        self.tables = {
            f"order_{o}": nn.Embedding(hash_size, embed_dim)
            for o in orders
        }
        for tbl in self.tables.values():
            tbl.weight = tbl.weight * 0.01
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        self.proj.weight = self.proj.weight * 0.01
        self.gate_proj = nn.Linear(embed_dim, len(orders), bias=True)
        self.gate_proj.bias = mx.full((len(orders),), -2.0)
        self.gate_proj.weight = self.gate_proj.weight * 0.01

    def _hash_ngram(self, tokens: mx.array, order: int, head_idx: int):
        """Hash n-gram context for given order and hash head."""
        prime = self._primes[head_idx]
        if order == 2:
            t_prev = tokens[:, :-1]
            t_curr = tokens[:, 1:]
            idx = mx.remainder(t_prev * prime + t_curr, self.hash_size)
            valid_start = 1
        elif order == 3:
            t_prev2 = tokens[:, :-2]
            t_prev1 = tokens[:, 1:-1]
            t_curr  = tokens[:, 2:]
            idx = mx.remainder(
                t_prev2 * (prime * prime) + t_prev1 * prime + t_curr,
                self.hash_size
            )
            valid_start = 2
        else:
            raise ValueError(f"n-gram order {order} not supported")
        return idx, valid_start

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens: (B, T) → (B, T, output_dim) additive logit bias"""
        B, T = tokens.shape
        combined = mx.zeros((B, T, self.embed_dim), dtype=mx.float32)
        for order in self.orders:
            tbl = self.tables[f"order_{order}"]
            head_sum = None
            for hi in range(self.n_heads):
                idx, valid_start = self._hash_ngram(tokens, order, hi)
                emb = tbl(idx).astype(mx.float32)
                pad = mx.zeros((B, valid_start, self.embed_dim), dtype=mx.float32)
                emb = mx.concatenate([pad, emb], axis=1)
                head_sum = emb if head_sum is None else head_sum + emb
            combined = combined + head_sum / self.n_heads
        gate = mx.sigmoid(self.gate_proj(combined))
        gate_scalar = gate.mean(axis=-1, keepdims=True)
        return self.proj(combined) * gate_scalar

class SkipGramHashEmbedding(nn.Module):
    def __init__(self, hash_size: int = 4096, dim: int = 1024,
                 skip_patterns: list = None):
        super().__init__()
        self.hash_size = hash_size
        self.dim = dim
        self.skip_patterns = skip_patterns if skip_patterns is not None else [[-1, -3], [-1, -5], [-2, -4]]
        self.tables = {
            f"skip_{i}": nn.Embedding(hash_size, dim)
            for i in range(len(self.skip_patterns))
        }
        for tbl in self.tables.values():
            tbl.weight = tbl.weight * 0.01

    def __call__(self, tokens: mx.array) -> mx.array:
        """tokens: (B, T) → (B, T, dim) additive logit bias"""
        B, T = tokens.shape
        output = mx.zeros((B, T, self.dim), dtype=mx.float32)
        for i, pattern in enumerate(self.skip_patterns):
            tbl = self.tables[f"skip_{i}"]
            min_offset = min(pattern)
            valid_start = abs(min_offset)
            if valid_start >= T:
                continue
            hash_val = mx.zeros((B, T - valid_start), dtype=mx.int32)
            prime = 31337
            for offset in pattern:
                start = valid_start + offset
                end = T + offset
                hash_val = hash_val * prime + tokens[:, start:end]
            idx = mx.remainder(mx.abs(hash_val), self.hash_size)
            emb = tbl(idx).astype(mx.float32)
            pad = mx.zeros((B, valid_start, self.dim), dtype=mx.float32)
            emb = mx.concatenate([pad, emb], axis=1)
            output = output + emb
        return output

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.full((dim,), 3.0, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        g = mx.sigmoid(self.gate).astype(x.dtype)
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        return g * x + (1.0 - g) * x_prev

def fake_quant_int6(w: mx.array) -> mx.array:
    """Simulate int6 quantization during training (STE for gradients)."""
    scale = mx.max(mx.abs(w), keepdims=True) / 31.0 + 1e-8
    w_q = mx.clip(mx.round(w / scale), -32, 31) * scale
    return w + mx.stop_gradient(w_q - w)

class EMABuffer:
    def __init__(self, model, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        for k, v in tree_flatten(model.parameters()):
            key = ".".join(str(p) for p in k) if isinstance(k, (list, tuple)) else k
            self.shadow[key] = mx.array(v)

    def update(self, model):
        d = self.decay
        for k, v in tree_flatten(model.parameters()):
            key = ".".join(str(p) for p in k) if isinstance(k, (list, tuple)) else k
            if key in self.shadow:
                self.shadow[key] = d * self.shadow[key] + (1.0 - d) * v
        mx.eval(list(self.shadow.values()))

    def apply(self, model):
        """Replace model weights with EMA weights."""
        model.update(tree_unflatten(list(self.shadow.items())))

    def state_dict(self):
        return dict(self.shadow)

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array, use_qat: bool = False) -> mx.array:
        w = self.weight
        if use_qat:
            w = fake_quant_int6(w)
        return x @ w.astype(x.dtype).T

class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float,
                 rope_dims: int = 0, use_xsa: bool = False):
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
        if rope_dims > 0:
            self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_base)
        else:
            self.rope = None
        self.scale = self.head_dim ** -0.5
        self.use_xsa = use_xsa

    def _xsa(self, y: mx.array, v: mx.array) -> mx.array:
        """XSA: subtract self-value component (PR #198)."""
        B, H, T, D = y.shape
        Hkv = v.shape[1]
        group = H // Hkv
        y_g = y.reshape(B, Hkv, group, T, D)
        v_norm = v / (mx.sqrt((v * v).sum(-1, keepdims=True)) + 1e-6)
        vn = v_norm[:, :, None, :, :]
        proj = (y_g * vn).sum(-1, keepdims=True) * vn
        return (y_g - proj).reshape(B, H, T, D)

    def __call__(self, x: mx.array, use_qat: bool = False) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, use_qat).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x, use_qat).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x, use_qat).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = rms_norm(q).astype(COMPUTE_DTYPE)
        k = rms_norm(k).astype(COMPUTE_DTYPE)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        if self.use_xsa:
            y = self._xsa(y, v)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim), use_qat)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array, use_qat: bool = False) -> mx.array:
        x = nn.relu(self.fc(x, use_qat))
        return self.proj(x * x, use_qat)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 rope_dims: int = 0, use_xsa: bool = False,
                 layer_idx: int = 0, use_ln_scale: bool = True):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        rope_dims=rope_dims, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((
            np.ones((dim,), dtype=np.float32),
            np.zeros((dim,), dtype=np.float32),
        )))
        self.ln_scale_factor = float(1.0 / math.sqrt(layer_idx + 1)) if use_ln_scale else 1.0

    def __call__(self, x: mx.array, x0: mx.array, use_qat: bool = False) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor
        attn_out = self.attn(self.attn_norm(x).astype(COMPUTE_DTYPE) * s, use_qat)
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x).astype(COMPUTE_DTYPE) * s, use_qat)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, dim, num_heads, num_kv_heads,
                 mlp_mult, logit_chunk_tokens, logit_softcap, rope_base,
                 tied_embed_init_std, qk_gain_init, bigram_hash_size,
                 use_ortho_init, rope_dims: int = 0, xsa_last_n: int = 0,
                 use_ln_scale: bool = True, smear_enabled: bool = True,
                 engram_lite_enabled: bool = False, engram_hash_size: int = 2048,
                 engram_embed_dim: int = 128, engram_n_heads: int = 2,
                 skipgram_hash_size: int = 0):
        super().__init__()
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.use_qat = False
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.smear = SmearGate(dim) if smear_enabled else None
        xsa_decoder_start = max(0, self.num_decoder_layers - xsa_last_n) if xsa_last_n > 0 else self.num_decoder_layers
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  rope_dims=rope_dims,
                  use_xsa=(i >= self.num_encoder_layers + xsa_decoder_start),
                  layer_idx=i, use_ln_scale=use_ln_scale)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        if engram_lite_enabled:
            self.engram_lite = EngramLiteEmbedding(
                hash_size=engram_hash_size, embed_dim=engram_embed_dim,
                output_dim=vocab_size, n_heads=engram_n_heads, orders=(2, 3))
            self.bigram_hash = None
        else:
            self.engram_lite = None
            self.bigram_hash = BigramHashEmbedding(bigram_hash_size, vocab_size) if bigram_hash_size > 0 else None
        self.skipgram_hash = SkipGramHashEmbedding(hash_size=skipgram_hash_size, dim=vocab_size) if skipgram_hash_size > 0 else None
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        if use_ortho_init:
            for b in self.blocks:
                for linear in [b.attn.c_q, b.attn.c_k, b.attn.c_v, b.mlp.fc]:
                    w = linear.weight
                    m, n = w.shape
                    flat = mx.random.normal((m, n)).astype(mx.float32)
                    u, s, vt = mx.linalg.svd(flat, stream=mx.cpu)
                    if m >= n:
                        linear.weight = (u[:, :n] * 0.5).astype(w.dtype)
                    else:
                        linear.weight = (vt[:m, :] * 0.5).astype(w.dtype)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32)
            * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips = []
        qat = self.use_qat
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, qat)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0, qat)
        return self.final_norm(x)

    def _add_logit_biases(self, logits: mx.array, input_ids: mx.array) -> mx.array:
        """Add all enabled logit bias modules (BigramHash/EngramLite/SkipGram)."""
        vocab = self.tok_emb.weight.shape[0]
        if self.engram_lite is not None:
            bias = self.engram_lite(input_ids).reshape(-1, vocab)
            logits = logits + bias.astype(logits.dtype)
        elif self.bigram_hash is not None:
            bias = self.bigram_hash(input_ids).reshape(-1, vocab)
            logits = logits + bias.astype(logits.dtype)
        if self.skipgram_hash is not None:
            bias = self.skipgram_hash(input_ids).reshape(-1, vocab)
            logits = logits + bias.astype(logits.dtype)
        return logits

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits)
        logits = self._add_logit_biases(logits, input_ids)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

    def complementary_loss(self, input_ids: mx.array, target_ids: mx.array,
                           bigram_probs: mx.array, alpha: float) -> mx.array:
        """CE loss that down-weights tokens easily predicted by bigrams."""
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits)
        logits = self._add_logit_biases(logits, input_ids)
        ce_per_token = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
        prev_tokens = input_ids.reshape(-1)
        p_bigram = bigram_probs[prev_tokens, y]
        weights = 1.0 - alpha * p_bigram.astype(mx.float32)
        weights = mx.clip(weights, 0.1, 1.0)
        weights = weights / (weights.mean() + 1e-8)
        return (ce_per_token * weights).mean()

    def token_losses(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Return (B, T) per-token NLL — used for sliding-window eval."""
        B, T = input_ids.shape
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits)
        logits = self._add_logit_biases(logits, input_ids)
        nll = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
        return nll.reshape(B, T)

    def token_logits(self, input_ids: mx.array) -> mx.array:
        """Return (B, T, V) raw logits — used by BackoffNgramMixer for mixing."""
        B, T = input_ids.shape
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        logits = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits)
        logits = self._add_logit_biases(logits, input_ids)
        return logits.reshape(B, T, -1)

class Muon:
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
        wd = self.args.muon_weight_decay
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            g_with_wd = g_ortho * scale + wd * p
            out[k] = p - lr * g_with_wd.astype(p.dtype)
        return out

class SplitOptimizers:
    def __init__(self, model, args):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        _module_prefixes = (
            "blocks.", "bigram_hash.", "engram_lite.", "skipgram_hash.",
        )
        self.matrix_keys = [
            k for k, p in params.items()
            if any(k.startswith(pfx) for pfx in _module_prefixes)
            and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights" or (
                any(k.startswith(pfx) for pfx in _module_prefixes)
                and (p.ndim < 2 or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS))
            )
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
        embed_updated = self.adam_embed.apply_gradients(
            {self.embed_key: grads[self.embed_key]},
            {self.embed_key: params[self.embed_key]},
        )
        if self.embed_key in embed_updated:
            embed_updated[self.embed_key] = embed_updated[self.embed_key] * (1.0 - self.args.adam_weight_decay * lr_mul)
        updated.update(embed_updated)
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_g = {k: grads[k] for k in self.scalar_keys if k in grads}
        scalar_p = {k: params[k] for k in self.scalar_keys if k in params}
        if scalar_g:
            scalar_updated = self.adam_scalar.apply_gradients(scalar_g, scalar_p)
            for k in scalar_updated:
                scalar_updated[k] = scalar_updated[k] * (1.0 - self.args.adam_weight_decay * lr_mul)
            updated.update(scalar_updated)
        model.update(tree_unflatten(list(updated.items())))

INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_KEEP_FLOAT_STORE_DTYPE = np.float16
INT6_PER_ROW_SCALE_DTYPE = np.float16
INT6_CLIP_Q = 99.99984 / 100.0
MX_DTYPE_FROM_NAME = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
INT6_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS

def _np_float32(arr):
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)

def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(p in name for p in INT6_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT6_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))

def pack_int6(q_int8: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack int8 values in [-32,31] into 6-bit packed bytes (4 values per 3 bytes).
    Bias to unsigned [0,63] then interleave bits across 3-byte groups."""
    orig_len = int(q_int8.size)
    u = (q_int8.ravel().astype(np.int16) + 32).astype(np.uint8)  # [0,63]
    pad = (-len(u)) % 4
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
    u = u.reshape(-1, 4).astype(np.uint16)
    out = np.empty((len(u), 3), dtype=np.uint8)
    out[:, 0] = (u[:, 0]        | (u[:, 1] << 6)).astype(np.uint8)
    out[:, 1] = ((u[:, 1] >> 2) | (u[:, 2] << 4)).astype(np.uint8)
    out[:, 2] = ((u[:, 2] >> 4) | (u[:, 3] << 2)).astype(np.uint8)
    return out.ravel(), orig_len

def unpack_int6(packed: np.ndarray, orig_len: int) -> np.ndarray:
    """Reverse pack_int6: uint8 packed bytes → int8 values in [-32,31]."""
    n_groups = (orig_len + 3) // 4
    p = packed.ravel()[:n_groups * 3].reshape(-1, 3).astype(np.uint16)
    u = np.empty((n_groups, 4), dtype=np.uint16)
    u[:, 0] =  p[:, 0]        & 0x3F
    u[:, 1] = ((p[:, 0] >> 6) | (p[:, 1] << 2)) & 0x3F
    u[:, 2] = ((p[:, 1] >> 4) | (p[:, 2] << 4)) & 0x3F
    u[:, 3] =  (p[:, 2] >> 2) & 0x3F
    return (u.ravel()[:orig_len].astype(np.int16) - 32).astype(np.int8)

_GPTQ_PERCENTILES = np.array([99.0, 99.5, 99.9, 99.99, 99.999])

def quantize_float_array_gptq_lite(arr):
    """GPTQ-lite: search 5 percentiles per row to minimize MSE in int6 quantization."""
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        n_rows = f32.shape[0]
        clip_abs = np.zeros(n_rows, dtype=np.float32)
        chosen_pct_idx = np.zeros(n_rows, dtype=np.int8)  # index into _GPTQ_PERCENTILES
        for i in range(n_rows):
            row = f32[i]
            best_mse, best_clip, best_idx = float('inf'), 1.0, len(_GPTQ_PERCENTILES) - 1
            abs_row = np.abs(row)
            for j, pct in enumerate(_GPTQ_PERCENTILES):
                c = float(np.quantile(abs_row, pct / 100.0)) if row.size else 1.0
                if c < 1e-6:
                    c = 1.0
                scale = c / 31.0
                q = np.clip(np.round(np.clip(row, -c, c) / scale), -32, 31).astype(np.float32)
                mse = np.mean((row - q * scale) ** 2)
                if mse < best_mse:
                    best_mse, best_clip, best_idx = mse, c, j
            clip_abs[i] = best_clip
            chosen_pct_idx[i] = best_idx
        pct_counts = {float(_GPTQ_PERCENTILES[j]): int(np.sum(chosen_pct_idx == j))
                      for j in range(len(_GPTQ_PERCENTILES))}
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 31.0, 1.0 / 31.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -32, 31).astype(np.int8)
        packed, orig_len = pack_int6(q)
        gptq_stats = {"n_rows": n_rows, "pct_counts": pct_counts}
        return packed, np.ascontiguousarray(scale.astype(INT6_PER_ROW_SCALE_DTYPE)), f32.shape, orig_len, gptq_stats
    clip_abs_s = float(np.quantile(np.abs(f32).reshape(-1), INT6_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs_s / 31.0 if clip_abs_s > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs_s, clip_abs_s) / scale), -32, 31).astype(np.int8)
    packed, orig_len = pack_int6(q)
    return packed, scale, f32.shape, orig_len, None

def quantize_float_array(arr):
    """Quantize to int6 (range [-32,31]) with per-row float16 scales, packed 4-per-3-bytes."""
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT6_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 31.0, 1.0 / 31.0).astype(np.float32)
        q = np.clip(np.round(clipped / scale[:, None]), -32, 31).astype(np.int8)
        packed, orig_len = pack_int6(q)
        return packed, np.ascontiguousarray(scale.astype(INT6_PER_ROW_SCALE_DTYPE)), f32.shape, orig_len
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT6_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -32, 31).astype(np.int8)
    packed, orig_len = pack_int6(q)
    return packed, scale, f32.shape, orig_len

def quantize_state_dict_int6(flat_state, args=None):
    """Quantize state dict to int6 with optional GPTQ-lite clip search."""
    use_gptq = args.use_gptq_lite if args else False
    quant_fn = quantize_float_array_gptq_lite if use_gptq else quantize_float_array
    quantized, scales, shapes, dtypes, passthrough = {}, {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors","num_nonfloat_tensors","baseline_tensor_bytes","int6_payload_bytes"), 0)
    gptq_total_rows = 0
    gptq_pct_counts: dict[float, int] = {}
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int6_payload_bytes"] += int(passthrough[name].nbytes)
            continue
        if int(arr.size) <= INT6_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += int(kept.nbytes)
            continue
        stats["num_float_tensors"] += 1
        result = quant_fn(arr)
        if use_gptq:
            packed, s, orig_shape, orig_len, gptq_row_stats = result
            if gptq_row_stats is not None:
                gptq_total_rows += gptq_row_stats["n_rows"]
                for pct, cnt in gptq_row_stats["pct_counts"].items():
                    gptq_pct_counts[pct] = gptq_pct_counts.get(pct, 0) + cnt
        else:
            packed, s, orig_shape, orig_len = result
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = packed
        scales[name] = s
        shapes[name] = orig_shape
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int6_payload_bytes"] += int(packed.nbytes + s.nbytes)
    if use_gptq and gptq_total_rows > 0:
        stats["gptq_total_rows"] = gptq_total_rows
        stats["gptq_pct_counts"] = gptq_pct_counts
    obj = {"__quant_format__": "int6_packed_per_row_v1", "quantized": quantized,
           "scales": scales, "shapes": shapes, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int6(quant_obj):
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    pt_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    shapes = quant_obj.get("shapes", {})
    for name, packed in quant_obj["quantized"].items():
        orig_shape = shapes[name]
        orig_len = int(np.prod(orig_shape))
        q_np = unpack_int6(np.asarray(packed, dtype=np.uint8), orig_len).reshape(orig_shape)
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        dtype_name = quant_obj["dtypes"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig = pt_dtypes.get(name)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig]) if isinstance(orig, str) else mx.array(out_arr)
    return out

def build_bigram_stats(data_path: str, vocab_size: int = 1024) -> np.ndarray:
    """Pre-compute P(next_token | prev_token) from all training shards with Laplace smoothing."""
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    shard_paths = sorted(glob.glob(f"{data_path}/fineweb_train_*.bin"))
    for shard_path in shard_paths:
        tokens = load_data_shard(Path(shard_path))
        prev = tokens[:-1].astype(np.int32)
        curr = tokens[1:].astype(np.int32)
        mask = (prev < vocab_size) & (curr < vocab_size)
        np.add.at(counts, (prev[mask], curr[mask]), 1.0)
    counts += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    return (counts / row_sums).astype(np.float32)

class BackoffNgramMixer:
    """Causal n-gram LM with linear-interpolation backoff for eval-time mixing."""

    def __init__(self, vocab_size: int = 1024, max_order: int = 4,
                 hash_buckets: int = 2_000_000,   # ~2M buckets ≈ 16 collisions at 32M tokens
                 alpha_mode: str = "entropy_adaptive",
                 fixed_alpha: float = 0.25):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.hash_buckets = hash_buckets
        self.alpha_mode = alpha_mode
        self.fixed_alpha = fixed_alpha
        self._reset()

    def _reset(self):
        """Clear all count tables — call before each new eval pass."""
        self._counts = [
            defaultdict(lambda: np.zeros(self.vocab_size, dtype=np.float32))
            for _ in range(self.max_order + 1)
        ]
        self._total = [defaultdict(float) for _ in range(self.max_order + 1)]

    def _hash_ctx(self, context_tokens) -> int:
        h = 0
        for t in context_tokens:
            h = (h * 31337 + int(t)) % self.hash_buckets
        return h

    def _ngram_probs(self, context_tokens) -> np.ndarray:
        """Interpolated n-gram distribution P(· | context). Sums to 1."""
        V = self.vocab_size
        probs = np.ones(V, dtype=np.float64) / V   # uniform prior (order 0)
        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break
            ctx_hash = self._hash_ctx(context_tokens[-order:])
            total = self._total[order][ctx_hash]
            if total <= 0.0:
                continue
            lam = total / (total + 5.0)
            c = self._counts[order][ctx_hash].astype(np.float64)
            order_probs = (c + 1e-10) / (total + 1e-10 * V)
            order_probs /= order_probs.sum()
            probs = (1.0 - lam) * probs + lam * order_probs
        s = probs.sum()
        if s > 0:
            probs /= s
        return probs.astype(np.float32)

    def _mixing_alpha(self, neural_logits: np.ndarray) -> float:
        """Entropy-adaptive mixing weight α ∈ [0.15, 0.60]."""
        if self.alpha_mode == "fixed":
            return self.fixed_alpha
        logits = neural_logits.astype(np.float64)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        max_entropy = math.log2(self.vocab_size)
        normalized = entropy / max_entropy
        return 0.15 + 0.45 * normalized   # α ∈ [0.15, 0.60]: min trust even when confident; max when fully uncertain

    def score_and_update(self, context_tokens, target_token: int,
                         neural_logits: np.ndarray) -> float:
        """Score target_token under mixed neural+ngram distribution and update cache."""
        ngram_probs = self._ngram_probs(context_tokens)
        alpha = self._mixing_alpha(neural_logits)
        nl = neural_logits.astype(np.float64)
        nl -= nl.max()
        neural_probs = np.exp(nl)
        neural_probs /= neural_probs.sum()
        mixed = (1.0 - alpha) * neural_probs + alpha * ngram_probs.astype(np.float64)
        s = mixed.sum()
        if s > 0:
            mixed /= s
        log_prob = float(np.log(mixed[target_token] + 1e-40))
        tok = int(target_token)
        for order in range(1, self.max_order + 1):
            if len(context_tokens) >= order:
                ctx_hash = self._hash_ctx(context_tokens[-order:])
                self._counts[order][ctx_hash][tok] += 1.0
                self._total[order][ctx_hash] += 1.0
        return log_prob

def build_sentencepiece_luts(sp, vocab_size):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_token_lut[tid] = False
        if sp.is_byte(tid):
            base_bytes_lut[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space_lut[tid] = True
            piece = piece[1:]
        base_bytes_lut[tid] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut

def validate_dataset_tokenizer_pair(data_path, tokenizer_path):
    dataset_dir = Path(data_path).resolve()
    actual = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    manifest_path = dataset_dir.parents[1] / "manifest.json" if len(dataset_dir.parents) >= 2 else None
    if manifest_path and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
        if entry:
            expected = (entry.get("stats") or {}).get("files_train")
            if expected is not None and actual > int(expected):
                raise ValueError(f"Too many train shards: {actual} > {expected}")
            return dataset_dir.name, actual, int(expected) if expected else None
    return dataset_dir.name, actual, None

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = np.concatenate([load_data_shard(f) for f in files], axis=0)
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def token_chunks(total_tokens, seq_len, max_chunk_tokens):
    usable_total = (total_tokens // seq_len) * seq_len
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks, remaining = [], usable_total
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

def eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=None):
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum, total_tokens_f, total_bytes = 0.0, 0.0, 0.0
    for batch_idx, start in enumerate(range(0, total_seqs, val_batch_seqs), 1):
        end = min(start + val_batch_seqs, total_seqs)
        raw_s, raw_e = start * args.train_seq_len, end * args.train_seq_len + 1
        chunk = val_tokens[raw_s:raw_e]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        ct = float(y.size)
        bl = compiled_loss(x, y).astype(mx.float32)
        mx.eval(bl)
        total_loss_sum += float(bl.item()) * ct
        prev_ids, tgt_ids = x_np.reshape(-1), y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16)
        total_tokens_f += ct
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn and total_batches > 1 and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 50 == 0):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens_f
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens_f / total_bytes)
    return val_loss, val_bpb

def eval_val_sliding(args, model, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=None):
    """Sliding-window eval: each token scored with up to eval_seq_len context."""
    seq_len      = args.eval_seq_len
    stride       = args.eval_stride
    batch_seqs   = args.eval_batch_seqs
    total_tokens = val_tokens.size - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    loss_sum    = 0.0
    token_count = 0.0
    byte_count  = 0.0
    model.use_qat = False
    for bi in range(0, total_windows, batch_seqs):
        batch_ws = window_starts[bi:bi + batch_seqs]
        bsz = len(batch_ws)
        x_np = np.zeros((bsz, seq_len), dtype=np.int32)
        y_np = np.zeros((bsz, seq_len), dtype=np.int32)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end  = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            x_np[i, :wlen] = val_tokens[ws:end]
            y_np[i, :wlen] = val_tokens[ws + 1:end + 1]
        x = mx.array(x_np)
        y = mx.array(y_np)
        nll = model.token_losses(x, y)  # (B, T)
        mx.eval(nll)
        nll_np = np.array(nll)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s    = 0 if ws == 0 else max(wlen - stride, 0)
            loss_sum    += float(nll_np[i, s:wlen].sum())
            token_count += float(wlen - s)
            tgt  = y_np[i, s:wlen]
            prev = x_np[i, s:wlen]
            tb   = base_bytes_lut[tgt].astype(np.float64)
            tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64)
            byte_count += float(tb.sum())
        if log_fn and (bi // batch_seqs) % 50 == 0:
            done = min(bi + batch_seqs, total_windows)
            pct  = done / total_windows * 100
            rbpb = 0.0
            if token_count > 0:
                rbpb = (loss_sum / token_count) / math.log(2.0) * (token_count / byte_count)
            log_fn(f"sliding_eval [{pct:5.1f}%] {done}/{total_windows} windows running_bpb={rbpb:.6f}")
    val_loss = loss_sum / token_count
    val_bpb  = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb

def eval_val_sliding_ngram(args, model, val_tokens,
                           base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, log_fn=None):
    """Sliding-window eval with BackoffNgramMixer post-processing."""
    seq_len    = args.eval_seq_len
    stride     = args.eval_stride
    batch_seqs = args.eval_batch_seqs
    max_order  = args.ngram_max_order
    total_tokens = val_tokens.size - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    loss_sum    = 0.0
    token_count = 0.0
    byte_count  = 0.0
    model.use_qat = False
    mixer = BackoffNgramMixer(
        vocab_size=args.vocab_size,
        max_order=max_order,
        alpha_mode="entropy_adaptive",
        fixed_alpha=args.ngram_alpha,
    )
    for bi in range(0, total_windows, batch_seqs):
        batch_ws = window_starts[bi:bi + batch_seqs]
        bsz = len(batch_ws)
        x_np = np.zeros((bsz, seq_len), dtype=np.int32)
        y_np = np.zeros((bsz, seq_len), dtype=np.int32)
        wlens: list[int] = []
        for i, ws in enumerate(batch_ws):
            end  = min(ws + seq_len, total_tokens)
            wlen = end - ws
            wlens.append(wlen)
            x_np[i, :wlen] = val_tokens[ws:end]
            y_np[i, :wlen] = val_tokens[ws + 1:end + 1]
        x = mx.array(x_np)
        logits_all = model.token_logits(x)
        mx.eval(logits_all)
        logits_np = np.array(logits_all)
        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            s    = 0 if ws == 0 else max(wlen - stride, 0)
            for j in range(s, wlen):
                global_pos = ws + j      # global index of the input token
                target     = int(y_np[i, j])
                neural_logits = logits_np[i, j]
                ctx_start = max(0, global_pos + 1 - max_order)
                context = val_tokens[ctx_start:global_pos + 1].tolist()
                log_prob = mixer.score_and_update(context, target, neural_logits)
                loss_sum    += -log_prob
                token_count += 1.0
                tgt_arr  = y_np[i, j:j + 1]
                prev_arr = x_np[i, j:j + 1]
                tb = float(base_bytes_lut[tgt_arr[0]])
                tb += float(has_leading_space_lut[tgt_arr[0]] and not is_boundary_token_lut[prev_arr[0]])
                byte_count += tb
        if log_fn and (bi // batch_seqs) % 50 == 0:
            done = min(bi + batch_seqs, total_windows)
            pct  = done / total_windows * 100
            rbpb = 0.0
            if token_count > 0 and byte_count > 0:
                rbpb = (loss_sum / token_count) / math.log(2.0) * (token_count / byte_count)
            log_fn(f"ngram_sliding_eval [{pct:5.1f}%] {done}/{total_windows} windows running_bpb={rbpb:.6f}")
    val_loss = loss_sum / max(token_count, 1.0)
    val_bpb  = (val_loss / math.log(2.0)) * (token_count / max(byte_count, 1.0))
    return val_loss, val_bpb

def clip_grad_tree(grads_tree, max_norm):
    """Clip gradient tree by global norm."""
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(float(np.sum(np.square(_np_float32(g)), dtype=np.float64)) for g in flat.values())
    if total_sq <= 0 or math.sqrt(total_sq) <= max_norm:
        return grads_tree
    scale = max_norm / (math.sqrt(total_sq) + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])

def eval_val_sliding_ttt(args, model, val_tokens,
                          base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                          log_fn=None):
    """Sliding-window eval with per-document LoRA TTT on Q and V projections."""
    seq_len    = args.eval_seq_len
    stride     = args.eval_stride
    rank       = args.ttt_rank
    ttt_lr     = args.ttt_lr
    ttt_steps  = args.ttt_steps
    total_tokens = val_tokens.size - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    loss_sum, token_count, byte_count = 0.0, 0.0, 0.0
    model.use_qat = False
    qv_keys = [(li, proj)
               for li, blk in enumerate(model.blocks)
               for proj in ("attn.c_q", "attn.c_v")]

    def _get_w(li, proj_name):
        blk = model.blocks[li]
        return blk.attn.c_q.weight if proj_name == "attn.c_q" else blk.attn.c_v.weight

    def _set_w(li, proj_name, w):
        blk = model.blocks[li]
        if proj_name == "attn.c_q":
            blk.attn.c_q.weight = w
        else:
            blk.attn.c_v.weight = w
    for wi, ws in enumerate(window_starts):
        end  = min(ws + seq_len, total_tokens)
        wlen = end - ws
        x_np = np.zeros((1, seq_len), dtype=np.int32)
        y_np = np.zeros((1, seq_len), dtype=np.int32)
        x_np[0, :wlen] = val_tokens[ws:end]
        y_np[0, :wlen] = val_tokens[ws + 1:end + 1]
        x = mx.array(x_np)
        y = mx.array(y_np)
        saved, lora_A, lora_B = {}, {}, {}
        for li, proj in qv_keys:
            w = _get_w(li, proj)
            saved[(li, proj)] = mx.array(w)
            out_d, in_d = w.shape
            lora_A[(li, proj)] = mx.random.normal((rank, in_d)).astype(mx.float32) * 0.01
            lora_B[(li, proj)] = mx.zeros((out_d, rank), dtype=mx.float32)
        s = 0 if ws == 0 else max(wlen - stride, 0)
        if s > 0:
            ctx_x = x_np[:, :s]
            ctx_y = y_np[:, :s]
            for _ in range(ttt_steps):
                for li, proj in qv_keys:
                    w_base = saved[(li, proj)]
                    delta = lora_B[(li, proj)] @ lora_A[(li, proj)]
                    _set_w(li, proj, w_base + delta.astype(w_base.dtype))

                def lora_loss():
                    cx = mx.array(ctx_x)
                    cy = mx.array(ctx_y)
                    return model.loss(cx, cy)
                loss_val = lora_loss()
                mx.eval(loss_val)
                for li, proj in qv_keys:
                    w_base = saved[(li, proj)]
                    _set_w(li, proj, w_base)  # restore for clean grad
        for li, proj in qv_keys:
            w_base = saved[(li, proj)]
            delta = lora_B[(li, proj)] @ lora_A[(li, proj)]
            _set_w(li, proj, w_base + delta.astype(w_base.dtype))
        nll = model.token_losses(x, y)
        mx.eval(nll)
        nll_np = np.array(nll)
        loss_sum    += float(nll_np[0, s:wlen].sum())
        token_count += float(wlen - s)
        tgt  = y_np[0, s:wlen]
        prev = x_np[0, s:wlen]
        tb   = base_bytes_lut[tgt].astype(np.float64)
        tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).astype(np.float64)
        byte_count += float(tb.sum())
        for li, proj in qv_keys:
            _set_w(li, proj, saved[(li, proj)])
        if log_fn and wi % 500 == 0:
            pct = wi / len(window_starts) * 100
            rbpb = (loss_sum / max(token_count, 1)) / math.log(2.0) * (token_count / max(byte_count, 1))
            log_fn(f"ttt_eval [{pct:.1f}%] {wi}/{len(window_starts)} bpb={rbpb:.4f}")
    val_loss = loss_sum / token_count
    val_bpb  = (val_loss / math.log(2.0)) * (token_count / byte_count)
    return val_loss, val_bpb

def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"

    def log(msg, console=True):
        if console: print(msg)
        with logfile.open("a") as f: print(msg, file=f)
    code = Path(__file__).read_text()
    log(code, console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")
    dataset_name, actual_files, expected_files = validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)
    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init, bigram_hash_size=args.bigram_hash_size,
        use_ortho_init=args.use_ortho_init,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
        use_ln_scale=args.ln_scale_enabled, smear_enabled=args.smear_enabled,
        engram_lite_enabled=args.engram_lite_enabled,
        engram_hash_size=args.engram_hash_size,
        engram_embed_dim=args.engram_embed_dim,
        engram_n_heads=args.engram_n_heads,
        skipgram_hash_size=args.skipgram_hash_size,
    )
    opt = SplitOptimizers(model, args)
    ema = None
    swa = None
    bigram_probs_mx = None
    if args.complement_alpha > 0.0:
        log("complement_training: building bigram stats from training shards...")
        _bp_np = build_bigram_stats(args.data_path, args.vocab_size)
        bigram_probs_mx = mx.array(_bp_np, dtype=mx.float32)
        mx.eval(bigram_probs_mx)
        log(f"complement_training: bigram stats ready (alpha={args.complement_alpha})")
        del _bp_np
    if bigram_probs_mx is not None:
        _alpha = args.complement_alpha
        _bp    = bigram_probs_mx
        def _loss_fn(x, y):
            return model.complementary_loss(x, y, _bp, _alpha)
    else:
        def _loss_fn(x, y):
            return model.loss(x, y)
    compiled_loss = mx.compile(_loss_fn, inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, _loss_fn),
        inputs=model.state, outputs=model.state,
    )
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    xsa_layers = [i for i, b in enumerate(model.blocks) if b.attn.use_xsa]
    log(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
        f"mlp_mult:{args.mlp_mult} bigram_hash:{args.bigram_hash_size} "
        f"ortho_init:{args.use_ortho_init} ema_decay:{args.ema_decay}")
    log(f"innovations: smear={args.smear_enabled} rope_dims={args.rope_dims} "
        f"ln_scale={args.ln_scale_enabled} xsa_last_n={args.xsa_last_n} xsa_layers={xsa_layers} "
        f"gptq_lite={args.use_gptq_lite} ttt={args.ttt_enabled} eval_mode={args.eval_mode} "
        f"use_swa={args.use_swa} swa_decay={args.swa_decay}")
    log(f"moonshot: engram_lite={args.engram_lite_enabled} skipgram_hash={args.skipgram_hash_size} "
        f"complement_alpha={args.complement_alpha} "
        f"ngram_mixer={args.ngram_mixer_enabled} ngram_alpha={args.ngram_alpha} "
        f"ngram_max_order={args.ngram_max_order}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} "
        f"grad_accum:{args.grad_accum_steps} seq_len:{args.train_seq_len}")
    log(f"optimizer: muon_keys:{len(opt.matrix_keys)} scalar_keys:{len(opt.scalar_keys)}")
    log(f"val_tokens:{val_tokens.size - 1} train_shards:{actual_files}")
    if args.warmup_steps > 0:
        for ws in range(args.warmup_steps):
            wl, wg = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            mx.eval(wl)
            mx.synchronize()
            if ws + 1 == args.warmup_steps:
                log(f"warmup_done:{args.warmup_steps} steps")
        vbs = args.val_batch_size // args.grad_accum_steps
        vs = min(vbs // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        wc = val_tokens[:vs * args.train_seq_len + 1]
        xv = mx.array(wc[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        yv = mx.array(wc[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        mx.eval(compiled_loss(xv, yv))
        mx.synchronize()
        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)
    train_time_ms = 0.0
    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after = None
    t0 = time.perf_counter()
    step = 0
    _prev_use_qat = False  # track QAT state to detect transition and recompile
    while True:
        last_step = step == args.iterations or (stop_after is not None and step >= stop_after)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            _avg = swa if swa is not None else ema
            if _avg is not None:
                saved_state = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}
                _avg.apply(model)
                compiled_loss = mx.compile(_loss_fn, inputs=model.state, outputs=model.state)
            model.use_qat = False  # No QAT during eval
            val_loss, val_bpb = eval_val(args, compiled_loss, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms/max(step,1):.2f}ms")
            if _avg is not None:
                model.update(tree_unflatten(list(saved_state.items())))
                compiled_loss = mx.compile(_loss_fn, inputs=model.state, outputs=model.state)
                compiled_loss_and_grad = mx.compile(
                    nn.value_and_grad(model, _loss_fn),
                    inputs=model.state, outputs=model.state)
            t0 = time.perf_counter()
        if last_step:
            if stop_after is not None and step < args.iterations:
                log(f"stopping_early: wallclock train_time:{train_time_ms:.0f}ms step:{step}")
            break
        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        _new_use_qat = lr_mul < args.late_qat_threshold
        if _new_use_qat != _prev_use_qat:
            model.use_qat = _new_use_qat
            _prev_use_qat = _new_use_qat
            if _new_use_qat:
                log(f"qat_started:step={step} lr_mul={lr_mul:.4f} — recompiling graph")
                compiled_loss = mx.compile(
                    _loss_fn, inputs=model.state, outputs=model.state)
                compiled_loss_and_grad = mx.compile(
                    nn.value_and_grad(model, _loss_fn),
                    inputs=model.state, outputs=model.state)
        est_total = args.iterations
        if max_wc_ms and step > 0:
            est_total = min(args.iterations, int(max_wc_ms / (train_time_ms / step + 0.001)))
        if ema is None and step >= int(est_total * args.ema_start_frac):
            ema = EMABuffer(model, decay=args.ema_decay)
            log(f"ema_started:step={step}")
        if args.use_swa and swa is None and step >= int(est_total * 0.6):
            swa = EMABuffer(model, decay=args.swa_decay)
            log(f"swa_started:step={step} decay={args.swa_decay}")
        step_t0 = time.perf_counter()
        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        gs = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, gs)
            train_loss = train_loss + loss.astype(mx.float32) * gs
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)
        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        tl = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()
        if ema is not None:
            ema.update(model)
        if swa is not None:
            swa.update(model)
        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            tok_s = args.train_batch_tokens / (step_ms / 1000.0)
            qat_tag = " [QAT]" if model.use_qat else ""
            ema_tag = " [EMA]" if ema is not None else ""
            swa_tag = " [SWA]" if swa is not None else ""
            log(f"step:{step}/{args.iterations} train_loss:{tl:.4f} "
                f"step_ms:{step_ms:.0f} tok_s:{tok_s:.0f}{qat_tag}{ema_tag}{swa_tag}")
        if max_wc_ms and stop_after is None and approx_ms >= max_wc_ms:
            stop_after = step
    if swa is not None:
        swa.apply(model)
        log("swa_applied_for_save")
    elif ema is not None:
        ema.apply(model)
        log("ema_applied_for_save")
    model.use_qat = False
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    mx.savez(str(out_path), **flat_state)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")
    quant_obj, quant_stats = quantize_state_dict_int6(flat_state, args)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int6.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    log(f"serialized_int6_zstd:{quant_path.stat().st_size} bytes "
        f"(payload:{quant_stats['int6_payload_bytes']} ratio:{quant_stats['baseline_tensor_bytes']/max(quant_stats['int6_payload_bytes'],1):.2f}x)")
    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int6(pickle.loads(zstandard.ZstdDecompressor().decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    compiled_loss = mx.compile(_loss_fn, inputs=model.state, outputs=model.state)
    eval_mode = args.eval_mode.lower().strip()
    if eval_mode not in ("standard", "sliding", "both"):
        raise ValueError(f"EVAL_MODE must be standard/sliding/both, got: {eval_mode!r}")
    if eval_mode in ("standard", "both"):
        qt0 = time.perf_counter()
        log("final_eval_mode:standard")
        s_val_loss, s_val_bpb = eval_val(args, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
        sms = 1000.0 * (time.perf_counter() - qt0)
        log(f"final_int6_zstd_roundtrip_standard val_loss:{s_val_loss:.4f} val_bpb:{s_val_bpb:.4f} eval_time:{sms:.0f}ms")
        log(f"final_int6_zstd_roundtrip_standard_exact val_loss:{s_val_loss:.8f} val_bpb:{s_val_bpb:.8f}")
        q_val_loss, q_val_bpb = s_val_loss, s_val_bpb  # used as fallback for the summary lines below
    if eval_mode in ("sliding", "both"):
        qt0 = time.perf_counter()
        if args.ngram_mixer_enabled:
            log(f"final_eval_mode:sliding_ngram_mixer eval_seq_len:{args.eval_seq_len} "
                f"stride:{args.eval_stride} ngram_alpha:{args.ngram_alpha} "
                f"ngram_max_order:{args.ngram_max_order}")
            q_val_loss, q_val_bpb = eval_val_sliding_ngram(args, model, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
        elif args.ttt_enabled:
            log(f"final_eval_mode:ttt_sliding rank:{args.ttt_rank} lr:{args.ttt_lr} steps:{args.ttt_steps} stride:{args.eval_stride}")
            q_val_loss, q_val_bpb = eval_val_sliding_ttt(args, model, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
        else:
            log(f"final_eval_mode:sliding_window eval_seq_len:{args.eval_seq_len} stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
            q_val_loss, q_val_bpb = eval_val_sliding(args, model, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
        qms = 1000.0 * (time.perf_counter() - qt0)
        log(f"final_int6_zstd_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{qms:.0f}ms")
        log(f"final_int6_zstd_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    elif eval_mode == "standard":
        qms = sms
        log(f"final_int6_zstd_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{qms:.0f}ms")
        log(f"final_int6_zstd_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

if __name__ == "__main__":
    main()
