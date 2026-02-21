#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing MLX. Install with: python3 -m pip install mlx") from exc


# FineWeb shard format used by train_gpt.py
_SHARD_MAGIC = 20240520
_SHARD_VERSION = 1
_SHARD_HEADER_INTS = 256


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


TOKENIZER_KIND = os.environ.get("TOKENIZER_KIND", "sp").lower()


def _maybe_find_manifest_for_data_path(data_path: str) -> Path | None:
    p = Path(data_path).resolve()
    candidates = [p] + list(p.parents[:6])
    for cand in candidates:
        manifest = cand / "manifest.json"
        if manifest.is_file():
            return manifest
    return None


def _default_sentencepiece_tokenizer_path() -> str:
    # Try to infer from a matched dataset manifest first (works for DATA_PATH=.../datasets/fineweb10B_sp2048).
    data_path = _env_str("DATA_PATH", "./data/matched_10B/datasets/fineweb10B_sp2048")
    vocab_size = _env_int("VOCAB_SIZE", 2048)
    manifest_path = _maybe_find_manifest_for_data_path(data_path)
    if manifest_path is not None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for tok in manifest.get("tokenizers", []):
                if not isinstance(tok, dict):
                    continue
                if int(tok.get("vocab_size", -1)) != vocab_size:
                    continue
                kind = str(tok.get("kind", "")).lower()
                if "sentencepiece" not in kind and tok.get("name", "").startswith("sp_") is False:
                    continue
                model_path = tok.get("model_path")
                if not isinstance(model_path, str):
                    continue
                # manifest paths may be repo-relative ("data/...") or basename-only in other setups.
                direct = Path(model_path)
                if direct.is_file():
                    return str(direct)
                sibling = manifest_path.parent / "tokenizers" / Path(model_path).name
                if sibling.is_file():
                    return str(sibling)
        except Exception:
            pass

    # Fallbacks for common repo layouts/checkouts.
    candidates = [
        f"data/matched_10B_docs2m_seed1337/tokenizers/fineweb_{vocab_size}_bpe.model",
        "data/matched_10B/tokenizers/fineweb_2048_bpe.model",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


FIXED_TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", _default_sentencepiece_tokenizer_path())
_BYTES_LUT_CACHE_NP: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def _fro_norm(x: mx.array) -> mx.array:
    return mx.sqrt(mx.sum(x * x))


def _zeropower_via_svd(g: mx.array) -> mx.array:
    u, _, vh = mx.linalg.svd(g, full_matrices=False)
    return u @ vh


def _zeropower_via_newtonschulz5(g: mx.array, steps: int = 5, eps: float = 1e-7) -> mx.array:
    # Newton-Schulz iteration approximating the orthogonal factor used by Muon.
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (_fro_norm(x) + eps)
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.T
        transposed = True
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def _load_data_shard(file: Path) -> np.ndarray:
    header = np.fromfile(file, dtype=np.int32, count=_SHARD_HEADER_INTS)
    if header.size != _SHARD_HEADER_INTS:
        raise ValueError(f"{file}: truncated header")
    if int(header[0]) != _SHARD_MAGIC:
        raise ValueError(f"{file}: magic mismatch {int(header[0])} != {_SHARD_MAGIC}")
    if int(header[1]) != _SHARD_VERSION:
        raise ValueError(f"{file}: version mismatch {int(header[1])} != {_SHARD_VERSION}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype=np.uint16, count=num_tokens, offset=_SHARD_HEADER_INTS * 4)
    if tokens.size != num_tokens:
        raise ValueError(f"{file}: token count mismatch {tokens.size} != {num_tokens}")
    return tokens.astype(np.int32, copy=False)


def bytes_per_token_np(input_ids: np.ndarray, target_ids: np.ndarray, model_vocab_size: int) -> np.ndarray:
    """
    Return per-target UTF-8 byte counts for compression metrics (NumPy port of train_gpt.py logic).
    """
    if input_ids.shape != target_ids.shape:
        raise ValueError(f"shape mismatch: input_ids={input_ids.shape} target_ids={target_ids.shape}")

    if TOKENIZER_KIND in {"gpt2", "gpt-2"}:
        tokenizer_cache_key = "gpt2"
    elif TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
        tokenizer_cache_key = f"sentencepiece:{Path(FIXED_TOKENIZER_PATH).resolve()}"
    else:
        tokenizer_cache_key = TOKENIZER_KIND

    cache_key = (tokenizer_cache_key, int(model_vocab_size))
    cached = _BYTES_LUT_CACHE_NP.get(cache_key)
    if cached is None:
        if TOKENIZER_KIND in {"gpt2", "gpt-2"}:
            try:
                import tiktoken
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "tiktoken is required for TOKENIZER_KIND=gpt2 bytes-per-token accounting"
                ) from exc

            enc = tiktoken.get_encoding("gpt2")
            table_size = max(int(enc.n_vocab), int(model_vocab_size))
            base_bytes = np.zeros((table_size,), dtype=np.int16)
            has_leading_space = np.zeros((table_size,), dtype=np.bool_)
            is_boundary_token = np.ones((table_size,), dtype=np.bool_)

            eot_id = int(enc.eot_token)
            for token_id in range(min(int(enc.n_vocab), table_size)):
                if token_id == eot_id:
                    base_bytes[token_id] = 0
                    is_boundary_token[token_id] = True
                    continue
                piece_bytes = enc.decode_single_token_bytes(token_id)
                base_bytes[token_id] = len(piece_bytes)
                is_boundary_token[token_id] = False
        elif TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
            tokenizer_path = Path(FIXED_TOKENIZER_PATH)
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"Tokenizer not found at {tokenizer_path}. "
                    "Set TOKENIZER_PATH to the SentencePiece model used to build the shards."
                )
            try:
                import sentencepiece as spm
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentencepiece is required for TOKENIZER_KIND=sentencepiece bytes-per-token accounting"
                ) from exc

            sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            sp_vocab_size = int(sp.vocab_size())
            table_size = max(sp_vocab_size, int(model_vocab_size))
            base_bytes = np.zeros((table_size,), dtype=np.int16)
            has_leading_space = np.zeros((table_size,), dtype=np.bool_)
            is_boundary_token = np.ones((table_size,), dtype=np.bool_)

            has_is_byte = hasattr(sp, "is_byte")
            for token_id in range(sp_vocab_size):
                is_control = bool(sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id))
                if is_control:
                    base_bytes[token_id] = 0
                    is_boundary_token[token_id] = True
                    continue

                is_boundary_token[token_id] = False
                if has_is_byte and bool(sp.is_byte(token_id)):
                    base_bytes[token_id] = 1
                    continue

                piece = sp.id_to_piece(token_id)
                leading = piece.startswith("▁")
                if leading:
                    piece = piece[1:]
                has_leading_space[token_id] = leading
                base_bytes[token_id] = len(piece.encode("utf-8"))
        else:
            raise ValueError(
                f"Unsupported TOKENIZER_KIND={TOKENIZER_KIND!r}. Expected 'gpt2' or 'sentencepiece'."
            )

        cached = (base_bytes, has_leading_space, is_boundary_token)
        _BYTES_LUT_CACHE_NP[cache_key] = cached

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = cached
    flat_prev = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    flat_target = np.asarray(target_ids, dtype=np.int64).reshape(-1)

    if flat_prev.size == 0:
        return np.zeros_like(target_ids, dtype=np.int16)
    if flat_prev.max(initial=0) >= base_bytes_lut.shape[0] or flat_target.max(initial=0) >= base_bytes_lut.shape[0]:
        raise ValueError(
            "Token id exceeds tokenizer/model vocab table size "
            f"(table={base_bytes_lut.shape[0]}, prev_max={int(flat_prev.max())}, target_max={int(flat_target.max())})"
        )

    out = base_bytes_lut[flat_target].astype(np.int16, copy=True)
    needs_space = has_leading_space_lut[flat_target]
    prev_allows_space = ~is_boundary_token_lut[flat_prev]
    out += (needs_space & prev_allows_space).astype(np.int16, copy=False)
    return out.reshape(target_ids.shape)


class TokenSource:
    def take(self, n: int) -> np.ndarray:
        raise NotImplementedError


class ShardTokenStream(TokenSource):
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        remaining = n
        while remaining > 0:
            avail = int(self.tokens.size - self.pos)
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)


class ArrayTokenStream(TokenSource):
    def __init__(self, tokens: np.ndarray):
        if tokens.ndim != 1:
            raise ValueError("tokens must be rank-1")
        if tokens.size < 2:
            raise ValueError("need at least 2 tokens")
        self.tokens = tokens.astype(np.int32, copy=False)
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0,), dtype=np.int32)
        out = np.empty((n,), dtype=np.int32)
        filled = 0
        size = int(self.tokens.size)
        while filled < n:
            if self.pos >= size:
                self.pos = 0
            k = min(n - filled, size - self.pos)
            out[filled : filled + k] = self.tokens[self.pos : self.pos + k]
            self.pos += k
            filled += k
        return out


class TokenLoader:
    """
    Simple single-device version of DistributedTokenLoader from train_gpt.py.
    Batch size is expressed in tokens, then packed into [B, T].
    """

    def __init__(self, source: TokenSource):
        self.source = source

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        if batch_tokens <= 0:
            raise ValueError("batch_tokens must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        usable_tokens = (batch_tokens // seq_len) * seq_len
        if usable_tokens <= 0:
            raise ValueError(f"Sequence length {seq_len} too long for token budget {batch_tokens}")
        chunk = self.source.take(usable_tokens + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, eps=self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rope_base: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.v_mix = mx.array(0.5, dtype=mx.float32)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array, v1: mx.array | None) -> tuple[mx.array, mx.array]:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if v1 is None:
            v1 = v
        mix = self.v_mix.astype(v.dtype)
        v = (1.0 - mix) * v + mix * v1

        q = _rms_norm(q)
        k = _rms_norm(k)
        q = self.rope(q)
        k = self.rope(k)

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y), v1


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        x = x * x
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, rope_base: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, rope_base=rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.resid_mix = mx.array([1.0, 0.0], dtype=mx.float32)

    def __call__(self, x: mx.array, x0: mx.array, v1: mx.array | None) -> tuple[mx.array, mx.array]:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0] * x + mix[1] * x0
        attn_out, v1 = self.attn(self.attn_norm(x), v1)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, v1


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        mlp_mult: int,
        max_seq_len: int,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        tie_embeddings: bool,
        tied_embed_init_std: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = mx.ones((self.num_decoder_layers,), dtype=mx.float32)
        self.blocks = [Block(model_dim, num_heads, mlp_mult, rope_base=rope_base) for _ in range(num_layers)]
        self.final_norm = RMSNormNoWeight()
        self.lm_head = None if tie_embeddings else nn.Linear(model_dim, vocab_size, bias=False)

        # Match train_gpt.py behavior: zero-init proj and head.
        for blk in self.blocks:
            blk.attn.proj.weight = mx.zeros_like(blk.attn.proj.weight)
            blk.mlp.proj.weight = mx.zeros_like(blk.mlp.proj.weight)
        if self.lm_head is not None:
            self.lm_head.weight = mx.zeros_like(self.lm_head.weight)

        if tie_embeddings and tied_embed_init_std > 0:
            self.tok_emb.weight = (
                mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * float(tied_embed_init_std)
            )

    def _apply_softcap(self, logits: mx.array) -> mx.array:
        if self.logit_softcap <= 0:
            return logits
        c = float(self.logit_softcap)
        return c * mx.tanh(logits / c)

    def _head(self, x: mx.array) -> mx.array:
        if self.tie_embeddings:
            return x @ self.tok_emb.weight.T
        assert self.lm_head is not None
        return self.lm_head(x)

    def __call__(self, input_ids: mx.array) -> mx.array:
        _, seqlen = input_ids.shape
        if seqlen > self.max_seq_len:
            raise ValueError(f"Input sequence length {seqlen} exceeds max_seq_len {self.max_seq_len}")

        x = self.tok_emb(input_ids)
        x = _rms_norm(x)
        x0 = x
        v1 = None
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x, v1 = self.blocks[i](x, x0, v1)
            skips.append(x)

        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                skip_weight = self.skip_weights[i].astype(x.dtype)
                x = x + skip_weight * skip
            x, v1 = self.blocks[self.num_encoder_layers + i](x, x0, v1)

        x = self.final_norm(x)
        return x

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids)
        x = x.reshape(-1, x.shape[-1])
        targets = target_ids.reshape(-1)
        chunk_tokens = int(self.logit_chunk_tokens)

        if chunk_tokens <= 0 or x.shape[0] <= chunk_tokens:
            logits = self._apply_softcap(self._head(x))
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for start in range(0, n, chunk_tokens):
            end = min(start + chunk_tokens, n)
            logits = self._apply_softcap(self._head(x[start:end]))
            ce = nn.losses.cross_entropy(logits, targets[start:end], reduction="sum")
            loss_sum = loss_sum + ce
        return loss_sum / float(n)


@dataclass
class Hyperparameters:
    data_path: str = _env_str("DATA_PATH", "./data/matched_10B/datasets/fineweb10B_sp2048")
    train_files: str = os.path.join(data_path, "fineweb_train_*.bin")
    val_files: str = os.path.join(data_path, "fineweb_val_*.bin")
    run_id: str = _env_str("RUN_ID", str(uuid.uuid4()))
    seed: int = _env_int("SEED", 1337)

    iterations: int = _env_int("ITERATIONS", 200)
    val_loss_every: int = _env_int("VAL_LOSS_EVERY", 25)
    val_tokens: int = _env_int("VAL_TOKENS", 16_384)
    val_batch_tokens: int = _env_int("VAL_BATCH_TOKENS", 4_096)
    enable_val_bpb: bool = _env_bool("ENABLE_VAL_BPB", True)
    # This is the optimizer-step token budget (global batch for single-device MLX).
    train_batch_tokens: int = _env_int("TRAIN_BATCH_TOKENS", 4_096)
    grad_accum_steps: int = _env_int("GRAD_ACCUM_STEPS", 1)
    train_max_seq_len: int = _env_int("TRAIN_MAX_SEQ_LEN", 128)

    warmup_steps: int = _env_int("WARMUP_STEPS", 10)
    warmdown_iters: int = _env_int("WARMDOWN_ITERS", 0)

    vocab_size: int = _env_int("VOCAB_SIZE", 2048)
    num_layers: int = _env_int("NUM_LAYERS", 6)
    model_dim: int = _env_int("MODEL_DIM", 256)
    num_heads: int = _env_int("NUM_HEADS", 4)
    mlp_mult: int = _env_int("MLP_MULT", 4)
    tie_embeddings: bool = _env_bool("TIE_EMBEDDINGS", True)
    tied_embed_init_std: float = _env_float("TIED_EMBED_INIT_STD", 0.005)
    logit_chunk_tokens: int = _env_int("LOGIT_CHUNK_TOKENS", 0)
    logit_softcap: float = _env_float("LOGIT_SOFTCAP", 30.0)
    rope_base: float = _env_float("ROPE_BASE", 10000.0)

    base_lr: float = _env_float("BASE_LR", 3e-4)
    embed_lr: float = _env_float("EMBED_LR", 0.6)
    head_lr: float = _env_float("HEAD_LR", 0.008)
    tied_embed_lr: float = _env_float("TIED_EMBED_LR", 0.05)
    matrix_lr: float = _env_float("MATRIX_LR", 0.04)
    scalar_lr: float = _env_float("SCALAR_LR", 0.04)
    beta1: float = _env_float("BETA1", 0.9)
    beta2: float = _env_float("BETA2", 0.95)
    adam_eps: float = _env_float("ADAM_EPS", 1e-8)
    weight_decay: float = _env_float("WEIGHT_DECAY", 0.0)
    use_muon_split: bool = _env_bool("USE_MUON_SPLIT", True)
    muon_momentum: float = _env_float("MUON_MOMENTUM", 0.95)
    muon_nesterov: bool = _env_bool("MUON_NESTEROV", True)
    muon_backend: str = _env_str("MUON_BACKEND", "newtonschulz5")
    muon_backend_steps: int = _env_int("MUON_BACKEND_STEPS", 5)
    muon_momentum_warmup_start: float = _env_float("MUON_MOMENTUM_WARMUP_START", 0.85)
    muon_momentum_warmup_steps: int = _env_int("MUON_MOMENTUM_WARMUP_STEPS", 500)

    save_model: bool = _env_bool("SAVE_MODEL", True)
    out_dir: str = _env_str("OUT_DIR", "logs")

    # Fallback toy data for local Mac verification when FineWeb shards are absent.
    toy_text_path: str = _env_str("TOY_TEXT_PATH", "")
    toy_repeat: int = _env_int("TOY_REPEAT", 512)


class TrainingManager:
    def __init__(self, args: Hyperparameters):
        self.args = args

    def lr_for_step(self, step: int) -> float:
        lr = self.args.base_lr
        if self.args.warmup_steps > 0 and step < self.args.warmup_steps:
            lr *= (step + 1) / self.args.warmup_steps
        if self.args.warmdown_iters > 0:
            wd_start = max(self.args.iterations - self.args.warmdown_iters, 0)
            if wd_start <= step < self.args.iterations:
                decay_ratio = (self.args.iterations - step) / max(self.args.warmdown_iters, 1)
                lr *= max(decay_ratio, 0.0)
        return float(lr)

    def microbatch_tokens(self) -> int:
        if self.args.grad_accum_steps <= 0:
            raise ValueError("GRAD_ACCUM_STEPS must be > 0")
        if self.args.train_batch_tokens % self.args.grad_accum_steps != 0:
            raise ValueError(
                "TRAIN_BATCH_TOKENS must be divisible by GRAD_ACCUM_STEPS "
                f"(got {self.args.train_batch_tokens} and {self.args.grad_accum_steps})"
            )
        return self.args.train_batch_tokens // self.args.grad_accum_steps

    def lr_mul_for_step(self, step: int) -> float:
        base = max(self.args.base_lr, 1e-12)
        return self.lr_for_step(step) / base


class MuonMLX:
    """
    SGD-momentum + orthogonalized updates for selected 2D parameters.
    """

    def __init__(
        self,
        keys: list[str],
        params_flat: dict[str, mx.array],
        lr: float,
        momentum: float,
        nesterov: bool,
        backend: str,
        backend_steps: int,
    ):
        self.keys = list(keys)
        self.base_lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.backend = backend
        self.backend_steps = int(backend_steps)
        self.momentum_buffers = {k: mx.zeros_like(params_flat[k]) for k in self.keys}

    def _orthogonalize(self, g: mx.array) -> mx.array:
        if self.backend == "svd":
            return _zeropower_via_svd(g)
        if self.backend == "newtonschulz5":
            return _zeropower_via_newtonschulz5(g, steps=self.backend_steps)
        raise ValueError(f"Unsupported MUON_BACKEND={self.backend!r}")

    def step(
        self,
        params_flat: dict[str, mx.array],
        grads_flat: dict[str, mx.array],
        lr_mul: float,
        momentum_override: float | None = None,
    ) -> dict[str, mx.array]:
        updated: dict[str, mx.array] = {}
        lr = self.base_lr * float(lr_mul)
        momentum = self.momentum if momentum_override is None else float(momentum_override)
        for k in self.keys:
            p = params_flat[k]
            g = grads_flat.get(k)
            if g is None:
                updated[k] = p
                continue
            buf_prev = self.momentum_buffers[k]
            buf = momentum * buf_prev + g
            self.momentum_buffers[k] = buf
            g_eff = g + momentum * buf if self.nesterov else buf
            g_ortho = self._orthogonalize(g_eff)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            updated[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return updated

    def state_for_eval(self):
        return self.momentum_buffers


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params_flat = dict(tree_flatten(model.parameters()))
        keys = list(params_flat.keys())

        block_keys = [k for k in keys if k.startswith("blocks.")]
        matrix_keys = [k for k in block_keys if params_flat[k].ndim == 2]
        scalar_keys = [k for k in block_keys if params_flat[k].ndim < 2]
        if "skip_weights" in params_flat:
            scalar_keys.append("skip_weights")

        self.matrix_keys = matrix_keys
        self.scalar_keys = scalar_keys
        self.embed_key = "tok_emb.weight" if "tok_emb.weight" in params_flat else None
        self.head_key = "lm_head.weight" if "lm_head.weight" in params_flat else None
        self.tied_embed_head = self.head_key is None

        embed_lr = args.tied_embed_lr if self.tied_embed_head else args.embed_lr
        self.embed_base_lr = float(embed_lr)
        self.head_base_lr = float(args.head_lr)
        self.scalar_base_lr = float(args.scalar_lr)

        self.muon = MuonMLX(
            keys=self.matrix_keys,
            params_flat=params_flat,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            nesterov=args.muon_nesterov,
            backend=args.muon_backend,
            backend_steps=args.muon_backend_steps,
        )
        self.adam_embed = optim.Adam(
            learning_rate=self.embed_base_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        ) if self.embed_key is not None else None
        self.adam_head = optim.Adam(
            learning_rate=self.head_base_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        ) if self.head_key is not None else None
        self.adam_scalar = optim.Adam(
            learning_rate=self.scalar_base_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        ) if self.scalar_keys else None

    def _apply_weight_decay_if_needed(
        self,
        params_sub: dict[str, mx.array],
        grads_sub: dict[str, mx.array],
    ) -> dict[str, mx.array]:
        if self.args.weight_decay <= 0:
            return grads_sub
        out = dict(grads_sub)
        for k, g in grads_sub.items():
            p = params_sub.get(k)
            if p is None:
                continue
            if p.ndim >= 2:
                out[k] = g + self.args.weight_decay * p
        return out

    def step(self, model: GPT, grads: dict, step: int, lr_mul: float) -> None:
        params_flat = dict(tree_flatten(model.parameters()))
        grads_flat = dict(tree_flatten(grads))
        updated_flat = dict(params_flat)

        if self.args.muon_momentum_warmup_steps > 0:
            frac = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            muon_momentum = (
                (1.0 - frac) * self.args.muon_momentum_warmup_start + frac * self.args.muon_momentum
            )
        else:
            muon_momentum = self.args.muon_momentum

        updated_flat.update(
            self.muon.step(
                params_flat=params_flat,
                grads_flat=grads_flat,
                lr_mul=lr_mul,
                momentum_override=muon_momentum,
            )
        )

        if self.adam_embed is not None and self.embed_key is not None:
            self.adam_embed.learning_rate = self.embed_base_lr * lr_mul
            p_sub = {self.embed_key: params_flat[self.embed_key]}
            g_sub = {self.embed_key: grads_flat[self.embed_key]} if self.embed_key in grads_flat else {}
            if g_sub:
                g_sub = self._apply_weight_decay_if_needed(p_sub, g_sub)
                updated_flat.update(self.adam_embed.apply_gradients(g_sub, p_sub))

        if self.adam_head is not None and self.head_key is not None:
            self.adam_head.learning_rate = self.head_base_lr * lr_mul
            p_sub = {self.head_key: params_flat[self.head_key]}
            g_sub = {self.head_key: grads_flat[self.head_key]} if self.head_key in grads_flat else {}
            if g_sub:
                g_sub = self._apply_weight_decay_if_needed(p_sub, g_sub)
                updated_flat.update(self.adam_head.apply_gradients(g_sub, p_sub))

        if self.adam_scalar is not None and self.scalar_keys:
            self.adam_scalar.learning_rate = self.scalar_base_lr * lr_mul
            p_sub = {k: params_flat[k] for k in self.scalar_keys}
            g_sub = {k: grads_flat[k] for k in self.scalar_keys if k in grads_flat}
            if g_sub:
                g_sub = self._apply_weight_decay_if_needed(p_sub, g_sub)
                updated_flat.update(self.adam_scalar.apply_gradients(g_sub, p_sub))

        model.update(tree_unflatten(list(updated_flat.items())))

    def state_for_eval(self):
        state = [self.muon.state_for_eval()]
        if self.adam_embed is not None:
            state.append(self.adam_embed.state)
        if self.adam_head is not None:
            state.append(self.adam_head.state)
        if self.adam_scalar is not None:
            state.append(self.adam_scalar.state)
        return state


def _build_toy_stream(args: Hyperparameters, split: str) -> TokenLoader:
    if args.toy_text_path:
        text = Path(args.toy_text_path).read_text(encoding="utf-8")
    else:
        text = (
            "OpenAI Parameter Golf MLX toy training corpus. "
            "This fallback is only for local Mac verification when FineWeb shards are not present. "
            "The model should overfit this text quickly. "
        )
    if split == "val":
        text = text[::-1]
    data = (text * max(args.toy_repeat, 1)).encode("utf-8")
    toks = np.frombuffer(data, dtype=np.uint8).astype(np.int32, copy=False)
    if toks.size < 2:
        raise ValueError("Toy text produced too few tokens")
    if toks.max(initial=0) >= args.vocab_size:
        raise ValueError(
            f"Toy byte fallback needs VOCAB_SIZE >= 256 (got {args.vocab_size})"
        )
    return TokenLoader(ArrayTokenStream(toks))


def _build_loader(args: Hyperparameters, pattern: str, split: str) -> tuple[TokenLoader, str]:
    files = sorted(glob.glob(pattern))
    if files:
        return TokenLoader(ShardTokenStream(pattern)), f"shards:{len(files)}"
    return _build_toy_stream(args, split=split), "toy-bytes"


def _eval_loss(model: GPT, loader: TokenLoader, args: Hyperparameters) -> tuple[float, float | None]:
    if args.val_tokens % args.val_batch_tokens != 0:
        raise ValueError("VAL_TOKENS must be divisible by VAL_BATCH_TOKENS")
    steps = args.val_tokens // args.val_batch_tokens
    total = mx.array(0.0, dtype=mx.float32)
    val_token_count = 0.0
    val_byte_count = 0.0
    val_bpb_enabled = args.enable_val_bpb
    for _ in range(steps):
        x, y = loader.next_batch(args.val_batch_tokens, args.train_max_seq_len)
        total = total + model.loss(x, y)
        if val_bpb_enabled:
            try:
                x_np = np.array(x)
                y_np = np.array(y)
                bytes_np = bytes_per_token_np(x_np, y_np, model_vocab_size=args.vocab_size)
                if bytes_np.shape != y_np.shape:
                    raise ValueError(
                        f"bytes_per_token_np returned shape {bytes_np.shape}, expected {y_np.shape}"
                    )
                val_token_count += float(y_np.size)
                val_byte_count += float(bytes_np.astype(np.float64).sum())
            except Exception as exc:
                val_bpb_enabled = False
                print(f"val_bpb:disabled error={exc}")
    total = total / float(steps)
    mx.eval(total)
    val_loss = float(total.item())

    val_bpb: float | None = None
    if val_bpb_enabled and val_byte_count > 0:
        bits_per_token = val_loss / math.log(2.0)
        tokens_per_byte = val_token_count / val_byte_count
        val_bpb = bits_per_token * tokens_per_byte
    elif val_bpb_enabled and val_byte_count <= 0:
        print("val_bpb:disabled zero_byte_count")
    return val_loss, val_bpb


def _flatten_params_for_save(model: GPT) -> dict[str, mx.array]:
    return {k: v for k, v in tree_flatten(model.parameters())}


def main() -> None:
    args = Hyperparameters()
    mx.random.seed(args.seed)

    train_loader, train_loader_kind = _build_loader(args, args.train_files, split="train")
    val_loader, val_loader_kind = _build_loader(args, args.val_files, split="val")

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        mlp_mult=args.mlp_mult,
        max_seq_len=args.train_max_seq_len,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
    )

    manager = TrainingManager(args)
    split_optimizers = SplitOptimizers(model, args) if args.use_muon_split else None
    optimizer = None
    if split_optimizers is None:
        optimizer = optim.Adam(
            learning_rate=args.base_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def loss_fn(m: GPT, x: mx.array, y: mx.array) -> mx.array:
        return m.loss(x, y)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"run_id:{args.run_id}")
    print(f"mlx_version:{mx.__version__}")
    print(f"train_loader:{train_loader_kind} pattern={args.train_files}")
    print(f"val_loader:{val_loader_kind} pattern={args.val_files}")
    print(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} seq_len:{args.train_max_seq_len} "
        f"tie_embeddings:{args.tie_embeddings}"
    )
    print(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} "
        f"grad_accum_steps:{args.grad_accum_steps} val_batch_tokens:{args.val_batch_tokens} "
        f"val_tokens:{args.val_tokens}"
    )
    microbatch_tokens = manager.microbatch_tokens()
    print(
        f"microbatch_tokens:{microbatch_tokens} "
        f"microbatch_batch_size:{microbatch_tokens // args.train_max_seq_len}"
    )
    if args.enable_val_bpb:
        msg = f"val_bpb:enabled tokenizer_kind={TOKENIZER_KIND}"
        if TOKENIZER_KIND in {"sentencepiece", "spm", "sp"}:
            msg += f" tokenizer_path={FIXED_TOKENIZER_PATH}"
        print(msg)
    else:
        print("val_bpb:disabled by ENABLE_VAL_BPB=0")
    if split_optimizers is not None:
        print(
            f"optimizer:muon+adam "
            f"muon_backend:{args.muon_backend} muon_matrix_params:{len(split_optimizers.matrix_keys)} "
            f"scalar_params:{len(split_optimizers.scalar_keys)} "
            f"embed_lr:{split_optimizers.embed_base_lr} "
            + (f"head_lr:{split_optimizers.head_base_lr} " if split_optimizers.head_key else "")
            + f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
        )
    else:
        print("optimizer:adam")

    t0 = time.perf_counter()
    running_ms = 0.0

    for step in range(args.iterations + 1):
        last_step = step == args.iterations
        run_eval = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if run_eval:
            val_loss, val_bpb = _eval_loss(model, val_loader, args)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            running_ms += elapsed_ms
            t0 = time.perf_counter()
            val_bpb_str = f" val_bpb:{val_bpb:.4f}" if val_bpb is not None else ""
            print(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} "
                f"train_time:{running_ms:.0f}ms step_avg:{running_ms / max(step, 1):.2f}ms"
                f"{val_bpb_str}"
            )

        if last_step:
            break

        lr = manager.lr_for_step(step)
        lr_mul = manager.lr_mul_for_step(step)
        if optimizer is not None:
            optimizer.learning_rate = lr

        step_t0 = time.perf_counter()
        train_loss = None
        accum_flat: dict[str, mx.array] | None = None
        for micro_step in range(args.grad_accum_steps):
            x, y = train_loader.next_batch(microbatch_tokens, args.train_max_seq_len)
            loss, grads = loss_and_grad(model, x, y)
            flat_grads = dict(tree_flatten(grads))
            scale = 1.0 / args.grad_accum_steps
            if accum_flat is None:
                accum_flat = {k: g * scale for k, g in flat_grads.items()}
            else:
                for k, g in flat_grads.items():
                    accum_flat[k] = accum_flat[k] + g * scale
            train_loss = loss
            # Materialize periodically so large grad-accum runs do not build up lazy graphs.
            if args.grad_accum_steps > 1 and (
                (micro_step + 1) % 8 == 0 or (micro_step + 1) == args.grad_accum_steps
            ):
                mx.eval(*accum_flat.values())
        if train_loss is None or accum_flat is None:
            raise RuntimeError("No microbatches were executed")
        grads = tree_unflatten(list(accum_flat.items()))

        if args.weight_decay > 0 and optimizer is not None:
            params_flat = dict(tree_flatten(model.parameters()))
            grads_flat = dict(tree_flatten(grads))
            for key, g in grads_flat.items():
                p = params_flat.get(key)
                if p is None:
                    continue
                # Skip simple scalar controls from weight decay.
                if p.ndim >= 2:
                    grads_flat[key] = g + args.weight_decay * p
            grads = dict(grads_flat)

        if split_optimizers is not None:
            split_optimizers.step(model, grads, step=step, lr_mul=lr_mul)
            mx.eval(loss, model.parameters(), *split_optimizers.state_for_eval())
        else:
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        step_ms = (time.perf_counter() - step_t0) * 1000.0
        approx_ms = running_ms + elapsed_ms
        step_toks_per_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
        print(
            f"step:{step + 1}/{args.iterations} train_loss:{float(train_loss.item()):.4f} lr:{lr:.2e} "
            f"step_ms:{step_ms:.0f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms / (step + 1):.2f}ms "
            f"tok_s:{step_toks_per_s:.0f}"
        )

    if args.save_model:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.run_id}_mlx_model.npz"
        mx.savez(str(out_path), **_flatten_params_for_save(model))
        size = out_path.stat().st_size
        print(f"saved_model:{out_path} bytes:{size}")


if __name__ == "__main__":
    main()
