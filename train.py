"""Single-file training and evaluation pipeline for constrained model artifacts."""
from __future__ import annotations

import collections
import copy
import datetime
import glob
import io
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

_ACCUM_WORLD_REFERENCE = 8
MAX_ARTIFACT_BYTES = 16 * 1_000_000

try:
    from flash_attn_interface import flash_attn_func as _fa3_impl

    _USE_FA3 = True


    def flash_attn_3_func(q, k, v, causal: bool = True):
        return _fa3_impl(q, k, v, causal=causal)

except ImportError:
    _USE_FA3 = False


    def flash_attn_3_func(q, k, v, causal: bool = True):
        B, T, Hq, D = q.shape
        Hkv = k.shape[2]
        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()
        if Hkv != Hq:
            repeat = Hq // Hkv
            k_ = k_.repeat_interleave(repeat, dim=1)
            v_ = v_.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal)
        return out.permute(0, 2, 1, 3).contiguous()


def _env_lookup(name_or_names, default):
    if isinstance(name_or_names, (tuple, list)):
        for key in name_or_names:
            if key in os.environ:
                return os.environ[key]
        return default
    return os.environ.get(name_or_names, default)


def _env_int(name_or_names, default: int) -> int:
    return int(_env_lookup(name_or_names, default))


def _env_float(name_or_names, default: float) -> float:
    return float(_env_lookup(name_or_names, default))


def _env_flag(name_or_names, default: bool) -> bool:
    return bool(int(_env_lookup(name_or_names, "1" if default else "0")))


def _default_lowbit_layers(num_layers: int) -> str:
    tail_a = max(0, num_layers - 2)
    tail_b = max(0, num_layers - 1)
    return f"blocks.{tail_a}.:5,blocks.{tail_b}.:5"


def _default_profile_values():
    # Kept in one table to avoid exposing a one-field-per-line signature.
    return {
        "seed": 1337,
        "iterations": 50000,
        "max_wallclock_seconds": 590.0,
        "warmdown_frac": 0.72,
        "warmup_steps": 20,
        "train_batch_tokens": 786_432,
        "train_seq_len": 2048,
        "train_log_every": 500,
        "val_loss_every": 4000,
        "val_batch_tokens": 524_288,
        "eval_seq_len": 2048,
        "num_layers": 11,
        "cleanup_last_n": 11,
        "model_dim": 512,
        "embedding_dim": 512,
        "num_kv_heads": 4,
        "num_heads": 8,
        "mlp_mult": 4.0,
        "logit_softcap": 30.0,
        "rope_base": 10_000.0,
        "rope_dims": 16,
        "rope_train_seq_len": 2048,
        "query_gain_seed": 5.25,
        "recurrent_passes": 2,
        "recurrent_begin": 3,
        "recurrent_end": 5,
        "recurrent_enable_frac": 0.35,
        "dual_path_from_layer": 7,
        "min_lr": 0.1,
        "embed_lr": 0.6,
        "head_lr": 0.008,
        "tied_embed_lr": 0.03,
        "tied_embed_init_std": 0.005,
        "matrix_lr": 0.022,
        "scalar_lr": 0.02,
        "ortho_momentum": 0.99,
        "ortho_steps": 5,
        "ortho_warmup_start": 0.92,
        "ortho_warmup_fraction": 0.22,
        "beta1": 0.9,
        "beta2": 0.95,
        "adam_eps": 1e-8,
        "grad_clip_norm": 0.3,
        "eval_stride": 64,
        "matrix_beta2": 0.95,
        "adam_wd": 0.005,
        "matrix_wd_attn": 0.095,
        "matrix_wd_mlp": 0.115,
        "embed_wd": 0.085,
        "ema_decay": 0.9965,
        "adapt_lr": 0.005,
        "adapt_epochs": 4,
        "adapt_momentum": 0.9,
        "adapt_chunk_tokens": 40960,
        "quant_calibration_batches": 64,
        "matrix_bits": 6,
        "embed_bits": 8,
        "matrix_clip_sigmas": 12.85,
        "embed_clip_sigmas": 20.0,
    }


_DEFAULTS = _default_profile_values()


class ExecutionProfile:
    _ENV_KEYS = {
        "iterations": "TRAIN_STEPS",
        "train_batch_tokens": "TOKENS_PER_STEP",
        "val_batch_tokens": "VAL_TOKENS_PER_STEP",
        "eval_seq_len": "VAL_SEQ_LEN",
        "query_gain_seed": "QUERY_KEY_GAIN",
        "dual_path_from_layer": "PARALLEL_FROM_LAYER",
        "cleanup_last_n": "CLEANUP_LAST_N",
        "recurrent_passes": "RECURRENT_PASSES",
        "recurrent_begin": "RECURRENT_BEGIN",
        "recurrent_end": "RECURRENT_END",
        "recurrent_enable_frac": "RECURRENT_ENABLE_FRAC",
        "bridge_gates_enabled": "BRIDGE_GATES_ENABLED",
        "adapt_enabled": "ADAPTIVE_EVAL_ENABLED",
        "adapt_lr": "ADAPT_LR",
        "adapt_epochs": "ADAPT_EPOCHS",
        "adapt_momentum": "ADAPT_MOMENTUM",
        "adapt_chunk_tokens": "ADAPT_CHUNK_TOKENS",
        "eval_stride": "SLIDING_STRIDE",
        "quant_calibration_batches": "QUANT_CALIBRATION_BATCHES",
        "ortho_momentum": "ORTHO_MOMENTUM",
        "ortho_steps": "ORTHO_STEPS",
        "ortho_warmup_start": "ORTHO_WARMUP_START",
        "ortho_warmup_fraction": "ORTHO_WARMUP_FRACTION",
        "matrix_row_normalize": "MATRIX_ROW_NORMALIZE",
        "matrix_wd_attn": "ATTN_MATRIX_WD",
        "matrix_wd_mlp": "MLP_MATRIX_WD",
    }
    _FIELD_SPECS = (
        ("seed", _env_int, _DEFAULTS["seed"]),
        ("iterations", _env_int, _DEFAULTS["iterations"]),
        ("max_wallclock_seconds", _env_float, _DEFAULTS["max_wallclock_seconds"]),
        ("warmdown_frac", _env_float, _DEFAULTS["warmdown_frac"]),
        ("warmup_steps", _env_int, _DEFAULTS["warmup_steps"]),
        ("train_batch_tokens", _env_int, _DEFAULTS["train_batch_tokens"]),
        ("train_seq_len", _env_int, _DEFAULTS["train_seq_len"]),
        ("train_log_every", _env_int, _DEFAULTS["train_log_every"]),
        ("val_loss_every", _env_int, _DEFAULTS["val_loss_every"]),
        ("val_batch_tokens", _env_int, _DEFAULTS["val_batch_tokens"]),
        ("eval_seq_len", _env_int, _DEFAULTS["eval_seq_len"]),
        ("sliding_window_enabled", _env_flag, True),
        ("num_layers", _env_int, _DEFAULTS["num_layers"]),
        ("cleanup_last_n", _env_int, _DEFAULTS["cleanup_last_n"]),
        ("model_dim", _env_int, _DEFAULTS["model_dim"]),
        ("embedding_dim", _env_int, _DEFAULTS["embedding_dim"]),
        ("num_kv_heads", _env_int, _DEFAULTS["num_kv_heads"]),
        ("num_heads", _env_int, _DEFAULTS["num_heads"]),
        ("mlp_mult", _env_float, _DEFAULTS["mlp_mult"]),
        ("bridge_gates_enabled", _env_flag, True),
        ("tie_embeddings", _env_flag, True),
        ("logit_softcap", _env_float, _DEFAULTS["logit_softcap"]),
        ("rope_base", _env_float, _DEFAULTS["rope_base"]),
        ("rope_dims", _env_int, _DEFAULTS["rope_dims"]),
        ("rope_train_seq_len", _env_int, _DEFAULTS["rope_train_seq_len"]),
        ("ln_scale", _env_flag, True),
        ("query_gain_seed", _env_float, _DEFAULTS["query_gain_seed"]),
        ("recurrent_passes", _env_int, _DEFAULTS["recurrent_passes"]),
        ("recurrent_begin", _env_int, _DEFAULTS["recurrent_begin"]),
        ("recurrent_end", _env_int, _DEFAULTS["recurrent_end"]),
        ("recurrent_enable_frac", _env_float, _DEFAULTS["recurrent_enable_frac"]),
        ("dual_path_from_layer", _env_int, _DEFAULTS["dual_path_from_layer"]),
        ("min_lr", _env_float, _DEFAULTS["min_lr"]),
        ("embed_lr", _env_float, _DEFAULTS["embed_lr"]),
        ("head_lr", _env_float, _DEFAULTS["head_lr"]),
        ("tied_embed_lr", _env_float, _DEFAULTS["tied_embed_lr"]),
        ("tied_embed_init_std", _env_float, _DEFAULTS["tied_embed_init_std"]),
        ("matrix_lr", _env_float, _DEFAULTS["matrix_lr"]),
        ("scalar_lr", _env_float, _DEFAULTS["scalar_lr"]),
        ("ortho_momentum", _env_float, _DEFAULTS["ortho_momentum"]),
        ("ortho_steps", _env_int, _DEFAULTS["ortho_steps"]),
        ("ortho_warmup_start", _env_float, _DEFAULTS["ortho_warmup_start"]),
        ("ortho_warmup_fraction", _env_float, _DEFAULTS["ortho_warmup_fraction"]),
        ("matrix_row_normalize", _env_flag, True),
        ("beta1", _env_float, _DEFAULTS["beta1"]),
        ("beta2", _env_float, _DEFAULTS["beta2"]),
        ("adam_eps", _env_float, _DEFAULTS["adam_eps"]),
        ("grad_clip_norm", _env_float, _DEFAULTS["grad_clip_norm"]),
        ("eval_stride", _env_int, _DEFAULTS["eval_stride"]),
        ("matrix_beta2", _env_float, _DEFAULTS["matrix_beta2"]),
        ("adam_wd", _env_float, _DEFAULTS["adam_wd"]),
        ("matrix_wd_attn", _env_float, _DEFAULTS["matrix_wd_attn"]),
        ("matrix_wd_mlp", _env_float, _DEFAULTS["matrix_wd_mlp"]),
        ("embed_wd", _env_float, _DEFAULTS["embed_wd"]),
        ("ema_decay", _env_float, _DEFAULTS["ema_decay"]),
        ("adapt_enabled", _env_flag, True),
        ("adapt_lr", _env_float, _DEFAULTS["adapt_lr"]),
        ("adapt_epochs", _env_int, _DEFAULTS["adapt_epochs"]),
        ("adapt_momentum", _env_float, _DEFAULTS["adapt_momentum"]),
        ("adapt_chunk_tokens", _env_int, _DEFAULTS["adapt_chunk_tokens"]),
        ("quant_calibration_batches", _env_int, _DEFAULTS["quant_calibration_batches"]),
        ("matrix_bits", _env_int, _DEFAULTS["matrix_bits"]),
        ("embed_bits", _env_int, _DEFAULTS["embed_bits"]),
        ("matrix_clip_sigmas", _env_float, _DEFAULTS["matrix_clip_sigmas"]),
        ("embed_clip_sigmas", _env_float, _DEFAULTS["embed_clip_sigmas"]),
    )

    def __init__(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:8]
        self.run_id = os.environ.get("RUN_ID", f"{ts}_{short}")
        self.vocab_size = _env_int(("SUBWORD_VOCAB_SIZE", "VOCAB_SIZE"), _DEFAULTS["vocab_size"])
        self.data_dir = os.environ.get("DATA_DIR", "./data")
        corpus_stem = os.environ.get("CORPUS_STEM", "fineweb")
        corpus_scale = os.environ.get("CORPUS_SCALE", "10B")
        tokenizer_suffix = os.environ.get("TOKENIZER_SUFFIX", "bpe")
        dataset_bundle = f"{corpus_stem}{corpus_scale}_sp{self.vocab_size}"
        self.datasets_dir = os.path.join(self.data_dir, "datasets", dataset_bundle)
        self.train_files = _env_lookup(
            ("TRAIN_SHARDS", "TRAIN_FILES"),
            os.path.join(self.datasets_dir, f"{corpus_stem}_train_*.bin"),
        )
        self.val_files = _env_lookup(
            ("VAL_SHARDS", "VAL_FILES"),
            os.path.join(self.datasets_dir, f"{corpus_stem}_val_*.bin"),
        )
        self.tokenizer_path = os.environ.get(
            "TOKENIZER_PATH",
            os.path.join(self.data_dir, "tokenizers", f"{corpus_stem}_{self.vocab_size}_{tokenizer_suffix}.model"),
        )
        self.compressor = os.environ.get("COMPRESSOR", "brotli")
        for name, parser, default in self._FIELD_SPECS:
            env_key = self._ENV_KEYS.get(name, name.upper())
            setattr(self, name, parser(env_key, default))
        self.lowbit_layers = _env_lookup("TARGET_INT5_LAYERS", _default_lowbit_layers(self.num_layers))
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = _env_int("RANK", 0)
        self.world_size = _env_int("WORLD_SIZE", 1)
        self.local_rank = _env_int("LOCAL_RANK", 0)
        self.is_main_process = self.rank == 0
        self.grad_accum_steps = max(1, _ACCUM_WORLD_REFERENCE // self.world_size)
        self.logfile = f"logs/{self.run_id}.txt"
        os.makedirs("ckpt", exist_ok=True)
        self.model_path = "ckpt/final_model.pt"
        self.quantized_model_path = "ckpt/final_model.int6.ptz"


_logger_hparams = None


def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h
    return _logger_hparams


def log(msg, console=True):
    logger_state = _logger_hparams
    if logger_state is None:
        print(msg)
        return
    if not logger_state.is_main_process:
        return
    if console:
        print(msg)
    log_path = logger_state.logfile
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            print(msg, file=handle)


class BinaryShardStore:
    def __init__(self):
        self.header_words = 256
        self.header_dtype = "<i4"
        self.token_dtype = "<u2"
        self.magic = 20240520
        self.version = 1
        self.header_bytes = self.header_words * np.dtype(self.header_dtype).itemsize
        self._num_tokens = {}
        self._mmaps = {}

    def _parse_header(self, file):
        header = np.fromfile(file, dtype=self.header_dtype, count=self.header_words)
        ok = (
            header.size == self.header_words
            and int(header[0]) == self.magic
            and int(header[1]) == self.version
        )
        if not ok:
            raise ValueError(f"Unexpected shard header for {file}")
        return int(header[2])

    def token_count(self, file):
        key = str(file)
        cached = self._num_tokens.get(key)
        if cached is not None:
            return cached
        n = self._parse_header(file)
        self._num_tokens[key] = n
        return n

    def mapped_tokens(self, file):
        key = str(file)
        mm = self._mmaps.get(key)
        if mm is not None:
            return mm
        n = self.token_count(file)
        mm = np.memmap(file, mode="r", dtype=self.token_dtype, offset=self.header_bytes, shape=(n,))
        self._mmaps[key] = mm
        return mm


_SHARD_STORE = BinaryShardStore()


def decode_shard_payload(file):
    header_bytes = _SHARD_STORE.header_bytes
    token_bytes = np.dtype(_SHARD_STORE.token_dtype).itemsize
    num_tokens = _SHARD_STORE.token_count(file)
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def gather_eval_stream(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([decode_shard_payload(f) for f in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_bpb_tables(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    if sp.piece_to_id("▁") == sp.unk_id():
        raise RuntimeError("Tokenizer must have '▁' as a token for BPB byte counting")
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


class CorpusEvalContext:
    def __init__(self, h, device):
        tokenizer = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(tokenizer.vocab_size()) != h.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(tokenizer.vocab_size())}"
            )
        self.sp = tokenizer
        self.val_tokens = gather_eval_stream(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_bpb_tables(
            tokenizer, h.vocab_size, device
        )


class SequenceSampler:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
        self.files = all_files[h.rank:: h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_SHARD_STORE.token_count(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        self._phase_epoch = [0] * len(self.files)
        self._phase_order = [self.rng.permutation(8).tolist() for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        if max_phase > 0:
            k = self._phase_order[si][self._phase_epoch[si] % 8]
            self._phase_epoch[si] += 1
            phase = min(k * (max_phase + 1) // 8, max_phase)
        else:
            phase = 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def sample_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        num_shards = len(self.files)
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        total = remaining.sum()
        if total <= 0:
            for si in range(num_shards):
                self._reset_shard(si)
            remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
            total = remaining.sum()
        target = device_batch_size * remaining / total
        quotas = np.floor(target).astype(np.int64)
        leftover = device_batch_size - int(quotas.sum())
        if leftover > 0:
            order = np.argsort(-(target - quotas))
            for i in order[:leftover]:
                quotas[int(i)] += 1
        quotas = np.minimum(quotas, remaining.astype(np.int64))
        shortfall = device_batch_size - int(quotas.sum())
        if shortfall > 0:
            extra = remaining.astype(np.int64) - quotas
            order = np.argsort(-extra)
            fill = 0
            while shortfall > 0 and fill < num_shards:
                j = int(order[fill])
                if extra[j] > 0:
                    quotas[j] += 1
                    extra[j] -= 1
                    shortfall -= 1
                else:
                    fill += 1
        shard_plan = np.concatenate([np.full(int(q), s, dtype=np.int64) for s, q in enumerate(quotas)])
        self.rng.shuffle(shard_plan)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            si = int(shard_plan[bi])
            if not self.start_inds[si]:
                for si2 in range(num_shards):
                    self._reset_shard(si2)
            start_ind = self.start_inds[si].pop()
            mm = _SHARD_STORE.mapped_tokens(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind: start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class ScaleNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        norm_shape = (x.shape[-1],)
        return F.rms_norm(x, norm_shape, eps=self.eps)


class DTypeLinear(nn.Linear):
    def forward(self, x):
        dtype = x.dtype
        w = self.weight.to(dtype=dtype)
        bias = None if self.bias is None else self.bias.to(dtype=dtype)
        return F.linear(x, w, bias)


class PhaseCache(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        cached_ok = (
            self._cos_cached is not None
            and self._sin_cached is not None
            and self._seq_len_cached == seq_len
            and self._cos_cached.device == device
        )
        if not cached_ok:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1.0 / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_phase_rotation(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)


class MultiheadMixer(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, query_gain_seed, train_seq_len):
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
        self.c_q = DTypeLinear(dim, dim, bias=False)
        self.c_k = DTypeLinear(dim, kv_dim, bias=False)
        self.c_v = DTypeLinear(dim, kv_dim, bias=False)
        self.proj = DTypeLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads, 2), query_gain_seed, dtype=torch.float32))
        self.attn_out_gate_width = 12
        self.attn_out_gate_w = nn.Parameter(
            torch.zeros(num_heads, self.attn_out_gate_width, dtype=torch.float32)
        )
        self.rope_dims = 0
        self.rotary = PhaseCache(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _suppress_value_projection(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.shape[-2]
        group = H // Hkv
        y_grouped = y.reshape(B, T, Hkv, group, D)
        v_unit = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_grouped * v_unit).sum(dim=-1, keepdim=True) * v_unit
        cleaned = y_grouped - proj
        return cleaned.reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_phase_rotation(q, cos, sin, self.rope_dims)
        k = apply_phase_rotation(k, cos, sin, self.rope_dims)
        rd = self.rope_dims
        gain = self.q_gain.to(dtype=q.dtype)
        if 0 < rd < q.size(-1):
            q_rope = q[..., :rd] * gain[None, None, :, 0:1]
            q_nope = q[..., rd:] * gain[None, None, :, 1:2]
            q = torch.cat((q_rope, q_nope), dim=-1)
        else:
            q = q * gain[None, None, :, 0:1]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._suppress_value_projection(y, v)
        gate_w = self.attn_out_gate_w.to(dtype=x.dtype)
        g = 2.0 * torch.sigmoid(
            F.linear(x[..., : self.attn_out_gate_width], gate_w)
        )
        y = y * g[..., None]
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class SquaredSiLUFeedForward(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = DTypeLinear(dim, hidden, bias=False)
        self.proj = DTypeLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        hidden = self.fc(x)
        hidden = F.silu(hidden)
        hidden = hidden * hidden
        return self.proj(hidden)


class ResidualMixerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            num_kv_heads,
            mlp_mult,
            rope_base,
            query_gain_seed,
            train_seq_len,
            layer_idx=0,
            ln_scale=False,
    ):
        super().__init__()
        self.attn_norm = ScaleNorm()
        self.mlp_norm = ScaleNorm()
        self.attn = MultiheadMixer(dim, num_heads, num_kv_heads, rope_base, query_gain_seed, train_seq_len)
        self.mlp = SquaredSiLUFeedForward(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        resid_a = mix[0][None, None, :]
        resid_b = mix[1][None, None, :]
        x_in = resid_a * x + resid_b * x0
        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed)
        attn_gain = self.attn_scale.to(dtype=x_in.dtype)[None, None, :]
        mlp_gain = self.mlp_scale.to(dtype=x_in.dtype)[None, None, :]
        if self.parallel:
            mlp_in = self.mlp_norm(x_in) * self.ln_scale_factor
            mlp_out = self.mlp(mlp_in)
            x_out = x_in + attn_gain * attn_out + mlp_gain * mlp_out
        else:
            x_mid = x_in + attn_gain * attn_out
            mlp_in = self.mlp_norm(x_mid) * self.ln_scale_factor
            x_out = x_mid + mlp_gain * self.mlp(mlp_in)
        return x_out


class TokenModel(nn.Module):
    def __init__(self, h):
        super().__init__()
        if h.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
        self.tie_embeddings = h.tie_embeddings
        self.tied_embed_init_std = h.tied_embed_init_std
        self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        self.embed_scale = nn.Parameter(torch.ones(h.embedding_dim, dtype=torch.float32))
        if h.embedding_dim != h.model_dim:
            self.embed_proj = DTypeLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = DTypeLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj = None
        self.num_encoder_layers = h.num_layers // 2
        self.num_decoder_layers = h.num_layers - self.num_encoder_layers
        self.blocks = nn.ModuleList(
            [
                ResidualMixerBlock(
                    h.model_dim,
                    h.num_heads,
                    h.num_kv_heads,
                    h.mlp_mult,
                    h.rope_base,
                    h.query_gain_seed,
                    h.train_seq_len,
                    layer_idx=i,
                    ln_scale=h.ln_scale,
                )
                for i in range(h.num_layers)
            ]
        )
        if h.rope_dims > 0:
            head_dim = h.model_dim // h.num_heads
            for block in self.blocks:
                block.attn.rope_dims = h.rope_dims
                block.attn.rotary = PhaseCache(
                    head_dim, base=h.rope_base, train_seq_len=h.rope_train_seq_len, rope_dims=h.rope_dims
                )
        self.final_norm = ScaleNorm()
        self.lm_head = None if h.tie_embeddings else DTypeLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        if h.cleanup_last_n > 0:
            for i in range(max(0, h.num_layers - h.cleanup_last_n), h.num_layers):
                self.blocks[i].attn.use_xsa = True
        if h.dual_path_from_layer >= 0:
            for i in range(h.dual_path_from_layer, h.num_layers):
                self.blocks[i].parallel = True

        if h.recurrent_passes > 0 and h.ln_scale:
            loop_scale = 1.0 / math.sqrt(h.recurrent_passes + 1)
            for i in range(h.recurrent_begin, h.recurrent_end + 1):
                self.blocks[i].ln_scale_factor = self.blocks[i].ln_scale_factor * loop_scale

        self.looping_active = False
        if h.recurrent_passes > 0:
            loop_seg = list(range(h.recurrent_begin, h.recurrent_end + 1))
            all_indices = list(range(h.recurrent_begin))
            for _ in range(h.recurrent_passes + 1):
                all_indices.extend(loop_seg)
            all_indices.extend(range(h.recurrent_end + 1, h.num_layers))
            num_enc = len(all_indices) // 2
            self.encoder_indices = all_indices[:num_enc]
            self.decoder_indices = all_indices[num_enc:]
        else:
            self.encoder_indices = list(range(self.num_encoder_layers))
            self.decoder_indices = list(range(self.num_encoder_layers, h.num_layers))

        self.num_bridge_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.bridge_weights = nn.Parameter(
            torch.ones(self.num_bridge_weights, h.model_dim, dtype=torch.float32)
        )
        self.bridge_gates = (
            nn.Parameter(torch.zeros(self.num_bridge_weights, h.model_dim, dtype=torch.float32))
            if h.bridge_gates_enabled
            else None
        )
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                    continue
                if module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x * self.embed_scale.to(x.dtype), (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips = []
        enc_iter = (
            self.encoder_indices
            if self.looping_active
            else range(self.num_encoder_layers)
        )
        dec_iter = (
            self.decoder_indices
            if self.looping_active
            else range(self.num_encoder_layers, self.num_encoder_layers + self.num_decoder_layers)
        )
        for i in enc_iter:
            x = self.blocks[i](x, x0)
            skips.append(x)
        for bridge_idx, i in enumerate(dec_iter):
            if bridge_idx < self.num_bridge_weights and skips:
                bridge_weight = self.bridge_weights[bridge_idx].to(dtype=x.dtype)[None, None, :]
                bridge_signal = bridge_weight * skips.pop()
                if self.bridge_gates is not None:
                    g = torch.sigmoid(self.bridge_gates[bridge_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(bridge_signal, x, g)
                else:
                    x = x + bridge_signal
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        flat_logits = logits.reshape(-1, logits.size(-1)).float()
        flat_targets = target_ids.reshape(-1)
        return F.cross_entropy(flat_logits, flat_targets, reduction="mean")


def param_family(name):
    if any(key in name for key in ("tok_emb", "lm_head")):
        return "embed"
    if ".mlp." in name:
        return "mlp"
    is_attention_proj = ".attn." in name or (".proj." in name and ".mlp." not in name)
    if is_attention_proj:
        return "attn"
    return "other"


@torch.compile
def orthogonalize_ns(G, steps=10, eps=1e-07):
    a, b, c = 3.4445, -4.775, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    needs_t = G.shape[0] > G.shape[1]
    if needs_t:
        X = X.transpose(0, 1)
    for _ in range(steps):
        gram = X @ X.transpose(0, 1)
        poly = b * gram + c * (gram @ gram)
        X = a * X + poly @ X
    return X.transpose(0, 1) if needs_t else X


class MatrixOrthoSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                row_normalize=row_normalize,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        maybe_loss = None
        if closure is not None:
            with torch.enable_grad():
                maybe_loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if len(params) == 0:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            total = sum(int(p.numel()) for p in params)
            packed = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            cursor = 0
            for index, param in enumerate(params):
                param_size = int(param.numel())
                should_own = index % world_size == rank and param.grad is not None
                if should_own:
                    grad = param.grad
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buf = state["momentum_buffer"]
                    momentum_buf.mul_(momentum).add_(grad)
                    if nesterov:
                        grad = grad.add(momentum_buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        norms = grad.float().norm(dim=-1, keepdim=True).clamp_min(1e-07)
                        grad = grad / norms.to(grad.dtype)
                    grad = orthogonalize_ns(grad, steps=backend_steps)
                    grad = grad * max(1, grad.shape[0] / grad.shape[1]) ** 0.5
                    packed[cursor:cursor + param_size] = grad.reshape(-1)
                cursor += param_size
            if distributed:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            cursor = 0
            for param in params:
                param_size = int(param.numel())
                if wd > 0.0:
                    param.data.mul_(1.0 - lr * wd)
                update = packed[cursor:cursor + param_size].view_as(param).to(dtype=param.dtype)
                param.add_(update, alpha=-lr)
                cursor += param_size
        return maybe_loss


AUX_PARAM_TAGS = tuple(
    pattern
    for pattern in os.environ.get(
        "AUX_PARAM_TAGS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,bridge_weight,bridge_weights,bridge_gates,attn_out_gate",
    ).split(",")
    if pattern
)


class OptimizerBundle:
    def __init__(self, h, base_model):
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_attn_params = []
        matrix_mlp_params = []
        for (name, p) in block_named_params:
            if p.ndim != 2 or any(pattern in name for pattern in AUX_PARAM_TAGS):
                continue
            if ".mlp." in name:
                matrix_mlp_params.append(p)
            else:
                matrix_attn_params.append(p)
        scalar_params = [
            p
            for (name, p) in block_named_params
            if p.ndim < 2 or any(pattern in name for pattern in AUX_PARAM_TAGS)
        ]
        if base_model.bridge_weights.numel() > 0:
            scalar_params.append(base_model.bridge_weights)
        if base_model.bridge_gates is not None and base_model.bridge_gates.numel() > 0:
            scalar_params.append(base_model.bridge_gates)
        if hasattr(base_model, "embed_scale"):
            scalar_params.append(base_model.embed_scale)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
        self.optimizer_tok = torch.optim.AdamW(
            tok_params, betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True
        )
        self.optimizer_matrix = MatrixOrthoSGD(
            [
                {"params": matrix_attn_params, "weight_decay": h.matrix_wd_attn},
                {"params": matrix_mlp_params, "weight_decay": h.matrix_wd_mlp},
            ],
            lr=h.matrix_lr,
            momentum=h.ortho_momentum,
            backend_steps=h.ortho_steps,
            weight_decay=h.matrix_wd_attn,
            row_normalize=h.matrix_row_normalize,
        )
        for group in self.optimizer_matrix.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2),
            eps=h.adam_eps,
            weight_decay=h.adam_wd,
            fused=True,
        )
        self.optimizers = [self.optimizer_tok, self.optimizer_matrix, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2),
                eps=h.adam_eps,
                fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)
        else:
            self.optimizer_head = None

    def __iter__(self):
        yield from self.optimizers

    def zero_grad_all(self):
        for opt in tuple(self.optimizers):
            opt.zero_grad(set_to_none=True)

    def step(self):
        for opt in tuple(self.optimizers):
            opt.step()
        self.zero_grad_all()


def stabilize_fp32_modules(model):
    for module in model.modules():
        if isinstance(module, DTypeLinear):
            module.float()
    for name, param in model.named_parameters():
        is_control_tensor = param.ndim < 2 or any(pattern in name for pattern in AUX_PARAM_TAGS)
        if is_control_tensor and param.dtype != torch.float32:
            param.data = param.data.float()


def collect_calibration_grams(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            gram = hessians.get(name)
            if gram is None:
                gram = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name] = gram
            gram.addmm_(x.T, x)

        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, DTypeLinear) and module.weight.numel() > 65536:
            cat = param_family(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)

            return hook_fn

        hooks.append(hook_module.register_forward_hook(make_output_hook("tok_emb.weight")))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.sample_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks:
        hook.remove()
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians


def quantize_with_curvature(w, H, clip_sigmas=3.0, clip_range=63, block_size=32):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H.diagonal().mul_(1.01)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s


def parse_lowbit_overrides(spec):
    m = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if item:
            pat, b = item.rsplit(":", 1)
            m[pat.strip()] = int(b)
    return m


def build_packed_state(state_dict, hessians, h):
    lowbit_map = parse_lowbit_overrides(getattr(h, "lowbit_layers", ""))
    result = {}
    meta = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        if "tok_emb" in name:
            bits = h.embed_bits
        else:
            bits = h.matrix_bits
            for pat, b in lowbit_map.items():
                if pat in name:
                    bits = b
                    break
        q, s = quantize_with_curvature(t, hessians[name], clip_sigmas=cs, clip_range=2 ** (bits - 1) - 1)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"quantized(int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta


def recover_packed_state(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        target_dtype = orig.dtype
        if "passthrough" in info:
            passthrough = result[name]
            if passthrough.dtype == torch.float16 and target_dtype in (torch.float32, torch.bfloat16):
                passthrough = passthrough.to(target_dtype)
            out[name] = passthrough
            continue
        q = result[name + ".q"]
        s = result[name + ".scale"]
        if s.ndim:
            scale = s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            out[name] = (q.float() * scale).to(target_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(target_dtype)
    return out


_PACK_MAGIC = b"PBYT"


def _shuffle_bytes(data, stride=2):
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    pieces = [src[pos::stride] for pos in range(stride)]
    payload = np.concatenate(pieces, axis=0)
    return _PACK_MAGIC + bytes((stride,)) + payload.tobytes()


def _unshuffle_bytes(data):
    if len(data) < 5 or data[:4] != _PACK_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = payload.size
    out = np.empty(n, dtype=np.uint8)
    cursor = 0
    for pos in range(stride):
        block = n // stride + (1 if pos < (n % stride) else 0)
        out[pos::stride][:block] = payload[cursor:cursor + block]
        cursor += block
    return out.tobytes()


def pack_blob(data, compressor):
    data = _shuffle_bytes(data)
    if compressor == "brotli":
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")


def unpack_blob(data, compressor):
    if compressor == "brotli":
        import brotli
        return _unshuffle_bytes(brotli.decompress(data))
    raise ValueError(f"Unknown compressor: {compressor!r}")


def export_artifact(h, base_model):
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        model_bytes = os.path.getsize(h.model_path)
        log(f"Serialized model: {model_bytes} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    device = torch.device("cuda", h.local_rank)
    log("quant:collecting Hessians from calibration data...")
    t0 = time.perf_counter()
    calib_loader = SequenceSampler(h, device)
    hessians = collect_calibration_grams(
        base_model,
        calib_loader,
        h,
        device,
        n_calibration_batches=h.quant_calibration_batches,
    )
    log(f"quant:collected {len(hessians)} Hessians in {time.perf_counter() - t0:.1f}s")
    quant_result, quant_meta = build_packed_state(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = pack_blob(quant_raw, h.compressor)
    quant_file_bytes = len(quant_blob)
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f:
            f.write(quant_blob)
        code_bytes = os.path.getsize(__file__)
        total_bytes = code_bytes + quant_file_bytes
        log(f"artifact_bytes: code={code_bytes} model={quant_file_bytes} total={total_bytes}")
        if total_bytes >= MAX_ARTIFACT_BYTES:
            raise RuntimeError(f"artifact too large: {total_bytes}")
    return quant_file_bytes


def import_artifact(h, device):
    eval_model = TokenModel(h).to(device).bfloat16()
    stabilize_fp32_modules(eval_model)
    template_state = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f:
        packed = f.read()
    decoded = unpack_blob(packed, h.compressor)
    quant_state = torch.load(io.BytesIO(decoded), map_location="cpu")
    recovered = recover_packed_state(quant_state["w"], quant_state["m"], template_state)
    eval_model.load_state_dict(recovered, strict=True)
    return eval_model


def compute_bpb_metrics(loss_sum, token_count, byte_count):
    mean_loss = (loss_sum / token_count).item()
    bits_per_token = mean_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    return mean_loss, bits_per_token * tokens_per_byte


def evaluate_regular(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"VAL_BATCH_TOKENS must provide at least one sequence per rank; got "
            f"VAL_BATCH_TOKENS={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, "
            f"GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = total_seqs * h.rank // h.world_size
    seq_end = total_seqs * (h.rank + 1) // h.world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_data.val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (val_data.has_leading_space_lut[tgt_ids] & ~val_data.is_boundary_token_lut[prev_ids]).to(
                dtype=torch.int16
            )
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    model.train()
    return compute_bpb_metrics(val_loss_sum, val_token_count, val_byte_count)


def evaluate_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = total_windows * h.rank // h.world_size
    my_e = total_windows * (h.rank + 1) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi: bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_data.val_tokens[ws: we + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                base_bytes = val_data.base_bytes_lut[tgt].to(torch.float64)
                lead_space = (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(
                    torch.float64
                )
                byte_count += (base_bytes + lead_space).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return compute_bpb_metrics(loss_sum, token_count, byte_count)


def evaluate_adaptive(h, device, val_data, base_model, batch_seqs=32):
    rank = h.rank
    world_size = h.world_size
    seq_len = h.eval_seq_len
    stride = h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    adapt_chunk = h.adapt_chunk_tokens
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride) if ws + context_size < total_tokens]
    num_chunks = (total_tokens + adapt_chunk - 1) // adapt_chunk
    chunk_windows = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        wlen = min(ws + seq_len, total_tokens) - ws
        s = 0 if ws == 0 else context_size
        scored_start = ws + s
        ci = min(scored_start // adapt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)
    log(f"adapt:start chunks={num_chunks} adapt_lr={h.adapt_lr} adapt_epochs={h.adapt_epochs}")
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    _all_adapt_candidates = [p for p in base_model.parameters()]
    adapt_params = [p for p in _all_adapt_candidates if p.numel() >= 10000]
    for p in _all_adapt_candidates:
        p.requires_grad_(p.numel() >= 10000)
    optimizer = torch.optim.SGD(adapt_params, lr=h.adapt_lr, momentum=h.adapt_momentum, nesterov=True)
    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * adapt_chunk
        chunk_end = min((ci + 1) * adapt_chunk, total_tokens)
        my_s = len(windows) * rank // world_size
        my_e = len(windows) * (rank + 1) // world_size
        my_windows = windows[my_s:my_e]
        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi: bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    we = min(ws + seq_len, total_tokens)
                    wlen = we - ws
                    wlens.append(wlen)
                    chunk_tok = val_data.val_tokens[ws: we + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk_tok[:-1]
                    y_batch[i, :wlen] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compiled_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none"
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else context_size
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                    tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(
                        torch.float64)
                    byte_count += tb.sum()
        is_last_chunk = ci == num_chunks - 1
        if not is_last_chunk and h.adapt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = h.adapt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = chunk_seqs * rank // world_size
                my_seq_e = chunk_seqs * (rank + 1) // world_size
                my_chunk_seqs = my_seq_e - my_seq_s
                for _ep in range(h.adapt_epochs):
                    for bs in range(0, my_chunk_seqs, batch_seqs):
                        be = min(bs + batch_seqs, my_chunk_seqs)
                        actual_bs = my_seq_s + bs
                        start_tok = chunk_start + actual_bs * seq_len
                        end_tok = chunk_start + (my_seq_s + be) * seq_len + 1
                        if end_tok > val_data.val_tokens.numel():
                            continue
                        local = val_data.val_tokens[start_tok:end_tok].to(device=device, dtype=torch.int64)
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _lf = base_model.forward_logits(x).reshape(-1, h.vocab_size).float()
                            loss = F.cross_entropy(_lf, y.reshape(-1)) + 1e-5 * torch.logsumexp(_lf, dim=-1).pow(
                                2).mean()
                        loss.backward()
                        if world_size > 1:
                            for p in adapt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(adapt_params, 1.0)
                        optimizer.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return compute_bpb_metrics(loss_sum, token_count, byte_count)


def run_timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    started = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed_ms = 1e3 * (time.perf_counter() - started)
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms")
    return val_loss, val_bpb


def run_training(h, device, val_data):
    base_model = TokenModel(h).to(device).bfloat16()
    stabilize_fp32_modules(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    if h.distributed:
        model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False)
    else:
        model = compiled_model
    log(f"model_params:{sum(p.numel() for p in base_model.parameters())}")
    optimizers = OptimizerBundle(h, base_model)
    train_loader = SequenceSampler(h, device)
    max_wallclock_ms = 1e3 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None

    def progress_fraction(step, elapsed_ms):
        if max_wallclock_ms is not None:
            return elapsed_ms / max(max_wallclock_ms, 1e-09)
        return step / max(h.iterations, 1)

    def schedule_scale(progress):
        if h.warmdown_frac <= 0:
            return 1.0
        warmdown_start = 1.0 - h.warmdown_frac
        if progress < warmdown_start:
            return 1.0
        linear = (1.0 - progress) / h.warmdown_frac
        return max(linear, h.min_lr)

    def ortho_momentum_frac(elapsed_ms):
        if h.ortho_warmup_fraction <= 0 or max_wallclock_ms is None:
            return 1.0
        warmup_ms = max_wallclock_ms * h.ortho_warmup_fraction
        return min(elapsed_ms / max(warmup_ms, 1e-9), 1.0)

    def step_fn(step, lr_scale, elapsed_ms):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.sample_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac_mm = ortho_momentum_frac(elapsed_ms)
        matrix_momentum = (1 - frac_mm) * h.ortho_warmup_start + frac_mm * h.ortho_momentum
        mm_blend = max(lr_scale, 0.25)
        matrix_momentum = mm_blend * matrix_momentum + (1.0 - mm_blend) * h.ortho_warmup_start
        for group in optimizers.optimizer_matrix.param_groups:
            group["momentum"] = matrix_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps > 0:
        initial_model_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(h.warmup_steps):
            step_fn(warmup_step, 1.0, 0.0)
            if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                log(f"warmup_step: {warmup_step + 1}/{h.warmup_steps}")
        if h.recurrent_passes > 0:
            base_model.looping_active = True
            log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
            for warmup_step in range(h.warmup_steps):
                step_fn(warmup_step, 1.0, 0.0)
                if warmup_step <= 5 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == h.warmup_steps:
                    log(f"loop_warmup_step: {warmup_step + 1}/{h.warmup_steps}")
            base_model.looping_active = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        optimizers.zero_grad_all()
        if h.distributed:
            model.require_backward_grad_sync = True
        train_loader = SequenceSampler(h, device)

    ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay = h.ema_decay
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1e3 * (time.perf_counter() - t0)
            val_loss, val_bpb = evaluate_regular(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(
                    f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms "
                    f"step: {step}/{h.iterations}"
                )
            break
        elapsed_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        frac = progress_fraction(step, elapsed_ms)
        scale = schedule_scale(frac)
        if h.recurrent_passes > 0 and not base_model.looping_active and frac >= h.recurrent_enable_frac:
            base_model.looping_active = True
            log(
                f"layer_loop:enabled step:{step} frac:{frac:.3f} "
                f"encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}"
            )
        train_loss = step_fn(step, scale, elapsed_ms)
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)
        step += 1
        approx_training_time_ms = training_time_ms + 1e3 * (time.perf_counter() - t0)
        should_log_train = h.train_log_every > 0 and (
                step <= 5 or step % h.train_log_every == 0 or stop_after_step is not None
        )
        if should_log_train:
            tok_per_sec = step * h.train_batch_tokens / (approx_training_time_ms / 1e3)
            log(
                f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} "
                f"train_time: {approx_training_time_ms / 60000:.1f}m tok/s: {tok_per_sec:.0f}"
            )
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
    log(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    log("ema:applying EMA weights")
    current_state = base_model.state_dict()
    avg_state = {name: t.to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
    base_model.load_state_dict(avg_state, strict=True)
    return base_model, compiled_model


def run_full_pipeline(h, device):
    random.seed(h.seed)
    np.random.seed(h.seed)
    torch.manual_seed(h.seed)
    torch.cuda.manual_seed_all(h.seed)
    val_data = CorpusEvalContext(h, device)
    log(f"train_shards: {len(list(Path(h.datasets_dir).resolve().glob(Path(h.train_files).name)))}")
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")
    base_model, compiled_model = run_training(h, device, val_data)
    torch._dynamo.reset()
    run_timed_eval("fp_eval_after_ema", evaluate_regular, h, device, val_data, compiled_model)
    export_artifact(h, base_model)
    if h.distributed:
        dist.barrier()
    eval_model = import_artifact(h, device)
    if h.recurrent_passes > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    run_timed_eval("packed_eval", evaluate_regular, h, device, val_data, compiled_model)
    if h.sliding_window_enabled:
        run_timed_eval("packed_eval_sliding", evaluate_sliding, h, device, val_data, eval_model)
    if h.adapt_enabled and h.sliding_window_enabled:
        del eval_model, compiled_model
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        adapt_model = import_artifact(h, device)
        if h.recurrent_passes > 0:
            adapt_model.looping_active = True
        run_timed_eval("packed_eval_adapt", evaluate_adaptive, h, device, val_data, adapt_model)
        del adapt_model


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if _ACCUM_WORLD_REFERENCE % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide {_ACCUM_WORLD_REFERENCE} so grad_accum_steps stays integral"
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = ExecutionProfile()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log(100 * "=", console=False)
        log("Run config:", console=True)
        for k, v in sorted(vars(h).items()):
            if not k.startswith("_"):
                log(f"  {k}: {v}", console=True)
        log("=" * 100, console=False)
        log(f"Running Python {sys.version}", console=False)
        log(f"Running PyTorch {torch.__version__}", console=False)
        log(
            subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
            ).stdout,
            console=False,
        )
        log("=" * 100, console=False)
    log(f"attention_backend:{'flash_attn_3' if _USE_FA3 else 'sdpa_fallback'}")
    run_full_pipeline(h, device)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

