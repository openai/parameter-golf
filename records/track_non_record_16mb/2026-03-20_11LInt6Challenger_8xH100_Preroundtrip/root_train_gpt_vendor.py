"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from lowbit_utils import (
    compress_quantized,
    decompress_quantized,
    dequantize_state_dict,
    export_state_dict_without_mtp,
    fake_quantize_tensor,
    load_export_state_dict,
    quantize_state_dict,
    tensor_nbytes,
)
from mlp_variants import build_mlp
from optimizer_variants import Muon, NorMuon, zeropower_via_newtonschulz5
from run_tracking import RunTracker, env_flag, extract_config, git_sha
from validation_utils import build_sentencepiece_luts, eval_val, eval_val_sliding

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    roundtrip_val_every = int(os.environ.get("ROUNDTRIP_VAL_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    save_best_by = os.environ.get("SAVE_BEST_BY", "roundtrip_val_bpb")

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", os.environ.get("TRAIN_SEQ_LEN", "1024")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_hidden = int(os.environ["MLP_HIDDEN"]) if "MLP_HIDDEN" in os.environ else None
    mlp_kind = os.environ.get("MLP_KIND", "relu2")
    swiglu_hidden_mult = float(os.environ.get("SWIGLU_HIDDEN_MULT", 1.333333))
    num_unique_blocks = int(os.environ.get("NUM_UNIQUE_BLOCKS", 0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.80))
    qat_lr_scale = float(os.environ.get("QAT_LR_SCALE", 0.20))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.01))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    optimizer_variant = os.environ.get("OPTIMIZER_VARIANT", "muon")
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.85))
    swa_every_steps = int(os.environ.get("SWA_EVERY_STEPS", 200))

    # Quantization + export.
    serial_compressor = os.environ.get("SERIAL_COMPRESSOR", "zlib")
    weight_quant_bits = int(os.environ.get("WEIGHT_QUANT_BITS", 8))
    embed_quant_bits = int(os.environ.get("EMBED_QUANT_BITS", 8))
    lowbit_name_patterns = tuple(pattern for pattern in os.environ.get("LOWBIT_NAME_PATTERNS", "").split(",") if pattern)
    keep_float_name_patterns = tuple(pattern for pattern in os.environ.get("KEEP_FLOAT_NAME_PATTERNS", "").split(",") if pattern)
    grouped_int8_name_patterns = tuple(pattern for pattern in os.environ.get("GROUPED_INT8_NAME_PATTERNS", "").split(",") if pattern)
    group_size = int(os.environ.get("GROUP_SIZE", 64))
    fp16_embed_export = bool(int(os.environ.get("FP16_EMBED_EXPORT", "0")))
    lowbit_ste = bool(int(os.environ.get("LOWBIT_STE", "0")))
    lowbit_ste_start_frac = float(os.environ.get("LOWBIT_STE_START_FRAC", 0.80))
    lowbit_ste_lr_scale = float(os.environ.get("LOWBIT_STE_LR_SCALE", 0.20))
    lowbit_ste_name_patterns = tuple(pattern for pattern in os.environ.get("LOWBIT_STE_NAME_PATTERNS", "").split(",") if pattern)

    # Run tracking + promotion.
    runs_dir = os.environ.get("RUNS_DIR", "artifacts/runs")
    best_dir = os.environ.get("BEST_DIR", "artifacts/best")
    promote_max_bytes = int(os.environ.get("PROMOTE_MAX_BYTES", 16_000_000))
    promote_metric = os.environ.get("PROMOTE_METRIC", "final_int8_zlib_roundtrip_exact_val_bpb")
    retain_top_k = max(1, int(os.environ.get("RETAIN_TOP_K", "3")))
    keep_nonbest_artifacts = env_flag("KEEP_NONBEST_ARTIFACTS", "0")

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
KEEP_FLOAT_MAX_NUMEL = 65_536
KEEP_FLOAT_STORE_DTYPE = torch.float16
PER_ROW_SCALE_DTYPE = torch.float16
CLIP_PERCENTILE = 99.99984
CLIP_Q = CLIP_PERCENTILE / 100.0


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    return quantize_state_dict(
        state_dict,
        weight_quant_bits=8,
        embed_quant_bits=8,
        lowbit_name_patterns=(),
        keep_float_name_patterns=(),
        grouped_int8_name_patterns=(),
        group_size=64,
        keep_float_max_numel=KEEP_FLOAT_MAX_NUMEL,
        keep_float_fp32_name_patterns=KEEP_FLOAT_FP32_NAME_PATTERNS,
        keep_float_store_dtype=KEEP_FLOAT_STORE_DTYPE,
        per_row_scale_dtype=PER_ROW_SCALE_DTYPE,
        clip_q=CLIP_Q,
        fp16_embed_export=False,
    )


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    return dequantize_state_dict(obj)


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fake_quant = False
        self.fake_quant_bits = 8

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = fake_quantize_tensor(self.weight, num_bits=self.fake_quant_bits, clip_q=CLIP_Q) if self.fake_quant else self.weight
        return F.linear(x, weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
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
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_gain: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int | None,
        mlp_kind: str,
        swiglu_hidden_mult: float,
        rope_base: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp = build_mlp(dim, mlp_mult, mlp_hidden, mlp_kind, swiglu_hidden_mult, CastedLinear)

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        *,
        resid_mix: Tensor,
        attn_scale: Tensor,
        mlp_scale: Tensor,
        q_gain: Tensor,
    ) -> Tensor:
        mix = resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), q_gain=q_gain)
        x = x + attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int | None,
        mlp_kind: str,
        swiglu_hidden_mult: float,
        num_unique_blocks: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int,
        mtp_loss_weight: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.qat_active = False
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_layers = num_layers
        unique_blocks = num_layers if num_unique_blocks <= 0 else num_unique_blocks
        if unique_blocks <= 0 or num_layers % unique_blocks != 0:
            raise ValueError(f"NUM_UNIQUE_BLOCKS={num_unique_blocks} must divide NUM_LAYERS={num_layers}")
        self.num_unique_blocks = unique_blocks
        self.block_map = [i % self.num_unique_blocks for i in range(num_layers)]
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.attn_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
        self.mlp_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
        self.resid_mixes = nn.Parameter(
            torch.stack(
                [torch.stack((torch.ones(model_dim), torch.zeros(model_dim))).float() for _ in range(num_layers)],
                dim=0,
            )
        )
        self.q_gains = nn.Parameter(torch.full((num_layers, num_heads), qk_gain_init, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    mlp_hidden,
                    mlp_kind,
                    swiglu_hidden_mult,
                    rope_base,
                )
                for _ in range(self.num_unique_blocks)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList([CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        for head in self.mtp_heads:
            head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def set_qat_active(self, enabled: bool) -> None:
        self.qat_active = enabled

        for module in self.modules():
            if isinstance(module, CastedLinear):
                module.fake_quant = enabled
                module.fake_quant_bits = 8

    def set_lowbit_ste(self, enabled: bool, name_patterns: tuple[str, ...], num_bits: int) -> None:
        for name, module in self.named_modules():
            if isinstance(module, CastedLinear):
                module.fake_quant = enabled and any(pattern in name for pattern in name_patterns)
                module.fake_quant_bits = num_bits

    def _encode(self, input_ids: Tensor) -> Tensor:
        emb_weight = fake_quantize_tensor(self.tok_emb.weight, num_bits=8, clip_q=CLIP_Q) if self.qat_active else self.tok_emb.weight
        x = F.embedding(input_ids, emb_weight)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[self.block_map[i]](
                x,
                x0,
                resid_mix=self.resid_mixes[i],
                attn_scale=self.attn_scales[i],
                mlp_scale=self.mlp_scales[i],
                q_gain=self.q_gains[i],
            )
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            logical_idx = self.num_encoder_layers + i
            x = self.blocks[self.block_map[logical_idx]](
                x,
                x0,
                resid_mix=self.resid_mixes[logical_idx],
                attn_scale=self.attn_scales[logical_idx],
                mlp_scale=self.mlp_scales[logical_idx],
                q_gain=self.q_gains[logical_idx],
            )
        return self.final_norm(x)

    def _project_logits(self, x: Tensor) -> Tensor:
        if self.tie_embeddings:
            head_weight = fake_quantize_tensor(self.tok_emb.weight, num_bits=8, clip_q=CLIP_Q) if self.qat_active else self.tok_emb.weight
            logits_proj = F.linear(x, head_weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward_per_position(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._encode(input_ids)
        logits = self._project_logits(hidden)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="none").view_as(target_ids)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        hidden = self._encode(input_ids)
        logits = self._project_logits(hidden)
        main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            mtp_loss_sum = hidden.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                offset = k + 1
                if hidden.size(1) <= offset:
                    break
                mtp_logits = self.logit_softcap * torch.tanh(mtp_head(hidden[:, :-offset, :]) / self.logit_softcap)
                mtp_targets = target_ids[:, offset:]
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.size(-1)).float(),
                    mtp_targets.reshape(-1),
                    reduction="mean",
                )
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)
        return main_loss


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    code_bytes = len(code.encode("utf-8"))
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    run_dir = Path(args.runs_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_state_path = run_dir / "selected_state.pt"
    tracker = (
        RunTracker(
            run_id=args.run_id,
            trainer_name="train_gpt.py",
            backend="torch",
            config={**extract_config(args), "distributed": distributed, "world_size": world_size, "local_rank": local_rank},
            runs_dir=args.runs_dir,
            best_dir=args.best_dir,
            promote_max_bytes=args.promote_max_bytes,
            promote_metric=args.promote_metric,
            retain_top_k=args.retain_top_k,
            keep_nonbest_artifacts=args.keep_nonbest_artifacts,
        )
        if master_process
        else None
    )

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    def log0(msg: str, console: bool = True, event_type: str = "log", step: int | None = None, train_time_ms: float | None = None, **payload) -> None:
        if not master_process:
            return
        if tracker is None:
            raise RuntimeError("tracker must be initialized on the master process")
        tracker.log(
            msg,
            console=console,
            event_type=event_type,
            step=step,
            train_time_ms=train_time_ms,
            **payload,
        )

    if master_process and tracker is not None:
        print(tracker.log_path)
    log0(code, console=False, event_type="source_snapshot")
    log0("=" * 100, console=False, event_type="log_separator")
    log0(f"Running Python {sys.version}", console=False, event_type="runtime")
    log0(f"Running PyTorch {torch.__version__}", console=False, event_type="runtime")
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
        event_type="runtime",
    )
    log0("=" * 100, console=False, event_type="log_separator")

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, max(args.train_seq_len, args.eval_seq_len))
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}",
        event_type="tokenizer_setup",
        tokenizer_path=args.tokenizer_path,
    )
    log0(
        f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}",
        event_type="dataset_setup",
        dataset_name=dataset_dir.name,
        train_shards=actual_train_files,
    )
    log0(
        f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}",
        event_type="dataset_setup",
        val_pattern=args.val_files,
        val_tokens=int(val_tokens.numel() - 1),
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        mlp_kind=args.mlp_kind,
        swiglu_hidden_mult=args.swiglu_hidden_mult,
        num_unique_blocks=args.num_unique_blocks,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model
    roundtrip_model = None
    if args.roundtrip_val_every > 0 or args.save_best_by.startswith("roundtrip"):
        roundtrip_model = GPT(
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            mlp_hidden=args.mlp_hidden,
            mlp_kind=args.mlp_kind,
            swiglu_hidden_mult=args.swiglu_hidden_mult,
            num_unique_blocks=args.num_unique_blocks,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
            mtp_num_heads=args.mtp_num_heads,
            mtp_loss_weight=args.mtp_loss_weight,
        ).to(device).bfloat16()
        for module in roundtrip_model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(roundtrip_model)
        roundtrip_model.eval()
    eval_model: nn.Module = model if args.eval_seq_len == args.train_seq_len else base_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    named_params = list(base_model.named_parameters())
    matrix_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and (p.ndim != 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    if args.optimizer_variant == "muon":
        optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    elif args.optimizer_variant == "normuon":
        optimizer_muon = NorMuon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            beta2=args.beta2,
            eps=args.adam_eps,
        )
    else:
        raise ValueError(f"OPTIMIZER_VARIANT must be 'muon' or 'normuon', got {args.optimizer_variant!r}")
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}", event_type="model_setup", model_params=n_params)
    log0(
        f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}",
        event_type="runtime_setup",
        world_size=world_size,
        grad_accum_steps=grad_accum_steps,
    )
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False", event_type="runtime_setup")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads} "
        f"unique_blocks:{base_model.num_unique_blocks} mlp_kind:{args.mlp_kind} mlp_hidden:{args.mlp_hidden}",
        event_type="model_setup",
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_unique_blocks=base_model.num_unique_blocks,
        mlp_kind=args.mlp_kind,
        mlp_hidden=args.mlp_hidden,
        swiglu_hidden_mult=args.swiglu_hidden_mult,
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} optimizer_variant:{args.optimizer_variant}",
        event_type="optimizer_setup",
        tie_embeddings=args.tie_embeddings,
        token_lr=token_lr,
        head_lr=args.head_lr if base_model.lm_head is not None else 0.0,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        optimizer_variant=args.optimizer_variant,
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"eval_seq_len:{args.eval_seq_len} eval_stride:{args.eval_stride} iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}",
        event_type="training_setup",
        train_batch_tokens=args.train_batch_tokens,
        train_seq_len=args.train_seq_len,
        eval_seq_len=args.eval_seq_len,
        eval_stride=args.eval_stride,
        iterations=args.iterations,
        warmup_steps=args.warmup_steps,
        max_wallclock_seconds=args.max_wallclock_seconds,
        roundtrip_val_every=args.roundtrip_val_every,
        save_best_by=args.save_best_by,
        qat_start_frac=args.qat_start_frac,
        qat_lr_scale=args.qat_lr_scale,
    )
    log0(
        f"weight_quant_bits:{args.weight_quant_bits} embed_quant_bits:{args.embed_quant_bits} serial_compressor:{args.serial_compressor}",
        event_type="quantization_setup",
        weight_quant_bits=args.weight_quant_bits,
        embed_quant_bits=args.embed_quant_bits,
        serial_compressor=args.serial_compressor,
        lowbit_name_patterns=args.lowbit_name_patterns,
        keep_float_name_patterns=args.keep_float_name_patterns,
        grouped_int8_name_patterns=args.grouped_int8_name_patterns,
        fp16_embed_export=args.fp16_embed_export,
        lowbit_ste=args.lowbit_ste,
        lowbit_ste_name_patterns=args.lowbit_ste_name_patterns,
    )
    log0(
        f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} swa:{int(args.swa_enabled)}",
        event_type="model_setup",
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        swa_enabled=args.swa_enabled,
    )
    log0(f"seed:{args.seed}", event_type="training_setup", seed=args.seed)

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(
                    f"warmup_step:{warmup_step + 1}/{args.warmup_steps}",
                    event_type="warmup",
                    step=warmup_step + 1,
                )
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    latest_val_loss: float | None = None
    latest_val_bpb: float | None = None
    latest_roundtrip_loss: float | None = None
    latest_roundtrip_bpb: float | None = None
    latest_sliding_loss: float | None = None
    latest_sliding_bpb: float | None = None
    best_selection: dict[str, float | int | str] | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def progress_frac(step: int, elapsed_ms: float) -> float:
        frac_step = step / max(args.iterations, 1)
        frac_time = elapsed_ms / max(max_wallclock_ms, 1.0) if max_wallclock_ms is not None else 0.0
        return min(max(frac_step, frac_time), 1.0)

    def apply_quant_modes(enabled: bool) -> None:
        if args.lowbit_ste:
            base_model.set_lowbit_ste(enabled, args.lowbit_ste_name_patterns, args.weight_quant_bits)
        else:
            base_model.set_qat_active(enabled)

    def export_quantized() -> tuple[dict[str, object], bytes, dict[str, int], int]:
        export_state = export_state_dict_without_mtp(base_model.state_dict())
        quant_obj, quant_stats = quantize_state_dict(
            export_state,
            weight_quant_bits=args.weight_quant_bits,
            embed_quant_bits=args.embed_quant_bits,
            lowbit_name_patterns=args.lowbit_name_patterns,
            keep_float_name_patterns=args.keep_float_name_patterns,
            grouped_int8_name_patterns=args.grouped_int8_name_patterns,
            group_size=args.group_size,
            keep_float_max_numel=KEEP_FLOAT_MAX_NUMEL,
            keep_float_fp32_name_patterns=KEEP_FLOAT_FP32_NAME_PATTERNS,
            keep_float_store_dtype=KEEP_FLOAT_STORE_DTYPE,
            per_row_scale_dtype=PER_ROW_SCALE_DTYPE,
            clip_q=CLIP_Q,
            fp16_embed_export=args.fp16_embed_export,
        )
        quant_blob, quant_raw_bytes = compress_quantized(quant_obj, args.serial_compressor)
        return quant_obj, quant_blob, quant_stats, quant_raw_bytes

    def save_selected_checkpoint(metric_name: str, metric_value: float, step: int, train_time_ms: float) -> None:
        nonlocal best_selection
        candidate = {"metric_name": metric_name, "metric_value": metric_value, "step": step, "train_time_ms": train_time_ms}
        if best_selection is None or float(candidate["metric_value"]) < float(best_selection["metric_value"]):
            if master_process:
                torch.save(base_model.state_dict(), selected_state_path)
            if distributed:
                dist.barrier()
            best_selection = candidate
            log0(
                f"selected_checkpoint step:{step} metric:{metric_name} value:{metric_value:.8f}",
                event_type="checkpoint_selected",
                step=step,
                train_time_ms=train_time_ms,
                selected_metric=metric_name,
                selected_value=metric_value,
            )

    def eval_roundtrip_checkpoint(step: int, train_time_ms: float) -> tuple[float, float, int, int]:
        if roundtrip_model is None:
            raise RuntimeError("roundtrip_model is required for judged checkpointing")
        quant_obj, quant_blob, _, _ = export_quantized()
        bytes_total = len(quant_blob) + code_bytes
        quant_state = decompress_quantized(quant_blob, args.serial_compressor)
        load_export_state_dict(roundtrip_model, dequantize_state_dict(quant_state))
        torch.cuda.synchronize()
        t_q = time.perf_counter()
        q_loss, q_bpb = eval_val(
            val_batch_size=args.val_batch_size,
            eval_seq_len=args.eval_seq_len,
            model=roundtrip_model,
            rank=rank,
            world_size=world_size,
            device=device,
            grad_accum_steps=grad_accum_steps,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        if args.eval_stride > 0:
            q_loss, q_bpb = eval_val_sliding(
                eval_seq_len=args.eval_seq_len,
                eval_stride=args.eval_stride,
                model=roundtrip_model,
                rank=rank,
                world_size=world_size,
                device=device,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
        torch.cuda.synchronize()
        eval_time_ms = int(1000.0 * (time.perf_counter() - t_q))
        log0(
            f"step:{step}/{args.iterations} roundtrip_val_loss:{q_loss:.4f} roundtrip_val_bpb:{q_bpb:.4f} "
            f"bytes_total:{bytes_total} eval_time:{eval_time_ms}ms",
            event_type="roundtrip_validation",
            step=step,
            train_time_ms=train_time_ms,
            roundtrip_val_loss=q_loss,
            roundtrip_val_bpb=q_bpb,
            bytes_total=bytes_total,
            eval_time_ms=eval_time_ms,
        )
        return q_loss, q_bpb, bytes_total, eval_time_ms

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            latest_val_loss, latest_val_bpb = eval_val(
                val_batch_size=args.val_batch_size,
                eval_seq_len=args.eval_seq_len,
                model=eval_model,
                rank=rank,
                world_size=world_size,
                device=device,
                grad_accum_steps=grad_accum_steps,
                val_tokens=val_tokens,
                base_bytes_lut=base_bytes_lut,
                has_leading_space_lut=has_leading_space_lut,
                is_boundary_token_lut=is_boundary_token_lut,
            )
            if args.eval_stride > 0:
                latest_sliding_loss, latest_sliding_bpb = eval_val_sliding(
                    eval_seq_len=args.eval_seq_len,
                    eval_stride=args.eval_stride,
                    model=base_model,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                    val_tokens=val_tokens,
                    base_bytes_lut=base_bytes_lut,
                    has_leading_space_lut=has_leading_space_lut,
                    is_boundary_token_lut=is_boundary_token_lut,
                )
                log0(
                    f"step:{step}/{args.iterations} sliding_val_loss:{latest_sliding_loss:.4f} sliding_val_bpb:{latest_sliding_bpb:.4f}",
                    event_type="validation_sliding",
                    step=step,
                    train_time_ms=training_time_ms,
                    sliding_val_loss=latest_sliding_loss,
                    sliding_val_bpb=latest_sliding_bpb,
                )
            log0(
                f"step:{step}/{args.iterations} val_loss:{latest_val_loss:.4f} val_bpb:{latest_val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms",
                event_type="validation",
                step=step,
                train_time_ms=training_time_ms,
                val_loss=latest_val_loss,
                val_bpb=latest_val_bpb,
            )
            if args.save_best_by == "val_loss":
                save_selected_checkpoint("val_loss", latest_val_loss, step, training_time_ms)
            elif args.save_best_by == "val_bpb":
                save_selected_checkpoint("val_bpb", latest_val_bpb, step, training_time_ms)
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        should_roundtrip_validate = last_step or (
            args.roundtrip_val_every > 0 and step % args.roundtrip_val_every == 0
        )
        if should_roundtrip_validate and roundtrip_model is not None:
            latest_roundtrip_loss, latest_roundtrip_bpb, bytes_total_ckpt, eval_time_ckpt_ms = eval_roundtrip_checkpoint(step, training_time_ms)
            if args.save_best_by == "roundtrip_val_loss":
                save_selected_checkpoint("roundtrip_val_loss", latest_roundtrip_loss, step, training_time_ms)
            elif args.save_best_by == "roundtrip_val_bpb":
                save_selected_checkpoint("roundtrip_val_bpb", latest_roundtrip_bpb, step, training_time_ms)
            log0(
                f"checkpoint_bytes_total:{bytes_total_ckpt} checkpoint_eval_time:{eval_time_ckpt_ms}ms",
                console=False,
                event_type="roundtrip_validation_detail",
                step=step,
                train_time_ms=training_time_ms,
                bytes_total=bytes_total_ckpt,
                eval_time_ms=eval_time_ckpt_ms,
            )
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}",
                    event_type="stopping",
                    step=step,
                    train_time_ms=training_time_ms,
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        lowbit_phase = progress_frac(step, elapsed_ms) >= (args.lowbit_ste_start_frac if args.lowbit_ste else args.qat_start_frac)
        if lowbit_phase:
            scale *= args.lowbit_ste_lr_scale if args.lowbit_ste else args.qat_lr_scale
        apply_quant_modes(lowbit_phase)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"qat:{int(lowbit_phase)}",
                event_type="train",
                step=step,
                train_time_ms=approx_training_time_ms,
                train_loss=train_loss.item(),
                qat_active=lowbit_phase,
            )
        if args.swa_enabled and step > 0 and step % max(args.swa_every_steps, 1) == 0 and progress_frac(step, approx_training_time_ms) >= args.swa_start_frac:
            state = {name: tensor.detach().cpu() for name, tensor in base_model.state_dict().items() if "mtp_heads" not in name}
            if swa_state is None:
                swa_state = state
                swa_count = 1
            else:
                swa_count += 1
                for name, tensor in state.items():
                    swa_state[name].mul_((swa_count - 1) / swa_count).add_(tensor, alpha=1.0 / swa_count)

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
        event_type="memory",
        peak_memory_allocated_mib=torch.cuda.max_memory_allocated() // 1024 // 1024,
        peak_memory_reserved_mib=torch.cuda.max_memory_reserved() // 1024 // 1024,
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    if distributed:
        dist.barrier()
    if selected_state_path.is_file():
        selected_state = torch.load(selected_state_path, map_location="cpu")
        base_model.load_state_dict(selected_state, strict=True)
        if roundtrip_model is not None:
            roundtrip_model.load_state_dict(selected_state, strict=True)
        log0(
            f"loading_selected_checkpoint step:{int(best_selection['step']) if best_selection else -1} "
            f"metric:{best_selection['metric_name'] if best_selection else 'none'} "
            f"value:{float(best_selection['metric_value']) if best_selection else math.nan:.8f}",
            event_type="selected_checkpoint_loaded",
            step=int(best_selection["step"]) if best_selection else None,
            train_time_ms=training_time_ms,
        )
        if distributed:
            dist.barrier()
        if master_process:
            selected_state_path.unlink(missing_ok=True)
    if args.swa_enabled and swa_state is not None:
        load_export_state_dict(base_model, swa_state)
        if roundtrip_model is not None:
            load_export_state_dict(roundtrip_model, swa_state)
        log0(
            f"loading_swa_state count:{swa_count}",
            event_type="swa_loaded",
            step=step,
            train_time_ms=training_time_ms,
            swa_count=swa_count,
        )

    if best_selection is not None or latest_val_loss is None or latest_val_bpb is None:
        latest_val_loss, latest_val_bpb = eval_val(
            val_batch_size=args.val_batch_size,
            eval_seq_len=args.eval_seq_len,
            model=eval_model,
            rank=rank,
            world_size=world_size,
            device=device,
            grad_accum_steps=grad_accum_steps,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        log0(
            f"final_prequant val_loss:{latest_val_loss:.4f} val_bpb:{latest_val_bpb:.4f}",
            event_type="validation",
            step=step,
            train_time_ms=training_time_ms,
            val_loss=latest_val_loss,
            val_bpb=latest_val_bpb,
        )
    if args.eval_stride > 0:
        latest_sliding_loss, latest_sliding_bpb = eval_val_sliding(
            eval_seq_len=args.eval_seq_len,
            eval_stride=args.eval_stride,
            model=base_model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
        log0(
            f"final_sliding_prequant val_loss:{latest_sliding_loss:.4f} val_bpb:{latest_sliding_bpb:.4f}",
            event_type="validation_sliding",
            step=step,
            train_time_ms=training_time_ms,
            sliding_val_loss=latest_sliding_loss,
            sliding_val_bpb=latest_sliding_bpb,
        )

    raw_model_path = run_dir / "final_model.pt"
    quant_tag = f"w{args.weight_quant_bits}e{args.embed_quant_bits}_{args.serial_compressor}"
    quant_model_path = run_dir / f"final_model.{quant_tag}.ptz"
    if master_process:
        torch.save(base_model.state_dict(), raw_model_path)
        model_bytes = raw_model_path.stat().st_size
        log0(f"Serialized model: {model_bytes} bytes", event_type="serialization", raw_model_bytes=model_bytes)
        log0(f"Code size: {code_bytes} bytes", event_type="serialization", code_bytes=code_bytes)
        log0(
            f"Total submission size: {model_bytes + code_bytes} bytes",
            event_type="serialization",
            bytes_total_raw=model_bytes + code_bytes,
        )

    quant_obj, quant_blob, quant_stats, quant_raw_bytes = export_quantized()
    if master_process:
        with quant_model_path.open("wb") as f:
            f.write(quant_blob)
        quant_file_bytes = quant_model_path.stat().st_size
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model {quant_tag}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)",
            event_type="serialization",
            quantized_model_bytes=quant_file_bytes,
            quant_payload_bytes=quant_stats["int8_payload_bytes"],
            quant_raw_bytes=quant_raw_bytes,
            quant_payload_ratio=ratio,
            quant_tag=quant_tag,
        )
        log0(
            f"Total submission size {quant_tag}: {quant_file_bytes + code_bytes} bytes",
            event_type="serialization",
            bytes_total=quant_file_bytes + code_bytes,
        )

    if distributed:
        dist.barrier()
    with quant_model_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_state = decompress_quantized(quant_blob_disk, args.serial_compressor)
    load_export_state_dict(base_model, dequantize_state_dict(quant_state))
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        val_batch_size=args.val_batch_size,
        eval_seq_len=args.eval_seq_len,
        model=eval_model,
        rank=rank,
        world_size=world_size,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    q_slide_loss = None
    q_slide_bpb = None
    if args.eval_stride > 0:
        q_slide_loss, q_slide_bpb = eval_val_sliding(
            eval_seq_len=args.eval_seq_len,
            eval_stride=args.eval_stride,
            model=base_model,
            rank=rank,
            world_size=world_size,
            device=device,
            val_tokens=val_tokens,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    q_eval_ms = int(1000.0 * (time.perf_counter() - t_qeval))
    final_exact_loss = q_slide_loss if q_slide_loss is not None else q_val_loss
    final_exact_bpb = q_slide_bpb if q_slide_bpb is not None else q_val_bpb
    log0(
        f"final_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{q_eval_ms:.0f}ms",
        event_type="final_roundtrip",
        step=step,
        train_time_ms=training_time_ms,
        final_roundtrip_val_loss=q_val_loss,
        final_roundtrip_val_bpb=q_val_bpb,
        final_roundtrip_eval_time_ms=q_eval_ms,
    )
    log0(
        f"final_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}",
        event_type="final_roundtrip_exact",
        step=step,
        train_time_ms=training_time_ms,
        final_roundtrip_exact_val_loss=q_val_loss,
        final_roundtrip_exact_val_bpb=q_val_bpb,
    )
    if q_slide_loss is not None and q_slide_bpb is not None:
        log0(
            f"final_sliding_window_exact val_loss:{q_slide_loss:.8f} val_bpb:{q_slide_bpb:.8f} stride:{args.eval_stride}",
            event_type="final_sliding_window_exact",
            step=step,
            train_time_ms=training_time_ms,
            final_sliding_window_exact_val_loss=q_slide_loss,
            final_sliding_window_exact_val_bpb=q_slide_bpb,
            final_sliding_window_stride=args.eval_stride,
        )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{final_exact_loss:.8f} val_bpb:{final_exact_bpb:.8f}",
        event_type="final_roundtrip_exact_compat",
        step=step,
        train_time_ms=training_time_ms,
        final_int8_zlib_roundtrip_exact_val_loss=final_exact_loss,
        final_int8_zlib_roundtrip_exact_val_bpb=final_exact_bpb,
    )

    if master_process and tracker is not None:
        tracker.finalize(
            summary={
                "git_sha": git_sha(),
                "final_prequant_val_loss": latest_val_loss,
                "final_prequant_val_bpb": latest_val_bpb,
                "final_prequant_sliding_val_loss": latest_sliding_loss,
                "final_prequant_sliding_val_bpb": latest_sliding_bpb,
                "final_roundtrip_exact_val_loss": q_val_loss,
                "final_roundtrip_exact_val_bpb": q_val_bpb,
                "final_sliding_window_exact_val_loss": q_slide_loss,
                "final_sliding_window_exact_val_bpb": q_slide_bpb,
                "final_exact_val_loss": final_exact_loss,
                "final_exact_val_bpb": final_exact_bpb,
                "final_int8_zlib_roundtrip_exact_val_loss": final_exact_loss,
                "final_int8_zlib_roundtrip_exact_val_bpb": final_exact_bpb,
                "raw_model_bytes": raw_model_path.stat().st_size if raw_model_path.exists() else 0,
                "quantized_model_bytes": quant_model_path.stat().st_size if quant_model_path.exists() else 0,
                "bytes_code": code_bytes,
                "bytes_total": (quant_model_path.stat().st_size if quant_model_path.exists() else 0) + code_bytes,
                "train_time_ms": int(training_time_ms),
                "roundtrip_eval_time_ms": q_eval_ms,
                "quant_tag": quant_tag,
                "selection_metric": best_selection["metric_name"] if best_selection else args.save_best_by,
                "selection_metric_value": best_selection["metric_value"] if best_selection else None,
                "raw_model_path": str(raw_model_path),
                "quantized_model_path": str(quant_model_path),
            },
            artifact_paths={
                "raw_model": raw_model_path,
                "quantized_model": quant_model_path,
            },
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
