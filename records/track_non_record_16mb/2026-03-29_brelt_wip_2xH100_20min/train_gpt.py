"""
brelt is a byte-level latent recurrent architecture for compression-first language modeling.
the idea is that the model should not have to think over the full byte/token sequence
at one flat resolution all the time. instead, it compresses local spans into patch latents,
runs recurrent global mixing over a much shorter internal sequence, and then decodes back to
byte predictions

this implementation pulls from a few different directions:
- blt-style byte-to-latent modeling
- trm / rate-distortion style thinking, where the latent structure should pay for itself
- universal-transformer-style shared recurrent depth
- deep supervision to make hierarchical training less awful
- quantization-aware/export-aware tricks so the final checkpoint stays tiny without collapsing

in practice, this file contains:
- raw-byte training on top of sentencepiece shards by decoding tokens back into bytes
- byte-to-patch segmentation
- a lightweight local byte encoder
- patch commit into span latents
- a global recurrent latent prior over a compressed sequence
- a lightweight local decoder back to byte logits
- bits-per-byte evaluation
- export + int8/zlib roundtrip utilities used in the actual runs

if you want the comparison target (adapted openai baseline), that is in `train_gpt.adpt`
"""

from __future__ import annotations

import copy
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
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# hyperparameters
# almost everything interesting in brelt is wired through env vars so runs can
# mutate quickly without turning this file into a giant config parser.
#
# the default profile stays close to the t4-winning setup. when we want to stop
# babying the model and let it spend real compute, we flip BRELT_PROFILE=full.

BRELT_PROFILE = os.environ.get("BRELT_PROFILE", "t4").strip().lower()
BRELT_FULL_PROFILE = BRELT_PROFILE in {"full", "h100", "h100_30m", "scale"}


def env_str(name: str, default: object) -> str:
    return os.environ.get(name, str(default))


def env_int(name: str, default_t4: int, default_full: int | None = None) -> int:
    default = default_t4 if default_full is None or not BRELT_FULL_PROFILE else default_full
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default_t4: float, default_full: float | None = None) -> float:
    default = default_t4 if default_full is None or not BRELT_FULL_PROFILE else default_full
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default_t4: bool, default_full: bool | None = None) -> bool:
    default = default_t4 if default_full is None or not BRELT_FULL_PROFILE else default_full
    return bool(int(os.environ.get(name, "1" if default else "0")))


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_bytes = env_int("VAL_BATCH_BYTES", 262_144, 1_048_576)
    val_max_bytes = env_int("VAL_MAX_BYTES", 0, 16 * 1024 * 1024)
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1600))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_bytes = env_int("TRAIN_BATCH_BYTES", 262_144, 2_097_152)
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    patching_mode = os.environ.get("PATCHING_MODE", "learned")
    patch_size = int(float(os.environ.get("PATCH_SIZE", 6)))
    max_patch_len = env_int("MAX_PATCH_LEN", 24, 64)
    max_patches = env_int("MAX_PATCHES", 0, 192)
    entropy_threshold = float(os.environ.get("ENTROPY_THRESHOLD", 4.0))
    entropy_delta = float(os.environ.get("ENTROPY_DELTA", 0.35))
    entropy_aux_weight = float(os.environ.get("ENTROPY_AUX_WEIGHT", 0.05))
    prior_min_scale = float(os.environ.get("PRIOR_MIN_SCALE", 0.02))

    model_dim = env_int("MODEL_DIM", 1024, 1536)
    num_layers = env_int("NUM_LAYERS", 11, 14)
    num_heads = env_int("NUM_HEADS", 16, 24)
    num_kv_heads = env_int("NUM_KV_HEADS", 4, 8)
    mlp_mult = float(os.environ.get("MLP_MULT", 2.0))
    global_mlp_mult = env_float("GLOBAL_MLP_MULT", mlp_mult, 3.5)
    local_mlp_mult = env_float("LOCAL_MLP_MULT", mlp_mult, 2.5)
    global_mlp_kind = os.environ.get("GLOBAL_MLP_KIND", "lrelu2")
    local_mlp_kind = os.environ.get("LOCAL_MLP_KIND", "relu2")
    latent_rate_weight = float(os.environ.get("LATENT_RATE_WEIGHT", 0.05))
    deep_supervision_decay = float(os.environ.get("DEEP_SUPERVISION_DECAY", 0.5))
    deep_supervision_taps = int(os.environ.get("DEEP_SUPERVISION_TAPS", 2))
    latent_supervision_weight = float(os.environ.get("LATENT_SUPERVISION_WEIGHT", 0.0))
    aux_warmup_steps = int(os.environ.get("AUX_WARMUP_STEPS", 48))
    anti_collapse_strength = float(os.environ.get("ANTI_COLLAPSE_STRENGTH", 0.25))
    anti_collapse_last_n = int(os.environ.get("ANTI_COLLAPSE_LAST_N", 3))
    latent_align_weight = float(os.environ.get("LATENT_ALIGN_WEIGHT", 0.01))
    rope_base = float(os.environ.get("ROPE_BASE", 500000.0))
    global_partial_rope_fraction = float(os.environ.get("GLOBAL_PARTIAL_ROPE_FRACTION", 0.25))
    global_xsa = bool(int(os.environ.get("GLOBAL_XSA", "1")))
    global_xsa_last_n = env_int("GLOBAL_XSA_LAST_N", 3, 4)
    global_xsa_start_step = env_int("GLOBAL_XSA_START_STEP", 96, 32)
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    global_latent_stride = int(os.environ.get("GLOBAL_LATENT_STRIDE", 4))
    global_downsample_kernel = int(os.environ.get("GLOBAL_DOWNSAMPLE_KERNEL", 5))
    global_downsample_expansion = int(os.environ.get("GLOBAL_DOWNSAMPLE_EXPANSION", 2))
    recurrence_start_layers = env_int("RECURRENCE_START_LAYERS", 9, 10)
    recurrence_warmup_steps = int(os.environ.get("RECURRENCE_WARMUP_STEPS", 20))
    depth_scale_min = float(os.environ.get("DEPTH_SCALE_MIN", 0.75))
    adaptive_recurrence = bool(int(os.environ.get("ADAPTIVE_RECURRENCE", "1")))
    adaptive_recurrence_min_scale = env_float("ADAPTIVE_RECURRENCE_MIN_SCALE", 0.75, 0.95)
    adaptive_recurrence_target = env_float("ADAPTIVE_RECURRENCE_TARGET", 0.75, 0.35)
    adaptive_recurrence_super_target = env_float("ADAPTIVE_RECURRENCE_SUPER_TARGET", 24.0, 48.0)
    adaptive_recurrence_count_min = float(os.environ.get("ADAPTIVE_RECURRENCE_COUNT_MIN", 0.85))
    adaptive_recurrence_count_max = float(os.environ.get("ADAPTIVE_RECURRENCE_COUNT_MAX", 1.15))

    local_dim = env_int("LOCAL_DIM", 144, 192)
    local_encoder_layers = env_int("LOCAL_ENCODER_LAYERS", 1, 2)
    local_decoder_layers = int(os.environ.get("LOCAL_DECODER_LAYERS", 2))
    patch_refiner_layers = int(os.environ.get("PATCH_REFINER_LAYERS", 0))
    local_heads = int(os.environ.get("LOCAL_HEADS", 4))
    local_kv_heads = int(os.environ.get("LOCAL_KV_HEADS", 4))
    local_window = int(os.environ.get("LOCAL_WINDOW", 256))
    cross_attn_heads = int(os.environ.get("CROSS_ATTN_HEADS", 4))
    local_conv_kernel = int(os.environ.get("LOCAL_CONV_KERNEL", 5))
    local_conv_expansion = int(os.environ.get("LOCAL_CONV_EXPANSION", 2))
    boundary_head_weight = float(os.environ.get("BOUNDARY_HEAD_WEIGHT", 0.05))
    boundary_add_threshold = float(os.environ.get("BOUNDARY_ADD_THRESHOLD", 0.85))
    boundary_target_patch_size = env_float("BOUNDARY_TARGET_PATCH_SIZE", 8.0, 8.0)
    boundary_rate_weight = env_float("BOUNDARY_RATE_WEIGHT", 0.1, 0.12)
    boundary_smooth_alpha = float(os.environ.get("BOUNDARY_SMOOTH_ALPHA", 0.25))
    boundary_proxy_weight = env_float("BOUNDARY_PROXY_WEIGHT", 0.05, 0.02)
    boundary_bias_eta = env_float("BOUNDARY_BIAS_ETA", 0.5, 0.15)
    boundary_bias_min = float(os.environ.get("BOUNDARY_BIAS_MIN", -8.0))
    boundary_bias_max = env_float("BOUNDARY_BIAS_MAX", 8.0, 6.0)

    entropy_model_dim = int(os.environ.get("ENTROPY_MODEL_DIM", 128))
    entropy_model_layers = int(os.environ.get("ENTROPY_MODEL_LAYERS", 2))
    entropy_model_heads = int(os.environ.get("ENTROPY_MODEL_HEADS", 4))
    entropy_context = int(os.environ.get("ENTROPY_CONTEXT", 128))

    vocab_size = 256
    byte_bos_id = 256

    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.04))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 20))

    decode_chunk_tokens = int(os.environ.get("DECODE_CHUNK_TOKENS", 8192))
    enable_compile = env_bool("ENABLE_COMPILE", False, True)
    qat_enabled = env_bool("QAT_ENABLED", True, False)
    qat_start_step = env_int("QAT_START_STEP", 96, 256)
    qat_bits = int(os.environ.get("QAT_BITS", 8))
    dual_rate_enabled = bool(int(os.environ.get("DUAL_RATE_ENABLED", "1")))
    dual_rate_capacity = env_float("DUAL_RATE_CAPACITY", 1.0, 0.24)
    dual_rate_eta = env_float("DUAL_RATE_ETA", 0.01, 0.02)
    dual_rate_init = float(os.environ.get("DUAL_RATE_INIT", 0.05))
    ddp_find_unused = bool(int(os.environ.get("DDP_FIND_UNUSED", "0")))


# -----------------------------
# muon optimizer
# -----------------------------


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # orthogonalize a 2d update matrix with a fast newton-schulz iteration
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, weight_decay=weight_decay, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if weight_decay > 0:
                    p.add_(p, alpha=-lr * weight_decay)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# dataset decoding utilities
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def parse_sentencepiece_piece_to_bytes(sp: spm.SentencePieceProcessor, token_id: int) -> bytes:
    if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
        return b""
    piece = sp.id_to_piece(token_id)
    if sp.is_byte(token_id):
        match = re.fullmatch(r"<0x([0-9a-fA-F]{2})>", piece)
        if match is None:
            raise ValueError(f"unable to parse byte piece: {piece}")
        return bytes([int(match.group(1), 16)])
    leading_space = piece.startswith("▁")
    if leading_space:
        piece = piece[1:]
    out = piece.encode("utf-8")
    if leading_space:
        out = b" " + out
    return out


def build_token_bytes_table(sp: spm.SentencePieceProcessor) -> list[bytes]:
    return [parse_sentencepiece_piece_to_bytes(sp, token_id) for token_id in range(int(sp.vocab_size()))]


class TokenByteStream:
    # sequentially decodes sentencepiece token shards back into raw bytes
    def __init__(self, pattern: str, token_bytes_table: list[bytes], decode_chunk_tokens: int):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"no files found for pattern: {pattern}")
        self.token_bytes_table = token_bytes_table
        self.decode_chunk_tokens = decode_chunk_tokens
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
        self.buffer = bytearray()

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def _fill(self, min_bytes: int) -> None:
        while len(self.buffer) < min_bytes:
            if self.pos >= self.tokens.numel():
                self._advance_file()
                continue
            end = min(self.pos + self.decode_chunk_tokens, self.tokens.numel())
            token_chunk = self.tokens[self.pos:end].tolist()
            self.pos = end
            self.buffer.extend(b"".join(self.token_bytes_table[int(tok)] for tok in token_chunk))

    def take(self, n: int) -> Tensor:
        self._fill(n)
        arr = np.frombuffer(memoryview(self.buffer)[:n], dtype=np.uint8).copy()
        del self.buffer[:n]
        return torch.from_numpy(arr)


class DistributedByteLoader:
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        token_bytes_table: list[bytes],
        decode_chunk_tokens: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenByteStream(pattern, token_bytes_table, decode_chunk_tokens)

    def next_batch(self, global_bytes: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_bytes = global_bytes // (self.world_size * grad_accum_steps)
        per_rank_span = local_bytes + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def decode_tokens_to_bytes(tokens: Tensor, token_bytes_table: list[bytes], decode_chunk_tokens: int) -> Tensor:
    buf = bytearray()
    flat = tokens.tolist()
    for i in range(0, len(flat), decode_chunk_tokens):
        chunk = flat[i : i + decode_chunk_tokens]
        buf.extend(b"".join(token_bytes_table[int(tok)] for tok in chunk))
    arr = np.frombuffer(buf, dtype=np.uint8).copy()
    return torch.from_numpy(arr.astype(np.int64, copy=False))


def load_validation_bytes(pattern: str, seq_len: int, token_bytes_table: list[bytes], decode_chunk_tokens: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"no files found for pattern: {pattern}")
    parts = [decode_tokens_to_bytes(load_data_shard(file), token_bytes_table, decode_chunk_tokens) for file in files]
    bytes_all = torch.cat(parts).contiguous()
    usable = ((bytes_all.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return bytes_all[: usable + 1]


# -----------------------------
# quantization utils
# -----------------------------


CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weight",
    "norm",
    "bos",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
# quantization / export helpers
# keep sensitive tensors in float when needed, rotate large matrices into a
# friendlier geometry, then pack the rest down to int8
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
TURBOQUANT_ROTATE_DEFAULT = bool(int(os.environ.get("TURBOQUANT_ROTATE", "1")))
TURBOQUANT_MIN_COLS = int(os.environ.get("TURBOQUANT_MIN_COLS", 64))


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht_last_dim(x: Tensor) -> Tensor:
    n = x.size(-1)
    if n & (n - 1):
        raise ValueError("fwht requires power-of-two last dimension")
    y = x
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], -1, 2 * h)
        a = y[..., :h]
        b = y[..., h:]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*x.shape[:-1], n)
        h *= 2
    return y


def turboquant_rotation_seed(name: str) -> int:
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def turboquant_sign_perm(name: str, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(turboquant_rotation_seed(name))
    signs = torch.randint(0, 2, (width,), generator=gen, dtype=torch.int64)
    signs = torch.where(signs == 0, -torch.ones(width, dtype=torch.int64), torch.ones(width, dtype=torch.int64)).to(device=device, dtype=torch.float32)
    perm = torch.randperm(width, generator=gen, dtype=torch.int64).to(device=device)
    return signs, perm


def turboquant_rotate_matrix(name: str, t: Tensor) -> tuple[Tensor, dict[str, object]]:
    # the point here is not to be the google paper (it's cool though), just to steal the useful
    # part: spread energy across dimensions before per-row quantization so weird
    # spiky channels do not get wrecked by int8
    rows, cols = t.shape
    padded_cols = next_power_of_two(cols)
    x = t.float()
    if padded_cols != cols:
        x = F.pad(x, (0, padded_cols - cols))
    signs, perm = turboquant_sign_perm(name, padded_cols, x.device)
    x = x * signs.unsqueeze(0)
    x = x.index_select(1, perm)
    x = fwht_last_dim(x) / math.sqrt(float(padded_cols))
    meta: dict[str, object] = {
        "scheme": "per_row_hadamard",
        "axis": 0,
        "orig_cols": cols,
        "padded_cols": padded_cols,
    }
    return x.to(dtype=t.dtype), meta


def turboquant_inverse_rotate_matrix(name: str, t: Tensor, orig_cols: int, padded_cols: int) -> Tensor:
    x = t.float()
    if x.size(1) != padded_cols:
        raise ValueError("unexpected padded width in inverse turboquant rotation")
    signs, perm = turboquant_sign_perm(name, padded_cols, x.device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(padded_cols, device=x.device, dtype=perm.dtype)
    x = fwht_last_dim(x) / math.sqrt(float(padded_cols))
    x = x.index_select(1, inv_perm)
    x = x * signs.unsqueeze(0)
    return x[:, :orig_cols]


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # control tensors are kept safer because tiny numerical changes there can
    # swing segmentation / recurrence behavior a lot more than a normal weight mat
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        quant_tensor = t
        tensor_meta: dict[str, object] | None = None
        if TURBOQUANT_ROTATE_DEFAULT and t.ndim == 2 and t.shape[1] >= TURBOQUANT_MIN_COLS:
            quant_tensor, tensor_meta = turboquant_rotate_matrix(name, t)
        q, s = quantize_float_tensor(quant_tensor)
        if tensor_meta is not None:
            qmeta[name] = tensor_meta
        elif s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
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


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        if meta.get("scheme") in {"per_row", "per_row_hadamard"} or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            t = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).contiguous()
            if meta.get("scheme") == "per_row_hadamard":
                t = turboquant_inverse_rotate_matrix(name, t, int(meta["orig_cols"]), int(meta["padded_cols"]))
            out[name] = t.to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# model helpers
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, eps=self.eps)


def fake_quantize_weight_per_row(w: Tensor, bits: int) -> Tensor:
    if bits >= 16 or w.ndim < 2:
        return w
    qmax = float((1 << (bits - 1)) - 1)
    w32 = w.float()
    clip = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = clip / qmax
    q = torch.clamp(torch.round(w32 / scale), -qmax, qmax)
    return (q * scale).to(dtype=w.dtype)


class CastedLinear(nn.Linear):
    qat_enabled = False
    qat_start_step = 0
    qat_bits = 8
    qat_current_step = 0

    @classmethod
    def set_qat_context(cls, enabled: bool, start_step: int, bits: int, current_step: int) -> None:
        cls.qat_enabled = enabled
        cls.qat_start_step = start_step
        cls.qat_bits = bits
        cls.qat_current_step = current_step

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = self.weight
        if self.training and self.qat_enabled and self.qat_current_step >= self.qat_start_step:
            weight = fake_quantize_weight_per_row(weight, self.qat_bits)
        return F.linear(x, weight.to(x.dtype), bias)


class Rotary(nn.Module):
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


def apply_partial_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dim: int) -> Tensor:
    if rope_dim <= 0:
        return x
    rotated = apply_rotary_emb(x[..., :rope_dim], cos, sin)
    if rope_dim >= x.size(-1):
        return rotated
    return torch.cat((rotated, x[..., rope_dim:]), dim=-1)


def make_attn_bias(mask: Tensor, dtype: torch.dtype) -> Tensor:
    return torch.where(mask[:, None, :, :], torch.zeros((), device=mask.device, dtype=dtype), torch.full((), -1e4, device=mask.device, dtype=dtype))


def build_causal_mask(seq_len: int, device: torch.device, window: int | None = None) -> Tensor:
    q = torch.arange(seq_len, device=device)[:, None]
    k = torch.arange(seq_len, device=device)[None, :]
    mask = q >= k
    if window is not None and window > 0:
        mask &= (q - k) < window
    return mask


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        rope_fraction: float = 1.0,
        xsa: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.xsa = xsa
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rope")
        rope_dim = int(self.head_dim * rope_fraction)
        rope_dim = max(0, min(self.head_dim, rope_dim - (rope_dim % 2)))
        self.rope_dim = rope_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.rope_dim, base=rope_base) if self.rope_dim > 0 else None

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        self_v = v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        if self.rotary is not None:
            cos, sin = self.rotary(seqlen, x.device, q.dtype)
            q = apply_partial_rotary_emb(q, cos, sin, self.rope_dim)
            k = apply_partial_rotary_emb(k, cos, sin, self.rope_dim)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        else:
            bias = make_attn_bias(attn_mask, q.dtype)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False, enable_gqa=(self.num_kv_heads != self.num_heads))
            valid_queries = attn_mask.any(dim=-1)[:, None, :, None]
            y = y * valid_queries.to(dtype=y.dtype)
        if self.xsa:
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                self_v = self_v.repeat_interleave(repeat, dim=1)
            self_v = self_v.to(dtype=y.dtype)
            proj = (y * self_v).sum(dim=-1, keepdim=True) / self_v.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            y = y - proj * self_v
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.wq = CastedLinear(dim, dim, bias=False)
        self.wk = CastedLinear(dim, dim, bias=False)
        self.wv = CastedLinear(dim, dim, bias=False)
        self.wo = CastedLinear(dim, dim, bias=False)
        self.wo._zero_init = True

    def forward(self, q_in: Tensor, kv_in: Tensor, attn_mask: Tensor) -> Tensor:
        bsz, qlen, _ = q_in.shape
        _, klen, _ = kv_in.shape
        q = self.wq(self.norm_q(q_in)).view(bsz, qlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(self.norm_kv(kv_in)).view(bsz, klen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(self.norm_kv(kv_in)).view(bsz, klen, self.num_heads, self.head_dim).transpose(1, 2)
        bias = make_attn_bias(attn_mask, q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False)
        valid_queries = attn_mask.any(dim=-1)[:, None, :, None]
        y = y * valid_queries.to(dtype=y.dtype)
        y = y.transpose(1, 2).contiguous().reshape(bsz, qlen, self.dim)
        return self.wo(y)


class ReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(dim * mlp_mult)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class LeakyReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, negative_slope: float = 0.5):
        super().__init__()
        hidden = int(dim * mlp_mult)
        self.negative_slope = negative_slope
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=self.negative_slope)
        return self.proj(x.square())


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = max(1, int(dim * mlp_mult * (2.0 / 3.0)))
        self.w1 = CastedLinear(dim, hidden, bias=False)
        self.w2 = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        mlp_kind: str,
        rope_fraction: float = 1.0,
        xsa: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, num_kv_heads, rope_base, rope_fraction=rope_fraction, xsa=xsa)
        if mlp_kind == "swiglu":
            self.mlp = SwiGLUMLP(dim, mlp_mult)
        elif mlp_kind == "lrelu2":
            self.mlp = LeakyReluSquaredMLP(dim, mlp_mult)
        elif mlp_kind == "relu2":
            self.mlp = ReluSquaredMLP(dim, mlp_mult)
        else:
            raise ValueError(f"unknown mlp_kind: {mlp_kind}")

    def forward(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
        x = x + self.attn(self.attn_norm(x), attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class CausalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, expansion: int = 2):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        self.pad = kernel_size - 1
        hidden = dim * expansion
        self.conv = nn.Conv1d(dim, hidden, kernel_size, bias=False)
        self.proj = nn.Conv1d(hidden, dim, 1, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = F.pad(y, (self.pad, 0))
        y = F.gelu(self.conv(y))
        y = self.proj(y).transpose(1, 2)
        return residual + self.norm(y)


class CausalDownsampleBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, stride: int, expansion: int = 2):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if expansion < 1:
            raise ValueError("expansion must be positive")
        self.pad = kernel_size - 1
        self.stride = stride
        hidden = dim * expansion
        self.conv = nn.Conv1d(dim, hidden, kernel_size, stride=stride, bias=False)
        self.proj = nn.Conv1d(hidden, dim, 1, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        y = x.transpose(1, 2)
        y = F.pad(y, (self.pad, 0))
        y = F.gelu(self.conv(y))
        y = self.proj(y).transpose(1, 2)
        return self.norm(y)


class TinyEntropyModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        dim = args.entropy_model_dim
        self.bos = nn.Parameter(torch.zeros(dim))
        self.byte_emb = nn.Embedding(args.vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, args.entropy_model_heads, args.entropy_model_heads, 2.0, args.rope_base, "relu2")
                for _ in range(args.entropy_model_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.head = CastedLinear(dim, args.vocab_size, bias=False)
        self.local_window = args.entropy_context

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.byte_emb(tokens)
        x = x.roll(shifts=1, dims=1)
        x[:, 0] = self.bos.to(dtype=x.dtype)
        mask = build_causal_mask(tokens.size(1), tokens.device, self.local_window)[None, :, :]
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.norm(x))


# patching helpers
# patching decides how much sequence the global core actually has to process

def entropy_from_logits(logits: Tensor) -> Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def is_space_like_byte(tokens: Tensor) -> Tensor:
    return (
        (tokens < ord("0"))
        | ((ord("9") < tokens) & (tokens < ord("A")))
        | ((ord("Z") < tokens) & (tokens < ord("a")))
        | ((ord("z") < tokens) & (tokens < 0b1000_0000))
        | (0b1100_0000 <= tokens)
    )


def apply_max_patch_len(start_mask: Tensor, max_patch_len: int) -> Tensor:
    if max_patch_len <= 0:
        return start_mask
    out = start_mask.clone()
    bsz, seqlen = out.shape
    run = torch.zeros(bsz, dtype=torch.long, device=out.device)
    for t in range(seqlen):
        if t == 0:
            out[:, t] = True
            run.fill_(1)
            continue
        force = run >= max_patch_len
        out[:, t] |= force
        run = torch.where(out[:, t], torch.ones_like(run), run + 1)
    return out


def apply_max_patch_count(start_mask: Tensor, scores: Tensor, max_patches: int) -> Tensor:
    # safety valve: if learned segmentation gets too excited, keep only the
    # strongest proposed boundaries so one bad microbatch does not blow up memory
    if max_patches <= 0:
        return start_mask
    out = start_mask.clone()
    bsz, seqlen = out.shape
    for b in range(bsz):
        out[b, 0] = True
        current = torch.nonzero(out[b, 1:], as_tuple=False).flatten() + 1
        if current.numel() + 1 <= max_patches:
            continue
        keep = max(max_patches - 1, 0)
        new_mask = torch.zeros(seqlen, dtype=torch.bool, device=out.device)
        new_mask[0] = True
        if keep > 0:
            top_idx = torch.topk(scores[b, current], k=keep, sorted=False).indices
            new_mask[current[top_idx]] = True
        out[b] = new_mask
    return out


def patch_ids_from_starts(start_mask: Tensor) -> tuple[Tensor, Tensor, int]:
    patch_ids = torch.cumsum(start_mask.to(torch.long), dim=1) - 1
    patch_counts = start_mask.sum(dim=1)
    num_patches = int(patch_counts.max().item()) if patch_counts.numel() else 0
    return patch_ids, patch_counts, num_patches


def pool_patches(x: Tensor, patch_ids: Tensor, num_patches: int) -> tuple[Tensor, Tensor]:
    bsz, _, dim = x.shape
    pooled = x.new_zeros(bsz, num_patches, dim)
    pooled.scatter_add_(1, patch_ids.unsqueeze(-1).expand(-1, -1, dim), x)
    counts = x.new_zeros(bsz, num_patches)
    counts.scatter_add_(1, patch_ids, x.new_ones(patch_ids.shape, dtype=x.dtype))
    mean = pooled / counts.clamp_min(1.0).unsqueeze(-1)
    valid = counts > 0

    positions = torch.arange(x.size(1), device=x.device, dtype=torch.long)[None, :].expand(bsz, -1)
    last_pos = torch.full((bsz, num_patches), -1, device=x.device, dtype=torch.long)
    last_pos.scatter_reduce_(1, patch_ids, positions, reduce="amax", include_self=True)
    gather_idx = last_pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, dim)
    last = x.gather(1, gather_idx)
    last = last * valid.unsqueeze(-1)

    pooled = 0.5 * mean + 0.5 * last
    return pooled, valid


def build_patch_membership_mask(patch_ids: Tensor, num_patches: int) -> Tensor:
    patch_index = torch.arange(num_patches, device=patch_ids.device)[None, :, None]
    return patch_index == patch_ids[:, None, :]


def build_decoder_patch_mask(group_ids: Tensor, num_groups: int, include_current: bool = False) -> Tensor:
    group_index = torch.arange(num_groups, device=group_ids.device)[None, None, :]
    if include_current:
        return group_index <= group_ids[:, :, None]
    # bytes in group j may only attend to latent states strictly before j.
    # allowing the current group would leak future bytes when latent states are
    # pooled from a full hard span during training
    return group_index < group_ids[:, :, None]


def build_patch_causal_mask(valid: Tensor, include_self: bool = True) -> Tensor:
    num_patches = valid.size(1)
    diagonal = 0 if include_self else -1
    causal = torch.tril(torch.ones((num_patches, num_patches), dtype=torch.bool, device=valid.device), diagonal=diagonal)
    return causal[None, :, :] & valid[:, :, None] & valid[:, None, :]


def latent_alignment_loss(states: list[Tensor], valid: Tensor) -> Tensor:
    if len(states) < 3:
        return states[0].new_zeros(()) if states else valid.new_zeros((), dtype=torch.float32)
    deltas = [states[i + 1] - states[i] for i in range(len(states) - 1)]
    losses = []
    for i in range(len(deltas) - 1):
        a = F.normalize(deltas[i].float(), dim=-1, eps=1e-6)
        b = F.normalize(deltas[i + 1].float(), dim=-1, eps=1e-6)
        cosine = (a * b).sum(dim=-1)
        losses.append((1.0 - cosine)[valid].mean())
    return torch.stack(losses).mean()


def latent_supervision_loss(states: list[Tensor], final_state: Tensor, valid: Tensor, decay: float) -> Tensor:
    if not states:
        return final_state.new_zeros(())
    target = final_state.detach()
    losses = []
    weights = []
    for idx, state in enumerate(states):
        losses.append(F.smooth_l1_loss(state[valid].float(), target[valid].float(), reduction="mean"))
        weights.append(decay ** (len(states) - 1 - idx))
    weight_sum = max(sum(weights), 1e-8)
    return sum(w * l for w, l in zip(weights, losses, strict=True)) / weight_sum


# main model
# local bytes -> patch latents -> compressed super latents -> recurrent global
# mixing -> bridge back down -> byte logits again
class ByteLatentTransformer(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.train_step = 0
        self.rate_lambda = float(args.dual_rate_init)
        self.last_rate_total = 0.0
        self.last_patch_count = 0.0
        self.last_super_count = 0.0
        self.last_avg_patch_len = 0.0
        self.last_boundary_rate = 0.0
        self.last_hard_boundary_rate = 0.0
        self.last_latent_rate = 0.0
        self.last_active_layers = 0
        self.last_compute_proxy = 0.0
        self.byte_emb = nn.Embedding(args.vocab_size, args.local_dim)
        self.local_conv = CausalConvBlock(args.local_dim, args.local_conv_kernel, args.local_conv_expansion)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerBlock(args.local_dim, args.local_heads, args.local_kv_heads, args.local_mlp_mult, args.rope_base, args.local_mlp_kind)
                for _ in range(args.local_encoder_layers)
            ]
        )
        self.encoder_cross = CrossAttention(args.local_dim, args.cross_attn_heads)
        self.boundary_head = CastedLinear(args.local_dim, 1, bias=False)
        target_rate = 1.0 / max(args.boundary_target_patch_size, 1.0)
        target_rate = min(max(target_rate, 1e-4), 1.0 - 1e-4)
        self.boundary_bias = nn.Parameter(torch.tensor(math.log(target_rate / (1.0 - target_rate)), dtype=torch.float32))
        self.encoder_to_global = CastedLinear(args.local_dim, args.model_dim, bias=False)
        self.global_downsample = CausalDownsampleBlock(
            args.local_dim,
            args.global_downsample_kernel,
            args.global_latent_stride,
            args.global_downsample_expansion,
        )
        self.step_embed = nn.Embedding(args.num_layers, args.model_dim)
        self.global_prior_norm = RMSNorm(args.model_dim)
        self.global_prior = SelfAttention(
            args.model_dim,
            args.num_heads,
            args.num_kv_heads,
            args.rope_base,
            rope_fraction=args.global_partial_rope_fraction,
        )
        self.prior_mean = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.prior_log_scale = CastedLinear(args.model_dim, args.model_dim, bias=False)
        self.prior_mean._zero_init = True
        self.prior_log_scale._zero_init = True
        self.global_block = TransformerBlock(
            args.model_dim,
            args.num_heads,
            args.num_kv_heads,
            args.global_mlp_mult,
            args.rope_base,
            args.global_mlp_kind,
            rope_fraction=args.global_partial_rope_fraction,
            xsa=False,
        )
        self.step_gain = nn.Parameter(torch.ones(args.num_layers))
        self.global_anti_norm = RMSNorm(args.model_dim)
        self.global_norm = RMSNorm(args.model_dim)
        self.global_to_local = CastedLinear(args.model_dim, args.local_dim, bias=False)
        self.super_bridge_proj = CastedLinear(args.local_dim, args.local_dim, bias=False)
        self.patch_bridge_gate = CastedLinear(args.local_dim * 2, args.local_dim, bias=False)
        self.patch_bridge_gate._zero_init = True
        self.super_bridge_gate = CastedLinear(args.local_dim * 2, args.local_dim, bias=False)
        self.super_bridge_gate._zero_init = True
        commit_in_dim = args.local_dim * 4 + 1
        self.patch_token_score = CastedLinear(args.local_dim, 1, bias=False)
        self.patch_commit = CastedLinear(commit_in_dim, args.local_dim, bias=False)
        self.patch_commit_gate = CastedLinear(commit_in_dim, args.local_dim, bias=False)
        self.patch_commit_gate._zero_init = True
        self.patch_refiner = nn.ModuleList(
            [
                TransformerBlock(args.local_dim, args.local_heads, args.local_kv_heads, args.local_mlp_mult, args.rope_base, args.local_mlp_kind)
                for _ in range(args.patch_refiner_layers)
            ]
        )
        self.decoder_cross = nn.ModuleList([CrossAttention(args.local_dim, args.cross_attn_heads) for _ in range(args.local_decoder_layers)])
        self.decoder_gates = nn.ModuleList([CastedLinear(args.local_dim * 2, args.local_dim, bias=False) for _ in range(args.local_decoder_layers)])
        for gate in self.decoder_gates:
            gate._zero_init = True
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(args.local_dim, args.local_heads, args.local_kv_heads, args.local_mlp_mult, args.rope_base, args.local_mlp_kind)
                for _ in range(args.local_decoder_layers)
            ]
        )
        self.final_norm = RMSNorm(args.local_dim)
        self.byte_head = CastedLinear(args.local_dim, args.vocab_size, bias=False)
        self.entropy_model = TinyEntropyModel(args) if args.patching_mode == "entropy" else None
        self.logit_softcap = args.logit_softcap

        self._local_mask_cache: dict[tuple[torch.device, int, int | None], Tensor] = {}
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.step_embed.weight)

    def set_train_step(self, step: int) -> None:
        self.train_step = max(step, 0)
        CastedLinear.set_qat_context(self.args.qat_enabled, self.args.qat_start_step, self.args.qat_bits, self.train_step)

    def _aux_ramp(self) -> float:
        if not self.training or self.args.aux_warmup_steps <= 0:
            return 1.0
        return min(float(self.train_step) / float(self.args.aux_warmup_steps), 1.0)

    def _active_num_layers(self) -> int:
        if not self.training or self.args.recurrence_warmup_steps <= 0:
            return self.args.num_layers
        start_layers = max(1, min(self.args.recurrence_start_layers, self.args.num_layers))
        progress = min(float(self.train_step) / float(self.args.recurrence_warmup_steps), 1.0)
        layers = start_layers + progress * (self.args.num_layers - start_layers)
        return max(1, min(int(round(layers)), self.args.num_layers))

    def _adaptive_active_num_layers(self, surprise: Tensor, super_count: Tensor) -> int:
        layers = self._active_num_layers()
        if not self.training or not self.args.adaptive_recurrence:
            return layers
        surprise_value = float(surprise.float().mean().item())
        target = max(self.args.adaptive_recurrence_target, 1e-6)
        surprise_scale = math.sqrt(max(surprise_value, 1e-6) / target)
        surprise_scale = min(max(surprise_scale, self.args.adaptive_recurrence_min_scale), 1.25)
        super_value = max(float(super_count.float().mean().item()), 1.0)
        count_scale = math.sqrt(max(self.args.adaptive_recurrence_super_target, 1e-6) / super_value)
        count_scale = min(max(count_scale, self.args.adaptive_recurrence_count_min), self.args.adaptive_recurrence_count_max)
        scale = surprise_scale * count_scale
        return max(1, min(int(round(layers * scale)), self.args.num_layers))

    def _local_mask(self, device: torch.device, seq_len: int) -> Tensor:
        key = (device, seq_len, self.args.local_window)
        mask = self._local_mask_cache.get(key)
        if mask is None:
            mask = build_causal_mask(seq_len, device, self.args.local_window)[None, :, :]
            self._local_mask_cache[key] = mask
        return mask

    def _commit_patches(self, x: Tensor, patch_ids: Tensor, num_patches: int) -> tuple[Tensor, Tensor]:
        # a patch latent should feel like the span committed some state, not just
        # that we averaged a few tokens and hoped for the best
        bsz, seqlen, dim = x.shape
        summed = x.new_zeros(bsz, num_patches, dim)
        summed.scatter_add_(1, patch_ids.unsqueeze(-1).expand(-1, -1, dim), x)
        counts = x.new_zeros(bsz, num_patches)
        counts.scatter_add_(1, patch_ids, x.new_ones(patch_ids.shape, dtype=x.dtype))
        valid = counts > 0

        token_scores = self.patch_token_score(F.rms_norm(x, (dim,))).squeeze(-1).float()
        patch_max = torch.full((bsz, num_patches), -1e9, device=x.device, dtype=torch.float32)
        patch_max.scatter_reduce_(1, patch_ids, token_scores, reduce="amax", include_self=True)
        centered = token_scores - patch_max.gather(1, patch_ids)
        token_weights = centered.exp()
        weight_sums = torch.zeros((bsz, num_patches), device=x.device, dtype=torch.float32)
        weight_sums.scatter_add_(1, patch_ids, token_weights)
        weighted = torch.zeros((bsz, num_patches, dim), device=x.device, dtype=torch.float32)
        weighted.scatter_add_(1, patch_ids.unsqueeze(-1).expand(-1, -1, dim), x.float() * token_weights.unsqueeze(-1))
        summary = (weighted / weight_sums.clamp_min(1e-6).unsqueeze(-1)).to(dtype=x.dtype)

        positions = torch.arange(seqlen, device=x.device, dtype=torch.long)[None, :].expand(bsz, -1)
        first_pos = torch.full((bsz, num_patches), seqlen, device=x.device, dtype=torch.long)
        first_pos.scatter_reduce_(1, patch_ids, positions, reduce="amin", include_self=True)
        last_pos = torch.full((bsz, num_patches), -1, device=x.device, dtype=torch.long)
        last_pos.scatter_reduce_(1, patch_ids, positions, reduce="amax", include_self=True)

        first_idx = first_pos.clamp_max(seqlen - 1).unsqueeze(-1).expand(-1, -1, dim)
        last_idx = last_pos.clamp_min(0).unsqueeze(-1).expand(-1, -1, dim)
        first = x.gather(1, first_idx) * valid.unsqueeze(-1)
        last = x.gather(1, last_idx) * valid.unsqueeze(-1)
        delta = last - first
        length_feat = (torch.log1p(counts) / math.log(max(self.args.max_patch_len, 2) + 1.0)).unsqueeze(-1)
        commit_in = torch.cat(
            (
                F.rms_norm(first, (dim,)),
                F.rms_norm(summary, (dim,)),
                F.rms_norm(last, (dim,)),
                F.rms_norm(delta, (dim,)),
                length_feat,
            ),
            dim=-1,
        )
        commit_gate = torch.sigmoid(self.patch_commit_gate(commit_in))
        commit_update = self.patch_commit(commit_in)
        committed = (last + commit_gate * commit_update) * valid.unsqueeze(-1)
        return committed, valid

    def _patch_starts(self, tokens: Tensor, local_states: Optional[Tensor] = None) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor, Tensor | None]:
        # all segmentation modes end up here. learned is the interesting one, but
        # the fallback modes still matter because they let us keep the system honest
        # and debuggable while the learned policy matures
        bsz, seqlen = tokens.shape
        boundary_loss = None
        boundary_rate = tokens.new_zeros((), dtype=torch.float32)
        boundary_logits = None
        if self.args.patching_mode == "learned":
            if local_states is None:
                raise RuntimeError("learned patching requires local states")
            boundary_logits = self.boundary_head(F.rms_norm(local_states, (local_states.size(-1),))).squeeze(-1) + self.boundary_bias
            probs = torch.sigmoid(boundary_logits)
            if self.args.boundary_smooth_alpha > 0 and seqlen > 1:
                prev_probs = torch.zeros_like(probs)
                prev_probs[:, 1:] = probs[:, :-1]
                probs = (1.0 - self.args.boundary_smooth_alpha) * probs + self.args.boundary_smooth_alpha * prev_probs
            start = probs > 0.5
            start[:, 0] = True
            start = apply_max_patch_len(start, self.args.max_patch_len)
            start = apply_max_patch_count(start, boundary_logits.float(), self.args.max_patches)
            boundary_rate = probs[:, 1:].mean()
            target_rate = 1.0 / max(self.args.boundary_target_patch_size, 1.0)
            rate_loss = (boundary_rate - target_rate).square()
            entropy = -(probs[:, 1:] * torch.log(probs[:, 1:].clamp_min(1e-6)) + (1.0 - probs[:, 1:]) * torch.log((1.0 - probs[:, 1:]).clamp_min(1e-6))).mean()
            boundary_loss = self.args.boundary_rate_weight * rate_loss + 0.01 * entropy
        elif self.args.patching_mode == "static":
            start = torch.zeros((bsz, seqlen), dtype=torch.bool, device=tokens.device)
            start[:, :: max(self.args.patch_size, 1)] = True
            start[:, 0] = True
            start = apply_max_patch_len(start, self.args.max_patch_len)
        elif self.args.patching_mode == "space":
            start = torch.zeros((bsz, seqlen), dtype=torch.bool, device=tokens.device)
            start[:, 0] = True
            if seqlen > 1:
                start[:, 1:] = is_space_like_byte(tokens[:, :-1])
            start = apply_max_patch_len(start, self.args.max_patch_len)
        elif self.args.patching_mode == "entropy":
            if self.entropy_model is None:
                raise RuntimeError("entropy patching requested but entropy_model is missing")
            entropy_logits = self.entropy_model(tokens)
            ent = entropy_from_logits(entropy_logits).detach()
            start = torch.zeros((bsz, seqlen), dtype=torch.bool, device=tokens.device)
            start[:, 0] = True
            if seqlen > 1:
                delta = ent[:, 1:] - ent[:, :-1]
                start[:, 1:] = (ent[:, 1:] > self.args.entropy_threshold) & (delta > self.args.entropy_delta)
            start = apply_max_patch_len(start, self.args.max_patch_len)
            entropy_loss = F.cross_entropy(entropy_logits.reshape(-1, self.args.vocab_size), tokens.reshape(-1), reduction="mean")
        else:
            raise ValueError(f"unknown patching mode: {self.args.patching_mode}")

        if self.args.patching_mode != "learned" and local_states is not None and self.args.boundary_head_weight > 0:
            boundary_logits = self.boundary_head(F.rms_norm(local_states, (local_states.size(-1),))).squeeze(-1)
            base_target = start.float()
            target = base_target.clone()
            if seqlen > 1:
                local_delta = (local_states[:, 1:] - local_states[:, :-1]).float().pow(2).mean(dim=-1)
                local_delta = local_delta / local_delta.mean(dim=1, keepdim=True).clamp_min(1e-6)
                local_delta = local_delta.clamp(0.0, 2.0) / 2.0
                punctuation = is_space_like_byte(tokens[:, :-1]).float()
                target[:, 1:] = torch.maximum(base_target[:, 1:], 0.5 * local_delta + 0.5 * punctuation)
            boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits.float(), target, reduction="mean")
            extra_starts = torch.sigmoid(boundary_logits) > self.args.boundary_add_threshold
            extra_starts[:, 0] = True
            start = apply_max_patch_len(start | extra_starts, self.args.max_patch_len)
            boundary_rate = torch.sigmoid(boundary_logits[:, 1:]).mean()

        return start, entropy_loss if self.args.patching_mode == "entropy" else None, boundary_loss, boundary_rate, boundary_logits

    def _decode_logits(
        self,
        byte_states: Tensor,
        latent_local: Tensor,
        group_ids: Tensor,
        latent_valid: Tensor,
        local_mask: Tensor,
        patch_memory: Optional[Tensor] = None,
        include_current_groups: bool = False,
    ) -> Tensor:
        dec = byte_states if patch_memory is None else byte_states + patch_memory
        dec_cross_mask = build_decoder_patch_mask(group_ids, latent_local.size(1), include_current=include_current_groups) & latent_valid[:, None, :]
        for cross, gate_proj, block in zip(self.decoder_cross, self.decoder_gates, self.decoder_blocks, strict=True):
            cross_out = cross(dec, latent_local, dec_cross_mask)
            gate_in = torch.cat((F.rms_norm(dec, (dec.size(-1),)), F.rms_norm(cross_out, (cross_out.size(-1),))), dim=-1)
            gate = torch.sigmoid(gate_proj(gate_in))
            dec = dec + gate * cross_out
            dec = block(dec, local_mask)
        logits = self.byte_head(self.final_norm(dec))
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def _supervision_steps(self, active_layers: int) -> list[int]:
        taps = max(self.args.deep_supervision_taps, 1)
        if active_layers <= 1 or taps == 1:
            return [max(active_layers - 1, 0)]
        positions = torch.linspace(0, active_layers - 1, steps=taps)
        indices = sorted({int(round(x.item())) for x in positions})
        if indices[-1] != active_layers - 1:
            indices.append(active_layers - 1)
        return indices

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor | tuple[Tensor, Tensor]:
        # stage 1: local byte encoding. this is the high-resolution view of the
        # sequence before we decide what can be compressed into a span latent
        local_mask = self._local_mask(input_ids.device, input_ids.size(1))
        x = self.byte_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.local_conv(x)
        for block in self.encoder_blocks:
            x = block(x, local_mask)
        # stage 2: build patches. this is where sequence length can collapse
        # or explode, so all the control logging later is basically watching this
        start_mask, entropy_loss, boundary_loss, boundary_rate, boundary_logits = self._patch_starts(input_ids, local_states=x)
        patch_ids, patch_counts, num_patches = patch_ids_from_starts(start_mask)
        if num_patches <= 0:
            raise RuntimeError("patching produced zero patches")

        # stage 3: commit byte spans into patch latents, then let the patch
        # tokens look back at the raw local stream once before going global
        patch_repr, patch_valid = self._commit_patches(x, patch_ids, num_patches)
        enc_cross_mask = build_patch_membership_mask(patch_ids, num_patches)
        patch_repr = patch_repr + self.encoder_cross(patch_repr, x, enc_cross_mask)
        patch_local_repr = patch_repr
        # stage 4: downsample patch latents again into the shorter sequence
        # that the recurrent global prior will actually operate on
        super_repr = self.encoder_to_global(self.global_downsample(patch_local_repr))
        super_len = super_repr.size(1)
        patch_valid_counts = patch_valid.sum(dim=1)
        super_counts = (patch_valid_counts + self.args.global_latent_stride - 1) // self.args.global_latent_stride
        super_valid = torch.arange(super_len, device=input_ids.device)[None, :] < super_counts[:, None]
        super_repr = super_repr * super_valid.unsqueeze(-1)
        prior_mask = build_patch_causal_mask(super_valid, include_self=False)
        prior_context = self.global_prior(self.global_prior_norm(super_repr), prior_mask)
        prior_mean = self.prior_mean(prior_context)
        prior_log_scale = self.prior_log_scale(prior_context).clamp(min=math.log(self.args.prior_min_scale), max=3.0)
        prior_scale = prior_log_scale.exp()
        rate_error = ((super_repr - prior_mean) / prior_scale).float()
        latent_rate_loss = (0.5 * rate_error.square() + prior_log_scale.float())[super_valid].mean()

        aux_ramp = self._aux_ramp()
        hard_boundary_rate = ((patch_counts.float() - 1.0).clamp_min(0.0) / max(input_ids.size(1) - 1, 1)).mean()
        effective_boundary_rate = 0.5 * boundary_rate + 0.5 * hard_boundary_rate.detach()
        active_layers = self._adaptive_active_num_layers((latent_rate_loss + effective_boundary_rate).detach(), super_counts)
        self.last_patch_count = float(patch_counts.float().mean().item())
        self.last_super_count = float(super_counts.float().mean().item())
        self.last_avg_patch_len = float(input_ids.size(1) / max(self.last_patch_count, 1.0))
        self.last_boundary_rate = float(boundary_rate.detach().item())
        self.last_hard_boundary_rate = float(hard_boundary_rate.detach().item())
        self.last_latent_rate = float(latent_rate_loss.detach().item())
        self.last_active_layers = int(active_layers)
        self.last_compute_proxy = float(self.last_active_layers * (self.last_super_count ** 2))
        super_state = super_repr
        supervision_indices = self._supervision_steps(active_layers)
        latent_supervision_states: list[Tensor] = []
        recur_mask = build_patch_causal_mask(super_valid, include_self=False)
        patch_super_ids = (patch_ids // self.args.global_latent_stride).clamp_max(super_len - 1)
        patch_group_ids = torch.arange(num_patches, device=input_ids.device)[None, :].expand(input_ids.size(0), -1)
        patch_group_ids = patch_group_ids // self.args.global_latent_stride
        recursive_states = [super_state]
        anti_collapse_start = max(active_layers - self.args.anti_collapse_last_n, 0)
        xsa_start = max(active_layers - self.args.global_xsa_last_n, 0)
        # stage 5: shared recurrent global mixing. same block, multiple
        # passes, with step conditioning so each depth can still specialize
        for step_idx in range(active_layers):
            self.global_block.attn.xsa = self.args.global_xsa and self.train_step >= self.args.global_xsa_start_step and step_idx >= xsa_start
            step_cond = self.step_embed.weight[step_idx].to(dtype=super_state.dtype)[None, None, :]
            next_state = self.global_block(super_state + step_cond, recur_mask) - step_cond
            delta = next_state - super_state
            if self.args.anti_collapse_strength > 0 and step_idx >= anti_collapse_start:
                state_unit = F.normalize(self.global_anti_norm(super_state).float(), dim=-1, eps=1e-6)
                delta_float = delta.float()
                parallel = (delta_float * state_unit).sum(dim=-1, keepdim=True) * state_unit
                strength = self.args.anti_collapse_strength * aux_ramp
                delta = (delta_float - strength * parallel).to(dtype=super_state.dtype)
            depth_scale = max(self.args.depth_scale_min, float((step_idx + 1) ** -0.5))
            delta = delta * (self.step_gain[step_idx].to(dtype=super_state.dtype) * depth_scale)
            super_state = (super_state + delta) * super_valid.unsqueeze(-1)
            recursive_states.append(super_state)
            if step_idx in supervision_indices[:-1]:
                latent_supervision_states.append(self.global_norm(super_state))

        # stage 6: bring the compressed global state back into local space.
        # this is the bridge where the high-level summary becomes useful for bytes again
        super_local = self.global_to_local(self.global_norm(super_state))
        prev_patch_local = patch_local_repr.new_zeros(patch_local_repr.shape)
        prev_patch_local[:, 1:] = patch_local_repr[:, :-1]
        prev_super_patch_ids = (patch_group_ids - 1).clamp_min(0)
        prev_super_patch = super_local.gather(1, prev_super_patch_ids.unsqueeze(-1).expand(-1, -1, super_local.size(-1)))
        prev_super_patch = prev_super_patch * (patch_group_ids > 0).unsqueeze(-1)
        patch_super_bridge = self.super_bridge_proj(prev_super_patch)
        patch_refine_mask = build_patch_causal_mask(patch_valid, include_self=True)
        patch_bridge = patch_super_bridge
        for block in self.patch_refiner:
            patch_bridge = block(patch_bridge, patch_refine_mask)
        byte_prev_patch = prev_patch_local.gather(1, patch_ids.unsqueeze(-1).expand(-1, -1, prev_patch_local.size(-1)))
        byte_super_bridge = patch_bridge.gather(1, patch_ids.unsqueeze(-1).expand(-1, -1, patch_bridge.size(-1)))
        prev_patch_gate = torch.sigmoid(
            self.patch_bridge_gate(
                torch.cat((F.rms_norm(x, (x.size(-1),)), F.rms_norm(byte_prev_patch, (byte_prev_patch.size(-1),))), dim=-1)
            )
        )
        super_gate = torch.sigmoid(
            self.super_bridge_gate(
                torch.cat((F.rms_norm(x, (x.size(-1),)), F.rms_norm(byte_super_bridge, (byte_super_bridge.size(-1),))), dim=-1)
            )
        )
        byte_patch_bridge = prev_patch_gate * byte_prev_patch + super_gate * byte_super_bridge
        # stage 7: decode back to byte logits with the bridged latent memory
        logits = self._decode_logits(
            x,
            super_local,
            patch_super_ids,
            super_valid,
            local_mask,
            patch_memory=byte_patch_bridge,
            include_current_groups=False,
        )
        # inference path is simple: just return logits. all the weird losses
        # below only matter when we are teaching the hierarchy how to behave
        if target_ids is None:
            return logits, patch_ids
        token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="none").view_as(target_ids)
        # the byte loss is still the boss. the auxiliary terms are there to
        # shape the latent system, not to replace the actual prediction objective
        main_loss = token_loss.mean()
        total_loss = main_loss
        rate_total = latent_rate_loss + effective_boundary_rate
        self.last_rate_total = float(rate_total.detach().item())
        if self.training:
            if latent_supervision_states and self.args.latent_supervision_weight > 0:
                total_loss = total_loss + (self.args.latent_supervision_weight * aux_ramp) * latent_supervision_loss(
                    latent_supervision_states,
                    self.global_norm(super_state),
                    super_valid,
                    self.args.deep_supervision_decay,
                )
            if self.args.dual_rate_enabled:
                penalty = F.relu(rate_total - self.args.dual_rate_capacity)
                total_loss = total_loss + (self.rate_lambda * aux_ramp) * penalty
            else:
                total_loss = total_loss + (self.args.latent_rate_weight * aux_ramp) * latent_rate_loss
            if self.args.latent_align_weight > 0:
                total_loss = total_loss + (self.args.latent_align_weight * aux_ramp) * latent_alignment_loss(recursive_states, super_valid)
            if entropy_loss is not None:
                total_loss = total_loss + self.args.entropy_aux_weight * entropy_loss
            if boundary_loss is not None:
                total_loss = total_loss + self.args.boundary_head_weight * aux_ramp * boundary_loss
            if boundary_logits is not None and self.args.boundary_proxy_weight > 0 and target_ids.size(1) > 1:
                surprise = token_loss.detach()
                surprise = surprise / surprise.mean(dim=1, keepdim=True).clamp_min(1e-6)
                boundary_target = surprise[:, 1:].clamp(0.0, 2.0) / 2.0
                total_loss = total_loss + self.args.boundary_proxy_weight * aux_ramp * F.binary_cross_entropy_with_logits(
                    boundary_logits[:, 1:].float(),
                    boundary_target,
                    reduction="mean",
                )
        return total_loss, main_loss.detach()


# -----------------------------
# evaluation
# -----------------------------


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_bytes: Tensor,
) -> tuple[float, float]:
    local_batch_bytes = args.val_batch_bytes // (world_size * grad_accum_steps)
    if local_batch_bytes < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_BYTES must provide at least one sequence per rank; "
            f"got VAL_BATCH_BYTES={args.val_batch_bytes}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_bytes // args.train_seq_len
    total_seqs = (val_bytes.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    local_target_bytes = float(args.val_max_bytes) / float(max(world_size, 1)) if args.val_max_bytes > 0 else 0.0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            if local_target_bytes > 0 and val_byte_count.item() >= local_target_bytes:
                break
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_bytes[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                batch_loss, _ = model(x, y)
            batch_byte_count = float(y.numel())
            val_loss_sum += batch_loss.detach().to(torch.float64) * batch_byte_count
            val_byte_count += batch_byte_count

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_byte_count
    val_bpb = val_loss.item() / math.log(2.0)
    model.train()
    return float(val_loss.item()), float(val_bpb)


# -----------------------------
# training
# -----------------------------


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

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
        raise RuntimeError("cuda is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    compute_capability = torch.cuda.get_device_capability(device)
    use_flash = compute_capability[0] >= 8
    enable_cudnn_sdp(False)
    enable_flash_sdp(use_flash)
    enable_mem_efficient_sdp(False)
    # keep math sdp available as a fallback because some masked non-causal
    # attention calls in brelt do not have a flash kernel on all setups
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"running python {sys.version}", console=False)
    log0(f"running pytorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"script only supports sentencepiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    token_bytes_table = build_token_bytes_table(sp)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_bytes = load_validation_bytes(args.val_files, args.train_seq_len, token_bytes_table, args.decode_chunk_tokens)
    log0(f"dataset_dir:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:pattern={args.val_files} bytes:{val_bytes.numel() - 1}")
    log0(f"brelt_profile:{BRELT_PROFILE} full_profile:{int(BRELT_FULL_PROFILE)}")
    log0(f"patching_mode:{args.patching_mode} patch_size:{args.patch_size} max_patch_len:{args.max_patch_len}")
    log0("architecture:brelt_recursive_shared_v6_learned_xsa_qat")

    if args.enable_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    base_model = ByteLatentTransformer(args).to(device).to(dtype=amp_dtype)
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()

    compiled_model: nn.Module = base_model
    if args.enable_compile:
        compiled_model = torch.compile(base_model, dynamic=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=args.ddp_find_unused)
        if distributed
        else compiled_model
    )

    ema_state = {name: tensor.detach().clone() for name, tensor in base_model.state_dict().items()}
    dual_rate_lambda = float(args.dual_rate_init)

    def update_ema() -> None:
        if args.ema_decay <= 0:
            return
        with torch.no_grad():
            current_state = base_model.state_dict()
            for name, tensor in current_state.items():
                ema_tensor = ema_state[name]
                if not tensor.is_floating_point():
                    ema_tensor.copy_(tensor)
                    continue
                ema_tensor.lerp_(tensor.detach(), 1.0 - args.ema_decay)

    def load_state_dict_inplace(state: dict[str, Tensor]) -> None:
        current_state = base_model.state_dict()
        for name, tensor in state.items():
            current_state[name].copy_(tensor.to(device=current_state[name].device, dtype=current_state[name].dtype))

    named_params = list(base_model.named_parameters())
    embed_params = [p for name, p in named_params if p.requires_grad and p.ndim == 2 and "byte_emb" in name]
    head_params = [p for name, p in named_params if p.requires_grad and name == "byte_head.weight"]
    conv_params = [p for name, p in named_params if p.requires_grad and name.startswith("local_conv.")]
    control_name_patterns = ("boundary_head", "boundary_bias", "prior_mean", "prior_log_scale", "step_gain", "norm")
    control_params = [p for name, p in named_params if p.requires_grad and any(pattern in name for pattern in control_name_patterns)]
    control_param_ids = {id(p) for p in control_params}
    matrix_params = [
        p
        for name, p in named_params
        if p.requires_grad and id(p) not in control_param_ids and p.ndim == 2 and ("byte_emb" not in name and name != "byte_head.weight" and not name.startswith("local_conv."))
    ]
    scalar_params = [p for name, p in named_params if p.requires_grad and id(p) not in control_param_ids and p.ndim < 2 and not name.startswith("local_conv.")]

    optimizer_embed = torch.optim.AdamW(
        [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_conv = torch.optim.AdamW(
        [{"params": conv_params, "lr": args.matrix_lr, "base_lr": args.matrix_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizer_control = torch.optim.AdamW(
        [{"params": control_params, "lr": args.scalar_lr * 0.5, "base_lr": args.scalar_lr * 0.5}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=0.0,
        fused=True,
    )
    optimizer_head = torch.optim.AdamW(
        [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_embed, optimizer_head, optimizer_muon, optimizer_conv, optimizer_scalar, optimizer_control]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"val_max_bytes:{args.val_max_bytes}")
    log0(f"train_batch_bytes:{args.train_batch_bytes} train_seq_len:{args.train_seq_len}")
    log0(f"local_dim:{args.local_dim} global_dim:{args.model_dim}")
    log0(f"local_encoder_layers:{args.local_encoder_layers} local_decoder_layers:{args.local_decoder_layers} global_layers:{args.num_layers}")
    log0(f"global_partial_rope_fraction:{args.global_partial_rope_fraction}")
    log0(f"local_conv_kernel:{args.local_conv_kernel} local_conv_expansion:{args.local_conv_expansion}")
    log0(
        f"global_latent_stride:{args.global_latent_stride} "
        f"global_downsample_kernel:{args.global_downsample_kernel} "
        f"global_downsample_expansion:{args.global_downsample_expansion}"
    )
    log0(
        f"latent_rate_weight:{args.latent_rate_weight} "
        f"deep_supervision_decay:{args.deep_supervision_decay} "
        f"deep_supervision_taps:{args.deep_supervision_taps} "
        f"latent_supervision_weight:{args.latent_supervision_weight} "
        f"aux_warmup_steps:{args.aux_warmup_steps} "
        f"anti_collapse_strength:{args.anti_collapse_strength} "
        f"anti_collapse_last_n:{args.anti_collapse_last_n} "
        f"latent_align_weight:{args.latent_align_weight} "
        f"recurrence_start_layers:{args.recurrence_start_layers} "
        f"recurrence_warmup_steps:{args.recurrence_warmup_steps} "
        f"depth_scale_min:{args.depth_scale_min} "
        f"adaptive_recurrence:{int(args.adaptive_recurrence)} "
        f"adaptive_recurrence_min_scale:{args.adaptive_recurrence_min_scale} "
        f"adaptive_recurrence_target:{args.adaptive_recurrence_target} "
        f"adaptive_recurrence_super_target:{args.adaptive_recurrence_super_target} "
        f"adaptive_recurrence_count_min:{args.adaptive_recurrence_count_min} "
        f"adaptive_recurrence_count_max:{args.adaptive_recurrence_count_max} "
        f"prior_min_scale:{args.prior_min_scale} "
        f"global_xsa:{int(args.global_xsa)} global_xsa_last_n:{args.global_xsa_last_n} global_xsa_start_step:{args.global_xsa_start_step} "
        f"boundary_head_weight:{args.boundary_head_weight} "
        f"boundary_add_threshold:{args.boundary_add_threshold} "
        f"boundary_target_patch_size:{args.boundary_target_patch_size} "
        f"boundary_rate_weight:{args.boundary_rate_weight} "
        f"boundary_smooth_alpha:{args.boundary_smooth_alpha} "
        f"boundary_proxy_weight:{args.boundary_proxy_weight} "
        f"boundary_bias_eta:{args.boundary_bias_eta} "
        f"boundary_bias_min:{args.boundary_bias_min} boundary_bias_max:{args.boundary_bias_max} "
        f"max_patches:{args.max_patches} "
        f"patch_refiner_layers:{args.patch_refiner_layers} "
        f"global_mlp_kind:{args.global_mlp_kind} "
        f"local_mlp_kind:{args.local_mlp_kind} "
        f"global_mlp_mult:{args.global_mlp_mult} local_mlp_mult:{args.local_mlp_mult} "
        f"qat_enabled:{int(args.qat_enabled)} qat_start_step:{args.qat_start_step} qat_bits:{args.qat_bits} "
        f"dual_rate_enabled:{int(args.dual_rate_enabled)} dual_rate_capacity:{args.dual_rate_capacity} "
        f"dual_rate_eta:{args.dual_rate_eta} dual_rate_init:{args.dual_rate_init} "
        f"ema_decay:{args.ema_decay} ema_start_step:{args.ema_start_step} "
        f"ddp_find_unused:{int(args.ddp_find_unused)}"
    )
    log0(f"turboquant_rotate:{int(TURBOQUANT_ROTATE_DEFAULT)} turboquant_min_cols:{TURBOQUANT_MIN_COLS}")
    log0(
        f"embed_lr:{args.embed_lr} head_lr:{args.head_lr} matrix_lr:{args.matrix_lr} "
        f"scalar_lr:{args.scalar_lr} weight_decay:{args.weight_decay}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedByteLoader(
        args.train_files,
        rank,
        world_size,
        device,
        token_bytes_table,
        args.decode_chunk_tokens,
    )

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            base_model.set_train_step(0)
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_bytes, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    warmup_loss, _ = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedByteLoader(
            args.train_files,
            rank,
            world_size,
            device,
            token_bytes_table,
            args.decode_chunk_tokens,
        )

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    total_train_bytes = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        base_model.set_train_step(step)
        base_model.rate_lambda = dual_rate_lambda
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_main_loss = torch.zeros((), device=device)
        max_patch_count = 0.0
        max_super_count = 0.0
        max_compute_proxy = 0.0
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_bytes, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                loss, main_loss = model(x, y)
            train_loss += loss.detach()
            train_main_loss += main_loss.detach()
            max_patch_count = max(max_patch_count, base_model.last_patch_count)
            max_super_count = max(max_super_count, base_model.last_super_count)
            max_compute_proxy = max(max_compute_proxy, base_model.last_compute_proxy)
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_main_loss /= grad_accum_steps

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
        if args.dual_rate_enabled:
            dual_rate_lambda = max(0.0, dual_rate_lambda + args.dual_rate_eta * (base_model.last_rate_total - args.dual_rate_capacity))
        if args.patching_mode == "learned" and args.boundary_bias_eta > 0:
            target_hard_rate = 1.0 / max(args.boundary_target_patch_size, 1.0)
            bias_delta = args.boundary_bias_eta * (target_hard_rate - base_model.last_hard_boundary_rate)
            with torch.no_grad():
                base_model.boundary_bias.add_(bias_delta)
                base_model.boundary_bias.clamp_(min=args.boundary_bias_min, max=args.boundary_bias_max)
        if step + 1 >= args.ema_start_step:
            update_ema()
        zero_grad_all()

        step += 1
        total_train_bytes += args.train_batch_bytes
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_main_loss:{train_main_loss.item():.4f} "
                f"train_bpb:{train_main_loss.item() / math.log(2.0):.4f} active_layers:{base_model.last_active_layers} "
                f"patches:{base_model.last_patch_count:.1f} super:{base_model.last_super_count:.1f} "
                f"max_patches:{max_patch_count:.1f} max_super:{max_super_count:.1f} "
                f"avg_patch_len:{base_model.last_avg_patch_len:.2f} boundary_rate:{base_model.last_boundary_rate:.4f} "
                f"hard_boundary_rate:{base_model.last_hard_boundary_rate:.4f} boundary_bias:{base_model.boundary_bias.item():.4f} "
                f"latent_rate:{base_model.last_latent_rate:.4f} dual_lambda:{dual_rate_lambda:.4f} lr_scale:{scale:.4f} "
                f"compute_proxy:{base_model.last_compute_proxy:.1f} max_compute_proxy:{max_compute_proxy:.1f} "
                f"seen_gb:{total_train_bytes / 1e9:.3f} throughput_mib_s:{(total_train_bytes / (1024.0 * 1024.0)) / max(approx_training_time_ms / 1000.0, 1e-9):.1f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    raw_val_loss, raw_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
    log0(f"final_fp_raw val_loss:{raw_val_loss:.4f} val_bpb:{raw_val_bpb:.4f}")

    export_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
    export_label = "raw"
    best_val_loss = raw_val_loss
    if args.ema_decay > 0 and step >= args.ema_start_step:
        raw_state = export_state
        load_state_dict_inplace(ema_state)
        log0("using_ema_weights:1")
        ema_val_loss, ema_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
        log0(f"final_fp_ema val_loss:{ema_val_loss:.4f} val_bpb:{ema_val_bpb:.4f}")
        if ema_val_loss < best_val_loss:
            export_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
            export_label = "ema"
            best_val_loss = ema_val_loss
        else:
            load_state_dict_inplace(raw_state)
    log0(f"export_weights:{export_label}")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"serialized model: {model_bytes} bytes")
        log0(f"code size: {code_bytes} bytes")
        log0(f"total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_bytes)
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
