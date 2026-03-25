"""
Byte-level JEPA baseline for the Parameter Golf challenge.

This script intentionally drops SentencePiece and trains directly on the published
`byte260` shard format. Download that export first, for example:

python data/cached_challenge_fineweb.py --variant byte260

The model is JEPA-based rather than decoder-only:
- a shared byte encoder embeds context and target spans
- a predictor maps encoded context to future target embeddings
- a byte decoder produces target-token log-probs for challenge scoring
- a Gaussian latent regularizer keeps the embedding space well-conditioned

Hard stop: keep `train_gpt.py` readable and under 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
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
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


def env_flag(name: str, default: bool) -> bool:
    return bool(int(os.environ.get(name, "1" if default else "0")))


class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/byte260_export/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 65_536))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    max_val_tokens = int(os.environ.get("MAX_VAL_TOKENS", 131_072))
    final_max_val_tokens = int(os.environ.get("FINAL_MAX_VAL_TOKENS", 131_072))
    final_full_val = env_flag("FINAL_FULL_VAL", False)
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    iterations = int(os.environ.get("ITERATIONS", 16_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 131_072))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    target_seq_len = int(os.environ.get("TARGET_SEQ_LEN", 256))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    pad_id = int(os.environ.get("PAD_ID", 0))
    bos_id = int(os.environ.get("BOS_ID", 1))
    eos_id = int(os.environ.get("EOS_ID", 2))
    unk_id = int(os.environ.get("UNK_ID", 3))
    byte_offset = int(os.environ.get("BYTE_OFFSET", 4))
    byte_count = int(os.environ.get("BYTE_COUNT", 256))

    model_dim = int(os.environ.get("MODEL_DIM", 384))
    latent_dim = int(os.environ.get("LATENT_DIM", 128))
    encoder_layers = int(os.environ.get("ENCODER_LAYERS", 8))
    predictor_layers = int(os.environ.get("PREDICTOR_LAYERS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 6))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_activation = os.environ.get("MLP_ACTIVATION", "leaky_relu_sq")
    context_pool_stride = int(os.environ.get("CONTEXT_POOL_STRIDE", 1))
    predict_chunk_len = int(os.environ.get("PREDICT_CHUNK_LEN", 32))
    target_patch_size = int(os.environ.get("TARGET_PATCH_SIZE", 1))
    memory_tokens = int(os.environ.get("MEMORY_TOKENS", 64))
    memory_layers = int(os.environ.get("MEMORY_LAYERS", 2))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 20.0))
    decoder_layers = int(os.environ.get("DECODER_LAYERS", 2))
    predictor_recur_layers = int(os.environ.get("PREDICTOR_RECUR_LAYERS", 0))
    predictor_recur_passes = int(os.environ.get("PREDICTOR_RECUR_PASSES", 1))
    decoder_recur_layers = int(os.environ.get("DECODER_RECUR_LAYERS", 0))
    decoder_recur_passes = int(os.environ.get("DECODER_RECUR_PASSES", 1))

    ce_loss_weight = float(os.environ.get("CE_LOSS_WEIGHT", 1.0))
    latent_loss_weight = float(os.environ.get("LATENT_LOSS_WEIGHT", 1.0))
    sigreg_weight = float(os.environ.get("SIGREG_WEIGHT", 0.05))
    sigreg_cov_weight = float(os.environ.get("SIGREG_COV_WEIGHT", 0.05))
    lewm_style = env_flag("LEWM_STYLE", True)
    decoder_aux_weight = float(os.environ.get("DECODER_AUX_WEIGHT", 1.0))
    sigreg_lambda = float(os.environ.get("SIGREG_LAMBDA", 0.05))
    chunk_latent_weight = float(os.environ.get("CHUNK_LATENT_WEIGHT", 0.25))
    decoder_byte_mask_rate = float(os.environ.get("DECODER_BYTE_MASK_RATE", 0.15))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.996))

    embed_lr = float(os.environ.get("EMBED_LR", 0.25))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.035))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    use_compile = env_flag("USE_COMPILE", os.name != "nt")
    compile_fullgraph = env_flag("COMPILE_FULLGRAPH", False)

    # TTT (test-time training) — score-first, adapt-second protocol
    ttt_enabled = env_flag("TTT_ENABLED", False)
    ttt_lr = float(os.environ.get("TTT_LR", 0.001))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_train_mode = os.environ.get("TTT_TRAIN_MODE", "late")
    ttt_freeze_encoder_blocks = int(os.environ.get("TTT_FREEZE_ENCODER_BLOCKS", 6))
    ttt_lr_min_frac = float(os.environ.get("TTT_LR_MIN_FRAC", 0.2))
    ttt_pure_latent = env_flag("TTT_PURE_LATENT", True)

    # Eval-only causal byte cache for better bpb without changing training.
    byte_logit_cache_enabled = env_flag("BYTE_LOGIT_CACHE_ENABLED", False)
    byte_logit_cache_min_n = int(os.environ.get("BYTE_LOGIT_CACHE_MIN_N", 3))
    byte_logit_cache_max_n = int(os.environ.get("BYTE_LOGIT_CACHE_MAX_N", 8))
    byte_logit_cache_alpha = float(os.environ.get("BYTE_LOGIT_CACHE_ALPHA", 0.25))
    byte_logit_cache_min_count = int(os.environ.get("BYTE_LOGIT_CACHE_MIN_COUNT", 2))
    byte_logit_cache_count_scale = float(os.environ.get("BYTE_LOGIT_CACHE_COUNT_SCALE", 16.0))

    latent_knn_cache_enabled = env_flag("LATENT_KNN_CACHE_ENABLED", False)
    latent_knn_cache_k = int(os.environ.get("LATENT_KNN_CACHE_K", 3))
    latent_knn_cache_alpha = float(os.environ.get("LATENT_KNN_CACHE_ALPHA", 0.1))

    # Cross-attention decoder: each decoder layer cross-attends to pred_latents
    # instead of receiving them as an additive input bias.
    use_cross_decoder = env_flag("CROSS_DECODER", False)

    @property
    def context_seq_len(self) -> int:
        return self.train_seq_len - self.target_seq_len


# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.transpose(-2, -1)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.transpose(-2, -1) if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
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
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.float32)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad.float()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / max(g.size(1), 1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,mlp_scale,self_scale,cross_scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


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
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
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
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# -----------------------------
# DATA LOADING + BYTE METRIC
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
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


def load_validation_tokens(pattern: str, target_seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    if tokens.numel() < target_seq_len:
        raise ValueError(f"Validation split is too short for TARGET_SEQ_LEN={target_seq_len}")
    return tokens


def cap_validation_tokens(tokens: Tensor, max_tokens: int, target_seq_len: int) -> Tensor:
    if max_tokens <= 0 or tokens.numel() <= max_tokens:
        return tokens
    capped_tokens = max(max_tokens, target_seq_len)
    capped_tokens = min(capped_tokens, int(tokens.numel()))
    capped_tokens = (capped_tokens // target_seq_len) * target_seq_len
    if capped_tokens <= 0:
        capped_tokens = min(int(tokens.numel()), target_seq_len)
    return tokens[:capped_tokens].contiguous()


def build_byte_lut(vocab_size: int, byte_offset: int, byte_count: int, device: torch.device) -> Tensor:
    out = np.zeros((vocab_size,), dtype=np.int16)
    hi = min(vocab_size, byte_offset + byte_count)
    out[byte_offset:hi] = 1
    return torch.tensor(out, dtype=torch.int16, device=device)


class TokenStream:
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


class DistributedSequenceLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        denom = self.world_size * grad_accum_steps * seq_len
        if global_tokens % denom != 0:
            raise ValueError(
                f"TRAIN_BATCH_TOKENS={global_tokens} must be divisible by "
                f"WORLD_SIZE*GRAD_ACCUM_STEPS*TRAIN_SEQ_LEN={denom}"
            )
        local_seqs = global_tokens // denom
        per_rank_tokens = local_seqs * seq_len
        chunk = self.stream.take(per_rank_tokens * self.world_size)
        start = self.rank * per_rank_tokens
        local = chunk[start : start + per_rank_tokens].to(dtype=torch.int64)
        return local.reshape(local_seqs, seq_len).to(self.device, non_blocking=True)


# -----------------------------
# JEPA MODEL
# -----------------------------

def maybe_compile(obj, *, enabled: bool, fullgraph: bool):
    if not enabled or not hasattr(torch, "compile"):
        return obj
    try:
        return torch.compile(obj, dynamic=False, fullgraph=fullgraph)
    except Exception:
        return obj


def make_adam(param_groups: list[dict[str, object]], *, beta1: float, beta2: float, eps: float):
    kwargs = dict(betas=(beta1, beta2), eps=eps)
    if torch.cuda.is_available():
        try:
            return torch.optim.Adam(param_groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(param_groups, **kwargs)


def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def pool_sequence(x: Tensor, stride: int) -> Tensor:
    if stride <= 1:
        return x
    bsz, seqlen, dim = x.shape
    whole = seqlen // stride
    parts: list[Tensor] = []
    if whole > 0:
        parts.append(x[:, : whole * stride].reshape(bsz, whole, stride, dim).mean(dim=2))
    if whole * stride < seqlen:
        parts.append(x[:, whole * stride :].mean(dim=1, keepdim=True))
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


def sigreg_loss(z: Tensor, cov_weight: float, num_slices: int = 512, num_t: int = 17) -> Tensor:
    """SIGReg via Epps-Pulley characteristic function test (LeWM).

    Matches the distribution of latent embeddings to an isotropic Gaussian
    by testing random 1D projections against N(0,1).  The ``cov_weight``
    parameter is accepted for API compatibility but unused.
    """
    flat = z.float().reshape(-1, z.size(-1))
    N, D = flat.shape
    if N < 2:
        return flat.new_zeros(())
    # Random unit-norm projection directions
    A = torch.randn(D, num_slices, device=flat.device, dtype=torch.float32)
    A = F.normalize(A, dim=0)
    proj = flat @ A  # (N, M)
    # Quadrature points in [0.2, 4.0] (recommended by LeWM)
    t = torch.linspace(0.2, 4.0, num_t, device=flat.device, dtype=torch.float32)
    gauss_cf = torch.exp(-0.5 * t * t)  # target N(0,1) characteristic function
    # Empirical characteristic function via cos/sin (avoids complex tensors)
    phase = proj.unsqueeze(2) * t  # (N, M, T)
    ecf_real = torch.cos(phase).mean(dim=0)  # (M, T)
    ecf_imag = torch.sin(phase).mean(dim=0)  # (M, T)
    # Gaussian-weighted squared distance |ecf(t) - gauss_cf(t)|^2 * w(t)
    diff_sq = (ecf_real - gauss_cf).square() + ecf_imag.square()
    weighted = diff_sq * gauss_cf
    ep_stat = torch.trapezoid(weighted, t, dim=1)  # (M,)
    return ep_stat.mean()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.weight.to(dtype=x.dtype), eps=self.eps)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = F.rms_norm(q.reshape(bsz, seqlen, self.num_heads, self.head_dim), (self.head_dim,)).transpose(1, 2)
        k = F.rms_norm(k.reshape(bsz, seqlen, self.num_heads, self.head_dim), (self.head_dim,)).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=self.causal)
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, causal: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        bsz, qlen, dim = x.shape
        mlen = memory.size(1)
        q = self.q_proj(x).reshape(bsz, qlen, self.num_heads, self.head_dim)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)
        q = F.rms_norm(q, (self.head_dim,)).transpose(1, 2)
        k = F.rms_norm(k.reshape(bsz, mlen, self.num_heads, self.head_dim), (self.head_dim,)).transpose(1, 2)
        v = v.reshape(bsz, mlen, self.num_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=self.causal)
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, qlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mult: int, activation: str):
        super().__init__()
        hidden = dim * mult
        if activation not in {"relu_sq", "leaky_relu_sq"}:
            raise ValueError(f"Unsupported MLP_ACTIVATION={activation}")
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation == "leaky_relu_sq":
            x = F.leaky_relu(x, negative_slope=0.5)
        else:
            x = torch.relu(x)
        return self.proj(x.square())


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, mlp_activation: str):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult, mlp_activation)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, mlp_activation: str):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads, causal=True)
        self.mlp = MLP(dim, mlp_mult, mlp_activation)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class CrossDecoderBlock(nn.Module):
    """Decoder block with cross-attention to predicted latents.

    Unlike the plain DecoderBlock (which receives latents as an additive input
    bias), each layer here explicitly queries the full pred_latents sequence.
    Cross-attention over pred_latents is non-causal: position t can attend to
    all predicted latent positions because pred_latents are derived solely from
    the context encoder and carry no target-byte information.
    """

    def __init__(self, dim: int, num_heads: int, mlp_mult: int, mlp_activation: str):
        super().__init__()
        self.self_norm = RMSNorm(dim)
        self.cross_norm = RMSNorm(dim)
        self.latent_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, causal=True)
        self.cross_attn = CrossAttention(dim, num_heads)  # non-causal over latents
        self.mlp = MLP(dim, mlp_mult, mlp_activation)
        self.self_scale = nn.Parameter(torch.ones(dim))
        self.cross_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, latents: Tensor) -> Tensor:
        x = x + self.self_scale.to(dtype=x.dtype)[None, None, :] * self.self_attn(self.self_norm(x))
        x = x + self.cross_scale.to(dtype=x.dtype)[None, None, :] * self.cross_attn(self.cross_norm(x), self.latent_norm(latents))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class PredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, mlp_activation: str):
        super().__init__()
        self.self_norm = RMSNorm(dim)
        self.cross_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, causal=True)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult, mlp_activation)
        self.self_scale = nn.Parameter(torch.ones(dim))
        self.cross_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = x + self.self_scale.to(dtype=x.dtype)[None, None, :] * self.self_attn(self.self_norm(x))
        x = x + self.cross_scale.to(dtype=x.dtype)[None, None, :] * self.cross_attn(self.cross_norm(x), self.memory_norm(memory))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class MemoryResamplerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int, mlp_activation: str):
        super().__init__()
        self.cross_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.mlp_norm = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_mult, mlp_activation)
        self.cross_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

    def forward(self, slots: Tensor, memory: Tensor) -> Tensor:
        slots = slots + self.cross_scale.to(dtype=slots.dtype)[None, None, :] * self.cross_attn(
            self.cross_norm(slots), self.memory_norm(memory)
        )
        slots = slots + self.mlp_scale.to(dtype=slots.dtype)[None, None, :] * self.mlp(self.mlp_norm(slots))
        return slots


def causal_byte_logit_cache_ce(
    model_true_log_probs: Tensor,
    context_ids: Tensor,
    target_ids: Tensor,
    *,
    vocab_size: int,
    min_n: int,
    max_n: int,
    alpha: float,
    min_count: int,
    count_scale: float,
) -> Tensor:
    """Blend decoder probabilities with a strict prefix-only byte cache."""
    base_nll = -model_true_log_probs
    if alpha <= 0.0 or max_n < min_n or target_ids.numel() == 0:
        return base_nll.mean()

    smoothing = 0.25
    blended_nll = base_nll.detach().to(device="cpu", dtype=torch.float64).clone()
    model_true_probs = model_true_log_probs.detach().exp().to(device="cpu", dtype=torch.float64)
    context_cpu = context_ids.detach().to(device="cpu", dtype=torch.int64)
    target_cpu = target_ids.detach().to(device="cpu", dtype=torch.int64)

    def update_tables(
        history: list[int],
        next_token: int,
        tables: list[dict[tuple[int, ...], list]],
    ) -> None:
        pos = len(history)
        for n in range(min_n, min(max_n, pos) + 1):
            key = tuple(history[pos - n : pos])
            entry = tables[n].get(key)
            if entry is None:
                tables[n][key] = [1, {next_token: 1}]
                continue
            entry[0] += 1
            counts = entry[1]
            counts[next_token] = counts.get(next_token, 0) + 1

    for batch_idx in range(int(target_cpu.size(0))):
        history: list[int] = []
        tables: list[dict[tuple[int, ...], list]] = [dict() for _ in range(max_n + 1)]

        for token in context_cpu[batch_idx].tolist():
            update_tables(history, token, tables)
            history.append(token)

        for token_idx, token in enumerate(target_cpu[batch_idx].tolist()):
            weighted_true = 0.0
            weighted_total = 0.0
            for n in range(min(max_n, len(history)), min_n - 1, -1):
                entry = tables[n].get(tuple(history[-n:]))
                if entry is None:
                    continue
                total, counts = entry
                if total < min_count:
                    continue
                weight = float(n - min_n + 1)
                weighted_total += weight * float(total)
                weighted_true += weight * float(counts.get(token, 0))

            if weighted_total > 0.0:
                cache_prob = (weighted_true + smoothing) / (weighted_total + smoothing * vocab_size)
                alpha_eff = alpha * (weighted_total / (weighted_total + count_scale))
                model_prob = float(model_true_probs[batch_idx, token_idx])
                mixed_prob = (1.0 - alpha_eff) * model_prob + alpha_eff * cache_prob
                blended_nll[batch_idx, token_idx] = -math.log(max(mixed_prob, 1e-12))

            update_tables(history, token, tables)
            history.append(token)

    return blended_nll.to(device=model_true_log_probs.device, dtype=model_true_log_probs.dtype).mean()


class ByteJEPA(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        if args.context_seq_len <= 0:
            raise ValueError("TRAIN_SEQ_LEN must be larger than TARGET_SEQ_LEN")
        if args.target_seq_len % args.predict_chunk_len != 0:
            raise ValueError("TARGET_SEQ_LEN must be divisible by PREDICT_CHUNK_LEN")
        if args.predict_chunk_len % args.target_patch_size != 0:
            raise ValueError("PREDICT_CHUNK_LEN must be divisible by TARGET_PATCH_SIZE")
        if args.logit_softcap <= 0.0:
            raise ValueError("LOGIT_SOFTCAP must be positive")
        if args.use_cross_decoder and args.decoder_recur_passes > 1 and args.decoder_recur_layers > 0:
            raise ValueError("Decoder recurrence (DECODER_RECUR_PASSES>1 + DECODER_RECUR_LAYERS>0) is incompatible with CROSS_DECODER")
        self.vocab_size = args.vocab_size
        self.pad_id = args.pad_id
        self.bos_id = args.bos_id
        self.context_seq_len = args.context_seq_len
        self.target_seq_len = args.target_seq_len
        self.predict_chunk_len = args.predict_chunk_len
        self.num_target_chunks = args.target_seq_len // args.predict_chunk_len
        self.target_patch_size = args.target_patch_size
        self.predict_chunk_patches = args.predict_chunk_len // args.target_patch_size
        self.context_pool_stride = args.context_pool_stride
        self.memory_tokens = args.memory_tokens
        self.use_memory_resampler = args.memory_tokens > 0 and args.memory_layers > 0
        self.logit_softcap = args.logit_softcap
        self.latent_loss_weight = args.latent_loss_weight
        self.ce_loss_weight = args.ce_loss_weight
        self.sigreg_weight = args.sigreg_weight
        self.sigreg_cov_weight = args.sigreg_cov_weight
        self.lewm_style = args.lewm_style
        self.decoder_aux_weight = args.decoder_aux_weight
        self.sigreg_lambda = args.sigreg_lambda
        self.chunk_latent_weight = args.chunk_latent_weight
        self.decoder_byte_mask_rate = args.decoder_byte_mask_rate
        self.byte_logit_cache_enabled = args.byte_logit_cache_enabled
        self.byte_logit_cache_min_n = args.byte_logit_cache_min_n
        self.byte_logit_cache_max_n = args.byte_logit_cache_max_n
        self.byte_logit_cache_alpha = args.byte_logit_cache_alpha
        self.byte_logit_cache_min_count = args.byte_logit_cache_min_count
        self.byte_logit_cache_count_scale = args.byte_logit_cache_count_scale
        self.latent_knn_cache_enabled = args.latent_knn_cache_enabled
        self.latent_knn_cache_k = args.latent_knn_cache_k
        self.latent_knn_cache_alpha = args.latent_knn_cache_alpha
        self.use_cross_decoder = args.use_cross_decoder
        self.predictor_recur_layers = min(max(args.predictor_recur_layers, 0), args.predictor_layers)
        self.predictor_recur_passes = max(args.predictor_recur_passes, 1)
        self.decoder_recur_layers = min(max(args.decoder_recur_layers, 0), args.decoder_layers)
        self.decoder_recur_passes = max(args.decoder_recur_passes, 1)
        self.use_target_latent_losses = (
            (self.latent_loss_weight > 0)
            or (self.chunk_latent_weight > 0)
            or (self.lewm_style and self.sigreg_lambda > 0)
            or ((not self.lewm_style) and self.sigreg_weight > 0)
        )

        self.byte_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.pos_emb = nn.Parameter(torch.empty(args.train_seq_len, args.model_dim))
        self.future_queries = nn.Parameter(torch.empty(self.predict_chunk_patches, args.model_dim))
        self.rollout_step_emb = nn.Parameter(torch.empty(self.num_target_chunks, args.model_dim))
        self.query_seed = nn.Linear(args.model_dim, args.model_dim, bias=False)
        self.rollout_token_proj = nn.Linear(args.model_dim, args.model_dim, bias=False)

        if self.use_memory_resampler:
            self.memory_queries = nn.Parameter(torch.empty(args.memory_tokens, args.model_dim))
            self.memory_seed = nn.Linear(args.model_dim, args.model_dim, bias=False)
            self.memory_blocks = nn.ModuleList(
                [
                    MemoryResamplerBlock(args.model_dim, args.num_heads, args.mlp_mult, args.mlp_activation)
                    for _ in range(args.memory_layers)
                ]
            )
            self.memory_norm = RMSNorm(args.model_dim)
        else:
            self.register_parameter("memory_queries", None)
            self.memory_seed = nn.Identity()
            self.memory_blocks = nn.ModuleList()
            self.memory_norm = nn.Identity()

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(args.model_dim, args.num_heads, args.mlp_mult, args.mlp_activation)
                for _ in range(args.encoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(args.model_dim)

        self.predictor_blocks = nn.ModuleList(
            [
                PredictorBlock(args.model_dim, args.num_heads, args.mlp_mult, args.mlp_activation)
                for _ in range(args.predictor_layers)
            ]
        )
        self.predictor_norm = RMSNorm(args.model_dim)
        if self.predictor_recur_layers > 0 and self.predictor_recur_passes > 1:
            recur_shape = (self.predictor_recur_passes - 1, self.predictor_recur_layers, args.model_dim)
            self.predictor_recur_scale = nn.Parameter(torch.ones(recur_shape))
            self.predictor_recur_bias = nn.Parameter(torch.zeros(recur_shape))
        else:
            self.register_parameter("predictor_recur_scale", None)
            self.register_parameter("predictor_recur_bias", None)
        self.target_proj = nn.Linear(args.model_dim, args.latent_dim, bias=False)
        self.pred_proj = nn.Linear(args.model_dim, args.latent_dim, bias=False)
        self.decode_proj = nn.Linear(args.latent_dim, args.model_dim, bias=False)
        self.decode_norm = RMSNorm(args.model_dim)
        _decoder_block_cls = CrossDecoderBlock if args.use_cross_decoder else DecoderBlock
        self.decoder_blocks = nn.ModuleList(
            [
                _decoder_block_cls(args.model_dim, args.num_heads, args.mlp_mult, args.mlp_activation)
                for _ in range(args.decoder_layers)
            ]
        )
        if self.decoder_recur_layers > 0 and self.decoder_recur_passes > 1:
            recur_shape = (self.decoder_recur_passes - 1, self.decoder_recur_layers, args.model_dim)
            self.decoder_recur_scale = nn.Parameter(torch.ones(recur_shape))
            self.decoder_recur_bias = nn.Parameter(torch.zeros(recur_shape))
        else:
            self.register_parameter("decoder_recur_scale", None)
            self.register_parameter("decoder_recur_bias", None)
        self.decoder_final_norm = RMSNorm(args.model_dim)

        # EMA target encoder: slow-moving copy provides stable prediction
        # targets so the predictor isn't chasing a moving target.
        self.ema_decay = args.ema_decay
        self._build_ema_encoder(args)

        self._init_weights()
        self.zero_pad_embedding_()
        # Initialize EMA from online encoder after weight init
        self._sync_ema_encoder()

    def _build_ema_encoder(self, args: Hyperparameters) -> None:
        """Create EMA copies of encoder + target projection."""
        self.ema_byte_emb = copy.deepcopy(self.byte_emb)
        self.ema_pos_emb = nn.Parameter(self.pos_emb.data.clone())
        self.ema_encoder_blocks = copy.deepcopy(self.encoder_blocks)
        self.ema_encoder_norm = copy.deepcopy(self.encoder_norm)
        self.ema_target_proj = copy.deepcopy(self.target_proj)
        # EMA params don't receive optimizer gradients
        for p in self._ema_params():
            p.requires_grad_(False)

    def _ema_params(self):
        yield self.ema_pos_emb
        for m in [self.ema_byte_emb, self.ema_encoder_blocks,
                  self.ema_encoder_norm, self.ema_target_proj]:
            yield from m.parameters()

    @torch.no_grad()
    def _sync_ema_encoder(self) -> None:
        """Hard-copy online encoder weights to EMA encoder."""
        pairs = zip(self._ema_params(), self._online_target_params())
        for ema_p, online_p in pairs:
            ema_p.data.copy_(online_p.data)

    def _online_target_params(self):
        """Online encoder params that correspond to EMA params."""
        yield self.pos_emb
        for m in [self.byte_emb, self.encoder_blocks,
                  self.encoder_norm, self.target_proj]:
            yield from m.parameters()

    @torch.no_grad()
    def update_ema(self) -> None:
        """One EMA step — call after each optimizer step."""
        d = self.ema_decay
        for ema_p, online_p in zip(self._ema_params(), self._online_target_params()):
            ema_p.data.lerp_(online_p.data, 1.0 - d)

    def serializable_state_dict(self) -> dict[str, Tensor]:
        """State dict without EMA params (they're rebuilt from online params)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("ema_")}

    def load_serializable_state_dict(self, state_dict: dict[str, Tensor], **kwargs) -> None:
        """Load state dict and rebuild EMA from online params."""
        self.load_state_dict(state_dict, strict=False, **kwargs)
        self._sync_ema_encoder()

    def _init_weights(self) -> None:
        nn.init.normal_(self.byte_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)
        if self.memory_queries is not None:
            nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.rollout_step_emb, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @torch.no_grad()
    def zero_pad_embedding_(self) -> None:
        if 0 <= self.pad_id < self.byte_emb.num_embeddings:
            self.byte_emb.weight[self.pad_id].zero_()

    def embed_tokens(self, token_ids: Tensor, *, pos_offset: int = 0) -> Tensor:
        x = self.byte_emb(token_ids)
        x = x + self.pos_emb[pos_offset : pos_offset + token_ids.size(1)].to(dtype=x.dtype)[None, :, :]
        if self.pad_id >= 0:
            x = x * (token_ids != self.pad_id).unsqueeze(-1)
        return x

    def encode(self, token_ids: Tensor, *, pos_offset: int = 0) -> Tensor:
        x = self.embed_tokens(token_ids, pos_offset=pos_offset)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.encoder_blocks:
            x = block(x)
        return self.encoder_norm(x)

    def build_memory(self, context_states: Tensor) -> Tensor:
        pooled = pool_sequence(context_states, self.context_pool_stride)
        if not self.use_memory_resampler:
            return pooled
        summary = context_states[:, -1:, :] if context_states.size(1) > 0 else pooled.mean(dim=1, keepdim=True)
        slots = self.memory_queries[None, :, :].to(dtype=context_states.dtype) + self.memory_seed(summary)
        for block in self.memory_blocks:
            slots = block(slots, pooled)
        return self.memory_norm(slots)

    def apply_predictor_recurrence(self, x: Tensor, memory: Tensor) -> Tensor:
        if self.predictor_recur_scale is None or self.predictor_recur_bias is None:
            return x
        tail_blocks = self.predictor_blocks[-self.predictor_recur_layers :]
        for pass_idx in range(self.predictor_recur_passes - 1):
            for layer_idx, block in enumerate(tail_blocks):
                block_input = x + self.predictor_recur_bias[pass_idx, layer_idx][None, None, :].to(dtype=x.dtype)
                block_output = block(block_input, memory)
                scale = self.predictor_recur_scale[pass_idx, layer_idx][None, None, :].to(dtype=x.dtype)
                x = block_input + scale * (block_output - block_input)
        return x

    def apply_decoder_recurrence(self, x: Tensor) -> Tensor:
        if self.decoder_recur_scale is None or self.decoder_recur_bias is None:
            return x
        tail_blocks = self.decoder_blocks[-self.decoder_recur_layers :]
        for pass_idx in range(self.decoder_recur_passes - 1):
            for layer_idx, block in enumerate(tail_blocks):
                block_input = x + self.decoder_recur_bias[pass_idx, layer_idx][None, None, :].to(dtype=x.dtype)
                block_output = block(block_input)
                scale = self.decoder_recur_scale[pass_idx, layer_idx][None, None, :].to(dtype=x.dtype)
                x = block_input + scale * (block_output - block_input)
        return x

    def predict_hidden(self, context_states: Tensor) -> Tensor:
        memory = self.build_memory(context_states)
        summary = context_states[:, -1:, :] if context_states.size(1) > 0 else memory.mean(dim=1, keepdim=True)
        rollout_state = summary
        chunks: list[Tensor] = []
        for chunk_idx in range(self.num_target_chunks):
            x = (
                self.future_queries[None, :, :].to(dtype=context_states.dtype)
                + self.query_seed(rollout_state)
                + self.rollout_step_emb[chunk_idx][None, None, :].to(dtype=context_states.dtype)
            )
            for block in self.predictor_blocks:
                x = block(x, memory)
            x = self.apply_predictor_recurrence(x, memory)
            x = self.predictor_norm(x)
            chunks.append(x)
            # Feed full per-token outputs into memory for fine-grained
            # autoregressive context (not just a single chunk-mean summary).
            memory = torch.cat((memory, self.rollout_token_proj(x)), dim=1)
            rollout_state = x[:, -1:, :]
        return torch.cat(chunks, dim=1)

    def predict(self, context_states: Tensor) -> Tensor:
        x = self.predict_hidden(context_states)
        return self.pred_proj(x)

    @torch.no_grad()
    def ema_encode(self, token_ids: Tensor, *, pos_offset: int = 0) -> Tensor:
        """Encode using the EMA (slow-moving) target encoder."""
        x = self.ema_byte_emb(token_ids)
        x = x + self.ema_pos_emb[pos_offset : pos_offset + token_ids.size(1)].to(dtype=x.dtype)[None, :, :]
        if self.pad_id >= 0:
            x = x * (token_ids != self.pad_id).unsqueeze(-1)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.ema_encoder_blocks:
            x = block(x)
        return self.ema_encoder_norm(x)

    def target_latents(self, target_ids: Tensor) -> Tensor:
        """Target latents from EMA encoder — stable prediction targets."""
        enc = self.ema_encode(target_ids, pos_offset=self.context_seq_len)
        return self.ema_target_proj(enc)

    def build_shifted_target_emb(self, target_ids: Tensor) -> Tensor:
        """Right-shifted target byte embeddings for autoregressive decoding.

        During training, randomly zeros out a fraction of byte embeddings
        (controlled by decoder_byte_mask_rate) to prevent the decoder from
        relying solely on autoregressive byte context and force it to use
        latent predictions from the JEPA predictor.
        """
        bsz = target_ids.size(0)
        bos = target_ids.new_full((bsz, 1), self.bos_id)
        shifted_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)
        x = self.embed_tokens(shifted_ids, pos_offset=self.context_seq_len)
        x = F.rms_norm(x, (x.size(-1),))
        if self.training and self.decoder_byte_mask_rate > 0:
            mask = torch.rand(bsz, x.size(1), 1, device=x.device) > self.decoder_byte_mask_rate
            x = x * mask
        return x

    def decode_logits(self, pred_latents: Tensor, target_ids: Tensor) -> Tensor:
        # Project pred_latents from latent_dim → model_dim once; reused by both modes.
        latent_proj = self.decode_norm(self.decode_proj(pred_latents))
        if self.target_patch_size > 1:
            latent_proj = latent_proj.repeat_interleave(self.target_patch_size, dim=1)
        byte_ctx = self.build_shifted_target_emb(target_ids)
        tgt_len = byte_ctx.size(1)
        extra_pos = self.pos_emb[self.context_seq_len : self.context_seq_len + tgt_len].to(dtype=byte_ctx.dtype)[None, :, :]
        if self.use_cross_decoder:
            # Cross-decoder: byte context + position forms the input; each layer
            # cross-attends to latent_proj (non-causal — pred_latents carry no
            # target-byte information so attending to all positions is safe).
            x = byte_ctx + extra_pos
            for block in self.decoder_blocks:
                x = block(x, latent_proj)
        else:
            # Original additive-bias decoder.
            x = latent_proj + byte_ctx + extra_pos
            for block in self.decoder_blocks:
                x = block(x)
        x = self.apply_decoder_recurrence(x)
        x = self.decoder_final_norm(x)
        logits_proj = F.linear(x, self.byte_emb.weight)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def latent_knn_cache_logits(self, pred_latents: Tensor, context_latents: Tensor, context_ids: Tensor, k: int = 3) -> Tensor:
        # All ops in float32 to avoid dtype issues under autocast
        p_norm = F.normalize(pred_latents.float(), p=2, dim=-1)
        c_norm = F.normalize(context_latents.float(), p=2, dim=-1)[:, :-1, :]
        k = min(k, c_norm.size(1))
        sim = torch.bmm(p_norm, c_norm.transpose(1, 2))
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=-1)
        weights = F.softmax(topk_sim * 10.0, dim=-1)
        next_bytes = context_ids[:, 1:].long()
        B, T_tgt, _ = pred_latents.shape
        knn_probs = torch.zeros(B, T_tgt, self.vocab_size, device=pred_latents.device, dtype=torch.float32)
        for i in range(k):
            idx = topk_idx[:, :, i]
            w = weights[:, :, i]
            byte_val = torch.gather(next_bytes, 1, idx)
            knn_probs.scatter_add_(2, byte_val.unsqueeze(-1), w.unsqueeze(-1))
        return knn_probs

    def score(self, context_ids: Tensor, target_ids: Tensor) -> Tensor:
        context_states = self.encode(context_ids)
        pred_latents = self.predict(context_states)
        logits = self.decode_logits(pred_latents, target_ids)
        
        if self.latent_knn_cache_enabled and not torch.is_grad_enabled():
            enc = self.ema_encode(context_ids, pos_offset=0)
            context_latents = self.ema_target_proj(enc)
            if self.target_patch_size > 1:
                pred_latents_upsampled = pred_latents.repeat_interleave(self.target_patch_size, dim=1)
            else:
                pred_latents_upsampled = pred_latents
            knn_probs = self.latent_knn_cache_logits(pred_latents_upsampled, context_latents, context_ids, k=self.latent_knn_cache_k)
            model_probs = F.softmax(logits.float(), dim=-1)
            blended = (1.0 - self.latent_knn_cache_alpha) * model_probs + self.latent_knn_cache_alpha * knn_probs
            true_log_probs = torch.log(torch.clamp(blended, min=1e-12)).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        else:
            true_log_probs = F.log_softmax(logits.float(), dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            
        if self.byte_logit_cache_enabled and not torch.is_grad_enabled():
            return causal_byte_logit_cache_ce(
                true_log_probs,
                context_ids,
                target_ids,
                vocab_size=self.vocab_size,
                min_n=self.byte_logit_cache_min_n,
                max_n=self.byte_logit_cache_max_n,
                alpha=self.byte_logit_cache_alpha,
                min_count=self.byte_logit_cache_min_count,
                count_scale=self.byte_logit_cache_count_scale,
            )
        return -true_log_probs.mean()

    def forward(self, batch_ids: Tensor, latent_only: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        context_ids = batch_ids[:, : self.context_seq_len]
        target_ids = batch_ids[:, self.context_seq_len :]
        context_states = self.encode(context_ids)
        pred_hidden = self.predict_hidden(context_states)
        pred_latents = self.pred_proj(pred_hidden)

        if latent_only:
            target_latents = self.target_latents(target_ids)
            target_latents_pooled = pool_sequence(target_latents, self.target_patch_size)
            pred_chunk = pred_latents.reshape(
                pred_latents.size(0), self.num_target_chunks, self.predict_chunk_patches, pred_latents.size(-1)
            ).mean(dim=2)
            target_chunk = target_latents_pooled.reshape(
                target_latents_pooled.size(0), self.num_target_chunks, self.predict_chunk_patches, target_latents_pooled.size(-1)
            ).mean(dim=2)

            pred_norm = F.normalize(pred_latents.float(), p=2, dim=-1)
            target_norm = F.normalize(target_latents_pooled.float(), p=2, dim=-1)
            token_latent_loss = F.mse_loss(pred_norm, target_norm)

            pred_chunk_norm = F.normalize(pred_chunk.float(), p=2, dim=-1)
            target_chunk_norm = F.normalize(target_chunk.float(), p=2, dim=-1)
            chunk_latent_loss = F.mse_loss(pred_chunk_norm, target_chunk_norm)

            latent_total = self.latent_loss_weight * token_latent_loss + self.chunk_latent_weight * chunk_latent_loss
            zero = latent_total.new_zeros(())
            return latent_total, zero.detach(), latent_total.detach(), zero.detach()

        logits = self.decode_logits(pred_latents, target_ids)
        ce_loss = F.cross_entropy(logits.float().reshape(-1, self.vocab_size), target_ids.reshape(-1), reduction="mean")
        zero = ce_loss.new_zeros(())

        if not self.use_target_latent_losses:
            ce_scale = self.decoder_aux_weight if self.lewm_style else self.ce_loss_weight
            return ce_scale * ce_loss, ce_loss.detach(), zero.detach(), zero.detach()

        # EMA target encoder provides stable prediction targets (no grad).
        target_latents = self.target_latents(target_ids)
        target_latents_pooled = pool_sequence(target_latents, self.target_patch_size)

        # Chunk-level latent matching: predict 1 vector per chunk instead of
        # 1 per token.  This makes the prediction task ~chunk_len times easier
        # (8 vectors instead of 256) so the predictor can actually learn.
        pred_chunk = pred_latents.reshape(
            pred_latents.size(0), self.num_target_chunks, self.predict_chunk_patches, pred_latents.size(-1)
        ).mean(dim=2)
        target_chunk = target_latents_pooled.reshape(
            target_latents_pooled.size(0), self.num_target_chunks, self.predict_chunk_patches, target_latents_pooled.size(-1)
        ).mean(dim=2)
        
        pred_norm = F.normalize(pred_latents.float(), p=2, dim=-1)
        target_norm = F.normalize(target_latents_pooled.float(), p=2, dim=-1)
        token_latent_loss = F.mse_loss(pred_norm, target_norm)
        
        pred_chunk_norm = F.normalize(pred_chunk.float(), p=2, dim=-1)
        target_chunk_norm = F.normalize(target_chunk.float(), p=2, dim=-1)
        chunk_latent_loss = F.mse_loss(pred_chunk_norm, target_chunk_norm)
        
        sig_loss = sigreg_loss(target_latents_pooled, 1.0)
        latent_total = self.latent_loss_weight * token_latent_loss + self.chunk_latent_weight * chunk_latent_loss
        total = latent_total + self.sigreg_lambda * sig_loss + self.decoder_aux_weight * ce_loss
        return total, ce_loss.detach(), latent_total.detach(), sig_loss.detach()

# -----------------------------
# EVALUATION
# -----------------------------

def eval_val(
    args: Hyperparameters,
    model: ByteJEPA,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    byte_lut: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    total_seq_len = args.train_seq_len
    if args.val_batch_size % (world_size * total_seq_len) != 0:
        raise ValueError(
            f"VAL_BATCH_SIZE={args.val_batch_size} must be divisible by WORLD_SIZE*TRAIN_SEQ_LEN={world_size * total_seq_len}"
        )
    local_batch_seqs = args.val_batch_size // (world_size * total_seq_len)
    if local_batch_seqs <= 0:
        raise ValueError("VAL_BATCH_SIZE is too small for even one sequence per rank")

    pad = torch.full((args.context_seq_len,), args.pad_id, dtype=val_tokens.dtype)
    padded = torch.cat((pad, val_tokens), dim=0)
    windows = padded.unfold(0, total_seq_len, args.target_seq_len)
    total_windows = int(windows.size(0))
    win_start = (total_windows * rank) // world_size
    win_end = (total_windows * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_start in range(win_start, win_end, local_batch_seqs):
            batch_end = min(batch_start + local_batch_seqs, win_end)
            batch = windows[batch_start:batch_end].to(device=device, dtype=torch.int64, non_blocking=True)
            context_ids = batch[:, : args.context_seq_len]
            target_ids = batch[:, args.context_seq_len :]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                batch_loss = model.score(context_ids, target_ids).detach()
            batch_token_count = float(target_ids.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            val_byte_count += byte_lut[target_ids].to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / max(val_byte_count.item(), 1.0)
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_ttt(
    args: Hyperparameters,
    model: ByteJEPA,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    byte_lut: Tensor,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Score-first TTT evaluation (legal protocol).

    For each chunk of val tokens:
      1. SCORE the chunk under inference_mode() — no gradients, no weight mutation
      2. ADAPT the model on the already-scored chunk using JEPA losses
         (latent prediction + CE) so later chunks benefit from adaptation

    Chunk N is scored by a model adapted only on chunks 0..N-1.
    The last chunk is scored but never adapted on.
    """
    total_seq_len = args.train_seq_len
    chunk_tokens = args.ttt_chunk_tokens

    # Pad and create sliding windows (same as eval_val)
    pad = torch.full((args.context_seq_len,), args.pad_id, dtype=val_tokens.dtype)
    padded = torch.cat((pad, val_tokens), dim=0)
    windows = padded.unfold(0, total_seq_len, args.target_seq_len)
    total_windows = int(windows.size(0))

    # Split windows into chunks for TTT
    windows_per_chunk = max(1, chunk_tokens // args.target_seq_len)
    num_chunks = (total_windows + windows_per_chunk - 1) // windows_per_chunk

    # Batch size for scoring (windows per forward pass)
    if args.val_batch_size % (world_size * total_seq_len) != 0:
        raise ValueError(
            f"VAL_BATCH_SIZE={args.val_batch_size} must be divisible by WORLD_SIZE*TRAIN_SEQ_LEN={world_size * total_seq_len}"
        )
    score_batch_seqs = args.val_batch_size // (world_size * total_seq_len)
    # TTT adapt batch must be much smaller than score batch (backward needs ~3x memory)
    adapt_batch_seqs = max(1, score_batch_seqs // 4)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Free memory before TTT
    torch.cuda.empty_cache()

    # Save original weights to restore after TTT if needed
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if not k.startswith("ema_")}

    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    future_path_prefixes = (
        "memory_seed.",
        "memory_blocks.",
        "memory_norm.",
        "future_queries",
        "rollout_step_emb",
        "query_seed.",
        "rollout_token_proj.",
        "predictor_blocks.",
        "predictor_norm.",
        "predictor_recur_",
        "pred_proj.",
        "decode_proj.",
        "decode_norm.",
        "decoder_blocks.",
        "decoder_recur_",
        "decoder_final_norm.",
    )
    late_encoder_start = min(max(args.ttt_freeze_encoder_blocks, 0), len(model.encoder_blocks))
    ttt_params = []
    for name, p in model.named_parameters():
        if name.startswith("ema_"):
            p.requires_grad_(False)
            continue
        is_future_path = name == "memory_queries" or name.startswith(future_path_prefixes)
        is_late_encoder = any(name.startswith(f"encoder_blocks.{idx}.") for idx in range(late_encoder_start, len(model.encoder_blocks)))
        if args.ttt_train_mode == "all":
            trainable = True
        elif args.ttt_train_mode == "predictor_decoder":
            trainable = is_future_path
        elif args.ttt_train_mode == "late":
            trainable = is_future_path or is_late_encoder or name == "encoder_norm.weight"
        else:
            raise ValueError(f"Unsupported TTT_TRAIN_MODE={args.ttt_train_mode}")
        p.requires_grad_(trainable)
        if trainable:
            ttt_params.append(p)
    if not ttt_params:
        raise ValueError("TTT selected zero trainable parameters")
    ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    for chunk_idx in range(num_chunks):
        win_start = chunk_idx * windows_per_chunk
        win_end = min(win_start + windows_per_chunk, total_windows)

        # Distribute windows across ranks
        chunk_wins = win_end - win_start
        rank_start = win_start + (chunk_wins * rank) // world_size
        rank_end = win_start + (chunk_wins * (rank + 1)) // world_size

        # ---- PHASE 1: SCORE under inference_mode() ----
        model.eval()
        chunk_windows_list = []
        with torch.inference_mode():
            for batch_start in range(rank_start, rank_end, score_batch_seqs):
                batch_end = min(batch_start + score_batch_seqs, rank_end)
                batch = windows[batch_start:batch_end].to(device=device, dtype=torch.int64, non_blocking=True)
                context_ids = batch[:, : args.context_seq_len]
                target_ids = batch[:, args.context_seq_len :]
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    batch_loss = model.score(context_ids, target_ids).detach()
                batch_token_count = float(target_ids.numel())
                val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
                val_token_count += batch_token_count
                val_byte_count += byte_lut[target_ids].to(torch.float64).sum()
                chunk_windows_list.append(batch.detach())

        # ---- PHASE 2: ADAPT on already-scored chunk ----
        # Skip adaptation on the last chunk (nothing left to score)
        if chunk_idx < num_chunks - 1 and chunk_windows_list:
            model.train()
            if num_chunks > 1:
                progress = chunk_idx / max(num_chunks - 2, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_scale = args.ttt_lr_min_frac + (1.0 - args.ttt_lr_min_frac) * cosine
            else:
                lr_scale = 1.0
            for group in ttt_opt.param_groups:
                group["lr"] = args.ttt_lr * lr_scale
            # Disable byte masking during TTT — we want clean signal, not regularization noise
            saved_mask_rate = model.decoder_byte_mask_rate
            model.decoder_byte_mask_rate = 0.0
            scored_batch = torch.cat(chunk_windows_list, dim=0)
            for _epoch in range(args.ttt_epochs):
                # Shuffle windows within the chunk
                perm = torch.randperm(scored_batch.size(0), device=device)
                for i in range(0, scored_batch.size(0), adapt_batch_seqs):
                    mini = scored_batch[perm[i : i + adapt_batch_seqs]]
                    ttt_opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=False):
                        loss, _, _, _ = model(mini, latent_only=args.ttt_pure_latent)
                    if not torch.isfinite(loss):
                        continue  # skip NaN/inf mini-batches
                    loss.backward()
                    if args.ttt_grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        if not torch.isfinite(grad_norm):
                            ttt_opt.zero_grad(set_to_none=True)
                            continue
                    ttt_opt.step()
            model.decoder_byte_mask_rate = saved_mask_rate

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / max(val_byte_count.item(), 1.0)

    # Restore original weights (TTT is eval-time only, don't pollute saved model)
    model.load_state_dict(original_state, strict=False)
    model._sync_ema_encoder()
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad[name])
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    if args.vocab_size < args.byte_offset + args.byte_count:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} is too small for byte range [{args.byte_offset}, {args.byte_offset + args.byte_count})"
        )
    if args.context_seq_len <= 0:
        raise ValueError("TRAIN_SEQ_LEN must be greater than TARGET_SEQ_LEN")
    if args.context_pool_stride <= 0:
        raise ValueError("CONTEXT_POOL_STRIDE must be positive")
    if args.byte_logit_cache_min_n <= 0:
        raise ValueError("BYTE_LOGIT_CACHE_MIN_N must be positive")
    if args.byte_logit_cache_max_n < args.byte_logit_cache_min_n:
        raise ValueError("BYTE_LOGIT_CACHE_MAX_N must be >= BYTE_LOGIT_CACHE_MIN_N")
    if not (0.0 <= args.byte_logit_cache_alpha <= 1.0):
        raise ValueError("BYTE_LOGIT_CACHE_ALPHA must be in [0, 1]")
    if args.byte_logit_cache_min_count <= 0:
        raise ValueError("BYTE_LOGIT_CACHE_MIN_COUNT must be positive")
    if args.byte_logit_cache_count_scale <= 0.0:
        raise ValueError("BYTE_LOGIT_CACHE_COUNT_SCALE must be positive")
    if not (0.0 <= args.latent_knn_cache_alpha <= 1.0):
        raise ValueError("LATENT_KNN_CACHE_ALPHA must be in [0, 1]")
    if args.latent_knn_cache_k <= 0:
        raise ValueError("LATENT_KNN_CACHE_K must be positive")

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = env_world_size > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    world_size = env_world_size if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    default_grad_accum_steps = max(8 // world_size, 1) if 8 % world_size == 0 else 1
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", str(default_grad_accum_steps)))
    if grad_accum_steps <= 0:
        raise ValueError(f"GRAD_ACCUM_STEPS must be positive, got {grad_accum_steps}")
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
        dist.barrier()
    master_process = rank == 0

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = get_amp_dtype()

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
            try:
                with open(logfile, "a", encoding="utf-8") as f:
                    print(msg, file=f)
            except OSError as exc:
                print(f"log_write_failed:{exc}", file=sys.stderr)

    def safe_nvidia_smi() -> str:
        try:
            return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout
        except FileNotFoundError:
            return "nvidia-smi not found"

    log0(f"code_path:{Path(__file__).resolve()}", console=False)
    log0(f"code_bytes:{len(code.encode('utf-8'))}", console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(safe_nvidia_smi(), console=False)
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if actual_train_files == 0:
        raise FileNotFoundError(
            f"No byte shards found under {dataset_dir}. Download them with: python data/cached_challenge_fineweb.py --variant byte260"
        )
    sample_shard = load_data_shard(next(iter(sorted(dataset_dir.glob("fineweb_train_*.bin")))))
    sample_token_max = int(sample_shard.to(dtype=torch.int64).max().item())
    if sample_token_max >= args.vocab_size:
        raise ValueError(
            f"Dataset token id max={sample_token_max} exceeds VOCAB_SIZE={args.vocab_size}. "
            "This script expects the byte260 export, not the SentencePiece shards."
        )

    val_tokens = load_validation_tokens(args.val_files, args.target_seq_len)
    periodic_val_tokens = cap_validation_tokens(val_tokens, args.max_val_tokens, args.target_seq_len)
    final_val_tokens = val_tokens if args.final_full_val else cap_validation_tokens(
        val_tokens, args.final_max_val_tokens, args.target_seq_len
    )
    byte_lut = build_byte_lut(args.vocab_size, args.byte_offset, args.byte_count, device)
    log0(
        f"val_bpb:enabled tokenizer_kind=byte vocab_size:{args.vocab_size} byte_offset:{args.byte_offset} byte_count:{args.byte_count}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    final_eval_mode = "full" if args.final_full_val else "capped"
    log0(
        f"val_loader:shards pattern={args.val_files} full_tokens:{val_tokens.numel()} "
        f"periodic_tokens:{periodic_val_tokens.numel()} final_tokens:{final_val_tokens.numel()} "
        f"final_eval_mode:{final_eval_mode}"
    )

    base_model = ByteJEPA(args).to(device)
    zeropower_via_newtonschulz5 = maybe_compile(
        zeropower_via_newtonschulz5,
        enabled=args.use_compile,
        fullgraph=args.compile_fullgraph,
    )
    train_model = maybe_compile(base_model, enabled=args.use_compile, fullgraph=args.compile_fullgraph)
    if distributed:
        model = DDP(
            train_model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=not base_model.use_target_latent_losses,
        )
    else:
        model = train_model

    named_params = [(n, p) for n, p in base_model.named_parameters() if p.requires_grad]
    embed_param_names = {"byte_emb.weight", "pos_emb", "memory_queries", "future_queries", "rollout_step_emb"}
    control_param_name_patterns = ("_recur_scale", "_recur_bias")
    embed_params = [p for name, p in named_params if name in embed_param_names]
    matrix_params = [
        p
        for name, p in named_params
        if p.ndim == 2 and name not in embed_param_names and not any(pattern in name for pattern in control_param_name_patterns)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if p.ndim < 2 or any(pattern in name for pattern in control_param_name_patterns)
    ]

    optimizers: list[torch.optim.Optimizer] = []
    if embed_params:
        optimizer_embed = make_adam(
            [{"params": embed_params, "lr": args.embed_lr, "base_lr": args.embed_lr}],
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.adam_eps,
        )
        optimizers.append(optimizer_embed)
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizers.append(optimizer_muon)
    if scalar_params:
        optimizer_scalar = make_adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.adam_eps,
        )
        optimizers.append(optimizer_scalar)

    n_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    microbatch_denom = world_size * grad_accum_steps * args.train_seq_len
    microbatch_seqs = (
        args.train_batch_tokens // microbatch_denom
        if args.train_batch_tokens % microbatch_denom == 0
        else -1
    )
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps} amp_dtype:{str(amp_dtype).removeprefix('torch.')}")
    if microbatch_seqs > 0:
        log0(f"microbatch_seqs_per_rank:{microbatch_seqs}")
    log0(f"compile_enabled:{args.use_compile} compile_fullgraph:{args.compile_fullgraph}")
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} context_seq_len:{args.context_seq_len} "
        f"target_seq_len:{args.target_seq_len} predict_chunk_len:{args.predict_chunk_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps}"
    )
    log0(
        f"model_dim:{args.model_dim} latent_dim:{args.latent_dim} encoder_layers:{args.encoder_layers} "
        f"predictor_layers:{args.predictor_layers} heads:{args.num_heads} context_pool_stride:{args.context_pool_stride} "
        f"target_patch_size:{args.target_patch_size} memory_tokens:{args.memory_tokens} memory_layers:{args.memory_layers} "
        f"decoder_layers:{args.decoder_layers} cross_decoder:{int(args.use_cross_decoder)}"
    )
    log0(
        f"byte_logit_cache_enabled:{int(args.byte_logit_cache_enabled)} "
        f"byte_logit_cache_ngram:{args.byte_logit_cache_min_n}-{args.byte_logit_cache_max_n} "
        f"byte_logit_cache_alpha:{args.byte_logit_cache_alpha:.3f} "
        f"byte_logit_cache_min_count:{args.byte_logit_cache_min_count} "
        f"byte_logit_cache_count_scale:{args.byte_logit_cache_count_scale:.1f}"
    )
    log0(
        f"latent_knn_cache_enabled:{int(args.latent_knn_cache_enabled)} "
        f"latent_knn_cache_k:{args.latent_knn_cache_k} "
        f"latent_knn_cache_alpha:{args.latent_knn_cache_alpha:.3f}"
    )
    log0(
        f"mlp_activation:{args.mlp_activation} predictor_recur_layers:{args.predictor_recur_layers} "
        f"predictor_recur_passes:{args.predictor_recur_passes} decoder_recur_layers:{args.decoder_recur_layers} "
        f"decoder_recur_passes:{args.decoder_recur_passes}"
    )
    log0(
        f"loss_weights ce:{args.ce_loss_weight} latent:{args.latent_loss_weight} sigreg:{args.sigreg_weight} cov:{args.sigreg_cov_weight}"
    )
    log0(
        f"lewm_style:{int(args.lewm_style)} decoder_aux_weight:{args.decoder_aux_weight:.3f} "
        f"sigreg_lambda:{args.sigreg_lambda:.3f} chunk_latent_weight:{args.chunk_latent_weight:.3f} "
        f"decoder_byte_mask_rate:{args.decoder_byte_mask_rate:.3f} ema_decay:{args.ema_decay:.4f}"
    )
    log0(
        f"ttt_enabled:{int(args.ttt_enabled)} ttt_train_mode:{args.ttt_train_mode} "
        f"ttt_freeze_encoder_blocks:{args.ttt_freeze_encoder_blocks} ttt_lr_min_frac:{args.ttt_lr_min_frac:.3f} "
        f"ttt_pure_latent:{int(args.ttt_pure_latent)}"
    )
    log0(f"seed:{args.seed}")

    train_loader = DistributedSequenceLoader(args.train_files, rank, world_size, device)

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
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                batch = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    warmup_loss, _, _, _ = model(batch)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            base_model.zero_pad_embedding_()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        base_model._sync_ema_encoder()  # resync EMA after warmup reset
        base_model.zero_pad_embedding_()
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedSequenceLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            eval_tokens = final_val_tokens if last_step else periodic_val_tokens
            val_loss, val_bpb = eval_val(args, base_model, rank, world_size, device, eval_tokens, byte_lut, amp_dtype)
            val_scope = "final_full" if last_step and args.final_full_val else "final_capped" if last_step else "periodic"
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"val_scope:{val_scope} val_tokens:{eval_tokens.numel()} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_ce = torch.zeros((), device=device)
        train_latent = torch.zeros((), device=device)
        train_sig = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            batch = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                loss, ce_loss, latent_loss, sig_loss = model(batch)
            train_loss += loss.detach()
            train_ce += ce_loss
            train_latent += latent_loss
            train_sig += sig_loss
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_ce /= grad_accum_steps
        train_latent /= grad_accum_steps
        train_sig /= grad_accum_steps

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
        base_model.update_ema()
        base_model.zero_pad_embedding_()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_ce:{train_ce.item():.4f} "
                f"train_latent:{train_latent.item():.4f} train_sigreg:{train_sig.item():.4f} "
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

    if master_process:
        torch.save(base_model.serializable_state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.serializable_state_dict())
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
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_serializable_state_dict(dequantize_state_dict_int8(quant_state))
    base_model.zero_pad_embedding_()

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, base_model, rank, world_size, device, final_val_tokens, byte_lut, amp_dtype)
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_scope:{final_eval_mode} val_tokens:{final_val_tokens.numel()} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
        f"eval_scope:{final_eval_mode} val_tokens:{final_val_tokens.numel()}"
    )

    # TTT: score-first, adapt-second on the quantized model
    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_ttt(
            args, base_model, rank, world_size, device,
            final_val_tokens, byte_lut, amp_dtype,
        )
        torch.cuda.synchronize()
        ttt_time_ms = 1000.0 * (time.perf_counter() - t_ttt)
        log0(
            f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"ttt_gain:{ttt_bpb - q_val_bpb:.4f} "
            f"eval_scope:{final_eval_mode} val_tokens:{final_val_tokens.numel()} "
            f"ttt_time:{ttt_time_ms:.0f}ms"
        )
        log0(
            f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f} "
            f"eval_scope:{final_eval_mode} val_tokens:{final_val_tokens.numel()}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
