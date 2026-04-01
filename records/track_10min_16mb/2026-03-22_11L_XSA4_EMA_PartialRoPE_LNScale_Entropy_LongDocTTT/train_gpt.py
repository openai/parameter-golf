"""
Frontier candidate forked from PR315: 11L XSA4 + EMA + Partial RoPE + LN Scale,
with a custom packed export codec and gated long-document LoRA TTT.
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
from contextlib import nullcontext
from pathlib import Path

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    flash_attn_3_func = None
from custom_entropy_codec import decode_quantized_state as decode_custom_quantized_state
from custom_entropy_codec import encode_quantized_state as encode_custom_quantized_state

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
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 9000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    train_loader_mode = os.environ.get("TRAIN_LOADER_MODE", "stream")
    doc_boundary_bias = float(os.environ.get("DOC_BOUNDARY_BIAS", 0.0))
    allow_local_device_fallback = bool(int(os.environ.get("ALLOW_LOCAL_DEVICE_FALLBACK", "0")))
    local_device = os.environ.get("LOCAL_DEVICE", "").strip().lower()
    local_disable_compile = bool(int(os.environ.get("LOCAL_DISABLE_COMPILE", "0")))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    mtp_num_heads = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))
    muon_beta2 = float(os.environ.get("MUON_BETA2", 0.95))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_every = int(os.environ.get("SWA_EVERY", 200))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "0")))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat = bool(int(os.environ.get("LATE_QAT", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    export_codec = os.environ.get("EXPORT_CODEC", _COMPRESSOR)
    compare_export_codecs = bool(int(os.environ.get("COMPARE_EXPORT_CODECS", "0")))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_doc_percentile = float(os.environ.get("TTT_DOC_PERCENTILE", 95.0))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 4))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 16))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 2048))
    ttt_target_last_n = int(os.environ.get("TTT_TARGET_LAST_N", 2))
    ttt_min_pred_tokens = int(os.environ.get("TTT_MIN_PRED_TOKENS", 0))
    ttt_max_docs = int(os.environ.get("TTT_MAX_DOCS", 0))
    ttt_max_eval_seconds = float(os.environ.get("TTT_MAX_EVAL_SECONDS", 0.0))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    work_dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32
    X = G.to(dtype=work_dtype)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay),
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
            update_dtype = torch.bfloat16 if params[0].device.type == "cuda" else torch.float32
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=update_dtype)

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

            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
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


def device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def autocast_context(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
    return nullcontext()


def maybe_compile(obj, enabled: bool):
    return torch.compile(obj, dynamic=False, fullgraph=True) if enabled else obj


def metric_accum_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


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


def load_all_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return torch.cat([load_data_shard(file) for file in files]).contiguous()


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    accum_dtype = metric_accum_dtype(device)
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    val_token_count = torch.zeros((), device=device, dtype=accum_dtype)
    val_byte_count = torch.zeros((), device=device, dtype=accum_dtype)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with autocast_context(device, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(accum_dtype) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(accum_dtype).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we quantize aggressively, then serialize through either the legacy monolithic
# compressor path or the new structured custom-packed codec before roundtrip evaluation.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear",
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
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
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

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
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
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


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


def find_document_spans(tokens: Tensor, bos_token_id: int = 1) -> list[tuple[int, int]]:
    bos_positions = (tokens == bos_token_id).nonzero(as_tuple=False).flatten()
    if bos_positions.numel() == 0:
        return [(0, int(tokens.numel()))]
    docs: list[tuple[int, int]] = []
    for i, start in enumerate(bos_positions.tolist()):
        end = int(bos_positions[i + 1]) if i + 1 < bos_positions.numel() else int(tokens.numel())
        if end - int(start) >= 2:
            docs.append((int(start), end))
    return docs


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
    def __init__(
        self,
        pattern: str,
        rank: int,
        world_size: int,
        device: torch.device,
        sample_mode: str = "stream",
        doc_boundary_bias: float = 0.0,
        bos_token_id: int = 1,
        seed: int = 0,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
        self.sample_mode = sample_mode
        self.doc_boundary_bias = min(max(doc_boundary_bias, 0.0), 1.0)
        self.bos_token_id = bos_token_id
        self.rng = np.random.default_rng(seed)
        self.sampled_tokens_from_shard = 0
        self.docs = find_document_spans(self.stream.tokens, bos_token_id=self.bos_token_id)

    def _advance_random_file(self) -> None:
        self.stream._advance_file()
        self.docs = find_document_spans(self.stream.tokens, bos_token_id=self.bos_token_id)
        self.sampled_tokens_from_shard = 0

    def _sample_doc_window(self, window_len: int) -> Tensor:
        candidates = [(start, end) for start, end in self.docs if end - start >= window_len]
        if not candidates:
            return self.stream.take(window_len)
        weights = np.array([end - start - window_len + 1 for start, end in candidates], dtype=np.float64)
        weights = np.maximum(weights, 1.0)
        weights /= weights.sum()
        doc_idx = int(self.rng.choice(len(candidates), p=weights))
        doc_start, doc_end = candidates[doc_idx]
        if self.doc_boundary_bias > 0.0 and self.rng.random() < self.doc_boundary_bias:
            start = doc_start
        else:
            max_start = doc_end - window_len
            start = doc_start if max_start <= doc_start else int(self.rng.integers(doc_start, max_start + 1))
        window = self.stream.tokens[start : start + window_len].clone()
        self.sampled_tokens_from_shard += window_len
        if self.sampled_tokens_from_shard + window_len > self.stream.tokens.numel() and len(self.stream.files) > 1:
            self._advance_random_file()
        return window

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        if self.sample_mode == "stream":
            chunk = self.stream.take(per_rank_span * self.world_size)
            start = self.rank * per_rank_span
            local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        if self.sample_mode != "doc_random":
            raise ValueError(f"Unsupported TRAIN_LOADER_MODE={self.sample_mode}")
        batch_size = local_tokens // seq_len
        local = torch.stack([self._sample_doc_window(seq_len + 1) for _ in range(batch_size)], dim=0).to(dtype=torch.int64)
        x = local[:, :-1]
        y = local[:, 1:]
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
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # NTK-aware RoPE: auto-scales base frequency when seq_len exceeds train_seq_len.
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        rd = self.rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._cache_built_without_grad = False

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
            self._cache_built_without_grad = not torch.is_grad_enabled()
        cos_cached = self._cos_cached
        sin_cached = self._sin_cached
        # TTT mixes inference-only eval passes with later gradient-tracked LoRA updates.
        # If RoPE caches were first materialized under inference mode, clone them before
        # reusing them in autograd-enabled code.
        if torch.is_grad_enabled() and self._cache_built_without_grad:
            cos_cached = cos_cached.clone()
            sin_cached = sin_cached.clone()
            self._cos_cached = cos_cached
            self._sin_cached = sin_cached
            self._cache_built_without_grad = False
        return cos_cached.to(dtype=dtype), sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = cos.size(-1) * 2
    if rd < x.size(-1):
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
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
        qk_gain_init: float,
        rope_dims: int = 0,
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
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)
        self.use_xsa = False

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection via GQA-aware reshape (no repeat_interleave)."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, q_delta: Tensor | None = None, v_delta: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x)
        if q_delta is not None:
            q = q + q_delta
        if v_delta is not None:
            v = v + v_delta
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if flash_attn_3_func is not None:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            q_sdpa = q.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                repeat = self.num_heads // self.num_kv_heads
                k_sdpa = k_sdpa.repeat_interleave(repeat, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(repeat, dim=1)
            y = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=None,
                is_causal=True,
            ).transpose(1, 2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        layer_idx: int = 0,
        ln_scale: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor
        norm_x = self.attn_norm(x) * s
        q_delta = q_delta_fn(norm_x) if q_delta_fn is not None else None
        v_delta = v_delta_fn(norm_x) if v_delta_fn is not None else None
        attn_out = self.attn(norm_x, q_delta=q_delta, v_delta=v_delta)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
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
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.1,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.mtp_num_heads = mtp_num_heads
        self.mtp_loss_weight = mtp_loss_weight
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    rope_dims=rope_dims,
                    layer_idx=i,
                    ln_scale=ln_scale,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)]
        )
        for head in self.mtp_heads:
            head._zero_init = True
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            qd = lora.q_module(i) if lora is not None else None
            vd = lora.v_module(i) if lora is not None else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            qd = lora.q_module(block_idx) if lora is not None else None
            vd = lora.v_module(block_idx) if lora is not None else None
            x = self.blocks[block_idx](x, x0, qd, vd)

        x = self.final_norm(x)
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        if lora is not None:
            logits_proj = logits_proj + lora.lm_head_lora(x).reshape(-1, logits_proj.size(-1))
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        if lora is not None:
            return F.cross_entropy(logits.float(), targets, reduction="none").reshape_as(target_ids)

        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            mtp_loss_count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_hidden = x[:, :valid_t, :].reshape(-1, dim)
                mtp_targets = target_ids[:, k + 1 :].reshape(-1)
                mtp_logits_proj = mtp_head(mtp_hidden)
                mtp_logits = self.logit_softcap * torch.tanh(mtp_logits_proj / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_logits.float(), mtp_targets, reduction="mean")
                mtp_loss_count += 1
            if mtp_loss_count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / mtp_loss_count)

        return main_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# SLIDING WINDOW EVALUATION
# -----------------------------

def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = eval_seq_len or args.train_seq_len
    accum_dtype = metric_accum_dtype(device)
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    token_count = torch.zeros((), device=device, dtype=accum_dtype)
    byte_count = torch.zeros((), device=device, dtype=accum_dtype)

    base_model.eval()
    # Adaptive eval batches docs by similar size, but max_wlen still changes across chunks.
    # Compiling forward_logits with fullgraph=True causes Dynamo to exhaust its
    # recompile budget on these variable-length shapes, so keep this path eager.
    compiled_logits = base_model.forward_logits

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with autocast_context(device):
                logits = compiled_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(accum_dtype)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(accum_dtype)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(accum_dtype)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


BOS_ID = 1


class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class TargetedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int, target_block_ids: list[int]):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleDict()
        self.v_loras = nn.ModuleDict()
        self.target_block_ids = tuple(target_block_ids)
        for block_id in self.target_block_ids:
            block = model.blocks[block_id]
            self.q_loras[str(block_id)] = BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank)
            self.v_loras[str(block_id)] = BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank)

    def reset(self) -> None:
        self.lm_head_lora.reset()
        for module in self.q_loras.values():
            module.reset()
        for module in self.v_loras.values():
            module.reset()

    def q_module(self, block_id: int):
        key = str(block_id)
        return self.q_loras[key] if key in self.q_loras else None

    def v_module(self, block_id: int):
        key = str(block_id)
        return self.v_loras[key] if key in self.v_loras else None


def _reset_ttt_optimizer(opt) -> None:
    for group in opt.param_groups:
        for param in group["params"]:
            state = opt.state.get(param)
            if not state:
                continue
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            state["step"].fill_(0)


def _build_ttt_optimizer(lora: TargetedTTTLoRA, args: Hyperparameters):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)


def _find_docs(all_tokens: Tensor, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _split_docs_by_percentile(
    docs: list[tuple[int, int]],
    percentile: float,
    min_pred_tokens: int = 0,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    if not docs:
        return [], [], 0
    pred_lengths = np.asarray([doc_len - 1 for _, doc_len in docs], dtype=np.int32)
    threshold = int(np.quantile(pred_lengths, percentile / 100.0))
    if min_pred_tokens > 0:
        threshold = max(threshold, int(min_pred_tokens))
    short_docs = [doc for doc in docs if doc[1] - 1 < threshold]
    long_docs = [doc for doc in docs if doc[1] - 1 >= threshold]
    if not long_docs:
        long_docs = [max(docs, key=lambda doc: doc[1])]
        short_docs = [doc for doc in docs if doc not in long_docs]
        threshold = long_docs[0][1] - 1
    return short_docs, long_docs, threshold


def _compute_chunk_window(chunk_idx: int, pred_len: int, chunk_size: int, eval_seq_len: int) -> tuple[int, int, int, int]:
    chunk_start = chunk_idx * chunk_size
    chunk_end = min(pred_len, (chunk_idx + 1) * chunk_size)
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _accumulate_bpb(
    nll: Tensor,
    x: Tensor,
    y: Tensor,
    batch_i: int,
    chunk_offset: int,
    chunk_len: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    loss_sum: Tensor,
    byte_sum: Tensor,
    token_count: Tensor,
) -> None:
    accum_dtype = loss_sum.dtype
    labels = nll[batch_i, chunk_offset : chunk_offset + chunk_len].to(accum_dtype)
    prev = x[batch_i, chunk_offset : chunk_offset + chunk_len]
    tgt = y[batch_i, chunk_offset : chunk_offset + chunk_len]
    token_bytes = base_bytes_lut[tgt].to(accum_dtype)
    token_bytes += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(accum_dtype)
    loss_sum += labels.sum()
    byte_sum += token_bytes.sum()
    token_count += chunk_len


def _eval_docs_no_ttt(
    args: Hyperparameters,
    compiled_logits,
    docs: list[tuple[int, int]],
    all_tokens: Tensor,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    accum_dtype = metric_accum_dtype(device)
    loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    byte_sum = torch.zeros((), device=device, dtype=accum_dtype)
    token_count = torch.zeros((), device=device, dtype=accum_dtype)
    if not docs:
        return loss_sum, byte_sum, token_count

    docs = sorted(docs, key=lambda doc: doc[1])
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len

    with torch.inference_mode():
        for start in range(0, len(docs), args.ttt_batch_size):
            batch = docs[start : start + args.ttt_batch_size]
            pred_lens = [doc_len - 1 for _, doc_len in batch]
            num_chunks = [(pred_len + chunk_size - 1) // chunk_size for pred_len in pred_lens]
            for chunk_idx in range(max(num_chunks)):
                batch_meta = []
                max_wlen = 0
                for doc_i, (doc_start, _) in enumerate(batch):
                    if chunk_idx >= num_chunks[doc_i]:
                        batch_meta.append(None)
                        continue
                    win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                        chunk_idx, pred_lens[doc_i], chunk_size, eval_seq_len
                    )
                    batch_meta.append((doc_start, win_start, win_len, chunk_offset, chunk_len))
                    max_wlen = max(max_wlen, win_len)
                x = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
                y = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
                for doc_i, meta in enumerate(batch_meta):
                    if meta is None:
                        continue
                    doc_start, win_start, win_len, _, _ = meta
                    chunk = all_tokens[doc_start + win_start : doc_start + win_start + win_len + 1]
                    toks = chunk.to(dtype=torch.int64, device=device)
                    x[doc_i, :win_len] = toks[:-1]
                    y[doc_i, :win_len] = toks[1:]
                with autocast_context(device):
                    logits = compiled_logits(x)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y.reshape(-1),
                    reduction="none",
                ).reshape(len(batch), max_wlen)
                for doc_i, meta in enumerate(batch_meta):
                    if meta is None:
                        continue
                    _, _, _, chunk_offset, chunk_len = meta
                    _accumulate_bpb(
                        nll, x, y, doc_i, chunk_offset, chunk_len,
                        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        loss_sum, byte_sum, token_count,
                    )
    return loss_sum, byte_sum, token_count


def _eval_docs_ttt(
    args: Hyperparameters,
    base_model: GPT,
    compiled_logits,
    docs: list[tuple[int, int]],
    all_tokens: Tensor,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    accum_dtype = metric_accum_dtype(device)
    loss_sum = torch.zeros((), device=device, dtype=accum_dtype)
    byte_sum = torch.zeros((), device=device, dtype=accum_dtype)
    token_count = torch.zeros((), device=device, dtype=accum_dtype)
    if not docs:
        return loss_sum, byte_sum, token_count

    docs = sorted(docs, key=lambda doc: doc[1], reverse=True)
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    deadline = time.perf_counter() + args.ttt_max_eval_seconds if args.ttt_max_eval_seconds > 0 else None
    target_block_ids = list(range(max(0, len(base_model.blocks) - args.ttt_target_last_n), len(base_model.blocks)))
    reusable_lora = TargetedTTTLoRA(args.ttt_batch_size, base_model, args.ttt_lora_rank, target_block_ids).to(device)
    reusable_opt = _build_ttt_optimizer(reusable_lora, args)

    for start in range(0, len(docs), args.ttt_batch_size):
        if deadline is not None and time.perf_counter() >= deadline:
            rem_loss, rem_bytes, rem_tokens = _eval_docs_no_ttt(
                args, compiled_logits, docs[start:], all_tokens, device,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            loss_sum += rem_loss
            byte_sum += rem_bytes
            token_count += rem_tokens
            break
        batch = docs[start : start + args.ttt_batch_size]
        if len(batch) == args.ttt_batch_size:
            lora = reusable_lora
            opt = reusable_opt
            lora.reset()
            _reset_ttt_optimizer(opt)
        else:
            lora = TargetedTTTLoRA(len(batch), base_model, args.ttt_lora_rank, target_block_ids).to(device)
            opt = _build_ttt_optimizer(lora, args)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pred_len + chunk_size - 1) // chunk_size for pred_len in pred_lens]
        for chunk_idx in range(max(num_chunks)):
            batch_meta = []
            max_wlen = 0
            train_terms = []
            x = None
            y = None
            for doc_i, (doc_start, _) in enumerate(batch):
                if chunk_idx >= num_chunks[doc_i]:
                    batch_meta.append(None)
                    continue
                win_start, win_len, chunk_offset, chunk_len = _compute_chunk_window(
                    chunk_idx, pred_lens[doc_i], chunk_size, eval_seq_len
                )
                batch_meta.append((doc_start, win_start, win_len, chunk_offset, chunk_len))
                max_wlen = max(max_wlen, win_len)
            x = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            y = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            for doc_i, meta in enumerate(batch_meta):
                if meta is None:
                    continue
                doc_start, win_start, win_len, _, _ = meta
                chunk = all_tokens[doc_start + win_start : doc_start + win_start + win_len + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[doc_i, :win_len] = toks[:-1]
                y[doc_i, :win_len] = toks[1:]

            needs_train = any(chunk_idx < num_chunks[doc_i] - 1 for doc_i in range(len(batch)))
            if needs_train:
                with autocast_context(device):
                    ptl = base_model(x, y, lora=lora)
            else:
                with torch.no_grad(), autocast_context(device):
                    ptl = base_model(x, y, lora=lora)

            with torch.no_grad():
                for doc_i, meta in enumerate(batch_meta):
                    if meta is None:
                        continue
                    _, _, _, chunk_offset, chunk_len = meta
                    _accumulate_bpb(
                        ptl, x, y, doc_i, chunk_offset, chunk_len,
                        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        loss_sum, byte_sum, token_count,
                    )

            if needs_train:
                for doc_i, meta in enumerate(batch_meta):
                    if meta is None or chunk_idx >= num_chunks[doc_i] - 1:
                        continue
                    _, _, _, chunk_offset, chunk_len = meta
                    train_terms.append(ptl[doc_i, chunk_offset : chunk_offset + chunk_len].mean())
                if train_terms:
                    opt.zero_grad()
                    torch.stack(train_terms).mean().backward()
                    opt.step()

    return loss_sum, byte_sum, token_count


def eval_val_adaptive_longdoc_ttt(
    args: Hyperparameters,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float, int, int]:
    all_tokens = load_all_validation_tokens(args.val_files)
    docs = _find_docs(all_tokens)
    short_docs, long_docs, threshold = _split_docs_by_percentile(
        docs,
        args.ttt_doc_percentile,
        args.ttt_min_pred_tokens,
    )
    if args.ttt_max_docs > 0 and len(long_docs) > args.ttt_max_docs:
        long_docs = sorted(long_docs, key=lambda doc: doc[1], reverse=True)
        kept_long_docs = long_docs[: args.ttt_max_docs]
        fallback_docs = long_docs[args.ttt_max_docs :]
        short_docs = short_docs + fallback_docs
        long_docs = kept_long_docs
    rank_short_docs = short_docs[(len(short_docs) * rank) // world_size : (len(short_docs) * (rank + 1)) // world_size]
    rank_long_docs = long_docs[(len(long_docs) * rank) // world_size : (len(long_docs) * (rank + 1)) // world_size]

    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad_(False)
    # Adaptive eval sees variable-length chunk windows across docs, which causes
    # repeated recompiles in torch.compile(fullgraph=True). Keep this path eager
    # so final long-doc TTT eval finishes reliably under the challenge budget.
    compiled_logits = base_model.forward_logits

    short_loss, short_bytes, short_tokens = _eval_docs_no_ttt(
        args, compiled_logits, rank_short_docs, all_tokens, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    long_loss, long_bytes, long_tokens = _eval_docs_ttt(
        args, base_model, compiled_logits, rank_long_docs, all_tokens, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )

    loss_sum = short_loss + long_loss
    byte_sum = short_bytes + long_bytes
    token_count = short_tokens + long_tokens

    selected_docs = torch.tensor(len(long_docs), device=device, dtype=torch.int64)
    threshold_tensor = torch.tensor(threshold, device=device, dtype=torch.int64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(selected_docs, op=dist.ReduceOp.MAX)
        dist.all_reduce(threshold_tensor, op=dist.ReduceOp.MAX)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    base_model.train()
    return val_loss, val_bpb, int(selected_docs.item()), int(threshold_tensor.item())


# -----------------------------
# INT6 MIXED QUANTIZATION (transplanted from working diagnostic scripts)
# -----------------------------

def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"

def quantize_int6_per_row(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 31.0).clamp_min(1.0 / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -32, 31).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 31.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -32, 31).to(torch.int8)
    return q, scale

def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))

    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        # tok_emb.weight falls through to int8 via "embed" category
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    return result, meta

def dequantize_mixed_int6(result: dict[str, Tensor], meta: dict[str, object],
                          template_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code_paths = [Path(__file__), Path(__file__).with_name("custom_entropy_codec.py")]
    code_chunks = [path.read_text(encoding="utf-8") for path in code_paths if path.exists()]
    code = "\n\n".join(code_chunks)
    code_bytes = sum(len(chunk.encode("utf-8")) for chunk in code_chunks)
    args = Hyperparameters()

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
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        if not args.allow_local_device_fallback:
            raise RuntimeError("CUDA is required unless ALLOW_LOCAL_DEVICE_FALLBACK=1")
        if distributed or world_size != 1:
            raise RuntimeError("Local device fallback only supports single-process runs")
        if args.local_device == "cpu":
            device = torch.device("cpu")
        elif args.local_device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("LOCAL_DEVICE=mps requested but MPS is not available")
            device = torch.device("mps")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    compile_enabled = device.type == "cuda" and not args.local_disable_compile
    zeropower_via_newtonschulz5 = maybe_compile(zeropower_via_newtonschulz5, enabled=compile_enabled)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)

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
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    if device.type == "cuda":
        log0(
            subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
            console=False,
        )
    else:
        log0(f"local_device_fallback:{device.type}", console=False)
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
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
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    CastedLinear._qat_enabled = args.qat_enabled

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mtp_num_heads=args.mtp_num_heads,
        mtp_loss_weight=args.mtp_loss_weight,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
    ).to(device)
    if device.type == "cuda":
        base_model = base_model.bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = maybe_compile(base_model, enabled=compile_enabled)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.mtp_num_heads > 0:
        matrix_params.extend([p for p in base_model.mtp_heads.parameters() if p.ndim == 2])
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=device.type == "cuda",
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=device.type == "cuda",
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=device.type == "cuda",
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    mtp_params = sum(p.numel() for p in base_model.mtp_heads.parameters())
    log0(f"model_params:{n_params}")
    log0(f"mtp_num_heads:{args.mtp_num_heads} mtp_loss_weight:{args.mtp_loss_weight} mtp_params:{mtp_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"train_loader_mode:{args.train_loader_mode} doc_boundary_bias:{args.doc_boundary_bias:.3f}")
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(
        args.train_files,
        rank,
        world_size,
        device,
        sample_mode=args.train_loader_mode,
        doc_boundary_bias=args.doc_boundary_bias,
        bos_token_id=max(int(sp.bos_id()), 1),
        seed=args.seed + rank,
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
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

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
                with autocast_context(device, enabled=True):
                    warmup_loss = model(x, y)
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
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    swa_state: dict[str, Tensor] | None = None
    swa_count = 0

    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}

    training_time_ms = 0.0
    stop_after_step: int | None = None
    device_sync(device)
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            device_sync(device)
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            device_sync(device)
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        qat_threshold = float(os.environ.get("QAT_THRESHOLD", "0.1"))
        if args.late_qat and scale < qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with autocast_context(device, enabled=True):
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

        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        if args.swa_enabled and not args.ema_enabled and scale < 0.5 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name].add_(t.detach().float())
                swa_count += 1

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    if device.type == "cuda":
        log0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
        )
    else:
        log0(f"peak memory allocated: n/a reserved: n/a device:{device.type}")

    if ema_state is not None:
        log0("ema:applying EMA weights")
        avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype)
                     for name, t in ema_state.items()}
        del ema_state
        base_model.load_state_dict(avg_state, strict=True)
    elif args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        avg_state = {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype)
                     for name, t in swa_state.items()}
        del swa_state
        base_model.load_state_dict(avg_state, strict=True)

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    full_state_dict = base_model.state_dict()
    export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}
    excluded_mtp = sum(int(t.numel()) for k, t in full_state_dict.items() if "mtp_heads" in k)
    if excluded_mtp > 0:
        log0(f"export_excluding_mtp_params:{excluded_mtp}")

    if master_process:
        torch.save(export_sd, "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
    legacy_quant_blob = None
    if args.compare_export_codecs or args.export_codec != "custom":
        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        legacy_quant_blob = (
            zstandard.ZstdCompressor(level=22).compress(quant_raw)
            if _COMPRESSOR == "zstd"
            else zlib.compress(quant_raw, 9)
        )
    use_custom_codec = args.export_codec == "custom"
    custom_quant_blob = None
    if args.compare_export_codecs or use_custom_codec:
        quant_arrays = {name: tensor.numpy() for name, tensor in quant_result.items()}
        custom_quant_blob = encode_custom_quantized_state(quant_arrays, quant_meta)
    if not use_custom_codec and legacy_quant_blob is None:
        raise RuntimeError("Legacy export requested but legacy blob was not built")
    if use_custom_codec and custom_quant_blob is None:
        raise RuntimeError("Custom export requested but custom blob was not built")
    quant_blob = custom_quant_blob if use_custom_codec else legacy_quant_blob
    artifact_label = "mixed+custom-packed" if use_custom_codec else f"mixed+{_COMPRESSOR}"
    artifact_name = "final_model.quant.pgec" if use_custom_codec else "final_model.int6.ptz"
    if master_process:
        if args.compare_export_codecs and legacy_quant_blob is not None:
            log0(f"Serialized model mixed+{_COMPRESSOR}: {len(legacy_quant_blob)} bytes")
            log0(f"Total submission size mixed+{_COMPRESSOR}: {len(legacy_quant_blob) + code_bytes} bytes")
        if custom_quant_blob is not None:
            log0(f"Serialized model mixed+custom-packed: {len(custom_quant_blob)} bytes")
            log0(f"Total submission size mixed+custom-packed: {len(custom_quant_blob) + code_bytes} bytes")
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        log0(f"Serialized model {artifact_label}: {quant_file_bytes} bytes")
        log0(f"Total submission size {artifact_label}: {quant_file_bytes + code_bytes} bytes")

    # Roundtrip: decompress + dequantize into fresh model + eval
    if distributed:
        dist.barrier()
    with open(artifact_name, "rb") as f:
        quant_blob_disk = f.read()
    if use_custom_codec:
        decoded_arrays, decoded_meta = decode_custom_quantized_state(quant_blob_disk)
        decoded_result = {name: torch.from_numpy(arr.copy()) for name, arr in decoded_arrays.items()}
        deq_state = dequantize_mixed_int6(decoded_result, decoded_meta, sd_cpu)
    else:
        quant_state = torch.load(
            io.BytesIO(
                zstandard.ZstdDecompressor().decompress(quant_blob_disk)
                if _COMPRESSOR == "zstd"
                else zlib.decompress(quant_blob_disk)
            ),
            map_location="cpu",
        )
        deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
    ).to(device)
    if device.type == "cuda":
        eval_model = eval_model.bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = maybe_compile(eval_model, enabled=compile_enabled)

    # Standard non-overlapping eval (sanity check)
    device_sync(device)
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    device_sync(device)
    log0(
        f"final_quant_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_quant_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval (submission score)
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        device_sync(device)
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        device_sync(device)
        log0(
            f"final_quant_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_quant_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    # Second sliding window eval at stride=64 for submission comparison
    if args.eval_stride > 0 and args.eval_stride != 64 and 64 < sw_seq_len:
        device_sync(device)
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        device_sync(device)
        log0(
            f"final_quant_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_quant_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if args.ttt_enabled:
        device_sync(device)
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb, ttt_doc_count, ttt_threshold = eval_val_adaptive_longdoc_ttt(
            args, eval_model, rank, world_size, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        device_sync(device)
        log0(
            f"final_longdoc_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"doc_percentile:{args.ttt_doc_percentile:.1f} min_pred_tokens:{args.ttt_min_pred_tokens} "
            f"selected_docs:{ttt_doc_count} "
            f"threshold_tokens:{ttt_threshold} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_longdoc_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
