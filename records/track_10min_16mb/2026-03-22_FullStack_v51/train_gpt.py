"""
train_gpt_submit.py — Submission v2: wider MLP + STE int6 QAT + MTP + seq2048 + NTK RoPE +
fp16 embed + late-K passthrough + sliding window eval.
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

import zstandard
_COMPRESSOR = "zstd"

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
    flash_attn_3_func = None  # Falls back to SDPA (works on A100/non-Hopper GPUs)

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
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

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
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 200))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    late_qat = bool(int(os.environ.get("LATE_QAT", "0")))
    # TTT (Test-Time Training) — LoRA TTT (per-document, from PR #548)
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 20))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_min_doc_len = int(os.environ.get("TTT_MIN_DOC_LEN", 1024))
    ttt_surprise_topk = float(os.environ.get("TTT_SURPRISE_TOPK", 0.5))  # Fraction of highest-loss tokens to train on (0.5 = top 50%)
    # In-Place TTT (ICLR 2026 — update MLP output projections as fast weights)
    inplace_ttt_enabled = bool(int(os.environ.get("INPLACE_TTT_ENABLED", "1")))
    inplace_ttt_lr = float(os.environ.get("INPLACE_TTT_LR", 0.001))
    inplace_ttt_chunk = int(os.environ.get("INPLACE_TTT_CHUNK", 256))
    # Legacy AdamW TTT params (kept for fallback)
    ttt_lr = float(os.environ.get("TTT_LR", 0.0005))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    # TrigramHash (our unique addition)
    trigram_vocab_size = int(os.environ.get("TRIGRAM_VOCAB_SIZE", 4096))
    trigram_dim = int(os.environ.get("TRIGRAM_DIM", 128))
    # Value Residual (from PR #413, ResFormer — -0.015 BPB for 18 params)
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "1")))
    # Gradient-Guided Quantization (from PR #332)
    grad_quant = bool(int(os.environ.get("GRAD_QUANT", "1")))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
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
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

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
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

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
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


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
        value_residual: bool = False,
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
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value projection via GQA-aware reshape (no repeat_interleave)."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, v0: Tensor | None = None, q_delta: Tensor | None = None, v_delta: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen, dim = x.shape
        q_out = self.c_q(x)
        if q_delta is not None:
            q_out = q_out + q_delta
        q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v_out = self.c_v(x)
        if v_delta is not None:
            v_out = v_out + v_delta
        v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if flash_attn_3_func is not None:
            y = flash_attn_3_func(q, k, v, causal=True)
        else:
            qt, kt, vt = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
            if kt.shape[1] != qt.shape[1]:
                nr = qt.shape[1]//kt.shape[1]
                kt = kt[:,:,None,:,:].expand(-1,-1,nr,-1,-1).reshape(kt.shape[0],-1,kt.shape[-2],kt.shape[-1])
                vt = vt[:,:,None,:,:].expand(-1,-1,nr,-1,-1).reshape(vt.shape[0],-1,vt.shape[-2],vt.shape[-1])
            y = torch.nn.functional.scaled_dot_product_attention(qt,kt,vt,is_causal=True).transpose(1,2)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y), raw_v


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


# ── LoRA TTT Modules (from PR #548) ──

class BatchedLinearLoRA(nn.Module):
    """Per-batch-element LoRA adapter. Delta = x @ A^T @ B^T."""
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


class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch: LM head + Q/V per block."""
    def __init__(self, bsz: int, model, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            q_out = block.attn.c_q.weight.shape[0]
            v_out = block.attn.c_v.weight.shape[0]
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, q_out, rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, v_out, rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


BOS_ID = 1  # SentencePiece BOS token ID


def _find_docs(all_tokens: Tensor) -> list[tuple[int, int]]:
    """Return (start_offset, length) for each document at BOS boundaries."""
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    docs: list[tuple[int, int]] = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) + 1 if i + 1 < len(bos_positions) else all_tokens.numel()
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def _reset_ttt_optimizer(opt: torch.optim.Adam) -> None:
    for group in opt.param_groups:
        for p in group["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            s["exp_avg"].zero_()
            s["exp_avg_sq"].zero_()
            s["step"].fill_(0)


def _compute_chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int):
    """Return (win_start, win_len, chunk_offset, chunk_len) for chunk ci of a doc."""
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len


def _build_ttt_optimizer(lora, args):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)


def eval_val_ttt_lora(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    log_fn=None,
) -> tuple[float, float]:
    """Batched LoRA TTT evaluation: per-document adaptation, sharded across GPUs."""
    docs = _find_docs(val_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
    short_docs = [d for d in rank_docs if d[1] < args.ttt_min_doc_len]
    long_docs = [d for d in rank_docs if d[1] >= args.ttt_min_doc_len]
    master = rank == 0
    if master and log_fn:
        log_fn(f"lora_ttt: {len(docs)} total docs, rank0: {len(long_docs)} long + {len(short_docs)} short, batch={args.ttt_batch_seqs}")

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    # Score short docs without TTT
    t0 = time.perf_counter()
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds: ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
            y = val_tokens[ds + 1: ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mean_loss = base_model(x, y)
            loss_sum += mean_loss.to(torch.float64) * n
            token_count += n
            tgt = y.reshape(-1)
            px = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            byte_sum += tb.sum()
    if master and log_fn:
        log_fn(f"lora_ttt: short_docs time={time.perf_counter()-t0:.1f}s tokens={int(token_count.item())}")

    # Sort long docs by chunk count for efficient batching
    long_docs.sort(key=lambda d: (d[1] - 2) // args.ttt_chunk_size)
    batch_size = args.ttt_batch_seqs
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len

    # Create reusable LoRA and optimizer
    lora = BatchedTTTLoRA(batch_size, base_model, args.ttt_lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)

    t1 = time.perf_counter()
    for bi in range(0, len(long_docs), batch_size):
        batch = long_docs[bi: bi + batch_size]
        bsz = len(batch)
        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, args.ttt_lora_rank).to(device)
            cur_opt = _build_ttt_optimizer(cur_lora, args)
        pred_lens = [dl - 1 for _, dl in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        for epoch in range(args.ttt_epochs):
            for ci in range(max_nc):
                active = [ci < nc for nc in num_chunks]
                ws_ref, wl_ref, _, _ = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
                x = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                y = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                doc_info = []
                for b in range(bsz):
                    if not active[b]:
                        doc_info.append((0, 0))
                        continue
                    ds, dl = batch[b]
                    ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                    toks = val_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                    x[b, :wl] = toks[:-1]
                    y[b, :wl] = toks[1:]
                    doc_info.append((co, cl))
                needs_train = any(ci < nc - 1 for nc in num_chunks)
                if needs_train:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = base_model(x, y, lora=cur_lora)
                else:
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ptl = base_model(x, y, lora=cur_lora)
                if epoch == args.ttt_epochs - 1:
                    with torch.no_grad():
                        for b in range(bsz):
                            if not active[b]:
                                continue
                            co, cl = doc_info[b]
                            loss_sum += ptl[b, co: co + cl].to(torch.float64).sum()
                            token_count += cl
                            tgt = y[b, co: co + cl]
                            px = x[b, co: co + cl]
                            tb = base_bytes_lut[tgt].to(torch.float64)
                            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
                            byte_sum += tb.sum()
                if needs_train:
                    train_loss = torch.zeros(bsz, device=device)
                    topk_frac = args.ttt_surprise_topk
                    for b in range(bsz):
                        if ci >= num_chunks[b] - 1:
                            continue
                        co, cl = doc_info[b]
                        if cl > 0:
                            chunk_losses = ptl[b, co: co + cl]
                            if topk_frac < 1.0 and cl > 1:
                                # Surprise gating: only train on highest-loss tokens
                                k = max(1, int(cl * topk_frac))
                                topk_vals, _ = torch.topk(chunk_losses, k)
                                train_loss[b] = topk_vals.mean()
                            else:
                                train_loss[b] = chunk_losses.mean()
                    cur_opt.zero_grad()
                    train_loss.sum().backward()
                    cur_opt.step()
        if master and log_fn and (bi + batch_size) % (batch_size * 5) == 0:
            elapsed = time.perf_counter() - t1
            avg_l = loss_sum.item() / max(token_count.item(), 1)
            log_fn(f"lora_ttt: batch {bi // batch_size + 1}/{(len(long_docs) + batch_size - 1) // batch_size} time={elapsed:.1f}s avg_loss={avg_l:.4f}")

    if master and log_fn:
        log_fn(f"lora_ttt: long_docs time={time.perf_counter()-t1:.1f}s docs={len(long_docs)}")

    # All-reduce across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    base_model.train()
    for p in base_model.parameters():
        p.requires_grad_(True)

    val_loss = float(loss_sum.item() / max(token_count.item(), 1))
    val_bpb = float((loss_sum.item() / math.log(2.0)) / max(byte_sum.item(), 1))
    if master and log_fn:
        log_fn(f"lora_ttt:complete loss:{val_loss:.4f} bpb:{val_bpb:.4f} time:{time.perf_counter()-t0:.1f}s")
    return val_loss, val_bpb


# ── In-Place TTT (ICLR 2026 Oral — update MLP output projection as fast weights) ──

def inplace_ttt_eval(
    args, model, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    chunk_size: int = 256, lr: float = 0.001,
    rank: int = 0, world_size: int = 1, log_fn=None,
) -> tuple[float, float]:
    """In-Place TTT eval: per-document MLP projection adaptation with apply-then-update.

    For each document:
    1. Reset MLP proj weights to original
    2. For each chunk: SCORE (accumulate bpb) → then UPDATE MLP proj weights
    3. Apply-then-update preserves strict causality (score before adapt)

    Returns (val_loss, val_bpb) like eval_val_ttt_lora.
    """
    docs = _find_docs(val_tokens)
    rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
    short_docs = [d for d in rank_docs if d[1] < args.ttt_min_doc_len]
    long_docs = [d for d in rank_docs if d[1] >= args.ttt_min_doc_len]
    master = rank == 0

    # Collect MLP proj weights (the fast weights)
    mlp_projs = [block.mlp.proj.weight for block in model.blocks]
    original_weights = [w.data.clone() for w in mlp_projs]

    if master and log_fn:
        log_fn(f"inplace_ttt:start docs={len(rank_docs)} ({len(long_docs)} long, {len(short_docs)} short) layers={len(mlp_projs)} lr={lr}")

    t0 = time.perf_counter()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    # Score short docs without TTT (same as LoRA TTT)
    with torch.no_grad():
        for ds, dl in short_docs:
            x = val_tokens[ds: ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
            y = val_tokens[ds + 1: ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
            n = dl - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                mean_loss = model(x, y)
            loss_sum += mean_loss.to(torch.float64) * n
            token_count += n
            tgt = y.reshape(-1)
            px = x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            byte_sum += tb.sum()

    # Long docs: apply-then-update per chunk
    for doc_idx, (ds, dl) in enumerate(long_docs):
        # Reset fast weights to original for each document
        for w, orig in zip(mlp_projs, original_weights):
            w.data.copy_(orig)

        pred_len = dl - 1
        num_chunks = (pred_len + chunk_size - 1) // chunk_size

        for ci in range(num_chunks):
            chunk_start = ci * chunk_size
            chunk_end = min((ci + 1) * chunk_size, pred_len)
            cl = chunk_end - chunk_start
            if cl < 2:
                continue

            win_start = max(0, chunk_end - 1024)
            win_len = chunk_end - win_start
            chunk_offset = chunk_start - win_start

            toks = val_tokens[ds + win_start: ds + win_start + win_len + 1].to(device=device, dtype=torch.int64)
            x = toks[:-1].unsqueeze(0)
            y = toks[1:].unsqueeze(0)

            # APPLY: Score this chunk (no grad)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                per_token_loss_all = F.cross_entropy(
                    model.forward_logits(x).reshape(-1, args.vocab_size).float(),
                    y.reshape(-1),
                    reduction="none",
                )
            # Accumulate only the chunk's tokens (not the context window)
            chunk_losses = per_token_loss_all[chunk_offset: chunk_offset + cl]
            loss_sum += chunk_losses.to(torch.float64).sum()
            token_count += cl
            tgt = y[0, chunk_offset: chunk_offset + cl]
            px = x[0, chunk_offset: chunk_offset + cl]
            tb = base_bytes_lut[tgt].to(torch.float64)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
            byte_sum += tb.sum()

            # UPDATE: gradient step on MLP proj (skip last chunk — nothing to update for)
            if ci < num_chunks - 1:
                for w in mlp_projs:
                    w.requires_grad_(True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    update_loss = model(x, y)  # mean NTP loss
                update_loss.backward()
                with torch.no_grad():
                    for w in mlp_projs:
                        if w.grad is not None:
                            w.data -= lr * w.grad
                            w.grad = None
                        w.requires_grad_(False)

        if master and log_fn and (doc_idx + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0
            avg_l = loss_sum.item() / max(token_count.item(), 1)
            log_fn(f"inplace_ttt: doc {doc_idx+1}/{len(long_docs)} time={elapsed:.1f}s avg_loss={avg_l:.4f}")

    # Reset to original weights (don't leave model in mutated state)
    for w, orig in zip(mlp_projs, original_weights):
        w.data.copy_(orig)

    # All-reduce across ranks
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

    val_loss = float(loss_sum.item() / max(token_count.item(), 1))
    val_bpb = float((loss_sum.item() / math.log(2.0)) / max(byte_sum.item(), 1))
    if master and log_fn:
        log_fn(f"inplace_ttt:complete loss:{val_loss:.4f} bpb:{val_bpb:.4f} time:{time.perf_counter()-t0:.1f}s")
    return val_loss, val_bpb


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


class TrigramHashEmbedding(nn.Module):
    def __init__(self, trigram_vocab_size: int, trigram_dim: int, model_dim: int):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(trigram_dim, model_dim, bias=False) if trigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.03, dtype=torch.float32))

    def trigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.trigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1] = mod
        out[..., 2:] = (
            torch.bitwise_xor(
                torch.bitwise_xor(36313 * t[..., 2:], 27191 * t[..., 1:-1]),
                51497 * t[..., :-2],
            )
            % mod
        )
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.trigram_hash(token_ids))
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
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
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
        value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims, value_residual=value_residual)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v0: Tensor | None = None, q_delta_fn=None, v_delta_fn=None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale_factor
        n = self.attn_norm(x) * s
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        attn_out, raw_v = self.attn(n, v0=v0, q_delta=qd, v_delta=vd)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        return x, raw_v


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
        trigram_vocab_size: int = 0,
        trigram_dim: int = 128,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale: bool = False,
        value_residual: bool = False,
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
        self.trigram = TrigramHashEmbedding(trigram_vocab_size, trigram_dim, model_dim) if trigram_vocab_size > 0 else None
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
                    value_residual=value_residual,
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
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            qd_fn = lora.q_loras[i] if lora is not None else None
            vd_fn = lora.v_loras[i] if lora is not None else None
            x, raw_v = self.blocks[i](x, x0, v0=v0, q_delta_fn=qd_fn, v_delta_fn=vd_fn)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            qd_fn = lora.q_loras[bi] if lora is not None else None
            vd_fn = lora.v_loras[bi] if lora is not None else None
            x, _ = self.blocks[bi](x, x0, v0=v0, q_delta_fn=qd_fn, v_delta_fn=vd_fn)

        x = self.final_norm(x)

        # LoRA path: per-token loss with LM head LoRA delta
        if lora is not None:
            if self.tie_embeddings:
                logits_proj = F.linear(x, self.tok_emb.weight)
            else:
                logits_proj = self.lm_head(x)
            logits_proj = logits_proj + lora.lm_head_lora(x)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(input_ids.shape)

        # Standard path: mean loss
        x_flat = x.reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
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
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        v0 = None
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x, raw_v = self.blocks[i](x, x0, v0=v0)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, _ = self.blocks[self.num_encoder_layers + i](x, x0, v0=v0)
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
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()  # PyTorch inference mode
    try:
        compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
        # Quick test to catch compile failures early
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _test = compiled_logits(torch.zeros(1, seq_len, dtype=torch.int64, device=device))
    except Exception:
        compiled_logits = base_model.forward_logits  # fallback: no compile

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

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
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

def gptq_calibrate(model: nn.Module, train_pattern: str, device: torch.device,
                   n_samples: int = 256, seq_len: int = 2048) -> dict[str, Tensor]:
    """Collect Hessian H = X^T X for each linear layer using training data."""
    hessians: dict[str, Tensor] = {}
    n_seen: dict[str, int] = {}
    hooks = []
    def make_hook(name: str):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=torch.float32)
                n_seen[name] = 0
            hessians[name].addmm_(x.t(), x)
            n_seen[name] += x.shape[0]
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, CastedLinear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    stream = TokenStream(train_pattern)
    model.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            tokens = stream.take(seq_len + 1).to(device=device, dtype=torch.int64)
            x = tokens[:-1].unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.forward_logits(x)
    for h in hooks:
        h.remove()
    for name in hessians:
        hessians[name] /= max(n_seen[name], 1)
    model.train()
    return hessians


def _find_best_row_scales(W: Tensor, clip_range: int = 31) -> Tensor:
    t32 = W.float()
    best_s = t32.abs().amax(dim=1) / clip_range
    best_s = best_s.clamp_min(1.0 / clip_range)
    best_err = torch.full((t32.shape[0],), float('inf'))
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        if pct < 1.0:
            row_clip = torch.quantile(t32.abs(), pct, dim=1)
        else:
            row_clip = t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
        recon = q * s[:, None]
        err = (t32 - recon).pow(2).mean(dim=1)
        improved = err < best_err
        best_s[improved] = s[improved]
        best_err[improved] = err[improved]
    return best_s


def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 31,
                          block_size: int = 128, percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    """GPTQ quantize weight W using Hessian H for error compensation."""
    W = W.float().clone()
    rows, cols = W.shape
    row_scale = _find_best_row_scales(W, clip_range)
    H = H.float().clone()
    damp = percdamp * H.diag().mean()
    H.diagonal().add_(damp)
    perm = torch.argsort(H.diag())
    invperm = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except torch._C._LinAlgError:
        Hinv = torch.diag(1.0 / H.diag().clamp_min(1e-6))
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros_like(W_block)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            h_inv_jj = Hinv_block[j, j].clamp_min(1e-8)
            q_col = torch.clamp(torch.round(w_col / row_scale), -clip_range, clip_range)
            deq_col = q_col * row_scale
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - deq_col) / h_inv_jj
            Err[:, j] = err
            if j + 1 < i2 - i1:
                W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    Q = Q[:, invperm]
    return Q, row_scale.to(torch.float16)


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

def _quantize_adaptive_bits(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Quantize tensor to specified bit-width (5/6/7)."""
    max_val = (1 << (bits - 1)) - 1  # 5→15, 6→31, 7→63
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / float(max_val)).clamp_min(1.0 / float(max_val)).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -max_val - 1, max_val).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / float(max_val) if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -max_val - 1, max_val).to(torch.int8)
    return q, scale


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str],
                        grad_sensitivity: dict[str, float] | None = None,
                        hessians: dict[str, Tensor] | None = None):
    num_layers_total = max(
        (int(k.split(".")[1]) for k in state_dict if k.startswith("blocks.")),
        default=0,
    ) + 1
    late_k_layers = set(range(num_layers_total - 2, num_layers_total))

    # Pre-compute sensitivity thresholds for gradient-guided quantization
    p20, p90 = 0.0, float('inf')
    if grad_sensitivity:
        all_sens = sorted(grad_sensitivity.values())
        if len(all_sens) > 4:
            p20 = all_sens[len(all_sens) // 5]
            p90 = all_sens[9 * len(all_sens) // 10]

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
            # Gradient-guided: adaptive bits per tensor
            bits = 6
            if grad_sensitivity and name in grad_sensitivity:
                sens = grad_sensitivity[name]
                if sens >= p90:
                    bits = 7  # Top 10%: high sensitivity → more precision
                elif sens <= p20:
                    bits = 5  # Bottom 20%: low sensitivity → more compression
            # Try GPTQ if Hessian available (31% quant gap reduction)
            module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
            H = hessians.get(module_name) if hessians else None
            if H is not None and t.ndim == 2 and bits == 6:
                q, s = gptq_quantize_weight(t, H.cpu())
            elif bits == 6:
                q, s = quantize_int6_per_row(t)
            else:
                q, s = _quantize_adaptive_bits(t, bits)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{bits}"}
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
# TTT (TEST-TIME TRAINING)
# -----------------------------

def ttt_adapt(args, base_model, device, val_tokens, rank=0, world_size=1, log_fn=None):
    """Full-weight TTT with cosine LR decay and per-layer LR (from PR #481)."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs
    ttt_cosine = bool(int(os.environ.get("TTT_COSINE", "1")))
    ttt_perlayer = bool(int(os.environ.get("TTT_PERLAYER", "1")))

    frozen_params = set()
    if args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)
                    frozen_params.add(id(p))

    # Per-layer LR: MLP output projections get 3× LR (most quant damage),
    # MLP input projections get 0.5× LR (least damage)
    ttt_use_adamw = bool(int(os.environ.get("TTT_ADAMW", "1")))
    if ttt_perlayer:
        proj_params = [p for n, p in base_model.named_parameters()
                      if "mlp.proj" in n and p.requires_grad and id(p) not in frozen_params]
        fc_params = [p for n, p in base_model.named_parameters()
                    if "mlp.fc" in n and p.requires_grad and id(p) not in frozen_params]
        other_params = [p for p in base_model.parameters()
                       if p.requires_grad and id(p) not in frozen_params
                       and id(p) not in {id(q) for q in proj_params + fc_params}]
        param_groups = [g for g in [
            {"params": proj_params, "lr": args.ttt_lr * 3.0},
            {"params": fc_params, "lr": args.ttt_lr * 0.5},
            {"params": other_params, "lr": args.ttt_lr},
        ] if g["params"]]
        ttt_params = proj_params + fc_params + other_params
    else:
        ttt_params = [p for p in base_model.parameters() if p.requires_grad and id(p) not in frozen_params]
        param_groups = [{"params": ttt_params, "lr": args.ttt_lr}]

    if ttt_use_adamw:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    else:
        optimizer = torch.optim.SGD(param_groups, momentum=args.ttt_momentum)

    # Store initial LR for cosine schedule
    if ttt_cosine:
        for g in optimizer.param_groups:
            g["initial_lr"] = g["lr"]

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size
    steps_per_epoch = (my_end - my_start) // max(batch_seqs, 1)
    total_steps = args.ttt_epochs * steps_per_epoch
    global_step = 0

    base_model.train()
    t0 = time.perf_counter()

    if log_fn:
        n_ttt = sum(p.numel() for p in ttt_params)
        log_fn(f"ttt:config params:{n_ttt} cosine:{ttt_cosine} perlayer:{ttt_perlayer}")

    for epoch in range(args.ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)

        for batch_start in range(my_start, my_end, batch_seqs):
            # Cosine LR decay
            if ttt_cosine and total_steps > 0:
                progress = global_step / total_steps
                mul = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                for g in optimizer.param_groups:
                    g["lr"] = g["initial_lr"] * mul

            batch_end = min(batch_start + batch_seqs, my_end)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()

            if world_size > 1:
                for p in ttt_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
            optimizer.step()

            epoch_loss_sum += loss.detach().to(torch.float64) * y.numel()
            epoch_tokens += float(y.numel())
            global_step += 1

        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)

        elapsed = time.perf_counter() - t0
        if log_fn:
            log_fn(f"ttt_epoch:{epoch+1}/{args.ttt_epochs} loss:{epoch_loss_sum.item()/max(epoch_tokens.item(),1):.4f} time:{elapsed:.1f}s")

    for p in base_model.parameters():
        p.requires_grad_(True)

    if log_fn:
        log_fn(f"ttt:done elapsed={time.perf_counter()-t0:.1f}s")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5) if os.environ.get("TORCH_COMPILE","1")!="0" else zeropower_via_newtonschulz5

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

    # Fast math knobs
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
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

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
        trigram_vocab_size=args.trigram_vocab_size,
        trigram_dim=args.trigram_dim,
        xsa_last_n=0,  # Train WITHOUT XSA — enable only at eval time
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if os.environ.get("TORCH_COMPILE","1")!="0" else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=args.value_residual) if distributed else compiled_model

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
    if base_model.trigram is not None:
        scalar_params.append(base_model.trigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)
    if base_model.trigram is not None:
        tok_params.append({"params": [base_model.trigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.trigram.proj is not None:
            matrix_params.append(base_model.trigram.proj.weight)
    optimizer_tok = torch.optim.AdamW(
        tok_params,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
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
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
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
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
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
            torch.cuda.synchronize()
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

        # Gradient sensitivity accumulation for Gradient-Guided Quantization
        if args.grad_quant and scale < 0.1 and scale > 0:
            if not hasattr(main, '_grad_sens'):
                main._grad_sens = {}
                main._grad_count = 0
            main._grad_count += 1
            for name, p in base_model.named_parameters():
                if p.grad is not None and p.ndim == 2 and p.shape[0] >= 64:
                    gs = p.grad.detach().float().pow(2).mean().item()
                    main._grad_sens[name] = main._grad_sens.get(name, 0.0) + gs

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

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

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
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    sd_cpu = {k: v.detach().cpu() for k, v in export_sd.items()}
    grad_sens = None
    if args.grad_quant and hasattr(main, '_grad_sens') and main._grad_sens:
        count = getattr(main, '_grad_count', 1)
        grad_sens = {name: val / count for name, val in main._grad_sens.items()}
        log0(f"grad_quant: adaptive precision for {len(grad_sens)} tensors based on gradient sensitivity")

    # Full GPTQ: Hessian-aware quantization (31% quant gap reduction)
    gptq_hessians = None
    if torch.cuda.is_available():
        t_gptq = time.perf_counter()
        log0("gptq:calibrating...")
        gptq_hessians = gptq_calibrate(base_model, args.train_files, device, n_samples=256, seq_len=args.train_seq_len)
        log0(f"gptq:calibrated {len(gptq_hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")

    quant_result, quant_meta = mixed_quantize_int6(sd_cpu, {"mlp", "attn"}, grad_sensitivity=grad_sens, hessians=gptq_hessians)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw) if _COMPRESSOR == "zstd" else zlib.compress(quant_raw, 9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+{_COMPRESSOR}: {quant_file_bytes + code_bytes} bytes")

    # Roundtrip: decompress + dequantize into fresh model + eval
    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob_disk) if _COMPRESSOR == "zstd" else zlib.decompress(quant_blob_disk)),
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
        trigram_vocab_size=args.trigram_vocab_size, trigram_dim=args.trigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)

    # LoRA TTT: per-document adaptation during eval (from PR #548)
    # CRITICAL: Must use a fresh UNCOMPILED model — torch.compile silently ignores LoRA deltas
    if args.ttt_enabled:
        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        torch._dynamo.reset()
        ttt_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads, model_dim=args.model_dim,
            num_heads=args.num_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            trigram_vocab_size=args.trigram_vocab_size, trigram_dim=args.trigram_dim,
            xsa_last_n=args.xsa_last_n,  # XSA enabled for eval
            rope_dims=args.rope_dims, ln_scale=args.ln_scale,
            value_residual=args.value_residual,
        ).to(device).bfloat16()
        for m in ttt_model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        restore_low_dim_params_to_fp32(ttt_model)
        ttt_model.load_state_dict(deq_state, strict=True)

        # TTT eval: choose between LoRA TTT (batched) or In-Place TTT (per-doc MLP adaptation)
        t_ttt = time.perf_counter()
        if args.inplace_ttt_enabled:
            # In-Place TTT: apply-then-update on MLP proj weights per document
            log0(f"inplace_ttt:start lr={args.inplace_ttt_lr} chunk={args.inplace_ttt_chunk}")
            ttt_val_loss, ttt_val_bpb = inplace_ttt_eval(
                args, ttt_model, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                chunk_size=args.inplace_ttt_chunk, lr=args.inplace_ttt_lr,
                rank=rank, world_size=world_size, log_fn=log0,
            )
        else:
            # LoRA TTT: batched per-document adaptation on Q/V/LM head
            log0(f"lora_ttt:start rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} epochs={args.ttt_epochs} batch={args.ttt_batch_seqs}")
            ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
                args, ttt_model, rank, world_size, device, val_tokens,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                log_fn=log0,
            )
        log0(f"ttt:complete val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} time:{time.perf_counter() - t_ttt:.1f}s")
        del ttt_model
        if distributed:
            dist.barrier()

    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True) if os.environ.get("TORCH_COMPILE","1")!="0" else eval_model

    # Standard non-overlapping eval (sanity check)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args, compiled_eval, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    # Sliding window eval (submission score)
    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    # Second sliding window eval at stride=64 for submission comparison
    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
