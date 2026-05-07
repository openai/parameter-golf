"""
train_gpt.py — Hymba: Hybrid Attention + Mamba SSM language model
==================================================================
Competition submission for OpenAI Parameter Golf challenge.
https://github.com/openai/parameter-golf

Architecture: Hymba (Hybrid Attention + Mamba SSM)
  - Every block: parallel Attention + SSM with learned merge
  - SmearGate + BigramHash embedding
  - U-Net skip connections (encoder/decoder halves)
  - Leaky ReLU² MLP with learnable attn/mlp scales
  - Logit softcap, residual mixing (x + x0 blend)
  - EMA weight averaging
  - Tied embeddings + ortho init

Training budget: TIME_BUDGET_SECONDS (default 600 s = 10 min wall clock)
Primary metric : val_bpb  (bits per byte, lower is better)

Requirements: NVIDIA GPU, mamba-ssm + causal-conv1d
  pip install mamba-ssm[causal-conv1d]

Single GPU:
  python train_gpt.py

Multi-GPU (DDP, e.g. 8x H100):
  torchrun --standalone --nproc_per_node=8 train_gpt.py
"""

from __future__ import annotations
import glob, io, math, os, time, zlib
from pathlib import Path
from typing import Tuple
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

DATA_PATH      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
TRAIN_FILES    = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_FILES      = os.path.join(DATA_PATH, "fineweb_val_*.bin")
VOCAB_SIZE     = int(os.environ.get("VOCAB_SIZE", 1024))
SHARD_HEADER_SIZE  = 256
SHARD_MAGIC_NUMBER = 20240520
SHARD_VERSION      = 1

def load_data_shard(file: Path) -> Tensor:
    header_bytes = SHARD_HEADER_SIZE * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=SHARD_HEADER_SIZE)
    if (
        header.size != SHARD_HEADER_SIZE
        or int(header[0]) != SHARD_MAGIC_NUMBER
        or int(header[1]) != SHARD_VERSION
    ):
        raise ValueError(f"Unexpected shard header: {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read: {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str) -> None:
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No shard files found: {pattern}")
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
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> Tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No validation files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np           = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np    = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np    = np.ones( (table_size,), dtype=np.bool_)
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
        torch.tensor(base_bytes_np,        dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np,  dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np,  dtype=torch.bool,  device=device),
    )


def eval_val(
    model: nn.Module, rank: int, world_size: int, device: torch.device,
    grad_accum_steps: int, val_tokens: Tensor, seq_len: int, val_batch_size: int,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
) -> Tuple[float, float]:
    local_batch_tokens = val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            f"val_batch_size too small: {val_batch_size}, "
            f"need >= {world_size * grad_accum_steps * seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs       = (val_tokens.numel() - 1) // seq_len
    seq_start        = (total_seqs * rank)       // world_size
    seq_end          = (total_seqs * (rank + 1)) // world_size

    val_loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end   = batch_seq_end   * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum    += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            token_bytes  = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)

    val_loss        = val_loss_sum / val_token_count
    bits_per_token  = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


BUDGET_BYTES = 16_000_000
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0
_DEFAULT_FP32_NAME_PATTERNS = ()


def _tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _keep_float_tensor(
    name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str],
    fp32_patterns: tuple[str, ...],
) -> Tensor:
    if any(p in name for p in fp32_patterns):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def _quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    fp32_name_patterns: tuple[str, ...] = _DEFAULT_FP32_NAME_PATTERNS,
):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += _tensor_nbytes(t)
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float_tensor(name, t, passthrough_orig_dtypes, fp32_name_patterns)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += _tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = _quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
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
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def measure_and_save_artifact(
    model: nn.Module,
    code_paths: list[str],
    save_dir: str,
    fp32_name_patterns: tuple[str, ...] = _DEFAULT_FP32_NAME_PATTERNS,
    log_fn=print,
) -> dict:
    total_code_bytes = 0
    for p in code_paths:
        sz = len(Path(p).read_bytes())
        log_fn(f"  code: {p} = {sz:,} bytes")
        total_code_bytes += sz

    # Int8 quantize + zlib compress
    quant_obj, quant_stats = quantize_state_dict_int8(
        model.state_dict(), fp32_name_patterns=fp32_name_patterns,
    )
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    model_bytes = len(quant_blob)

    total = total_code_bytes + model_bytes
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)

    log_fn(f"  model raw bf16:   {quant_stats['baseline_tensor_bytes']:,} bytes")
    log_fn(f"  model int8 payload: {quant_stats['int8_payload_bytes']:,} bytes")
    log_fn(f"  model int8+zlib:  {model_bytes:,} bytes (ratio: {ratio:.2f}x)")
    log_fn(f"  code total:       {total_code_bytes:,} bytes")
    log_fn(f"  ARTIFACT TOTAL:   {total:,} / {BUDGET_BYTES:,} bytes "
           f"({'OK' if total <= BUDGET_BYTES else 'OVER BUDGET!'})")
    log_fn(f"  headroom:         {BUDGET_BYTES - total:,} bytes")

    # Save compressed model
    os.makedirs(save_dir, exist_ok=True)
    ptz_path = os.path.join(save_dir, "final_model.int8.ptz")
    with open(ptz_path, "wb") as f:
        f.write(quant_blob)
    log_fn(f"  compressed model saved: {ptz_path}")

    return {
        "code_bytes": total_code_bytes,
        "model_bytes": model_bytes,
        "total": total,
        "within_budget": total <= BUDGET_BYTES,
        "quant_blob": quant_blob,
        "quant_stats": quant_stats,
        "ptz_path": ptz_path,
    }


def verify_artifact_roundtrip(
    model: nn.Module,
    ptz_path: str,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    seq_len: int,
    val_batch_size: int,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log_fn=print,
) -> Tuple[float, float]:
    log_fn("── Roundtrip verification (int8+zlib → dequantize → eval) ──")
    with open(ptz_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zlib.decompress(quant_blob_disk)),
        map_location="cpu", weights_only=False,
    )
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        model, rank, world_size, device, grad_accum_steps,
        val_tokens, seq_len, val_batch_size,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    log_fn(
        f"  int8_roundtrip val_loss:{q_val_loss:.4f}  val_bpb:{q_val_bpb:.4f}  "
        f"eval_time:{time.perf_counter() - t_qeval:.1f}s"
    )
    return q_val_loss, q_val_bpb




import copy
import random
import subprocess
import sys
import uuid
from typing import Optional

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# SSM dependencies
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from causal_conv1d import causal_conv1d_fn
except Exception as _err:
    raise ImportError(
        "mamba-ssm + causal-conv1d are required. "
        "Install with: pip install mamba-ssm[causal-conv1d]"
    ) from _err

# =============================================================================
# HYPERPARAMETERS  ← agent edits this section freely
# =============================================================================

TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", 600.0))

# Control tensor patterns — small scalars/gains that should stay FP32 and use Adam
CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "smeargate", "merge_alpha",
)


class Hyperparameters:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_path       = DATA_PATH
    train_files     = TRAIN_FILES
    val_files       = VAL_FILES
    tokenizer_path  = TOKENIZER_PATH
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    # ── Validation ────────────────────────────────────────────────────────────
    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    # ── Training length ───────────────────────────────────────────────────────
    max_iterations  = int(os.environ.get("MAX_ITERATIONS", 20_000))
    warmdown_iters  = int(os.environ.get("WARMDOWN_ITERS", 700))
    warmdown_shape  = os.environ.get("WARMDOWN_SHAPE", "cosine")
    warmup_steps    = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 65_536))
    train_seq_len   = int(os.environ.get("TRAIN_SEQ_LEN", 1024))

    # ── Model shape ───────────────────────────────────────────────────────────
    vocab_size      = VOCAB_SIZE
    num_layers      = int(os.environ.get("NUM_LAYERS", 7))
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim       = int(os.environ.get("MODEL_DIM", 512))
    mlp_mult        = int(os.environ.get("MLP_MULT", 4))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.9))
    qk_gain_init    = float(os.environ.get("QK_GAIN_INIT", 1.5))
    partial_rope_dims = int(os.environ.get("PARTIAL_ROPE_DIMS", 0))

    # ── Hymba SSM ─────────────────────────────────────────────────────────────
    hymba_expand    = int(os.environ.get("HYMBA_EXPAND", 1))
    hymba_conv_kernel = int(os.environ.get("HYMBA_CONV_KERNEL", 3))
    hymba_dt_rank   = int(os.environ.get("HYMBA_DT_RANK", 0))
    hymba_ssm_state = int(os.environ.get("HYMBA_SSM_STATE", 4))

    # ── MLA (Multi-Head Latent Attention) ────────────────────────────────────
    mla_kv_lora_rank = int(os.environ.get("MLA_KV_LORA_RANK", 0))   # 0 = disabled (use GQA)
    mla_q_lora_rank  = int(os.environ.get("MLA_Q_LORA_RANK", 0))    # 0 = no Q compression
    mla_qk_rope_dim  = int(os.environ.get("MLA_QK_ROPE_DIM", 0))    # 0 = auto (head_dim // 2)

    # ── Embedding features ────────────────────────────────────────────────────
    use_smeargate   = bool(int(os.environ.get("USE_SMEARGATE", "0")))
    use_bigram_hash = bool(int(os.environ.get("USE_BIGRAM_HASH", "1")))
    bigram_buckets  = int(os.environ.get("BIGRAM_BUCKETS", 4096))
    bigram_hash_dim = int(os.environ.get("BIGRAM_HASH_DIM", 128))

    # ── U-Net skip / init ────────────────────────────────────────────────────
    use_unet_skip   = bool(int(os.environ.get("USE_UNET_SKIP", "1")))
    use_ortho_init  = bool(int(os.environ.get("USE_ORTHO_INIT", "1")))
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    kv_share_every  = int(os.environ.get("KV_SHARE_EVERY", 1))

    # ── EMA ──────────────────────────────────────────────────────────────────
    ema_enabled     = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay       = float(os.environ.get("EMA_DECAY", 0.96))
    ema_every       = int(os.environ.get("EMA_EVERY", 1))

    # ── Optimizer ─────────────────────────────────────────────────────────────
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR", 0.05))
    embed_lr        = float(os.environ.get("EMBED_LR", 0.6))
    head_lr         = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr       = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr       = float(os.environ.get("SCALAR_LR", 0.02))
    weight_decay    = float(os.environ.get("WEIGHT_DECAY", 0.02))
    grad_clip_norm  = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1           = float(os.environ.get("BETA1", 0.9))
    beta2           = float(os.environ.get("BETA2", 0.95))
    adam_eps         = float(os.environ.get("ADAM_EPS", 1e-8))

    # ── MTP (Multi-Token Prediction) ─────────────────────────────────────────
    mtp_heads       = int(os.environ.get("MTP_HEADS", 0))

    # ── Sequence Length Curriculum (SLC) ─────────────────────────────────────
    slc_enabled       = bool(int(os.environ.get("SLC_ENABLED", "1")))
    slc_min_seq_len   = int(os.environ.get("SLC_MIN_SEQ_LEN", 128))
    slc_warmup_iters  = int(os.environ.get("SLC_WARMUP_ITERS", 2000))
    slc_schedule      = os.environ.get("SLC_SCHEDULE", "linear")  # linear | quadratic | root

    # ── torch.compile ─────────────────────────────────────────────────────────
    torch_compile   = bool(int(os.environ.get("TORCH_COMPILE", "1")))


# =============================================================================
# MUON OPTIMIZER
# =============================================================================

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
            weight_decay = group.get("weight_decay", 0.0)

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
                if weight_decay > 0:
                    p.data.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# =============================================================================
# MODEL ARCHITECTURE — Hymba (Hybrid Attention + Mamba SSM)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Linear layer that keeps weights in FP32 but casts to input dtype for compute."""
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class SmearGate(nn.Module):
    """Learnable gate that blends current and previous token embeddings."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev


class BigramHash(nn.Module):
    """Hash-based bigram embedding with learnable projection."""
    def __init__(self, num_buckets: int, hash_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.table = nn.Embedding(num_buckets, hash_dim)
        self.proj = CastedLinear(hash_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        prev_ids = torch.cat([
            torch.zeros(bsz, 1, dtype=input_ids.dtype, device=input_ids.device),
            input_ids[:, :-1],
        ], dim=1)
        h = ((prev_ids.long() * 92821 + input_ids.long()) % self.num_buckets).long()
        return self.proj(self.table(h))


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """Ensure 1-D params and control tensors stay in FP32 for training stability."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """Pre-computed RoPE — full table up to max_seq_len, sliced in forward.
    No forward-time mutation means torch.compile never sees an object-identity
    guard on _cos/_sin_cached, avoiding spurious recompilations with SLC.
    """
    def __init__(self, dim: int, base: float = 10000.0, partial_dims: int = 0, max_seq_len: int = 2048):
        super().__init__()
        self.rope_dims = partial_dims if partial_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("_sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        return (
            self._cos_cached[:, :, :seq_len, :].to(dtype=dtype),
            self._sin_cached[:, :, :seq_len, :].to(dtype=dtype),
        )


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rope_dims = cos.size(-1) * 2
    if rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class HymbaAttention(nn.Module):
    """Hymba-style hybrid: attention + Mamba SSM in parallel within one block."""
    def __init__(
        self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
        qk_gain_init: float, mamba_expand: int = 2, conv_kernel_size: int = 4,
        dt_rank: int = 0, ssm_state_size: int = 16, shared_kv_proj=None,
        partial_rope_dims: int = 0, use_rope: bool = True,
        mla_kv_lora_rank: int = 0, mla_q_lora_rank: int = 0, mla_qk_rope_dim: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.intermediate_size = mamba_expand * dim
        self.ssm_state_size = ssm_state_size
        self.dt_rank = dt_rank if dt_rank > 0 else max(dim // 16, 1)
        self.use_rope = use_rope
        self.use_mla = mla_kv_lora_rank > 0

        if self.use_mla:
            # MLA: Multi-Head Latent Attention (DeepSeek-V2 style)
            self.kv_lora_rank = mla_kv_lora_rank
            self.qk_rope_dim = mla_qk_rope_dim if mla_qk_rope_dim > 0 else self.head_dim // 2
            self.qk_nope_dim = self.head_dim - self.qk_rope_dim

            # Q path (optional compression)
            if mla_q_lora_rank > 0:
                self.q_a_proj = CastedLinear(dim, mla_q_lora_rank, bias=False)
                self.q_a_norm = RMSNorm()
                self.q_b_proj = CastedLinear(mla_q_lora_rank, num_heads * self.head_dim, bias=False)
            else:
                self.c_q = CastedLinear(dim, dim, bias=False)

            # KV down-projection (fused: compressed KV + decoupled RoPE key)
            if shared_kv_proj is not None:
                self.kv_a_proj = shared_kv_proj
            else:
                self.kv_a_proj = CastedLinear(dim, mla_kv_lora_rank + self.qk_rope_dim, bias=False)
            self.kv_a_norm = RMSNorm()
            # KV up-projection → per-head [k_nope, v]
            self.kv_b_proj = CastedLinear(
                mla_kv_lora_rank, num_heads * (self.qk_nope_dim + self.head_dim), bias=False,
            )

            # Decoupled RoPE (applied only to rope portions of Q and K)
            self.rotary = Rotary(self.qk_rope_dim, base=rope_base) if use_rope else None
        else:
            # Standard GQA path
            kv_dim = num_kv_heads * self.head_dim
            self.kv_dim = kv_dim
            self.c_q = CastedLinear(dim, dim, bias=False)
            if shared_kv_proj is not None:
                self.kv_proj = shared_kv_proj
            else:
                self.kv_proj = CastedLinear(dim, kv_dim * 2, bias=False)
            self.rotary = Rotary(self.head_dim, base=rope_base, partial_dims=partial_rope_dims) if use_rope else None

        self.mamba_proj = CastedLinear(dim, self.intermediate_size * 2, bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))

        self.conv1d = nn.Conv1d(
            self.intermediate_size, self.intermediate_size, bias=True,
            kernel_size=conv_kernel_size, groups=self.intermediate_size,
            padding=conv_kernel_size - 1,
        )
        self.x_proj = CastedLinear(self.intermediate_size, self.dt_rank + ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.intermediate_size, bias=True)

        A = torch.arange(1, ssm_state_size + 1, dtype=torch.float32)[None, :].expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

        self.mamba_out_proj = CastedLinear(self.intermediate_size, dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.merge_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @torch.compiler.disable
    def _ssm_forward(self, x_ssm: Tensor, gate: Tensor) -> Tensor:
        """SSM (Mamba) path — disabled from torch.compile to avoid graph breaks on CUDA extensions."""
        _conv_dtype = x_ssm.dtype
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2)).to(_conv_dtype)
        conv_bias = self.conv1d.bias.to(_conv_dtype) if self.conv1d.bias is not None else None
        x_ssm = causal_conv1d_fn(x_ssm, conv_weights, conv_bias, activation="silu")

        ssm_params = self.x_proj(x_ssm.transpose(1, 2))
        dt, B_ssm, C_ssm = torch.split(
            ssm_params, [self.dt_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        dt_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        dt = self.dt_proj(dt).transpose(1, 2)
        self.dt_proj.bias = dt_proj_bias

        A = -torch.exp(self.A_log.float())
        dt_proj_bias_f = dt_proj_bias.float() if dt_proj_bias is not None else None

        scan_out = selective_scan_fn(
            x_ssm, dt, A,
            B_ssm.transpose(1, 2), C_ssm.transpose(1, 2),
            self.D.float(), z=gate,
            delta_bias=dt_proj_bias_f,
            delta_softplus=True,
            return_last_state=False,
        )
        return self.mamba_out_proj(scan_out.transpose(1, 2))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape

        # SSM projections (shared by both MLA and GQA paths)
        mamba_out_proj = self.mamba_proj(x)
        x_ssm, gate = mamba_out_proj.split([self.intermediate_size, self.intermediate_size], dim=-1)

        if self.use_mla:
            # --- MLA Attention Path ---
            # Q projection (with optional compression)
            if hasattr(self, 'q_a_proj'):
                q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
            else:
                q = self.c_q(x)
            q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))

            # Split Q into nope and rope parts
            q_nope, q_rope = q.split([self.qk_nope_dim, self.qk_rope_dim], dim=-1)

            # KV down-projection: [compressed_kv, decoupled_rope_key]
            kv_a = self.kv_a_proj(x)
            c_kv, k_rope = kv_a.split([self.kv_lora_rank, self.qk_rope_dim], dim=-1)

            # KV up-projection → per-head [k_nope, v]
            kv_up = self.kv_b_proj(self.kv_a_norm(c_kv))
            kv_up = kv_up.reshape(bsz, seqlen, self.num_heads, self.qk_nope_dim + self.head_dim)
            kv_up = kv_up.transpose(1, 2)
            k_nope, v = kv_up.split([self.qk_nope_dim, self.head_dim], dim=-1)

            # Apply decoupled RoPE to rope portions
            k_rope = k_rope.unsqueeze(1)  # [B, 1, L, qk_rope_dim] — shared across heads
            if self.rotary is not None:
                cos, sin = self.rotary(seqlen, x.device, q.dtype)
                q_rope = apply_rotary_emb(q_rope, cos, sin)
                k_rope = apply_rotary_emb(k_rope, cos, sin)

            # Assemble full Q and K: [nope ; rope]
            k_rope = k_rope.expand(-1, self.num_heads, -1, -1)
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)

            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        else:
            # --- Standard GQA Attention Path ---
            q_out = self.c_q(x)
            kv_out = self.kv_proj(x)
            k_out, v_out = kv_out.split([self.kv_dim, self.kv_dim], dim=-1)

            q = q_out.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v_out.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            if self.rotary is not None:
                cos, sin = self.rotary(seqlen, x.device, q.dtype)
                q = apply_rotary_emb(q, cos, sin)
                k = apply_rotary_emb(k, cos, sin)
            q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)

            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)

        # SSM (Mamba) path — runs eagerly via _ssm_forward to avoid CUDA-extension graph breaks
        mamba_out = self._ssm_forward(x_ssm.transpose(1, 2), gate.transpose(1, 2))

        # Merge attention + SSM
        w = torch.sigmoid(self.merge_alpha).to(dtype=x.dtype)
        merged = w * attn_out + (1.0 - w) * mamba_out
        return self.proj(merged)


class MLP(nn.Module):
    """SwiGLU MLP — gated SiLU activation, iso-param to original."""
    def __init__(self, dim: int, mlp_mult: int, leaky_relu_slope: float = 0.9):
        super().__init__()
        hidden = (mlp_mult * dim * 2 // 3 + 63) // 64 * 64  # iso-param SwiGLU hidden
        self.gate_up = CastedLinear(dim, hidden * 2, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        gu = self.gate_up(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.proj(F.silu(gate) * up)


class Block(nn.Module):
    """Hymba block: parallel Attention+SSM, then MLP, with learnable scales and residual mixing."""
    def __init__(
        self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
        rope_base: float, qk_gain_init: float, hymba_expand: int = 2,
        hymba_conv_kernel: int = 4, hymba_dt_rank: int = 0, hymba_ssm_state: int = 16,
        shared_kv_proj=None, leaky_relu_slope: float = 0.9,
        partial_rope_dims: int = 0, use_rope: bool = True,
        mla_kv_lora_rank: int = 0, mla_q_lora_rank: int = 0, mla_qk_rope_dim: int = 0,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = HymbaAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            mamba_expand=hymba_expand, conv_kernel_size=hymba_conv_kernel,
            dt_rank=hymba_dt_rank, ssm_state_size=hymba_ssm_state,
            shared_kv_proj=shared_kv_proj,
            partial_rope_dims=partial_rope_dims,
            use_rope=use_rope,
            mla_kv_lora_rank=mla_kv_lora_rank,
            mla_q_lora_rank=mla_q_lora_rank,
            mla_qk_rope_dim=mla_qk_rope_dim,
        )
        self.mlp = MLP(dim, mlp_mult, leaky_relu_slope=leaky_relu_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class HymbaModel(nn.Module):
    """Hymba: Hybrid Attention + Mamba SSM language model with U-Net skip connections."""

    def __init__(self, args: Hyperparameters) -> None:
        super().__init__()
        self.vocab_size = args.vocab_size
        self.tie_embeddings = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap = args.logit_softcap
        self.use_ortho_init = args.use_ortho_init
        self.use_unet_skip = args.use_unet_skip
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        self.smeargate = SmearGate(args.model_dim) if args.use_smeargate else None
        self.bigram_hash = BigramHash(
            args.bigram_buckets, args.bigram_hash_dim, args.model_dim
        ) if args.use_bigram_hash else None

        num_layers = args.num_layers
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        self.skip_weights = nn.Parameter(
            torch.ones(1, self.num_skip_weights, args.model_dim, dtype=torch.float32)
        ) if args.use_unet_skip else None

        # Build blocks with optional cross-layer KV sharing
        use_mla = args.mla_kv_lora_rank > 0
        blocks = []
        for i in range(num_layers):
            shared_kv = None
            if args.kv_share_every > 1 and (i % args.kv_share_every) != 0:
                leader_idx = (i // args.kv_share_every) * args.kv_share_every
                if use_mla:
                    shared_kv = blocks[leader_idx].attn.kv_a_proj
                else:
                    shared_kv = blocks[leader_idx].attn.kv_proj
            blocks.append(Block(
                args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                args.rope_base, args.qk_gain_init,
                hymba_expand=args.hymba_expand, hymba_conv_kernel=args.hymba_conv_kernel,
                hymba_dt_rank=args.hymba_dt_rank, hymba_ssm_state=args.hymba_ssm_state,
                shared_kv_proj=shared_kv, leaky_relu_slope=args.leaky_relu_slope,
                partial_rope_dims=args.partial_rope_dims,
                use_rope=(i % 2 == 0),  # RNoPE: even=RoPE, odd=NoPE
                mla_kv_lora_rank=args.mla_kv_lora_rank,
                mla_q_lora_rank=args.mla_q_lora_rank,
                mla_qk_rope_dim=args.mla_qk_rope_dim,
            ))
        self.blocks = nn.ModuleList(blocks)

        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

        # MTP auxiliary prediction heads (training-only, stripped before save)
        self.mtp_projs = None
        if args.mtp_heads > 0:
            self.mtp_projs = nn.ModuleList([
                CastedLinear(args.model_dim, args.model_dim, bias=False)
                for _ in range(args.mtp_heads)
            ])

        # Attention gate init: early layers favor SSM, late layers favor attention
        with torch.no_grad():
            for i, block in enumerate(self.blocks):
                # Linear interpolation: layer 0 → -2.0 (SSM), last layer → +2.0 (Attn)
                t = i / max(num_layers - 1, 1)
                block.attn.merge_alpha.fill_(-2.0 + 4.0 * t)

        # MTP projections: identity init so aux heads start ≈ primary head
        if self.mtp_projs is not None:
            with torch.no_grad():
                for proj in self.mtp_projs:
                    nn.init.eye_(proj.weight)

        n_params = sum(p.numel() for pid, p in {id(p): p for p in self.parameters()}.items())
        mtp_str = f" mtp_heads={args.mtp_heads}" if args.mtp_heads > 0 else ""
        mla_str = f" MLA(kv_rank={args.mla_kv_lora_rank}, rope={args.mla_qk_rope_dim or self.blocks[0].attn.head_dim // 2})" if use_mla else ""
        print(f"HymbaModel: {num_layers} layers | params={n_params:,} "
              f"dim={args.model_dim} heads={args.num_heads} kv_heads={args.num_kv_heads}{mtp_str}{mla_str}")

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif self.use_ortho_init and module.weight.ndim == 2 and min(module.weight.shape) >= 16:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name and name.split(".")[-1] in ("proj",):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _embed(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram_hash is not None:
            x = x + self.bigram_hash(input_ids)
        if self.smeargate is not None:
            x = self.smeargate(x)
        return F.rms_norm(x, (x.size(-1),))

    def _run_blocks(self, x: Tensor, x0: Tensor) -> Tensor:
        if self.use_unet_skip:
            skips: list[Tensor] = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[0, i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        else:
            for block in self.blocks:
                x = block(x, x0)
        return x

    def _logits_loss(self, h_flat: Tensor, targets: Tensor) -> Tensor:
        """Compute logits (with softcap) and cross-entropy from normed hidden states."""
        if self.tie_embeddings:
            logits = F.linear(h_flat, self.tok_emb.weight)
        else:
            logits = self.lm_head(h_flat)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def _compute_logits_and_loss(self, x: Tensor, target_ids: Tensor) -> Tensor:
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward(self, input_ids: Tensor, target_ids: Optional[Tensor] = None) -> Tensor:
        x = self._embed(input_ids)
        x0 = x
        x = self._run_blocks(x, x0)
        if target_ids is not None:
            # MTP training: predict t+1, t+2, ... t+n with shared trunk
            if self.training and self.mtp_projs is not None and len(self.mtp_projs) > 0:
                n = len(self.mtp_projs) + 1
                h = self.final_norm(x)
                d = h.size(-1)
                # Primary head: predict t+1
                loss = self._logits_loss(h.reshape(-1, d), target_ids.reshape(-1))
                # Auxiliary heads: predict t+2, t+3, ...
                for k, proj in enumerate(self.mtp_projs, 1):
                    sl = h.size(1) - k
                    if sl <= 0:
                        continue
                    hk = proj(h[:, :sl, :]).reshape(-1, d)
                    loss = loss + self._logits_loss(hk, target_ids[:, k:].reshape(-1))
                return loss / n
            return self._compute_logits_and_loss(x, target_ids)
        # Return logits for inference (no MTP)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        w = self.tok_emb.weight if self.tie_embeddings else self.lm_head.weight
        logits = self.logit_softcap * torch.tanh(F.linear(x, w) / self.logit_softcap)
        return logits.reshape(input_ids.size(0), input_ids.size(1), -1)


# =============================================================================
# SEQUENCE LENGTH CURRICULUM
# =============================================================================

def get_slc_seq_len(step: int, args) -> int:
    """Return the current training sequence length under SLC.

    Lengths are rounded down to the nearest power of 2 to keep the number of
    distinct shapes small (e.g. 128→256→512→1024). With dynamic=True this is
    not strictly required, but fewer distinct shapes means fewer guard checks.
    """
    if not args.slc_enabled or args.slc_min_seq_len >= args.train_seq_len:
        return args.train_seq_len
    if step >= args.slc_warmup_iters:
        return args.train_seq_len
    t = step / args.slc_warmup_iters          # progress in [0, 1)
    if args.slc_schedule == "quadratic":
        t = t ** 2
    elif args.slc_schedule == "root":
        t = t ** 0.5
    raw = args.slc_min_seq_len + (args.train_seq_len - args.slc_min_seq_len) * t
    # Floor to largest power of 2 that fits in raw
    p = 1
    while p * 2 <= raw:
        p *= 2
    return min(p, args.train_seq_len)


def _slc_distinct_lengths(args) -> list[int]:
    """All distinct seq_lens that SLC will use — for warmup precompilation."""
    if not args.slc_enabled or args.slc_min_seq_len >= args.train_seq_len:
        return []
    seen: set[int] = set()
    for i in range(101):
        seen.add(get_slc_seq_len(round(i / 100 * args.slc_warmup_iters), args))
    return sorted(seen)


# =============================================================================
# TRAINING
# =============================================================================

def main() -> None:
    global zeropower_via_newtonschulz5

    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.config.cache_size_limit = 16
    except (ImportError, AttributeError):
        pass

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    if args.torch_compile:
        zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ── Distributed + device setup ────────────────────────────────────────────
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    # Single H100: grad_accum_steps=1 for max throughput
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
    grad_scale       = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run train.py")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    from torch.backends.cuda import (
        enable_cudnn_sdp, enable_flash_sdp,
        enable_math_sdp, enable_mem_efficient_sdp,
    )
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt" if master_process else None

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
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── Tokenizer + validation metric ─────────────────────────────────────────
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={sp.vocab_size()}"
        )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled  tokenizer={args.tokenizer_path}")
    log0(f"val_tokens:{val_tokens.numel() - 1:,}  train_files={args.train_files}")
    log0(f"time_budget:{TIME_BUDGET_SECONDS:.0f}s  world_size:{world_size}  "
         f"grad_accum:{grad_accum_steps}")

    # ── Model ─────────────────────────────────────────────────────────────────
    base_model = HymbaModel(args).to(device).bfloat16()
    # Keep CastedLinear weights in FP32
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    if args.torch_compile:
        compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    else:
        compiled_model = base_model
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    # ── Optimizer — separate LR groups per SOTA ───────────────────────────────
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights is not None and base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.smeargate is not None:
        scalar_params.append(base_model.smeargate.gate)
    if base_model.bigram_hash is not None:
        scalar_params.append(base_model.bigram_hash.table.weight)
        matrix_params.append(base_model.bigram_hash.proj.weight)
    if base_model.mtp_projs is not None:
        for proj in base_model.mtp_projs:
            matrix_params.append(proj.weight)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params:,}  num_layers:{args.num_layers}  "
         f"model_dim:{args.model_dim}  mlp_mult:{args.mlp_mult}")

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # ── LR schedule — cosine warmdown based on wallclock ──────────────────────
    max_wallclock_ms = TIME_BUDGET_SECONDS * 1000.0

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            frac = remaining_ms / max(warmdown_ms, 1e-9)
        else:
            frac = 1.0
        if args.warmdown_shape == "cosine" and frac < 1.0:
            return 0.5 * (1.0 + math.cos(math.pi * (1.0 - frac)))
        return frac

    # ── EMA setup (BEMA: bias-corrected EMA, arxiv 2508.00180) ──────────────
    ema_state: dict[str, Tensor] | None = None
    ema_n_updates = 0  # count of EMA updates for bias correction
    if args.ema_enabled:
        # Initialize to zeros for BEMA (bias correction handles the init)
        ema_state = {name: torch.zeros_like(t, dtype=torch.float32) for name, t in base_model.state_dict().items()}

    # ── Warmup: compile/trace paths without counting against time budget ──────
    if args.warmup_steps > 0:
        init_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states  = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        train_loader_wu  = DistributedTokenLoader(args.train_files, rank, world_size, device)
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader_wu.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
                log0(f"warmup {ws + 1}/{args.warmup_steps}")
        # ── SLC precompile: trigger one fwd+bwd per shorter discrete length ──
        # This runs during the already-untimed warmup phase so recompilations
        # don't eat into the training time budget.
        slc_extra_lens = [l for l in _slc_distinct_lengths(args) if l != args.train_seq_len]
        if slc_extra_lens:
            log0(f"slc: precompiling {len(slc_extra_lens)} extra seq_lens {slc_extra_lens}")
            for sl in slc_extra_lens:
                zero_grad_all()
                for _ in range(grad_accum_steps):
                    x, y = train_loader_wu.next_batch(args.train_batch_tokens, sl, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y)
                    (loss * grad_scale).backward()
                zero_grad_all()

        base_model.load_state_dict(init_model_state, strict=True)
        for opt, state in zip(optimizers, init_opt_states):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        # Re-init BEMA after warmup reset
        if ema_state is not None:
            ema_state = {name: torch.zeros_like(t, dtype=torch.float32) for name, t in base_model.state_dict().items()}
            ema_n_updates = 0
        # Reset train loader
        train_loader_init = DistributedTokenLoader(args.train_files, rank, world_size, device)
    else:
        train_loader_init = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Main training loop ────────────────────────────────────────────────────
    train_loader     = train_loader_init
    training_time_ms = 0.0
    stop_at_step: int | None = None

    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0

    while True:
        last_step = (
            step == args.max_iterations
            or (stop_at_step is not None and step >= stop_at_step)
        )

        # ── Validation ────────────────────────────────────────────────────────
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            # Use BEMA weights for evaluation if available (bias-corrected)
            if ema_state is not None and ema_n_updates > 0:
                current_state = base_model.state_dict()
                bias_correction = 1.0 - args.ema_decay ** ema_n_updates
                avg_state = {
                    name: (t / bias_correction).to(dtype=current_state[name].dtype)
                    for name, t in ema_state.items()
                }
                base_model.load_state_dict(avg_state, strict=True)

            val_loss, val_bpb = eval_val(
                model, rank, world_size, device, grad_accum_steps,
                val_tokens, args.train_seq_len, args.val_batch_size,
                base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )

            # Restore training weights after eval
            if ema_state is not None and ema_n_updates > 0:
                base_model.load_state_dict(
                    {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in current_state.items()},
                    strict=True,
                )

            log0(
                f"step:{step}  val_loss:{val_loss:.4f}  val_bpb:{val_bpb:.4f}  "
                f"train_time:{training_time_ms:.0f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            log0(f"{'early_stop:wallclock  ' if stop_at_step else ''}final val_bpb:{val_bpb:.4f}")
            break

        # ── LR update ─────────────────────────────────────────────────────────
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        # ── Forward / backward ────────────────────────────────────────────────
        current_seq_len = get_slc_seq_len(step, args)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch(
                args.train_batch_tokens, current_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1

        # ── BEMA update (bias-corrected EMA) ─────────────────────────────────
        if ema_state is not None and step % args.ema_every == 0:
            d = args.ema_decay ** args.ema_every
            ema_n_updates += 1
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (
            step <= 10 or step % args.train_log_every == 0
        ):
            slc_tag = f"  seq_len:{current_seq_len}" if args.slc_enabled and current_seq_len != args.train_seq_len else ""
            log0(
                f"step:{step}  train_loss:{train_loss.item():.4f}{slc_tag}  "
                f"train_time:{approx_ms:.0f}ms  step_avg:{approx_ms / step:.2f}ms"
            )

        # ── Wallclock cap ─────────────────────────────────────────────────────
        reached_cap = approx_ms >= max_wallclock_ms
        if distributed:
            cap_t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_t, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_t.item())
        if stop_at_step is None and reached_cap:
            stop_at_step = step

    # ── Apply BEMA for final save (bias-corrected) ─────────────────────────
    if ema_state is not None and ema_n_updates > 0:
        bias_correction = 1.0 - args.ema_decay ** ema_n_updates
        log0(f"bema:applying bias-corrected EMA weights (n_updates={ema_n_updates}, correction={bias_correction:.6f})")
        current_state = base_model.state_dict()
        avg_state = {name: (t / bias_correction).to(dtype=current_state[name].dtype) for name, t in ema_state.items()}
        base_model.load_state_dict(avg_state, strict=True)

    # ── Strip MTP heads (training-only) before saving ───────────────────────
    if base_model.mtp_projs is not None:
        mtp_n = len(base_model.mtp_projs)
        mtp_params = sum(p.numel() for proj in base_model.mtp_projs for p in proj.parameters())
        base_model.mtp_projs = None
        log0(f"mtp: stripped {mtp_n} auxiliary heads ({mtp_params:,} params)")

    # ── Save checkpoint + artifact measurement + roundtrip verification ─────
    if master_process:
        torch.save(base_model.state_dict(), "logs/final_model.pt")
        log0(f"checkpoint saved: logs/final_model.pt")

        # Measure artifact size and save compressed model
        code_paths = [str(Path(__file__).resolve())]
        log0("── Artifact size measurement ──")
        artifact = measure_and_save_artifact(
            base_model, code_paths, save_dir="logs",
            fp32_name_patterns=(), log_fn=log0,
        )

        # Roundtrip: load int8 from disk → dequantize → eval
        q_val_loss, q_val_bpb = verify_artifact_roundtrip(
            base_model, artifact["ptz_path"],
            rank, world_size, device, grad_accum_steps,
            val_tokens, args.train_seq_len, args.val_batch_size,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            log_fn=log0,
        )
        log0(f"  bpb_delta_from_quantization: {q_val_bpb - val_bpb:+.4f}")

        log0(
            f"peak_memory: "
            f"{torch.cuda.max_memory_allocated() // 1024 // 1024} MiB allocated  "
            f"{torch.cuda.max_memory_reserved() // 1024 // 1024} MiB reserved"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
