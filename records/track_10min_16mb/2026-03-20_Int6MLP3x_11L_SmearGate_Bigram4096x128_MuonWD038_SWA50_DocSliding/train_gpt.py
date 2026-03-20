"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.
Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
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
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel
try:
    import zstandard
except ImportError:
    zstandard = None
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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_unique_layers = int(os.environ.get("NUM_UNIQUE_LAYERS", 0))
    num_recurrence = int(os.environ.get("NUM_RECURRENCE", 1))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    smeargate_enabled = bool(int(os.environ.get("SMEARGATE_ENABLED", "0")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 64))
    prevlogit_rank = int(os.environ.get("PREVLOGIT_RANK", 0))
    init_ckpt_path = os.environ.get("INIT_CKPT_PATH", "").strip()
    init_ckpt_strict = bool(int(os.environ.get("INIT_CKPT_STRICT", "1")))
    trainable_name_patterns = tuple(
        pattern
        for pattern in os.environ.get("TRAINABLE_NAME_PATTERNS", "").split(",")
        if pattern
    )
    torch_compile_disable = bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0")))
    # Compression-aware training.
    compression_aware_mode = os.environ.get("COMPRESSION_AWARE_MODE", "").strip().lower()
    compression_aware_start_frac = float(os.environ.get("COMPRESSION_AWARE_START_FRAC", 0.60))
    compression_aware_every = int(os.environ.get("COMPRESSION_AWARE_EVERY", 4))
    compression_aware_kl_weight = float(os.environ.get("COMPRESSION_AWARE_KL_WEIGHT", 0.10))
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
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.0))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    cache_train_shards = bool(int(os.environ.get("CACHE_TRAIN_SHARDS", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 0))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
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
            weight_decay = group["weight_decay"]
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
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
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
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear.gate,bigram.scale,prevlogit.scale",
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
INT8_KEEP_FLOAT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("INT8_KEEP_FLOAT_NAME_PATTERNS", "").split(",")
    if pattern
)
FORCE_QUANTIZE_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("FORCE_QUANTIZE_NAME_PATTERNS", "").split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
LOWBIT_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get("LOWBIT_NAME_PATTERNS", "").split(",")
    if pattern
)
LOWBIT_BITS = int(os.environ.get("LOWBIT_BITS", 8))
LOWBIT_CLIP_PERCENTILE = float(os.environ.get("LOWBIT_CLIP_PERCENTILE", INT8_CLIP_PERCENTILE))
LOWBIT_CLIP_Q = LOWBIT_CLIP_PERCENTILE / 100.0
SERIAL_COMPRESSOR = os.environ.get("SERIAL_COMPRESSOR", "zlib").strip().lower()
SERIAL_ZLIB_LEVEL = int(os.environ.get("SERIAL_ZLIB_LEVEL", 9))
SERIAL_ZSTD_LEVEL = int(os.environ.get("SERIAL_ZSTD_LEVEL", 22))
def parse_group_overrides(raw: str) -> tuple[tuple[str, int], ...]:
    parsed: list[tuple[str, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        pattern, sep, group_size_str = item.partition(":")
        if not sep:
            raise ValueError(f"Expected PATTERN:GROUP_SIZE in INT8_GROUP_OVERRIDES, got {item!r}")
        parsed.append((pattern, int(group_size_str)))
    return tuple(parsed)
INT8_GROUP_OVERRIDES = parse_group_overrides(os.environ.get("INT8_GROUP_OVERRIDES", ""))
def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in name for pattern in patterns)
def group_size_for_name(name: str) -> int | None:
    for pattern, group_size in INT8_GROUP_OVERRIDES:
        if pattern in name:
            return group_size
    return None
def lowbit_is_active_for_name(name: str) -> bool:
    return LOWBIT_BITS < 8 and matches_any(name, LOWBIT_NAME_PATTERNS)
def export_variant_tag() -> str:
    if LOWBIT_BITS < 8 and LOWBIT_NAME_PATTERNS:
        return f"mixed_int{LOWBIT_BITS}_{SERIAL_COMPRESSOR}"
    return f"int8_{SERIAL_COMPRESSOR}"
def export_variant_desc() -> str:
    if LOWBIT_BITS < 8 and LOWBIT_NAME_PATTERNS:
        return f"mixed-int{LOWBIT_BITS}+{SERIAL_COMPRESSOR}"
    return f"int8+{SERIAL_COMPRESSOR}"
def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())
def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if matches_any(name, INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if matches_any(name, INT8_KEEP_FLOAT_NAME_PATTERNS):
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t
def quant_range(bits: int) -> tuple[int, int]:
    qmax = 127 if bits >= 8 else max((1 << (bits - 1)) - 1, 1)
    return -qmax, qmax
def quantize_float_tensor(
    t: Tensor,
    *,
    bits: int = 8,
    group_size: int | None = None,
    clip_q: float = INT8_CLIP_Q,
) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    qmin, qmax = quant_range(bits)
    scale_floor = 1.0 / max(qmax, 1)
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        if group_size is not None and group_size > 0 and t32.shape[1] > group_size:
            rows, cols = t32.shape
            groups = math.ceil(cols / group_size)
            pad_cols = groups * group_size - cols
            tpad = F.pad(t32, (0, pad_cols)) if pad_cols else t32
            grouped = tpad.view(rows, groups, group_size)
            clip_abs = (
                torch.quantile(grouped.abs(), clip_q, dim=2)
                if grouped.numel()
                else torch.empty((rows, groups), dtype=torch.float32)
            )
            clipped = torch.maximum(torch.minimum(grouped, clip_abs[..., None]), -clip_abs[..., None])
            scale = (clip_abs / qmax).clamp_min(scale_floor)
            q = torch.clamp(torch.round(clipped / scale[..., None]), qmin, qmax).to(torch.int8)
            q = q.view(rows, groups * group_size)[..., :cols].contiguous()
            return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
        clip_abs = (
            torch.quantile(t32.abs(), clip_q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(scale_floor)
        q = torch.clamp(torch.round(clipped / scale[:, None]), qmin, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), qmin, qmax).to(torch.int8).contiguous()
    return q, scale
def fake_quantize_tensor_export_surrogate(t: Tensor) -> Tensor:
    # Compile-friendly surrogate for the final exporter used only during training.
    # We avoid torch.quantile here because it breaks the current fullgraph compile path.
    if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
        return t
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = t32.abs().amax(dim=1)
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
        deq = (q * scale[:, None]).to(dtype=t.dtype)
        return t + (deq - t).detach()
    clip_abs = t32.abs().amax()
    scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
    clipped = torch.clamp(t32, -clip_abs, clip_abs)
    q = torch.clamp(torch.round(clipped / scale), -127, 127)
    deq = (q * scale).to(dtype=t.dtype)
    return t + (deq - t).detach()
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
def quantize_state_dict_serial(state_dict: dict[str, Tensor]):
    # Configurable serializer used by the dense integrated lane:
    # - optional int6 for selected large matrix groups
    # - optional grouped int8 for selected matrices
    # - fp16 passthrough for explicitly selected sensitive tensors
    # - zlib or zstd outer compression
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "payload_bytes"),
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
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        force_quantize = matches_any(name, FORCE_QUANTIZE_NAME_PATTERNS)
        if (
            (t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL and not force_quantize)
            or matches_any(name, INT8_KEEP_FLOAT_NAME_PATTERNS)
            or matches_any(name, CONTROL_TENSOR_NAME_PATTERNS)
        ):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        bits = LOWBIT_BITS if lowbit_is_active_for_name(name) else 8
        group_size = group_size_for_name(name)
        q, s = quantize_float_tensor(
            t,
            bits=bits,
            group_size=group_size if t.ndim == 2 else None,
            clip_q=LOWBIT_CLIP_Q if bits < 8 else INT8_CLIP_Q,
        )
        scheme = "per_row_group" if t.ndim == 2 and group_size is not None and group_size > 0 and t.shape[1] > group_size else (
            "per_row" if s.ndim > 0 else "per_tensor"
        )
        qmeta[name] = {"scheme": scheme, "bits": bits}
        if scheme == "per_row_group":
            qmeta[name]["group_size"] = int(group_size)
            qmeta[name]["axis"] = 1
        elif scheme == "per_row":
            qmeta[name]["axis"] = 0
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "mixed_lowbit_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
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
        scheme = qmeta.get(name, {}).get("scheme")
        if scheme == "per_row_group":
            group_size = int(qmeta[name]["group_size"])
            scale = s.to(dtype=torch.float32)
            expanded = torch.repeat_interleave(scale, repeats=group_size, dim=1)[:, : q.shape[1]]
            out[name] = (q.float() * expanded).to(dtype=dtype).contiguous()
        elif scheme == "per_row" or s.ndim > 0:
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
def compress_serialized_model(raw: bytes) -> bytes:
    if SERIAL_COMPRESSOR == "zlib":
        return zlib.compress(raw, level=SERIAL_ZLIB_LEVEL)
    if SERIAL_COMPRESSOR == "zstd":
        if zstandard is None:
            raise RuntimeError("SERIAL_COMPRESSOR=zstd requires the zstandard package")
        return zstandard.ZstdCompressor(level=SERIAL_ZSTD_LEVEL).compress(raw)
    raise ValueError(f"Unsupported SERIAL_COMPRESSOR={SERIAL_COMPRESSOR!r}")
def decompress_serialized_model(blob: bytes) -> bytes:
    if SERIAL_COMPRESSOR == "zlib":
        return zlib.decompress(blob)
    if SERIAL_COMPRESSOR == "zstd":
        if zstandard is None:
            raise RuntimeError("SERIAL_COMPRESSOR=zstd requires the zstandard package")
        return zstandard.ZstdDecompressor().decompress(blob)
    raise ValueError(f"Unsupported SERIAL_COMPRESSOR={SERIAL_COMPRESSOR!r}")
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
    def __init__(self, pattern: str, cache_shards: bool = False):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.cache_shards = cache_shards
        self._cached_tokens = [load_data_shard(file) for file in self.files] if cache_shards else None
        self.tokens = self._cached_tokens[0] if self._cached_tokens is not None else load_data_shard(self.files[0])
        self.pos = 0
    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = (
            self._cached_tokens[self.file_idx]
            if self._cached_tokens is not None
            else load_data_shard(self.files[self.file_idx])
        )
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
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device, cache_shards: bool = False):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern, cache_shards=cache_shards)
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
    def forward(self, x: Tensor, use_fake_quant: bool = False) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        weight = fake_quantize_tensor_export_surrogate(self.weight) if use_fake_quant else self.weight
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
        qk_gain_init: float,
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
        self.rotary = Rotary(self.head_dim, base=rope_base)
    def forward(self, x: Tensor, use_fake_quant: bool = False) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, use_fake_quant=use_fake_quant).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x, use_fake_quant=use_fake_quant).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x, use_fake_quant=use_fake_quant).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y, use_fake_quant=use_fake_quant)
class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x: Tensor, use_fake_quant: bool = False) -> Tensor:
        x = torch.relu(self.fc(x, use_fake_quant=use_fake_quant))
        return self.proj(x.square(), use_fake_quant=use_fake_quant)
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1.0 - gate) * x + gate * x_prev
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        if bigram_vocab_size <= 1:
            raise ValueError(f"bigram_vocab_size must be > 1, got {bigram_vocab_size}")
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, token_ids: Tensor) -> Tensor:
        tokens = token_ids.to(torch.int32)
        out = torch.empty_like(tokens)
        mod = self.bigram_vocab_size - 1
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * tokens[..., 1:], 27191 * tokens[..., :-1]) % mod
        return out.to(dtype=torch.int64)
    def forward(self, token_ids: Tensor, use_fake_quant: bool = False) -> Tensor:
        embed_weight = (
            fake_quantize_tensor_export_surrogate(self.embed.weight) if use_fake_quant else self.embed.weight
        )
        x = F.embedding(self.bigram_hash(token_ids), embed_weight)
        if self.proj is not None:
            x = self.proj(x, use_fake_quant=use_fake_quant)
        return x * self.scale.to(dtype=x.dtype)
class PrevTokenLogitHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int, model_dim: int):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"prevlogit rank must be positive, got {rank}")
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size + 1, rank)
        self.proj = CastedLinear(rank, vocab_size, bias=False)
        self.gate = CastedLinear(model_dim, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        nn.init.zeros_(self.embed.weight)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    def previous_tokens(self, token_ids: Tensor) -> Tensor:
        prev = torch.empty_like(token_ids)
        prev[..., 0] = self.vocab_size
        prev[..., 1:] = token_ids[..., :-1]
        return prev
    def forward(self, token_ids: Tensor, hidden: Tensor, use_fake_quant: bool = False) -> Tensor:
        embed_weight = (
            fake_quantize_tensor_export_surrogate(self.embed.weight) if use_fake_quant else self.embed.weight
        )
        prev = F.embedding(self.previous_tokens(token_ids).reshape(-1), embed_weight)
        logits = self.proj(prev, use_fake_quant=use_fake_quant)
        gate = torch.sigmoid(self.gate(hidden, use_fake_quant=use_fake_quant))
        return logits * gate * self.scale.to(dtype=logits.dtype)
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
    def forward(self, x: Tensor, x0: Tensor, use_fake_quant: bool = False) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x), use_fake_quant=use_fake_quant)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x),
            use_fake_quant=use_fake_quant,
        )
        return x
class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_unique_layers: int,
        num_recurrence: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        compression_aware_kl_weight: float,
        smeargate_enabled: bool,
        bigram_vocab_size: int,
        bigram_dim: int,
        prevlogit_rank: int,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if num_unique_layers <= 0:
            num_unique_layers = num_layers
        if num_recurrence <= 0:
            raise ValueError(f"num_recurrence must be positive, got {num_recurrence}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_unique_layers = num_unique_layers
        self.num_recurrence = num_recurrence
        self.num_layers = self.num_unique_layers * self.num_recurrence
        self.compression_aware_kl_weight = compression_aware_kl_weight
        self.reuse_blocks_across_halves = self.num_layers > self.num_unique_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim) if smeargate_enabled else None
        self.prevlogit = PrevTokenLogitHead(vocab_size, prevlogit_rank, model_dim) if prevlogit_rank > 0 else None
        self.num_encoder_layers = self.num_layers // 2
        self.num_decoder_layers = self.num_layers - self.num_encoder_layers
        self.decoder_block_offset = 0 if self.reuse_blocks_across_halves else self.num_encoder_layers
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
                )
                for i in range(self.num_unique_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
    def forward_logits(self, input_ids: Tensor, use_fake_quant: bool = False) -> Tensor:
        emb_weight = fake_quantize_tensor_export_surrogate(self.tok_emb.weight) if use_fake_quant else self.tok_emb.weight
        x = F.embedding(input_ids, emb_weight) if use_fake_quant else self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids, use_fake_quant=use_fake_quant)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i % self.num_unique_layers](x, x0, use_fake_quant=use_fake_quant)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            block_idx = (i % self.num_unique_layers) if self.reuse_blocks_across_halves else (self.decoder_block_offset + i)
            x = self.blocks[block_idx](x, x0, use_fake_quant=use_fake_quant)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, emb_weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x, use_fake_quant=use_fake_quant)
        if self.prevlogit is not None:
            logits_proj = logits_proj + self.prevlogit(input_ids, x, use_fake_quant=use_fake_quant)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    def forward(self, input_ids: Tensor, target_ids: Tensor, compression_aware: bool = False) -> Tensor:
        logits = self.forward_logits(input_ids, use_fake_quant=False)
        targets = target_ids.reshape(-1)
        ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if not compression_aware or self.compression_aware_kl_weight <= 0.0:
            return ce_loss
        shadow_logits = self.forward_logits(input_ids, use_fake_quant=True)
        teacher_probs = F.softmax(logits.detach().float(), dim=-1)
        shadow_log_probs = F.log_softmax(shadow_logits.float(), dim=-1)
        kl_loss = F.kl_div(shadow_log_probs, teacher_probs, reduction="batchmean")
        return ce_loss + self.compression_aware_kl_weight * kl_loss
# -----------------------------
# TRAINING
# -----------------------------
def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if not args.torch_compile_disable:
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
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_unique_layers=args.num_unique_layers,
        num_recurrence=args.num_recurrence,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        compression_aware_kl_weight=args.compression_aware_kl_weight,
        smeargate_enabled=args.smeargate_enabled,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        prevlogit_rank=args.prevlogit_rank,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.init_ckpt_path:
        init_state = torch.load(args.init_ckpt_path, map_location="cpu")
        if isinstance(init_state, dict) and "state_dict" in init_state and isinstance(init_state["state_dict"], dict):
            init_state = init_state["state_dict"]
        if not isinstance(init_state, dict):
            raise TypeError(f"Expected checkpoint state dict at {args.init_ckpt_path}, got {type(init_state)!r}")
        incompatible = base_model.load_state_dict(init_state, strict=args.init_ckpt_strict)
        missing_keys = list(getattr(incompatible, "missing_keys", ()))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", ()))
        log0(
            f"init_ckpt:path:{args.init_ckpt_path} strict:{args.init_ckpt_strict} "
            f"missing:{len(missing_keys)} unexpected:{len(unexpected_keys)}"
        )
        if missing_keys:
            log0(f"init_ckpt:missing_keys:{','.join(missing_keys[:16])}", console=False)
        if unexpected_keys:
            log0(f"init_ckpt:unexpected_keys:{','.join(unexpected_keys[:16])}", console=False)
    if args.trainable_name_patterns:
        trainable_params = 0
        frozen_params = 0
        for name, param in base_model.named_parameters():
            keep_trainable = matches_any(name, args.trainable_name_patterns)
            param.requires_grad_(keep_trainable)
            if keep_trainable:
                trainable_params += int(param.numel())
            else:
                frozen_params += int(param.numel())
        if trainable_params == 0:
            raise ValueError(
                f"TRAINABLE_NAME_PATTERNS={args.trainable_name_patterns!r} matched no parameters"
            )
        log0(
            f"trainable_filter:patterns:{','.join(args.trainable_name_patterns)} "
            f"trainable_params:{trainable_params} frozen_params:{frozen_params}"
        )
    model_impl: nn.Module
    if args.torch_compile_disable:
        model_impl = base_model
    else:
        model_impl = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(model_impl, device_ids=[local_rank], broadcast_buffers=False) if distributed else model_impl
    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    token_params: list[Tensor] = []
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    head_params: list[Tensor] = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "tok_emb.weight" or name == "bigram.embed.weight" or name == "prevlogit.embed.weight":
            token_params.append(param)
        elif name == "lm_head.weight" or name == "prevlogit.proj.weight":
            head_params.append(param)
        elif param.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
            matrix_params.append(param)
        else:
            scalar_params.append(param)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizers: list[torch.optim.Optimizer] = []
    if token_params:
        optimizer_tok = torch.optim.AdamW(
            [{"params": token_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
            fused=True,
        )
        optimizers.append(optimizer_tok)
    if head_params:
        optimizer_head = torch.optim.AdamW(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
            fused=True,
        )
        optimizers.append(optimizer_head)
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
            weight_decay=args.muon_weight_decay,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_muon)
    if scalar_params:
        optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
            fused=True,
        )
        optimizers.append(optimizer_scalar)
    if not optimizers:
        raise RuntimeError("No trainable parameters were assigned to any optimizer")
    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"depth_layout:effective_layers:{base_model.num_layers} "
        f"unique_layers:{base_model.num_unique_layers} recurrence:{base_model.num_recurrence}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if head_params else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"weight_decay:muon:{args.muon_weight_decay} adam:{args.adam_weight_decay} "
        f"swa_every:{args.swa_every} swa_start_frac:{args.swa_start_frac:.2f}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"compression_aware:mode:{args.compression_aware_mode or 'off'} "
        f"start_frac:{args.compression_aware_start_frac:.2f} "
        f"every:{args.compression_aware_every} "
        f"kl_weight:{args.compression_aware_kl_weight:.4f}"
    )
    log0(f"torch_compile:enabled:{not args.torch_compile_disable}")
    log0(f"seed:{args.seed}")
    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, cache_shards=args.cache_train_shards)
    swa_model = AveragedModel(base_model, device=torch.device("cpu")) if args.swa_every > 0 else None
    swa_updates = 0
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
                    warmup_loss = model(x, y, compression_aware=False)
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
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device, cache_shards=args.cache_train_shards)
    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------
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
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        compression_aware_active = (
            args.compression_aware_mode == "shadow_int8_kl"
            and args.compression_aware_every > 0
            and max_wallclock_ms is not None
            and elapsed_ms >= max_wallclock_ms * args.compression_aware_start_frac
            and step % args.compression_aware_every == 0
        )
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, compression_aware=compression_aware_active)
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
                f"compression_aware_active:{compression_aware_active} "
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
        if (
            swa_model is not None
            and max_wallclock_ms is not None
            and args.swa_start_frac < 1.0
            and approx_training_time_ms >= max_wallclock_ms * args.swa_start_frac
            and step % args.swa_every == 0
        ):
            swa_model.update_parameters(base_model)
            swa_updates += 1
    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the configured compressed artifact and validate the round-tripped weights.
    if swa_model is not None and swa_updates > 0:
        log0(f"swa_updates:{swa_updates}")
        base_model.load_state_dict(swa_model.module.state_dict(), strict=True)
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")
    quant_obj, quant_stats = quantize_state_dict_serial(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = compress_serialized_model(quant_raw)
    quant_raw_bytes = len(quant_raw)
    artifact_tag = export_variant_tag()
    artifact_desc = export_variant_desc()
    artifact_path = "final_model.quant.ptz"
    legacy_artifact_path = "final_model.int8.ptz"
    if master_process:
        with open(artifact_path, "wb") as f:
            f.write(quant_blob)
        # Keep the legacy filename as an alias because local helper scripts copy it.
        with open(legacy_artifact_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(artifact_path)
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["payload_bytes"], 1)
        log0(
            f"Serialized model {artifact_desc}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {artifact_desc}: {quant_file_bytes + code_bytes} bytes")
    if distributed:
        dist.barrier()
    with open(artifact_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(decompress_serialized_model(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
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
    torch.cuda.synchronize()
    log0(
        f"final_{artifact_tag}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{artifact_tag}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
    if distributed:
        dist.destroy_process_group()
if __name__ == "__main__":
    main()
