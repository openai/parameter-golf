"""
Masked Diffusion Language Model, adapted to train_gpt.py from https://github.com/kuleshov-group/mdlm
This is a discrete diffusion model that masks out some of the tokens in the sequence and then denoises them.
Other than the objective, sampling schedule, and conditioning, this is largely architecturally similar to train_gpt.py for A2A comparison.
The hypers have been untouched since I don't respect diffusion language models, but in theory diffusion models may be useful for this challenge since they can spend arbitrary amounts of compute per-token.
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
try:
    import zstandard as zstd_mod
except ImportError:
    zstd_mod = None
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

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
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 512))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    cond_dim = int(os.environ.get("COND_DIM", 64))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    dropout = float(os.environ.get("DROPOUT", 0.0))

    # Diffusion / denoising setup.
    noise_eps = float(os.environ.get("NOISE_EPS", 1e-3))
    sampling_eps = float(os.environ.get("SAMPLING_EPS", 1e-3))
    sampling_steps = int(os.environ.get("SAMPLING_STEPS", 256))
    var_eval_steps = int(os.environ.get("VAR_EVAL_STEPS", 256))
    sampler = os.environ.get("SAMPLER", "ddpm_cache")
    sampling_schedule = os.environ.get("SAMPLING_SCHEDULE", "linear")
    antithetic_sampling = bool(int(os.environ.get("ANTITHETIC_SAMPLING", "1")))
    time_conditioning = bool(int(os.environ.get("TIME_CONDITIONING", "1")))

    # Compression / export knobs.
    int6_weights = bool(int(os.environ.get("INT6_WEIGHTS", "1")))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))
    use_zstd = bool(int(os.environ.get("USE_ZSTD", "1")))
    zstd_level = int(os.environ.get("ZSTD_LEVEL", 22))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
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
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

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
    model: MDLM,
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
    # - val_loss: continuous-time denoising objective used for training
    # - val_var_bpb: byte-normalized variational upper bound on NLL
    #   This is the quantity we can compare most honestly to AR compression metrics.
    #   It is still an upper bound rather than an exact NLL, but unlike the raw
    #   denoising loss it corresponds to a bona fide latent-variable codelength.
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
    val_vlb_bits_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            prev_ids = local[:-1].reshape(-1, args.train_seq_len)
            x0 = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss, batch_vlb_bits = model.validation_metrics_on_batch(
                    x0,
                    deterministic_t=True,
                    t_offset=float(batch_seq_start),
                )
            batch_token_count = float(x0.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_vlb_bits_sum += batch_vlb_bits.to(torch.float64).sum()
            val_token_count += batch_token_count
            prev_ids = prev_ids.reshape(-1)
            tgt_ids = x0.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_vlb_bits_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    val_var_bpb = val_vlb_bits_sum / val_byte_count
    model.train()
    return float(val_loss.item()), float(val_var_bpb.item())

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
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
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

def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """Int6-equivalent export: store int8 values snapped to multiples of 4."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q_raw = torch.round(clipped / scale[:, None])
        q = torch.clamp((torch.round(q_raw / 4) * 4), -128, 124).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q_raw = torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale)
    q = torch.clamp((torch.round(q_raw / 4) * 4), -128, 124).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(
    state_dict: dict[str, Tensor],
    int6_weights: bool = False,
):
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
        use_int6 = int6_weights and name != "vocab_embed.weight"
        q, s = quantize_float_tensor_int6(t) if use_int6 else quantize_float_tensor(t)
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
    # one disjoint span per rank and reshapes it into fixed-length clean sequences.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> Tensor:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        if local_tokens % seq_len != 0:
            raise ValueError(
                f"Per-rank token count {local_tokens} must be divisible by seq_len={seq_len} "
                f"(TRAIN_BATCH_TOKENS={global_tokens}, WORLD_SIZE={self.world_size}, "
                f"GRAD_ACCUM_STEPS={grad_accum_steps})"
            )
        chunk = self.stream.take(local_tokens * self.world_size)
        start = self.rank * local_tokens
        local = chunk[start : start + local_tokens].to(dtype=torch.int64)
        return local.reshape(-1, seq_len).to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    _qat: bool = False
    _qat_int6: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self._qat and self.training and w.ndim == 2:
            w_f = w.float()
            amax = w_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            scale = amax / 127.0
            q_raw = (w_f / scale).round()
            if self._qat_int6:
                q = (q_raw / 4).round() * 4
                q = q.clamp(-128, 124)
            else:
                q = q_raw.clamp(-127, 127)
            w_q = q * scale
            w = w + (w_q - w_f).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


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


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float | None = None):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.dim,), eps=self.eps)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            CastedLinear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            CastedLinear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class BidirectionalSelfAttention(nn.Module):
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
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=True)
        self.proj = CastedLinear(hidden, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        cond_dim: int,
        mlp_mult: int,
        rope_base: float,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attn = BidirectionalSelfAttention(dim, num_heads, num_kv_heads, rope_base)
        self.mlp = MLP(dim, mlp_mult)
        self.dropout = nn.Dropout(dropout)
        self.adaLN_modulation = CastedLinear(cond_dim, 6 * dim, bias=True)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_msa[:, None, :].to(dtype=x.dtype) * self.dropout(attn_out)
        mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp[:, None, :].to(dtype=x.dtype) * self.dropout(mlp_out)
        return x


class DDitFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        cond_dim: int,
    ):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = CastedLinear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = CastedLinear(cond_dim, 2 * hidden_size, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class LogLinearNoise(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.total_noise(t), self.rate_noise(t)

    def rate_noise(self, t: Tensor) -> Tensor:
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t: Tensor) -> Tensor:
        return -torch.log1p(-(1 - self.eps) * t)


def sample_categorical(categorical_probs: Tensor) -> Tensor:
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def hashed_uniform_like(reference: Tensor, offset: float) -> Tensor:
    # Validation metrics need deterministic latent-noise samples so repeated evals on the
    # same checkpoint are stable. We use a simple hash of token position + offset rather than
    # an RNG stateful stream so distributed workers can reproduce the same q-samples exactly.
    positions = torch.arange(reference.numel(), device=reference.device, dtype=torch.float32).reshape_as(reference)
    hashed = torch.sin(positions * 12.9898 + (1.0 + offset) * 78.233) * 43758.5453
    return torch.remainder(hashed, 1.0)


class MDLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        cond_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        dropout: float,
        noise_eps: float,
        sampling_eps: float,
        sampling_steps: int,
        var_eval_steps: int,
        sampler: str,
        sampling_schedule: str,
        antithetic_sampling: bool,
        time_conditioning: bool,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if (model_dim // num_heads) % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.base_vocab_size = vocab_size
        self.mask_index = vocab_size
        self.vocab_size = vocab_size + 1
        self.cond_dim = cond_dim
        self.sampling_eps = sampling_eps
        self.sampling_steps = sampling_steps
        self.var_eval_steps = var_eval_steps
        self.sampler = sampler
        self.sampling_schedule = sampling_schedule
        self.antithetic_sampling = antithetic_sampling
        self.time_conditioning = time_conditioning
        self.neg_infinity = -1_000_000.0
        if self.sampler not in {"ddpm", "ddpm_cache"}:
            raise ValueError(f"Unsupported sampler: {self.sampler}")
        if self.sampling_schedule not in {"linear", "quadratic", "cubic"}:
            raise ValueError(f"Unsupported sampling_schedule: {self.sampling_schedule}")

        self.vocab_embed = nn.Embedding(self.vocab_size, model_dim)
        nn.init.kaiming_uniform_(self.vocab_embed.weight, a=math.sqrt(5))
        self.sigma_map = TimestepEmbedder(self.cond_dim)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    self.cond_dim,
                    mlp_mult,
                    rope_base,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = DDitFinalLayer(model_dim, self.vocab_size, self.cond_dim)
        self.noise = LogLinearNoise(noise_eps)

    def _process_sigma(self, sigma: Tensor) -> Tensor:
        sigma = sigma.reshape(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        return sigma

    def backbone_logits(self, xt: Tensor, sigma: Tensor) -> Tensor:
        sigma = self._process_sigma(sigma)
        x = self.vocab_embed(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, c)
        return self.output_layer(x, c)

    def subs_log_probs(self, xt: Tensor, sigma: Tensor) -> Tensor:
        logits = self.backbone_logits(xt, sigma).float()
        logits[:, :, self.mask_index] = self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        frozen_logits = torch.full_like(logits, self.neg_infinity)
        frozen_logits.scatter_(-1, xt[..., None], 0.0)
        visible_mask = (xt != self.mask_index)[..., None]
        return torch.where(visible_mask, frozen_logits, logits)

    def q_xt(self, x0: Tensor, move_chance: Tensor, uniform_noise: Tensor | None = None) -> Tensor:
        if uniform_noise is None:
            uniform_noise = torch.rand(x0.shape, device=x0.device)
        move_indices = uniform_noise < move_chance
        return torch.where(move_indices, torch.full_like(x0, self.mask_index), x0)

    def sample_t(
        self,
        n: int,
        device: torch.device,
        deterministic_t: bool = False,
        t_offset: float = 0.0,
    ) -> Tensor:
        if deterministic_t:
            t = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / max(n, 1)
            if t_offset:
                t = torch.remainder(t + 0.61803398875 * t_offset, 1.0)
        else:
            t = torch.rand(n, device=device)
            if self.antithetic_sampling:
                t = (t / n + torch.arange(n, device=device) / n) % 1.0
        return (1 - self.sampling_eps) * t + self.sampling_eps

    def build_sampling_timesteps(self, num_steps: int, device: torch.device) -> Tensor:
        u = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        if self.sampling_schedule == "linear":
            curve = 1.0 - u
        elif self.sampling_schedule == "quadratic":
            curve = (1.0 - u).square()
        elif self.sampling_schedule == "cubic":
            curve = (1.0 - u).pow(3)
        else:
            raise ValueError(f"Unknown sampling_schedule: {self.sampling_schedule!r}")
        return self.sampling_eps + (1.0 - self.sampling_eps) * curve

    def loss_on_batch(
        self,
        x0: Tensor,
        deterministic_t: bool = False,
        t_offset: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        t = self.sample_t(x0.shape[0], x0.device, deterministic_t=deterministic_t, t_offset=t_offset)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        uniform_noise = None
        if deterministic_t:
            uniform_noise = hashed_uniform_like(x0, t_offset)
        xt = self.q_xt(x0, move_chance, uniform_noise=uniform_noise)
        log_probs = self.subs_log_probs(xt, sigma)
        log_p_theta = torch.gather(log_probs, dim=-1, index=x0[:, :, None]).squeeze(-1)
        loss = -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        return loss.mean(), loss

    def variational_bits_on_batch(
        self,
        x0: Tensor,
        deterministic_t: bool = False,
        t_offset: float = 0.0,
    ) -> Tensor:
        # Why this exists:
        # ----------------
        # The continuous-time SUBS training loss above is a denoising objective, not an exact
        # token codelength. Converting that loss directly into "bpb" units gives a number,
        # but it is not theoretically comparable to autoregressive validation BPB.
        #
        # For a compression-facing metric we need a proper latent-variable codelength. The exact
        # code for the current continuous process is not tractable because it would require
        # integrating over all latent corruption trajectories. We therefore evaluate a *discrete*
        # absorbing-mask variational bound instead:
        #
        #   KL(q(x_T | x_0) || p(x_T))
        #   + sum_t E_q KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))
        #
        # using the same denoiser p_theta(x_0 | x_t). This is the ELBO given in the MDLM paper.
        # We train with the efficient continuous objective, but report a tractable
        # variational upper bound when we want something closer to a true coding cost.
        #
        # Two modeling decisions are worth spelling out:
        # 1. We discretize time into VAR_EVAL_STEPS. Without that discretization there is no
        #    finite sum of KL terms to evaluate. This is interesting, as the closer to continuous this gets, the lower the upper bound on our BPB.
        # 2. The log-linear schedule never reaches a perfectly masked terminal state; at t=1 a
        #    small fraction of tokens are still visible. To make the terminal distribution
        #    codeable without peeking at x_0, we choose a simple prior that matches the mask mass
        #    and allocates the remaining visible mass uniformly over the clean vocabulary. This
        #    adds the terminal KL term below.
        if self.var_eval_steps <= 0:
            raise ValueError("VAR_EVAL_STEPS must be positive")
        batch_size, seq_len = x0.shape
        device = x0.device
        total_bits = torch.zeros((batch_size,), device=device, dtype=torch.float32)

        t_grid = torch.arange(1, self.var_eval_steps + 1, device=device, dtype=torch.float32) / self.var_eval_steps
        sigma_grid, _ = self.noise(t_grid)
        alpha_grid = torch.exp(-sigma_grid)

        # Terminal prior p(x_T): keep the same mask mass as q(x_T | x_0), but spread any
        # residual visible-token mass uniformly over the clean vocabulary so the prior does not
        # depend on the target sequence itself.
        alpha_T = alpha_grid[-1]
        total_bits = total_bits + (seq_len * alpha_T * math.log(self.base_vocab_size) / math.log(2.0))

        alpha_prev = torch.tensor(1.0, device=device, dtype=torch.float32)
        for step_idx in range(self.var_eval_steps):
            t_curr = t_grid[step_idx].expand(batch_size)
            sigma_curr = sigma_grid[step_idx].expand(batch_size)
            alpha_curr = alpha_grid[step_idx]
            move_chance = 1 - alpha_curr
            uniform_noise = None
            if deterministic_t:
                uniform_noise = hashed_uniform_like(x0, t_offset + 17.0 + 1.61803398875 * step_idx)
            xt = self.q_xt(x0, move_chance.expand(batch_size, 1), uniform_noise=uniform_noise)
            log_probs = self.subs_log_probs(xt, sigma_curr)
            log_p_x0 = torch.gather(log_probs, dim=-1, index=x0[:, :, None]).squeeze(-1)

            # For the absorbing-mask chain, if x_t is already visible then x_{t-1} is determined.
            # Only masked positions contribute a KL term. The coefficient below is the exact
            # posterior probability that a token was still visible at the previous discrete step
            # given that it is masked at the current step.
            reveal_prob = (alpha_prev - alpha_curr) / max(1.0 - float(alpha_curr.item()), 1e-12)
            step_nats = -reveal_prob * log_p_x0 * (xt == self.mask_index)
            total_bits = total_bits + step_nats.sum(dim=-1) / math.log(2.0)
            alpha_prev = alpha_curr
        return total_bits

    def validation_metrics_on_batch(
        self,
        x0: Tensor,
        deterministic_t: bool = False,
        t_offset: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        batch_loss, _ = self.loss_on_batch(x0, deterministic_t=deterministic_t, t_offset=t_offset)
        var_bits = self.variational_bits_on_batch(x0, deterministic_t=deterministic_t, t_offset=t_offset)
        return batch_loss, var_bits

    def forward(self, x0: Tensor) -> Tensor:
        loss, _ = self.loss_on_batch(x0)
        return loss

    @torch.no_grad()
    def ddpm_update(self, x: Tensor, t: Tensor, t_next: Tensor) -> Tensor:
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t_next)
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        log_p_x0 = self.subs_log_probs(x, sigma_t)
        q_xs = log_p_x0.exp() * (move_chance_t[:, None, None] - move_chance_s[:, None, None])
        q_xs[:, :, self.mask_index] = move_chance_s[:, None]
        sampled = sample_categorical(q_xs)
        return torch.where(x != self.mask_index, x, sampled)

    @torch.no_grad()
    def ddpm_caching_update(self, x: Tensor, t: Tensor, t_next: Tensor, p_x0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        # This function is a variant of the DDPM update step
        # that utilizes a cached distribution p_x0 for greater efficiency. This allows us to skip computation of p_x0 for each step.
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t_next)
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        if p_x0 is None:
            p_x0 = self.subs_log_probs(x, sigma_t).exp()
        q_xs = p_x0 * (move_chance_t[:, None, None] - move_chance_s[:, None, None])
        q_xs[:, :, self.mask_index] = move_chance_s[:, None]
        sampled = sample_categorical(q_xs)
        return p_x0, torch.where(x != self.mask_index, x, sampled)

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, device: torch.device, num_steps: int | None = None) -> Tensor:
        if num_steps is None:
            num_steps = self.sampling_steps
        x = torch.full((batch_size, seq_len), self.mask_index, dtype=torch.long, device=device)
        timesteps = self.build_sampling_timesteps(num_steps, device)
        p_x0_cache = None
        for i in range(num_steps):
            t = timesteps[i].expand(batch_size)
            t_next = timesteps[i + 1].expand(batch_size)
            if self.sampler == "ddpm":
                x = self.ddpm_update(x, t, t_next)
            else:
                p_x0_cache, x_next = self.ddpm_caching_update(x, t, t_next, p_x0=p_x0_cache)
                if self.time_conditioning or not torch.equal(x_next, x):
                    p_x0_cache = None
                x = x_next
        sigma, _ = self.noise(timesteps[-1].expand(batch_size))
        return self.subs_log_probs(x, sigma).argmax(dim=-1)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
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
    log0(
        f"val_var_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path} "
        f"var_eval_steps:{args.var_eval_steps}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = MDLM(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        cond_dim=args.cond_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        dropout=args.dropout,
        noise_eps=args.noise_eps,
        sampling_eps=args.sampling_eps,
        sampling_steps=args.sampling_steps,
        var_eval_steps=args.var_eval_steps,
        sampler=args.sampler,
        sampling_schedule=args.sampling_schedule,
        antithetic_sampling=args.antithetic_sampling,
        time_conditioning=args.time_conditioning,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.qat_enabled:
        for module in base_model.modules():
            if isinstance(module, CastedLinear):
                module._qat = True
                module._qat_int6 = args.int6_weights
    compiled_model = torch.compile(base_model, dynamic=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - output projection (Adam) uses HEAD_LR
    # - matrix params in the denoiser body use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    named_params = list(base_model.named_parameters())
    excluded_param_ids = {
        id(base_model.vocab_embed.weight),
        id(base_model.output_layer.linear.weight),
    }
    matrix_params = [
        p
        for name, p in named_params
        if id(p) not in excluded_param_ids and p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named_params
        if id(p) not in excluded_param_ids and (p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]
    token_lr = args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.vocab_embed.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_head = torch.optim.Adam(
        [{"params": [base_model.output_layer.linear.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_head, optimizer_muon, optimizer_scalar]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:bidirectional_gqa num_heads:{args.num_heads} "
        f"num_kv_heads:{args.num_kv_heads} cond_dim:{base_model.cond_dim}"
    )
    log0(
        f"base_vocab_size:{args.vocab_size} model_vocab_size:{base_model.vocab_size} "
        f"mask_index:{base_model.mask_index}"
    )
    log0(
        f"embed_lr:{token_lr} head_lr:{args.head_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"noise:loglinear noise_eps:{args.noise_eps} sampling_eps:{args.sampling_eps} "
        f"sampler:{args.sampler} sampling_schedule:{args.sampling_schedule} "
        f"sampling_steps:{args.sampling_steps} var_eval_steps:{args.var_eval_steps} "
        f"time_conditioning:{int(args.time_conditioning)} antithetic:{int(args.antithetic_sampling)}"
    )
    log0(
        f"export:int6_weights:{int(args.int6_weights)} "
        f"qat:{int(args.qat_enabled)} "
        f"use_zstd:{int(args.use_zstd)} zstd_level:{args.zstd_level}"
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
                x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x0)
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
            val_loss, val_var_bpb = eval_val(
                args,
                base_model,
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
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_var_bpb:{val_var_bpb:.4f} "
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
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x0 = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x0)
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

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(
        base_model.state_dict(),
        int6_weights=args.int6_weights,
    )
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    weight_label = "int6" if args.int6_weights else "int8"
    if args.use_zstd and zstd_mod is not None:
        cctx = zstd_mod.ZstdCompressor(level=args.zstd_level)
        quant_blob = cctx.compress(quant_raw)
        compress_label = f"{weight_label}+zstd{args.zstd_level}"
    else:
        quant_blob = zlib.compress(quant_raw, level=9)
        compress_label = f"{weight_label}+zlib"
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model {compress_label}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size {compress_label}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if args.use_zstd and zstd_mod is not None:
        dctx = zstd_mod.ZstdDecompressor()
        quant_state = torch.load(io.BytesIO(dctx.decompress(quant_blob_disk)), map_location="cpu")
    else:
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_var_bpb = eval_val(
        args,
        base_model,
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
        f"final_quant_roundtrip val_loss:{q_val_loss:.4f} val_var_bpb:{q_val_var_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_quant_roundtrip_exact val_loss:{q_val_loss:.8f} val_var_bpb:{q_val_var_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
