"""
Mercury-style compact diffusion submission candidate.

This fork keeps exact Parameter Golf BPB evaluation and export path, but it
adds a diffusion-native training mode and extra reporting for parallel
coarse-to-fine refinement. The goal is a non-record submission that is
interesting specifically because it acts like a text diffusion model rather
than an AR model with a tiny auxiliary loss.
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
    tokenizer_meta_path = os.environ.get("TOKENIZER_META_PATH", "")
    tokenizer_meta_validate = bool(int(os.environ.get("TOKENIZER_META_VALIDATE", "0")))
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
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

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
    use_compile = bool(int(os.environ.get("USE_COMPILE", "0")))

    # Text-diffusion hybrid knobs.
    diffusion_enabled = bool(int(os.environ.get("DIFFUSION_ENABLED", "1")))
    diffusion_train_mode = os.environ.get("DIFFUSION_TRAIN_MODE", "mercury").strip().lower()
    diffusion_corruption_mode = os.environ.get("DIFFUSION_CORRUPTION_MODE", "hybrid").strip().lower()
    diffusion_hybrid_mask_prob = float(os.environ.get("DIFFUSION_HYBRID_MASK_PROB", 0.5))
    diffusion_loss_weight = float(os.environ.get("DIFFUSION_LOSS_WEIGHT", 1.0))
    diffusion_kd_temperature = float(os.environ.get("DIFFUSION_KD_TEMPERATURE", 1.0))
    diffusion_span_len = int(os.environ.get("DIFFUSION_SPAN_LEN", 64))
    diffusion_spans_per_seq = int(os.environ.get("DIFFUSION_SPANS_PER_SEQ", 8))
    diffusion_min_mask_ratio = float(os.environ.get("DIFFUSION_MIN_MASK_RATIO", 0.25))
    diffusion_max_mask_ratio = float(os.environ.get("DIFFUSION_MAX_MASK_RATIO", 0.35))
    diffusion_progressive = bool(int(os.environ.get("DIFFUSION_PROGRESSIVE", "1")))
    diffusion_warmstart_ce_only_steps = int(os.environ.get("DIFFUSION_WARMSTART_CE_ONLY_STEPS", 100))
    diffusion_warmstart_ce_only_seconds = float(os.environ.get("DIFFUSION_WARMSTART_CE_ONLY_SECONDS", 0.0))
    diffusion_warmstart_ce_only_fraction = float(os.environ.get("DIFFUSION_WARMSTART_CE_ONLY_FRACTION", 0.0))
    mercury_task_mode = os.environ.get("MERCURY_TASK_MODE", "mixed").strip().lower()
    mercury_continuation_prob = float(os.environ.get("MERCURY_CONTINUATION_PROB", 0.75))
    mercury_clean_loss_weight = float(os.environ.get("MERCURY_CLEAN_LOSS_WEIGHT", 0.10))
    mercury_self_condition_prob = float(os.environ.get("MERCURY_SELF_CONDITION_PROB", 1.0))
    mercury_self_condition_fraction = float(os.environ.get("MERCURY_SELF_CONDITION_FRACTION", 0.75))
    mercury_unroll_steps = int(os.environ.get("MERCURY_UNROLL_STEPS", 1))
    mercury_first_pass_loss_weight = float(os.environ.get("MERCURY_FIRST_PASS_LOSS_WEIGHT", 0.5))
    mercury_second_pass_loss_weight = float(os.environ.get("MERCURY_SECOND_PASS_LOSS_WEIGHT", 1.0))
    mercury_commit_strategy = os.environ.get("MERCURY_COMMIT_STRATEGY", "confidence").strip().lower()
    report_parallel_metrics = bool(int(os.environ.get("REPORT_PARALLEL_METRICS", "1")))
    parallel_eval_examples = int(os.environ.get("PARALLEL_EVAL_EXAMPLES", 16))
    parallel_eval_prefix_len = int(os.environ.get("PARALLEL_EVAL_PREFIX_LEN", 128))
    parallel_eval_suffix_len = int(os.environ.get("PARALLEL_EVAL_SUFFIX_LEN", 64))
    parallel_eval_span_len = int(os.environ.get("PARALLEL_EVAL_SPAN_LEN", 64))
    parallel_eval_steps = os.environ.get("PARALLEL_EVAL_STEPS", "1,2,4,8")

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

TOKENIZER_META_FORMAT_VERSION = 1
TOKENIZER_META_SUFFIX = ".meta.npz"


def _derive_tokenizer_meta_path(tokenizer_path: str) -> Path:
    tokenizer = Path(tokenizer_path)
    if tokenizer.suffix == ".model":
        return tokenizer.with_suffix(TOKENIZER_META_SUFFIX)
    return tokenizer.with_name(tokenizer.name + TOKENIZER_META_SUFFIX)


def _torch_luts_from_numpy(
    base_bytes_np: np.ndarray,
    has_leading_space_np: np.ndarray,
    is_boundary_token_np: np.ndarray,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def build_sentencepiece_luts_np(sp, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return base_bytes_np, has_leading_space_np, is_boundary_token_np


def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return _torch_luts_from_numpy(*build_sentencepiece_luts_np(sp, vocab_size), device=device)


def load_tokenizer_meta_luts_np(meta_path: Path, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(meta_path, allow_pickle=False) as data:
        format_version = int(data["format_version"])
        if format_version != TOKENIZER_META_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported tokenizer metadata format_version={format_version} in {meta_path}; "
                f"expected {TOKENIZER_META_FORMAT_VERSION}"
            )
        meta_vocab_size = int(data["vocab_size"])
        table_size = max(meta_vocab_size, vocab_size)
        base_bytes_np = np.zeros((table_size,), dtype=np.int16)
        has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
        is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
        base_src = np.asarray(data["base_bytes"], dtype=np.int16)
        lead_src = np.asarray(data["has_leading_space"], dtype=np.bool_)
        boundary_src = np.asarray(data["is_boundary_token"], dtype=np.bool_)
        if not (len(base_src) == len(lead_src) == len(boundary_src) == meta_vocab_size):
            raise ValueError(f"Tokenizer metadata arrays in {meta_path} do not match vocab_size={meta_vocab_size}")
        base_bytes_np[:meta_vocab_size] = base_src
        has_leading_space_np[:meta_vocab_size] = lead_src
        is_boundary_token_np[:meta_vocab_size] = boundary_src
        metadata = {
            "format_version": format_version,
            "tokenizer_kind": str(data["tokenizer_kind"].item()),
            "source_model_name": str(data["source_model_name"].item()),
            "vocab_size": meta_vocab_size,
        }
    return base_bytes_np, has_leading_space_np, is_boundary_token_np, metadata


def load_tokenizer_luts(
    *,
    tokenizer_path: str,
    tokenizer_meta_path: str,
    vocab_size: int,
    device: torch.device,
    validate_meta: bool,
) -> tuple[tuple[Tensor, Tensor, Tensor], dict[str, object]]:
    explicit_meta = Path(tokenizer_meta_path).expanduser() if tokenizer_meta_path else None
    meta_path = explicit_meta if explicit_meta is not None else _derive_tokenizer_meta_path(tokenizer_path)
    meta_path = meta_path.resolve()
    tokenizer = Path(tokenizer_path).expanduser().resolve()
    if meta_path.is_file():
        base_bytes_np, has_leading_space_np, is_boundary_token_np, metadata = load_tokenizer_meta_luts_np(
            meta_path, vocab_size
        )
        if validate_meta:
            sp_luts = load_tokenizer_luts_from_sentencepiece_np(tokenizer, vocab_size)
            if not (
                np.array_equal(base_bytes_np, sp_luts[0])
                and np.array_equal(has_leading_space_np, sp_luts[1])
                and np.array_equal(is_boundary_token_np, sp_luts[2])
            ):
                raise ValueError(
                    f"Tokenizer metadata parity check failed for meta_path={meta_path} tokenizer_path={tokenizer}"
                )
        metadata.update(
            {
                "meta_path": str(meta_path),
                "tokenizer_path": str(tokenizer),
                "source": "meta",
            }
        )
        return _torch_luts_from_numpy(base_bytes_np, has_leading_space_np, is_boundary_token_np, device=device), metadata
    if explicit_meta is not None:
        raise FileNotFoundError(f"TOKENIZER_META_PATH does not exist: {meta_path}")
    luts = load_tokenizer_luts_from_sentencepiece_np(tokenizer, vocab_size)
    metadata = {
        "format_version": None,
        "tokenizer_kind": "sentencepiece",
        "source_model_name": tokenizer.name,
        "vocab_size": max(vocab_size, len(luts[0])),
        "meta_path": str(meta_path),
        "tokenizer_path": str(tokenizer),
        "source": "sentencepiece-fallback",
    }
    return _torch_luts_from_numpy(*luts, device=device), metadata


def load_tokenizer_luts_from_sentencepiece_np(
    tokenizer_path: Path, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError(
            f"Tokenizer metadata not found at { _derive_tokenizer_meta_path(str(tokenizer_path)) } and "
            "sentencepiece is unavailable for compatibility fallback"
        ) from exc
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if int(sp.vocab_size()) != vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    return build_sentencepiece_luts_np(sp, vocab_size)


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


def sample_diffusion_target_mask(args: Hyperparameters, batch_size: int, seq_len: int, step: int, device: torch.device) -> Tensor:
    if args.diffusion_span_len <= 0:
        raise ValueError(f"DIFFUSION_SPAN_LEN must be positive, got {args.diffusion_span_len}")
    if args.diffusion_min_mask_ratio <= 0.0 or args.diffusion_max_mask_ratio <= 0.0:
        raise ValueError("Diffusion mask ratios must be positive")
    if args.diffusion_min_mask_ratio > args.diffusion_max_mask_ratio:
        raise ValueError("DIFFUSION_MIN_MASK_RATIO must be <= DIFFUSION_MAX_MASK_RATIO")

    progress = min(step / max(args.iterations, 1), 1.0) if args.diffusion_progressive else 0.0
    target_ratio = (1.0 - progress) * args.diffusion_min_mask_ratio + progress * args.diffusion_max_mask_ratio
    mask_tokens = max(1, min(int(round(seq_len * target_ratio)), seq_len))
    span_len = min(args.diffusion_span_len, seq_len)
    num_spans = max(args.diffusion_spans_per_seq, math.ceil(mask_tokens / span_len))
    mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)

    for batch_idx in range(batch_size):
        tokens_left = mask_tokens
        for _ in range(num_spans):
            if tokens_left <= 0:
                break
            current_span = min(span_len, tokens_left)
            max_start = seq_len - current_span
            start = int(torch.randint(max_start + 1, (), device=device).item())
            mask[batch_idx, start : start + current_span] = True
            tokens_left -= current_span
    return mask


def sample_mercury_target_mask(
    args: Hyperparameters,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> Tensor:
    if not (0.0 <= args.mercury_continuation_prob <= 1.0):
        raise ValueError(f"MERCURY_CONTINUATION_PROB must be in [0, 1], got {args.mercury_continuation_prob}")
    if args.mercury_task_mode not in {"mixed", "continuation", "infill", "random"}:
        raise ValueError(f"Unsupported MERCURY_TASK_MODE={args.mercury_task_mode!r}")
    if args.mercury_task_mode == "random":
        return sample_diffusion_target_mask(args, batch_size, seq_len, step, device)

    progress = min(step / max(args.iterations, 1), 1.0) if args.diffusion_progressive else 0.0
    target_ratio = (1.0 - progress) * args.diffusion_min_mask_ratio + progress * args.diffusion_max_mask_ratio
    mask_tokens = max(1, min(int(round(seq_len * target_ratio)), seq_len - 2))
    mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)

    for batch_idx in range(batch_size):
        if args.mercury_task_mode == "continuation":
            task = "continuation"
        elif args.mercury_task_mode == "infill":
            task = "infill"
        else:
            task = "continuation" if torch.rand((), device=device).item() < args.mercury_continuation_prob else "infill"

        if task == "continuation":
            mask[batch_idx, seq_len - mask_tokens :] = True
            continue

        max_start = seq_len - mask_tokens - 1
        min_start = 1
        start = min_start if max_start <= min_start else int(torch.randint(min_start, max_start + 1, (), device=device).item())
        mask[batch_idx, start : start + mask_tokens] = True
    return mask


def build_diffusion_corruption(
    args: Hyperparameters,
    input_ids: Tensor,
    step: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if args.diffusion_train_mode == "mercury":
        target_mask = sample_mercury_target_mask(args, input_ids.size(0), input_ids.size(1), step, input_ids.device)
    else:
        target_mask = sample_diffusion_target_mask(args, input_ids.size(0), input_ids.size(1), step, input_ids.device)
    mask_token_mask = torch.zeros_like(target_mask)
    corrupted_ids = input_ids.clone()

    if not target_mask.any():
        return corrupted_ids, target_mask, mask_token_mask

    if args.diffusion_corruption_mode == "mask":
        mask_token_mask = target_mask
        return corrupted_ids, target_mask, mask_token_mask

    random_ids = torch.randint(args.vocab_size, input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
    random_ids = torch.where(random_ids == input_ids, (random_ids + 1) % args.vocab_size, random_ids)

    if args.diffusion_corruption_mode == "uniform":
        corrupted_ids = torch.where(target_mask, random_ids, corrupted_ids)
        return corrupted_ids, target_mask, mask_token_mask

    if args.diffusion_corruption_mode == "hybrid":
        if not (0.0 <= args.diffusion_hybrid_mask_prob <= 1.0):
            raise ValueError(
                f"DIFFUSION_HYBRID_MASK_PROB must be in [0, 1], got {args.diffusion_hybrid_mask_prob}"
            )
        choose_mask = torch.rand(input_ids.shape, device=input_ids.device) < args.diffusion_hybrid_mask_prob
        mask_token_mask = target_mask & choose_mask
        uniform_mask = target_mask & ~choose_mask
        corrupted_ids = torch.where(uniform_mask, random_ids, corrupted_ids)
        return corrupted_ids, target_mask, mask_token_mask

    raise ValueError(f"Unsupported DIFFUSION_CORRUPTION_MODE={args.diffusion_corruption_mode!r}")


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    flat_mask = mask.reshape(-1).float()
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        targets.reshape(-1),
        reduction="none",
    )
    denom = flat_mask.sum().clamp_min(1.0)
    return (losses * flat_mask).sum() / denom


def masked_kl_divergence(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
    if temperature <= 0.0:
        raise ValueError(f"DIFFUSION_KD_TEMPERATURE must be positive, got {temperature}")
    flat_mask = mask.reshape(-1).float()
    student = student_logits.reshape(-1, student_logits.size(-1)).float()
    teacher = teacher_logits.reshape(-1, teacher_logits.size(-1)).float().detach()
    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    teacher_probs = F.softmax(teacher / temperature, dim=-1)
    per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    denom = flat_mask.sum().clamp_min(1.0)
    return (per_token * flat_mask).sum() / denom * (temperature * temperature)


def parse_parallel_eval_steps(steps_csv: str) -> list[int]:
    steps: list[int] = []
    for raw in steps_csv.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"PARALLEL_EVAL_STEPS values must be positive, got {value}")
        steps.append(value)
    if not steps:
        raise ValueError("PARALLEL_EVAL_STEPS must contain at least one positive integer")
    return sorted(set(steps))


@torch.no_grad()
def mercury_commit_predictions(
    model: GPT,
    corrupted_ids: Tensor,
    target_mask: Tensor,
    mask_token_mask: Tensor,
    *,
    commit_fraction: float,
    strategy: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if not (0.0 <= commit_fraction <= 1.0):
        raise ValueError(f"commit_fraction must be in [0, 1], got {commit_fraction}")
    if strategy not in {"confidence", "random"}:
        raise ValueError(f"Unsupported MERCURY_COMMIT_STRATEGY={strategy!r}")
    if not target_mask.any() or commit_fraction <= 0.0:
        return corrupted_ids, target_mask, mask_token_mask

    hidden = model.encode_hidden(corrupted_ids, mask_token_mask=mask_token_mask)
    logits = model.token_logits(hidden)
    probs = F.softmax(logits.float(), dim=-1)
    pred_ids = probs.argmax(dim=-1)
    pred_conf = probs.max(dim=-1).values

    refined_ids = corrupted_ids.clone()
    remaining_mask = target_mask.clone()
    remaining_mask_tokens = mask_token_mask.clone()

    for batch_idx in range(corrupted_ids.size(0)):
        active = torch.nonzero(target_mask[batch_idx], as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        if commit_fraction >= 1.0:
            chosen = active
        else:
            k = max(1, min(active.numel(), math.ceil(active.numel() * commit_fraction)))
            if strategy == "confidence":
                conf = pred_conf[batch_idx, active]
                _, order = torch.topk(conf, k=k, largest=True, sorted=False)
                chosen = active[order]
            else:
                perm = torch.randperm(active.numel(), device=active.device)[:k]
                chosen = active[perm]
        refined_ids[batch_idx, chosen] = pred_ids[batch_idx, chosen]
        remaining_mask[batch_idx, chosen] = False
        remaining_mask_tokens[batch_idx, chosen] = False

    return refined_ids, remaining_mask, remaining_mask_tokens

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
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


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
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
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
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
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
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.mask_emb = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))
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
                )
                for i in range(num_layers)
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
        nn.init.normal_(self.mask_emb, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def encode_hidden(self, input_ids: Tensor, mask_token_mask: Tensor | None = None) -> Tensor:
        x = self.tok_emb(input_ids)
        if mask_token_mask is not None:
            x = torch.where(mask_token_mask[..., None], self.mask_emb.to(dtype=x.dtype)[None, None, :], x)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def token_logits(self, hidden: Tensor) -> Tensor:
        x = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.reshape(hidden.size(0), hidden.size(1), -1)

    def forward(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
        diffusion_input_ids: Tensor | None = None,
        diffusion_target_mask: Tensor | None = None,
        diffusion_mask_token_mask: Tensor | None = None,
        diffusion_train_mode: str = "aux",
        diffusion_loss_weight: float = 0.0,
        diffusion_kd_temperature: float = 1.0,
        mercury_clean_loss_weight: float = 0.05,
        return_metrics: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
        if diffusion_train_mode not in {"aux", "block", "consistency", "mercury"}:
            raise ValueError(f"Unsupported DIFFUSION_TRAIN_MODE={diffusion_train_mode!r}")

        hidden = self.encode_hidden(input_ids)
        logits = self.token_logits(hidden)
        clean_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction="mean")

        diffusion_loss = clean_loss.new_zeros(())
        mask_ratio = clean_loss.new_zeros(())
        total_loss = clean_loss
        if diffusion_target_mask is not None and diffusion_loss_weight > 0.0 and diffusion_train_mode == "block":
            mask_present = diffusion_target_mask.float().amax()
            corrupted_ids = diffusion_input_ids if diffusion_input_ids is not None else input_ids
            masked_hidden = self.encode_hidden(corrupted_ids, mask_token_mask=diffusion_mask_token_mask)
            masked_logits = self.token_logits(masked_hidden)
            block_loss = masked_cross_entropy(masked_logits, target_ids, diffusion_target_mask)
            mask_ratio = diffusion_target_mask.float().mean()
            diffusion_loss = block_loss.detach()
            clean_loss = torch.lerp(clean_loss.detach(), block_loss.detach(), mask_present)
            total_loss = torch.lerp(total_loss, block_loss, mask_present)
        elif diffusion_target_mask is not None and diffusion_loss_weight > 0.0 and diffusion_train_mode == "mercury":
            mask_present = diffusion_target_mask.float().amax()
            corrupted_ids = diffusion_input_ids if diffusion_input_ids is not None else input_ids
            masked_hidden = self.encode_hidden(corrupted_ids, mask_token_mask=diffusion_mask_token_mask)
            masked_logits = self.token_logits(masked_hidden)
            diffusion_loss = masked_cross_entropy(masked_logits, input_ids, diffusion_target_mask)
            mask_ratio = diffusion_target_mask.float().mean()
            mercury_total_loss = diffusion_loss_weight * diffusion_loss + mercury_clean_loss_weight * clean_loss
            total_loss = torch.lerp(total_loss, mercury_total_loss, mask_present)
        elif diffusion_target_mask is not None and diffusion_loss_weight > 0.0 and diffusion_train_mode == "consistency":
            corrupted_ids = diffusion_input_ids if diffusion_input_ids is not None else input_ids
            masked_hidden = self.encode_hidden(corrupted_ids, mask_token_mask=diffusion_mask_token_mask)
            masked_logits = self.token_logits(masked_hidden)
            diffusion_loss = masked_kl_divergence(
                masked_logits,
                logits,
                diffusion_target_mask,
                diffusion_kd_temperature,
            )
            mask_ratio = diffusion_target_mask.float().mean()
            total_loss = clean_loss + diffusion_loss_weight * diffusion_loss
        elif diffusion_target_mask is not None and diffusion_loss_weight > 0.0:
            corrupted_ids = diffusion_input_ids if diffusion_input_ids is not None else input_ids
            masked_hidden = self.encode_hidden(corrupted_ids, mask_token_mask=diffusion_mask_token_mask)
            masked_logits = self.token_logits(masked_hidden)
            diffusion_loss = masked_cross_entropy(masked_logits, input_ids, diffusion_target_mask)
            mask_ratio = diffusion_target_mask.float().mean()
            total_loss = clean_loss + diffusion_loss_weight * diffusion_loss

        # Keep mask_emb in the autograd graph even during CE-only warmup/control runs
        # so DDP does not flag it as an unused parameter.
        total_loss = total_loss + self.mask_emb[:1].float().sum() * 0.0

        if return_metrics:
            return total_loss, clean_loss.detach(), diffusion_loss.detach(), mask_ratio.detach()
        return total_loss


def iterative_parallel_refine(
    model: GPT,
    seed_ids: Tensor,
    target_mask: Tensor,
    num_steps: int,
) -> Tensor:
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    refined_ids = seed_ids.clone()
    remaining_mask = target_mask.clone()
    if not remaining_mask.any():
        return refined_ids

    for step_idx in range(num_steps):
        hidden = model.encode_hidden(refined_ids, mask_token_mask=remaining_mask)
        logits = model.token_logits(hidden)
        probs = F.softmax(logits.float(), dim=-1)
        pred_ids = probs.argmax(dim=-1)
        pred_conf = probs.max(dim=-1).values

        steps_left = num_steps - step_idx
        for batch_idx in range(refined_ids.size(0)):
            active = torch.nonzero(remaining_mask[batch_idx], as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            if steps_left == 1:
                chosen = active
            else:
                fill_count = max(1, math.ceil(active.numel() / steps_left))
                conf = pred_conf[batch_idx, active]
                _, order = torch.topk(conf, k=min(fill_count, active.numel()), largest=True, sorted=False)
                chosen = active[order]
            refined_ids[batch_idx, chosen] = pred_ids[batch_idx, chosen]
            remaining_mask[batch_idx, chosen] = False
    return refined_ids


def sample_parallel_eval_batch(
    val_tokens: Tensor,
    *,
    batch_size: int,
    seq_len: int,
    prefix_len: int,
    suffix_len: int,
    span_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if prefix_len + suffix_len > seq_len:
        raise ValueError(
            f"prefix_len + suffix_len must be <= seq_len, got {prefix_len}+{suffix_len}>{seq_len}"
        )
    if prefix_len <= 0 or suffix_len <= 0 or span_len <= 0:
        raise ValueError("Parallel eval lengths must be positive")
    total_seqs = (val_tokens.numel() - 1) // seq_len
    if total_seqs < batch_size:
        raise ValueError(f"Need at least {batch_size} validation sequences, found {total_seqs}")

    seq_ids = torch.randperm(total_seqs)[:batch_size]
    base = []
    for seq_id in seq_ids.tolist():
        start = seq_id * seq_len
        base.append(val_tokens[start : start + seq_len].to(device=device, dtype=torch.int64, non_blocking=True))
    clean = torch.stack(base, dim=0)

    continuation_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
    continuation_mask[:, prefix_len : prefix_len + suffix_len] = True

    infill_mask = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
    max_span_start = seq_len - span_len
    min_span_start = min(prefix_len, max_span_start)
    for batch_idx in range(batch_size):
        span_start = int(torch.randint(min_span_start, max_span_start + 1, (), device=device).item())
        infill_mask[batch_idx, span_start : span_start + span_len] = True

    return clean, continuation_mask, infill_mask


def eval_parallel_metrics(
    args: Hyperparameters,
    model: GPT,
    val_tokens: Tensor,
    device: torch.device,
) -> list[str]:
    batch_size = min(args.parallel_eval_examples, 32)
    steps_list = parse_parallel_eval_steps(args.parallel_eval_steps)
    clean, continuation_mask, infill_mask = sample_parallel_eval_batch(
        val_tokens,
        batch_size=batch_size,
        seq_len=args.train_seq_len,
        prefix_len=args.parallel_eval_prefix_len,
        suffix_len=args.parallel_eval_suffix_len,
        span_len=args.parallel_eval_span_len,
        device=device,
    )
    log_lines: list[str] = []

    model.eval()
    with torch.inference_mode():
        for steps in steps_list:
            for task_name, task_mask in (("continuation", continuation_mask), ("infill", infill_mask)):
                seed_ids = clean.clone()
                seed_ids[task_mask] = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                refined = iterative_parallel_refine(model, seed_ids, task_mask, steps)
                torch.cuda.synchronize()
                elapsed = max(time.perf_counter() - t0, 1e-9)

                masked_tokens = task_mask.sum().item()
                token_acc = (refined[task_mask] == clean[task_mask]).float().mean().item()
                exact_seq = (refined[task_mask] == clean[task_mask]).reshape(batch_size, -1).all(dim=1).float().mean().item()
                tok_per_sec = masked_tokens / elapsed
                log_lines.append(
                    f"parallel_eval task:{task_name} steps:{steps} "
                    f"token_acc:{token_acc:.4f} exact_seq:{exact_seq:.4f} "
                    f"tokens:{masked_tokens} tok_per_sec:{tok_per_sec:.2f}"
                )
    model.train()
    return log_lines


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.use_compile:
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

    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut), tokenizer_meta = load_tokenizer_luts(
        tokenizer_path=args.tokenizer_path,
        tokenizer_meta_path=args.tokenizer_meta_path,
        vocab_size=args.vocab_size,
        device=device,
        validate_meta=args.tokenizer_meta_validate,
    )
    meta_path = tokenizer_meta.get("meta_path")
    log0(
        "val_bpb:enabled "
        f"tokenizer_kind={tokenizer_meta['tokenizer_kind']} "
        f"tokenizer_path={tokenizer_meta['tokenizer_path']} "
        f"tokenizer_source={tokenizer_meta['source']} "
        f"tokenizer_meta_path={meta_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

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
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if args.use_compile else base_model
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
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.mask_emb)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
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
    log0(f"model_params:{n_params}")
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
    log0(
        f"diffusion_enabled:{args.diffusion_enabled} diffusion_train_mode:{args.diffusion_train_mode} "
        f"corruption_mode:{args.diffusion_corruption_mode} hybrid_mask_prob:{args.diffusion_hybrid_mask_prob:.3f} "
        f"diffusion_loss_weight:{args.diffusion_loss_weight} kd_temperature:{args.diffusion_kd_temperature} "
        f"span_len:{args.diffusion_span_len} spans_per_seq:{args.diffusion_spans_per_seq} "
        f"mask_ratio:{args.diffusion_min_mask_ratio:.3f}->{args.diffusion_max_mask_ratio:.3f} "
        f"progressive:{args.diffusion_progressive} warmstart_ce_only_steps:{args.diffusion_warmstart_ce_only_steps} "
        f"warmstart_ce_only_seconds:{args.diffusion_warmstart_ce_only_seconds:.1f} "
        f"warmstart_ce_only_fraction:{args.diffusion_warmstart_ce_only_fraction:.3f} "
        f"use_compile:{args.use_compile}"
    )
    log0(
        f"mercury_task_mode:{args.mercury_task_mode} mercury_continuation_prob:{args.mercury_continuation_prob:.3f} "
        f"mercury_clean_loss_weight:{args.mercury_clean_loss_weight:.3f} "
        f"mercury_self_condition_prob:{args.mercury_self_condition_prob:.3f} "
        f"mercury_self_condition_fraction:{args.mercury_self_condition_fraction:.3f} "
        f"mercury_unroll_steps:{args.mercury_unroll_steps} "
        f"mercury_first_pass_loss_weight:{args.mercury_first_pass_loss_weight:.3f} "
        f"mercury_second_pass_loss_weight:{args.mercury_second_pass_loss_weight:.3f} "
        f"mercury_commit_strategy:{args.mercury_commit_strategy}"
    )
    log0(
        f"report_parallel_metrics:{args.report_parallel_metrics} "
        f"parallel_eval_examples:{args.parallel_eval_examples} "
        f"parallel_eval_prefix_len:{args.parallel_eval_prefix_len} "
        f"parallel_eval_suffix_len:{args.parallel_eval_suffix_len} "
        f"parallel_eval_span_len:{args.parallel_eval_span_len} "
        f"parallel_eval_steps:{args.parallel_eval_steps}"
    )

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

    def diffusion_is_active(step: int, elapsed_ms: float) -> bool:
        if not args.diffusion_enabled:
            return False
        if step < args.diffusion_warmstart_ce_only_steps:
            return False
        if args.diffusion_warmstart_ce_only_seconds > 0.0 and elapsed_ms < 1000.0 * args.diffusion_warmstart_ce_only_seconds:
            return False
        if (
            args.diffusion_warmstart_ce_only_fraction > 0.0
            and max_wallclock_ms is not None
            and elapsed_ms < args.diffusion_warmstart_ce_only_fraction * max_wallclock_ms
        ):
            return False
        return True

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
        train_clean_loss = torch.zeros((), device=device)
        train_diffusion_loss = torch.zeros((), device=device)
        train_mask_ratio = torch.zeros((), device=device)
        diffusion_weight = args.diffusion_loss_weight if diffusion_is_active(step, elapsed_ms) else 0.0
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            diffusion_input_ids = None
            diffusion_target_mask = None
            diffusion_mask_token_mask = None
            if diffusion_weight > 0.0:
                diffusion_input_ids, diffusion_target_mask, diffusion_mask_token_mask = build_diffusion_corruption(
                    args, x, step
                )
                if (
                    args.diffusion_train_mode == "mercury"
                    and args.mercury_unroll_steps <= 1
                    and args.mercury_self_condition_prob > 0.0
                    and diffusion_target_mask is not None
                    and diffusion_target_mask.any()
                    and torch.rand((), device=device).item() < args.mercury_self_condition_prob
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        with torch.no_grad():
                            sc_hidden = base_model.encode_hidden(
                                diffusion_input_ids,
                                mask_token_mask=diffusion_mask_token_mask,
                            )
                            sc_logits = base_model.token_logits(sc_hidden)
                            sc_pred_ids = sc_logits.argmax(dim=-1)
                    replace_mask = diffusion_target_mask & (
                        torch.rand(diffusion_target_mask.shape, device=device) < args.mercury_self_condition_fraction
                    )
                    diffusion_input_ids = torch.where(replace_mask, sc_pred_ids, diffusion_input_ids)
                    diffusion_mask_token_mask = diffusion_mask_token_mask & ~replace_mask
            if (
                diffusion_weight > 0.0
                and args.diffusion_train_mode == "mercury"
                and args.mercury_unroll_steps > 1
                and diffusion_target_mask is not None
                and diffusion_target_mask.any()
            ):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    first_loss_raw, clean_loss, first_diffusion_loss, mask_ratio = model(
                        x,
                        y,
                        diffusion_input_ids=diffusion_input_ids,
                        diffusion_target_mask=diffusion_target_mask,
                        diffusion_mask_token_mask=diffusion_mask_token_mask,
                        diffusion_train_mode=args.diffusion_train_mode,
                        diffusion_loss_weight=diffusion_weight,
                        diffusion_kd_temperature=args.diffusion_kd_temperature,
                        mercury_clean_loss_weight=0.0,
                        return_metrics=True,
                    )
                    clean_prior_loss = model(x, y)
                refined_input_ids, remaining_target_mask, refined_mask_token_mask = mercury_commit_predictions(
                    base_model,
                    diffusion_input_ids,
                    diffusion_target_mask,
                    diffusion_mask_token_mask,
                    commit_fraction=args.mercury_self_condition_fraction,
                    strategy=args.mercury_commit_strategy,
                )
                if remaining_target_mask.any():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        second_loss_raw, _, second_diffusion_loss, second_mask_ratio = model(
                            x,
                            y,
                            diffusion_input_ids=refined_input_ids,
                            diffusion_target_mask=remaining_target_mask,
                            diffusion_mask_token_mask=refined_mask_token_mask,
                            diffusion_train_mode=args.diffusion_train_mode,
                            diffusion_loss_weight=diffusion_weight,
                            diffusion_kd_temperature=args.diffusion_kd_temperature,
                            mercury_clean_loss_weight=0.0,
                            return_metrics=True,
                        )
                    total_weight = (
                        args.mercury_first_pass_loss_weight
                        + args.mercury_second_pass_loss_weight
                        + args.mercury_clean_loss_weight
                    )
                    loss = (
                        args.mercury_first_pass_loss_weight * first_loss_raw
                        + args.mercury_second_pass_loss_weight * second_loss_raw
                        + args.mercury_clean_loss_weight * clean_prior_loss
                    ) / max(total_weight, 1e-9)
                    diffusion_loss = (
                        args.mercury_first_pass_loss_weight * first_diffusion_loss
                        + args.mercury_second_pass_loss_weight * second_diffusion_loss
                    ) / max(args.mercury_first_pass_loss_weight + args.mercury_second_pass_loss_weight, 1e-9)
                    mask_ratio = second_mask_ratio
                else:
                    total_weight = args.mercury_first_pass_loss_weight + args.mercury_clean_loss_weight
                    loss = (
                        args.mercury_first_pass_loss_weight * first_loss_raw
                        + args.mercury_clean_loss_weight * clean_prior_loss
                    ) / max(total_weight, 1e-9)
                    diffusion_loss = first_diffusion_loss
                clean_loss = clean_prior_loss.detach()
            else:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss, clean_loss, diffusion_loss, mask_ratio = model(
                        x,
                        y,
                        diffusion_input_ids=diffusion_input_ids,
                        diffusion_target_mask=diffusion_target_mask,
                        diffusion_mask_token_mask=diffusion_mask_token_mask,
                        diffusion_train_mode=args.diffusion_train_mode,
                        diffusion_loss_weight=diffusion_weight,
                        diffusion_kd_temperature=args.diffusion_kd_temperature,
                        mercury_clean_loss_weight=args.mercury_clean_loss_weight,
                        return_metrics=True,
                    )
            train_loss += loss.detach()
            train_clean_loss += clean_loss
            train_diffusion_loss += diffusion_loss
            train_mask_ratio += mask_ratio
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_clean_loss /= grad_accum_steps
        train_diffusion_loss /= grad_accum_steps
        train_mask_ratio /= grad_accum_steps

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
                f"clean_loss:{train_clean_loss.item():.4f} diffusion_loss:{train_diffusion_loss.item():.4f} "
                f"mask_ratio:{train_mask_ratio.item():.4f} "
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
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
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
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if args.report_parallel_metrics:
        if distributed:
            dist.barrier()
        if master_process:
            for line in eval_parallel_metrics(args, base_model, val_tokens, device):
                log0(line)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
