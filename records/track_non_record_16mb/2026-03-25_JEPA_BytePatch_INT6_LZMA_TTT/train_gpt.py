"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import json
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
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
# Default JEPA run:
# - pure-byte FineWeb export (`byte260`)
# - byte-patch JEPA with latent next-patch prediction plus a small causal byte decoder
# - 5 encoder layers × 2 depth-recurrent repeats at width 480, 6 heads with 3 KV heads
# - 7-layer causal decoder at width 480, 4 heads
# - sequence length 2047 so the reconstructed AR stream has 2048 bytes, cleanly divisible by patch size 8
# - 524,032 train tokens per step for 20,000 iterations with a ~10 minute cap


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_byte260")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_pure_byte_260.json"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_160))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    val_sliding_stride = int(os.environ.get("VAL_SLIDING_STRIDE", 256))
    val_sliding_batch = int(os.environ.get("VAL_SLIDING_BATCH", 32))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 2))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_032))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2047))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    use_compile = bool(int(os.environ.get("USE_COMPILE", "1")))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 260))
    num_layers = int(os.environ.get("NUM_LAYERS", 5))
    encoder_repeats = int(os.environ.get("ENCODER_REPEATS", 2))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 3))
    model_dim = int(os.environ.get("MODEL_DIM", 480))
    num_heads = int(os.environ.get("NUM_HEADS", 6))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    patch_size = int(os.environ.get("PATCH_SIZE", 8))
    latent_dim = int(os.environ.get("LATENT_DIM", 192))
    decoder_layers = int(os.environ.get("DECODER_LAYERS", 7))
    decoder_heads = int(os.environ.get("DECODER_HEADS", 4))
    sigreg_weight = float(os.environ.get("SIGREG_WEIGHT", 0.02))
    sigreg_knots = int(os.environ.get("SIGREG_KNOTS", 17))
    sigreg_num_proj = int(os.environ.get("SIGREG_NUM_PROJ", 256))
    jepa_pred_weight = float(os.environ.get("JEPA_PRED_WEIGHT", 0.5))
    jepa_ce_weight = float(os.environ.get("JEPA_CE_WEIGHT", 3.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.1))
    head_lr = float(os.environ.get("HEAD_LR", 0.02))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
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


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
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
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
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
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

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
# We score BPB (bits-per-byte), but the model is fixed to a pure-byte vocabulary:
# 4 special ids followed by raw UTF-8 bytes. That makes byte accounting exact.
def build_pure_byte_luts(
    vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    table_size = max(vocab_size, 260)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    base_bytes_np[4 : min(table_size, 260)] = 1
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    return (
        torch.tensor(base_bytes_np[:vocab_size], dtype=torch.int16, device=device),
        torch.tensor(
            has_leading_space_np[:vocab_size], dtype=torch.bool, device=device
        ),
        torch.tensor(
            is_boundary_token_np[:vocab_size], dtype=torch.bool, device=device
        ),
    )


def load_pure_byte_luts(
    tokenizer_path: str, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    path = Path(tokenizer_path)
    if path.suffix != ".json":
        raise ValueError(
            f"Pure-byte JEPA expects a tokenizer JSON at {tokenizer_path!r}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_type = payload.get("tokenizer_type") or payload.get("kind")
    json_vocab_size = int(payload.get("vocab_size", vocab_size))
    if tokenizer_type != "pure_byte":
        raise ValueError(
            f"Unsupported tokenizer JSON {tokenizer_path}: expected pure_byte, got {tokenizer_type!r}"
        )
    if json_vocab_size != vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={json_vocab_size}"
        )
    return build_pure_byte_luts(vocab_size, device)


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
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                _, batch_loss = model(x, y)
                batch_loss = batch_loss.detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
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


def eval_val_sliding_ttt(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    log_fn=print,
) -> tuple[float, float]:
    seq_len = args.train_seq_len
    stride = args.val_sliding_stride
    batch_seqs = args.ttt_batch_seqs
    total_tokens = val_tokens.numel() - 1
    ttt_chunk = args.ttt_chunk_tokens

    window_starts = [
        ws
        for ws in range(0, total_tokens, stride)
        if min(ws + seq_len, total_tokens) - ws >= seq_len
    ]
    num_chunks = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s = 0 if ws == 0 else max(wlen - stride, 0)
        scored_start = ws + s
        ci = min(scored_start // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    log_fn(
        f"ttt:start chunks={num_chunks} chunk_tokens={ttt_chunk} "
        f"windows={len(window_starts)} stride={stride} "
        f"lr={args.ttt_lr} epochs={args.ttt_epochs} freeze={args.ttt_freeze_blocks}"
    )

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    frozen_block_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        freeze = any(f"blocks.{bi}." in name for bi in frozen_block_ids)
        if freeze:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)
            ttt_params.append(p)

    optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)
    t0 = time.perf_counter()
    t_infer_total = 0.0
    t_adapt_total = 0.0

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue

        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        torch.cuda.synchronize()
        t_infer = time.perf_counter()
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi : bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                for i, ws in enumerate(batch_ws):
                    chunk_tok = val_tokens[ws : ws + seq_len + 1].to(
                        dtype=torch.int64, device=device
                    )
                    x_batch[i] = chunk_tok[:-1]
                    y_batch[i] = chunk_tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch, y_batch)
                nll = F.cross_entropy(
                    logits[:, 1:].reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    s = 0 if ws == 0 else seq_len - stride
                    scored = nll[i, s:seq_len].to(torch.float64)
                    loss_sum += scored.sum()
                    token_count += float(seq_len - s)
                    tgt = y_batch[i, s:seq_len]
                    prev = x_batch[i, s:seq_len]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (
                        has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
                    ).to(torch.float64)
                    byte_count += tb.sum()

        torch.cuda.synchronize()
        t_infer_total += time.perf_counter() - t_infer

        is_last_chunk = ci == num_chunks - 1
        t_adapt = time.perf_counter()
        if not is_last_chunk and args.ttt_epochs > 0:
            base_model.train()
            chunk_start = ci * ttt_chunk
            chunk_end = min((ci + 1) * ttt_chunk, total_tokens)
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                cos_lr = (
                    args.ttt_lr
                    * 0.5
                    * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                for _ep in range(args.ttt_epochs):
                    for bs in range(my_seq_s, my_seq_e, args.ttt_batch_seqs):
                        be = min(bs + args.ttt_batch_seqs, my_seq_e)
                        start_tok = chunk_start + bs * seq_len
                        end_tok = chunk_start + be * seq_len + 1
                        if end_tok > val_tokens.numel():
                            continue
                        local = val_tokens[start_tok:end_tok].to(
                            device=device, dtype=torch.int64
                        )
                        x = local[:-1].reshape(-1, seq_len)
                        y = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = base_model.forward_logits(x, y)
                        ce = F.cross_entropy(
                            logits[:, 1:].reshape(-1, logits.size(-1)).float(),
                            y.reshape(-1),
                        )
                        ce.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        torch.cuda.synchronize()
        t_adapt_total += time.perf_counter() - t_adapt

        if rank == 0 and (ci % 50 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl = loss_sum.item() / max(token_count.item(), 1)
            rbpb = (
                rl / math.log(2.0) * (token_count.item() / max(byte_count.item(), 1))
                if token_count.item() > 0
                else 0.0
            )
            log_fn(
                f"  ttt_chunk [{ci + 1}/{num_chunks}] bpb={rbpb:.6f} "
                f"time={elapsed:.1f}s infer={t_infer_total:.1f}s adapt={t_adapt_total:.1f}s"
            )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    val_bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    return val_loss, val_bpb


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
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


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
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


# ------------------------------------
# INT6 OPTIMAL-CLIP QUANTIZATION
# ------------------------------------


def _classify_param_jepa(name: str) -> str:
    if "tok_emb" in name or "decoder_token_emb" in name or "decoder_out" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name:
        return "attn"
    return "other"


def quantize_int6_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
            if pct < 1.0:
                row_clip = torch.quantile(t32.abs(), pct, dim=1)
            else:
                row_clip = t32.abs().amax(dim=1)
            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(
                torch.round(t32 / s.float()[:, None]), -clip_range, clip_range
            ).to(torch.int8)
            recon = q.float() * s.float()[:, None]
            err = (t32 - recon).pow(2).mean().item()
            if err < best_err:
                best_q, best_s, best_err = q, s, err
        return best_q, best_s
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(
        torch.int8
    )
    return q, scale


def mixed_quantize_int6(
    state_dict: dict[str, Tensor], int6_cats: set[str]
) -> tuple[dict[str, Tensor], dict[str, object]]:
    result: dict[str, Tensor] = {}
    meta: dict[str, object] = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param_jepa(name)
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
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


def dequantize_mixed_int6(
    result: dict[str, Tensor],
    meta: dict[str, object],
    template_sd: dict[str, Tensor],
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if info in ("passthrough", "passthrough_ctrl"):
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (
                torch.float32,
                torch.bfloat16,
            ):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (
                q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
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
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
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

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


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
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (
                    torch.clamp(torch.round(w32 / scale[:, None]), -31, 31)
                    * scale[:, None]
                ).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
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

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
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
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
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
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        return x


class SIGReg(nn.Module):
    # Sketch regularizer from LeWM, adapted to local (per-rank) batches.
    def __init__(self, knots: int = 17, num_proj: int = 256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / max(knots - 1, 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        if knots > 1:
            weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, proj: Tensor) -> Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects (T, B, D), got {tuple(proj.shape)}")
        A = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype
        )
        A = A / (A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-6))
        x_t = (proj @ A).unsqueeze(-1) * self.t.to(dtype=proj.dtype)
        err = (
            x_t.cos().mean(-3) - self.phi.to(dtype=proj.dtype)
        ).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights.to(dtype=proj.dtype)) * proj.size(-2)
        return statistic.mean().float()


class LatentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = hidden_mult * input_dim
        self.norm = RMSNorm()
        self.fc = CastedLinear(input_dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, output_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = F.silu(self.fc(x))
        return self.proj(x)


class BytePatchJEPA(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_layers: int,
        encoder_repeats: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        patch_size: int,
        latent_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        sigreg_knots: int,
        sigreg_num_proj: int,
        sigreg_weight: float,
        jepa_pred_weight: float,
        jepa_ce_weight: float,
    ):
        super().__init__()
        if patch_size < 2:
            raise ValueError(f"PATCH_SIZE must be >=2, got {patch_size}")
        if decoder_heads <= 0 or model_dim % decoder_heads != 0:
            raise ValueError(
                f"DECODER_HEADS={decoder_heads} must divide MODEL_DIM={model_dim}"
            )
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.encoder_repeats = encoder_repeats
        self.sigreg_weight = sigreg_weight
        self.jepa_pred_weight = jepa_pred_weight
        self.jepa_ce_weight = jepa_ce_weight
        self.bos_id = 1

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.patch_pos = nn.Parameter(
            torch.zeros(patch_size, model_dim, dtype=torch.float32)
        )
        self.patch_in = CastedLinear(patch_size * model_dim, model_dim, bias=False)
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
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.projector = LatentMLP(model_dim, latent_dim)
        self.predictor = LatentMLP(model_dim, latent_dim)
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)

        self.start_latent = nn.Parameter(torch.zeros(latent_dim, dtype=torch.float32))
        self.decoder_token_emb = nn.Embedding(vocab_size, model_dim)
        self.decoder_cond = CastedLinear(latent_dim, model_dim, bias=False)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    decoder_heads,
                    decoder_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm()
        self.decoder_out = CastedLinear(model_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder_token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.patch_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.start_latent, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _build_full_sequence(
        self, input_ids: Tensor, target_ids: Tensor | None
    ) -> Tensor:
        if target_ids is None:
            raise ValueError(
                "BytePatchJEPA requires target_ids so it can reconstruct the full autoregressive stream"
            )
        if input_ids.shape != target_ids.shape:
            raise ValueError(
                f"input_ids and target_ids must match, got {tuple(input_ids.shape)} vs {tuple(target_ids.shape)}"
            )
        full = torch.cat((input_ids[:, :1], target_ids), dim=1)
        if full.size(1) % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {full.size(1)} must be divisible by PATCH_SIZE={self.patch_size}; "
                "set TRAIN_SEQ_LEN so TRAIN_SEQ_LEN+1 is divisible by PATCH_SIZE"
            )
        return full

    def _patchify(self, full_ids: Tensor) -> Tensor:
        bsz, seqlen = full_ids.shape
        num_patches = seqlen // self.patch_size
        return full_ids.view(bsz, num_patches, self.patch_size)

    def _encode_patches(self, patches: Tensor) -> Tensor:
        x = self.tok_emb(patches)
        x = x + self.patch_pos.to(dtype=x.dtype)[None, None, :, :]
        x = F.rms_norm(x, (x.size(-1),))
        return self.patch_in(x.reshape(x.size(0), x.size(1), -1))

    def _contextualize(self, patch_emb: Tensor) -> Tensor:
        x = F.rms_norm(patch_emb, (patch_emb.size(-1),))
        x0 = x
        for _ in range(self.encoder_repeats):
            for block in self.blocks:
                x = block(x, x0)
        return self.final_norm(x)

    def _decode_logits(self, cond_latent: Tensor, target_patches: Tensor) -> Tensor:
        bsz, num_patches, patch_size = target_patches.shape
        total_bytes = num_patches * patch_size
        flat_bytes = target_patches.reshape(bsz, total_bytes)
        prev = torch.cat(
            [flat_bytes.new_full((bsz, 1), self.bos_id), flat_bytes[:, :-1]], dim=1
        )
        x = self.decoder_token_emb(prev)
        cond = self.decoder_cond(cond_latent).to(dtype=x.dtype)
        x = x + cond.repeat_interleave(patch_size, dim=1)
        x0 = x
        for block in self.decoder_blocks:
            x = block(x, x0)
        x = self.decoder_norm(x)
        return self.decoder_out(x).reshape(
            bsz, num_patches, patch_size, self.vocab_size
        )

    def forward(
        self, input_ids: Tensor, target_ids: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        full = self._build_full_sequence(input_ids, target_ids)
        patches = self._patchify(full)
        patch_emb = self._encode_patches(patches)
        target_latent = self.projector(patch_emb)
        context = self._contextualize(patch_emb)
        pred_latent = self.predictor(context[:, :-1])
        pred_loss = F.mse_loss(
            pred_latent.float(), target_latent[:, 1:].detach().float(), reduction="mean"
        )
        sigreg_loss = self.sigreg(target_latent.transpose(0, 1))

        start = self.start_latent.to(dtype=pred_latent.dtype)[None, None, :].expand(
            patches.size(0), 1, -1
        )
        cond_latent = torch.cat((start, pred_latent), dim=1)
        logits = self._decode_logits(cond_latent, patches)
        ce = F.cross_entropy(
            logits.reshape(-1, self.vocab_size).float(),
            patches.reshape(-1),
            reduction="none",
        )
        ce = ce.reshape_as(patches).float()
        mask = torch.ones_like(ce)
        mask[:, 0, 0] = 0.0
        nll = (ce * mask).sum() / mask.sum()
        total = (
            self.jepa_ce_weight * nll
            + self.jepa_pred_weight * pred_loss
            + self.sigreg_weight * sigreg_loss
        )
        return total, nll

    def forward_logits(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        full = self._build_full_sequence(input_ids, target_ids)
        patches = self._patchify(full)
        patch_emb = self._encode_patches(patches)
        context = self._contextualize(patch_emb)
        pred_latent = self.predictor(context[:, :-1])
        start = self.start_latent.to(dtype=pred_latent.dtype)[None, None, :].expand(
            patches.size(0), 1, -1
        )
        cond_latent = torch.cat((start, pred_latent), dim=1)
        logits = self._decode_logits(cond_latent, patches)
        bsz = logits.size(0)
        return logits.reshape(bsz, -1, self.vocab_size)


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
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
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
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

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
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
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

    if (args.train_seq_len + 1) % args.patch_size != 0:
        raise ValueError(
            f"JEPA requires TRAIN_SEQ_LEN+1 to be divisible by PATCH_SIZE; "
            f"got TRAIN_SEQ_LEN={args.train_seq_len}, PATCH_SIZE={args.patch_size}"
        )
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_pure_byte_luts(
        args.tokenizer_path, args.vocab_size, device
    )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    log0(f"val_bpb:enabled tokenizer_kind=byte tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model: nn.Module = BytePatchJEPA(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        encoder_repeats=args.encoder_repeats,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sigreg_weight=args.sigreg_weight,
        jepa_pred_weight=args.jepa_pred_weight,
        jepa_ce_weight=args.jepa_ce_weight,
    )
    base_model = base_model.to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=False)
        if args.use_compile
        else base_model
    )
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    embedding_tags = ("tok_emb", "decoder_token_emb", "patch_pos")
    head_names = {"decoder_out.weight"}
    embedding_params: list[Tensor] = []
    head_params: list[Tensor] = []
    matrix_params: list[Tensor] = []
    scalar_params: list[Tensor] = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name in head_names:
            head_params.append(param)
        elif any(tag in name for tag in embedding_tags):
            embedding_params.append(param)
        elif param.ndim == 2 and not any(
            pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS
        ):
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    token_lr = args.embed_lr
    optimizers: list[torch.optim.Optimizer] = []
    if embedding_params:
        optimizer_embed = torch.optim.Adam(
            [{"params": embedding_params, "lr": token_lr, "base_lr": token_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_embed)
    optimizer_muon = None
    if matrix_params:
        optimizer_muon = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizers.append(optimizer_muon)
    if head_params:
        optimizer_head = torch.optim.Adam(
            [{"params": head_params, "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_head)
    if scalar_params:
        optimizer_scalar = torch.optim.Adam(
            [
                {
                    "params": scalar_params,
                    "lr": args.scalar_lr,
                    "base_lr": args.scalar_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_scalar)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"model_family:jepa attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    log0(
        f"embed_lr:{token_lr} head_lr:{args.head_lr if head_params else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"jepa:patch_size:{args.patch_size} latent_dim:{args.latent_dim} "
        f"encoder_layers:{args.num_layers}x{args.encoder_repeats} "
        f"decoder_layers:{args.decoder_layers} decoder_heads:{args.decoder_heads} "
        f"sigreg_weight:{args.sigreg_weight} pred_weight:{args.jepa_pred_weight} ce_weight:{args.jepa_ce_weight}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_loss, _ = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    ema_state = {
        name: t.detach().float().clone() for name, t in base_model.state_dict().items()
    }
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        should_validate = last_step or (
            args.val_loss_every > 0 and step % args.val_loss_every == 0
        )
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
        if (
            args.late_qat_threshold > 0
            and scale < args.late_qat_threshold
            and not CastedLinear._qat_enabled
        ):
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        train_nll = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, nll = model(x, y)
            train_loss += loss.detach()
            train_nll += nll.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        train_nll /= grad_accum_steps

        if optimizer_muon is not None:
            frac = (
                min(step / args.muon_momentum_warmup_steps, 1.0)
                if args.muon_momentum_warmup_steps > 0
                else 1.0
            )
            muon_momentum = (
                1 - frac
            ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
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

        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(args.ema_decay).add_(
                    t.detach().float(), alpha=1.0 - args.ema_decay
                )
        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms "
                f"train_nll:{train_nll.item():.4f}"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
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

    current_sd = base_model.state_dict()
    ema_avg = {
        name: t.to(dtype=current_sd[name].dtype) for name, t in ema_state.items()
    }
    base_model.load_state_dict(ema_avg, strict=True)
    log0(f"ema:applied EMA weights (decay={args.ema_decay})")

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

    template_sd = {k: v.cpu() for k, v in base_model.state_dict().items()}
    int6_cats = {"mlp", "attn", "other", "embed"}
    quant_result, quant_meta = mixed_quantize_int6(base_model.state_dict(), int6_cats)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = lzma.compress(quant_raw, preset=9)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+lzma: {quant_file_bytes} bytes")
        log0(f"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(lzma.decompress(quant_blob_disk)), map_location="cpu"
    )
    CastedLinear._qat_enabled = False
    deq_sd = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)
    base_model.load_state_dict(deq_sd, strict=True)

    if args.ttt_enabled:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_loss, ttt_bpb = eval_val_sliding_ttt(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log0,
        )
        torch.cuda.synchronize()
        log0(
            f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
