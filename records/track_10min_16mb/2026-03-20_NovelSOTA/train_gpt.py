"""
11L int5/int6 + XSA + online TTT w/ decay prior.
Built on #198, #180, #162, #164, #265, #254.
torchrun --standalone --nproc_per_node=8 train_gpt.py
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# zstandard is required - fail fast if missing rather than silently falling back
# to zlib (which produces a different, larger artifact that may exceed 16MB).
import zstandard as _zstd

_ZSTD_COMPRESSOR = _zstd.ZstdCompressor(level=22)
_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()

def compress_blob(data: bytes) -> bytes:
    return _ZSTD_COMPRESSOR.compress(data)

def decompress_blob(data: bytes) -> bytes:
    return _ZSTD_DECOMPRESSOR.decompress(data)

COMPRESS_EXT = "int6.zst"

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

class Hyp:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # SWA
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 200))
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))

    # SmearGate + BigramHash
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    # XSA (Exclusive Self Attention)
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 3))

    # Extra RMSNorm before every linear projection is experimental and off by default.
    extra_linear_rmsnorm = bool(int(os.environ.get("EXTRA_LINEAR_RMSNORM", "0")))

    # Sliding window eval
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Safety: checkpoint interval
    ckpt_every = int(os.environ.get("CKPT_EVERY", 2000))

    # Reptile Meta-TTT
    meta_ttt_enabled = bool(int(os.environ.get("META_TTT_ENABLED", "1")))
    meta_ttt_start_frac = float(os.environ.get("META_TTT_START_FRAC", 0.90))
    meta_ttt_inner_steps = int(os.environ.get("META_TTT_INNER_STEPS", 1))
    meta_ttt_inner_lr = float(os.environ.get("META_TTT_INNER_LR", 2e-3))
    meta_ttt_epsilon = float(os.environ.get("META_TTT_EPSILON", 0.3))
    meta_ttt_log_every = int(os.environ.get("META_TTT_LOG_EVERY", 50))

    # Online Causal TTT Eval
    ttt_eval_lr = float(os.environ.get("TTT_EVAL_LR", 2e-3))
    ttt_eval_momentum = float(os.environ.get("TTT_EVAL_MOMENTUM", 0.9))
    ttt_eval_grad_clip = float(os.environ.get("TTT_EVAL_GRAD_CLIP", 1.0))
    ttt_eval_decay = float(os.environ.get("TTT_EVAL_DECAY", 0.02))
    ttt_eval_adapt_last_n = int(os.environ.get("TTT_EVAL_ADAPT_LAST_N", 3))


CONTROL_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain",
    "skip_weight", "skip_weights", "smear", "bigram_scale",
)

# ─────────────────────────────────────────────
# MUON OPTIMIZER WITH WEIGHT DECAY
# ─────────────────────────────────────────────

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
                 weight_decay: float = 0.0, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
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
            wd = group["weight_decay"]

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
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ─────────────────────────────────────────────
# TOKENIZER + BPB UTILS
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
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
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def _reduce_bpb(val_loss_sum, val_token_count, val_byte_count):
    """All-reduce across ranks and compute (val_loss, bits_per_byte)."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = (val_loss_sum / val_token_count).item()
    bpb = (val_loss / math.log(2.0)) * (val_token_count.item() / val_byte_count.item())
    return val_loss, bpb


# ─────────────────────────────────────────────
# SLIDING WINDOW EVAL (stride=64)
# ─────────────────────────────────────────────

def eval_val_sliding(
    base_model, val_tokens, eval_seq_len, eval_stride, device,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    rank, world_size, batch_size=32,
):
    """Sliding window eval with batching for GPU efficiency.

    Windows are padded to eval_seq_len and processed in batches of `batch_size`,
    giving ~30x speedup over single-window iteration. With ~10M val tokens,
    stride=64, 8 GPUs: ~19.5K windows/rank / 32 per batch = ~610 forward
    passes, well within the 10-minute eval budget.
    """
    total_tokens = val_tokens.numel()
    window_starts = list(range(0, total_tokens - 1, eval_stride))
    my_starts = [ws for i, ws in enumerate(window_starts) if i % world_size == rank]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    sc = base_model.logit_softcap
    base_model.eval()
    with torch.inference_mode():
        for batch_idx in range(0, len(my_starts), batch_size):
            batch_ws_list = my_starts[batch_idx : batch_idx + batch_size]

            # Pre-filter valid windows and collect metadata
            x_list = []
            y_list = []
            score_ranges = []
            for ws in batch_ws_list:
                wend = min(ws + eval_seq_len + 1, total_tokens)
                wlen = wend - ws - 1
                if wlen < 1:
                    continue
                score_start = 0 if ws == 0 else max(eval_seq_len - eval_stride, 0)
                if score_start >= wlen:
                    continue
                chunk = val_tokens[ws : wend].to(dtype=torch.int64)
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
                score_ranges.append((score_start, wlen))

            if not x_list:
                continue

            # Pad to uniform length and stack into a batch tensor
            max_len = max(t.numel() for t in x_list)
            B = len(x_list)
            x_batch = torch.zeros(B, max_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(B, max_len, dtype=torch.int64, device=device)
            for i, (xi, yi) in enumerate(zip(x_list, y_list)):
                L = xi.numel()
                x_batch[i, :L] = xi
                y_batch[i, :L] = yi

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                raw_logits = base_model.forward_logits(x_batch)

            logits = sc * torch.tanh(raw_logits / sc)

            for i, (score_start, wlen) in enumerate(score_ranges):
                window_logits = logits[i, :wlen].float()
                window_y = y_batch[i, :wlen]
                nll = F.cross_entropy(window_logits, window_y, reduction="none")

                scored_nll = nll[score_start:wlen]
                scored_x = x_batch[i, score_start:wlen]
                scored_y = y_batch[i, score_start:wlen]

                val_loss_sum += scored_nll.to(torch.float64).sum()
                val_token_count += float(scored_nll.numel())

                token_bytes = base_bytes_lut[scored_y].to(torch.int16)
                token_bytes += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).to(torch.int16)
                val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss, bpb = _reduce_bpb(val_loss_sum, val_token_count, val_byte_count)
    base_model.train()
    return val_loss, bpb


def eval_val_simple(args, model, rank, world_size, device, grad_accum_steps,
                    val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    """Non-sliding eval for mid-training validation (faster)."""
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
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

    val_loss, bpb = _reduce_bpb(val_loss_sum, val_token_count, val_byte_count)
    model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────
# ONLINE CAUSAL TTT EVAL WITH DECAY PRIOR
# ─────────────────────────────────────────────

def eval_val_ttt(
    base_model, val_tokens, eval_seq_len, eval_stride, device,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    rank, world_size, args, log_fn=None,
):
    """Online Causal TTT Eval: interleave scoring and adaptation with decay prior.

    TTT requires sequential window processing (each backward adapts the model
    for subsequent windows), so batching is not possible. Instead, we use a
    coarser stride for the adaptation+scoring loop to keep wall-clock time
    within budget:

      ttt_stride = max(eval_stride, eval_seq_len // 4)

    With eval_seq_len=2048 this gives ttt_stride=512, producing ~19.5K total
    windows (~2.4K per rank with 8 GPUs). At ~5ms per forward+backward,
    that's ~12 seconds -- well within the 10-minute eval cap.

    For each sliding window:
      1. Forward pass -> score tokens
      2. Backward pass -> SGD update on MLP params in last N blocks
      3. Apply decay prior: p.data += decay * (theta_original - p.data)
    """
    total_tokens = val_tokens.numel()
    # Use coarser stride for TTT to stay within wall-clock budget.
    # Standard sliding-window with stride=64 would create ~156K windows,
    # each needing forward+backward -- far too slow for a 10-minute cap.
    ttt_stride = max(eval_stride, eval_seq_len // 4)
    window_starts = list(range(0, total_tokens - 1, ttt_stride))
    my_starts = [ws for i, ws in enumerate(window_starts) if i % world_size == rank]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Identify MLP params in last N blocks to adapt
    adapt_last_n = args.ttt_eval_adapt_last_n
    num_blocks = len(base_model.blocks)
    adapt_block_indices = list(range(max(0, num_blocks - adapt_last_n), num_blocks))
    adapt_params = []
    for bi in adapt_block_indices:
        block = base_model.blocks[bi]
        for name, p in block.mlp.named_parameters():
            if "weight" in name and ("fc" in name or "proj" in name):
                adapt_params.append(p)

    # Save original params for decay prior
    theta_original = {id(p): p.data.clone() for p in adapt_params}

    # Set up SGD optimizer for adaptation
    ttt_optimizer = torch.optim.SGD(
        adapt_params, lr=args.ttt_eval_lr, momentum=args.ttt_eval_momentum,
    )

    decay = args.ttt_eval_decay
    sc = base_model.logit_softcap

    base_model.eval()
    # We need gradients for TTT adaptation, so no inference_mode
    for p in base_model.parameters():
        p.requires_grad_(False)
    for p in adapt_params:
        p.requires_grad_(True)

    total_my_windows = len(my_starts)
    for window_idx, ws in enumerate(my_starts):
        wend = min(ws + eval_seq_len + 1, total_tokens)
        wlen = wend - ws - 1
        if wlen < 1:
            continue
        chunk = val_tokens[ws : wend].to(device=device, dtype=torch.int64)
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            raw_logits = base_model.forward_logits(x)

        logits = sc * torch.tanh(raw_logits.squeeze(0) / sc)
        nll = F.cross_entropy(logits.float(), y, reduction="none")

        score_start = 0 if ws == 0 else max(eval_seq_len - ttt_stride, 0)
        if score_start >= wlen:
            continue
        scored_nll = nll[score_start:wlen]
        scored_x = x.squeeze(0)[score_start:wlen]
        scored_y = y[score_start:wlen]

        val_loss_sum += scored_nll.detach().to(torch.float64).sum()
        val_token_count += float(scored_nll.numel())

        token_bytes = base_bytes_lut[scored_y].to(torch.int16)
        token_bytes += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).to(torch.int16)
        val_byte_count += token_bytes.to(torch.float64).sum()

        # Backward pass for adaptation (use mean NLL over full window)
        adapt_loss = nll.mean()
        ttt_optimizer.zero_grad()
        adapt_loss.backward()

        if args.ttt_eval_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(adapt_params, args.ttt_eval_grad_clip)

        ttt_optimizer.step()

        if decay > 0:
            with torch.no_grad():
                for p in adapt_params:
                    p.data.add_(theta_original[id(p)] - p.data, alpha=decay)

        if log_fn is not None and total_my_windows > 0 and (
            (window_idx + 1) % 50 == 0 or (window_idx + 1) == total_my_windows
        ):
            log_fn(
                f"ttt_eval:progress windows:{window_idx + 1}/{total_my_windows} "
                f"rank:{rank} partial_bpb:{(val_loss_sum.item() / max(val_token_count.item(), 1.0)) / math.log(2.0) * (val_token_count.item() / max(val_byte_count.item(), 1.0)):.4f}"
            )

    # Restore original params after TTT eval
    with torch.no_grad():
        for p in adapt_params:
            p.data.copy_(theta_original[id(p)])

    # Re-enable gradients for all params
    for p in base_model.parameters():
        p.requires_grad_(True)

    val_loss, bpb = _reduce_bpb(val_loss_sum, val_token_count, val_byte_count)
    base_model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────
# MIXED INT5/INT6 QUANTIZATION + ZSTD
# ─────────────────────────────────────────────
# Int5 for MLP weights ([-16,15], 3 zero high bits → zstd compresses at ~1.88x)
# Int6 for attention weights ([-32,31], 2 zero high bits → zstd compresses at ~1.51x)
# This saves ~1.8MB vs all-int6, funding BigramHash(10240).

INT6_CLIP = 31  # [-32, 31]
INT5_CLIP = 15  # [-16, 15]
FP16_KEEP_PATTERNS = ("tok_emb",)
MLP_PATTERNS = (".mlp.",)  # tensors matching these get int5

def _quantize_intN_per_row(t: Tensor, clip: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    row_max = t32.abs().amax(dim=1)
    scale = (row_max / clip).clamp_min(1.0 / clip).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale[:, None].float()), -(clip + 1), clip).to(torch.int8)
    return q.contiguous(), scale.contiguous()

def quantize_state_dict_int6(state_dict):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, passthrough_orig_dtypes = {}, {}
    stats = {"param_count": 0, "payload_bytes": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()

        # FP16 passthrough for embeddings
        if any(p in name for p in FP16_KEEP_PATTERNS):
            pt = t.to(torch.float16).contiguous()
            passthrough[name] = pt
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["payload_bytes"] += pt.numel() * pt.element_size()
            continue

        # Small tensors: keep as fp16
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_PATTERNS):
                passthrough[name] = t.float().contiguous()
                stats["payload_bytes"] += t.numel() * 4
            elif t.is_floating_point():
                pt = t.to(torch.float16).contiguous()
                passthrough[name] = pt
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                stats["payload_bytes"] += pt.numel() * 2
            else:
                passthrough[name] = t
                stats["payload_bytes"] += t.numel() * t.element_size()
            continue

        # Large 2D float tensors: int5 for MLP, int6 for attention/other
        if t.is_floating_point() and t.ndim == 2:
            clip = INT5_CLIP if any(p in name for p in MLP_PATTERNS) else INT6_CLIP
            q, s = _quantize_intN_per_row(t, clip)
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["payload_bytes"] += q.numel() * 1 + s.numel() * 2
        elif t.is_floating_point():
            # 1D float: per-tensor int8
            clip_abs = float(t.float().abs().max().item()) if t.numel() else 0.0
            sc = max(clip_abs / 127.0, 1.0 / 127.0)
            q = torch.clamp(torch.round(t.float() / sc), -127, 127).to(torch.int8).contiguous()
            quantized[name] = q
            scales[name] = torch.tensor(sc, dtype=torch.float32)
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["payload_bytes"] += q.numel() * 1 + 4
        else:
            passthrough[name] = t
            stats["payload_bytes"] += t.numel() * t.element_size()

    obj = {
        "__quant_format__": "int6_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes,
        "passthrough": passthrough, "passthrough_orig_dtypes": passthrough_orig_dtypes,
    }
    return obj, stats

def dequantize_state_dict_int6(obj):
    out = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    actual_size = Path(file).stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size}, got {actual_size}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}: expected {num_tokens}, got {tokens_np.size}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class SmearGate(nn.Module):
    """Learned per-dim gate blending each token with previous token embedding."""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHash(nn.Module):
    """Hash-based bigram embedding for additive token-pair context."""
    def __init__(self, bigram_vocab, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab = bigram_vocab
        self.embed = nn.Embedding(bigram_vocab, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.bigram_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, token_ids):
        t = token_ids.to(torch.int32)
        mod = self.bigram_vocab - 1
        h = torch.empty_like(t)
        h[..., 0] = mod  # sentinel for position 0
        h[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        emb = self.embed(h.long())
        return self.proj(emb) * self.bigram_scale.to(dtype=emb.dtype)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = (0, None, None)

    def forward(self, seq_len, device, dtype):
        if self._cache[0] != seq_len or self._cache[1] is None or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.use_xsa = use_xsa
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def _xsa_efficient(self, y, v):
        """Exclusive Self Attention: remove value-direction component from output.
        y: [B, T, H, D], v: [B, T, Hkv, D]"""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)  # [B, T, Hkv, 1, D]
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        # Extra RMSNorm before Q/K projections (Steinmetz 2025): stabilizes activation
        # scales entering the fragile RoPE path, improving post-quantization quality.
        x_qk = F.rms_norm(x, (x.size(-1),))
        q = self.c_q(x_qk).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x_qk).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        # y: [B, H, T, D], v: [B, Hkv, T, D]
        if self.use_xsa:
            # Transpose to [B, T, H, D] and [B, T, Hkv, D]
            y_bt = y.transpose(1, 2)  # [B, T, H, D]
            v_bt = v.transpose(1, 2)  # [B, T, Hkv, D]
            y_bt = self._xsa_efficient(y_bt, v_bt)
            return self.proj(y_bt.contiguous().reshape(bsz, seqlen, dim))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, use_xsa=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, bigram_vocab, bigram_dim, xsa_last_n=3):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.smear_gate = SmearGate(model_dim)
        self.bigram_hash = BigramHash(bigram_vocab, bigram_dim, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  use_xsa=(i >= num_layers - xsa_last_n))
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights(num_layers, tied_embed_init_std)

    def _init_weights(self, num_layers, std):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        out_scale = 1.0 / math.sqrt(2 * num_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    # muP: scale output projections in transformer blocks only
                    if "blocks." in name and ".proj." in name and "c_" not in name:
                        module.weight.data.mul_(out_scale)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        targets = target_ids.reshape(-1)
        logits_flat = logits.reshape(-1, logits.size(-1))
        logits_capped = self.logit_softcap * torch.tanh(logits_flat / self.logit_softcap)
        return F.cross_entropy(logits_capped.float(), targets, reduction="mean")

    def forward_logits(self, input_ids):
        x = self.tok_emb(input_ids)
        x = self.smear_gate(x)
        x = x + self.bigram_hash(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            return F.linear(x, self.tok_emb.weight)
        return self.lm_head(x)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyp()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    if grad_accum_steps <= 0:
        raise ValueError(f"WORLD_SIZE={world_size} too large: grad_accum_steps would be 0")
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        if console: print(msg, flush=True)
        if logfile:
            with open(logfile, "a") as f: print(msg, file=f)

    log0(code, console=False)
    log0("=" * 80, console=False)

    # Seed
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer vocab_size={int(sp.vocab_size())}")

    # Validation data - load and align to seq_len for simple eval
    val_files_list = [Path(p) for p in sorted(glob.glob(args.val_files))]
    val_tokens_raw = torch.cat([load_data_shard(f) for f in val_files_list]).contiguous()
    usable = ((val_tokens_raw.numel() - 1) // args.train_seq_len) * args.train_seq_len
    val_tokens = val_tokens_raw[:usable + 1]
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel()} (raw:{val_tokens_raw.numel()})")

    # Model
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab=args.bigram_vocab_size, bigram_dim=args.bigram_dim, xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizers
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named if p.ndim == 2 and not any(c in n for c in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_named if p.ndim < 2 or any(c in n for c in CONTROL_PATTERNS)]

    # BigramHash: proj weight goes to Muon (it's a real weight matrix),
    # but the embedding table goes to Adam (it's an embedding, not a weight matrix)
    for n, p in base_model.bigram_hash.named_parameters():
        if n == "proj.weight":
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    # SmearGate params -> Adam
    for p in base_model.smear_gate.parameters():
        scalar_params.append(p)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                    backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    optimizers = [opt_tok, opt_muon, opt_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
        )
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} mlp_mult:{args.mlp_mult}")
    log0(f"matrix_lr:{args.matrix_lr} muon_wd:{args.muon_wd} adam_wd:{args.adam_wd} grad_clip:{args.grad_clip_norm}")
    log0(f"seq_len:{args.train_seq_len} warmdown:{args.warmdown_iters} swa:{args.swa_enabled}/{args.swa_every}")
    log0(f"seed:{args.seed} world_size:{world_size} xsa_last_n:{args.xsa_last_n}")
    log0(f"meta_ttt:{args.meta_ttt_enabled} start_frac:{args.meta_ttt_start_frac} inner_steps:{args.meta_ttt_inner_steps}")
    log0(f"extra_linear_rmsnorm:{args.extra_linear_rmsnorm} meta_ttt_log_every:{args.meta_ttt_log_every}")

    # Data loader
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    def apply_optimizers(step, scale):
        """Muon momentum warmup, LR schedule, grad clip, optimizer step, zero grad."""
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = muon_mom
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if warmdown_start <= step < args.iterations:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
            return 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def sync_reached_cap(approx_ms):
        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            t_cap = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(t_cap, op=dist.ReduceOp.MAX)
            reached_cap = bool(t_cap.item())
        return reached_cap

    # Warmup (torch.compile priming)
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opt_states): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # SWA state
    swa_state = None
    swa_count = 0

    # Training loop
    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Pre-compute Reptile meta-TTT param set (constant across steps)
    _num_blocks = len(base_model.blocks)
    _adapt_indices = range(max(0, _num_blocks - args.ttt_eval_adapt_last_n), _num_blocks)
    meta_adapt_ids = set()
    for _bi in _adapt_indices:
        for _n, _p in base_model.blocks[_bi].mlp.named_parameters():
            if "weight" in _n and ("fc" in _n or "proj" in _n):
                meta_adapt_ids.add(id(_p))

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val_simple(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}")
            break

        # Safety checkpoint
        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0 and master:
            torch.save(base_model.state_dict(), f"ckpt_step{step}.pt")
            log0(f"checkpoint saved: ckpt_step{step}.pt")

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Compute elapsed fraction for meta-TTT check
        if max_wallclock_ms is not None and max_wallclock_ms > 0:
            elapsed_frac = elapsed_ms / max_wallclock_ms
        else:
            elapsed_frac = step / max(args.iterations, 1)

        # SWA collection
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                swa_count = 1
            else:
                for k, v in base_model.state_dict().items():
                    swa_state[k] += v.detach().cpu()
                swa_count += 1

        # ─── Reptile Meta-TTT (last 15% of training) ───
        if args.meta_ttt_enabled and elapsed_frac >= args.meta_ttt_start_frac:
            if args.meta_ttt_log_every > 0 and (
                step == 0
                or step == args.iterations - 1
                or step % args.meta_ttt_log_every == 0
            ):
                log0(
                    f"meta_ttt:start step:{step}/{args.iterations} "
                    f"elapsed_frac:{elapsed_frac:.4f} inner_steps:{args.meta_ttt_inner_steps}"
                )
            # Save current params theta_0 (only adapted MLP params need cloning)
            theta_0 = {n: p.data.clone() for n, p in base_model.named_parameters()
                       if id(p) in meta_adapt_ids}

            # Inner loop: K SGD steps on consecutive batches (simulating TTT)
            for _inner in range(args.meta_ttt_inner_steps):
                approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
                if sync_reached_cap(approx_ms):
                    if stop_after_step is None:
                        stop_after_step = step
                    log0(
                        f"meta_ttt:aborting_for_wallclock step:{step}/{args.iterations} "
                        f"inner_step:{_inner}/{args.meta_ttt_inner_steps} train_time:{approx_ms:.0f}ms"
                    )
                    break
                zero_grad_all()
                for ms in range(grad_accum_steps):
                    if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        inner_loss = model(x, y)
                    (inner_loss * grad_scale).backward()
                # SGD step only on MLP params in last N blocks (matching eval TTT)
                with torch.no_grad():
                    for p in base_model.parameters():
                        if p.grad is not None and id(p) in meta_adapt_ids:
                            p.data.sub_(args.meta_ttt_inner_lr * p.grad)
                zero_grad_all()
                if args.meta_ttt_log_every > 0 and (
                    (_inner + 1) == args.meta_ttt_inner_steps
                    or step % args.meta_ttt_log_every == 0
                ):
                    log0(
                        f"meta_ttt:inner step:{step}/{args.iterations} "
                        f"inner:{_inner + 1}/{args.meta_ttt_inner_steps} loss:{inner_loss.item():.4f}"
                    )

            if stop_after_step is not None and step >= stop_after_step:
                continue

            # Reptile interpolate only adapted params
            with torch.no_grad():
                for n, p in base_model.named_parameters():
                    if id(p) in meta_adapt_ids:
                        p.data.copy_(theta_0[n] + args.meta_ttt_epsilon * (p.data - theta_0[n]))

            # Get eval batch and compute eval_loss for the normal optimizer step
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    eval_loss = model(x, y)
                (eval_loss * grad_scale).backward()

            apply_optimizers(step, scale)

            step += 1
            approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
                log0(f"step:{step}/{args.iterations} [meta-ttt] train_loss:{eval_loss.item():.4f} "
                     f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        else:
            # ─── Normal training step ───
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps

            apply_optimizers(step, scale)

            step += 1
            approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
            if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
                log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                     f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        reached_cap = sync_reached_cap(approx_ms)
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_mem: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # ─── POST-TRAINING PIPELINE ───

    # 1. Apply SWA
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"SWA: averaging {swa_count} checkpoints")
        avg = {k: (v / swa_count).to(dtype=base_model.state_dict()[k].dtype) for k, v in swa_state.items()}
        base_model.load_state_dict(avg, strict=True)
    else:
        log0("post_train: skipping SWA averaging")

    # 2. Save raw checkpoint
    log0("post_train: saving raw checkpoint")
    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"Raw checkpoint saved: {os.path.getsize('final_model.pt')} bytes")

    # 3. Int6 + zstd quantization
    log0("post_train: quantizing artifact")
    quant_obj, quant_stats = quantize_state_dict_int6(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = compress_blob(quant_buf.getvalue())
    artifact_name = f"final_model.{COMPRESS_EXT}"
    if master:
        with open(artifact_name, "wb") as f:
            f.write(quant_blob)
        artifact_bytes = os.path.getsize(artifact_name)
        code_bytes = len(code.encode("utf-8"))
        total_bytes = artifact_bytes + code_bytes
        log0(f"Artifact: {artifact_bytes} bytes, code: {code_bytes} bytes, total: {total_bytes} bytes")
        if total_bytes > 16_000_000:
            raise RuntimeError(f"artifact too large: total {total_bytes} exceeds 16MB limit")

    # 4. Verify artifact < 16MB (already done in step 3 logging)

    # 5. Load quantized model (roundtrip)
    log0("post_train: loading quantized roundtrip")
    if distributed: dist.barrier()
    with open(artifact_name, "rb") as f:
        rt_blob = f.read()
    rt_state = torch.load(io.BytesIO(decompress_blob(rt_blob)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(rt_state), strict=True)

    # 6. Run simple eval for baseline number
    log0("post_train: starting roundtrip eval")
    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    q_loss, q_bpb = eval_val_simple(
        args, model, rank, world_size, device, grad_accum_steps,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(f"final_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} eval_time:{1000*(time.perf_counter()-t_eval):.0f}ms")
    log0(f"final_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    # 7. Run online TTT eval with decay prior (the real score)
    log0("post_train: starting online TTT eval")
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_loss, ttt_bpb = eval_val_ttt(
        base_model, val_tokens_raw, args.eval_seq_len, args.eval_stride, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, rank, world_size, args, log_fn=log0,
    )
    torch.cuda.synchronize()
    log0(f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
    log0(f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
