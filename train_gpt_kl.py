#!/usr/bin/env python3
"""
KaiLean's Parameter Golf — CUDA/PyTorch version
Based on stock train_gpt.py with the following innovations:
  1. BigramHashEmbedding  — hash-based bigram logit features (BIGRAM_HASH_SIZE=10240)
  2. Int6 QAT with STE    — fake-quant during training (QAT_START_FRAC=0.15)
  3. EMA weight averaging — replaces SWA (EMA_DECAY=0.995, EMA_START_FRAC=0.5)
  4. OrthoInit            — SVD orthogonal init for Q/K/V/fc matrices (USE_ORTHO_INIT=1)
  5. 11-layer default     — NUM_LAYERS=11 (was 9)
  6. 3× MLP default       — MLP_MULT=3  (was 2)

Compatible with torchrun + DDP (8×H100, 4×H100, single GPU, etc.)
Run:  torchrun --nproc_per_node=8 train_gpt_kl.py
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
import zstandard
import brotli
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP


# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

class Hyperparameters:
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations          = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters      = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps        = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens  = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len       = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len        = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride         = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs     = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init        = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — KL defaults: 11 layers, 3× MLP
    vocab_size      = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers      = int(os.environ.get("NUM_LAYERS", 11))     # was 9
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim       = int(os.environ.get("MODEL_DIM", 512))
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult        = int(os.environ.get("MLP_MULT", 3))        # was 2
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # KL innovations
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 16384))
    qat_start_frac   = float(os.environ.get("QAT_START_FRAC", 0.15))  # UNUSED — late_qat_threshold controls QAT
    ema_decay        = float(os.environ.get("EMA_DECAY", 0.9965))
    ema_start_frac   = float(os.environ.get("EMA_START_FRAC", 0.5))
    use_ortho_init   = bool(int(os.environ.get("USE_ORTHO_INIT", "1")))
    # SWA — optional complement/replacement for EMA; starts at 60% of iterations
    use_swa          = bool(int(os.environ.get("USE_SWA", "0")))
    swa_decay        = float(os.environ.get("SWA_DECAY", "0.4"))

    # SOTA techniques
    smear_enabled    = bool(int(os.environ.get("USE_SMEARGATE", os.environ.get("SMEAR_ENABLED", "1"))))  # USE_SMEARGATE alias
    rope_dims        = int(os.environ.get("ROPE_DIMS", 16))             # partial RoPE dims; 0 = full RoPE
    ln_scale_enabled = bool(int(os.environ.get("LN_SCALE_ENABLED", os.environ.get("USE_LN_SCALE", "1"))))  # 1/sqrt(layer+1) depth scale
    xsa_last_n       = int(os.environ.get("XSA_LAST_N", 4))
    use_gptq_lite    = bool(int(os.environ.get("USE_GPTQ_LITE", "1")))

    # TODO(SOTA-diff): VALUE EMBEDDING — difficulty: MEDIUM (~40 lines)
    # SOTA reinjects token identity into attention values at specific late layers via a
    # shared low-dim embedding (ve_dim=128) projected to kv_dim, added as v_embed to V.
    # Applied at layers 9,10 with per-layer learned scales.

    # Depth Recurrence
    depth_recurrence  = bool(int(os.environ.get("DEPTH_RECURRENCE", "1")))
    recurrence_layers = [int(x) for x in os.environ.get("RECURRENCE_LAYERS", "3,4,5").split(",") if x.strip()]
    recurrence_loops  = int(os.environ.get("RECURRENCE_LOOPS", "2"))

    # Parallel Residuals (GPT-J style)
    parallel_residuals   = bool(int(os.environ.get("PARALLEL_RESIDUALS", "1")))
    parallel_res_start   = int(os.environ.get("PARALLEL_RES_START", "7"))

    # TTT (Test-Time Training)
    ttt_enabled        = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_rank           = int(os.environ.get("TTT_RANK", "4"))
    ttt_lr             = float(os.environ.get("TTT_LR", "0.001"))
    ttt_steps          = int(os.environ.get("TTT_STEPS", "3"))
    ttt_entropy_weight = float(os.environ.get("TTT_ENTROPY_WEIGHT", "1.0"))
    ttt_doc_independent = bool(int(os.environ.get("TTT_DOC_INDEPENDENT", "1")))

    # Optimizer
    embed_lr        = float(os.environ.get("EMBED_LR", 0.6))
    head_lr         = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR", 0.03))
    matrix_lr       = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr       = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))
    adam_weight_decay = float(os.environ.get("ADAM_WEIGHT_DECAY", 0.04))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))
    beta1           = float(os.environ.get("BETA1", 0.9))
    beta2           = float(os.environ.get("BETA2", 0.95))
    adam_eps        = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm  = float(os.environ.get("GRAD_CLIP_NORM", 0.3))


# ─────────────────────────────────────────────────────────────
# CONTROL TENSOR PATTERNS (shared by quantizer + optimizer split)
# ─────────────────────────────────────────────────────────────

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear",
    ).split(",") if p
)
INT6_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT6_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT6_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT6_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT6_PER_ROW_SCALE_DTYPE    = torch.float16
INT6_CLIP_Q                 = 99.99984 / 100.0
# GPTQ-lite: 5 percentile candidates to search per row
_GPTQ_PERCENTILES = [99.90, 99.95, 99.99, 99.999, 100.0]


def pack_int6_np(q_int8: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack int8 values in [-32,31] into 6-bit packed bytes (4 values per 3 bytes)."""
    orig_len = int(q_int8.size)
    u = (q_int8.ravel().astype(np.int16) + 32).astype(np.uint8)  # bias to [0,63]
    pad = (-len(u)) % 4
    if pad:
        u = np.concatenate([u, np.zeros(pad, dtype=np.uint8)])
    u = u.reshape(-1, 4).astype(np.uint16)
    out = np.empty((len(u), 3), dtype=np.uint8)
    out[:, 0] = (u[:, 0]        | (u[:, 1] << 6)).astype(np.uint8)
    out[:, 1] = ((u[:, 1] >> 2) | (u[:, 2] << 4)).astype(np.uint8)
    out[:, 2] = ((u[:, 2] >> 4) | (u[:, 3] << 2)).astype(np.uint8)
    return out.ravel(), orig_len


def unpack_int6_np(packed: np.ndarray, orig_len: int) -> np.ndarray:
    """Reverse pack_int6_np: uint8 packed bytes → int8 values in [-32,31]."""
    n_groups = (orig_len + 3) // 4
    p = packed.ravel()[:n_groups * 3].reshape(-1, 3).astype(np.uint16)
    u = np.empty((n_groups, 4), dtype=np.uint16)
    u[:, 0] =  p[:, 0]        & 0x3F
    u[:, 1] = ((p[:, 0] >> 6) | (p[:, 1] << 2)) & 0x3F
    u[:, 2] = ((p[:, 1] >> 4) | (p[:, 2] << 4)) & 0x3F
    u[:, 3] =  (p[:, 2] >> 2) & 0x3F
    return (u.ravel()[:orig_len].astype(np.int16) - 32).astype(np.int8)


# ─────────────────────────────────────────────────────────────
# MUON OPTIMIZER  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size  = dist.get_world_size() if distributed else 1
        rank        = dist.get_rank()       if distributed else 0

        for group in self.param_groups:
            params        = group["params"]
            lr            = group["lr"]
            momentum      = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov      = group["nesterov"]

            if not params:
                continue

            total_params  = sum(int(p.numel()) for p in params)
            updates_flat  = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
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

            wd = group["weight_decay"]
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                curr += p.numel()

        return loss


# ─────────────────────────────────────────────────────────────
# INNOVATION 1: Int6 QAT — fake quantization with STE
# ─────────────────────────────────────────────────────────────

def fake_quant_int6(w: Tensor) -> Tensor:
    """Simulate int6 symmetric quantization. Gradients pass through via STE."""
    scale = (w.abs().amax(dim=-1, keepdim=True) / 31.0).clamp_min(1e-8)
    w_q   = (w / scale).clamp(-32, 31).round_() * scale
    # Straight-through estimator: forward uses w_q, backward treats as identity on w
    return w + (w_q - w).detach()


# ─────────────────────────────────────────────────────────────
# INNOVATION 2: BigramHashEmbedding — hash-based bigram logit bias
# ─────────────────────────────────────────────────────────────

class BigramHashEmbedding(nn.Module):
    """
    For position t, computes hash(token[t-1], token[t]) → vocab-sized logit bias.
    Near-zero parameter overhead — the hash table is Muon-optimized like other 2D matrices.
    """
    def __init__(self, hash_size: int, vocab_size: int):
        super().__init__()
        self.hash_size = hash_size
        self.table     = nn.Embedding(hash_size, vocab_size)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        """tokens: (B, T) int64 → bigram bias: (B, T, vocab)"""
        t_prev    = tokens[:, :-1]
        t_curr    = tokens[:, 1:]
        idx       = (t_prev * 31337 + t_curr) % self.hash_size          # (B, T-1)
        bigram    = self.table(idx)                                       # (B, T-1, vocab)
        pad       = bigram.new_zeros(tokens.shape[0], 1, bigram.shape[-1])
        return torch.cat([pad, bigram], dim=1)                            # (B, T, vocab)


# ─────────────────────────────────────────────────────────────
# INNOVATION 3: EMA weight averaging
# ─────────────────────────────────────────────────────────────

class EMABuffer:
    """Maintains exponential moving averages of all model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay  = decay
        self.shadow = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
        }

    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def apply(self, model: nn.Module) -> dict[str, Tensor]:
        """Replace model weights with EMA weights. Returns saved originals."""
        saved = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                saved[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return saved

    def restore(self, model: nn.Module, saved: dict[str, Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in saved:
                param.data.copy_(saved[name])


# ─────────────────────────────────────────────────────────────
# TOKENIZER-AGNOSTIC EVALUATION  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size    = max(sp_vocab_size, vocab_size)
    base_bytes_np          = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np   = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np   = np.ones((table_size,),  dtype=np.bool_)
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
        torch.tensor(has_leading_space_np, dtype=torch.bool,  device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool,  device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
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
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            f"VAL_BATCH_SIZE too small for WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs       = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start        = (total_seqs * rank) // world_size
    seq_end          = (total_seqs * (rank + 1)) // world_size
    val_loss_sum     = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count  = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count   = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end   = batch_seq_end   * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count   = float(y.numel())
            val_loss_sum        += batch_loss.to(torch.float64) * batch_token_count
            val_token_count     += batch_token_count
            prev_ids  = x.reshape(-1)
            tgt_ids   = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
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


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    seq_len: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    """Sliding-window eval: each token scored with up to seq_len context.

    Windows of seq_len advance by stride.  Only the final stride tokens per
    window contribute to the metric (first window scores all tokens in it).
    With seq_len=2048 and stride=64, every token sees up to 2048 tokens of
    context instead of the 1024-token chunks used during training.
    """
    total_tokens  = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    base_model = model.module if isinstance(model, DDP) else model
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end  = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)  # (B, T, vocab)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s    = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum    += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt  = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb   = base_bytes_lut[tgt].to(torch.float64)
                tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct  = done / len(my_windows) * 100
                rbpb = 0.0
                if token_count.item() > 0:
                    rbpb = (loss_sum / token_count).item() / math.log(2.0) * (token_count.item() / byte_count.item())
                print(f"  sliding_eval [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={rbpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count,  op=dist.ReduceOp.SUM)

    val_loss        = (loss_sum / token_count).item()
    bits_per_token  = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


def eval_val_sliding_ttt(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    seq_len: int,
    batch_seqs: int = 32,
) -> tuple[float, float]:
    """Sliding-window eval with entropy-weighted LoRA TTT (score-first, per-document).

    For each validation window:
    1. Create low-rank LoRA adapters on Q and V projections.
    2. Run TTT_STEPS gradient steps on the prefix, weighted by predictive entropy.
    3. Compute final loss on the full window with adapted model.
    4. Reset LoRA (if TTT_DOC_INDEPENDENT).
    """
    total_tokens  = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    base_model = model.module if isinstance(model, DDP) else model
    base_model.eval()

    # Collect Q and V weight matrices for LoRA
    qv_pairs: list[tuple[str, str, int, int]] = []  # (q_name, v_name, out_f, in_f)
    for block in base_model.blocks:
        q_w = block.attn.c_q.weight  # (dim, dim)
        v_w = block.attn.c_v.weight  # (kv_dim, dim)
        qv_pairs.append((block.attn.c_q.weight, block.attn.c_v.weight))

    ttt_rank = args.ttt_rank
    ttt_lr   = args.ttt_lr
    ttt_steps = args.ttt_steps
    ent_weight = args.ttt_entropy_weight
    max_ent = math.log(args.vocab_size)  # max possible entropy

    # Create per-window LoRA adapters
    def create_lora_adapters():
        adapters = []
        for q_w, v_w in qv_pairs:
            out_f_q, in_f_q = q_w.shape
            out_f_v, in_f_v = v_w.shape
            a_q = torch.randn(out_f_q, ttt_rank, device=device, dtype=torch.bfloat16) * 0.01
            b_q = torch.zeros(ttt_rank, in_f_q, device=device, dtype=torch.bfloat16)
            a_v = torch.randn(out_f_v, ttt_rank, device=device, dtype=torch.bfloat16) * 0.01
            b_v = torch.zeros(ttt_rank, in_f_v, device=device, dtype=torch.bfloat16)
            adapters.append((a_q, b_q, a_v, b_v))
        return adapters

    def apply_lora(adapters):
        """Monkey-patch LoRA into Q/V forward passes. Returns originals for restoration."""
        originals = []
        for i, (block, (a_q, b_q, a_v, b_v)) in enumerate(zip(base_model.blocks, adapters)):
            orig_q_fwd = block.attn.c_q.forward
            orig_v_fwd = block.attn.c_v.forward
            w_q = block.attn.c_q.weight
            w_v = block.attn.c_v.weight

            def make_lora_forward(w, a, b, orig_fwd):
                def lora_forward(x):
                    return F.linear(x, w.to(x.dtype), None) + (x @ b.T.to(x.dtype)) @ a.T.to(x.dtype)
                return lora_forward

            block.attn.c_q.forward = make_lora_forward(w_q, a_q, b_q, orig_q_fwd)
            block.attn.c_v.forward = make_lora_forward(w_v, a_v, b_v, orig_v_fwd)
            originals.append((i, orig_q_fwd, orig_v_fwd))
        return originals

    def restore_lora(originals):
        for i, orig_q_fwd, orig_v_fwd in originals:
            base_model.blocks[i].attn.c_q.forward = orig_q_fwd
            base_model.blocks[i].attn.c_v.forward = orig_v_fwd

    # Process windows with TTT
    with torch.inference_mode(False):  # Need grad for TTT
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end  = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            # Create fresh LoRA adapters for this batch
            if args.ttt_doc_independent:
                adapters = create_lora_adapters()
            # Make LoRA params require grad
            lora_params = []
            for a_q, b_q, a_v, b_v in adapters:
                a_q.requires_grad_(True); b_q.requires_grad_(True)
                a_v.requires_grad_(True); b_v.requires_grad_(True)
                lora_params.extend([a_q, b_q, a_v, b_v])

            # Apply LoRA
            originals = apply_lora(adapters)

            # TTT: gradient steps on prefix tokens (score-first)
            # Score-first: compute baseline loss, then adapt, then only keep if improved
            prefix_len = max(seq_len // 4, 1)  # Use first 25% as TTT prefix
            
            # Save original LoRA state for score-first rollback
            orig_lora_data = [(a_q.data.clone(), b_q.data.clone(), a_v.data.clone(), b_v.data.clone()) for a_q, b_q, a_v, b_v in adapters]
            
            # Compute baseline loss on prefix before TTT
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    bl_logits = base_model.forward_logits(x_batch[:, :prefix_len])
                baseline_loss = F.cross_entropy(
                    bl_logits.reshape(-1, bl_logits.size(-1)).float(),
                    y_batch[:, :prefix_len].reshape(-1),
                    reduction="mean",
                ).item()
            
            for ttt_step in range(ttt_steps):
                with torch.enable_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = base_model.forward_logits(x_batch[:, :prefix_len])
                    # Standard NLL loss (no entropy weighting — that was hurting)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y_batch[:, :prefix_len].reshape(-1),
                        reduction="mean",
                    )
                # Gradient step
                grads = torch.autograd.grad(loss, lora_params)
                for p, g in zip(lora_params, grads):
                    p.data -= ttt_lr * g

            # Score-first check: compute loss after TTT, rollback if worse
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ad_logits = base_model.forward_logits(x_batch[:, :prefix_len])
                adapted_loss = F.cross_entropy(
                    ad_logits.reshape(-1, ad_logits.size(-1)).float(),
                    y_batch[:, :prefix_len].reshape(-1),
                    reduction="mean",
                ).item()
            
            # Rollback if TTT made things worse
            if adapted_loss > baseline_loss:
                for (a_q, b_q, a_v, b_v), (oa_q, ob_q, oa_v, ob_v) in zip(adapters, orig_lora_data):
                    a_q.data.copy_(oa_q); b_q.data.copy_(ob_q)
                    a_v.data.copy_(oa_v); b_v.data.copy_(ob_v)
                # Re-apply the original (rolled-back) LoRA
                restore_lora(originals)
                originals = apply_lora(adapters)

            # Score full window with adapted model
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)  # (B, T, vocab)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)

            # Restore original forwards
            restore_lora(originals)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s    = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum    += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt  = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb   = base_bytes_lut[tgt].to(torch.float64)
                tb  += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if not args.ttt_doc_independent:
                # Keep adapters across documents
                pass

            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct  = done / len(my_windows) * 100
                rbpb = 0.0
                if token_count.item() > 0:
                    rbpb = (loss_sum / token_count).item() / math.log(2.0) * (token_count.item() / byte_count.item())
                print(f"  sliding_eval_ttt [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={rbpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count,  op=dist.ReduceOp.SUM)

    val_loss        = (loss_sum / token_count).item()
    bits_per_token  = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# ─────────────────────────────────────────────────────────────
# POST-TRAINING QUANTIZATION  (int6 packed + zstandard)
# ─────────────────────────────────────────────────────────────

def tensor_nbytes(t) -> int:
    if isinstance(t, np.ndarray):
        return int(t.nbytes)
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(p in name for p in INT6_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT6_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor):
    """Quantize to int6 (range [-32,31]) with per-row float16 scales, packed 4-per-3-bytes.
    Returns (packed_np_uint8, scale_tensor, orig_shape, orig_len)."""
    t32 = t.float()
    t_np = t32.cpu().numpy()
    orig_shape = t_np.shape
    if t_np.ndim == 2:
        clip_abs = np.quantile(np.abs(t_np), INT6_CLIP_Q, axis=1) if t_np.size else np.empty((t_np.shape[0],), dtype=np.float32)
        clipped  = np.clip(t_np, -clip_abs[:, None], clip_abs[:, None])
        scale    = np.maximum(clip_abs / 31.0, 1.0 / 31.0).astype(np.float32)
        q        = np.clip(np.round(clipped / scale[:, None]), -32, 31).astype(np.int8)
        packed, orig_len = pack_int6_np(q)
        scale_t  = torch.from_numpy(scale).to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()
        return packed, scale_t, orig_shape, orig_len
    clip_abs = float(np.quantile(np.abs(t_np).ravel(), INT6_CLIP_Q)) if t_np.size else 0.0
    scale    = np.array(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q        = np.clip(np.round(np.clip(t_np, -clip_abs, clip_abs) / scale), -32, 31).astype(np.int8)
    packed, orig_len = pack_int6_np(q)
    scale_t  = torch.tensor(float(scale), dtype=torch.float32)
    return packed, scale_t, orig_shape, orig_len


def quantize_float_tensor_gptq_lite(t: Tensor):
    """GPTQ-lite: search 5 clip percentiles per row to minimize MSE. Returns same tuple as quantize_float_tensor."""
    t_np = t.float().cpu().numpy()
    orig_shape = t_np.shape
    if t_np.ndim == 2:
        n_rows = t_np.shape[0]
        clip_abs = np.zeros(n_rows, dtype=np.float32)
        for i in range(n_rows):
            row = t_np[i]
            best_mse, best_c = float('inf'), 1.0
            abs_row = np.abs(row)
            for pct in _GPTQ_PERCENTILES:
                c = float(np.quantile(abs_row, pct / 100.0)) if row.size else 1.0
                c = max(c, 1e-6)
                scale = c / 31.0
                q = np.clip(np.round(np.clip(row, -c, c) / scale), -32, 31).astype(np.float32)
                mse = float(np.mean((row - q * scale) ** 2))
                if mse < best_mse:
                    best_mse, best_c = mse, c
            clip_abs[i] = best_c
        clipped  = np.clip(t_np, -clip_abs[:, None], clip_abs[:, None])
        scale    = np.maximum(clip_abs / 31.0, 1.0 / 31.0).astype(np.float32)
        q        = np.clip(np.round(clipped / scale[:, None]), -32, 31).astype(np.int8)
        packed, orig_len = pack_int6_np(q)
        scale_t  = torch.from_numpy(scale).to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()
        return packed, scale_t, orig_shape, orig_len
    # Scalar tensors: fall back to fixed clip
    clip_abs = float(np.quantile(np.abs(t_np).ravel(), INT6_CLIP_Q)) if t_np.size else 0.0
    scale    = np.array(clip_abs / 31.0 if clip_abs > 0 else 1.0, dtype=np.float32)
    q        = np.clip(np.round(np.clip(t_np, -clip_abs, clip_abs) / scale), -32, 31).astype(np.int8)
    packed, orig_len = pack_int6_np(q)
    return packed, torch.tensor(float(scale), dtype=torch.float32), orig_shape, orig_len


def quantize_state_dict_int6(state_dict: dict[str, Tensor], use_gptq_lite: bool = False, embed_bits: int = 6):
    """Quantize state dict to int6 (or mixed int8/int6 for embeddings).
    
    embed_bits=6: all weights int6 (default, competition standard)
    embed_bits=8: embedding matrices (tok_emb, lm_head) use int8, rest int6 (SOTA approach)
    """
    _EMBED_PATTERNS = ("tok_emb.weight", "lm_head.weight")
    quantized: dict[str, np.ndarray]    = {}
    scales:    dict[str, Tensor]        = {}
    shapes:    dict[str, tuple]         = {}
    dtypes:    dict[str, str]           = {}
    passthrough: dict[str, Tensor]      = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict]              = {}
    stats = dict.fromkeys(
        ("param_count","num_tensors","num_float_tensors","num_nonfloat_tensors",
         "baseline_tensor_bytes","int6_payload_bytes"), 0,
    )
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"]           += int(t.numel())
        stats["num_tensors"]           += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue
        if t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        
        # Mixed quantization: int8 for embeddings, int6 for weights
        use_int8 = (embed_bits == 8 and any(p in name for p in _EMBED_PATTERNS))
        if use_int8:
            q_int8, s_int8 = quantize_float_tensor_int8(t)
            # Store int8 quantized data in the same format as int6 but mark it
            packed_raw = q_int8.cpu().numpy().tobytes()
            packed_np = np.frombuffer(packed_raw, dtype=np.uint8).copy()
            quantized[name] = packed_np
            scales[name]    = s_int8
            shapes[name]    = t.shape
            dtypes[name]    = str(t.dtype).removeprefix("torch.")
            qmeta[name] = {"scheme": "per_row_int8", "axis": 0}
            stats["int6_payload_bytes"] += len(packed_raw) + tensor_nbytes(s_int8)
        else:
            _quant_fn = quantize_float_tensor_gptq_lite if use_gptq_lite else quantize_float_tensor
            packed, s, orig_shape, orig_len = _quant_fn(t)
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            quantized[name] = packed
            scales[name]    = s
            shapes[name]    = orig_shape
            dtypes[name]    = str(t.dtype).removeprefix("torch.")
            stats["int6_payload_bytes"] += tensor_nbytes(packed) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int6_packed_per_row_v1" if embed_bits == 6 else "mixed_int8_int6_packed_v1",
        "quantized": quantized, "scales": scales, "shapes": shapes,
        "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:                   obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


# ─────────────────────────────────────────────────────────────
# INT8 + ZLIB COMPRESSION (Competition-compliant submission format)
# ─────────────────────────────────────────────────────────────

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
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


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    """Competition-compliant int8 quantization: per-row int8 for 2D, per-tensor for others."""
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
        q, s = quantize_float_tensor_int8(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough,
    }
    if qmeta:                   obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
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


def dequantize_state_dict_int6(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta  = obj.get("qmeta", {})
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    shapes = obj.get("shapes", {})
    quant_format = obj.get("__quant_format__", "int6_packed_per_row_v1")
    for name, packed in obj["quantized"].items():
        orig_shape = shapes[name]
        orig_len   = int(np.prod(orig_shape))
        meta = qmeta.get(name, {})
        # Handle int8 quantized tensors (mixed int8/int6 format)
        if meta.get("scheme") == "per_row_int8":
            q_int8 = np.frombuffer(packed.tobytes() if isinstance(packed, np.ndarray) else packed, dtype=np.int8).copy().reshape(orig_shape)
            dtype = getattr(torch, obj["dtypes"][name])
            s = obj["scales"][name]
            s32 = s.to(dtype=torch.float32)
            q_t = torch.from_numpy(q_int8.astype(np.float32))
            out[name] = (q_t * s32.view(q_t.shape[0], *([1] * (q_t.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            q_np = unpack_int6_np(np.asarray(packed, dtype=np.uint8), orig_len).reshape(orig_shape)
            dtype = getattr(torch, obj["dtypes"][name])
            s = obj["scales"][name]
            if meta.get("scheme") == "per_row" or s.ndim > 0:
                s32 = s.to(dtype=torch.float32)
                q_t = torch.from_numpy(q_np.astype(np.float32))
                out[name] = (q_t * s32.view(q_t.shape[0], *([1] * (q_t.ndim - 1)))).to(dtype=dtype).contiguous()
            else:
                out[name] = (torch.from_numpy(q_np.astype(np.float32)) * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous() if isinstance(t, Tensor) else torch.from_numpy(np.array(t))
        orig  = pt_dtypes.get(name)
        out[name] = out_t.to(dtype=getattr(torch, orig)).contiguous() if isinstance(orig, str) else out_t
    return out


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes  = np.dtype("<u2").itemsize
    header       = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens    = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files    = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens   = load_data_shard(self.files[self.file_idx])
        self.pos      = 0

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
            self.pos  += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank       = rank
        self.world_size = world_size
        self.device     = device
        self.stream     = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens  = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ─────────────────────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """
    Keeps weights in fp32; casts to bf16 at matmul time.
    When use_qat=True, weights are fake-quantized to int6 before the cast.
    torch.compile specialises on use_qat and retraces on the single toggle.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.use_qat = False  # Toggled True once QAT phase starts

    def forward(self, x: Tensor) -> Tensor:
        w    = fake_quant_int6(self.weight) if self.use_qat else self.weight
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


class SmearGate(nn.Module):
    """Blend each token embedding with the previous token's embedding.
    gate init=3.0 → sigmoid≈0.95 (nearly pass-through at start). PR #102/#135."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


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
            self._cos_cached is None or self._sin_cached is None
            or self._seq_len_cached != seq_len or self._cos_cached.device != device
        ):
            t     = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached    = freqs.cos()[None, None, :, :]
            self._sin_cached    = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float,
                 use_xsa: bool = False, rope_dims: int = 0):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        kv_dim            = self.num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        # Partial RoPE: rotate only first rope_dims of each head; 0 = full head_dim
        _rot_dims = rope_dims if rope_dims > 0 else self.head_dim
        self.rotary   = Rotary(_rot_dims, base=rope_base)
        self.rope_dims = rope_dims  # 0 means full rotation
        self.use_xsa = use_xsa

    def _xsa(self, y: Tensor, v: Tensor) -> Tensor:
        """Subtract self-value component — forces attention to other tokens (XSA, PR #198).
        y: (B, H, T, D), v: (B, Hkv, T, D) → GQA-aware subtraction."""
        B, H, T, D = y.shape
        Hkv = v.shape[1]
        group = H // Hkv
        y_g = y.reshape(B, Hkv, group, T, D)
        v_norm = v / (v.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6)
        vn = v_norm[:, :, None, :, :]   # (B, Hkv, 1, T, D)
        proj = (y_g * vn).sum(-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, H, T, D)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        if self.rope_dims > 0 and self.rope_dims < self.head_dim:
            # Partial RoPE: rotate first rope_dims, leave remainder unrotated
            q_rot, q_pass = q[..., :self.rope_dims], q[..., self.rope_dims:]
            k_rot, k_pass = k[..., :self.rope_dims], k[..., self.rope_dims:]
            q = torch.cat([apply_rotary_emb(q_rot, cos, sin), q_pass], dim=-1)
            k = torch.cat([apply_rotary_emb(k_rot, cos, sin), k_pass], dim=-1)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        if self.use_xsa:
            y = self._xsa(y, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden    = mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float,
                 use_xsa: bool = False, rope_dims: int = 0, layer_idx: int = 0, use_ln_scale: bool = True):
        super().__init__()
        self.attn_norm  = RMSNorm()
        self.mlp_norm   = RMSNorm()
        self.attn       = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                              use_xsa=use_xsa, rope_dims=rope_dims)
        self.mlp        = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # 1/sqrt(layer+1) depth scale — dampens deep layer activations (ln_scale)
        self.ln_scale_factor = float(1.0 / math.sqrt(layer_idx + 1)) if use_ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, parallel_residual: bool = False) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x   = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s   = self.ln_scale_factor
        if parallel_residual:
            # GPT-J / PaLM style: attention and MLP in parallel
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x) * s) \
                   + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        else:
            # Standard sequential: attention then MLP
            x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x) * s)
            x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :]  * self.mlp(self.mlp_norm(x) * s)
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
        bigram_hash_size: int,
        use_ortho_init: bool,
        smear_enabled: bool = True,
        xsa_last_n: int = 0,
        rope_dims: int = 0,
        ln_scale_enabled: bool = True,
        depth_recurrence: bool = False,
        recurrence_layers: list[int] | None = None,
        recurrence_loops: int = 2,
        parallel_residuals: bool = False,
        parallel_res_start: int = 7,
    ):
        super().__init__()
        self.tie_embeddings     = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap      = logit_softcap
        self.depth_recurrence   = depth_recurrence
        self.recurrence_layers  = recurrence_layers if recurrence_layers is not None else []
        self.recurrence_loops   = recurrence_loops
        self.parallel_residuals = parallel_residuals
        self.parallel_res_start = parallel_res_start

        self.tok_emb            = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights       = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # SmearGate: blend each token embedding with predecessor's
        self.smear = SmearGate(model_dim) if smear_enabled else None

        # XSA on last xsa_last_n decoder layers
        xsa_decoder_start = max(0, self.num_decoder_layers - xsa_last_n) if xsa_last_n > 0 else self.num_decoder_layers
        self.blocks             = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  use_xsa=(i >= self.num_encoder_layers + xsa_decoder_start),
                  rope_dims=rope_dims, layer_idx=i, use_ln_scale=ln_scale_enabled)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head    = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # INNOVATION 2: BigramHash (None if hash_size == 0)
        self.bigram_hash = BigramHashEmbedding(bigram_hash_size, vocab_size) if bigram_hash_size > 0 else None

        self._init_weights(use_ortho_init)

    def _init_weights(self, use_ortho_init: bool = False) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
        # INNOVATION 4: OrthoInit — semi-orthogonal init for Q/K/V and MLP up-proj
        # Applied after zero-init so output projections remain zero.
        if use_ortho_init:
            for block in self.blocks:
                for linear in [block.attn.c_q, block.attn.c_k, block.attn.c_v, block.mlp.fc]:
                    nn.init.orthogonal_(linear.weight, gain=0.5)

    def _run_blocks(self, x: Tensor, x0: Tensor) -> Tensor:
        """Run all blocks with depth recurrence and parallel residuals support."""
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            par = self.parallel_residuals and i >= self.parallel_res_start
            x = self.blocks[i](x, x0, parallel_residual=par)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            bi = self.num_encoder_layers + i
            par = self.parallel_residuals and bi >= self.parallel_res_start
            x = self.blocks[bi](x, x0, parallel_residual=par)
        # Depth recurrence: re-run specified layers additional times
        if self.depth_recurrence and self.recurrence_layers and self.recurrence_loops > 1:
            for _ in range(self.recurrence_loops - 1):
                for li in self.recurrence_layers:
                    if li < len(self.blocks):
                        par = self.parallel_residuals and li >= self.parallel_res_start
                        x = self.blocks[li](x, x0, parallel_residual=par)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x  = self.tok_emb(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        x = self._run_blocks(x, x0)
        x       = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        # INNOVATION 2: add bigram hash bias to logits
        if self.bigram_hash is not None:
            bigram_bias = self.bigram_hash(input_ids)             # (B, T, vocab)
            logits = logits + bigram_bias.reshape(-1, bigram_bias.shape[-1]).to(logits.dtype)

        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return (B, T, vocab) logits without computing loss — used for sliding-window eval."""
        x  = self.tok_emb(input_ids)
        x  = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        x = self._run_blocks(x, x0)
        x = self.final_norm(x)  # (B, T, d_model) — no reshape
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if self.bigram_hash is not None:
            logits = logits + self.bigram_hash(input_ids).to(logits.dtype)
        return logits  # (B, T, vocab)


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # ── distributed + CUDA setup ──────────────────────────────
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK",       "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale       = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32        = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile: str | None = None
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
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
    log0("=" * 100, console=False)

    # ── tokenizer + validation metric setup ───────────────────
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    dataset_dir       = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens         = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"val_bpb:enabled tokenizer_path={args.tokenizer_path}")
    log0(f"dataset:{dataset_dir.name} train_shards:{actual_train_files} val_tokens:{val_tokens.numel()-1}")

    # ── model ─────────────────────────────────────────────────
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_hash_size=args.bigram_hash_size, use_ortho_init=args.use_ortho_init,
        smear_enabled=args.smear_enabled, xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims, ln_scale_enabled=args.ln_scale_enabled,
        depth_recurrence=args.depth_recurrence,
        recurrence_layers=args.recurrence_layers,
        recurrence_loops=args.recurrence_loops,
        parallel_residuals=args.parallel_residuals,
        parallel_res_start=args.parallel_res_start,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed else compiled_model
    )

    n_params = sum(p.numel() for p in base_model.parameters())
    xsa_layers = [i for i, b in enumerate(base_model.blocks) if b.attn.use_xsa]
    log0(f"model_params:{n_params} layers:{args.num_layers} dim:{args.model_dim} "
         f"mlp_mult:{args.mlp_mult} bigram_hash:{args.bigram_hash_size} "
         f"ortho_init:{args.use_ortho_init} ema_decay:{args.ema_decay}")
    log0(f"innovations: smear={args.smear_enabled} rope_dims={args.rope_dims} "
         f"ln_scale={args.ln_scale_enabled} xsa_last_n={args.xsa_last_n} xsa_layers={xsa_layers} "
         f"gptq_lite={args.use_gptq_lite} use_swa={args.use_swa} swa_decay={args.swa_decay}")
    log0(f"depth_recurrence={args.depth_recurrence} recurrence_layers={args.recurrence_layers} recurrence_loops={args.recurrence_loops}")
    log0(f"parallel_residuals={args.parallel_residuals} parallel_res_start={args.parallel_res_start}")
    log0(f"ttt_enabled={args.ttt_enabled} ttt_rank={args.ttt_rank} ttt_lr={args.ttt_lr} ttt_steps={args.ttt_steps} ttt_entropy_weight={args.ttt_entropy_weight}")

    # ── optimizer split ────────────────────────────────────────
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for name, p in block_named_params
        if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    # BigramHash table weight goes to Muon (it's a 2D matrix)
    if base_model.bigram_hash is not None:
        matrix_params.append(base_model.bigram_hash.table.weight)

    token_lr      = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        weight_decay=args.adam_weight_decay,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        weight_decay=args.adam_weight_decay,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
            weight_decay=args.adam_weight_decay,
        )
        optimizers.insert(1, optimizer_head)

    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} warmup_steps:{args.warmup_steps}")
    log0(f"seed:{args.seed}")

    # ── data loader ───────────────────────────────────────────
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
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations else 1.0
            )
        step_ms      = elapsed_ms / max(step, 1)
        warmdown_ms  = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # ── warmup: prime compiled paths then restore state ────────
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opt_states  = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
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
        for opt, state in zip(optimizers, initial_opt_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── main training loop ─────────────────────────────────────
    training_time_ms: float   = 0.0
    stop_after_step: int | None = None
    # Innovation state
    use_qat_active: bool = False
    ema: EMABuffer | None = None
    swa: EMABuffer | None = None

    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0

    while True:
        last_step    = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            # Apply SWA (preferred) or EMA weights for eval, then restore
            _avg = swa if swa is not None else ema
            saved_params: dict[str, Tensor] = {}
            if _avg is not None:
                saved_params = _avg.apply(base_model)

            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            qat_tag = " [QAT]" if use_qat_active else ""
            avg_tag = " [SWA]" if swa is not None else (" [EMA]" if ema is not None else "")
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms"
                f"{qat_tag}{avg_tag}"
            )

            if _avg is not None:
                _avg.restore(base_model, saved_params)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        # ── estimate total steps for fractional thresholds ────
        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        est_total  = args.iterations
        if max_wallclock_ms is not None and step > 0:
            step_ms_avg = training_time_ms / step
            est_total   = min(args.iterations, int(max_wallclock_ms / max(step_ms_avg, 1e-6)))

        scale = lr_mul(step, elapsed_ms)

        # Toggle QAT when lr_mul drops below late_qat_threshold (warmdown-triggered)
        # Only check after warmup completes to avoid triggering during warmup ramp
        if not use_qat_active and step > args.warmup_steps and scale < args.late_qat_threshold:
            use_qat_active = True
            for m in base_model.modules():
                if isinstance(m, CastedLinear):
                    m.use_qat = True
            # Recompile so the new static QAT branch is traced
            compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
            model = (
                DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
                if distributed else compiled_model
            )
            log0(f"qat_started:step={step} lr_mul={scale:.4f} — recompiled")

        # Initialise EMA once; initialise SWA at 60% of total iterations
        if ema is None and step >= int(est_total * args.ema_start_frac):
            ema = EMABuffer(base_model, decay=args.ema_decay)
            log0(f"ema_started:step={step}")
        if args.use_swa and swa is None and step >= int(est_total * 0.6):
            swa = EMABuffer(base_model, decay=args.swa_decay)
            log0(f"swa_started:step={step} decay={args.swa_decay}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # Update EMA and SWA after each optimizer step
        if ema is not None:
            ema.update(base_model)
        if swa is not None:
            swa.update(base_model)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            qat_tag = " [QAT]" if use_qat_active else ""
            ema_tag = " [EMA]" if ema is not None else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms{qat_tag}{ema_tag}"
            )

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
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

    # ── serialization + roundtrip validation ──────────────────

    # Apply SWA (preferred) or EMA to the final saved weights
    _final_avg = swa if swa is not None else ema
    if _final_avg is not None:
        _final_avg.apply(base_model)
        log0(f"{'swa' if swa is not None else 'ema'}_applied_for_save")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes  = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj_int8, quant_stats_int8 = quantize_state_dict_int8(base_model.state_dict())
    quant_buf_int8 = io.BytesIO()
    torch.save(quant_obj_int8, quant_buf_int8)
    quant_raw_int8 = quant_buf_int8.getvalue()
    quant_blob_int8 = zlib.compress(quant_raw_int8, level=9)
    quant_raw_bytes_int8 = len(quant_raw_int8)

    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob_int8)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio_int8 = quant_stats_int8["baseline_tensor_bytes"] / max(quant_stats_int8["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats_int8['int8_payload_bytes']} raw_torch:{quant_raw_bytes_int8} payload_ratio:{ratio_int8:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    # Also produce int6+zstd for comparison (our internal format)
    quant_obj_int6, quant_stats_int6 = quantize_state_dict_int6(base_model.state_dict(), use_gptq_lite=args.use_gptq_lite)
    quant_buf_int6 = io.BytesIO()
    torch.save(quant_obj_int6, quant_buf_int6)
    quant_raw_int6 = quant_buf_int6.getvalue()

    # ── int6+brotli (competition-winning format, smallest) ──
    quant_blob_int6_brotli = brotli.compress(quant_raw_int6, quality=11)

    if master_process:
        with open("final_model.int6.brotli.ptz", "wb") as f:
            f.write(quant_blob_int6_brotli)
        int6_brotli_bytes = os.path.getsize("final_model.int6.brotli.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio_int6 = quant_stats_int6["baseline_tensor_bytes"] / max(quant_stats_int6["int6_payload_bytes"], 1)
        log0(
            f"serialized_int6_brotli:{int6_brotli_bytes} bytes "
            f"(payload:{quant_stats_int6['int6_payload_bytes']} ratio:{ratio_int6:.2f}x)"
        )
        log0(f"Total submission size int6+brotli: {int6_brotli_bytes + code_bytes} bytes")

        # Also save int6+zstd for comparison
        quant_blob_int6_zstd = zstandard.ZstdCompressor(level=22).compress(quant_raw_int6)
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob_int6_zstd)
        int6_zstd_bytes = os.path.getsize("final_model.int6.ptz")
        log0(f"serialized_int6_zstd:{int6_zstd_bytes} bytes (comparison)")
        log0(f"Total submission size int6+zstd: {int6_zstd_bytes + code_bytes} bytes (comparison)")

    if distributed:
        dist.barrier()
    # Load and roundtrip the int6+brotli model (competition format)
    with open("final_model.int6.brotli.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(brotli.decompress(quant_blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int6(quant_state), strict=True)

    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    if args.eval_stride > 0:
        log0(f"final_eval_mode:sliding_window eval_seq_len:{args.eval_seq_len} stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args, model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, seq_len=args.eval_seq_len, batch_seqs=args.eval_batch_seqs,
        )
        # TTT eval (score-first test-time training)
        if args.ttt_enabled:
            log0(f"final_eval_mode:sliding_window_ttt ttt_rank:{args.ttt_rank} ttt_lr:{args.ttt_lr} ttt_steps:{args.ttt_steps} ent_weight:{args.ttt_entropy_weight}")
            ttt_val_loss, ttt_val_bpb = eval_val_sliding_ttt(
                args, model, rank, world_size, device,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                stride=args.eval_stride, seq_len=args.eval_seq_len, batch_seqs=args.eval_batch_seqs,
            )
            log0(f"final_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f}")
    else:
        log0("final_eval_mode:standard")
        q_val_loss, q_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    log0(
        f"final_int6_brotli_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_brotli_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
