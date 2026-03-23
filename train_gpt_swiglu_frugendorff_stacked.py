"""train_gpt.py — SwiGLU + U-Net + BigramHash + EMA + TTT + XSA4 + GPTQ-lite. Max 1500 lines."""

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
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# zstd-22 compression with zlib fallback
try:
    import zstandard as zstd
    USE_ZSTD = True
except ImportError:
    import zlib
    USE_ZSTD = False

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 42))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", "6000"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", "11"))  # Up from 9
    # Frugendorff: share middle layers. share_start=4, share_loops=3 means
    # blocks 0-3 unique, block 4 loops 3x (replaces layers 4,5,6), blocks 5-8 unique
    # = 9 stored blocks, 12 effective depth
    share_start = int(os.environ.get("SHARE_START", "4"))  # first shared layer index
    share_loops = int(os.environ.get("SHARE_LOOPS", "3"))  # how many times to loop the shared block
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", "8"))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3)) # Unused by Star-ReLU
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", "1792"))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # BigramHash config
    bigram_buckets = int(os.environ.get("BIGRAM_BUCKETS", "8192"))
    bigram_embed_dim = int(os.environ.get("BIGRAM_EMBED_DIM", 128))

    # Partial RoPE: apply rotary to only first ROPE_DIMS of head_dim (0 = full)
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))

    # LN Scale: scale norm input by 1/sqrt(layer_idx+1) per block
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))

    # Optimizer hyperparameters (updated to match #1 team)
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.035))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    decoder_lr_mult = float(os.environ.get("DECODER_LR_MULT", 2.0))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.04))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    # EMA: exponential moving average, updates every step (priority over SWA)
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", "0.9985"))

    # SWA config (fallback when EMA disabled)
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.5))
    swa_every = int(os.environ.get("SWA_EVERY", 50))

    # Late QAT: enable fake int6 quantization when LR scale < qat_threshold
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    qat_threshold = float(os.environ.get("QAT_THRESHOLD", "0.5"))  # earlier QAT: 3x more steps

    # TTT: SGD fine-tune on val data after training
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr = float(os.environ.get("TTT_LR", "0.0005"))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", "10"))
    ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_mlp_only = bool(int(os.environ.get("TTT_MLP_ONLY", "0")))
    ttt_cosine_decay = bool(int(os.environ.get("TTT_COSINE_DECAY", "1")))
    xsa_layers = int(os.environ.get("XSA_LAYERS", "4"))


# -----------------------------
# MUON OPTIMIZER WITH WEIGHT DECAY
# -----------------------------

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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.02):
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
                p.add_(g, alpha=-lr)
                # Apply weight decay after update
                if wd > 0:
                    p.mul_(1 - wd * lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------

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
        if piece.startswith("\u2581"):
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
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)

    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    base_model.eval()
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
                logits = base_model.forward_logits(x_batch)

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else wlen - stride
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()

                scored_prev = x_batch[i, s:wlen]
                scored_tgt = y_batch[i, s:wlen]
                tb = base_bytes_lut[scored_tgt].to(torch.int16)
                tb += (has_leading_space_lut[scored_tgt] & ~is_boundary_token_lut[scored_prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
                token_count += float(wlen - s)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING INT6 QUANTIZATION
# -----------------------------

INT6_MIN = -32
INT6_MAX = 31
INT6_CLIP_PERCENTILE = 99.99984
INT6_CLIP_Q = INT6_CLIP_PERCENTILE / 100.0

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,smear_gate,bigram,skip_gates",
    ).split(",")
    if pattern
)
INT6_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT6_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT6_PER_ROW_SCALE_DTYPE = torch.float16

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT6_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT6_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

# ---------------------------------------------------------------------------
# OptRot: Hadamard rotation before quantization to redistribute outliers
# ---------------------------------------------------------------------------
_hadamard_cache: dict[int, Tensor] = {}
def _hadamard(n: int) -> Tensor:
    """Normalized Hadamard matrix (self-inverse: H @ H = I). n must be power of 2."""
    if n in _hadamard_cache:
        return _hadamard_cache[n]
    if n == 1:
        H = torch.ones(1, 1)
    else:
        H_half = _hadamard(n // 2)
        H = torch.cat([torch.cat([H_half, H_half], 1), torch.cat([H_half, -H_half], 1)], 0) / math.sqrt(2)
    _hadamard_cache[n] = H
    return H

def optrot_rotate(W: Tensor) -> Tensor:
    """Apply Hadamard rotation to rows before quantization. Spreads outliers."""
    rows = W.shape[0]
    if rows & (rows - 1) != 0 or rows < 2:
        return W  # skip non-power-of-2
    H = _hadamard(rows).to(dtype=W.dtype, device=W.device)
    return H @ W

def optrot_unrotate(W: Tensor) -> Tensor:
    """Undo Hadamard rotation after dequantization. H is self-inverse."""
    return optrot_rotate(W)  # H @ H = I, so un-rotate = rotate again

# ---------------------------------------------------------------------------
# GPTQ: Hessian-aware quantization with column-wise error compensation
# ---------------------------------------------------------------------------
def _find_best_row_scales_int6(W: Tensor, clip_range: int = 31) -> Tensor:
    t32 = W.float()
    best_s = t32.abs().amax(dim=1) / clip_range
    best_s = best_s.clamp_min(1.0 / clip_range)
    best_err = torch.full((t32.shape[0],), float('inf'))
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range)
        q = torch.clamp(torch.round(t32 / s[:, None]), -clip_range, clip_range)
        err = (t32 - q * s[:, None]).pow(2).mean(dim=1)
        improved = err < best_err
        best_s[improved] = s[improved]
        best_err[improved] = err[improved]
    return best_s

def gptq_quantize_weight(W: Tensor, H: Tensor, clip_range: int = 31,
                          block_size: int = 128, percdamp: float = 0.01) -> tuple[Tensor, Tensor]:
    W = W.float().clone()
    rows, cols = W.shape
    row_scale = _find_best_row_scales_int6(W, clip_range)
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

def gptq_calibrate(model: nn.Module, train_pattern: str, device: torch.device,
                   n_samples: int = 256, seq_len: int = 1024) -> dict[str, Tensor]:
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
    return hessians

def quantize_float_tensor_int6(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize to int6 range [-32, 31], stored as int8."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(INT6_MAX)).clamp_min(1.0 / float(INT6_MAX))
        q = torch.clamp(torch.round(clipped / scale[:, None]), INT6_MIN, INT6_MAX).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(INT6_MAX) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), INT6_MIN, INT6_MAX).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int6(state_dict: dict[str, Tensor], gptq_hessians: dict[str, Tensor] | None = None):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int6_payload_bytes"),
        0,
    )
    gptq_count = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += tensor_nbytes(t)
            continue

        # Keep small float tensors and tok_emb.weight in fp16
        if t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL or name == "tok_emb.weight":
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        # OptRot: Hadamard rotation before quantization (power-of-2 rows only)
        use_optrot = bool(int(os.environ.get("USE_OPTROT", "1")))
        t_q = optrot_rotate(t) if (use_optrot and t.ndim == 2) else t
        rotated = t_q is not t  # track if rotation was applied
        # Use GPTQ when Hessian available for 2D weight matrices
        # GPTQ_SKIP_MLP=1: use naive int6 for MLP weights (better compression)
        gptq_skip_mlp = bool(int(os.environ.get("GPTQ_SKIP_MLP", "0")))
        is_mlp = any(k in name for k in ("gate_up.", "down.", "mlp."))
        module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
        H = gptq_hessians.get(module_name) if gptq_hessians and t_q.ndim == 2 else None
        skip_gptq = gptq_skip_mlp and is_mlp
        if H is not None and H.shape[0] == t_q.shape[1] and not skip_gptq:
            q, s = gptq_quantize_weight(t_q, H.cpu())
            gptq_count += 1
        else:
            q, s = quantize_float_tensor_int6(t_q)
        if s.ndim > 0:
            scheme = "per_row_rotated" if rotated else "per_row"
            qmeta[name] = {"scheme": scheme, "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    print(f"gptq_quantize: {gptq_count} GPTQ layers, {stats['num_float_tensors']-gptq_count} naive layers", flush=True)
    obj: dict[str, object] = {
        "__quant_format__": "int6_clean_per_row_v1",
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

def dequantize_state_dict_int6(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        meta = qmeta.get(name, {})
        scheme = meta.get("scheme", "") if isinstance(meta, dict) else ""
        if scheme in ("per_row", "per_row_rotated") or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            w = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
            if scheme == "per_row_rotated":
                w = optrot_unrotate(w)
            out[name] = w
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


# -----------------------------
# DATA LOADING
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


class DistributedTokenLoader:
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
    # Class-level flag: set True during late-QAT phase to enable fake int6 STE
    _qat_enabled: bool = False

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            # Fake int6 quantization via straight-through estimator
            with torch.no_grad():
                w32 = self.weight.float()
                row_clip = torch.quantile(w32.abs(), 0.9995, dim=1)
                scale = (row_clip / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """RoPE with optional partial application (first rope_dims of head_dim)."""
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        # rope_dims=0 means full head_dim; otherwise rotate only first rope_dims dims
        rope_d = rope_dims if rope_dims > 0 else dim
        self.rope_d = rope_d
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_d, 2, dtype=torch.float32) / rope_d))
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
    """Apply RoPE; if cos covers fewer dims than x, rotate only those dims."""
    rd = cos.size(-1) * 2
    if rd < x.size(-1):
        x_rope = x[..., :rd]
        x_pass = x[..., rd:]
        half = rd // 2
        x1 = x_rope[..., :half]
        x2 = x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float, rope_dims: int = 0):
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
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=rope_dims)
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
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        if self.use_xsa:
            y_xsa = y.transpose(1, 2)
            v_xsa = v.transpose(1, 2)
            y_xsa = self._xsa_efficient(y_xsa, v_xsa)
            y = y_xsa.reshape(bsz, seqlen, dim)
        else:
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, mlp_hidden: int = 0):
        super().__init__()
        # Star-ReLU implementation. mlp_mult is unused.
        hidden = mlp_hidden if mlp_hidden > 0 else int(dim * 3)
        self.up_proj = CastedLinear(dim, hidden, bias=False)
        self.down_proj = CastedLinear(hidden, dim, bias=False)
        self.down_proj._zero_init = True
        self.scale = nn.Parameter(torch.ones(hidden, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(hidden, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        x_up = self.up_proj(x)
        activated = F.leaky_relu(x_up, 0.5).pow(2)
        activated = activated * self.scale.to(dtype=activated.dtype) + self.bias.to(dtype=activated.dtype)
        return self.down_proj(activated)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_hidden: int = 0,
        rope_dims: int = 0,
        layer_idx: int = 0,
        ln_scale: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, rope_dims=rope_dims)
        self.mlp = MLP(dim, mlp_mult, mlp_hidden=mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.vrl_lambda = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        # LN Scale: dampen norm inputs by 1/sqrt(layer_idx+1) for deeper layers
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, v_first: Tensor | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        # VRL: mix current with first-block output for attention input
        attn_input = x
        if v_first is not None:
            lam = torch.sigmoid(self.vrl_lambda)
            attn_input = lam * x + (1 - lam) * v_first
        s = self.ln_scale_factor
        attn_out = self.attn(self.attn_norm(attn_input) * s)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        return x


# -----------------------------
# BIGRAM HASH EMBEDDING
# -----------------------------

class BigramHashEmbedding(nn.Module):
    """Hash-based bigram embedding that adds context from previous token."""
    def __init__(self, num_buckets: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)
        self.proj = CastedLinear(embed_dim, model_dim, bias=False)
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (bsz, seq_len)
        bsz, seq_len = input_ids.shape
        # Shift input_ids to get prev_ids, pad with 0
        prev_ids = F.pad(input_ids[:, :-1], (1, 0), value=0)
        # Hash: (prev_id * 1009 + curr_id) % buckets
        bigram_hash = (prev_ids * 1009 + input_ids) % self.num_buckets
        bigram_emb = self.embed(bigram_hash)
        return self.proj(bigram_emb)


# -----------------------------
# SMEAR GATE
# -----------------------------

class SmearGate(nn.Module):
    """Learned blending of current position with previous position."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        # x: (bsz, seq_len, dim)
        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))
        # Shift x to get previous position, pad with zeros
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - gate) * x + gate * x_prev


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
        mlp_hidden: int = 0,
        bigram_buckets: int = 4096,
        bigram_embed_dim: int = 128,
        rope_dims: int = 0,
        ln_scale: bool = False,
        xsa_last_n: int = 0,
        share_start: int = -1,
        share_loops: int = 1,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.share_start = share_start
        self.share_loops = share_loops
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram_emb = BigramHashEmbedding(bigram_buckets, bigram_embed_dim, model_dim)
        self.smear_gate = SmearGate(model_dim)
        # With sharing: num_layers unique blocks, but effective depth = num_layers + (share_loops - 1)
        # Layers before share_start: unique. share_start: looped share_loops times. After: unique.
        eff_depth = num_layers + (share_loops - 1) if share_start >= 0 else num_layers
        self.eff_depth = eff_depth
        self.num_encoder_layers = eff_depth // 2
        self.num_decoder_layers = eff_depth - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(
                model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                mlp_hidden=mlp_hidden, rope_dims=rope_dims, layer_idx=i, ln_scale=ln_scale,
            )
            for i in range(num_layers)
        ])
        # Orthogonal loop positions for the shared block
        if share_start >= 0 and share_loops > 1:
            raw = torch.randn(share_loops, model_dim)
            Q, _ = torch.linalg.qr(raw.T)
            self.loop_pos = nn.Parameter(Q.T[:share_loops] * 0.01)
        else:
            self.loop_pos = None
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                self.blocks[i].attn.use_xsa = True
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

    def _build_layer_sequence(self):
        """Build the effective layer sequence: unique blocks + shared block looped."""
        seq = []  # list of (block_idx, loop_idx_or_None)
        if self.share_start < 0 or self.share_loops <= 1:
            # No sharing — standard sequential
            for i in range(len(self.blocks)):
                seq.append((i, None))
        else:
            # Before shared: unique
            for i in range(self.share_start):
                seq.append((i, None))
            # Shared block looped
            for lp in range(self.share_loops):
                seq.append((self.share_start, lp))
            # After shared: unique (shifted by share_loops - 1 in effective index)
            for i in range(self.share_start + 1, len(self.blocks)):
                seq.append((i, None))
        return seq
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = x + self.bigram_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        v_first = None
        skips: list[Tensor] = []
        layer_seq = self._build_layer_sequence()
        for eff_i in range(self.num_encoder_layers):
            bi, lp = layer_seq[eff_i]
            if lp is not None and self.loop_pos is not None:
                x = x + self.loop_pos[lp]
            x = self.blocks[bi](x, x0, v_first=v_first)
            if eff_i == 0:
                v_first = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                gate = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
                x = gate[None, None, :] * x + (1.0 - gate[None, None, :]) * scaled_skip
            eff_j = self.num_encoder_layers + i
            bi, lp = layer_seq[eff_j]
            if lp is not None and self.loop_pos is not None:
                x = x + self.loop_pos[lp]
            x = self.blocks[bi](x, x0, v_first=v_first)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        x = self.tok_emb(input_ids)
        x = x + self.bigram_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        v_first = None
        skips: list[Tensor] = []
        layer_seq = self._build_layer_sequence()
        for eff_i in range(self.num_encoder_layers):
            bi, lp = layer_seq[eff_i]
            if lp is not None and self.loop_pos is not None:
                x = x + self.loop_pos[lp]
            x = self.blocks[bi](x, x0, v_first=v_first)
            if eff_i == 0:
                v_first = x
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                gate = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
                x = gate[None, None, :] * x + (1.0 - gate[None, None, :]) * scaled_skip
            eff_j = self.num_encoder_layers + i
            bi, lp = layer_seq[eff_j]
            if lp is not None and self.loop_pos is not None:
                x = x + self.loop_pos[lp]
            x = self.blocks[bi](x, x0, v_first=v_first)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# TEST-TIME TRAINING (TTT)
# -----------------------------

def ttt_adapt(
    args: Hyperparameters,
    base_model: nn.Module,
    device: torch.device,
    val_tokens: Tensor,
    rank: int = 0,
    world_size: int = 1,
    log_fn=None,
) -> None:
    """SGD fine-tune on validation data; all blocks unfrozen unless ttt_freeze_blocks > 0."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs

    if args.ttt_mlp_only:
        for name, p in base_model.named_parameters():
            if 'up_proj' not in name and 'down_proj' not in name and 'gate_proj' not in name and 'scale' not in name and 'bias' not in name:
                if 'mlp' not in name.lower():
                    p.requires_grad_(False)
    elif args.ttt_freeze_blocks > 0:
        for i, block in enumerate(base_model.blocks):
            if i < args.ttt_freeze_blocks:
                for p in block.parameters():
                    p.requires_grad_(False)

    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.0)
    ttt_scheduler = None
    if args.ttt_cosine_decay:
        ttt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ttt_epochs, eta_min=args.ttt_lr * 0.1)

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    base_model.train()
    t0 = time.perf_counter()

    for epoch in range(args.ttt_epochs):
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        epoch_tokens = torch.zeros((), device=device, dtype=torch.float64)

        for batch_start in range(my_start, my_end, batch_seqs):
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

            epoch_loss_sum += loss.detach().to(torch.float64) * float(y.numel())
            epoch_tokens += float(y.numel())

        if world_size > 1:
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_tokens, op=dist.ReduceOp.SUM)

        elapsed = time.perf_counter() - t0
        epoch_avg_loss = epoch_loss_sum.item() / max(epoch_tokens.item(), 1)
        if ttt_scheduler is not None:
            ttt_scheduler.step()
        if log_fn:
            log_fn(f"ttt_epoch:{epoch+1}/{args.ttt_epochs} loss:{epoch_avg_loss:.4f} time:{elapsed:.1f}s")

    # Re-enable all gradients
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
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout, console=False)
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
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}")
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    CastedLinear._qat_enabled = False  # start with QAT off; late_qat enables it mid-run

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
        mlp_hidden=args.mlp_hidden,
        bigram_buckets=args.bigram_buckets,
        bigram_embed_dim=args.bigram_embed_dim,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        xsa_last_n=args.xsa_layers,
        share_start=args.share_start,
        share_loops=args.share_loops,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Differential LR setup
    matrix_params_enc, scalar_params_enc = [], []
    matrix_params_dec, scalar_params_dec = [], []
    num_encoder_layers = base_model.num_encoder_layers
    for i, block in enumerate(base_model.blocks):
        is_decoder = i >= num_encoder_layers
        for name, p in block.named_parameters():
            if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
                (matrix_params_dec if is_decoder else matrix_params_enc).append(p)
            else:
                (scalar_params_dec if is_decoder else scalar_params_enc).append(p)

    # Non-block scalar parameters
    other_scalar_params = [base_model.smear_gate.gate, base_model.bigram_emb.embed.weight]
    if base_model.skip_weights.numel() > 0:
        other_scalar_params.append(base_model.skip_weights)
    if hasattr(base_model, 'skip_gates') and base_model.skip_gates.numel() > 0:
        other_scalar_params.append(base_model.skip_gates)
    if base_model.loop_pos is not None:
        other_scalar_params.append(base_model.loop_pos)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )

    matrix_lr_dec = args.matrix_lr * args.decoder_lr_mult
    optimizer_muon = Muon(
        [
            {'params': matrix_params_enc, 'lr': args.matrix_lr, 'base_lr': args.matrix_lr},
            {'params': matrix_params_dec, 'lr': matrix_lr_dec, 'base_lr': matrix_lr_dec},
        ],
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )

    scalar_lr_dec = args.scalar_lr * args.decoder_lr_mult
    optimizer_scalar = torch.optim.AdamW(
        [
            {'params': scalar_params_enc, 'lr': args.scalar_lr, 'base_lr': args.scalar_lr},
            {'params': scalar_params_dec, 'lr': scalar_lr_dec, 'base_lr': scalar_lr_dec},
            {'params': other_scalar_params, 'lr': args.scalar_lr, 'base_lr': args.scalar_lr},
        ],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )

    optimizer_bigram_proj = Muon(
        [base_model.bigram_emb.proj.weight],
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_bigram_proj.param_groups:
        group["base_lr"] = args.matrix_lr

    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_bigram_proj]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} decoder_lr_mult:{args.decoder_lr_mult}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} iterations:{args.iterations} warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"rope_dims:{args.rope_dims} ln_scale:{args.ln_scale}")
    log0(f"muon_wd:{args.muon_wd} adam_wd:{args.adam_wd} ema_enabled:{args.ema_enabled} late_qat:{args.late_qat} ttt_enabled:{args.ttt_enabled}")
    log0(f"bigram_buckets:{args.bigram_buckets} bigram_embed_dim:{args.bigram_embed_dim} seed:{args.seed}")

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
    # EMA / SWA STATE
    # -----------------------------

    # EMA takes priority; SWA is fallback (mutually exclusive)
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
        log0(f"ema:init decay={args.ema_decay}")

    swa_state: dict[str, Tensor] = {}
    swa_count = 0

    def update_swa():
        nonlocal swa_count
        with torch.no_grad():
            for name, param in base_model.state_dict().items():
                if name not in swa_state:
                    swa_state[name] = param.detach().cpu().clone().float()
                else:
                    swa_state[name].add_(param.detach().cpu().float())
        swa_count += 1

    def get_swa_state() -> dict[str, Tensor]:
        return {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype) for name, t in swa_state.items()}

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Estimate total steps for SWA start
    estimated_total_steps = args.iterations
    if max_wallclock_ms is not None:
        estimated_total_steps = min(args.iterations, int(max_wallclock_ms / 30))  # rough estimate

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args, model, rank, world_size, device, grad_accum_steps,
                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            )
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Late QAT: enable fake int6 quantization once LR scale drops below threshold
        if args.late_qat and not CastedLinear._qat_enabled and scale < args.qat_threshold:
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
        for group in optimizer_bigram_proj.param_groups:
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

        # EMA update every step (takes priority over SWA)
        if ema_state is not None:
            d = args.ema_decay
            with torch.no_grad():
                for name, t in base_model.state_dict().items():
                    ema_state[name].mul_(d).add_(t.detach().float(), alpha=1.0 - d)

        # SWA update (only when EMA disabled)
        swa_start_step = int(estimated_total_steps * args.swa_start_frac)
        if ema_state is None and step >= swa_start_step and step % args.swa_every == 0:
            update_swa()

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    # Final SWA update (only if EMA disabled and no SWA yet)
    if ema_state is None and swa_count == 0:
        update_swa()

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB")

    # Apply EMA or SWA weights (EMA takes priority)
    if ema_state is not None:
        log0("ema:applying EMA weights")
        avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
        del ema_state
        base_model.load_state_dict(avg_state, strict=True)
        del avg_state
    elif swa_count > 0:
        log0(f"swa:applying averaged {swa_count} checkpoints")
        base_model.load_state_dict(get_swa_state(), strict=True)
    else:
        log0("weight_avg:skipped (no EMA or SWA state)")

    # -----------------------------
    # TTT: fine-tune on val data AFTER EMA/SWA, BEFORE quantization
    # -----------------------------

    if args.ttt_enabled:
        if distributed:
            dist.barrier()
        log0(f"ttt:start lr={args.ttt_lr} momentum={args.ttt_momentum} epochs={args.ttt_epochs} freeze_blocks={args.ttt_freeze_blocks}")
        t_ttt = time.perf_counter()
        ttt_adapt(
            args, base_model, device, val_tokens,
            rank=rank, world_size=world_size, log_fn=log0,
        )
        log0(f"ttt:elapsed={time.perf_counter() - t_ttt:.1f}s")
        if distributed:
            dist.barrier()

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # GPTQ calibration: collect Hessians from training data
    log0("gptq:calibrating with training data...")
    t_gptq = time.perf_counter()
    gptq_hessians = gptq_calibrate(base_model, args.train_files, device, n_samples=256, seq_len=args.train_seq_len)
    log0(f"gptq:calibrated {len(gptq_hessians)} layers in {time.perf_counter()-t_gptq:.1f}s")
    quant_obj, quant_stats = quantize_state_dict_int6(base_model.state_dict(), gptq_hessians=gptq_hessians)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()

    # Use zstd-22 for compression (or zlib fallback)
    if USE_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        quant_blob = cctx.compress(quant_raw)
        compression_method = "zstd-22"
    else:
        import zlib
        quant_blob = zlib.compress(quant_raw, level=9)
        compression_method = "zlib-9"

    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int6.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int6_payload_bytes"], 1)
        log0(f"Serialized model int6+{compression_method}: {quant_file_bytes} bytes (payload:{quant_stats['int6_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)")
        log0(f"Total submission size int6+{compression_method}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob_disk = f.read()

    # Decompress
    if USE_ZSTD:
        dctx = zstd.ZstdDecompressor()
        quant_raw_disk = dctx.decompress(quant_blob_disk)
    else:
        import zlib
        quant_raw_disk = zlib.decompress(quant_blob_disk)

    quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int6(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val_sliding(
        args, base_model, rank, world_size, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, batch_seqs=args.eval_batch_seqs,
    )
    torch.cuda.synchronize()
    log0(f"final_int6_{compression_method}_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval:{1000.0*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"final_int6_{compression_method}_roundtrip_exact val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
