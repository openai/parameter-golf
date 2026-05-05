from __future__ import annotations
# =============================================================================
# Parameter Golf — Podium build
# Based on PR #549 (LeakyReLU² + TTT + Parallel Muon) stack
# New features:
#   [1] SP8192 defaults   — vocab_size=8192, bigger token alphabet
#   [2] Depth Recurrence  — RECURRENCE_LAYERS / RECURRENCE_LOOPS
#   [3] Parallel Residuals— PARALLEL_RESIDUALS (attn + MLP read same normed x)
#   [4] QK-Gain 5.0       — sharper attention, confirmed on leaderboard
#   [5] MuonEq-R          — equalized Newton-Schulz (unit-norm updates)
#   [6] Hessian SDClip    — std-based GPTQ clipping (beats fixed percentile)
#   [7] GPTQ Embeddings   — int8 quantise tok_emb + bigram + ve tables
#   [8] TTT Adam option   — faster TTT adaptation via Adam vs SGD
# =============================================================================
import copy
import glob
import importlib
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Optional, cast
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
REPO_ROOT = Path(__file__).resolve().parents[3]
zstandard: Any = None
try:
    zstandard = importlib.import_module("zstandard")
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"
try:
    flash_attn_3_func = cast(
        Callable[[Tensor, Tensor, Tensor, bool], Tensor],
        getattr(importlib.import_module("flash_attn_interface"), "flash_attn_func"),
    )
except ImportError:
    def flash_attn_3_func(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
        scale = 1.0 / math.sqrt(q.size(-1))
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        if q_sdpa.size(1) != k_sdpa.size(1):
            if q_sdpa.size(1) % k_sdpa.size(1) != 0:
                raise ValueError(
                    f"Incompatible attention head counts for fallback SDPA: "
                    f"q has {q_sdpa.size(1)} heads, k/v have {k_sdpa.size(1)} heads."
                )
            repeat_factor = q_sdpa.size(1) // k_sdpa.size(1)
            k_sdpa = k_sdpa.repeat_interleave(repeat_factor, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(repeat_factor, dim=1)
        out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, is_causal=causal, scale=scale
        )
        return out.transpose(1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
class Hyperparameters:
    # --- Data / tokenizer (SP8192 by default) ---
    data_path       = os.environ.get("DATA_PATH", str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp8192"))
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", str(REPO_ROOT / "data" / "tokenizers" / "fineweb_8192_bpe.model"))
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    # --- Evaluation ---
    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 4000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 500))
    eval_stride     = int(os.environ.get("EVAL_STRIDE", 64))
    eval_seq_len    = int(os.environ.get("EVAL_SEQ_LEN", 2048))

    # --- Training schedule ---
    iterations              = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters          = int(os.environ.get("WARMDOWN_ITERS", 3500))
    warmup_steps            = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens      = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len           = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds   = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # --- Model architecture ---
    vocab_size          = int(os.environ.get("VOCAB_SIZE", 8192))       # [1] SP8192
    num_layers          = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads        = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim           = int(os.environ.get("MODEL_DIM", 512))
    num_heads           = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult            = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings      = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base           = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap       = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # --- [2] Depth Recurrence ---
    # "4,5" means loop block indices 4 and 5. RECURRENCE_LOOPS=3 means each
    # specified block is called 3× total (1 normal + 2 extra passes).
    recurrence_layers   = os.environ.get("RECURRENCE_LAYERS", "4,5")
    recurrence_loops    = int(os.environ.get("RECURRENCE_LOOPS", 3))

    # --- [3] Parallel Residuals ---
    # When enabled, attn and MLP both read from norm(x_in) and their outputs
    # are added together: x_out = x_in + attn(norm(x)) + mlp(norm(x))
    parallel_residuals  = bool(int(os.environ.get("PARALLEL_RESIDUALS", "1")))

    # --- [4] QK-Gain ---
    qk_gain_init        = float(os.environ.get("QK_GAIN_INIT", 5.0))   # was 1.5

    # --- Attention extras ---
    xsa_last_n          = int(os.environ.get("XSA_LAST_N", 4))
    rope_dims           = int(os.environ.get("ROPE_DIMS", 16))
    ln_scale            = bool(int(os.environ.get("LN_SCALE", "1")))
    dtg_enabled         = bool(int(os.environ.get("DTG_ENABLED", "0")))
    gated_attention     = bool(int(os.environ.get("GATED_ATTENTION", "0")))
    value_residual      = bool(int(os.environ.get("VALUE_RESIDUAL", "0")))

    # --- Embeddings ---
    bigram_vocab_size   = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    bigram_dim          = int(os.environ.get("BIGRAM_DIM", 128))
    ve_enabled          = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim              = int(os.environ.get("VE_DIM", 128))
    ve_layers           = os.environ.get("VE_LAYERS", "9,10")

    # --- MTP ---
    mtp_num_heads       = int(os.environ.get("MTP_NUM_HEADS", 0))
    mtp_loss_weight     = float(os.environ.get("MTP_LOSS_WEIGHT", 0.2))

    # --- Optimizer learning rates ---
    tied_embed_lr               = float(os.environ.get("TIED_EMBED_LR", 0.035))
    embed_lr                    = float(os.environ.get("EMBED_LR", 0.6))
    head_lr                     = float(os.environ.get("HEAD_LR", 0.008))
    matrix_lr                   = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr                   = float(os.environ.get("SCALAR_LR", 0.025))
    beta1                       = float(os.environ.get("BETA1", 0.9))
    beta2                       = float(os.environ.get("BETA2", 0.95))
    adam_eps                    = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm              = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    muon_wd                     = float(os.environ.get("MUON_WD", 0.04))
    adam_wd                     = float(os.environ.get("ADAM_WD", 0.04))

    # --- [5] MuonEq-R ---
    # When enabled, Newton-Schulz updates are normalized so every weight
    # matrix gets an equal-magnitude step (unit Frobenius norm scaled by
    # sqrt(numel)), removing the implicit per-shape LR bias.
    muoneq_r                    = bool(int(os.environ.get("MUONEQ_R", "1")))

    # --- Muon momentum schedule ---
    muon_momentum               = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps          = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start  = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps  = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_beta2                  = float(os.environ.get("MUON_BETA2", 0.95))

    # --- Weight averaging ---
    swa_enabled     = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every       = int(os.environ.get("SWA_EVERY", 50))
    lawa_enabled    = bool(int(os.environ.get("LAWA_ENABLED", "0")))
    lawa_k          = int(os.environ.get("LAWA_K", 10))
    lawa_freq       = int(os.environ.get("LAWA_FREQ", 100))

    # --- QAT ---
    qat_enabled         = bool(int(os.environ.get("QAT_ENABLED", "0")))
    late_qat_threshold  = float(os.environ.get("LATE_QAT_THRESHOLD", 0.15))

    # --- [6] Hessian SDClip ---
    # std-based clipping for GPTQ: clip at (mean_abs + sdclip_k * row_std)
    # instead of a fixed percentile.  sdclip_k=0 disables (falls back to
    # percentile search).  Recommended range: 2.0–3.0.
    sdclip_k            = float(os.environ.get("SDCLIP_K", "2.5"))

    # --- [7] GPTQ embeddings ---
    # Quantise tok_emb / bigram embed / VE embed to int8 per-row before
    # saving, then dequantise for the int6 roundtrip eval.
    gptq_embed          = bool(int(os.environ.get("GPTQ_EMBED", "1")))

    # --- TTT ---
    ttt_enabled         = bool(int(os.environ.get("TTT_ENABLED", "0")))
    ttt_lr              = float(os.environ.get("TTT_LR", 0.002))
    ttt_epochs          = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens    = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_freeze_blocks   = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_momentum        = float(os.environ.get("TTT_MOMENTUM", 0.9))
    ttt_batch_seqs      = int(os.environ.get("TTT_BATCH_SEQS", 32))
    ttt_grad_clip       = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    # [8] "adam" or "sgd"
    ttt_optimizer       = os.environ.get("TTT_OPTIMIZER", "sgd")

    # --- Legacy compatibility aliases used by lab_runner.py ---
    warmdown_steps          = warmdown_iters
    weight_decay            = adam_wd
    grad_clip               = grad_clip_norm
    schedule_mode           = os.environ.get("SCHEDULE_MODE", "cosine")
    use_ema                 = bool(int(os.environ.get("USE_EMA", "1")))
    ema_decay               = float(os.environ.get("EMA_DECAY", 0.997))
    use_swa                 = swa_enabled
    swa_start               = int(os.environ.get("SWA_START", max(iterations - warmdown_iters, 0)))
    use_sliding_window_eval = bool(int(os.environ.get("USE_SLIDING_EVAL", "0")))
    use_depth_recurrence    = recurrence_loops > 1
    recurrence_interval     = int(os.environ.get("RECURRENCE_INTERVAL", 3))


def get_lr_schedule(step, total_steps, warmup_steps, warmdown_steps, mode="cosine", base_lr=1.0):
    """Legacy LR helper kept for lab_runner compatibility."""
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * max(step, 1) / warmup_steps
    if mode == "linear":
        return base_lr
    if warmdown_steps > 0:
        warmdown_start = max(total_steps - warmdown_steps, 0)
        if step >= warmdown_start:
            frac = max(total_steps - step, 0) / max(warmdown_steps, 1)
            return base_lr * frac
    main_steps = max(total_steps - warmup_steps - max(warmdown_steps, 0), 1)
    progress = min(max(step - warmup_steps, 0) / main_steps, 1.0)
    return base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))


class EMAState:
    """Lightweight EMA helper retained for older tooling."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].lerp_(p.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))


class SWAAccumulator:
    """Simple SWA accumulator retained for older tooling."""
    def __init__(self):
        self._sum: Optional[dict[str, Tensor]] = None
        self._count = 0

    def add(self, state_dict):
        snapshot = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        if self._sum is None:
            self._sum = snapshot
        else:
            for name, value in snapshot.items():
                self._sum[name].add_(value)
        self._count += 1

    def get_count(self):
        return self._count

    def get_state_dict(self):
        if self._sum is None or self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sum.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Newton-Schulz orthogonalization (batched)
# ──────────────────────────────────────────────────────────────────────────────
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Batched Newton-Schulz orthogonalization. G: (B,M,N) or (M,N)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


# ──────────────────────────────────────────────────────────────────────────────
# [5] MuonEq-R: Parallel Muon with equalized update magnitudes
# ──────────────────────────────────────────────────────────────────────────────
class Muon(torch.optim.Optimizer):
    """Parallel Muon with optional MuonEq-R equalization.

    MuonEq-R: after Newton-Schulz, the update for each shard is rescaled so
    that its Frobenius norm equals sqrt(M*N) — the same value an orthogonal
    matrix of that shape would have.  This removes the implicit per-shape LR
    difference caused by varying aspect ratios and ensures every weight matrix
    trains at exactly the same effective step size.
    """
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0,
                 muoneq_r: bool = False):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, weight_decay=weight_decay,
                 muoneq_r=muoneq_r),
        )
        self._built = False
        self._rs_futures: list[Any] = []

    def _build(self):
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size

        self._bank_meta = []
        for group in self.param_groups:
            for p in group["params"]:
                B = p.shape[0]
                padded_B = ((B + ws - 1) // ws) * ws
                shard_B = padded_B // ws
                tail = p.shape[1:]
                dev = p.device
                self._bank_meta.append({
                    'p': p,
                    'B': B,
                    'padded_grad': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'shard_mom': torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    'full_update': torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    # MuonEq-R: scale so ||update||_F = sqrt(M*N) for any shape
                    'eq_scale': math.sqrt(p.shape[-2] * p.shape[-1]) if p.ndim >= 2 else 1.0,
                    'scale': max(1, p.shape[-2] / p.shape[-1]) ** 0.5,  # legacy fallback
                })
        self._bank_meta.sort(key=lambda m: -m['p'].numel())
        self._built = True

    def launch_reduce_scatters(self):
        if not self._built:
            self._build()
        if not self._distributed:
            return
        self._rs_futures = []
        for m in self._bank_meta:
            p = m['p']
            if p.grad is None:
                self._rs_futures.append(None)
                continue
            pg = m['padded_grad']
            pg[:m['B']].copy_(p.grad.bfloat16())
            if pg.shape[0] > m['B']:
                pg[m['B']:].zero_()
            fut = dist.reduce_scatter_tensor(m['shard'], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> float:
        loss = 0.0
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self._built:
            self._build()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            do_eq = group.get("muoneq_r", False)

            prev_ag_handle: Any = None
            prev_m: Optional[dict[str, Any]] = None
            sharded = self._distributed and hasattr(self, '_rs_futures')

            for i, m in enumerate(self._bank_meta):
                p = m['p']
                if p.grad is None:
                    continue

                if prev_ag_handle is not None:
                    prev_ag_handle.wait()
                    assert prev_m is not None
                    pp = prev_m['p']
                    upd = prev_m['full_update'][:prev_m['B']]
                    if wd > 0.0:
                        pp.data.mul_(1.0 - lr * wd)
                    pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

                if sharded and self._rs_futures[i] is not None:
                    self._rs_futures[i].wait()
                    g = m['shard']
                    buf = m['shard_mom']
                else:
                    g = p.grad.bfloat16()
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, steps=backend_steps)

                # [5] MuonEq-R: normalize to unit Frobenius then re-scale
                if do_eq:
                    fro = update.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                    target = m['eq_scale']   # target Frobenius norm is sqrt(M*N)
                    update = update * (target / fro)

                if sharded:
                    prev_ag_handle = dist.all_gather_into_tensor(
                        m['full_update'], update, async_op=True)
                    prev_m = m
                else:
                    if wd > 0.0:
                        p.data.mul_(1.0 - lr * wd)
                    p.add_(update.to(dtype=p.dtype), alpha=-lr * m['scale'])

            if prev_ag_handle is not None:
                prev_ag_handle.wait()
                assert prev_m is not None
                pp = prev_m['p']
                upd = prev_m['full_update'][:prev_m['B']]
                if wd > 0.0:
                    pp.data.mul_(1.0 - lr * wd)
                pp.add_(upd.to(dtype=pp.dtype), alpha=-lr * prev_m['scale'])

            if hasattr(self, '_rs_futures'):
                del self._rs_futures

        return float(loss)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
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


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[:usable + 1]


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut,
             is_boundary_token_lut, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end   = (total_seqs * (rank + 1)) // world_size
    val_loss_sum   = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end   = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum   += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids  = y.reshape(-1)
            token_bytes  = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count,  op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token  = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# ──────────────────────────────────────────────────────────────────────────────
# [6] Quantization — Hessian SDClip + [7] GPTQ Embeddings
# ──────────────────────────────────────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,smear,dtg_gate,ve_layer_scales,"
        "ve_shared.scale,attn_gate,vr_lambda",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t): return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name, t, passthrough_orig_dtypes):
    if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped  = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale    = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if "bigram" in name and "embed" in name:
        return "embed"
    if "ve_shared" in name and "embed" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_int6_per_row(t: Tensor, clip_range: int = 31,
                           sdclip_k: float = 0.0) -> tuple[Tensor, Tensor]:
    """Int6 GPTQ-lite with optional Hessian SDClip.

    sdclip_k > 0  → std-based clipping: clip_abs = mean_abs + k * row_std
    sdclip_k == 0 → 5-percentile search (original GPTQ-lite behaviour)
    """
    t32 = t.float()
    if t32.ndim == 2:
        if sdclip_k > 0.0:
            # [6] Hessian SDClip: clip at mean_abs + k * std (per row)
            row_std      = t32.std(dim=1)
            row_mean_abs = t32.abs().mean(dim=1)
            clip_abs     = (row_mean_abs + sdclip_k * row_std).clamp_min(1e-8)
            s = (clip_abs / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
            return q, s
        else:
            # Original 5-percentile search
            best_q: Optional[Tensor] = None
            best_s: Optional[Tensor] = None
            best_err = float('inf')
            for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
                row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
                s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
                q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)
                err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
                if err < best_err:
                    best_q, best_s, best_err = q, s, err
            assert best_q is not None and best_s is not None
            return best_q, best_s
    # 1-D fallback
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8)
    return q, scale


def quantize_int8_per_row_embed(t: Tensor) -> tuple[Tensor, Tensor]:
    """[7] Per-row int8 for embedding tables."""
    t32 = t.float()
    row_max = t32.abs().amax(dim=1).clamp_min(1e-8)
    scale = (row_max / 127.0).to(torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _unbank_state_dict(sd, num_layers):
    out = {}
    n = num_layers
    for name, tensor in sd.items():
        if name == "qo_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_q.weight"]   = tensor[i]
                out[f"blocks.{i}.attn.proj.weight"]   = tensor[n + i]
        elif name == "kv_bank":
            for i in range(n):
                out[f"blocks.{i}.attn.c_k.weight"]   = tensor[i]
                out[f"blocks.{i}.attn.c_v.weight"]   = tensor[n + i]
        elif name == "mlp_up_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.fc.weight"]      = tensor[i]
        elif name == "mlp_down_bank":
            for i in range(n):
                out[f"blocks.{i}.mlp.proj.weight"]    = tensor[i]
        else:
            out[name] = tensor
    return out


def _rebank_state_dict(sd, num_layers, template_sd):
    out = {}
    n = num_layers
    qo_slices = [None] * (2 * n)
    kv_slices = [None] * (2 * n)
    up_slices = [None] * n
    down_slices = [None] * n
    consumed = set()
    for i in range(n):
        for key, store, idx in [
            (f"blocks.{i}.attn.c_q.weight",  qo_slices,   i),
            (f"blocks.{i}.attn.proj.weight",  qo_slices,   n + i),
            (f"blocks.{i}.attn.c_k.weight",  kv_slices,   i),
            (f"blocks.{i}.attn.c_v.weight",  kv_slices,   n + i),
            (f"blocks.{i}.mlp.fc.weight",    up_slices,   i),
            (f"blocks.{i}.mlp.proj.weight",  down_slices, i),
        ]:
            if key in sd:
                store[idx] = sd[key]
                consumed.add(key)
    out["qo_bank"]       = torch.stack(qo_slices).to(dtype=template_sd["qo_bank"].dtype)
    out["kv_bank"]       = torch.stack(kv_slices).to(dtype=template_sd["kv_bank"].dtype)
    out["mlp_up_bank"]   = torch.stack(up_slices).to(dtype=template_sd["mlp_up_bank"].dtype)
    out["mlp_down_bank"] = torch.stack(down_slices).to(dtype=template_sd["mlp_down_bank"].dtype)
    for name, tensor in sd.items():
        if name not in consumed:
            out[name] = tensor
    return out


def mixed_quantize_int6(state_dict, int6_cats, sdclip_k=0.0, gptq_embed=False):
    """Quantise model weights to int6 (attn+mlp) with optional GPTQ embeddings.

    [6] sdclip_k > 0  → Hessian SDClip instead of percentile search
    [7] gptq_embed     → int8 per-row for embedding matrices
    """
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name]   = "passthrough"
            continue
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name]   = "passthrough_ctrl"
            continue
        # [7] GPTQ embeddings: int8 per-row for embedding tables
        if gptq_embed and cat == "embed" and t.ndim == 2:
            q, s = quantize_int8_per_row_embed(t)
            result[name + ".q"]     = q
            result[name + ".scale"] = s
            meta[name]              = {"type": "int8_embed"}
            continue
        if cat in int6_cats and t.ndim >= 1:
            q, s = quantize_int6_per_row(t, sdclip_k=sdclip_k)
            result[name + ".q"]     = q
            result[name + ".scale"] = s
            meta[name]              = {"type": "int6"}
        else:
            q, s = quantize_float_tensor(t)
            result[name + ".q"]     = q
            result[name + ".scale"] = s
            meta[name]              = {"type": "int8"}
    return result, meta


def dequantize_mixed_int6(result, meta, template_sd):
    out = {}
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


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes  = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
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
        self.files   = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens   = load_data_shard(self.files[0])
        self.pos      = 0
    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size; self.device = device
        self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens   = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span  = local_tokens + 1
        chunk          = self.stream.take(per_rank_span * self.world_size)
        start          = self.rank * per_rank_span
        local          = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


class DataLoader:
    """Compatibility loader used by lab_runner.py.

    This intentionally keeps the older surface area alive without affecting the
    newer distributed training path below.
    """
    def __init__(self, pattern, seq_len, device):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.seq_len = seq_len
        self.device = device
        self.eval_offset = 0
        self._train_tokens = None
        self._eval_tokens = None
        self._train_file_idx = 0
        self._eval_file_idx = 0
        self._load_train_file(0)
        self._load_eval_file(0)

    def _load_train_file(self, index):
        self._train_file_idx = index % len(self.files)
        self._train_tokens = load_data_shard(self.files[self._train_file_idx]).contiguous()
        assert self._train_tokens is not None
        if self._train_tokens.numel() <= self.seq_len:
            raise ValueError(f"Shard too short for seq_len={self.seq_len}: {self.files[self._train_file_idx]}")

    def _load_eval_file(self, index):
        self._eval_file_idx = index % len(self.files)
        self._eval_tokens = load_data_shard(self.files[self._eval_file_idx]).contiguous()
        assert self._eval_tokens is not None
        if self._eval_tokens.numel() <= self.seq_len:
            raise ValueError(f"Shard too short for seq_len={self.seq_len}: {self.files[self._eval_file_idx]}")
        self.eval_offset = 0

    def get_batch(self, batch_size=1):
        assert self._train_tokens is not None
        need = self.seq_len + 1
        max_start = self._train_tokens.numel() - need
        if max_start <= 0:
            self._load_train_file(self._train_file_idx + 1)
            assert self._train_tokens is not None
            max_start = self._train_tokens.numel() - need
        starts = torch.randint(0, max_start + 1, (batch_size,))
        xs, ys = [], []
        for start in starts.tolist():
            chunk = self._train_tokens[start:start + need]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.stack(xs).to(self.device, dtype=torch.int64, non_blocking=True)
        y = torch.stack(ys).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y

    def get_eval_batch_sliding(self, stride=1):
        assert self._eval_tokens is not None
        need = self.seq_len + 1
        if self.eval_offset + need > self._eval_tokens.numel():
            self._load_eval_file(self._eval_file_idx + 1)
            assert self._eval_tokens is not None
        start = self.eval_offset
        chunk = self._eval_tokens[start:start + need]
        if chunk.numel() < need:
            self._load_eval_file(self._eval_file_idx + 1)
            assert self._eval_tokens is not None
            start = self.eval_offset
            chunk = self._eval_tokens[start:start + need]
        self.eval_offset += max(1, stride)
        x = chunk[:-1].unsqueeze(0).to(self.device, dtype=torch.int64, non_blocking=True)
        y = chunk[1:].unsqueeze(0).to(self.device, dtype=torch.int64, non_blocking=True)
        return x, y, start


# ──────────────────────────────────────────────────────────────────────────────
# Model modules
# ──────────────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, input: Tensor) -> Tensor:
        w = self.weight.to(input.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32   = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q   = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(input.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, w, bias)


def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) \
                    and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale    = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = cast(Tensor, self.inv_freq).to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope  = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                 gated_attention=False, value_residual=False):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.q_gain       = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims    = 0
        self.rotary       = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)
        self.use_xsa      = False
        self.gated_attention = gated_attention
        if gated_attention:
            self.attn_gate = nn.Linear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)
        self.value_residual = value_residual
        if value_residual:
            self.vr_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        B, T, H, D = y.shape
        Hkv   = v.size(-2)
        group = H // Hkv
        y_g   = y.reshape(B, T, Hkv, group, D)
        vn    = F.normalize(v, dim=-1).unsqueeze(-2)
        proj  = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(
        self,
        x: Tensor,
        q_w: Tensor,
        k_w: Tensor,
        v_w: Tensor,
        out_w: Tensor,
        v_embed: Optional[Tensor] = None,
        v0: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, q_w.to(x.dtype)).reshape(bsz, seqlen, self.num_heads,    self.head_dim)
        k = F.linear(x, k_w.to(x.dtype)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = F.linear(x, v_w.to(x.dtype))
        if v_embed is not None:
            v = v + v_embed
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        raw_v = v if self.value_residual else None
        if self.value_residual and v0 is not None:
            lam = self.vr_lambda.to(dtype=v.dtype)
            v   = lam[0] * v0 + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        if self.gated_attention:
            gate = torch.sigmoid(self.attn_gate(x)).unsqueeze(-1)
            y    = y * gate
        y = y.reshape(bsz, seqlen, dim)
        return F.linear(y, out_w.to(x.dtype)), raw_v


class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x):
        g      = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj  = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens):
        t   = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0]  = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size, ve_dim, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj  = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids):
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
    def forward(self, x, up_w, down_w):
        # LeakyReLU(0.5)² activation — confirmed best on leaderboard
        x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
        return F.linear(x.square(), down_w.to(x.dtype))


class Block(nn.Module):
    """Transformer block with optional [3] Parallel Residuals."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base,
                 qk_gain_init, layer_idx=0, ln_scale=False, dtg=False,
                 gated_attention=False, value_residual=False,
                 parallel_residual=False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base,
                                        qk_gain_init, gated_attention=gated_attention,
                                        value_residual=value_residual)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim,  dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        # [3] Parallel residual flag — stored for forward()
        self.parallel_residual = parallel_residual
        if dtg:
            self.dtg_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.dtg_gate.weight)
            nn.init.constant_(self.dtg_gate.bias, 2.0)
        else:
            self.dtg_gate = None

    def forward(self, x, x0, q_w, k_w, v_w, out_w, up_w, down_w,
                v_embed=None, v0=None):
        mix  = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        lsf  = self.ln_scale_factor

        if self.parallel_residual:
            # [3] Both streams read from the same normed input simultaneously.
            # This lets attention and MLP specialise independently and trains
            # ~2% faster because the MLP norm doesn't wait for attn output.
            h = self.attn_norm(x_in) * lsf
            attn_out, raw_v = self.attn(h, q_w, k_w, v_w, out_w, v_embed=v_embed, v0=v0)
            mlp_out = self.mlp(self.mlp_norm(x_in) * lsf, up_w, down_w)
            x_out = (x_in
                     + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out
                     + self.mlp_scale.to(x_in.dtype)[None, None, :] * mlp_out)
        else:
            # Original sequential path
            attn_out, raw_v = self.attn(self.attn_norm(x_in) * lsf, q_w, k_w, v_w, out_w,
                                        v_embed=v_embed, v0=v0)
            x_out = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out
            x_out = x_out + self.mlp_scale.to(x_out.dtype)[None, None, :] \
                    * self.mlp(self.mlp_norm(x_out) * lsf, up_w, down_w)

        if self.dtg_gate is not None:
            gate  = torch.sigmoid(self.dtg_gate(x_in.detach()))
            x_out = x_in + gate * (x_out - x_in)

        return x_out, raw_v


class GPT(nn.Module):
    """GPT with [2] Depth Recurrence and all stacked innovations."""
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init, mtp_num_heads=0, mtp_loss_weight=0.1,
                 bigram_vocab_size=0, bigram_dim=128, xsa_last_n=0, rope_dims=0,
                 ln_scale=False, dtg=False, ve_enabled=False, ve_dim=128,
                 ve_layers="9,10", gated_attention=False, value_residual=False,
                 parallel_residual=False,
                 recurrence_layer_indices=None, recurrence_loops=1):
        super().__init__()
        self._ve_target_dim = num_kv_heads * (model_dim // num_heads)
        self.tie_embeddings     = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap      = logit_softcap
        self.value_residual     = value_residual
        self.mtp_num_heads      = mtp_num_heads
        self.mtp_loss_weight    = mtp_loss_weight

        # [2] Depth Recurrence state
        self.recurrence_layer_set = set(recurrence_layer_indices or [])
        self.recurrence_loops     = max(1, recurrence_loops)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram  = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear   = SmearGate(model_dim)

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights       = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        # Parameter banks (batched Newton-Schulz)
        head_dim = model_dim // num_heads
        kv_dim   = num_kv_heads * head_dim
        mlp_dim  = int(mlp_mult * model_dim)
        self.num_layers   = num_layers
        self.qo_bank      = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.kv_bank      = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.mlp_up_bank  = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
        self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  layer_idx=i, ln_scale=ln_scale, dtg=dtg,
                  gated_attention=gated_attention, value_residual=value_residual,
                  parallel_residual=parallel_residual)
            for i in range(num_layers)
        ])
        blocks = cast(list[Block], list(self.blocks))

        # Partial RoPE
        if rope_dims > 0:
            for block in blocks:
                block.attn.rope_dims = rope_dims
                block.attn.rotary    = Rotary(head_dim, base=rope_base, train_seq_len=1024, rope_dims=rope_dims)

        # Value embeddings
        self.ve_layer_indices = [int(x) for x in ve_layers.split(",") if x.strip()] if ve_enabled else []
        if self.ve_layer_indices:
            kv_dim_ve = self._ve_target_dim
            self.ve_shared      = ValueEmbedding(vocab_size, ve_dim, kv_dim_ve)
            self.ve_layer_scales = nn.ParameterList(
                [nn.Parameter(torch.ones(1, dtype=torch.float32)) for _ in self.ve_layer_indices])
        else:
            self.ve_shared      = None
            self.ve_layer_scales = nn.ParameterList()
        self.value_embeds = nn.ModuleList()

        # XSA on last N layers
        if xsa_last_n > 0:
            for i in range(max(0, num_layers - xsa_last_n), num_layers):
                blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head    = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.mtp_heads = nn.ModuleList(
            [CastedLinear(model_dim, vocab_size, bias=False) for _ in range(mtp_num_heads)])
        for head in self.mtp_heads:
            head._zero_init = True
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        n = self.num_layers
        proj_scale = 1.0 / math.sqrt(2 * n)
        for i in range(n):
            nn.init.orthogonal_(self.qo_bank.data[i],     gain=1.0)
            nn.init.zeros_(self.qo_bank.data[n + i])
            nn.init.orthogonal_(self.kv_bank.data[i],     gain=1.0)
            nn.init.orthogonal_(self.kv_bank.data[n + i], gain=1.0)
            nn.init.orthogonal_(self.mlp_up_bank.data[i], gain=1.0)
            nn.init.zeros_(self.mlp_down_bank.data[i])
            self.qo_bank.data[n + i].mul_(proj_scale)
            self.mlp_down_bank.data[i].mul_(proj_scale)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _get_ve(
        self,
        layer_idx: int,
        input_ids: Tensor,
        ve_cache: dict[str, Tensor],
    ) -> Optional[Tensor]:
        if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
            return None
        if 've' not in ve_cache:
            ve_cache['ve'] = self.ve_shared(input_ids)
        ve_base = ve_cache['ve']
        ve_idx  = self.ve_layer_indices.index(layer_idx)
        return ve_base * self.ve_layer_scales[ve_idx].to(dtype=ve_base.dtype)

    def _run_block(
        self,
        i: int,
        x: Tensor,
        x0: Tensor,
        v0: Optional[Tensor],
        input_ids: Tensor,
        ve_cache: dict[str, Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Run block i once, handling bank indexing."""
        n  = self.num_layers
        ve = self._get_ve(i, input_ids, ve_cache)
        block = cast(Block, self.blocks[i])
        x, raw_v = block(
            x, x0,
            self.qo_bank[i],      self.kv_bank[i],
            self.kv_bank[n + i],  self.qo_bank[n + i],
            self.mlp_up_bank[i],  self.mlp_down_bank[i],
            v_embed=ve, v0=v0,
        )
        return x, raw_v

    def _forward_body(self, input_ids: Tensor) -> Tensor:
        """Shared encoder-decoder body for forward() and forward_logits()."""
        x    = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x    = F.rms_norm(x, (x.size(-1),))
        x    = self.smear(x)
        x0   = x
        v0   = None
        skips: list[Tensor] = []
        ve_cache: dict[str, Tensor] = {}

        # ── Encoder ──────────────────────────────────────────────────────────
        for i in range(self.num_encoder_layers):
            x, raw_v = self._run_block(i, x, x0, v0, input_ids, ve_cache)
            if v0 is None and raw_v is not None:
                v0 = raw_v
            # [2] Depth Recurrence: run extra loops (weights shared — no extra params)
            if i in self.recurrence_layer_set and self.recurrence_loops > 1:
                for _ in range(self.recurrence_loops - 1):
                    x, _ = self._run_block(i, x, x0, v0, input_ids, ve_cache)
            skips.append(x)

        # ── Decoder ──────────────────────────────────────────────────────────
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, _ = self._run_block(bi, x, x0, v0, input_ids, ve_cache)
            # [2] Depth Recurrence in decoder blocks
            if bi in self.recurrence_layer_set and self.recurrence_loops > 1:
                for _ in range(self.recurrence_loops - 1):
                    x, _ = self._run_block(bi, x, x0, v0, input_ids, ve_cache)

        return self.final_norm(x)

    def _logits(self, x: Tensor) -> Tensor:
        x_flat = x.reshape(-1, x.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            assert self.lm_head is not None
            logits_proj = self.lm_head(x_flat)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x       = self._forward_body(input_ids)
        logits  = self._logits(x)
        targets = target_ids.reshape(-1)
        main_loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if self.training and self.mtp_num_heads > 0 and self.mtp_loss_weight > 0.0:
            _, seqlen, dim = x.shape
            mtp_loss_sum = x.new_zeros(())
            count = 0
            for k, mtp_head in enumerate(self.mtp_heads):
                valid_t = seqlen - (k + 1)
                if valid_t <= 0:
                    continue
                mtp_h = x[:, :valid_t, :].reshape(-1, dim)
                mtp_t = target_ids[:, k + 1:].reshape(-1)
                mtp_l = self.logit_softcap * torch.tanh(mtp_head(mtp_h) / self.logit_softcap)
                mtp_loss_sum = mtp_loss_sum + F.cross_entropy(mtp_l.float(), mtp_t, reduction="mean")
                count += 1
            if count > 0:
                main_loss = main_loss + self.mtp_loss_weight * (mtp_loss_sum / count)
        return main_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._forward_body(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            assert self.lm_head is not None
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# ──────────────────────────────────────────────────────────────────────────────
# Sliding-window evaluation
# ──────────────────────────────────────────────────────────────────────────────
def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens  = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byt_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)

    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end  = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s    = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum  += nll[i, s:wlen].to(torch.float64).sum()
                tok_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb  = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byt_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, tok_count, byt_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / tok_count).item()
    base_model.train()
    return val_loss, val_loss / math.log(2.0) * (tok_count.item() / byt_count.item())


# ──────────────────────────────────────────────────────────────────────────────
# [8] Legal score-first TTT with Adam option
# ──────────────────────────────────────────────────────────────────────────────
def eval_val_sliding_ttt(args, base_model, rank, world_size, device, val_tokens,
                         base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                         stride, batch_seqs=32, log0=print):
    """Score-first TTT: each chunk is scored under inference_mode (stateless)
    then trained on with SGD or Adam ([8]).  Adam converges faster per chunk
    so more adaptation happens within the fixed time budget."""
    seq_len      = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    ttt_chunk    = args.ttt_chunk_tokens

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride or ws == 0]
    num_chunks    = (total_tokens + ttt_chunk - 1) // ttt_chunk
    chunk_windows: list[list[int]] = [[] for _ in range(num_chunks)]
    for ws in window_starts:
        end  = min(ws + seq_len, total_tokens)
        wlen = end - ws
        s    = 0 if ws == 0 else max(wlen - stride, 0)
        ci   = min((ws + s) // ttt_chunk, num_chunks - 1)
        chunk_windows[ci].append(ws)

    frozen_ids = set(range(min(args.ttt_freeze_blocks, len(base_model.blocks))))
    ttt_params = []
    for name, p in base_model.named_parameters():
        frozen = any(f"blocks.{bi}." in name for bi in frozen_ids)
        p.requires_grad_(not frozen)
        if not frozen:
            ttt_params.append(p)

    use_adam = args.ttt_optimizer.lower() == "adam"
    if use_adam:
        # [8] Adam for TTT: converges ~2× faster than SGD per chunk, allowing
        #     more effective adaptation within the ~400s TTT time budget.
        optimizer = torch.optim.Adam(ttt_params, lr=args.ttt_lr, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    log0(f"ttt:start optimizer={args.ttt_optimizer} chunks={num_chunks} "
         f"chunk_tokens={ttt_chunk} lr={args.ttt_lr} epochs={args.ttt_epochs} "
         f"freeze={args.ttt_freeze_blocks}")

    loss_sum  = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byt_count = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    for ci in range(num_chunks):
        windows = chunk_windows[ci]
        if not windows:
            continue
        chunk_start = ci * ttt_chunk
        chunk_end   = min((ci + 1) * ttt_chunk, total_tokens)
        my_s = (len(windows) * rank) // world_size
        my_e = (len(windows) * (rank + 1)) // world_size
        my_windows = windows[my_s:my_e]

        # Phase 1: SCORE (stateless)
        base_model.eval()
        with torch.inference_mode():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_b = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end  = min(ws + seq_len, total_tokens)
                    wlen = end - ws
                    wlens.append(wlen)
                    tok  = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                    x_b[i, :wlen] = tok[:-1]; y_b[i, :wlen] = tok[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_b)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_b.reshape(-1), reduction="none").reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s    = 0 if ws == 0 else max(wlen - stride, 0)
                    loss_sum  += nll[i, s:wlen].to(torch.float64).sum()
                    tok_count += float(wlen - s)
                    tgt = y_b[i, s:wlen]; prev = x_b[i, s:wlen]
                    tb  = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byt_count += tb.sum()

        # Phase 2: TRAIN on the already-scored chunk
        is_last = (ci == num_chunks - 1)
        if not is_last and args.ttt_epochs > 0:
            base_model.train()
            chunk_seqs = (chunk_end - chunk_start) // seq_len
            if chunk_seqs > 0:
                # Cosine LR decay across chunks (warm → cool)
                cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
                for pg in optimizer.param_groups:
                    pg['lr'] = cos_lr
                my_seq_s = (chunk_seqs * rank) // world_size
                my_seq_e = (chunk_seqs * (rank + 1)) // world_size
                my_n = my_seq_e - my_seq_s
                for _ep in range(args.ttt_epochs):
                    for bs in range(0, my_n, args.ttt_batch_seqs):
                        be  = min(bs + args.ttt_batch_seqs, my_n)
                        tok_start = chunk_start + (my_seq_s + bs) * seq_len
                        tok_end   = chunk_start + (my_seq_s + be) * seq_len + 1
                        if tok_end > val_tokens.numel():
                            continue
                        local = val_tokens[tok_start:tok_end].to(device=device, dtype=torch.int64)
                        xb = local[:-1].reshape(-1, seq_len)
                        yb = local[1:].reshape(-1, seq_len)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = base_model(xb, yb)
                        loss.backward()
                        if world_size > 1:
                            for p in ttt_params:
                                if p.grad is not None:
                                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.ttt_grad_clip)
                        optimizer.step()

        if rank == 0 and (ci % 10 == 0 or ci == num_chunks - 1):
            elapsed = time.perf_counter() - t0
            rl   = loss_sum.item() / max(tok_count.item(), 1)
            rbpb = rl / math.log(2.0) * (tok_count.item() / max(byt_count.item(), 1))
            log0(f"  ttt [{ci+1}/{num_chunks}] bpb={rbpb:.6f} t={elapsed:.1f}s")

    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, tok_count, byt_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    for p in base_model.parameters():
        p.requires_grad_(True)
    base_model.eval()
    val_loss = (loss_sum / tok_count).item()
    val_bpb  = val_loss / math.log(2.0) * (tok_count.item() / byt_count.item())
    log0(f"ttt:done val_loss={val_loss:.6f} val_bpb={val_bpb:.6f} "
         f"elapsed={time.perf_counter() - t0:.1f}s")
    return val_loss, val_bpb


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────
def make_model(args, device):
    recurrence_layer_indices = [int(x) for x in args.recurrence_layers.split(",") if x.strip()] \
        if args.recurrence_layers else []
    model = GPT(
        vocab_size          = args.vocab_size,
        num_layers          = args.num_layers,
        model_dim           = args.model_dim,
        num_heads           = args.num_heads,
        num_kv_heads        = args.num_kv_heads,
        mlp_mult            = args.mlp_mult,
        tie_embeddings      = args.tie_embeddings,
        tied_embed_init_std = args.tied_embed_init_std,
        logit_softcap       = args.logit_softcap,
        rope_base           = args.rope_base,
        qk_gain_init        = args.qk_gain_init,        # [4] 5.0
        mtp_num_heads       = args.mtp_num_heads,
        mtp_loss_weight     = args.mtp_loss_weight,
        bigram_vocab_size   = args.bigram_vocab_size,
        bigram_dim          = args.bigram_dim,
        xsa_last_n          = args.xsa_last_n,
        rope_dims           = args.rope_dims,
        ln_scale            = args.ln_scale,
        dtg                 = args.dtg_enabled,
        ve_enabled          = args.ve_enabled,
        ve_dim              = args.ve_dim,
        ve_layers           = args.ve_layers,
        gated_attention     = args.gated_attention,
        value_residual      = args.value_residual,
        parallel_residual   = args.parallel_residuals,  # [3]
        recurrence_layer_indices = recurrence_layer_indices,  # [2]
        recurrence_loops    = args.recurrence_loops,    # [2]
    ).to(device).bfloat16()
    # Banks stay FP32; cast to BF16 in forward via bank indexing
    for bank in [model.qo_bank, model.kv_bank, model.mlp_up_bank, model.mlp_down_bank]:
        bank.data = bank.data.float()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model


def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
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
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
    enable_cudnn_sdp(False); enable_flash_sdp(True)
    enable_mem_efficient_sdp(False); enable_math_sdp(False)

    logfile: Optional[str] = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer {int(sp.vocab_size())}")

    dataset_dir       = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens  = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = \
        build_sentencepiece_luts(sp, args.vocab_size, device)

    log0(f"vocab:{args.vocab_size} tokenizer:{args.tokenizer_path}")
    log0(f"train shards:{actual_train_files} val_tokens:{val_tokens.numel()-1}")
    log0(f"recurrence_layers:{args.recurrence_layers} loops:{args.recurrence_loops}")
    log0(f"parallel_residuals:{args.parallel_residuals} qk_gain:{args.qk_gain_init}")
    log0(f"muoneq_r:{args.muoneq_r} sdclip_k:{args.sdclip_k} gptq_embed:{args.gptq_embed}")

    CastedLinear._qat_enabled = args.qat_enabled
    base_model     = make_model(args, device)
    compiled_model = cast(nn.Module, torch.compile(base_model, dynamic=False, fullgraph=True))
    model          = compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")

    # ── Optimizers ────────────────────────────────────────────────────────────
    matrix_params = [base_model.qo_bank, base_model.kv_bank,
                     base_model.mlp_up_bank, base_model.mlp_down_bank]

    block_named_params = list(base_model.blocks.named_parameters())
    scalar_params: list[nn.Parameter] = [
        p for name, p in block_named_params
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(cast(nn.Parameter, base_model.skip_weights))
    scalar_params.append(base_model.smear.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr  = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_params: list[dict[str, Any]] = [
        {"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}
    ]
    if base_model.bigram is not None:
        tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            scalar_params.append(cast(nn.Parameter, base_model.bigram.proj.weight))
    if base_model.ve_shared is not None:
        tok_params.append({"params": [base_model.ve_shared.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.ve_shared.proj is not None:
            scalar_params.append(cast(nn.Parameter, base_model.ve_shared.proj.weight))
        scalar_params.append(base_model.ve_shared.scale)
        for s in base_model.ve_layer_scales:
            scalar_params.append(s)

    optimizer_tok = torch.optim.AdamW(
        tok_params, betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(
        matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd,
        muoneq_r=args.muoneq_r)                                          # [5]
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=args.adam_wd, fused=True)

    optimizer_head: Optional[torch.optim.Optimizer] = None
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)

    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if optimizer_head is not None:
        optimizers.append(optimizer_head)

    replicated_params: list[nn.Parameter] = list(optimizer_tok.param_groups[0]["params"])
    for pg in optimizer_tok.param_groups[1:]:
        replicated_params.extend(cast(list[nn.Parameter], pg["params"]))
    replicated_params.extend(scalar_params)
    if optimizer_head is not None:
        assert base_model.lm_head is not None
        replicated_params.append(base_model.lm_head.weight)

    # ── Training loop ─────────────────────────────────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) \
                if warmdown_start <= step < args.iterations else 1.0
        step_ms       = elapsed_ms / max(step, 1)
        warmdown_ms   = args.warmdown_iters * step_ms
        remaining_ms  = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup (run a few steps then reset, so optimizer state is warm)
    if args.warmup_steps > 0:
        init_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt_states  = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for _ in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            if distributed:
                for p in base_model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
        base_model.load_state_dict(init_model_state, strict=True)
        for opt, st in zip(optimizers, init_opt_states):
            opt.load_state_dict(st)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    swa_state, swa_count = None, 0
    from collections import deque
    lawa_queue: deque = deque(maxlen=args.lawa_k)
    ema_state  = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
    ema_decay  = 0.997

    training_time_ms = 0.0
    stop_after_step  = None
    torch.cuda.synchronize()
    t0   = time.perf_counter()
    step = 0

    while True:
        last_step      = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early step:{step}/{args.iterations} time:{training_time_ms:.0f}ms")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale      = lr_mul(step, elapsed_ms)

        if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
            CastedLinear._qat_enabled = True
            log0(f"late_qat:enabled step:{step} scale:{scale:.4f}")

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_mom

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        # Three-phase overlapped optimizer step (banks via Parallel Muon)
        optimizer_muon.launch_reduce_scatters()
        if distributed:
            for p in replicated_params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer_tok.step(); optimizer_scalar.step()
        if optimizer_head is not None:
            optimizer_head.step()
        optimizer_muon.step()
        zero_grad_all()

        # EMA
        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        if args.swa_enabled and scale < 0.2 and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1
        if args.lawa_enabled and step % args.lawa_freq == 0:
            lawa_queue.append({n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()})

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rc = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rc, op=dist.ReduceOp.MAX)
            reached_cap = bool(rc.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    # Apply weight averaging
    if args.lawa_enabled and len(lawa_queue) > 1:
        log0(f"lawa:applying k={len(lawa_queue)}")
        cur = base_model.state_dict()
        avg = {n: torch.zeros(t.shape, dtype=torch.float32) for n, t in cur.items()}
        for snap in lawa_queue:
            for n in avg:
                avg[n] += snap[n].float()
        for n in avg:
            avg[n] /= len(lawa_queue)
            avg[n]  = avg[n].to(dtype=cur[n].dtype)
        base_model.load_state_dict(avg, strict=True)
    else:
        log0("ema:applying")
        cur     = base_model.state_dict()
        avg_ema = {n: t.to(dtype=cur[n].dtype) for n, t in ema_state.items()}
        base_model.load_state_dict(avg_ema, strict=True)

    # Diagnostic eval after EMA
    torch.cuda.synchronize(); t_diag = time.perf_counter()
    dl, dbpb = eval_val(args, compiled_model, rank, world_size, device, grad_accum_steps,
                        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"DIAGNOSTIC post_ema val_loss:{dl:.4f} val_bpb:{dbpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_diag):.0f}ms")

    # Save full model
    full_sd   = base_model.state_dict()
    export_sd = {k: v for k, v in full_sd.items() if "mtp_heads" not in k}
    if master_process:
        torch.save(export_sd, "final_model.pt")
        log0(f"model_bytes:{os.path.getsize('final_model.pt')}")

    # Quantise + compress
    sd_cpu     = {k: v.detach().cpu() for k, v in export_sd.items()}
    unbanked   = _unbank_state_dict(sd_cpu, args.num_layers)

    # [6] Hessian SDClip + [7] GPTQ Embeddings
    quant_result, quant_meta = mixed_quantize_int6(
        unbanked,
        int6_cats={"mlp", "attn"},
        sdclip_k=args.sdclip_k,
        gptq_embed=args.gptq_embed,
    )
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = lzma.compress(quant_buf.getvalue(), preset=6)

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        log0(f"int6+lzma bytes:{len(quant_blob)} code_bytes:{len(code.encode())}")
        log0(f"total_submission_bytes:{len(quant_blob)+len(code.encode('utf-8'))}")

    if distributed:
        dist.barrier()

    # Roundtrip eval on dequantised model
    with open("final_model.int6.ptz", "rb") as f:
        blob_disk = f.read()
    qs = torch.load(io.BytesIO(lzma.decompress(blob_disk)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(qs["w"], qs["m"], unbanked)
    deq_state    = _rebank_state_dict(deq_unbanked, args.num_layers, sd_cpu)

    eval_model = make_model(args, device)
    eval_model.mtp_num_heads = 0
    eval_model.load_state_dict(deq_state, strict=True)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)

    torch.cuda.synchronize(); t_q = time.perf_counter()
    ql, qbpb = eval_val(args, compiled_eval, rank, world_size, device, grad_accum_steps,
                        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                        eval_seq_len=effective_eval_seq_len)
    log0(f"final_int6_roundtrip val_loss:{ql:.4f} val_bpb:{qbpb:.4f} "
         f"eval_time:{1000*(time.perf_counter()-t_q):.0f}ms")

    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize(); t_sw = time.perf_counter()
        swl, swbpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, eval_seq_len=sw_seq_len)
        log0(f"final_int6_sliding_window val_loss:{swl:.4f} val_bpb:{swbpb:.4f} "
             f"stride:{args.eval_stride} eval_time:{1000*(time.perf_counter()-t_sw):.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{swl:.8f} val_bpb:{swbpb:.8f}")

    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize(); t_s64 = time.perf_counter()
        s64l, s64bpb = eval_val_sliding(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=64, eval_seq_len=sw_seq_len)
        log0(f"final_int6_sliding_window_s64 val_loss:{s64l:.4f} val_bpb:{s64bpb:.4f} "
             f"stride:64 eval_time:{1000*(time.perf_counter()-t_s64):.0f}ms")

    # Legal score-first TTT
    if args.ttt_enabled:
        torch.cuda.synchronize(); t_ttt = time.perf_counter()
        tl, tbpb = eval_val_sliding_ttt(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, log0=log0)
        log0(f"legal_ttt val_loss:{tl:.4f} val_bpb:{tbpb:.4f} "
             f"eval_time:{1000*(time.perf_counter()-t_ttt):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
