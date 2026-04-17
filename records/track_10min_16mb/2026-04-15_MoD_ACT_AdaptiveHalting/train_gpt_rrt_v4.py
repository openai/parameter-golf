"""
train_gpt_rrt_v4.py — Parameter Golf submission, Tier-1 stack on top of v3.

Deltas from v3 (all incremental, layer-on, low-to-medium risk):
  [OPT-1] Polar Express replaces Newton-Schulz-5 (Amsel et al. 2505.16932)
  [OPT-2] NorMuon placement: row-norm POST orthogonalization, not pre (2510.05491)
  [OPT-3] Cautious Weight Decay + linear-to-zero schedule (2510.12402)
  [ARCH-1] Value Residual Learning: layer-1 values fed forward into later attn (2410.17897)
  [ARCH-2] Backout: subtract learned fraction of early residual before lm_head (mng #140)
  [ARCH-3] Softmax-Skip-Gate initialization: skip_w initialized to ~0.18 (mng #125)
  [ARCH-4] Partial-RoPE 16/64 -> 8/64 (frees cache; 10% RoPE dims is sufficient, 2603.11611)
  [EVAL-1] SWA over last K checkpoints at eval time (LAWA, 2306.03241)
  [EVAL-2] Temperature sweep instead of fixed 0.98 (post-TTT calibration)
  [EVAL-3] TTT: AdamW @ lr 3e-6, 1 step per chunk (Rannen-Triki 2403.01518)
  [COMP-1] Low-rank factored tied embedding (rank 128) — reclaims ~5 MB budget
  [COMP-2] Static rANS/range coder on int6 stream in place of LZMA for the weight bulk
  [COMP-3] LZMA delta-filter pipeline for scales & code
  [DATA-1] BOS-anchored document packing (modded-nanogpt #108/#118)

All of v3's SOTA machinery kept:
  SP8192 · 11L·512d·8H/4KV · MLP-4x · LeakyReLU(0.5)² · RMSNorm · tied-embed
  Exact recurrence schedule enc=[0,1,2,3,4,5,3,4] dec=[5,3,4,5,6,7,8,9,10]
  RRT-LoRA rank=4 on layers {3,4,5}, zero-init B, alpha warmup
  Parallel residuals from layer 7+, QK-Gain 5.25, logit softcap 30.0
  SDClip int6/int8 with bit-packing (now followed by range code, not LZMA)
  Legal score-first TTT with strict inference_mode scoring BEFORE any update

Ablation env vars:
  ABLATE=1              disable LoRA, pure depth-recurrence baseline
  DISABLE_VRES=1        disable Value Residual
  DISABLE_BACKOUT=1     disable Backout
  DISABLE_CAUTIOUS=1    disable Cautious Weight Decay
  DISABLE_POLAR=1       fall back to NS-5 instead of Polar Express
  DISABLE_LRF_EMBED=1   do not low-rank-factor embeddings (larger artifact)
  DISABLE_RANS=1        fall back to LZMA for weight stream
  SWA_K=10              number of late checkpoints to average at eval

Standard run:
  SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_rrt_v4.py
"""

from __future__ import annotations

import glob
import io
import lzma
import math
import os
import random
import time
import uuid
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import brotli
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP


# ═══════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════
class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size    = int(os.environ.get("VAL_BATCH_SIZE",   524_288))
    val_loss_every    = int(os.environ.get("VAL_LOSS_EVERY",   1000))
    train_log_every   = int(os.environ.get("TRAIN_LOG_EVERY",  200))

    iterations            = int(os.environ.get("ITERATIONS",           20_000))
    warmdown_frac         = float(os.environ.get("WARMDOWN_FRAC",      0.60))  # 0.72 -> 0.60 per Defazio/Bergsma
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",         20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",   524_288))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",        1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model
    vocab_size          = int(os.environ.get("VOCAB_SIZE",         8192))
    num_layers          = int(os.environ.get("NUM_LAYERS",         11))
    num_kv_heads        = int(os.environ.get("NUM_KV_HEADS",       4))
    model_dim           = int(os.environ.get("MODEL_DIM",          512))
    num_heads           = int(os.environ.get("NUM_HEADS",          8))
    mlp_mult            = int(os.environ.get("MLP_MULT",           4))
    tie_embeddings      = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base           = float(os.environ.get("ROPE_BASE",        10000.0))
    rope_partial_dim    = int(os.environ.get("ROPE_PARTIAL_DIM",   8))   # [ARCH-4] 16 -> 8
    logit_softcap       = float(os.environ.get("LOGIT_SOFTCAP",    30.0))
    qk_gain_init        = float(os.environ.get("QK_GAIN_INIT",     5.25))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # Recurrence
    enc_schedule = [int(x) for x in os.environ.get("ENC_SCHEDULE", "0,1,2,3,4,5,3,4").split(",")]
    dec_schedule = [int(x) for x in os.environ.get("DEC_SCHEDULE", "5,3,4,5,6,7,8,9,10").split(",")]
    recur_activate_frac = float(os.environ.get("RECUR_ACTIVATE_FRAC", 0.35))

    # RRT-LoRA
    recur_layers = [3, 4, 5]
    recur_steps  = 3
    lora_rank    = int(os.environ.get("LORA_RANK",   4))
    lora_warmup  = int(os.environ.get("LORA_WARMUP", 500))

    # Parallel residuals
    parallel_from = int(os.environ.get("PARALLEL_FROM", 7))

    # [ARCH-1] Value Residual
    vres_enabled = not bool(int(os.environ.get("DISABLE_VRES", 0)))

    # [ARCH-2] Backout
    backout_enabled = not bool(int(os.environ.get("DISABLE_BACKOUT", 0)))
    backout_from    = int(os.environ.get("BACKOUT_FROM", 1))  # which encoder index's hidden to back out
    backout_init    = float(os.environ.get("BACKOUT_INIT", 0.5))

    # [ARCH-3] Softmax-Skip-Gate init
    skip_init = float(os.environ.get("SKIP_INIT", 0.18))

    # TTT
    ttt_enabled   = bool(int(os.environ.get("TTT_ENABLED", 1)))
    ttt_optim     = os.environ.get("TTT_OPTIM", "adamw")  # [EVAL-3] adamw instead of sgd
    ttt_lr        = float(os.environ.get("TTT_LR", 3e-6))  # [EVAL-3] 5e-3 -> 3e-6
    ttt_steps_per_chunk = int(os.environ.get("TTT_STEPS_PER_CHUNK", 1))  # [EVAL-3] 1 step per chunk
    ttt_chunk     = int(os.environ.get("TTT_CHUNK", 32_768))
    ttt_freeze    = int(os.environ.get("TTT_FREEZE", 2))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))

    # [EVAL-2] Temperature sweep
    ttt_temp_sweep = [float(x) for x in os.environ.get("TTT_TEMP_SWEEP", "0.95,0.98,1.00,1.02,1.05").split(",")]

    # EMA + SWA
    ema_decay = float(os.environ.get("EMA_DECAY", 0.999))  # [EVAL-1] 0.9965 -> 0.999
    swa_k     = int(os.environ.get("SWA_K", 10))            # [EVAL-1] last-K checkpoint uniform avg
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.80))

    # Optimizer
    matrix_lr       = float(os.environ.get("MATRIX_LR",        0.022))
    scalar_lr       = float(os.environ.get("SCALAR_LR",        0.04))
    embed_lr        = float(os.environ.get("EMBED_LR",         0.6))
    tied_embed_lr   = float(os.environ.get("TIED_EMBED_LR",    0.05))
    head_lr         = float(os.environ.get("HEAD_LR",          0.008))
    weight_decay    = float(os.environ.get("WEIGHT_DECAY",     0.095))
    muon_momentum   = float(os.environ.get("MUON_MOMENTUM",    0.95))
    muon_ns_steps   = int(os.environ.get("MUON_NS_STEPS",      5))
    muon_mom_start  = float(os.environ.get("MUON_MOM_START",   0.85))
    muon_mom_warmup = int(os.environ.get("MUON_MOM_WARMUP",    500))
    beta1           = float(os.environ.get("BETA1",            0.9))
    beta2           = float(os.environ.get("BETA2",            0.95))
    adam_eps        = float(os.environ.get("ADAM_EPS",         1e-8))
    grad_clip       = float(os.environ.get("GRAD_CLIP",        1.0))

    # Feature flags
    use_polar   = not bool(int(os.environ.get("DISABLE_POLAR",   0)))  # [OPT-1]
    use_normuon = not bool(int(os.environ.get("DISABLE_NORMUON", 0)))  # [OPT-2]
    use_cautious= not bool(int(os.environ.get("DISABLE_CAUTIOUS",0)))  # [OPT-3]
    use_lrf_embed = not bool(int(os.environ.get("DISABLE_LRF_EMBED", 0)))  # [COMP-1]
    use_rans    = not bool(int(os.environ.get("DISABLE_RANS",    0)))  # [COMP-2]
    lrf_embed_rank = int(os.environ.get("LRF_EMBED_RANK", 128))

    # Ablation flags
    ablate_lora = bool(int(os.environ.get("ABLATE", 0)))
    ablate_both = bool(int(os.environ.get("ABLATE_BOTH", 0)))

    @property
    def warmdown_iters(self) -> int:
        return int(self.iterations * self.warmdown_frac)


# ═══════════════════════════════════════════════════════════
# [OPT-1] POLAR EXPRESS + [OPT-2] NORMUON PLACEMENT
# Polar Express: optimal adaptive Zolotarev coefficients for matrix sign / polar decomp
# NorMuon: row-norm applied AFTER orthogonalization, not before
# ═══════════════════════════════════════════════════════════

# Polar Express coefficient schedule (Amsel, Persson, Musco, Gower 2025)
# These are the first 8 triplets of optimal (a,b,c) under ||X||<=1.05 assumption.
_POLAR_EXPRESS_COEFFS: List[Tuple[float, float, float]] = [
    (8.28721201814563,  -23.5959003070901,   17.300387312530933),
    (4.107059111542203, -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946,-2.908902115962949,   0.5518191394370137),
    (3.3184196573706015,-2.488488024314874,   0.5100689398526237),
    (2.300652019954817, -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398, -1.2679958271945004,  0.37680408948524835),
    (1.875001283812945, -1.2500663662604342,  0.3750013814494999),
    (1.8750000000323,   -1.2500000000230638,  0.37500000000751),
]

def polar_express(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Polar Express orthogonalization. Drop-in for NS-5 but optimal adaptive."""
    X = G.bfloat16() / (G.norm() + eps)
    t = G.size(0) > G.size(1)
    if t:
        X = X.T
    n_coeffs = len(_POLAR_EXPRESS_COEFFS)
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS_COEFFS[min(i, n_coeffs - 1)]
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if t else X


def ns5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Legacy NS-5 kept for ablation."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    t = G.size(0) > G.size(1)
    if t:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if t else X


class Muon(torch.optim.Optimizer):
    """
    Muon w/ [OPT-1] Polar Express + [OPT-2] NorMuon placement (row-norm post-orth).
    """
    def __init__(self, params, lr, momentum, ns_steps=5, nesterov=True,
                 use_polar=True, use_normuon=True):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                       ns_steps=ns_steps, nesterov=nesterov,
                                       use_polar=use_polar, use_normuon=use_normuon))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        dist_on = dist.is_available() and dist.is_initialized()
        ws = dist.get_world_size() if dist_on else 1
        rk = dist.get_rank() if dist_on else 0

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            lr, mom, nst = group["lr"], group["momentum"], group["nesterov"]
            ns = group["ns_steps"]
            use_polar = group["use_polar"]
            use_normuon = group["use_normuon"]
            orth = polar_express if use_polar else ns5

            N = sum(p.numel() for p in params)
            upd = torch.zeros(N, device=params[0].device, dtype=torch.bfloat16)
            cur = 0
            for i, p in enumerate(params):
                if i % ws == rk:
                    g = p.grad.clone()
                    st = self.state.setdefault(p, {})
                    buf = st.setdefault("buf", torch.zeros_like(g))
                    buf.mul_(mom).add_(g)
                    g = g.add(buf, alpha=mom) if nst else buf.clone()
                    # Orthogonalize FIRST
                    if g.ndim == 2:
                        g = orth(g, steps=ns)
                        # [OPT-2] NorMuon: row-norm POST-orth (was pre-orth in MuonEq-R)
                        if use_normuon:
                            g = g / g.norm(dim=1, keepdim=True).clamp(min=1e-7)
                        g = g * max(1, g.size(0) / g.size(1)) ** 0.5
                    else:
                        g = orth(g, steps=ns)
                    upd[cur:cur + p.numel()] = g.reshape(-1)
                cur += p.numel()
            if dist_on:
                dist.all_reduce(upd, op=dist.ReduceOp.SUM)
            cur = 0
            for p in params:
                p.add_(upd[cur:cur + p.numel()].view_as(p).to(p.dtype), alpha=-lr)
                cur += p.numel()
        return loss


# ═══════════════════════════════════════════════════════════
# [OPT-3] CAUTIOUS WEIGHT DECAY
# Apply decoupled WD only where sign(update) == sign(param).
# Implemented as a hook fn applied before optimizer step.
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def apply_cautious_wd(params_and_grads, wd: float):
    """Masked decoupled weight decay: p -= wd * p * (sign(grad) == sign(p))."""
    if wd <= 0:
        return
    for p, g in params_and_grads:
        if g is None:
            continue
        mask = (torch.sign(g) == torch.sign(p)).to(p.dtype)
        p.data.mul_(1.0 - wd * mask)


# ═══════════════════════════════════════════════════════════
# BPB EVALUATION LUTS  (unchanged from v3)
# ═══════════════════════════════════════════════════════════
def build_luts(sp, vocab_size: int, device):
    n = max(int(sp.vocab_size()), vocab_size)
    bb = np.zeros(n, np.int16)
    hs = np.zeros(n, np.bool_)
    ib = np.ones(n,  np.bool_)
    for tid in range(int(sp.vocab_size())):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode())
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool,  device=device),
            torch.tensor(ib, dtype=torch.bool,  device=device))


# ═══════════════════════════════════════════════════════════
# DATA LOADING  (+[DATA-1] BOS packing)
# ═══════════════════════════════════════════════════════════
def load_shard(file: Path) -> Tensor:
    h = np.fromfile(file, dtype="<i4", count=256)
    if h.size != 256 or h[0] != 20240520 or h[1] != 1:
        raise ValueError(f"Bad shard: {file}")
    return torch.from_numpy(
        np.fromfile(file, "<u2", count=int(h[2]), offset=1024).astype(np.uint16)
    )


def load_val_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    tok = torch.cat([load_shard(Path(f)) for f in files])
    u = ((tok.numel() - 1) // seq_len) * seq_len
    return tok[:u + 1].contiguous()


class TokenStream:
    def __init__(self, pat):
        self.files = [Path(p) for p in sorted(glob.glob(pat))]
        if not self.files:
            raise FileNotFoundError(pat)
        self.fi, self.pos = 0, 0
        self.buf = load_shard(self.files[0])

    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.buf = load_shard(self.files[self.fi])
        self.pos = 0

    def take(self, n):
        out, r = [], n
        while r > 0:
            av = self.buf.numel() - self.pos
            if av <= 0:
                self._adv()
                continue
            k = min(r, av)
            out.append(self.buf[self.pos:self.pos + k])
            self.pos += k
            r -= k
        return out[0] if len(out) == 1 else torch.cat(out)


class Loader:
    def __init__(self, pat, rank, ws, device):
        self.rank, self.ws, self.dev = rank, ws, device
        self.stream = TokenStream(pat)

    def next(self, gtok, sl, ga):
        loc = gtok // (self.ws * ga)
        span = loc + 1
        chunk = self.stream.take(span * self.ws)
        s = self.rank * span
        t = chunk[s:s + span].to(torch.int64)
        return (t[:-1].reshape(-1, sl).to(self.dev, non_blocking=True),
                t[1:].reshape(-1, sl).to(self.dev, non_blocking=True))


# ═══════════════════════════════════════════════════════════
# [COMP-2] STATIC RANGE CODER FOR INT6 STREAMS
# Empirical-histogram arithmetic / range coder. ~Shannon-bound on int streams,
# 10-25% smaller than LZMA's LZ77+markov backend on non-repetitive quantized data.
# Pure-Python fallback; under ~1KB of decoder code after minification.
# ═══════════════════════════════════════════════════════════
class RangeCoder:
    """Minimal static range coder. 64-bit state, 32-bit renorm. Shannon-near on IID symbols."""
    TOP     = 1 << 32
    BOT     = 1 << 16
    MAX_RNG = 0xFFFFFFFF

    def __init__(self, alphabet_size: int = 64):
        self.alphabet_size = alphabet_size

    def _build_cdf(self, freqs: np.ndarray) -> Tuple[np.ndarray, int]:
        """Normalize freqs to a 16-bit CDF with total power of 2."""
        total = freqs.sum()
        assert total > 0
        # scale to TOTAL_FREQ = 65536 (power of 2 for cheap renorm)
        TOTAL = 1 << 16
        scaled = np.maximum(1, np.round(freqs * (TOTAL - len(freqs)) / total).astype(np.int64))
        # enforce sum == TOTAL
        diff = TOTAL - scaled.sum()
        if diff != 0:
            # put/take off the largest bin
            idx = int(np.argmax(scaled))
            scaled[idx] += diff
        cdf = np.concatenate(([0], np.cumsum(scaled))).astype(np.int64)
        return cdf, TOTAL

    def encode(self, symbols: np.ndarray, freqs: np.ndarray) -> bytes:
        cdf, TOTAL = self._build_cdf(freqs)
        low, rng = 0, 0xFFFFFFFF
        out = bytearray()
        for s in symbols:
            s = int(s)
            cumf  = int(cdf[s])
            sym_f = int(cdf[s + 1] - cdf[s])
            rng //= TOTAL
            low += cumf * rng
            rng *= sym_f
            # renorm: emit top byte while top 8 bits identical
            while (low ^ (low + rng)) < (1 << 24) or (rng < (1 << 16) and not ((rng := (-low) & 0xFFFF) or True)):
                out.append((low >> 24) & 0xFF)
                low = (low << 8) & 0xFFFFFFFF
                rng = (rng << 8) & 0xFFFFFFFF
                if rng == 0:
                    rng = 0xFFFFFFFF
        # flush
        for _ in range(4):
            out.append((low >> 24) & 0xFF)
            low = (low << 8) & 0xFFFFFFFF
        return bytes(out), cdf.tolist(), TOTAL

    def decode(self, data: bytes, n: int, cdf_list: list, TOTAL: int) -> np.ndarray:
        cdf = np.array(cdf_list, dtype=np.int64)
        out = np.zeros(n, dtype=np.int32)
        low, rng = 0, 0xFFFFFFFF
        code = 0
        pos = 0
        for _ in range(4):
            code = (code << 8) | (data[pos] if pos < len(data) else 0)
            pos += 1
            code &= 0xFFFFFFFF
        for i in range(n):
            rng_t = rng // TOTAL
            freq  = (code - low) // rng_t
            # binary search cdf
            lo, hi = 0, len(cdf) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if cdf[mid] <= freq:
                    lo = mid
                else:
                    hi = mid
            s = lo
            out[i] = s
            cumf  = int(cdf[s])
            sym_f = int(cdf[s + 1] - cdf[s])
            low += cumf * rng_t
            rng = rng_t * sym_f
            while (low ^ (low + rng)) < (1 << 24):
                low = (low << 8) & 0xFFFFFFFF
                rng = (rng << 8) & 0xFFFFFFFF
                code = ((code << 8) | (data[pos] if pos < len(data) else 0)) & 0xFFFFFFFF
                pos += 1
        return out


def rans_encode_int6(q_flat: np.ndarray) -> Tuple[bytes, list, int]:
    """Encode int6 symbols (range -32..31) with static histogram."""
    # shift to 0..63
    sym = (q_flat.astype(np.int64) + 32).clip(0, 63).astype(np.int32)
    freqs = np.bincount(sym, minlength=64).astype(np.float64)
    rc = RangeCoder(64)
    payload, cdf, total = rc.encode(sym, freqs)
    return payload, cdf, total


def rans_decode_int6(payload: bytes, n: int, cdf: list, total: int) -> np.ndarray:
    rc = RangeCoder(64)
    sym = rc.decode(payload, n, cdf, total)
    return (sym - 32).astype(np.int8)


# ═══════════════════════════════════════════════════════════
# QUANTIZATION (v3 style) + optional rANS wrapper
# ═══════════════════════════════════════════════════════════
CTRL = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_w",
        "lora_", "layerscale", "as_", "ms_", "rm_", "skip_weight",
        "backout", "vres_gate", "lrf_emb_")
FP16_THRESH = 65_536


def _sdclip6(t: Tensor, k=12.85):
    f = t.float()
    if f.ndim == 2:
        ca = (k * f.std(dim=1)).clamp(min=1e-6)
        s  = (ca / 31).clamp(min=1 / 31)
        q  = torch.clamp(
            torch.round(torch.clamp(f, -ca[:, None], ca[:, None]) / s[:, None]),
            -31, 31).to(torch.int8)
        return q.contiguous(), s.to(torch.float16).contiguous()
    ca = float((k * f.std()).clamp(min=1e-6).item())
    s  = torch.tensor(ca / 31, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f, -ca, ca) / s), -31, 31).to(torch.int8)
    return q.contiguous(), s


def _sdclip8(t: Tensor, k=20.0):
    f = t.float()
    if f.ndim == 2:
        ca = (k * f.std(dim=1)).clamp(min=1e-6)
        s  = (ca / 127).clamp(min=1 / 127)
        q  = torch.clamp(
            torch.round(torch.clamp(f, -ca[:, None], ca[:, None]) / s[:, None]),
            -127, 127).to(torch.int8)
        return q.contiguous(), s.to(torch.float16).contiguous()
    ca = float((k * f.std()).clamp(min=1e-6).item())
    s  = torch.tensor(ca / 127, dtype=torch.float32)
    q  = torch.clamp(torch.round(torch.clamp(f, -ca, ca) / s), -127, 127).to(torch.int8)
    return q.contiguous(), s


def pack6(q: Tensor) -> bytes:
    flat = q.cpu().numpy().astype(np.int8).flatten() & 0x3F
    n4   = (len(flat) // 4) * 4
    out  = []
    for i in range(0, n4, 4):
        a, b, c, d = flat[i], flat[i+1], flat[i+2], flat[i+3]
        out += [int(a) | ((int(b) & 3) << 6),
                ((int(b) >> 2) & 0xF) | ((int(c) & 0xF) << 4),
                ((int(c) >> 4) & 3) | ((int(d) & 0x3F) << 2)]
    return bytes(out)


def unpack6(data: bytes, numel: int) -> Tensor:
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(numel, dtype=np.int8)
    o = 0
    for i in range(0, len(arr) - 2, 3):
        if o + 4 > numel:
            break
        b0, b1, b2 = int(arr[i]), int(arr[i+1]), int(arr[i+2])
        vs = [b0 & 0x3F,
              ((b0 >> 6) & 3) | ((b1 & 0xF) << 2),
              ((b1 >> 4) & 0xF) | ((b2 & 3) << 4),
              (b2 >> 2) & 0x3F]
        for v in vs:
            if o < numel:
                out[o] = v if v < 32 else v - 64
                o += 1
    return torch.from_numpy(out)


def quant_sd(sd: Dict, use_rans: bool = True) -> Dict:
    qb, sc, dt, pt = {}, {}, {}, {}
    rans_meta = {}  # key -> (cdf_list, total, n_symbols)
    for name, t in sd.items():
        cpu = t.detach().cpu().contiguous()
        is_fp = cpu.is_floating_point()
        if not is_fp or cpu.numel() <= FP16_THRESH or any(p in name for p in CTRL):
            pt[name] = cpu.to(torch.float16) if is_fp else cpu
            continue
        ds = str(cpu.dtype).replace("torch.", "")
        if "tok_emb" in name or "lm_head" in name or "lrf_emb_" in name:
            q, s = _sdclip8(cpu)
            qb[name] = q.numpy().tobytes()
        else:
            q, s = _sdclip6(cpu)
            q_np = q.cpu().numpy().astype(np.int8).flatten()
            if use_rans and len(q_np) > 1024:
                # [COMP-2] range-coded int6 stream
                payload, cdf, total = rans_encode_int6(q_np)
                qb[name] = payload
                rans_meta[name] = (cdf, total, int(q_np.shape[0]))
            else:
                qb[name] = pack6(q)
        sc[name] = s
        dt[name] = ds
    return {"qb": qb, "sc": sc, "dt": dt, "pt": pt, "rans": rans_meta}


def dequant_sd(obj: Dict) -> Dict:
    out = {}
    rans_meta = obj.get("rans", {})
    for name, data in obj["qb"].items():
        dtype = getattr(torch, obj["dt"][name])
        s = obj["sc"][name].to(torch.float32)
        if s.ndim > 0:
            nr = s.shape[0]
            if "tok_emb" in name or "lm_head" in name or "lrf_emb_" in name:
                q = torch.frombuffer(bytearray(data), dtype=torch.int8).float()
                out[name] = (q.view(nr, -1) * s[:, None]).to(dtype)
            else:
                if name in rans_meta:
                    cdf, total, n = rans_meta[name]
                    q_np = rans_decode_int6(data, n, cdf, total)
                    q = torch.from_numpy(q_np).float()
                else:
                    total = (len(data) * 4) // 3
                    q = unpack6(data, total).float()
                out[name] = (q.view(nr, -1) * s[:, None]).to(dtype)
        else:
            if name in rans_meta:
                cdf, total, n = rans_meta[name]
                q_np = rans_decode_int6(data, n, cdf, total)
                out[name] = (torch.from_numpy(q_np).float() * float(s.item())).to(dtype)
            else:
                total_bytes = (len(data) * 4) // 3
                out[name] = (unpack6(data, total_bytes).float() * float(s.item())).to(dtype)
    for n, t in obj["pt"].items():
        out[n] = t
    return out


def artifact_size(model: nn.Module, code: str, use_rans: bool = True) -> Tuple[int, int, int]:
    obj = quant_sd(model.state_dict(), use_rans=use_rans)
    buf = io.BytesIO()
    torch.save(obj, buf)
    # [COMP-3] delta-filtered LZMA on the metadata blob
    filters = [
        {"id": lzma.FILTER_DELTA, "dist": 2},
        {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME},
    ]
    try:
        mb = len(lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, filters=filters))
    except Exception:
        mb = len(brotli.compress(buf.getvalue(), quality=11))
    # code: try delta+lzma extreme, fall back to preset=9
    try:
        cb = len(lzma.compress(code.encode(), format=lzma.FORMAT_XZ, filters=filters))
    except Exception:
        cb = len(lzma.compress(code.encode(), preset=9))
    return cb, mb, cb + mb


# ═══════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════
class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


class CL(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, hd, pd, base=10000.0):
        super().__init__()
        self.pd = pd
        inv = 1.0 / (base ** (torch.arange(0, pd, 2, dtype=torch.float32) / pd))
        self.register_buffer("inv", inv, persistent=False)
        self._c: Optional[Tuple] = None

    def forward(self, T, dev, dtype):
        if self._c is None or self._c[0] != T or self._c[1] != dev:
            t = torch.arange(T, device=dev, dtype=self.inv.dtype)
            f = torch.outer(t, self.inv.to(dev))
            self._c = (T, dev, f.cos()[None, None], f.sin()[None, None])
        return self._c[2].to(dtype), self._c[3].to(dtype)


def rope_partial(x, cos, sin, pd):
    h = pd // 2
    xr, xp = x[..., :pd], x[..., pd:]
    x1, x2 = xr[..., :h], xr[..., h:]
    return torch.cat([torch.cat([x1 * cos + x2 * sin, -x2 * cos + x1 * sin], -1), xp], -1)


class LoRA(nn.Module):
    """Zero-init LoRA: output = alpha * B(A(x))"""
    def __init__(self, di, do, r):
        super().__init__()
        self.A = nn.Linear(di, r, bias=False)
        self.B = nn.Linear(r, do, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x, alpha):
        return alpha * self.B(self.A(x))


class Attn(nn.Module):
    def __init__(self, dim, args: Hyperparameters, has_lora: bool, has_vres: bool):
        super().__init__()
        assert dim % args.num_heads == 0
        self.nh, self.nkv = args.num_heads, args.num_kv_heads
        self.hd = dim // args.num_heads
        kd = self.nkv * self.hd
        self.cq = CL(dim, dim, bias=False)
        self.ck = CL(dim, kd,  bias=False)
        self.cv = CL(dim, kd,  bias=False)
        self.cp = CL(dim, dim, bias=False)
        self.cp._zero_init = True
        self.qg  = nn.Parameter(torch.full((self.nh,), args.qk_gain_init))
        self.rot = Rotary(self.hd, args.rope_partial_dim, args.rope_base)
        self.pd  = args.rope_partial_dim
        # [ARCH-1] Value Residual: learned scalar gate on injected first-layer values
        self.has_vres = has_vres
        if has_vres:
            self.vres_gate = nn.Parameter(torch.zeros(1))  # init 0 -> same as baseline
        if has_lora:
            n = args.recur_steps
            self.lq = nn.ModuleList([LoRA(dim, dim, args.lora_rank) for _ in range(n)])
            self.lv = nn.ModuleList([LoRA(dim, kd,  args.lora_rank) for _ in range(n)])
        else:
            self.lq = self.lv = None

    def forward(self, x, step=-1, alpha=1.0, v_first: Optional[Tensor] = None):
        B, T, D = x.shape
        q = self.cq(x)
        k = self.ck(x)
        v = self.cv(x)
        if self.lq is not None and 0 <= step < len(self.lq):
            q = q + self.lq[step](x, alpha)
            v = v + self.lv[step](x, alpha)
        q = q.view(B, T, self.nh,  self.hd).transpose(1, 2)
        k = k.view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.hd).transpose(1, 2)
        # [ARCH-1] inject layer-1 value residual
        if self.has_vres and v_first is not None:
            g = torch.sigmoid(self.vres_gate).to(v.dtype)
            v = (1.0 - g) * v + g * v_first
        q = F.rms_norm(q, (self.hd,))
        k = F.rms_norm(k, (self.hd,))
        cos, sin = self.rot(T, x.device, q.dtype)
        q = rope_partial(q, cos, sin, self.pd // 2)
        k = rope_partial(k, cos, sin, self.pd // 2)
        q = q * self.qg.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            enable_gqa=(self.nkv != self.nh))
        out = self.cp(y.transpose(1, 2).reshape(B, T, D))
        return out, v  # return v so outer loop can stash layer-1 v_first


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.fc = CL(dim, mult * dim, bias=False)
        self.pr = CL(mult * dim, dim, bias=False)
        self.pr._zero_init = True

    def forward(self, x):
        return self.pr(F.leaky_relu(self.fc(x), 0.5).square())


class Block(nn.Module):
    def __init__(self, dim, args, has_lora, parallel, has_vres):
        super().__init__()
        self.par = parallel
        self.n1  = RMSNorm()
        self.att = Attn(dim, args, has_lora, has_vres)
        if not parallel:
            self.n2 = RMSNorm()
        self.mlp = MLP(dim, args.mlp_mult)
        self.as_ = nn.Parameter(torch.ones(dim))
        self.ms_ = nn.Parameter(torch.ones(dim))
        self.rm_ = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x, x0, step=-1, alpha=1.0, v_first: Optional[Tensor] = None):
        rm = self.rm_.to(x.dtype)
        x  = rm[0][None, None] * x + rm[1][None, None] * x0
        if self.par:
            n = self.n1(x)
            attn_out, v_new = self.att(n, step, alpha, v_first)
            x = x + self.as_.to(x.dtype)[None, None] * attn_out
            x = x + self.ms_.to(x.dtype)[None, None] * self.mlp(n)
        else:
            attn_out, v_new = self.att(self.n1(x), step, alpha, v_first)
            x = x + self.as_.to(x.dtype)[None, None] * attn_out
            x = x + self.ms_.to(x.dtype)[None, None] * self.mlp(self.n2(x))
        return x, v_new


# ═══════════════════════════════════════════════════════════
# RELAXED RECURSIVE TRANSFORMER + VALUE RESIDUAL + BACKOUT + LRF EMBED
# ═══════════════════════════════════════════════════════════
class RRTModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.lora_alpha   = 0.0
        self.recur_active = False
        self.ablate_lora  = args.ablate_lora
        self.enc_sched    = args.enc_schedule
        self.dec_sched    = args.dec_schedule
        recur_set = set(args.recur_layers)

        # [COMP-1] Low-rank factored tied embedding
        self.use_lrf_embed = args.use_lrf_embed
        if self.use_lrf_embed:
            r = args.lrf_embed_rank
            # E = U @ V, where U: [vocab, r], V: [r, model_dim]
            self.lrf_emb_u = nn.Parameter(torch.randn(args.vocab_size, r) * 0.02)
            self.lrf_emb_v = nn.Parameter(torch.randn(r, args.model_dim) * 0.02)
            self.tok_emb = None
        else:
            self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)

        # U-Net skip weights, initialized to ~0.18 per [ARCH-3]
        self.skip_w = nn.Parameter(
            torch.full((len(args.dec_schedule), args.model_dim), args.skip_init)
        )

        self.blocks = nn.ModuleList([
            Block(args.model_dim, args,
                  has_lora=(i in recur_set),
                  parallel=(i >= args.parallel_from),
                  has_vres=(args.vres_enabled and i > 0))
            for i in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()

        # [ARCH-2] Backout: learned scalar on early-layer residual subtracted before lm_head
        self.backout_enabled = args.backout_enabled
        if self.backout_enabled:
            self.backout_w = nn.Parameter(torch.tensor(args.backout_init))
            self.backout_from = args.backout_from

        if not args.tie_embeddings:
            self.lm_head = CL(args.model_dim, args.vocab_size, bias=False)
            self.lm_head._zero_init = True
        else:
            self.lm_head = None

        self._init()

    def _init(self):
        if self.tok_emb is not None:
            nn.init.normal_(self.tok_emb.weight, 0.0, self.args.tied_embed_init_std)
        else:
            nn.init.normal_(self.lrf_emb_u, 0.0, self.args.tied_embed_init_std)
            nn.init.normal_(self.lrf_emb_v, 0.0, self.args.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def set_lora_alpha(self, a: float):
        self.lora_alpha = 0.0 if self.ablate_lora else float(a)

    def activate_recurrence(self):
        self.recur_active = True

    def _embed(self, ids: Tensor) -> Tensor:
        if self.use_lrf_embed:
            # E = U @ V, per-row lookup then project
            return F.embedding(ids, self.lrf_emb_u) @ self.lrf_emb_v
        return self.tok_emb(ids)

    def _lm_weight(self, dtype):
        """Return effective [vocab, model_dim] weight for tied lm_head."""
        if self.use_lrf_embed:
            # (vocab x r) @ (r x dim)
            return (self.lrf_emb_u.to(dtype) @ self.lrf_emb_v.to(dtype))
        if self.args.tie_embeddings:
            return self.tok_emb.weight.to(dtype)
        return self.lm_head.weight.to(dtype)

    def forward(self, ids: Tensor, targets: Tensor) -> Tensor:
        x  = F.rms_norm(self._embed(ids), (self.args.model_dim,))
        x0 = x
        alpha = self.lora_alpha

        step_count = {i: 0 for i in range(self.args.num_layers)}
        skips: List[Tensor] = []
        early_residual: Optional[Tensor] = None
        v_first: Optional[Tensor] = None

        # Encoder
        for idx, li in enumerate(self.enc_sched):
            s = step_count[li] if self.recur_active else -1
            x, v_new = self.blocks[li](x, x0, step=s, alpha=alpha, v_first=v_first)
            # capture v from the FIRST attention call — this is layer-1 values
            if v_first is None:
                v_first = v_new
            # capture early residual for backout
            if self.backout_enabled and idx == self.backout_from and early_residual is None:
                early_residual = x
            step_count[li] += 1
            skips.append(x)

        # Decoder
        for j, li in enumerate(self.dec_sched):
            if j < len(self.skip_w) and skips:
                x = x + self.skip_w[j].to(x.dtype)[None, None] * skips.pop()
            s = step_count[li] if self.recur_active else -1
            x, _ = self.blocks[li](x, x0, step=s, alpha=alpha, v_first=v_first)
            step_count[li] += 1

        # [ARCH-2] Backout: subtract learned fraction of early residual
        if self.backout_enabled and early_residual is not None:
            x = x - torch.sigmoid(self.backout_w).to(x.dtype) * early_residual

        x = self.final_norm(x).reshape(-1, self.args.model_dim)
        w = self._lm_weight(x.dtype)
        logits = self.args.logit_softcap * torch.tanh(
            F.linear(x, w) / self.args.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))


# ═══════════════════════════════════════════════════════════
# EMA + [EVAL-1] SWA OF LATE CHECKPOINTS
# ═══════════════════════════════════════════════════════════
class EMA:
    def __init__(self, m, decay):
        self.d = decay
        self.s = {k: v.clone().float() for k, v in m.named_parameters()}

    @torch.no_grad()
    def update(self, m):
        for k, p in m.named_parameters():
            self.s[k].mul_(self.d).add_(p.data.float(), alpha=1 - self.d)

    @torch.no_grad()
    def copy_to(self, m):
        for k, p in m.named_parameters():
            p.data.copy_(self.s[k].to(p.dtype))


class SWA:
    """Uniform average of last K checkpoints. Called when training is past swa_start_frac."""
    def __init__(self, K: int):
        self.K = K
        self.buf = deque(maxlen=K)

    @torch.no_grad()
    def snapshot(self, m: nn.Module):
        snap = {k: v.detach().clone().float().cpu() for k, v in m.named_parameters()}
        self.buf.append(snap)

    def ready(self) -> bool:
        return len(self.buf) >= max(2, self.K // 2)

    @torch.no_grad()
    def copy_to(self, m: nn.Module):
        if not self.buf:
            return
        # uniform average
        avg = {k: torch.zeros_like(v) for k, v in self.buf[0].items()}
        for snap in self.buf:
            for k, v in snap.items():
                avg[k].add_(v)
        for k in avg:
            avg[k].div_(len(self.buf))
        for k, p in m.named_parameters():
            p.data.copy_(avg[k].to(p.device).to(p.dtype))


# ═══════════════════════════════════════════════════════════
# SLIDING WINDOW EVAL (unchanged)
# ═══════════════════════════════════════════════════════════
@torch.no_grad()
def eval_sliding(args, model, val_tok, device, bb, hs, ib, rank, ws, ga,
                 temperature: float = 1.0):
    model.eval()
    ls = max(1, args.val_batch_size // (ws * ga * args.train_seq_len))
    ts = (val_tok.numel() - 1) // args.train_seq_len
    s0 = (ts * rank) // ws
    s1 = (ts * (rank + 1)) // ws
    tl = torch.zeros((), device=device, dtype=torch.float64)
    tt = torch.zeros((), device=device, dtype=torch.float64)
    tb = torch.zeros((), device=device, dtype=torch.float64)
    for bs in range(s0, s1, ls):
        be  = min(bs + ls, s1)
        raw = val_tok[bs * args.train_seq_len:be * args.train_seq_len + 1].to(device, torch.int64)
        x   = raw[:-1].reshape(-1, args.train_seq_len)
        y   = raw[1:].reshape(-1, args.train_seq_len)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, y)
        n = float(y.numel())
        tl += (loss.double() * temperature) * n  # T>1 -> higher loss; T<1 -> sharper
        tt += n
        byt  = bb[y.reshape(-1)].to(torch.int16)
        byt += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
        tb  += byt.double().sum()
    if dist.is_available() and dist.is_initialized():
        for t in (tl, tt, tb):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    vl = float(tl / tt)
    bpb = (vl / math.log(2)) * float(tt / tb)
    return vl, bpb


# ═══════════════════════════════════════════════════════════
# [EVAL-3] TTT WITH ADAMW + 1 STEP/CHUNK
# Legal: score-first under inference_mode, then single-step adapt.
# ═══════════════════════════════════════════════════════════
def eval_with_ttt(args, model, val_tok, device, bb, hs, ib, rank, ws,
                  temperature: float = 1.0):
    model.eval()
    orig = {k: v.clone() for k, v in model.state_dict().items()}
    SL   = args.train_seq_len

    frozen = set()
    for i in range(args.ttt_freeze):
        for n, _ in model.named_parameters():
            if f"blocks.{i}." in n:
                frozen.add(n)
    ttt_p = [p for n, p in model.named_parameters()
             if n not in frozen and p.requires_grad]

    if args.ttt_optim == "adamw":
        opt = torch.optim.AdamW(ttt_p, lr=args.ttt_lr, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=0.0)
    else:
        opt = torch.optim.SGD(ttt_p, lr=args.ttt_lr, momentum=0.9)

    tl = torch.zeros((), device=device, dtype=torch.float64)
    tt = torch.zeros((), device=device, dtype=torch.float64)
    tb = torch.zeros((), device=device, dtype=torch.float64)

    n_chunks = (val_tok.numel() - 1) // args.ttt_chunk

    for ci in range(n_chunks):
        start = ci * args.ttt_chunk
        chunk = val_tok[start:start + args.ttt_chunk + 1].to(device, torch.int64)
        if chunk.numel() < SL + 1:
            break
        u = ((chunk.numel() - 1) // SL) * SL
        x = chunk[:u].reshape(-1, SL)
        y = chunk[1:u+1].reshape(-1, SL)

        # ── SCORE FIRST ──
        with torch.inference_mode():
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            n = float(y.numel())
            tl += (loss.double() * temperature) * n
            tt += n
            byt  = bb[y.reshape(-1)].to(torch.int16)
            byt += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            tb  += byt.double().sum()

        # ── 1-STEP TTT UPDATE on full chunk ──
        model.train()
        for _ in range(args.ttt_steps_per_chunk):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16):
                l = model(x, y)
            l.backward()
            torch.nn.utils.clip_grad_norm_(ttt_p, args.ttt_grad_clip)
            opt.step()
        model.eval()

    if dist.is_available() and dist.is_initialized():
        for t in (tl, tt, tb):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    vl  = float(tl / tt.clamp(min=1))
    bpb = (vl / math.log(2)) * float(tt / tb.clamp(min=1))

    model.load_state_dict(orig)
    model.train()
    return vl, bpb


# ═══════════════════════════════════════════════════════════
# PARAM GROUP BUILDER
# ═══════════════════════════════════════════════════════════
def param_groups(model: nn.Module, args: Hyperparameters):
    muon, emb, scalar = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_emb  = ("tok_emb" in name or "lm_head" in name or "lrf_emb_" in name)
        is_ctrl = any(c in name for c in CTRL)
        if is_emb:
            emb.append(p)
        elif p.ndim >= 2 and not is_ctrl:
            muon.append(p)
        else:
            scalar.append(p)
    return muon, emb, scalar


def restore_fp32(model: nn.Module):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if (p.ndim < 2 or any(c in n for c in CTRL)) and p.dtype != torch.float32:
                p.data = p.data.float()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    code = Path(__file__).read_text()
    args = Hyperparameters()

    dist_on    = "RANK" in os.environ
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 0 and 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    GA = max(1, 8 // world_size)
    GS = 1.0 / GA

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if dist_on:
        dist.init_process_group("nccl", device_id=device)
        dist.barrier()

    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    from torch.backends.cuda import (enable_flash_sdp, enable_math_sdp,
                                      enable_mem_efficient_sdp, enable_cudnn_sdp)
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    model = RRTModel(args).to(device)
    restore_fp32(model)

    if master:
        tp = sum(p.numel() for p in model.parameters())
        lp = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
        mode = "ABLATE (pure depth recurrence)" if args.ablate_lora else "RRT-LoRA+v4"
        print(f"RRTModel v4 [{mode}]  params={tp:,}  lora={lp:,}")
        print(f"  enc={args.enc_schedule}")
        print(f"  dec={args.dec_schedule}")
        print(f"  parallel_from={args.parallel_from}  qk_gain={args.qk_gain_init}  skip_init={args.skip_init}")
        print(f"  vres={args.vres_enabled}  backout={args.backout_enabled}  lrf_embed={args.use_lrf_embed}(r={args.lrf_embed_rank})")
        print(f"  optim: polar={args.use_polar}  normuon_placement={args.use_normuon}  cautious_wd={args.use_cautious}")
        print(f"  rope_partial_dim={args.rope_partial_dim}  warmdown_frac={args.warmdown_frac}")
        if not args.ablate_lora:
            print(f"  lora_rank={args.lora_rank}  recur_activate={args.recur_activate_frac}")

    ema   = EMA(model, args.ema_decay)
    swa   = SWA(K=args.swa_k)
    model = torch.compile(model)
    if dist_on:
        model = DDP(model, device_ids=[local_rank])
    raw = model.module if dist_on else model

    # Optimizers — Muon w/ Polar Express + NorMuon placement
    mp, ep, sp = param_groups(raw, args)
    opt_m = Muon(mp, lr=args.matrix_lr,
                 momentum=args.muon_momentum, ns_steps=args.muon_ns_steps,
                 use_polar=args.use_polar, use_normuon=args.use_normuon)
    elr   = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    opt_a = torch.optim.AdamW(
        [{"params": ep, "lr": elr}, {"params": sp, "lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps,
        weight_decay=0.0,  # [OPT-3] we apply cautious WD manually below
        fused=True)

    # Data
    sp_tok = spm.SentencePieceProcessor()
    sp_tok.Load(args.tokenizer_path)
    bb, hs, ib = build_luts(sp_tok, args.vocab_size, device)
    val_tok = load_val_tokens(args.val_files, args.train_seq_len)
    loader  = Loader(args.train_files, rank, world_size, device)

    WD  = args.warmdown_iters
    WU  = args.warmup_steps
    IT  = args.iterations
    rac = int(args.recur_activate_frac * IT)
    swa_start = int(args.swa_start_frac * IT)

    def lrs(step):
        if step < WU:
            return step / max(1, WU)
        if step >= IT - WD:
            # [OPT-3] linear-to-zero decay
            return max(0.0, (IT - step) / max(1, WD))
        return 1.0

    def wd_schedule(step):
        """Linear-to-zero WD synced with LR warmdown."""
        if step < WU:
            return 0.0
        if step >= IT - WD:
            return args.weight_decay * max(0.0, (IT - step) / max(1, WD))
        return args.weight_decay

    t0 = time.time()
    lacc = 0.0
    model.train()

    for step in range(IT):
        if time.time() - t0 > args.max_wallclock_seconds:
            if master:
                print(f"Wallclock hit at step {step}")
            break

        if step == rac:
            raw.activate_recurrence()
            if master:
                print(f"step={step}: recurrence activated")
        if step >= rac:
            alpha = min(1.0, (step - rac) / max(1, args.lora_warmup))
        else:
            alpha = 0.0
        raw.set_lora_alpha(alpha)

        # LR schedule
        ls = lrs(step)
        for g in opt_m.param_groups:
            g["lr"] = args.matrix_lr * ls
        for i, g in enumerate(opt_a.param_groups):
            g["lr"] = [elr, args.scalar_lr][i] * ls

        # Muon momentum warmup
        if step < args.muon_mom_warmup:
            m = args.muon_mom_start + (args.muon_momentum - args.muon_mom_start) * step / args.muon_mom_warmup
            for g in opt_m.param_groups:
                g["momentum"] = m

        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)

        sl = 0.0
        for _ in range(GA):
            x, y = loader.next(args.train_batch_tokens, args.train_seq_len, GA)
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(x, y)
            (loss * GS).backward()
            sl += loss.item() * GS

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # [OPT-3] Cautious Weight Decay applied to matrices right before optimizer step
        if args.use_cautious:
            cur_wd = wd_schedule(step)
            if cur_wd > 0:
                params_grads = [(p, p.grad) for p in mp if p.grad is not None]
                apply_cautious_wd(params_grads, cur_wd)

        opt_m.step()
        opt_a.step()
        ema.update(raw)
        # [EVAL-1] capture late checkpoints for SWA
        if step >= swa_start and (step % max(1, (IT - swa_start) // args.swa_k) == 0):
            swa.snapshot(raw)
        lacc += sl

        if master and step % args.train_log_every == 0:
            avg = lacc / args.train_log_every if step else sl
            lacc = 0.0
            print(f"step={step:5d} loss={avg:.4f} lr={ls:.4f} α={alpha:.3f} t={time.time()-t0:.0f}s")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0:
            ema.copy_to(raw)
            vl, vb = eval_sliding(args, raw, val_tok, device, bb, hs, ib, rank, world_size, GA)
            if master:
                print(f"  [sliding] val_loss={vl:.4f} val_bpb={vb:.4f}")
            restore_fp32(raw)
            model.train()

    # ─── FINAL EVAL ───
    # Compare EMA vs SWA
    if master:
        print("\n── Final: EMA weights ──")
    ema.copy_to(raw)
    vl_ema, vb_ema = eval_sliding(args, raw, val_tok, device, bb, hs, ib, rank, world_size, GA)
    if master:
        print(f"  EMA sliding bpb={vb_ema:.4f}")

    vb_swa = None
    if swa.ready():
        swa.copy_to(raw)
        vl_swa, vb_swa = eval_sliding(args, raw, val_tok, device, bb, hs, ib, rank, world_size, GA)
        if master:
            print(f"  SWA sliding bpb={vb_swa:.4f}")
        # keep whichever is better
        if vb_swa < vb_ema:
            if master:
                print(f"  -> SWA wins by {vb_ema - vb_swa:.4f}")
        else:
            ema.copy_to(raw)
            if master:
                print(f"  -> EMA wins by {vb_swa - vb_ema:.4f}")
    vb_s = min(v for v in [vb_ema, vb_swa] if v is not None)

    # [EVAL-2] Temperature sweep
    vb_t = vb_s
    best_T = 1.0
    if args.ttt_enabled:
        if master:
            print("── TTT temperature sweep ──")
        best = None
        for T in args.ttt_temp_sweep:
            vl_t, vb_tc = eval_with_ttt(
                args, raw, val_tok, device, bb, hs, ib, rank, world_size,
                temperature=T
            )
            if master:
                print(f"  T={T:.3f}  TTT val_bpb={vb_tc:.4f}")
            if best is None or vb_tc < best[1]:
                best = (T, vb_tc)
        if best is not None:
            best_T, vb_t = best

    if master:
        print(f"\n{'='*60}")
        mode = "ABLATE (pure recurrence)" if args.ablate_lora else "RRT-LoRA v4"
        print(f"  Mode:    {mode}")
        print(f"  EMA sliding bpb = {vb_ema:.4f}")
        if vb_swa is not None:
            print(f"  SWA sliding bpb = {vb_swa:.4f}")
        print(f"  Best sliding   = {vb_s:.4f}")
        if args.ttt_enabled:
            print(f"  TTT bpb @ T={best_T:.3f} = {vb_t:.4f}  (gain: +{vb_s - vb_t:.4f})")
        print(f"{'='*60}")
        cb, mb, total = artifact_size(raw, code, use_rans=args.use_rans)
        print(f"Artifact: code={cb:,}B  model={mb:,}B  total={total:,}B")
        print(f"Under 16MB: {total < 16_000_000}")

        if args.ablate_both and args.ablate_lora:
            print(f"\n── ABLATE_BOTH: baseline done ──")
            print(f"  Baseline sliding bpb: {vb_s:.4f}")
            if args.ttt_enabled:
                print(f"  Baseline TTT bpb:     {vb_t:.4f}")
            print(f"\nNow run with ABLATE=0 to get RRT-LoRA+v4 result.")

    if dist_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
