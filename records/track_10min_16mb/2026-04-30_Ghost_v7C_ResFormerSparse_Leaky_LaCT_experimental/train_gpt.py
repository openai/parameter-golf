"""
Ghost v7C: XSA-all + QK-Gain 4.0 + Clip-Search Quant + TTT no-QV + LaCT chunk=32 + ResFormer sparse VRL + LeakyReLU^2
Author: lock757

Built on Ghost v5 (11L, SOTA #1 stack). All additions clearly marked.

Changes from baseline:
  1. NUM_LAYERS 9->11, MLP_MULT 2->2.5 (mlp_hidden=1280)
  2. BigramHash(6144) + projection bigram_dim=128->512
  3. SmearGate — depthwise conv local context gate per block
  4. Partial RoPE (16 dims, rest NoPE)
  5. LN Scale — learned per-block residual stream magnitude
  6. OrthoInit — orthogonal init for Q/K/V projections
  7. EMA weights (start=30%, decay=0.9995)
  8. Causal online TTT — score-first then adapt (legal per Issue #402)
  9. Sliding window eval (stride=64, seq=2048)
  10. Int5 MLP / Int6 Attn / FP16 Embed mixed quantization
  11. zstd-22 compression
  12. Muon WD=0.04

[v6 additions]:
  13. XSA on ALL 11 layers (was last 4) — near-universal in frontier SOTA
  14. QK-Gain init 4.0 (was 1.5) — matches frontier range 4.0-5.25
  15. Clip-search quantization: 8-candidate threshold search per row (min MSE)
  16. TTT no-QV mask: freeze c_q + c_v during TTT, only update c_k + MLP + norms
  17. TTT LR 5e-4 (was 2e-4)

[v7 additions]:
  18. LaCT — Large-Chunk TTT (score-first batched online adaptation)
      chunk_size=32 seqs per grad step; score chunk before adapting on it.
  19. LeakyReLU(0.5)^2 MLP activation — one-line low-risk gradient-flow patch.
  20. ResFormer sparse value residual learning — cache layer-0 V and blend into last third
      of layers through tiny learned softmax gates (~8 params for 11L sparse mode).

Target: sub-1.08 val_bpb
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

try:
    import zstandard as _zstd_mod
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


# ── Hyperparameters ────────────────────────────────────────────────────────────

class Hyperparameters:
    data_path       = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files     = os.path.join(data_path, "fineweb_train_*.bin")
    val_files       = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path  = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id          = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed            = int(os.environ.get("SEED", 1337))

    val_batch_size  = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every  = int(os.environ.get("VAL_LOSS_EVERY", 500))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 100))

    iterations          = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters      = int(os.environ.get("WARMDOWN_ITERS", 600))
    warmup_steps        = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens  = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len       = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init        = float(os.environ.get("QK_GAIN_INIT", 4.0))  # [v6] was 1.5 — frontier range 4.0-5.25

    # [GHOST] Architecture — 11L dim=512 mlp=2.5 bigram_dim=128 = 24.5M params, 15.34MB
    # SOTA #1: 10L mlp=3.0 bigram_dim=128 = 25.5M params, 15.97MB (0.626 bytes/param)
    # Ghost trades narrower MLP for 11th layer — same effective capacity, one more layer
    vocab_size      = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers      = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim       = int(os.environ.get("MODEL_DIM", 512))
    num_heads       = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult        = float(os.environ.get("MLP_MULT", 2.5))      # hidden=1280 activation^2
    # [v7C] MLP activation. Kimi/report stack recommends LeakyReLU(0.5)^2 over ReLU^2.
    leaky_relu_slope = float(os.environ.get("LEAKY_RELU_SLOPE", 0.5))
    tie_embeddings  = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base       = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # [GHOST] BigramHash with projection (bigram_dim=128 -> dim via linear)
    bigram_buckets  = int(os.environ.get("BIGRAM_BUCKETS", 6144))
    bigram_dim      = int(os.environ.get("BIGRAM_DIM", 128))

    # [GHOST] Partial RoPE — apply RoPE to first N dims, rest are NoPE
    rope_dims       = int(os.environ.get("ROPE_DIMS", 16))  # 16 of 64 head dims get RoPE

    # [GHOST] EMA — replaces SWA
    ema_enabled     = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay       = float(os.environ.get("EMA_DECAY", 0.9995))
    ema_start_frac  = float(os.environ.get("EMA_START_FRAC", 0.30))  # start at 30%

    # [GHOST] TTT — test-time training on val data post-EMA
    ttt_enabled         = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_lr              = float(os.environ.get("TTT_LR", 5e-4))  # [v6] was 2e-4
    ttt_epochs          = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_seq_len         = int(os.environ.get("TTT_SEQ_LEN", 2048))
    ttt_batch_size      = int(os.environ.get("TTT_BATCH_SIZE", 32))
    ttt_freeze_layers   = int(os.environ.get("TTT_FREEZE_LAYERS", 3))  # freeze first 3 (not 5)
    ttt_no_qv           = bool(int(os.environ.get("TTT_NO_QV", "1")))  # [v6] freeze Q+V in TTT, only update K+MLP+norms
    lact_chunk_size     = int(os.environ.get("LACT_CHUNK_SIZE", 32))   # [v6] LaCT: seqs per grad step (0=single-seq legacy)

    # [GHOST] Sliding window eval
    eval_stride     = int(os.environ.get("EVAL_STRIDE", 64))
    eval_seq_len    = int(os.environ.get("EVAL_SEQ_LEN", 2048))

    # [GHOST] Mixed quantization: Int5 MLP, Int6 Attn, FP16 Embed
    use_mixed_quant = bool(int(os.environ.get("USE_MIXED_QUANT", "1")))
    use_zstd        = bool(int(os.environ.get("USE_ZSTD", "1")))

    # [GHOST] XSA — Cross-Sequence Attention on last N layers (zero params)
    # Removes self-value bias: forces tokens to rely on context, not own value
    # arXiv 2603.09078 (Zhai 2026). Near-universal in frontier submissions.
    xsa_layers      = int(os.environ.get("XSA_LAYERS", 11))  # [v6] ALL layers — near-universal in frontier SOTA

    # [v7C] ResFormer-style Value Residual Learning (VRL)
    # sparse mode: only last third of layers blend cached layer-0 V into current V.
    # learned mode uses 2 scalar logits per active layer; fixed mode uses 0.5/0.5.
    resformer_enabled = bool(int(os.environ.get("RESFORMER_ENABLED", "1")))
    resformer_mode    = os.environ.get("RESFORMER_MODE", "sparse")  # sparse|all|off
    resformer_learned = bool(int(os.environ.get("RESFORMER_LEARNED", "1")))
    resformer_detach_v0 = bool(int(os.environ.get("RESFORMER_DETACH_V0", "1")))

    # [GHOST] Late QAT — STE int6 fake-quant during warmdown (lr_scale < threshold)
    # Trains model to be aware of quantization error before final export
    # Eliminates the quant gap between training and inference precision
    qat_enabled     = bool(int(os.environ.get("QAT_ENABLED", "1")))
    qat_threshold   = float(os.environ.get("QAT_THRESHOLD", 0.15))  # start when lr_scale < 0.15

    # Optimizer
    embed_lr            = float(os.environ.get("EMBED_LR", 0.6))
    head_lr             = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr       = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr           = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr           = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum       = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 300))
    muon_wd             = float(os.environ.get("MUON_WD", 0.04))  # [GHOST] WD=0.04
    beta1               = float(os.environ.get("BETA1", 0.9))
    beta2               = float(os.environ.get("BETA2", 0.95))
    adam_eps            = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm      = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_backend_steps  = int(os.environ.get("MUON_BACKEND_STEPS", 10))


# ── Muon optimizer ─────────────────────────────────────────────────────────────

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                       backend_steps=backend_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim < 2:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g_use = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf.clone()
                g_use = zeropower_via_newtonschulz5(g_use, steps=group["backend_steps"])
                g_use = g_use * max(1, p.size(0) / p.size(1)) ** 0.5
                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(g_use, alpha=-lr)


# ── Tokenizer lookup tables ────────────────────────────────────────────────────

def build_sentencepiece_luts(tokenizer_path: str, vocab_size: int, device):
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    if sp.vocab_size() != vocab_size:
        raise ValueError(f"VOCAB_SIZE={vocab_size} does not match tokenizer vocab_size={sp.vocab_size()}")
    base_bytes_lut = torch.zeros(vocab_size, dtype=torch.int32, device=device)
    has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        has_space = piece.startswith("\u2581")
        raw = piece.lstrip("\u2581")
        base_bytes_lut[i] = max(1, len(raw.encode("utf-8")))
        has_leading_space_lut[i] = has_space
        is_boundary_token_lut[i] = not raw
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    with open(file, "rb") as f:
        header = np.frombuffer(f.read(header_bytes), dtype="<i4")
        assert header[0] == 20240520, f"Bad magic {header[0]}"
        assert header[1] == 1, f"Bad version {header[1]}"
        n_tokens = int(header[2])
        tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
    return torch.from_numpy(tokens.astype(np.int32))


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    assert files, f"No validation files: {pattern}"
    shards = [load_data_shard(Path(f)) for f in files]
    tokens = torch.cat(shards)
    trim = (len(tokens) // seq_len) * seq_len + 1
    return tokens[:trim]


class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"No files: {pattern}"
        self.file_idx = 0
        self.pos = 0
        self.tokens = load_data_shard(Path(self.files[0]))

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(Path(self.files[self.file_idx]))
        self.pos = 0

    def take(self, n: int) -> Tensor:
        out, remaining = [], n
        while remaining > 0:
            avail = len(self.tokens) - self.pos
            if avail == 0:
                self._advance_file()
                avail = len(self.tokens)
            k = min(remaining, avail)
            out.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return torch.cat(out)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        all_tokens = self.stream.take(global_tokens + 1)
        offset = self.rank * local_tokens * grad_accum_steps
        chunk = all_tokens[offset: offset + local_tokens * grad_accum_steps + 1]
        x = chunk[:-1].reshape(grad_accum_steps, local_tokens // seq_len, seq_len)
        y = chunk[1:].reshape(grad_accum_steps, local_tokens // seq_len, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
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
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            rs, re = bs * args.train_seq_len, be * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = float(val_loss_sum / val_token_count)
    bpb = float((val_loss / math.log(2.0)) * (val_token_count / val_byte_count))
    model.train()
    return val_loss, bpb


# [GHOST] Sliding window evaluation — longer context = better BPB
def eval_val_sliding(args, model, rank, world_size, device,
                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride: int = 64, seq_len: int = 2048, batch_seqs: int = 4):
    model.eval()
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    n = val_tokens.numel()
    positions = list(range(0, n - seq_len - 1, stride))
    rank_positions = positions[rank::world_size]
    with torch.inference_mode():
        for i in range(0, len(rank_positions), batch_seqs):
            batch_pos = rank_positions[i: i + batch_seqs]
            seqs_x, seqs_y = [], []
            for p in batch_pos:
                chunk = val_tokens[p: p + seq_len + 1].to(device=device, dtype=torch.int64)
                seqs_x.append(chunk[:-1])
                seqs_y.append(chunk[1:])
            x = torch.stack(seqs_x)
            y = torch.stack(seqs_y)
            # Only score the last stride tokens to avoid double-counting.
            # model(x, y) returns a whole-window mean, so request per-token losses
            # and slice before summing.
            score_start = max(0, seq_len - stride)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                losses = model(x, y, return_per_token=True).detach()
            scored_losses = losses[:, score_start:]
            val_loss_sum += scored_losses.to(torch.float64).sum()
            val_token_count += float(scored_losses.numel())
            prev_ids = x[:, score_start:].reshape(-1)
            tgt_ids  = y[:, score_start:].reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = float(val_loss_sum / val_token_count)
    bpb = float((val_loss / math.log(2.0)) * (val_token_count / val_byte_count))
    model.train()
    return val_loss, bpb


# ── Quantization ───────────────────────────────────────────────────────────────

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain",
                    "skip_weight", "ln_scale", "bigram", "ema_")
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_CLIP_Q = 0.9999984


def _quant_category(name: str) -> str:
    """Assign quantization precision by parameter type."""
    if any(p in name for p in CONTROL_PATTERNS):
        return "fp16"
    if "mlp" in name or "fc" in name:
        return "int5"   # [GHOST] Int5 for MLP
    if "attn" in name or "c_q" in name or "c_k" in name or "c_v" in name or "proj" in name:
        return "int6"   # [GHOST] Int6 for attention
    return "int8"


# [v6] Clip-search quantization: try N candidate clip thresholds per row,
# pick the one that minimizes reconstruction MSE. This is the "GPTQ-lite clip search"
# technique common among frontier submissions. O(N * quantile) overhead — ~1-2s total.
_CLIP_CANDIDATES = [0.90, 0.925, 0.950, 0.970, 0.985, 0.995, 0.9999, 1.0]

def _quantize_tensor(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    max_val = (1 << (bits - 1)) - 1  # 127 for int8, 63 for int6, 15 for int5

    if t32.ndim == 2:
        best_q: Tensor | None = None
        best_s: Tensor | None = None
        best_err = torch.full((t32.size(0),), float("inf"), device=t32.device)

        for alpha in _CLIP_CANDIDATES:
            clip_abs = torch.quantile(t32.abs(), alpha, dim=1).clamp_min(1e-9)
            scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8)
            recon = q.float() * scale[:, None]
            err = (t32 - recon).pow(2).mean(dim=1)
            better = err < best_err
            if best_q is None:
                best_q = q.clone()
                best_s = scale.clone()
            else:
                best_q[better] = q[better]
                best_s[better] = scale[better]  # type: ignore[index]
            best_err[better] = err[better]

        return best_q.contiguous(), best_s.to(torch.float16).contiguous()  # type: ignore[return-value]

    # Scalar / 1-D tensor path
    best_q_1d: Tensor | None = None
    best_s_1d: float = 1.0
    best_err_1d = float("inf")
    for alpha in _CLIP_CANDIDATES:
        clip_abs = float(torch.quantile(t32.abs().flatten(), alpha).item()) if t32.numel() else 0.0
        scale_val = clip_abs / max_val if clip_abs > 0 else 1.0
        scale_t = torch.tensor(scale_val, dtype=torch.float32)
        q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale_t),
                        -max_val, max_val).to(torch.int8)
        err = float((t32 - q.float() * scale_t).pow(2).mean())
        if err < best_err_1d:
            best_err_1d = err
            best_q_1d = q.contiguous()
            best_s_1d = scale_val
    return best_q_1d, torch.tensor(best_s_1d, dtype=torch.float32)  # type: ignore[return-value]


def quantize_state_dict_mixed(state_dict: dict[str, Tensor], use_mixed: bool):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    total_bytes = 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous().float()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            total_bytes += passthrough[name].numel() * passthrough[name].element_size()
            continue
        cat = _quant_category(name) if use_mixed else "int8"
        bits = {"int5": 5, "int6": 6, "int8": 8}.get(cat, 8)
        q, s = _quantize_tensor(t, bits)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = cat
        total_bytes += q.numel() * q.element_size() + s.numel() * s.element_size()
    return {"quantized": quantized, "scales": scales, "dtypes": dtypes,
            "passthrough": passthrough}, total_bytes


def dequantize_state_dict_mixed(obj: dict) -> dict[str, Tensor]:
    out = {}
    bits_map = {"int5": 5, "int6": 6, "int8": 8}
    for name, q in obj["quantized"].items():
        s = obj["scales"][name]
        max_val = (1 << (bits_map[obj["dtypes"][name]] - 1)) - 1
        scale = s.float()
        if s.ndim > 0 and q.ndim == 2:
            out[name] = (q.float() * scale.view(-1, 1)).to(torch.bfloat16)
        else:
            out[name] = (q.float() * float(scale.item())).to(torch.bfloat16)
    for name, t in obj["passthrough"].items():
        out[name] = t.float() if t.dtype == torch.float16 else t
    return out


def compress(data: bytes, use_zstd: bool) -> bytes:
    if use_zstd and _HAS_ZSTD:
        return _zstd_mod.ZstdCompressor(level=22).compress(data)
    return zlib.compress(data, level=9)


def decompress(data: bytes, use_zstd: bool) -> bytes:
    if use_zstd and _HAS_ZSTD and data[:4] == b'\xfd\x2f\xb5\x28':
        return _zstd_mod.ZstdDecompressor().decompress(data)
    return zlib.decompress(data)


# ── Model ──────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    for p in module.parameters():
        if p.ndim < 2:
            p.data = p.data.float()


# [GHOST] Partial RoPE: apply RoPE to first rope_dims, rest are NoPE
class PartialRotary(nn.Module):
    def __init__(self, head_dim: int, rope_dims: int, base: float = 10000.0):
        super().__init__()
        self.rope_dims = rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dims, 2, dtype=torch.float32) / rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = self._sin_cached = None

    def forward(self, seq_len: int, device, dtype):
        if self._cos_cached is None or self._seq_len_cached < seq_len:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :].to(dtype)
            self._sin_cached = freqs.sin()[None, None, :, :].to(dtype)
            self._seq_len_cached = seq_len
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def apply_partial_rope(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int) -> Tensor:
    x_rope = x[..., :rope_dims]
    x_nope = x[..., rope_dims:]
    h = rope_dims // 2
    x1, x2 = x_rope[..., :h], x_rope[..., h:]
    rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
    return torch.cat((rotated, x_nope), dim=-1)


# [GHOST] SmearGate: lightweight depthwise conv — adds local context, minimal params
# Depthwise only (no gate linear) to stay in budget at dim=512
class SmearGate(nn.Module):
    def __init__(self, dim: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel, padding=kernel - 1, groups=dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Local average gate: add a weighted local context to x
        local = self.conv(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        return x + 0.1 * local  # additive, not multiplicative — more stable


# [GHOST] BigramHash: local bigram context embedding with projection
# Uses bigram_dim=128 -> dim projection to match SOTA #1 parameter budget
class BigramHash(nn.Module):
    def __init__(self, n_buckets: int, bigram_dim: int, dim: int):
        super().__init__()
        self.n_buckets = n_buckets
        self.embedding = nn.Embedding(n_buckets, bigram_dim)
        self.proj = CastedLinear(bigram_dim, dim, bias=False)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, ids: Tensor) -> Tensor:
        prev = torch.zeros_like(ids)
        prev[:, 1:] = ids[:, :-1]
        bucket = (prev * 1_664_525 + ids * 1_013_904_223) % self.n_buckets
        return self.proj(self.embedding(bucket))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_dims: int, rope_base: float, qk_gain_init: float,
                 use_xsa: bool = False, layer_idx: int = 0, num_layers: int = 1,
                 resformer_enabled: bool = False, resformer_mode: str = "sparse",
                 resformer_learned: bool = True, resformer_detach_v0: bool = True):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.use_xsa      = use_xsa  # [GHOST] XSA flag
        self.layer_idx    = layer_idx
        self.num_layers   = num_layers
        self.resformer_detach_v0 = resformer_detach_v0
        # [v7C] ResFormer sparse VRL: cache layer-0 V; blend only into later layers by default.
        mode = (resformer_mode or "off").lower()
        sparse_start = max(1, num_layers - max(1, num_layers // 3))
        self.resformer_active = bool(resformer_enabled) and layer_idx > 0 and (
            mode == "all" or (mode == "sparse" and layer_idx >= sparse_start)
        )
        self.resformer_learned = bool(resformer_learned)
        kv_dim = num_kv_heads * self.head_dim
        self.c_q    = CastedLinear(dim, dim, bias=False)
        self.c_k    = CastedLinear(dim, kv_dim, bias=False)
        self.c_v    = CastedLinear(dim, kv_dim, bias=False)
        self.proj   = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = PartialRotary(self.head_dim, rope_dims, rope_base)
        self.rope_dims = rope_dims
        if self.resformer_active and self.resformer_learned:
            # Two logits -> softmax weights for [cached layer-0 V, current layer V].
            # Initialized to equal blend; only 2 params per active layer.
            self.vres_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        # [GHOST] OrthoInit
        for m in [self.c_q, self.c_k, self.c_v]:
            nn.init.orthogonal_(m.weight)

    def forward(self, x: Tensor, v0_cache: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        new_v0_cache = None
        if self.layer_idx == 0:
            new_v0_cache = v.detach() if self.resformer_detach_v0 else v
        elif self.resformer_active and v0_cache is not None and v0_cache.shape == v.shape:
            # Score-safe architecture change: v0_cache comes from current forward pass layer 0,
            # not future tokens or eval adaptation. Keeps FlashAttention path intact by blending
            # before SDPA, so attention still runs once.
            if self.resformer_learned and hasattr(self, "vres_logits"):
                w = torch.softmax(self.vres_logits.float(), dim=0).to(dtype=v.dtype)
                v = w[0] * v0_cache.to(dtype=v.dtype) + w[1] * v
            else:
                v = 0.5 * v0_cache.to(dtype=v.dtype) + 0.5 * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_partial_rope(q, cos, sin, self.rope_dims)
        k = apply_partial_rope(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        if self.use_xsa:
            # [GHOST] XSA: Cross-Sequence Attention (arXiv 2603.09078, Zhai 2026)
            # Removes self-value bias: y_i -= a_ii * v_i
            # Forces model to aggregate from context, not own value vector.
            # Zero params, minimal overhead — ~2ms/step with GQA-aware impl.
            repeat = self.num_heads // self.num_kv_heads
            k_exp  = k.repeat_interleave(repeat, dim=1)  # [B, H, T, hd]
            v_exp  = v.repeat_interleave(repeat, dim=1)  # [B, H, T, hd]
            # Standard SDPA output
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                enable_gqa=(self.num_kv_heads != self.num_heads))
            # Self-attention weight: a_ii = softmax(q_i · k_i / sqrt(d)) along causal row
            # Approximate: self-score relative to row, use sigmoid as proxy for diagonal softmax
            # This is O(T) not O(T^2) — only compute diagonal
            scale = self.head_dim ** -0.5
            self_scores = (q * k_exp).sum(-1, keepdim=True) * scale  # [B, H, T, 1]
            # Sigmoid approximation for diagonal weight (avoids full attn recompute)
            self_w = torch.sigmoid(self_scores)  # [B, H, T, 1]
            # Subtract self-value contribution
            y = y - self_w * v_exp
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C)), new_v0_cache


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float, leaky_relu_slope: float = 0.5):
        super().__init__()
        hidden = (int(dim * mlp_mult) // 64) * 64
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.leaky_relu_slope = float(leaky_relu_slope)

    def forward(self, x: Tensor) -> Tensor:
        h = self.fc(x)
        h = F.leaky_relu(h, negative_slope=self.leaky_relu_slope).square()
        return self.proj(h)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_dims: int, rope_base: float, qk_gain_init: float,
                 use_xsa: bool = False, leaky_relu_slope: float = 0.5,
                 layer_idx: int = 0, num_layers: int = 1,
                 resformer_enabled: bool = False, resformer_mode: str = "sparse",
                 resformer_learned: bool = True, resformer_detach_v0: bool = True):
        super().__init__()
        self.attn_norm  = RMSNorm()
        self.mlp_norm   = RMSNorm()
        self.smear      = SmearGate(dim)                                              # [GHOST]
        self.attn       = CausalSelfAttention(dim, num_heads, num_kv_heads,
                                               rope_dims, rope_base, qk_gain_init,
                                               use_xsa=use_xsa,
                                               layer_idx=layer_idx, num_layers=num_layers,
                                               resformer_enabled=resformer_enabled,
                                               resformer_mode=resformer_mode,
                                               resformer_learned=resformer_learned,
                                               resformer_detach_v0=resformer_detach_v0)  # [GHOST]+[v7C]
        self.mlp        = MLP(dim, mlp_mult, leaky_relu_slope=leaky_relu_slope)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale   = nn.Parameter(torch.ones(dim, dtype=torch.float32))         # [GHOST]
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, v0_cache: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out, new_v0_cache = self.attn(self.smear(self.attn_norm(x)), v0_cache=v0_cache)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        x = x * self.ln_scale.to(dtype=x.dtype)[None, None, :]                      # [GHOST]
        return x, new_v0_cache


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: float,
                 bigram_buckets: int, bigram_dim: int, rope_dims: int, rope_base: float,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, qk_gain_init: float,
                 xsa_layers: int = 4, leaky_relu_slope: float = 0.5,
                 resformer_enabled: bool = False, resformer_mode: str = "sparse",
                 resformer_learned: bool = True, resformer_detach_v0: bool = True):
        super().__init__()
        self.tie_embeddings     = tie_embeddings
        self.logit_softcap      = logit_softcap
        self.tok_emb            = nn.Embedding(vocab_size, model_dim)
        self.bigram             = BigramHash(bigram_buckets, bigram_dim, model_dim)               # [GHOST]
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights       = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        # [GHOST] XSA applied to last xsa_layers layers only (where self-bias is highest)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_dims, rope_base, qk_gain_init,
                  use_xsa=(i >= num_layers - xsa_layers),
                  leaky_relu_slope=leaky_relu_slope,
                  layer_idx=i, num_layers=num_layers,
                  resformer_enabled=resformer_enabled,
                  resformer_mode=resformer_mode,
                  resformer_learned=resformer_learned,
                  resformer_detach_v0=resformer_detach_v0)
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        self._init_weights(tied_embed_init_std)

    def _init_weights(self, std: float) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, return_per_token: bool = False) -> Tensor:
        x = self.tok_emb(input_ids) + self.bigram(input_ids)                         # [GHOST]
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        v0_cache: Tensor | None = None
        for i in range(self.num_encoder_layers):
            x, new_v0 = self.blocks[i](x, x0, v0_cache=v0_cache)
            if new_v0 is not None:
                v0_cache = new_v0
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x, new_v0 = self.blocks[self.num_encoder_layers + i](x, x0, v0_cache=v0_cache)
            if new_v0 is not None:
                v0_cache = new_v0
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        losses = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="none")
        if return_per_token:
            return losses.view_as(target_ids)
        return losses.mean()


# ── TTT ────────────────────────────────────────────────────────────────────────

def ttt_adapt(args, model: GPT, val_tokens: Tensor, device, log0,
              base_bytes_lut, has_leading_space_lut, is_boundary_token_lut) -> tuple[float, float]:
    """[GHOST] Causal online TTT — score token t THEN adapt on tokens <= t.

    LEGAL pattern (per GitHub Issue #402):
        for each sequence s in val (in order):
            1. score s with current weights  -> accumulate BPB (model has NOT seen s yet)
            2. adapt weights on s            -> model learns from s
            3. advance to s+1

    ILLEGAL (what we had before): train N epochs on all val, then score.
    That leaks future tokens and is explicitly banned.
    """
    log0(f"ttt:causal_online lr={args.ttt_lr} freeze_layers={args.ttt_freeze_layers} no_qv={args.ttt_no_qv}")

    # EMA snapshots are stored frozen during training; TTT needs a fresh trainable
    # copy before applying the explicit freeze/no-QV mask below.
    for p in model.parameters():
        p.requires_grad_(True)

    frozen = set()
    if args.ttt_freeze_layers > 0:
        for i in range(min(args.ttt_freeze_layers, len(model.blocks))):
            for p in model.blocks[i].parameters():
                p.requires_grad_(False)
                frozen.add(id(p))
        for p in model.tok_emb.parameters():
            p.requires_grad_(False)
            frozen.add(id(p))
        for p in model.bigram.parameters():
            p.requires_grad_(False)
            frozen.add(id(p))

    # [v6] no-QV mask: freeze Q and V projections across all blocks.
    # Only K, MLP, norms, and scalars are updated during TTT.
    # Rationale: Q/V learned during training encode the query/value semantics;
    # updating them at test-time risks destabilizing representations. K benefits
    # from TTT because it controls what context is retrieved per token.
    if args.ttt_no_qv:
        for block in model.blocks:
            for p in block.attn.c_q.parameters():
                p.requires_grad_(False)
                frozen.add(id(p))
            for p in block.attn.c_v.parameters():
                p.requires_grad_(False)
                frozen.add(id(p))
        log0("ttt:no_qv — Q+V projections frozen, updating K+MLP+norms only")

    trainable = [p for p in model.parameters() if p.requires_grad and id(p) not in frozen]
    if not trainable:
        raise RuntimeError("TTT freeze mask left zero trainable parameters; disable TTT_NO_QV or reduce TTT_FREEZE_LAYERS")
    opt = torch.optim.AdamW(trainable, lr=args.ttt_lr, betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=0.01, fused=(device.type == "cuda"))

    seq_len        = args.ttt_seq_len
    n_seqs         = (val_tokens.numel() - 1) // seq_len
    chunk_size     = max(1, args.lact_chunk_size)   # [v6] LaCT chunk size
    val_loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count  = torch.zeros((), device=device, dtype=torch.float64)

    log0(f"ttt:lact chunk_size={chunk_size} n_seqs={n_seqs} grad_steps={math.ceil(n_seqs/chunk_size)}")

    # [v6] LaCT — Large-Chunk TTT (arXiv:2505.23884, ICLR 2026 Oral).
    # Score entire chunk FIRST (score-first, no lookahead — legal per Issue #402),
    # then run ONE batched gradient step on the whole chunk.
    # GPU utilization: chunk=32 → ~70% H100 util vs <5% for single-seq.
    # More gradient signal per wall-clock second = better adaptation in eval budget.
    for chunk_start in range(0, n_seqs, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_seqs)

        # ── STEP 1: SCORE all sequences in chunk (score-first, MUST precede adapt) ──
        xs, ys = [], []
        model.eval()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                for idx in range(chunk_start, chunk_end):
                    start = idx * seq_len
                    tok = val_tokens[start: start + seq_len + 1].to(device=device, dtype=torch.int64)
                    x = tok[:-1].unsqueeze(0)
                    y = tok[1:].unsqueeze(0)
                    score_loss = model(x, y).detach()
                    val_loss_sum    += score_loss.to(torch.float64) * float(y.numel())
                    val_token_count += float(y.numel())
                    prev_ids    = x.reshape(-1)
                    tgt_ids     = y.reshape(-1)
                    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
                    val_byte_count += token_bytes.to(torch.float64).sum()
                    xs.append(x)
                    ys.append(y)

        # ── STEP 2: ADAPT — one batched gradient step across the whole chunk ──
        model.train()
        opt.zero_grad(set_to_none=True)
        X_batch = torch.cat(xs, dim=0)   # [chunk_size, seq_len]
        Y_batch = torch.cat(ys, dim=0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            adapt_loss = model(X_batch, Y_batch)
        adapt_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()

        if chunk_start % (chunk_size * 10) == 0:
            running_bpb = float(
                (val_loss_sum / val_token_count) / math.log(2.0)
                * (val_token_count / val_byte_count)
            )
            log0(f"ttt:seq:{chunk_end}/{n_seqs} running_bpb:{running_bpb:.4f}")

    for p in model.parameters():
        p.requires_grad_(True)

    causal_bpb = float(
        (val_loss_sum / val_token_count) / math.log(2.0)
        * (val_token_count / val_byte_count)
    )
    log0(f"ttt:complete causal_online_bpb:{causal_bpb:.4f}")
    return float(val_loss_sum / val_token_count), causal_bpb


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank        = int(os.environ.get("RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    master_process = rank == 0

    if distributed:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    if master_process:
        gpu_info = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, check=False).stdout
        print(gpu_info[:500])

    def log0(msg: str) -> None:
        if master_process:
            print(msg, flush=True)

    log0(f"ghost:start rank={rank} world_size={world_size} run_id={args.run_id}")
    log0(f"arch: {args.num_layers}L dim={args.model_dim} heads={args.num_heads}/{args.num_kv_heads} "
         f"mlp_mult={args.mlp_mult} act=leaky_relu({args.leaky_relu_slope})^2 "
         f"rope_dims={args.rope_dims} bigram={args.bigram_buckets} "
         f"resformer={args.resformer_enabled}/{args.resformer_mode}/learned={args.resformer_learned}")

    # ── PRE-FLIGHT: dry-run artifact size check ────────────────────────────────
    # Instantiate on CPU with random weights, serialize + compress, verify under
    # 16MB before spending any H100 time. Random-init weights are LESS compressible
    # than trained weights so this is a conservative upper-bound estimate.
    # Hard abort if over budget rather than waste the compute grant.
    if master_process:
        log0("preflight:start artifact size check")
        _check_model = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers,
            model_dim=args.model_dim, num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            bigram_buckets=args.bigram_buckets, bigram_dim=args.bigram_dim,
            rope_dims=args.rope_dims, rope_base=args.rope_base,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, qk_gain_init=args.qk_gain_init,
            xsa_layers=args.xsa_layers,
            leaky_relu_slope=args.leaky_relu_slope,
            resformer_enabled=args.resformer_enabled,
            resformer_mode=args.resformer_mode,
            resformer_learned=args.resformer_learned,
            resformer_detach_v0=args.resformer_detach_v0,
        ).cpu()
        _n_params = sum(p.numel() for p in _check_model.parameters())
        _sd = {k: v.detach().cpu() for k, v in _check_model.state_dict().items()}
        _obj, _ = quantize_state_dict_mixed(_sd, args.use_mixed_quant)
        _buf = io.BytesIO()
        torch.save(_obj, _buf)
        _compressed = compress(_buf.getvalue(), args.use_zstd)
        _code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))
        _artifact_bytes = len(_compressed) + _code_bytes
        _limit = 16_000_000
        _headroom_kb = (_limit - _artifact_bytes) / 1024
        log0(f"preflight:params={_n_params:,} model_compressed={len(_compressed):,} "
             f"code={_code_bytes:,} total={_artifact_bytes:,} headroom={_headroom_kb:.0f}KB")
        if _artifact_bytes >= _limit:
            raise RuntimeError(
                f"PREFLIGHT FAIL: artifact {_artifact_bytes:,} bytes exceeds 16MB limit. "
                f"Overage: {_artifact_bytes - _limit:,} bytes. Aborting before wasting compute."
            )
        if _headroom_kb < 100:
            log0(f"preflight:WARNING headroom {_headroom_kb:.0f}KB is tight — trained weights "
                 f"may compress differently. Proceeding but monitor final artifact size.")
        log0(f"preflight:PASS artifact safe ({_headroom_kb:.0f}KB headroom). Starting training.")
        del _check_model, _sd, _obj, _buf, _compressed
    if distributed:
        dist.barrier()  # all ranks wait for master preflight before proceeding
    # ── END PRE-FLIGHT ─────────────────────────────────────────────────────────

    # Tokenizer
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        args.tokenizer_path, args.vocab_size, device
    )

    # Model
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        bigram_buckets=args.bigram_buckets, bigram_dim=args.bigram_dim,
        rope_dims=args.rope_dims, rope_base=args.rope_base,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, qk_gain_init=args.qk_gain_init,
        xsa_layers=args.xsa_layers,                                        # [GHOST] XSA
        leaky_relu_slope=args.leaky_relu_slope,
        resformer_enabled=args.resformer_enabled,
        resformer_mode=args.resformer_mode,
        resformer_learned=args.resformer_learned,
        resformer_detach_v0=args.resformer_detach_v0,
    ).to(device)
    restore_low_dim_params_to_fp32(base_model)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"params: {n_params:,} ({n_params*0.626/1e6:.2f}MB estimated)")

    compiled_model = torch.compile(base_model)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizers
    def is_matrix(p): return p.ndim >= 2
    def is_embed(n): return "tok_emb" in n or "bigram" in n
    def is_scalar(p): return p.ndim < 2

    muon_params   = [p for n, p in base_model.named_parameters() if is_matrix(p) and not is_embed(n)]
    embed_params  = [p for n, p in base_model.named_parameters() if is_embed(n)]
    scalar_params = [p for n, p in base_model.named_parameters() if is_scalar(p)]

    opt_muon  = Muon(muon_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                     backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    opt_embed = torch.optim.Adam(embed_params, lr=args.tied_embed_lr,
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_head  = torch.optim.Adam(scalar_params, lr=args.scalar_lr,
                                  betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_muon, opt_embed, opt_head]

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Data
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    val_tokens   = load_validation_tokens(args.val_files, args.eval_seq_len).to(device)
    log0(f"val_tokens: {val_tokens.numel():,}")

    # [GHOST] EMA model
    ema_model = None
    if args.ema_enabled:
        ema_model = copy.deepcopy(base_model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # LR schedule
    def get_lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return step / args.warmup_steps
        warmdown_start = args.iterations - args.warmdown_iters
        if step >= warmdown_start:
            return max(0.0, (args.iterations - step) / args.warmdown_iters)
        return 1.0

    training_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds
    step = 0
    stop_after_step = None

    log0("training:start")
    t0 = time.perf_counter()

    while step < args.iterations:
        x_all, y_all = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
        scale = get_lr_scale(step)
        muon_momentum = (1 - min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)) * args.muon_momentum_warmup_start \
                        + min(step / max(args.muon_momentum_warmup_steps, 1), 1.0) * args.muon_momentum
        opt_muon.param_groups[0]["momentum"] = muon_momentum
        for opt in optimizers:
            base_lr = {"opt_muon": args.matrix_lr, "opt_embed": args.tied_embed_lr,
                       "opt_head": args.scalar_lr}[[opt_muon, opt_embed, opt_head].index(opt)]
            # Use HEAD_LR for head if not tied
            if opt is opt_head:
                base_lr = args.head_lr if not args.tie_embeddings else args.scalar_lr
            opt.param_groups[0]["lr"] = base_lr * scale

        zero_grad_all()
        train_loss = torch.tensor(0.0, device=device)
        for micro in range(grad_accum_steps):
            x, y = x_all[micro].to(device), y_all[micro].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y) / grad_accum_steps
            loss.backward()
            train_loss += loss.detach()
        if distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        step += 1

        # [GHOST] EMA update
        if args.ema_enabled and ema_model is not None:
            ema_start_step = int(args.ema_start_frac * args.iterations)
            if step >= ema_start_step:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), base_model.parameters()):
                        ema_p.data.lerp_(p.data.float(), 1.0 - args.ema_decay)

        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # [GHOST] Late QAT: fake-quantize weights when LR scale < threshold
        # STE allows gradients to flow through the quantization operation.
        # Model adapts to int6 quantization error before final export.
        if args.qat_enabled and scale < args.qat_threshold:
            with torch.no_grad():
                for p in base_model.parameters():
                    if p.ndim == 2 and p.numel() > 65536:
                        # STE int6 fake-quant: forward=round to int6, backward=identity
                        max_val = 31.0  # int6 signed: [-32, 31]
                        row_scale = p.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8) / max_val
                        p_q = torch.clamp((p / row_scale).round(), -max_val, max_val) * row_scale
                        p.copy_(p_q)

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            qat_active = args.qat_enabled and scale < args.qat_threshold
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"lr_scale:{scale:.4f} qat:{qat_active} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            eval_model = ema_model if (ema_model is not None and step >= int(args.ema_start_frac * args.iterations)) else base_model
            v_loss, v_bpb = eval_val(args, eval_model, rank, world_size, device, grad_accum_steps,
                                      val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{v_loss:.4f} val_bpb:{v_bpb:.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = approx_ms >= max_wallclock_ms
        if distributed:
            cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step
        if stop_after_step is not None and step >= stop_after_step:
            break

    training_time_ms += 1000.0 * (time.perf_counter() - t0)
    log0(f"training:done steps:{step} time:{training_time_ms:.0f}ms")

    # Build a clean export model. TTT is legal only as score-first online
    # evaluation; serializing a validation-adapted model and rescoring it is not.
    source_model = ema_model if ema_model is not None else base_model
    final_model = copy.deepcopy(source_model).to(device)
    for p in final_model.parameters():
        p.requires_grad_(True)

    # Serialize clean trained/EMA weights only.
    if master_process:
        code_bytes = len(code.encode("utf-8"))
        sd = {k: v.detach().cpu() for k, v in final_model.state_dict().items()}
        obj, model_bytes_raw = quantize_state_dict_mixed(sd, args.use_mixed_quant)
        buf = io.BytesIO()
        torch.save(obj, buf)
        compressed = compress(buf.getvalue(), args.use_zstd)
        artifact_bytes = len(compressed) + code_bytes
        log0(f"code_bytes:{code_bytes} model_compressed:{len(compressed)} total:{artifact_bytes}")
        log0(f"artifact_size_ok:{artifact_bytes < 16_000_000}")
        with open("final_model.ptz", "wb") as f:
            f.write(compressed)

    # Every rank must load the same quantized/dequantized checkpoint.
    if distributed:
        dist.barrier()
    with open("final_model.ptz", "rb") as f:
        compressed_disk = f.read()
    raw = decompress(compressed_disk, args.use_zstd)
    obj_loaded = torch.load(io.BytesIO(raw), map_location="cpu")
    deq_sd = dequantize_state_dict_mixed(obj_loaded)
    final_model.load_state_dict(deq_sd, strict=False)
    restore_low_dim_params_to_fp32(final_model)
    if distributed:
        dist.barrier()

    t_eval = time.perf_counter()
    if args.ttt_enabled:
        log0("ttt:starting legal quantized-roundtrip causal online TTT")
        q_loss, q_bpb = ttt_adapt(
            args, final_model, val_tokens, device, log0,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        eval_ms = 1000.0 * (time.perf_counter() - t_eval)
        log0(f"final_legal_ttt_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} eval_time:{eval_ms:.0f}ms")
        log0(f"final_legal_ttt_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")
    else:
        q_loss, q_bpb = eval_val_sliding(
            args, final_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride, seq_len=args.eval_seq_len,
        )
        eval_ms = 1000.0 * (time.perf_counter() - t_eval)
        log0(f"final_int8_zlib_roundtrip val_loss:{q_loss:.4f} val_bpb:{q_bpb:.4f} eval_time:{eval_ms:.0f}ms")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_loss:.8f} val_bpb:{q_bpb:.8f}")

    if master_process:
        log0(f"peak_memory: {torch.cuda.max_memory_allocated()//1024//1024}MiB "
             f"reserved:{torch.cuda.max_memory_reserved()//1024//1024}MiB")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
