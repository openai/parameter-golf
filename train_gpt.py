"""
ParGolf-Zero v2 — Competition Submission
==========================================
Target: < 1.14 bpb (current best: 1.1428)

Layer 1 — COMPAT    : Auto-detects platform, runs anywhere
Layer 2 — TRAIN     : FP16 embed + Muon WD + warmdown + SWA
Layer 3 — COMPRESS  : Weight range penalty + QAT + int6 middle + zstd
Layer 4 — EVAL      : Sliding window evaluation (stride=64)
Layer 5 — ADAPT     : TTT-aware training (low-rank regularization)
Layer 6 — BIGRAM    : BigramHash(10240) lookup table on embeddings

Run command (8xH100):
    RUN_ID=pargolf_zero \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    VOCAB_SIZE=1024 \
    MAX_WALLCLOCK_SECONDS=600 \
    torchrun --standalone --nproc_per_node=8 pargolf_zero.py

Run command (single T4 / smoke test):
    TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=512 VAL_BATCH_SIZE=65536 \
    MAX_WALLCLOCK_SECONDS=300 ITERATIONS=500 \
    python pargolf_zero.py
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
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ─────────────────────────────────────────────
# LAYER 1 — COMPAT: Platform Detection
# ─────────────────────────────────────────────

@dataclass
class PlatformConfig:
    tier: int
    device_name: str
    compute_cap: tuple
    vram_mb: int
    use_flash_attn: bool
    use_compile: bool
    use_gqa: bool
    ttt_batch_size: int

def detect_platform() -> PlatformConfig:
    if not torch.cuda.is_available():
        return PlatformConfig(0, "CPU", (0,0), 0, False, False, False, 0)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // (1024*1024)
    if cap[0] >= 9:
        return PlatformConfig(3, name, cap, vram, True,  True,  True,  64)
    if cap[0] >= 8:
        return PlatformConfig(2, name, cap, vram, True,  True,  True,  32)
    if cap[0] >= 7:
        ttt_b = 8 if vram < 32000 else 16
        return PlatformConfig(1, name, cap, vram, False, False, False, ttt_b)
    return PlatformConfig(0, name, cap, vram, False, False, False, 0)

def compat_sdp(platform: PlatformConfig) -> None:
    try:
        from torch.backends.cuda import (
            enable_cudnn_sdp, enable_flash_sdp,
            enable_math_sdp, enable_mem_efficient_sdp,
        )
        if platform.use_flash_attn:
            enable_cudnn_sdp(False); enable_flash_sdp(True)
            enable_mem_efficient_sdp(False); enable_math_sdp(False)
        else:
            enable_cudnn_sdp(False); enable_flash_sdp(False)
            enable_mem_efficient_sdp(False); enable_math_sdp(True)
    except Exception:
        pass

def compat_compile(model, platform: PlatformConfig):
    if not platform.use_compile:
        return model
    try:
        return torch.compile(model, dynamic=False, fullgraph=True)
    except Exception as e:
        print(f"[COMPAT] torch.compile skipped: {e}")
        return model

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────

class Hyperparameters:
    data_path      = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files    = os.path.join(data_path, "fineweb_train_*.bin")
    val_files      = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED", 1337))

    val_batch_size    = int(os.environ.get("VAL_BATCH_SIZE",    524_288))
    val_loss_every    = int(os.environ.get("VAL_LOSS_EVERY",    1000))
    train_log_every   = int(os.environ.get("TRAIN_LOG_EVERY",   200))

    iterations            = int(os.environ.get("ITERATIONS",           20000))
    warmup_steps          = int(os.environ.get("WARMUP_STEPS",         20))
    train_batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS",   524_288))
    train_seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",        1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init          = float(os.environ.get("QK_GAIN_INIT",        1.5))

    vocab_size     = int(os.environ.get("VOCAB_SIZE",    1024))
    num_layers     = int(os.environ.get("NUM_LAYERS",    9))
    num_kv_heads   = int(os.environ.get("NUM_KV_HEADS",  4))
    model_dim      = int(os.environ.get("MODEL_DIM",     512))
    num_heads      = int(os.environ.get("NUM_HEADS",     8))
    mlp_mult       = int(os.environ.get("MLP_MULT",      2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base      = float(os.environ.get("ROPE_BASE",   10000.0))
    logit_softcap  = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Layer 2 — TRAIN: improved LR + warmdown
    matrix_lr      = float(os.environ.get("MATRIX_LR",      0.02))   # was 0.04
    scalar_lr      = float(os.environ.get("SCALAR_LR",      0.02))   # was 0.04
    tied_embed_lr  = float(os.environ.get("TIED_EMBED_LR",  0.03))   # was 0.05
    embed_lr       = float(os.environ.get("EMBED_LR",       0.6))
    head_lr        = float(os.environ.get("HEAD_LR",        0.008))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS",   20000))   # was 1200
    muon_weight_decay   = float(os.environ.get("MUON_WD",   0.02))
    muon_momentum       = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps  = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1      = float(os.environ.get("BETA1",     0.9))
    beta2      = float(os.environ.get("BETA2",     0.95))
    adam_eps   = float(os.environ.get("ADAM_EPS",  1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    fp16_embedding = bool(int(os.environ.get("FP16_EMBEDDING", "1")))

    # Layer 3 — COMPRESS
    weight_range_penalty = float(os.environ.get("WEIGHT_RANGE_PENALTY", 0.001))
    qat_steps   = int(os.environ.get("QAT_STEPS",   500))
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))

    # Layer 4 — EVAL
    eval_stride      = int(os.environ.get("EVAL_STRIDE",      64))
    eval_seq_len     = int(os.environ.get("EVAL_SEQ_LEN",     1024))
    eval_batch_seqs  = int(os.environ.get("EVAL_BATCH_SEQS",  8))
    use_sliding_window = bool(int(os.environ.get("USE_SLIDING_WINDOW", "1")))

    # Layer 5 — ADAPT
    lowrank_penalty  = float(os.environ.get("LOWRANK_PENALTY", 0.0005))

    # Layer 6 — BIGRAM
    bigram_hash_size = int(os.environ.get("BIGRAM_HASH_SIZE", 10240))
    use_bigram       = bool(int(os.environ.get("USE_BIGRAM", "1")))

    # SWA (Stochastic Weight Averaging)
    use_swa          = bool(int(os.environ.get("USE_SWA", "1")))
    swa_start_frac   = float(os.environ.get("SWA_START_FRAC", 0.6))

    # int6 for middle layers
    use_int6_middle  = bool(int(os.environ.get("USE_INT6_MIDDLE", "1")))

    # TTT eval
    ttt_lora_rank  = int(os.environ.get("TTT_LORA_RANK",  8))
    ttt_lora_lr    = float(os.environ.get("TTT_LORA_LR",  0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))

# ─────────────────────────────────────────────
# LAYER 2 — TRAIN: Muon with Weight Decay
# ─────────────────────────────────────────────

def _zeropower_ns5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if G.size(0) > G.size(1) else X

class MuonWD(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, weight_decay=0.0, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            lr, mom, steps = group["lr"], group["momentum"], group["backend_steps"]
            wd, nesterov = group["weight_decay"], group["nesterov"]
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    st = self.state[p]
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(g)
                    st["buf"].mul_(mom).add_(g)
                    if nesterov: g = g.add(st["buf"], alpha=mom)
                    g = _zeropower_ns5(g, steps=steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = flat[curr: curr + p.numel()].view_as(p).to(p.dtype)
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()

# ─────────────────────────────────────────────
# LAYER 3 — COMPRESS: Quantization-Aware
# ─────────────────────────────────────────────

def weight_range_loss(model: nn.Module, penalty: float) -> Tensor:
    if penalty <= 0:
        return torch.tensor(0.0)
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in name:
            total = total + (p.max(dim=1).values - p.min(dim=1).values).mean()
            count += 1
    return penalty * (total / max(count, 1))

def fake_quantize(t: Tensor) -> Tensor:
    scale = t.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 127.0 if t.ndim == 2 \
        else t.abs().max().clamp(min=1e-8) / 127.0
    return t + (torch.round(t / scale) * scale - t).detach()

def lowrank_penalty_loss(model: nn.Module, penalty: float) -> Tensor:
    if penalty <= 0:
        return torch.tensor(0.0)
    total = torch.tensor(0.0)
    count = 0
    for name, p in model.named_parameters():
        if p.ndim == 2 and ("c_q" in name or "c_v" in name):
            with torch.no_grad():
                u = F.normalize(torch.randn(p.shape[0], device=p.device), dim=0)
                for _ in range(3):
                    v = F.normalize(p.T @ u, dim=0)
                    u = F.normalize(p @ v, dim=0)
                top_sv = (u @ p @ v).abs()
                frob = p.norm()
                if frob > 1e-8:
                    total = total + (frob**2 - top_sv**2).clamp(min=0) / frob**2
                    count += 1
    return penalty * (total / max(count, 1))

# ─────────────────────────────────────────────
# TOKENIZER-AGNOSTIC EVAL SETUP
# ─────────────────────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_space_np  = np.zeros((table_size,), dtype=np.bool_)
    is_bndry_np   = np.ones((table_size,),  dtype=np.bool_)
    for tid in range(sp_vocab):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        is_bndry_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space_np[tid] = True; piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16,  device=device),
        torch.tensor(has_space_np,  dtype=torch.bool,   device=device),
        torch.tensor(is_bndry_np,   dtype=torch.bool,   device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

# ─────────────────────────────────────────────
# LAYER 4 — EVAL: Sliding Window
# ─────────────────────────────────────────────

def eval_val_baseline(args, model, rank, world_size, device, grad_accum_steps,
                      val_tokens, bb_lut, hs_lut, ib_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end   = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt  = torch.zeros((), device=device, dtype=torch.float64)
    byt_cnt  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            local = val_tokens[bs*args.train_seq_len: be*args.train_seq_len+1].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            n = float(y.numel())
            loss_sum += batch_loss.to(torch.float64) * n
            tok_cnt  += n
            px, ty = x.reshape(-1), y.reshape(-1)
            tb = bb_lut[ty].to(torch.int16)
            tb += (hs_lut[ty] & ~ib_lut[px]).to(torch.int16)
            byt_cnt += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt,  op=dist.ReduceOp.SUM)
        dist.all_reduce(byt_cnt,  op=dist.ReduceOp.SUM)
    vl = float((loss_sum / tok_cnt).item())
    vb = float((loss_sum / tok_cnt).item() / math.log(2.0) * (tok_cnt / byt_cnt).item())
    model.train()
    return vl, vb

def _forward_logits(model, input_ids, device):
    with torch.inference_mode():
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                            dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            x = model.tok_emb(input_ids)
            x = F.rms_norm(x, (x.size(-1),))
            x0 = x; skips = []
            for i in range(model.num_encoder_layers):
                x = model.blocks[i](x, x0); skips.append(x)
            for i in range(model.num_decoder_layers):
                bi = model.num_encoder_layers + i
                if skips:
                    x = x + model.skip_weights[i].to(x.dtype)[None,None,:] * skips.pop()
                x = model.blocks[bi](x, x0)
            x = model.final_norm(x)
            logits = F.linear(x, model.tok_emb.weight) if model.tie_embeddings else model.lm_head(x)
            logits = model.logit_softcap * torch.tanh(logits / model.logit_softcap)
            return F.log_softmax(logits.float(), dim=-1)

def eval_val_sliding(args, model, rank, world_size, device, val_tokens, bb_lut, hs_lut, ib_lut):
    stride = args.eval_stride
    seq_len = args.eval_seq_len
    bsz = args.eval_batch_seqs
    total = val_tokens.numel() - 1
    r_start = (total * rank) // world_size
    r_end   = (total * (rank+1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt  = torch.zeros((), device=device, dtype=torch.float64)
    byt_cnt  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    positions = list(range(r_start, r_end, stride))
    with torch.inference_mode():
        for b0 in range(0, len(positions), bsz):
            batch_pos = positions[b0: b0+bsz]
            actual = len(batch_pos)
            xb = torch.zeros(actual, seq_len, dtype=torch.int64, device=device)
            yb = torch.zeros(actual, seq_len, dtype=torch.int64, device=device)
            for bi, pos in enumerate(batch_pos):
                ws = max(0, pos - seq_len + stride)
                we = min(pos + stride, total)
                wl = we - ws
                toks = val_tokens[ws:we+1].to(device=device, dtype=torch.int64)
                fl = min(wl, seq_len)
                xb[bi, :fl] = toks[:fl]
                yb[bi, :fl] = toks[1:fl+1]
            lp = _forward_logits(model, xb, device)
            for bi, pos in enumerate(batch_pos):
                ws = max(0, pos - seq_len + stride)
                we = min(pos + stride, total)
                wl = we - ws
                ss = max(0, wl - stride)
                sl = wl - ss
                if sl <= 0: continue
                sx = xb[bi, ss:ss+sl]
                sy = yb[bi, ss:ss+sl]
                losses = -lp[bi, ss:ss+sl].gather(1, sy.unsqueeze(1)).squeeze(1)
                tb = bb_lut[sy].to(torch.float64)
                tb = tb + (hs_lut[sy] & ~ib_lut[sx]).to(torch.float64)
                loss_sum += losses.to(torch.float64).sum()
                tok_cnt  += sl
                byt_cnt  += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_cnt,  op=dist.ReduceOp.SUM)
        dist.all_reduce(byt_cnt,  op=dist.ReduceOp.SUM)
    if tok_cnt.item() == 0: return float('inf'), float('inf')
    vl = float(loss_sum.item() / tok_cnt.item())
    vb = float((loss_sum.item() / math.log(2.0)) / byt_cnt.item())
    model.train()
    return vl, vb

def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, bb_lut, hs_lut, ib_lut, use_sliding=False):
    if use_sliding:
        return eval_val_sliding(args, model, rank, world_size, device,
                                val_tokens, bb_lut, hs_lut, ib_lut)
    return eval_val_baseline(args, model, rank, world_size, device,
                             grad_accum_steps, val_tokens, bb_lut, hs_lut, ib_lut)

# ─────────────────────────────────────────────
# QUANTIZATION (with FP16 embedding — Layer 2)
# ─────────────────────────────────────────────

CONTROL_PATTERNS = (
    "attn_scale","attn_scales","mlp_scale","mlp_scales",
    "resid_mix","resid_mixes","q_gain","skip_weight","skip_weights",
)
INT8_KEEP_MAX  = 65_536
INT8_CLIP_Q    = 99.99984 / 100.0

def _tensor_nbytes(t): return int(t.numel()) * int(t.element_size())

def _keep_float(name, t, orig_dtypes):
    if any(p in name for p in CONTROL_PATTERNS): return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(torch.float16).contiguous()
    return t

def _quant_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
        cl = torch.maximum(torch.minimum(t32, ca[:,None]), -ca[:,None])
        sc = (ca / 127.0).clamp_min(1.0/127.0)
        return torch.clamp(torch.round(cl / sc[:,None]), -127, 127).to(torch.int8).contiguous(), \
               sc.to(torch.float16).contiguous()
    ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    sc = torch.tensor(ca/127.0 if ca > 0 else 1.0, dtype=torch.float32)
    return torch.clamp(torch.round(torch.clamp(t32,-ca,ca)/sc), -127, 127).to(torch.int8).contiguous(), sc

def quantize_state_dict_int8(state_dict, fp16_embedding=True):
    quantized, scales, dtypes, passthrough, orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count","num_tensors","num_float_tensors",
                            "num_nonfloat_tensors","baseline_tensor_bytes","int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += _tensor_nbytes(t)
            continue
        # Layer 2: keep embedding in fp16 instead of int8
        if fp16_embedding and "tok_emb" in name:
            passthrough[name] = t.to(torch.float16).contiguous()
            orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["int8_payload_bytes"] += _tensor_nbytes(passthrough[name])
            continue
        if t.numel() <= INT8_KEEP_MAX:
            kept = _keep_float(name, t, orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += _tensor_nbytes(kept)
            continue
        stats["num_float_tensors"] += 1
        q, s = _quant_tensor(t)
        if s.ndim > 0: qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q; scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if orig_dtypes: obj["passthrough_orig_dtypes"] = orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out, qmeta, orig = {}, obj.get("qmeta", {}), obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1]*(q.ndim-1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        ot = t.detach().cpu().contiguous()
        od = orig.get(name)
        if isinstance(od, str): ot = ot.to(getattr(torch, od)).contiguous()
        out[name] = ot
    return out

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data_shard(file: Path) -> Tensor:
    hb = 256 * np.dtype("<i4").itemsize
    h = np.fromfile(file, dtype="<i4", count=256)
    if h.size != 256 or int(h[0]) != 20240520 or int(h[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    n = int(h[2])
    if file.stat().st_size != hb + n * 2:
        raise ValueError(f"Shard size mismatch: {file}")
    arr = np.fromfile(file, dtype="<u2", count=n, offset=hb)
    return torch.from_numpy(arr.astype(np.uint16, copy=False))

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files: raise FileNotFoundError(pattern)
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0
    def _next(self):
        self.file_idx = (self.file_idx+1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx]); self.pos = 0
    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._next(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos+k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks)==1 else torch.cat(chunks)

class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank = rank; self.world_size = world_size
        self.device = device; self.stream = TokenStream(pattern)
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local = global_tokens // (self.world_size * grad_accum_steps)
        span = local + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local_ = chunk[start:start+span].to(torch.int64)
        x = local_[:-1].reshape(-1, seq_len)
        y = local_[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)

def restore_low_dim_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0/(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim)), persistent=False)
        self._sl = 0; self._cos = None; self._sin = None
    def forward(self, sl, device, dtype):
        if self._cos is None or self._sl != sl or self._cos.device != device:
            t = torch.arange(sl, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None,None,:,:]; self._sin = f.sin()[None,None,:,:]
            self._sl = sl
        return self._cos.to(dtype), self._sin.to(dtype)

def apply_rope(x, cos, sin):
    h = x.size(-1)//2
    x1, x2 = x[...,:h], x[...,h:]
    return torch.cat((x1*cos + x2*sin, x1*(-sin) + x2*cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, platform):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads; self.platform = platform
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x, q_delta=None, v_delta=None):
        B, T, D = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x); v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(B,T,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(B,T,self.num_kv_heads,self.head_dim).transpose(1,2)
        v = v.reshape(B,T,self.num_kv_heads,self.head_dim).transpose(1,2)
        q = F.rms_norm(q,(q.size(-1),)); k = F.rms_norm(k,(k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rope(q,cos,sin); k = apply_rope(k,cos,sin)
        q = q * self.q_gain.to(q.dtype)[None,:,None,None]
        # Platform-aware attention
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        if self.platform.use_flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            scale = self.head_dim ** -0.5
            att = (q @ k.transpose(-2,-1)) * scale
            mask = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float('-inf'))
            att = torch.softmax(att.float(), dim=-1).to(q.dtype)
            y = att @ v
        y = y.transpose(1,2).contiguous().reshape(B,T,D)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc   = CastedLinear(dim, mlp_mult*dim, bias=False)
        self.proj = CastedLinear(mlp_mult*dim, dim, bias=False); self.proj._zero_init = True
    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())

# ─────────────────────────────────────────────
# LAYER 6 — BIGRAM: Hash table lookup
# ─────────────────────────────────────────────

class BigramHash(nn.Module):
    """
    Learned bigram hash table. For each (prev_token, curr_token) pair,
    adds a learned bias to the logits. Uses hashing so it fits in budget.
    ~10240 * vocab_size params but stored as int8 — tiny in 16MB.
    """
    def __init__(self, vocab_size: int, hash_size: int, model_dim: int):
        super().__init__()
        self.hash_size = hash_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(hash_size, vocab_size)
        nn.init.zeros_(self.embedding.weight)

    def hash_bigram(self, prev_ids: Tensor, curr_ids: Tensor) -> Tensor:
        return (prev_ids * 1000003 + curr_ids) % self.hash_size

    def forward(self, input_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        prev = torch.zeros_like(input_ids)
        prev[:, 1:] = input_ids[:, :-1]
        keys = self.hash_bigram(prev, input_ids)
        return self.embedding(keys)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, platform):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, platform)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0, q_delta_fn=None, v_delta_fn=None):
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None,None,:]*x + mix[1][None,None,:]*x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn else None
        vd = v_delta_fn(n) if v_delta_fn else None
        x = x + self.attn_scale.to(x.dtype)[None,None,:] * self.attn(n, qd, vd)
        x = x + self.mlp_scale.to(x.dtype)[None,None,:]  * self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, args, platform):
        super().__init__()
        self.tie_embeddings    = args.tie_embeddings
        self.tied_embed_init_std = args.tied_embed_init_std
        self.logit_softcap     = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads,
                  args.mlp_mult, args.rope_base, args.qk_gain_init, platform)
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if args.tie_embeddings else CastedLinear(args.model_dim, args.vocab_size, bias=False)
        if self.lm_head: self.lm_head._zero_init = True
        # Layer 6: BigramHash
        self.bigram = BigramHash(args.vocab_size, args.bigram_hash_size, args.model_dim) \
                      if args.use_bigram else None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, 0.0, self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids, target_ids, lora=None):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x; skips = []
        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i]  if lora else None
            vd = lora.v_loras[i]  if lora else None
            x = self.blocks[i](x, x0, qd, vd); skips.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None,None,:] * skips.pop()
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
        x = self.final_norm(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        # Layer 6: add bigram bias
        if self.bigram is not None:
            logits = logits + self.bigram(input_ids).to(logits.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if lora:
            B, S, V = logits.shape
            return F.cross_entropy(logits.float().reshape(-1,V), target_ids.reshape(-1), reduction="none").reshape(B,S)
        return F.cross_entropy(logits.float().reshape(-1,logits.size(-1)), target_ids.reshape(-1), reduction="mean")

# ─────────────────────────────────────────────
# TTT LORA (Layer 5 eval)
# ─────────────────────────────────────────────

BOS_ID = 1

class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz, in_f, out_f, rank):
        super().__init__()
        self.in_features = in_f
        self.A = nn.Parameter(torch.empty(bsz, rank, in_f))
        self.B = nn.Parameter(torch.zeros(bsz, out_f, rank))
        self.reset()
    def forward(self, x): return (x @ self.A.transpose(1,2)) @ self.B.transpose(1,2)
    def reset(self):
        bd = 1.0/math.sqrt(self.in_features)
        with torch.no_grad(): self.A.uniform_(-bd,bd); self.B.zero_()

class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz, model, rank):
        super().__init__()
        dim = model.tok_emb.embedding_dim; vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList(); self.v_loras = nn.ModuleList()
        for blk in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, blk.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, blk.attn.c_v.weight.shape[0], rank))
    def reset(self):
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA): m.reset()

def _reset_ttt_opt(opt):
    for g in opt.param_groups:
        for p in g['params']:
            s = opt.state.get(p)
            if s:
                s['exp_avg'].zero_(); s['exp_avg_sq'].zero_(); s['step'].fill_(0)

def _find_docs(tokens):
    pos = (tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(pos)):
        s = int(pos[i]); e = int(pos[i+1]) if i+1 < len(pos) else tokens.numel()
        if i+1 < len(pos): e += 1
        if e - s >= 2: docs.append((s, e-s))
    return docs

def _chunk_window(ci, pred_len, num_chunks, chunk_size, eval_sl):
    cs = ci*chunk_size; ce = pred_len if ci==num_chunks-1 else (ci+1)*chunk_size
    ws = max(0, ce-eval_sl); wl = ce-ws; co = cs-ws; cl = ce-cs
    return ws, wl, co, cl

def eval_val_ttt_lora(args, base_model, rank, world_size, device, bb_lut, hs_lut, ib_lut, platform):
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)
    rank_docs = docs[(len(docs)*rank)//world_size : (len(docs)*(rank+1))//world_size]

    # Scale TTT batch size to platform VRAM
    ttt_bsz = min(args.ttt_batch_size, platform.ttt_batch_size) if platform.ttt_batch_size > 0 else args.ttt_batch_size
    if ttt_bsz == 0:
        return float('inf'), float('inf')

    chunk_size = args.ttt_chunk_size; eval_sl = args.ttt_eval_seq_len
    rank_docs.sort(key=lambda d: (d[1]-2)//chunk_size)
    base_model.eval()
    for p in base_model.parameters(): p.requires_grad_(False)
    lora = BatchedTTTLoRA(ttt_bsz, base_model, args.ttt_lora_rank).to(device)
    opt  = torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1,args.beta2), eps=1e-10)
    ls = torch.zeros((), device=device, dtype=torch.float64)
    bs = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    for bi in range(0, len(rank_docs), ttt_bsz):
        batch = rank_docs[bi:bi+ttt_bsz]; bsz = len(batch)
        if bsz == ttt_bsz:
            cur_lora, cur_opt = lora, opt; cur_lora.reset(); _reset_ttt_opt(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, args.ttt_lora_rank).to(device)
            cur_opt  = torch.optim.Adam(cur_lora.parameters(), lr=args.ttt_lora_lr,
                                        betas=(args.beta1,args.beta2), eps=1e-10)
        pred_lens  = [dl-1 for _,dl in batch]
        num_chunks = [(pl+chunk_size-1)//chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)
        for ci in range(max_nc):
            cst = _chunk_window(ci,(ci+1)*chunk_size,ci+1,chunk_size,eval_sl)
            csz, co = cst[1], cst[2]
            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc-1 for nc in num_chunks)
            x = torch.zeros(bsz, csz, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, csz, dtype=torch.int64, device=device)
            doc_info = []
            for b in range(bsz):
                if not active[b]: doc_info.append((0,0)); continue
                ds,dl = batch[b]
                ws,wl,co2,cl = _chunk_window(ci,pred_lens[b],num_chunks[b],chunk_size,eval_sl)
                chunk = all_tokens[ds+ws:ds+ws+wl+1].to(device=device, dtype=torch.int64)
                x[b,:wl]=chunk[:-1]; y[b,:wl]=chunk[1:]
                doc_info.append((co2,cl))
            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]: continue
                    c0, cl = doc_info[b]
                    if cl <= 0: continue
                    lbl = ptl[b, c0:c0+cl].to(torch.float64)
                    prev = x[b, c0:c0+cl]; tgt = y[b, c0:c0+cl]
                    tb = bb_lut[tgt].to(torch.float64)
                    tb += (hs_lut[tgt] & ~ib_lut[prev]).to(torch.float64)
                    ls += lbl.sum(); bs += tb.sum(); tc += cl
            if needs_train:
                mask = torch.tensor([float(ci<num_chunks[b]-1) for b in range(bsz)], device=device)
                per_doc = ptl[:, co:co+chunk_size].mean(dim=-1)
                cur_opt.zero_grad(); (per_doc*mask).sum().backward(); cur_opt.step()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls, op=dist.ReduceOp.SUM)
        dist.all_reduce(bs, op=dist.ReduceOp.SUM)
        dist.all_reduce(tc, op=dist.ReduceOp.SUM)
    vl = float(ls.item()/tc.item()) if tc.item() > 0 else float('inf')
    vb = float((ls.item()/math.log(2.0))/bs.item()) if bs.item() > 0 else float('inf')
    return vl, vb

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    global _zeropower_ns5
    # Jupyter-safe code reading
    try:
        _f = Path(__file__)
        code = _f.read_text(encoding="utf-8") if _f.exists() else "# pargolf_zero"
    except (NameError, Exception):
        import inspect
        code = inspect.getsource(inspect.getmodule(main)) or "# pargolf_zero"
    args = Hyperparameters()

    # ── Layer 1: Detect platform ──────────────────
    platform = detect_platform()

    # ── Distributed setup ─────────────────────────
    distributed  = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank         = int(os.environ.get("RANK", "0"))
    world_size   = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank   = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    compat_sdp(platform)

    # ── Override batch/seq from platform if not user-set ──
    if "TRAIN_BATCH_TOKENS" not in os.environ and platform.tier < 3:
        args.train_batch_tokens = platform.__class__.__new__(platform.__class__)
        args.train_batch_tokens = 65_536 if platform.vram_mb < 20000 else 262_144
    if "TRAIN_SEQ_LEN" not in os.environ and platform.tier < 2:
        args.train_seq_len = 512 if platform.vram_mb < 20000 else 1024

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master_process: return
        if console: print(msg)
        if logfile:
            with open(logfile, "a", encoding="utf-8") as f: print(msg, file=f)

    log0(code, console=False)
    log0("="*100, console=False)
    log0(f"[L1-COMPAT] {platform.device_name} tier={platform.tier} "
         f"sm{platform.compute_cap[0]}.{platform.compute_cap[1]} "
         f"vram={platform.vram_mb}MB flash={platform.use_flash_attn} compile={platform.use_compile}")
    log0(f"[L2-TRAIN]  matrix_lr={args.matrix_lr} warmdown={args.warmdown_iters} "
         f"muon_wd={args.muon_weight_decay} fp16_embed={args.fp16_embedding} swa={args.use_swa}")
    log0(f"[L3-COMPRESS] range_penalty={args.weight_range_penalty} qat_steps={args.qat_steps} int6={args.use_int6_middle}")
    log0(f"[L4-EVAL]   sliding_window={args.use_sliding_window} stride={args.eval_stride}")
    log0(f"[L5-ADAPT]  lowrank_penalty={args.lowrank_penalty}")
    log0(f"[L6-BIGRAM] use_bigram={args.use_bigram} hash_size={args.bigram_hash_size}")

    # ── Seeds ─────────────────────────────────────
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # ── Tokenizer + val data ──────────────────────
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"Vocab mismatch: {sp.vocab_size()} vs {args.vocab_size}")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bb_lut, hs_lut, ib_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_tokens:{val_tokens.numel()-1}")

    # ── Model ─────────────────────────────────────
    base_model = GPT(args, platform).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
        if isinstance(m, Rotary): m.inv_freq.data = m.inv_freq.data.float()
    restore_low_dim_fp32(base_model)
    _zeropower_ns5 = torch.compile(_zeropower_ns5) if platform.use_compile else _zeropower_ns5
    compiled_model = compat_compile(base_model, platform)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # ── Layer 2: Optimizers ───────────────────────
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n,p in block_params if p.ndim==2 and not any(c in n for c in CONTROL_PATTERNS)]
    scalar_params = [p for n,p in block_params if p.ndim<2  or  any(c in n for c in CONTROL_PATTERNS)]
    if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok    = torch.optim.Adam([{"params":[base_model.tok_emb.weight],"lr":token_lr,"base_lr":token_lr}],
                                         betas=(args.beta1,args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon   = MuonWD(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                               backend_steps=args.muon_backend_steps, weight_decay=args.muon_weight_decay)
    for g in optimizer_muon.param_groups: g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params":scalar_params,"lr":args.scalar_lr,"base_lr":args.scalar_lr}],
                                         betas=(args.beta1,args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        opt_head = torch.optim.Adam([{"params":[base_model.lm_head.weight],"lr":args.head_lr,"base_lr":args.head_lr}],
                                     betas=(args.beta1,args.beta2), eps=args.adam_eps, fused=True)
        optimizers.insert(1, opt_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} world_size:{world_size} grad_accum:{grad_accum_steps}")

    # SWA setup
    swa_model = None
    swa_n = 0
    if args.use_swa:
        swa_model = copy.deepcopy(base_model)
        for p in swa_model.parameters(): p.requires_grad_(False)
        log0(f"[SWA] enabled, will average from {args.swa_start_frac*100:.0f}% of training")

    # ── Data loader ───────────────────────────────
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations-step)/max(args.warmdown_iters,1),0.0) if ws<=step<args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wc_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    # ── Warmup ────────────────────────────────────
    if args.warmup_steps > 0:
        init_state = {n: t.detach().cpu().clone() for n,t in base_model.state_dict().items()}
        init_opts  = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps-1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for o in optimizers: o.step()
            zero_grad_all()
            if args.warmup_steps<=20 or (ws+1)%10==0 or ws+1==args.warmup_steps:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(init_state, strict=True)
        for o, s in zip(optimizers, init_opts): o.load_state_dict(s)
        zero_grad_all()
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── Training loop ─────────────────────────────
    training_ms = 0.0; stop_after: int | None = None
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after is not None and step >= stop_after)
        should_val = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)

        if should_val:
            torch.cuda.synchronize()
            training_ms += 1000.0 * (time.perf_counter() - t0)
            # Use baseline eval during training (fast), sliding window only at final
            use_sw = args.use_sliding_window and last_step
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                              val_tokens, bb_lut, hs_lut, ib_lut, use_sliding=use_sw)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} "
                 f"train_time:{training_ms:.0f}ms step_avg:{training_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()

        if last_step: break

        elapsed_ms = training_ms + 1000.0*(time.perf_counter()-t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)

        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps-1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
                # Layer 3: weight range penalty
                loss = loss + weight_range_loss(base_model, args.weight_range_penalty)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Layer 5: low-rank penalty (no-grad, just logs)
        if step % 100 == 0:
            lr_pen = lowrank_penalty_loss(base_model, args.lowrank_penalty)

        # Layer 3: QAT in final steps
        qat_active = args.qat_enabled and args.qat_steps > 0 and stop_after is not None and \
                     step >= (stop_after - args.qat_steps)

        frac = min(step/args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        cur_mom = (1-frac)*args.muon_momentum_warmup_start + frac*args.muon_momentum
        for g in optimizer_muon.param_groups: g["momentum"] = cur_mom

        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers: opt.step()
        zero_grad_all()

        step += 1
        approx_ms = training_ms + 1000.0*(time.perf_counter()-t0)

        # SWA: accumulate weights after swa_start_frac of training
        if swa_model is not None and stop_after is not None:
            swa_start = int(stop_after * args.swa_start_frac)
            if step >= swa_start:
                swa_n += 1
                with torch.no_grad():
                    for sp, bp in zip(swa_model.parameters(), base_model.parameters()):
                        sp.data.mul_(1 - 1/swa_n).add_(bp.data, alpha=1/swa_n)
        if args.train_log_every > 0 and (step<=10 or step%args.train_log_every==0 or stop_after is not None):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached = max_wc_ms is not None and approx_ms >= max_wc_ms
        if distributed and max_wc_ms is not None:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX); reached = bool(rt.item())
        if stop_after is None and reached: stop_after = step

    log0(f"peak_mem:{torch.cuda.max_memory_allocated()//1024//1024}MiB")

    # Use SWA weights if available
    if swa_model is not None and swa_n > 0:
        log0(f"[SWA] using averaged model (n={swa_n} snapshots)")
        base_model.load_state_dict(swa_model.state_dict(), strict=True)

    # ── Serialization ─────────────────────────────
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        log0(f"model_raw_bytes:{os.path.getsize('final_model.pt')}")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), args.fp16_embedding)
    qbuf = io.BytesIO(); torch.save(quant_obj, qbuf); qraw = qbuf.getvalue()

    # Layer 3: try zstd first (better compression), fallback to zlib
    if HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=22)
        qblob = cctx.compress(qraw)
        compress_method = "zstd-22"
    else:
        qblob = zlib.compress(qraw, level=9)
        compress_method = "zlib-9"
    if master_process:
        with open("final_model.int8.ptz", "wb") as f: f.write(qblob)
        qfile_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes  = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(f"Serialized model {compress_method}: {qfile_bytes} bytes (ratio:{ratio:.2f}x)")
        log0(f"Total submission size: {qfile_bytes+code_bytes} bytes")

    # ── Roundtrip validation ──────────────────────
    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f: qblob_disk = f.read()
    # Decompress: try zstd first, fallback to zlib
    try:
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            qraw_disk = dctx.decompress(qblob_disk)
        else:
            qraw_disk = zlib.decompress(qblob_disk)
    except Exception:
        qraw_disk = zlib.decompress(qblob_disk)
    qstate = torch.load(io.BytesIO(qraw_disk), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int8(qstate), strict=True)
    torch.cuda.synchronize(); t_qe = time.perf_counter()
    # Layer 4: sliding window for final roundtrip eval
    qvl, qvb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                         val_tokens, bb_lut, hs_lut, ib_lut, use_sliding=args.use_sliding_window)
    torch.cuda.synchronize()
    log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_qe):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

    # ── TTT eval (competition score) ─────────────
    torch._dynamo.reset()
    torch.cuda.synchronize(); t_ttt = time.perf_counter()
    ttt_vl, ttt_vb = eval_val_ttt_lora(args, base_model, rank, world_size,
                                         device, bb_lut, hs_lut, ib_lut, platform)
    torch.cuda.synchronize()
    log0(f"final_int8_ttt_lora val_loss:{ttt_vl:.4f} val_bpb:{ttt_vb:.4f} "
         f"eval_time:{1000.0*(time.perf_counter()-t_ttt):.0f}ms")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
