"""
combined_v1.py — FarnsworthEngine + QAT + DecayTTT + Reptile

Base: PR #281 (score 1.1374 bpb) — U-Net skips, SmearGate, BigramHash, SWA, OrthoInit
Added from PR #295: QAT with STE (int5 MLP / int6 attn), Backout, BigramHash(10240), zstd-22
Added from PR #302: Online Causal TTT with decay prior, Reptile meta-learning

torchrun --standalone --nproc_per_node=8 combined_v1.py
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
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ── ZSTD (required, replaces zlib from #281) [#295, #302] ──
import zstandard as _zstd_mod
_ZSTD_COMPRESSOR = _zstd_mod.ZstdCompressor(level=22)
_ZSTD_DECOMPRESSOR = _zstd_mod.ZstdDecompressor()

def compress_blob(data: bytes) -> bytes:
    return _ZSTD_COMPRESSOR.compress(data)

def decompress_blob(data: bytes) -> bytes:
    return _ZSTD_DECOMPRESSOR.decompress(data)

# ── FLASH ATTENTION [#281] ──
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _HAS_FA3 = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_3_func
        _HAS_FA3 = True
    except ImportError:
        try:
            from flash_attn import flash_attn_func as flash_attn_3_func
            _HAS_FA3 = True
        except ImportError:
            _HAS_FA3 = False

# ─────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────
class Hyperparameters:
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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.0))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer
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
    muon_wd = float(os.environ.get("MUON_WD", 0.02))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    adam_wd = float(os.environ.get("ADAM_WD", 0.01))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))

    # SWA [#281]
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "1")))
    swa_every = int(os.environ.get("SWA_EVERY", 50))   # aggressive: every 50 steps like #281
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.4))

    # SmearGate + BigramHash [#281, #295: larger bucket 10240]
    smear_enabled = bool(int(os.environ.get("SMEAR_ENABLED", "1")))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 10240))  # #295: 10240 vs #281's 4096
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    # Backout [#295]
    backout_enabled = bool(int(os.environ.get("BACKOUT_ENABLED", "1")))
    backout_init = float(os.environ.get("BACKOUT_INIT", 0.2))

    # QAT [#295]
    qat_enabled = bool(int(os.environ.get("QAT_ENABLED", "1")))

    # Magnitude pruning [#295]
    prune_frac = float(os.environ.get("PRUNE_FRAC", 0.08))

    # Online Causal TTT with decay prior [#302]
    ttt_eval_lr = float(os.environ.get("TTT_EVAL_LR", 2e-3))
    ttt_eval_momentum = float(os.environ.get("TTT_EVAL_MOMENTUM", 0.9))
    ttt_eval_grad_clip = float(os.environ.get("TTT_EVAL_GRAD_CLIP", 1.0))
    ttt_eval_decay = float(os.environ.get("TTT_EVAL_DECAY", 0.02))
    ttt_eval_adapt_last_n = int(os.environ.get("TTT_EVAL_ADAPT_LAST_N", 3))

    # Reptile meta-learning [#302]
    meta_ttt_enabled = bool(int(os.environ.get("META_TTT_ENABLED", "1")))
    meta_ttt_start_frac = float(os.environ.get("META_TTT_START_FRAC", 0.90))
    meta_ttt_inner_steps = int(os.environ.get("META_TTT_INNER_STEPS", 1))
    meta_ttt_inner_lr = float(os.environ.get("META_TTT_INNER_LR", 2e-3))
    meta_ttt_epsilon = float(os.environ.get("META_TTT_EPSILON", 0.3))
    meta_ttt_log_every = int(os.environ.get("META_TTT_LOG_EVERY", 50))


# ─────────────────────────────────────────────
# MUON OPTIMIZER [#281]
# ─────────────────────────────────────────────
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,skip_weight,skip_weights,smear,backout_lambda,bigram.scale",
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_Q = 99.99984 / 100.0


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
            wd = group.get("weight_decay", 0.0)
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
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


# ─────────────────────────────────────────────
# TOKENIZER-AGNOSTIC EVALUATION [#281]
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
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split too short for seq_len={seq_len}")
    return tokens[: usable + 1]


def _reduce_bpb(val_loss_sum, val_token_count, val_byte_count):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)
    val_loss = (val_loss_sum / val_token_count).item()
    bpb = (val_loss / math.log(2.0)) * (val_token_count.item() / val_byte_count.item())
    return val_loss, bpb


def eval_val(args, model, rank, world_size, device, grad_accum_steps,
             val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
             eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(f"VAL_BATCH_SIZE too small: need at least {seq_len} tokens per rank")
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_tokens[bss * seq_len: bse * seq_len + 1].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            val_token_count += float(y.numel())
            tb = base_bytes_lut[y.reshape(-1)].to(dtype=torch.int16)
            tb += (has_leading_space_lut[y.reshape(-1)] & ~is_boundary_token_lut[x.reshape(-1)]).to(dtype=torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
    val_loss, bpb = _reduce_bpb(val_loss_sum, val_token_count, val_byte_count)
    model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────
# SLIDING WINDOW EVAL [#281]
# ─────────────────────────────────────────────
def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride, batch_seqs=32, eval_seq_len=None):
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi: bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws: end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y_batch.reshape(-1), reduction="none"
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bpb = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    base_model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────
# ONLINE CAUSAL TTT WITH DECAY PRIOR [#302]
# Replaces #281's full-weight SGD TTT with superior online variant
# ─────────────────────────────────────────────
def eval_val_ttt(base_model, val_tokens, eval_seq_len, eval_stride, device,
                 base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                 rank, world_size, args, log_fn=None):
    """Online Causal TTT: interleave scoring & adaptation with decay prior.
    p += λ(p₀ - p) after each update — prevents drift [#302].
    Coarser stride for wall-clock budget (ttt_stride = max(eval_stride, seq_len//4)).
    """
    total_tokens = val_tokens.numel()
    ttt_stride = max(eval_stride, eval_seq_len // 4)
    window_starts = list(range(0, total_tokens - 1, ttt_stride))
    my_starts = [ws for i, ws in enumerate(window_starts) if i % world_size == rank]

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Identify MLP params in last N blocks to adapt [#302]
    adapt_last_n = args.ttt_eval_adapt_last_n
    num_blocks = len(base_model.blocks)
    adapt_block_indices = list(range(max(0, num_blocks - adapt_last_n), num_blocks))
    adapt_params = []
    for bi in adapt_block_indices:
        block = base_model.blocks[bi]
        for name, p in block.mlp.named_parameters():
            if "weight" in name and ("fc" in name or "proj" in name):
                adapt_params.append(p)

    theta_original = {id(p): p.data.clone() for p in adapt_params}
    ttt_optimizer = torch.optim.SGD(adapt_params, lr=args.ttt_eval_lr, momentum=args.ttt_eval_momentum)
    decay = args.ttt_eval_decay
    sc = base_model.logit_softcap

    base_model.eval()
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
        chunk = val_tokens[ws: wend].to(device=device, dtype=torch.int64)
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

        # Backward pass + decay prior [#302]
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
            partial_bpb = (
                (val_loss_sum.item() / max(val_token_count.item(), 1.0)) / math.log(2.0)
                * (val_token_count.item() / max(val_byte_count.item(), 1.0))
            )
            log_fn(f"ttt_eval:progress windows:{window_idx+1}/{total_my_windows} rank:{rank} partial_bpb:{partial_bpb:.4f}")

    # Restore original params after TTT eval [#302]
    with torch.no_grad():
        for p in adapt_params:
            p.data.copy_(theta_original[id(p)])
    for p in base_model.parameters():
        p.requires_grad_(True)

    val_loss, bpb = _reduce_bpb(val_loss_sum, val_token_count, val_byte_count)
    base_model.train()
    return val_loss, bpb


# ─────────────────────────────────────────────
# MIXED INT5/INT6 QUANTIZATION + ZSTD-22 [#295, #302]
# Int5 for MLP (clip=15), Int6 for attention (clip=31), fp16 for embeddings
# ─────────────────────────────────────────────
FP16_KEEP_PATTERNS = ("tok_emb",)
MLP_PATTERNS = (".mlp.",)
INT5_CLIP = 15
INT6_CLIP = 31


def _classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if "bigram" in name:
        return "bigram"
    if ".attn." in name or (".proj." in name and ".mlp." not in name):
        return "attn"
    return "other"


def quantize_intN_per_row(t: Tensor, clip: int) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip).clamp_min(1.0 / clip).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip + 1), clip).to(torch.int8)
        return q.contiguous(), scale.contiguous()
    amax = t32.abs().max().item()
    scale = torch.tensor(max(amax / clip, 1e-12), dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -(clip + 1), clip).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def quantize_float_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    """Fallback int8 for tensors not in mlp/attn categories."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty((t32.shape[0],))
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_mixed(state_dict: dict) -> tuple[dict, dict]:
    """Mixed int5/int6/int8 quantization with per-row scales [#295, #302]."""
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        # Non-float or tiny tensors: passthrough
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough"
            continue
        # Control tensors: keep fp32
        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue
        # Embeddings: fp16
        if any(p in name for p in FP16_KEEP_PATTERNS):
            result[name] = t.to(torch.float16).contiguous()
            meta[name] = "passthrough_fp16"
            continue
        # Int5 MLP, Int6 attention/bigram [#295]
        if cat in ("mlp", "attn", "bigram") and t.ndim >= 1:
            clip = INT5_CLIP if cat == "mlp" else INT6_CLIP
            q, s = quantize_intN_per_row(t, clip)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": f"int{5 if cat == 'mlp' else 6}"}
            continue
        # Fallback: int8
        if t.ndim == 2:
            q, s = quantize_float_tensor_int8(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
        else:
            clip_abs = float(t.float().abs().max().item()) if t.numel() else 0.0
            sc = max(clip_abs / 127.0, 1.0 / 127.0)
            q = torch.clamp(torch.round(t.float() / sc), -127, 127).to(torch.int8).contiguous()
            result[name + ".q"] = q
            result[name + ".scale"] = torch.tensor(sc, dtype=torch.float32)
            meta[name] = {"type": "int8"}
    return result, meta


def dequantize_state_dict_mixed(result: dict, meta: dict, template_sd: dict) -> dict:
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        if isinstance(info, str) and info.startswith("passthrough"):
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


# ─────────────────────────────────────────────
# DATA LOADING [#281]
# ─────────────────────────────────────────────
def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * np.dtype("<u2").itemsize
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}")
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
            chunks.append(self.tokens[self.pos: self.pos + k])
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
        local = chunk[start: start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


# ─────────────────────────────────────────────
# TRANSFORMER MODULES
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Linear with optional QAT STE quantization during training [#295].
    Set qat_levels per-instance: 15 for MLP (int5), 31 for attn (int6), 0 = disabled.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_levels: int = 0  # per-instance; set in main() after model creation

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        # QAT with STE: straight-through estimator [#295]
        if self.training and w.ndim == 2 and self.qat_levels > 0:
            w_f = w.float()
            clip_abs = w_f.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
            scale = clip_abs / float(self.qat_levels)
            w_q = (w_f / scale).round().clamp(-self.qat_levels, self.qat_levels) * scale
            w = w + (w_q.to(w.dtype) - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """NTK-aware RoPE [#281]."""
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (self._cos_cached is None or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (self.dim / (self.dim - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=1024)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        if _HAS_FA3:
            y = flash_attn_3_func(q, k, v, causal=True)
            y = y.reshape(bsz, seqlen, dim)
        else:
            q2 = q.transpose(1, 2)
            k2 = k.transpose(1, 2)
            v2 = v.transpose(1, 2)
            if self.num_kv_heads != self.num_heads:
                rep = self.num_heads // self.num_kv_heads
                k2 = k2.repeat_interleave(rep, dim=1)
                v2 = v2.repeat_interleave(rep, dim=1)
            y = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
            y = y.transpose(1, 2).reshape(bsz, seqlen, dim)
        return self.proj(y)


class SmearGate(nn.Module):
    """Sigmoid-gated token-smoothing [#281]."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash-based bigram context [#281, #295: bucket size 10240]."""
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: float, rope_base: float, qk_gain_init: float):
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
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: float,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 smear_enabled: bool = True,
                 backout_enabled: bool = True, backout_init: float = 0.2):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.smear_enabled = smear_enabled
        self.backout_enabled = backout_enabled
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.smear = SmearGate(model_dim) if smear_enabled else None

        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Backout: learned residual subtraction [#295]
        self.backout_lambda = nn.Parameter(backout_init * torch.ones(1)) if backout_enabled else None

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if param.ndim == 2 and param.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
                nn.init.orthogonal_(param, gain=1.0)
                if "proj.weight" in name:
                    with torch.no_grad():
                        param.mul_(1.0 / math.sqrt(2 * self.num_layers))
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
            with torch.no_grad():
                U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
                target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
                self.tok_emb.weight.data = (U * target_S[None, :]) @ V
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)
        nl = len(self.blocks)
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(nl - 1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
                block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])

    def _run_layers(self, x: Tensor, x0: Tensor) -> Tensor:
        skips: list[Tensor] = []
        backout_layer = self.num_layers // 2
        x_backout: Tensor | None = None

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if i == backout_layer:
                x_backout = x

        for i in range(self.num_decoder_layers):
            li = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[li](x, x0)
            if li == backout_layer and x_backout is None:
                x_backout = x

        # Backout: subtract early-layer residual [#295]
        if self.backout_lambda is not None and x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout

        return x

    def _embed(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (self.tok_emb.weight.shape[1],))
        if self.smear is not None:
            x = self.smear(x)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x0 = self._embed(input_ids)
        x = self._run_layers(x0, x0)
        x_flat = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x0 = self._embed(input_ids)
        x = self.final_norm(self._run_layers(x0, x0))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be a divisor of 8")
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
    from torch.backends.cuda import (enable_cudnn_sdp, enable_flash_sdp,
                                     enable_math_sdp, enable_mem_efficient_sdp)
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
    log0(f"Python {sys.version}", console=False)
    log0(f"PyTorch {torch.__version__}", console=False)
    log0(subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, check=False).stdout, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    eval_sl = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, eval_sl)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_tokens:{val_tokens.numel() - 1}")

    # ── MODEL [#281 base + #295 QAT + Backout + #302 features] ──
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
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        smear_enabled=args.smear_enabled,
        backout_enabled=args.backout_enabled,
        backout_init=args.backout_init,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    # Enable QAT: int5 for MLP, int6 for attention [#295]
    if args.qat_enabled:
        for name, m in base_model.named_modules():
            if isinstance(m, CastedLinear) and m.weight.ndim == 2 and m.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
                m.qat_levels = INT5_CLIP if ".mlp." in name else INT6_CLIP
        log0(f"qat:enabled int5_mlp(clip={INT5_CLIP}) int6_attn(clip={INT6_CLIP})")

    compile_disabled = bool(int(os.environ.get("TORCH_COMPILE_DISABLE", "0")))
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if not compile_disabled else base_model
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # ── OPTIMIZERS [#281] ──
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]

    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if base_model.smear is not None:
        scalar_params.append(base_model.smear.gate)
    if base_model.backout_lambda is not None:
        scalar_params.append(base_model.backout_lambda)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.scale)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    tok_param_groups = [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}]
    if base_model.bigram is not None:
        tok_param_groups.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
        if base_model.bigram.proj is not None:
            matrix_params.append(base_model.bigram.proj.weight)

    optimizer_tok = torch.optim.AdamW(tok_param_groups,
                                      betas=(args.beta1, args.beta2), eps=args.adam_eps,
                                      weight_decay=args.adam_wd, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum,
                          backend_steps=args.muon_backend_steps, weight_decay=args.muon_wd)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, weight_decay=args.adam_wd, fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0(f"bigram_vocab_size:{args.bigram_vocab_size} backout:{args.backout_enabled} qat:{args.qat_enabled}")
    log0(f"meta_ttt:{args.meta_ttt_enabled} start_frac:{args.meta_ttt_start_frac}")

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
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if step >= warmdown_start else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem_ms / max(wd_ms, 1e-9) if rem_ms <= wd_ms else 1.0

    def apply_optimizers(step: int, scale: float) -> None:
        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for g in optimizer_muon.param_groups:
            g["momentum"] = muon_mom
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

    # ── WARMUP [#281] ──
    if args.warmup_steps > 0:
        initial_model_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (ws + 1) % 10 == 0:
                log0(f"warmup_step:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # ── TRAINING LOOP ──
    swa_state: dict | None = None
    swa_count = 0
    training_time_ms = 0.0
    stop_after_step: int | None = None

    # Pre-compute Reptile meta-TTT param set [#302]
    _num_blocks = len(base_model.blocks)
    _adapt_indices = range(max(0, _num_blocks - args.ttt_eval_adapt_last_n), _num_blocks)
    meta_adapt_ids: set[int] = set()
    for _bi in _adapt_indices:
        for _n, _p in base_model.blocks[_bi].mlp.named_parameters():
            if "weight" in _n and ("fc" in _n or "proj" in _n):
                meta_adapt_ids.add(id(_p))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                         val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Elapsed fraction for Reptile gate [#302]
        if max_wallclock_ms is not None and max_wallclock_ms > 0:
            elapsed_frac = elapsed_ms / max_wallclock_ms
        else:
            elapsed_frac = step / max(args.iterations, 1)

        # SWA collection [#281, every swa_every steps during warmdown]
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
                swa_count = 1
                log0(f"swa:start step:{step}")
            else:
                for n, t in base_model.state_dict().items():
                    swa_state[n] += t.detach().cpu()
                swa_count += 1

        # ── Reptile Meta-TTT (last ~10% of training) [#302] ──
        if args.meta_ttt_enabled and elapsed_frac >= args.meta_ttt_start_frac:
            if args.meta_ttt_log_every > 0 and (step % args.meta_ttt_log_every == 0):
                log0(f"meta_ttt:step:{step}/{args.iterations} frac:{elapsed_frac:.4f}")

            theta_0 = {n: p.data.clone() for n, p in base_model.named_parameters() if id(p) in meta_adapt_ids}

            # Inner loop: simulate TTT by gradient descent on training data
            for _inner in range(args.meta_ttt_inner_steps):
                approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
                reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
                if distributed and max_wallclock_ms is not None:
                    rct = torch.tensor(int(reached_cap), device=device)
                    dist.all_reduce(rct, op=dist.ReduceOp.MAX)
                    reached_cap = bool(rct.item())
                if reached_cap:
                    if stop_after_step is None:
                        stop_after_step = step
                    break
                zero_grad_all()
                for ms in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = ms == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        inner_loss = model(x, y)
                    (inner_loss * grad_scale).backward()
                with torch.no_grad():
                    for p in base_model.parameters():
                        if p.grad is not None and id(p) in meta_adapt_ids:
                            p.data.sub_(args.meta_ttt_inner_lr * p.grad)
                zero_grad_all()

            if stop_after_step is not None and step >= stop_after_step:
                step += 1
                continue

            # Reptile interpolation: move toward inner-adapted params [#302]
            with torch.no_grad():
                for n, p in base_model.named_parameters():
                    if id(p) in meta_adapt_ids:
                        p.data.copy_(theta_0[n] + args.meta_ttt_epsilon * (p.data - theta_0[n]))

            # Normal optimizer step after Reptile
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps
            apply_optimizers(step, scale)

        else:
            # ── Normal training step ──
            zero_grad_all()
            train_loss = torch.zeros((), device=device)
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    loss = model(x, y)
                train_loss += loss.detach()
                (loss * grad_scale).backward()
            train_loss /= grad_accum_steps
            apply_optimizers(step, scale)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            rct = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(rct, op=dist.ReduceOp.MAX)
            reached_cap = bool(rct.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB "
         f"reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")

    # ── POST-TRAINING PIPELINE ──

    # 1. SWA averaging [#281]
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"SWA: averaging {swa_count} checkpoints")
        avg = {n: (t / swa_count).to(dtype=base_model.state_dict()[n].dtype) for n, t in swa_state.items()}
        base_model.load_state_dict(avg, strict=True)
    else:
        log0("post_train: skipping SWA (no checkpoints collected)")

    # 2. Magnitude pruning [#295]
    if args.prune_frac > 0:
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if param.ndim == 2 and param.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
                    threshold = torch.quantile(param.abs().float().flatten(), args.prune_frac)
                    param.masked_fill_(param.abs() < threshold, 0.0)
        log0(f"Magnitude pruning: zeroed {args.prune_frac*100:.1f}% of large weight entries")

    # 3. Int5/Int6 quantization + zstd-22 [#295, #302]
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    quant_result, quant_meta = quantize_state_dict_mixed(sd_cpu)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    model_blob = compress_blob(quant_buf.getvalue())
    code_bytes = len(code.encode("utf-8"))
    model_bytes = len(model_blob)
    total_size = code_bytes + model_bytes
    log0(f"Compressed model: {model_bytes} bytes (zstd-22)")
    log0(f"Code size: {code_bytes} bytes")
    log0(f"Total submission size: {total_size} bytes ({total_size/1e6:.2f} MB)")
    if total_size > 16_000_000:
        log0(f"WARNING: Total size {total_size} exceeds 16MB limit!")
    else:
        log0(f"Size OK: {total_size/1e6:.2f} MB")

    if master_process:
        with open("final_model.int6.ptz", "wb") as f:
            f.write(model_blob)
    if distributed:
        dist.barrier()

    # 4. Roundtrip dequantize
    with open("final_model.int6.ptz", "rb") as f:
        model_blob_loaded = f.read()
    quant_state = torch.load(io.BytesIO(decompress_blob(model_blob_loaded)), map_location="cpu", weights_only=False)
    deq_state = dequantize_state_dict_mixed(quant_state["w"], quant_state["m"], sd_cpu)
    base_model.load_state_dict(deq_state, strict=True)

    # 5. Online Causal TTT Eval with decay prior [#302]
    log0("post_train: starting online TTT eval (decay prior)")
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    val_tokens_full = load_validation_tokens(args.val_files, eval_sl) if eval_sl != val_seq_len else val_tokens
    ttt_loss, ttt_bpb = eval_val_ttt(
        base_model, val_tokens_full, eval_sl, args.eval_stride, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        rank, world_size, args, log_fn=log0,
    )
    torch.cuda.synchronize()
    log0(f"final_ttt val_loss:{ttt_loss:.4f} val_bpb:{ttt_bpb:.4f} eval_time:{1000*(time.perf_counter()-t_ttt):.0f}ms")
    log0(f"final_ttt_exact val_loss:{ttt_loss:.8f} val_bpb:{ttt_bpb:.8f}")

    # 6. Sliding window eval (sanity check / fallback score)
    log0("post_train: starting sliding window eval")
    torch.cuda.synchronize()
    t_eval = time.perf_counter()
    q_vl, q_vb = eval_val_sliding(
        args, base_model, rank, world_size, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=args.eval_stride, batch_seqs=32, eval_seq_len=eval_sl,
    )
    torch.cuda.synchronize()
    log0(f"final_int6_sliding val_loss:{q_vl:.4f} val_bpb:{q_vb:.4f} eval_time:{1000*(time.perf_counter()-t_eval):.0f}ms")
    log0(f"final_int6_sliding_exact val_loss:{q_vl:.8f} val_bpb:{q_vb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
