"""
DepthScale-I4: Parameter-Shared Iterative Transformer with 4-bit Training

Novel architecture for Parameter Golf combining:
  - DepthScale (YOCO): 5 physical layers × N iterations = 5N effective depth
  - I4 quantization: 4-bit weight training via STE from step 0
  - Iteration-aware RoPE: distinguishes pass 1 from pass N

32M params at 4 bits = ~16MB. 20 effective layers from 5 physical.
Nobody in the competition has this combination.

Usage:
  # Single GPU test:
  TORCHDYNAMO_DISABLE=1 DEPTH_ITERS=4 python train_depthscale_i4.py

  # 8xH100 competition run:
  DEPTH_ITERS=4 torchrun --standalone --nproc_per_node=8 train_depthscale_i4.py

Env vars:
  DEPTH_ITERS=4          Number of iterations per forward pass (effective depth = 5 × this)
  NUM_PHYSICAL_LAYERS=5  Physical transformer layers (parameter-shared)
  MODEL_DIM=768          Hidden dimension
  I4_ENABLED=1           Enable 4-bit STE training (0=float32 baseline)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
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

# ─── Configuration ──────────────────────────────
DEPTH_ITERS = int(os.environ.get("DEPTH_ITERS", "4"))
I4_ENABLED = int(os.environ.get("I4_ENABLED", "1"))
I4_CLIP = 7  # int4: [-7, 7]
ROPE_ITER_FACTOR = float(os.environ.get("ROPE_ITER_FACTOR", "0.1"))
CURRICULUM = os.environ.get("CURRICULUM", "none")

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_PHYSICAL_LAYERS", 5))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 4))
    tie_embeddings = True
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# ─── I4 Quantized Linear (from anoLLM) ─────────

class I4Linear(nn.Linear):
    """Linear layer with 4-bit STE quantization during training.
    Weights stored in float32 for optimizer, quantized in forward pass.
    Compile-safe: all operations are static."""

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if I4_ENABLED and self.training and w.ndim == 2:
            # STE: quantize to int4 range, gradient flows through
            with torch.no_grad():
                w32 = w.float()
                row_max = w32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
                scale = row_max / I4_CLIP
                w_q = torch.clamp(torch.round(w32 / scale), -I4_CLIP, I4_CLIP)
                w_deq = (w_q * scale).to(w.dtype)
            # STE: forward uses quantized, backward uses original
            w = w + (w_deq - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


# ─── Muon Optimizer ─────────────────────────────

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
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
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
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                g = updates_flat[curr:curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()


# ─── BPB Evaluation ─────────────────────────────

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    base_bytes = np.zeros(sz, dtype=np.int16)
    has_space = np.zeros(sz, dtype=np.bool_)
    is_boundary = np.ones(sz, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary[tid] = False
        if sp.is_byte(tid):
            base_bytes[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes, dtype=torch.int16, device=device),
            torch.tensor(has_space, dtype=torch.bool, device=device),
            torch.tensor(is_boundary, dtype=torch.bool, device=device))


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    tokens_np = np.fromfile(file, dtype="<u2", count=int(header[2]), offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for: {pattern}")
        if CURRICULUM == "reverse":
            def _sd(p):
                h = np.fromfile(p, dtype="<i4", count=256)
                n = min(int(h[2]), 10000)
                t = np.fromfile(p, dtype="<u2", count=n, offset=256*4)
                return float(np.mean(t))
            self.files.sort(key=_sd, reverse=True)
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks, remaining = [], n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
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
        local = chunk[start:start + per_rank_span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, local_batch_seqs):
            be = min(bs + local_batch_seqs, seq_end)
            rs, re = bs * args.train_seq_len, be * args.train_seq_len + 1
            local = val_tokens[rs:re].to(device, dtype=torch.int64, non_blocking=True)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            n = float(y.numel())
            loss_sum += bl.to(torch.float64) * n
            tok_count += n
            tgt, prev = y.reshape(-1), x.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        for t in [loss_sum, tok_count, byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_count
    model.train()
    return float(vl.item()), float((vl.item() / math.log(2.0)) * (tok_count.item() / byte_count.item()))


# ─── Quantization (int8 post-training — proven from baseline) ───

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
INT8_CLIP_Q = 0.9999984

def quantize_float_tensor(t):
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    pt_dtypes = {}
    stats = {"baseline_bytes": 0, "int8_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["baseline_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            continue
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_PATTERNS):
                passthrough[name] = t.float().contiguous()
            else:
                pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(torch.float16).contiguous()
            continue
        q, s = quantize_float_tensor(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    obj = {"__quant_format__": "int8_per_row_v1", "quantized": quantized,
           "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if pt_dtypes:
        obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(torch.float32)
        if s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = pt_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out


# ─── Model Architecture ────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    """RoPE with iteration-aware encoding from DepthScale."""
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def forward(self, seq_len, device, dtype, iteration=0):
        key = (seq_len, device, iteration)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            # Iteration-aware: shift frequencies by iteration * factor
            freqs = freqs + ROPE_ITER_FACTOR * iteration
            self._cache[key] = (freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :])
        cos, sin = self._cache[key]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = I4Linear(dim, dim, bias=False)
        self.c_k = I4Linear(dim, kv_dim, bias=False)
        self.c_v = I4Linear(dim, kv_dim, bias=False)
        self.proj = I4Linear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x, iteration=0):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype, iteration=iteration)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        # GQA: repeat KV heads if needed (compatible with all PyTorch versions)
        if self.num_kv_heads != self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = I4Linear(dim, mlp_mult * dim, bias=False)
        self.proj = I4Linear(mlp_mult * dim, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())


class DepthScaleBlock(nn.Module):
    """Single physical block, reused across iterations via parameter sharing."""
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x, iteration=0):
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x), iteration)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class DepthScaleGPT(nn.Module):
    """
    Parameter-shared iterative GPT.
    5 physical layers × N iterations = 5N effective depth.
    The same weights are reused each iteration with iteration-aware RoPE.
    """
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, logit_softcap, rope_base, qk_gain_init, tied_embed_init_std,
                 depth_iters):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.depth_iters = depth_iters
        self.num_layers = num_layers

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)

        # Physical blocks (shared across iterations)
        self.blocks = nn.ModuleList([
            DepthScaleBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm()

        # Init zero-init weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids, target_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.weight.size(1),))

        # Iterative forward: same blocks, different iteration index
        for iteration in range(self.depth_iters):
            for block in self.blocks:
                x = block(x, iteration=iteration)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = F.linear(x, self.tok_emb.weight)  # Tied embeddings
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")


# ─── Restore fp32 for small params ──────────────

def restore_low_dim_params_to_fp32(module):
    with torch.no_grad():
        for name, p in module.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()


# ─── Main ───────────────────────────────────────

def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt"

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        with open(logfile, "a") as f:
            print(msg, file=f)

    log0(f"DepthScale-I4: depth_iters={DEPTH_ITERS} physical_layers={args.num_layers} effective_depth={DEPTH_ITERS*args.num_layers}")
    log0(f"  I4={I4_ENABLED} dim={args.model_dim} heads={args.num_heads} curriculum={CURRICULUM}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = DepthScaleGPT(
        args.vocab_size, args.num_layers, args.model_dim, args.num_heads,
        args.num_kv_heads, args.mlp_mult, args.logit_softcap, args.rope_base,
        args.qk_gain_init, args.tied_embed_init_std, DEPTH_ITERS,
    ).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, I4Linear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} ({n_params/1e6:.1f}M)")
    log0(f"seed:{args.seed}")

    # Optimizer
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]

    opt_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
                                betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                   betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step else 1.0
        step_ms = elapsed_ms / max(step, 1)
        wd_ms = args.warmdown_iters * step_ms
        rem = max(max_wallclock_ms - elapsed_ms, 0.0)
        return rem / max(wd_ms, 1e-9) if rem <= wd_ms else 1.0

    # Warmup (compile)
    if args.warmup_steps > 0:
        initial_state = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        initial_opts = [copy.deepcopy(o.state_dict()) for o in optimizers]
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad_all()
            for ms in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss * grad_scale).backward()
            for o in optimizers:
                o.step()
            zero_grad_all()
            if ws % 5 == 0 or ws == args.warmup_steps - 1:
                log0(f"warmup:{ws+1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_state, strict=True)
        for o, s in zip(optimizers, initial_opts):
            o.load_state_dict(s)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    training_time_ms, stop_after_step = 0.0, None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, *luts)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vb:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step,1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
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

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for g in opt_muon.param_groups:
            g["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for o in optimizers:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for o in optimizers:
            o.step()
        zero_grad_all()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} train_time:{approx_ms:.0f}ms step_avg:{approx_ms/step:.2f}ms")
        if max_wallclock_ms and approx_ms >= max_wallclock_ms and stop_after_step is None:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")

    # Quantize and save
    if master_process:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=9)
        code_bytes = len(code.encode("utf-8"))
        quant_file_bytes = len(quant_blob)
        log0(f"Serialized model int8+zlib: {quant_file_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)

    # Roundtrip eval
    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=False)
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    qvl, qvb = eval_val(args, base_model, rank, world_size, device, grad_accum_steps, val_tokens, *luts)
    log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvb:.4f} eval_time:{1000*(time.perf_counter()-t_qeval):.0f}ms")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{qvl:.8f} val_bpb:{qvb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
