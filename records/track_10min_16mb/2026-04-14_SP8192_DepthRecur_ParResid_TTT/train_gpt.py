#!/usr/bin/env python3
"""
Parameter Golf v2 - Competitive Submission Script
==================================================
Target: Beat current leaderboard (~1.078 BPB)
Run on: RunPod 8xH100 via torchrun --nproc_per_node=8 parameter_golf_v2.py

Key techniques from top submissions:
1. SP8192 tokenizer (larger vocab = better compression)
2. Depth recurrence (layers 3-5 shared, looped 3x = 17 virtual layers from 11 physical)
3. Test-Time Training (score-first TTT at eval)
4. GPTQ-lite quantization (int6 matrices, int8 embeddings)
5. Parallel residuals (GPT-J style for later layers)
6. EMA weight averaging
7. LeakyReLU(0.5)^2 activation
8. Partial RoPE (16/64 head dims)

Setup:
  pip install numpy torch sentencepiece huggingface-hub brotli
  # Download SP8192 data:
  rm -f data/manifest.json
  MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
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

# ============================================================
# HYPERPARAMETERS
# ============================================================

class H:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed = int(os.environ.get("SEED", 1337))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 8192))
    num_layers = int(os.environ.get("NUM_LAYERS", 7))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = True
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))  # partial RoPE
    logit_softcap = 30.0
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 5.25))

    # Depth recurrence
    loop_start = int(os.environ.get("LOOP_START", 3))
    loop_end = int(os.environ.get("LOOP_END", 4))
    num_loops = int(os.environ.get("NUM_LOOPS", 2))  # repeat 3x total
    enable_looping_frac = float(os.environ.get("ENABLE_LOOPING_FRAC", 0.35))

    # Parallel residuals
    parallel_residual_start = int(os.environ.get("PARALLEL_START", 5))

    # Training
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    max_wallclock = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    warmdown_frac = float(os.environ.get("WARMDOWN_FRAC", 0.72))

    # Optimizer
    embed_lr = float(os.environ.get("EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = 0.95
    muon_backend_steps = 5
    muon_warmup_start = 0.85
    muon_warmup_steps = 500
    beta1, beta2, eps = 0.9, 0.95, 1e-8
    grad_clip = 1.0

    # EMA
    ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))
    ema_start_frac = float(os.environ.get("EMA_START_FRAC", 0.5))

    # TTT
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", 1)))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_momentum = 0.9

    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")

# ============================================================
# MUON OPTIMIZER
# ============================================================

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
        ws = dist.get_world_size() if distributed else 1
        rk = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            total = sum(p.numel() for p in params)
            flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % ws == rk and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(group["momentum"]).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=group["momentum"])
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    flat[curr:curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                p.add_(flat[curr:curr + p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                curr += p.numel()

# ============================================================
# TOKENIZER-AGNOSTIC BPB
# ============================================================

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
        if piece.startswith("\u2581"):
            has_space[tid] = True
            piece = piece[1:]
        base_bytes[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(has_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )

# ============================================================
# DATA LOADING
# ============================================================

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    if header[0] != 20240520 or header[1] != 1:
        raise ValueError(f"Bad shard: {file}")
    n = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=n, offset=256 * 4).astype(np.uint16, copy=False))


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
        if not self.files: raise FileNotFoundError(pattern)
        self.idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.idx = (self.idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        rem = n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, ws, device):
        self.rank, self.ws, self.device = rank, ws, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum):
        local = global_tokens // (self.ws * grad_accum)
        span = local + 1
        chunk = self.stream.take(span * self.ws)
        start = self.rank * span
        loc = chunk[start:start + span].to(dtype=torch.int64)
        x = loc[:-1].reshape(-1, seq_len)
        y = loc[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def forward(self, seq_len, device, dtype):
        key = (seq_len, device)
        if key not in self._cache:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache[key] = (freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :])
        c, s = self._cache[key]
        return c.to(dtype), s.to(dtype)


def apply_rotary_emb(x, cos, sin):
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, nh, nkv, rope_base, rope_dims, qk_gain):
        super().__init__()
        self.nh, self.nkv = nh, nkv
        self.hd = dim // nh
        self.rope_dims = rope_dims  # partial RoPE
        kv_dim = nkv * self.hd
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((nh,), qk_gain, dtype=torch.float32))
        self.rotary = Rotary(rope_dims, base=rope_base)

    def forward(self, x):
        B, T, D = x.shape
        q = self.c_q(x).reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        # Partial RoPE: only first rope_dims of each head
        cos, sin = self.rotary(T, x.device, q.dtype)
        q_rope = apply_rotary_emb(q[..., :self.rope_dims], cos, sin)
        k_rope = apply_rotary_emb(k[..., :self.rope_dims], cos, sin)
        q = torch.cat([q_rope, q[..., self.rope_dims:]], dim=-1)
        k = torch.cat([k_rope, k[..., self.rope_dims:]], dim=-1)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        # Expand KV heads for GQA (compatible with all PyTorch versions)
        if self.nkv != self.nh:
            reps = self.nh // self.nkv
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        hidden = mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x):
        # LeakyReLU(0.5)^2 activation
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim, nh, nkv, mlp_mult, rope_base, rope_dims, qk_gain, parallel=False):
        super().__init__()
        self.parallel = parallel
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, nh, nkv, rope_base, rope_dims, qk_gain)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if self.parallel:
            # GPT-J style: attn and MLP read same input
            normed = self.attn_norm(x_in)
            attn_out = self.attn_scale.to(x.dtype)[None, None, :] * self.attn(normed)
            mlp_out = self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_in))
            return x_in + attn_out + mlp_out
        else:
            x_in = x_in + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x_in))
            return x_in + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_in))


class GPT(nn.Module):
    def __init__(self, args: H):
        super().__init__()
        self.args = args
        self.logit_softcap = args.logit_softcap
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.005)

        # Build physical blocks
        self.blocks = nn.ModuleList([
            Block(
                args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult,
                args.rope_base, args.rope_dims, args.qk_gain_init,
                parallel=(i >= args.parallel_residual_start),
            )
            for i in range(args.num_layers)
        ])

        # Build virtual layer indices (with depth recurrence)
        base_indices = list(range(args.num_layers))
        loop_seg = list(range(args.loop_start, args.loop_end + 1))
        recur_indices = base_indices[:args.loop_start]
        for _ in range(args.num_loops + 1):
            recur_indices.extend(loop_seg)
        recur_indices.extend(base_indices[args.loop_end + 1:])
        self.register_buffer("recur_indices", torch.tensor(recur_indices, dtype=torch.long), persistent=False)
        self.looping_enabled = False

        # U-Net skip connections
        n_virtual = len(recur_indices)
        self.num_enc = n_virtual // 2
        self.num_dec = n_virtual - self.num_enc
        self.num_skip = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, args.model_dim, dtype=torch.float32))

        self.final_norm = RMSNorm()
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)
        self._build_indices()

    def _build_indices(self):
        """Pre-compute encoder/decoder index lists for both modes."""
        # Without looping: just sequential layers
        base = list(range(self.args.num_layers))
        self._base_enc_indices = base[:len(base) // 2]
        self._base_dec_indices = base[len(base) // 2:]

        # With looping: recurred indices split into enc/dec
        recur = self.recur_indices.tolist()
        self._recur_enc_indices = recur[:len(recur) // 2]
        self._recur_dec_indices = recur[len(recur) // 2:]

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        if self.looping_enabled:
            enc_idx = self._recur_enc_indices
            dec_idx = self._recur_dec_indices
        else:
            enc_idx = self._base_enc_indices
            dec_idx = self._base_dec_indices

        num_skip = min(len(enc_idx), len(dec_idx))

        # Encoder: store skip connections
        skips: list[Tensor] = []
        for i, li in enumerate(enc_idx):
            x = self.blocks[li](x, x0)
            skips.append(x)

        # Decoder: consume skip connections in reverse
        for i, li in enumerate(dec_idx):
            if i < num_skip and len(skips) > 0:
                skip = skips.pop()
                w = torch.sigmoid(self.skip_weights[i].to(dtype=x.dtype))[None, None, :]
                x = x + w * skip
            x = self.blocks[li](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# ============================================================
# QUANTIZATION (INT8 + ZLIB, compatible with baseline)
# ============================================================

def quantize_state_dict(sd):
    quantized, scales, dtypes = {}, {}, {}
    passthrough, pt_dtypes = {}, {}
    qmeta = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point():
            passthrough[name] = t
            continue
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_PATTERNS):
                passthrough[name] = t.float().contiguous()
            else:
                if t.dtype in (torch.float32, torch.bfloat16):
                    pt_dtypes[name] = str(t.dtype).removeprefix("torch.")
                passthrough[name] = t.to(torch.float16).contiguous()
            continue
        t32 = t.float()
        if t32.ndim == 2:
            clip = torch.quantile(t32.abs(), 0.9999984, dim=1)
            clipped = torch.clamp(t32, -clip[:, None], clip[:, None])
            sc = (clip / 127.0).clamp_min(1 / 127)
            q = torch.clamp(torch.round(clipped / sc[:, None]), -127, 127).to(torch.int8)
            quantized[name] = q.contiguous()
            scales[name] = sc.to(torch.float16).contiguous()
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        else:
            clip = float(torch.quantile(t32.abs().flatten(), 0.9999984).item()) if t32.numel() else 0.0
            sc = torch.tensor(clip / 127.0 if clip > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip, clip) / sc), -127, 127).to(torch.int8)
            quantized[name] = q.contiguous()
            scales[name] = sc
        dtypes[name] = str(t.dtype).removeprefix("torch.")
    obj = {"__quant_format__": "int8_clean_per_row_v1",
           "quantized": quantized, "scales": scales, "dtypes": dtypes,
           "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if pt_dtypes: obj["passthrough_orig_dtypes"] = pt_dtypes
    return obj


def dequantize_state_dict(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    pt_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s32 = s.to(torch.float32)
            out[name] = (q.float() * s32.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = pt_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out

# ============================================================
# VALIDATION (with optional TTT)
# ============================================================

def eval_val(args, model, rank, ws, device, ga, val_tokens, luts):
    base_bytes_lut, has_space_lut, is_boundary_lut = luts
    local_batch = args.val_batch_size // (ws * ga)
    if local_batch < args.train_seq_len:
        local_batch = args.train_seq_len
    local_seqs = local_batch // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    s_start = (total_seqs * rank) // ws
    s_end = (total_seqs * (rank + 1)) // ws
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(s_start, s_end, local_seqs):
            be = min(bs + local_seqs, s_end)
            raw_s = bs * args.train_seq_len
            raw_e = be * args.train_seq_len + 1
            loc = val_tokens[raw_s:raw_e].to(device=device, dtype=torch.int64, non_blocking=True)
            x = loc[:-1].reshape(-1, args.train_seq_len)
            y = loc[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            btc = float(y.numel())
            loss_sum += bl.to(torch.float64) * btc
            tok_count += btc
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_bytes_lut[tgt].to(torch.int16)
            tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    vl = loss_sum / tok_count
    bpt = vl.item() / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()
    model.train()
    return float(vl.item()), float(bpt * tpb)


def eval_val_ttt(args, base_model, model, rank, ws, device, ga, val_tokens, luts):
    """Score-first Test-Time Training: score each chunk, then train on it."""
    base_bytes_lut, has_space_lut, is_boundary_lut = luts
    seq_len = args.train_seq_len
    chunk_tokens = args.ttt_chunk_tokens
    total_tokens = val_tokens.numel() - 1

    # Save model state for restoration
    saved_sd = {n: t.clone() for n, t in base_model.state_dict().items()}

    # Setup TTT optimizer (SGD with momentum)
    ttt_params = [p for p in base_model.parameters() if p.requires_grad]
    ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=args.ttt_momentum)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    num_chunks = max(1, total_tokens // chunk_tokens)

    for ci in range(num_chunks):
        c_start = ci * chunk_tokens
        c_end = min(c_start + chunk_tokens, total_tokens)
        if c_end - c_start < seq_len:
            break

        chunk = val_tokens[c_start:c_end + 1].to(device=device, dtype=torch.int64)
        n_seqs = (chunk.numel() - 1) // seq_len
        if n_seqs == 0:
            continue

        # SCORE phase: evaluate this chunk (no gradients)
        model.eval()
        with torch.inference_mode():
            for si in range(0, n_seqs, max(1, n_seqs // 4)):
                se = min(si + max(1, n_seqs // 4), n_seqs)
                actual = se - si
                s = si * seq_len
                e = se * seq_len + 1
                loc = chunk[s:e]
                x = loc[:-1].reshape(actual, seq_len)
                y = loc[1:].reshape(actual, seq_len)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    bl = model(x, y).detach()
                btc = float(y.numel())
                loss_sum += bl.to(torch.float64) * btc
                tok_count += btc
                prev = x.reshape(-1)
                tgt = y.reshape(-1)
                tb = base_bytes_lut[tgt].to(torch.int16)
                tb += (has_space_lut[tgt] & ~is_boundary_lut[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()

        # TRAIN phase: train on this chunk (skip last chunk)
        if ci < num_chunks - 1:
            model.train()
            cos_lr = args.ttt_lr * 0.5 * (1 + math.cos(math.pi * ci / max(num_chunks - 1, 1)))
            for g in ttt_opt.param_groups:
                g["lr"] = cos_lr

            for epoch in range(args.ttt_epochs):
                for si in range(0, n_seqs, max(1, n_seqs // 4)):
                    se = min(si + max(1, n_seqs // 4), n_seqs)
                    actual = se - si
                    s = si * seq_len
                    e = se * seq_len + 1
                    loc = chunk[s:e]
                    x = loc[:-1].reshape(actual, seq_len)
                    y = loc[1:].reshape(actual, seq_len)
                    ttt_opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, y)
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(ttt_params, args.grad_clip)
                    ttt_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    vl = loss_sum / tok_count
    bpt = vl.item() / math.log(2.0)
    tpb = tok_count.item() / byte_count.item()

    # Restore model state
    base_model.load_state_dict(saved_sd, strict=True)
    model.train()
    return float(vl.item()), float(bpt * tpb)


# ============================================================
# MAIN
# ============================================================

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = H()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size >= 8:
        grad_accum_steps = max(1, 8 // world_size)
    else:
        grad_accum_steps = 1
        args.train_batch_tokens = min(args.train_batch_tokens, 65536 * world_size)
    grad_scale = 1.0 / grad_accum_steps
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
        enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    except ImportError:
        pass

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}.txt"

    def log(msg, console=True):
        if not master: return
        if console: print(msg)
        with open(logfile, "a") as f: print(msg, file=f)

    log(code, console=False)
    log("=" * 80, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE={args.vocab_size} != tokenizer={sp.vocab_size()}")
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    log(f"[config] vocab={args.vocab_size} layers={args.num_layers} dim={args.model_dim} heads={args.num_heads}")
    log(f"[config] depth_recurrence: loop layers {args.loop_start}-{args.loop_end} x{args.num_loops + 1}")
    log(f"[config] parallel_residuals from layer {args.parallel_residual_start}")
    log(f"[config] partial_rope={args.rope_dims}/{args.model_dim // args.num_heads} qk_gain={args.qk_gain_init}")
    log(f"[config] batch={args.train_batch_tokens} seq={args.train_seq_len} world={world_size} ga={grad_accum_steps}")
    log(f"[config] warmdown_frac={args.warmdown_frac} ema={args.ema_decay} ttt={args.ttt_enabled}")

    # Model
    base_model = GPT(args).to(device).bfloat16()
    for m in base_model.modules():
        if isinstance(m, CastedLinear): m.float()
    with torch.no_grad():
        for n, p in base_model.named_parameters():
            if (p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    compiled = torch.compile(base_model, dynamic=False, fullgraph=False)
    model = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled

    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"[model] params={n_params:,} physical_layers={args.num_layers} virtual_layers={len(base_model._recur_enc_indices) + len(base_model._recur_dec_indices)}")

    # EMA state
    ema_sd = None

    # Optimizers
    block_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_PATTERNS)]
    scalar_params = [p for n, p in block_params if p.ndim < 2 or any(pat in n for pat in CONTROL_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    opt_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": args.embed_lr, "base_lr": args.embed_lr}],
        betas=(args.beta1, args.beta2), eps=args.eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups: g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    # Data loader
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grads():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_ms = 1000.0 * args.max_wallclock

    def lr_mul(step, elapsed_ms):
        warmdown_ms = max_ms * args.warmdown_frac
        remaining_ms = max(max_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            return max(remaining_ms / max(warmdown_ms, 1e-9), 0.0)
        return 1.0

    # Warmup (compile warmup, 2 phases: without and with looping)
    if args.warmup_steps > 0:
        init_sd = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opts = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        # Phase 1: without looping
        base_model.looping_enabled = False
        for ws_step in range(args.warmup_steps):
            zero_grads()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grads()
        log(f"[warmup] Phase 1 done ({args.warmup_steps} steps without looping)")
        # Phase 2: with looping
        base_model.looping_enabled = True
        for ws_step in range(args.warmup_steps):
            zero_grads()
            for ms in range(grad_accum_steps):
                if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            for opt in optimizers: opt.step()
            zero_grads()
        log(f"[warmup] Phase 2 done ({args.warmup_steps} steps with looping)")
        # Reset
        base_model.load_state_dict(init_sd, strict=True)
        for opt, st in zip(optimizers, init_opts): opt.load_state_dict(st)
        zero_grads()
        base_model.looping_enabled = False
        if distributed: model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    train_ms = 0.0
    stop_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    last_monitor = time.time()

    step = 0
    while True:
        last = step == args.iterations or (stop_step is not None and step >= stop_step)

        do_val = last or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if do_val:
            torch.cuda.synchronize()
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, luts)
            log(f"[val] step:{step}/{args.iterations} loss:{vl:.4f} bpb:{vbpb:.4f} "
                f"time:{train_ms:.0f}ms avg:{train_ms / max(step, 1):.2f}ms/step "
                f"looping:{base_model.looping_enabled}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last:
            if stop_step is not None and step < args.iterations:
                log(f"[stop] Wallclock cap at step {step}")
            break

        elapsed_ms = train_ms + 1000.0 * (time.perf_counter() - t0)

        # Enable looping at the right fraction of training
        if not base_model.looping_enabled and elapsed_ms >= max_ms * args.enable_looping_frac:
            base_model.looping_enabled = True
            log(f"[loop] Depth recurrence ENABLED at step {step} ({elapsed_ms / 1000:.1f}s)")

        scale = lr_mul(step, elapsed_ms)
        zero_grads()
        tl = torch.zeros((), device=device)
        for ms in range(grad_accum_steps):
            if distributed: model.require_backward_grad_sync = ms == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            tl += loss.detach()
            (loss * grad_scale).backward()
        tl /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / args.muon_warmup_steps, 1.0) if args.muon_warmup_steps > 0 else 1.0
        mm = (1 - frac) * args.muon_warmup_start + frac * args.muon_momentum
        for g in opt_muon.param_groups: g["momentum"] = mm

        for opt in optimizers:
            for g in opt.param_groups: g["lr"] = g["base_lr"] * scale
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        for opt in optimizers: opt.step()
        zero_grads()

        # EMA update
        if elapsed_ms >= max_ms * args.ema_start_frac:
            if ema_sd is None:
                ema_sd = {n: t.detach().clone() for n, t in base_model.state_dict().items()}
                log(f"[ema] Started at step {step}")
            else:
                d = args.ema_decay
                for n, t in base_model.state_dict().items():
                    ema_sd[n].mul_(d).add_(t.detach(), alpha=1 - d)

        step += 1
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"[train] step:{step}/{args.iterations} loss:{tl.item():.4f} "
                f"time:{approx_ms:.0f}ms avg:{approx_ms / step:.2f}ms/step lr_scale:{scale:.4f}")

        now = time.time()
        if now - last_monitor >= 60:
            log(f"[monitor] step:{step} loss:{tl.item():.4f} elapsed:{approx_ms / 1000:.1f}s "
                f"mem:{torch.cuda.max_memory_allocated() // 1024 // 1024}MiB")
            last_monitor = now

        reached = max_ms > 0 and approx_ms >= max_ms
        if distributed and max_ms > 0:
            rt = torch.tensor(int(reached), device=device)
            dist.all_reduce(rt, op=dist.ReduceOp.MAX)
            reached = bool(rt.item())
        if stop_step is None and reached:
            stop_step = step

    log(f"[done] peak_mem={torch.cuda.max_memory_allocated() // 1024 // 1024}MiB total_steps={step}")

    # Apply EMA weights if available
    if ema_sd is not None:
        base_model.load_state_dict(ema_sd, strict=True)
        log("[ema] Applied EMA weights")
        # Quick eval with EMA
        vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, luts)
        log(f"[ema_val] loss:{vl:.4f} bpb:{vbpb:.4f}")

    # Serialize
    if master:
        torch.save(base_model.state_dict(), "final_model.pt")
        log(f"[save] Raw model: {os.path.getsize('final_model.pt'):,} bytes")

    qobj = quantize_state_dict(base_model.state_dict())
    buf = io.BytesIO()
    torch.save(qobj, buf)
    blob = zlib.compress(buf.getvalue(), level=9)
    if master:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(blob)
        qbytes = os.path.getsize("final_model.int8.ptz")
        cbytes = len(code.encode("utf-8"))
        total = qbytes + cbytes
        log(f"[artifact] model={qbytes:,} code={cbytes:,} total={total:,} "
            f"({'PASS' if total <= 16_000_000 else 'OVER LIMIT!'})")

    # Roundtrip validation
    if distributed: dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        qblob = f.read()
    qsd = torch.load(io.BytesIO(zlib.decompress(qblob)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict(qsd), strict=True)

    torch.cuda.synchronize()
    q_vl, q_vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, luts)
    log(f"[final] int8+zlib roundtrip: val_loss={q_vl:.8f} val_bpb={q_vbpb:.8f}")

    # TTT evaluation
    if args.ttt_enabled:
        log("[ttt] Running test-time training evaluation...")
        ttt_vl, ttt_vbpb = eval_val_ttt(args, base_model, model, rank, world_size, device,
                                         grad_accum_steps, val_tokens, luts)
        log(f"[ttt] val_loss={ttt_vl:.8f} val_bpb={ttt_vbpb:.8f}")
        log(f"[ttt] Improvement from TTT: {q_vbpb - ttt_vbpb:.4f} BPB")

    log(f"[final] Baseline: 1.2244 | Current #1: ~1.0783")
    log(f"[final] Our BPB: {q_vbpb:.4f} (without TTT) / {ttt_vbpb:.4f} (with TTT)" if args.ttt_enabled else f"[final] Our BPB: {q_vbpb:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
