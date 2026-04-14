"""
Experiment: SURGE-M -- Surprise-gated Recurrent Generator for Evolving weight Matrices

Main experiment (Experiment A) from SURGE_M_architecture.md section 12.

The SURGE layers (layers 3, 4) have their attention output projection W_O evolve
multiplicatively during the forward pass:  W_t = (I + gate * u v^T) W_{t-1}
where (u, v, gate) are produced by a recurrent meta-network M (GRU, d_state=64)
that consumes the per-token prediction-error vector compressed to d_err=64.

Chunked forward (chunk_size=64):
  1. apply multiplicative update to W_3, W_4 based on GRU state s
  2. lower layers (0,1,2) -- standard, fixed weights
  3. SURGE layer 3 and 4 -- use W_O override
  4. upper layers (5..8) -- standard
  5. compute logits
  6. compute per-token error_t = one_hot(token_t) - softmax(logits_{t-1})
  7. sequentially update GRU state s over tokens in the chunk
FOMAML: W_t is detached after each update, gradients only flow through the
current-chunk's (u, v, gate); the GRU recurrence is kept in-graph over the full
sequence (4 chunks * 64 = 256 tokens is manageable for T=1024).

NOTE: This model cannot be torch.compile'd due to the dynamic chunk-wise graph.
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


# --- HYPERPARAMETERS ---
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 200))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 256))   # per SURGE-M spec
    num_heads = int(os.environ.get("NUM_HEADS", 4))      # per SURGE-M spec
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # SURGE-M specifics
    chunk_size = int(os.environ.get("CHUNK_SIZE", 64))
    d_err = int(os.environ.get("D_ERR", 64))
    d_state = int(os.environ.get("D_STATE", 64))
    max_drift_fraction = float(os.environ.get("MAX_DRIFT_FRACTION", 0.1))
    gate_init_bias = float(os.environ.get("GATE_INIT_BIAS", -4.6))
    uv_init_std = float(os.environ.get("UV_INIT_STD", 0.0))  # 0 = chicken-egg; >0 breaks symmetry
    surge_layers_str = os.environ.get("SURGE_LAYERS", "3,4")

    lr_meta = float(os.environ.get("LR_META", 3e-4))
    meta_weight_decay = float(os.environ.get("META_WEIGHT_DECAY", 0.01))
    grad_clip = float(os.environ.get("GRAD_CLIP", 1.0))

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
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# --- MUON OPTIMIZER ---
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
            lr, momentum, backend_steps, nesterov = group["lr"], group["momentum"], group["backend_steps"], group["nesterov"]
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
                curr += p.numel()
        return loss


# --- BPB EVAL HELPERS ---
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


# --- DATA LOADING ---
def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
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


# --- VALIDATION ---
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
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss, num_real = model(x, y)
                batch_loss = batch_loss.detach()
            val_loss_sum += batch_loss.to(torch.float64) * float(num_real)
            val_token_count += float(num_real)
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


# --- QUANTIZATION ---
CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
                                  "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights")

def quantize_float_tensor(t):
    t32 = t.float()
    clip_q = 99.99984 / 100.0
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), clip_q, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=torch.float16).contiguous()
    clip_abs = float(torch.quantile(t32.abs().flatten(), clip_q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict):
    quantized, scales, dtypes, passthrough, passthrough_orig_dtypes, qmeta = {}, {}, {}, {}, {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(t.numel()) * int(t.element_size())
        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += int(t.numel()) * int(t.element_size())
            continue
        if t.numel() <= 65536:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            else:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(dtype=torch.float16).contiguous()
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.numel()) * int(kept.element_size())
            continue
        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += int(q.numel()) * int(q.element_size()) + int(s.numel()) * int(s.element_size())
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# --- TRANSFORMER PRIMITIVES ---
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
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Attention layer with optional W_O override for SURGE multiplicative updates."""
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x, W_O_override: Tensor | None = None, return_pre_wo: bool = False):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)  # pre-W_O
        if W_O_override is not None:
            out = F.linear(y, W_O_override.to(y.dtype))
        else:
            out = self.proj(y)
        if return_pre_wo:
            return out, y
        return out


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
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x, x0, W_O_override: Tensor | None = None, return_pre_wo: bool = False):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        if return_pre_wo:
            attn_out, pre_wo = self.attn(self.attn_norm(x), W_O_override=W_O_override, return_pre_wo=True)
        else:
            attn_out = self.attn(self.attn_norm(x), W_O_override=W_O_override)
            pre_wo = None
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        if return_pre_wo:
            return x, pre_wo
        return x


# --- MIN GRU (parallelizable, linear recurrence) ---
class MinGRU(nn.Module):
    """
    Minimal GRU (Feng et al., 2024). Gates depend only on current input, not h_{t-1}.
    Recurrence: h_t = (1 - z_t) * h_{t-1} + z_t * tilde_h_t
       which is the linear form  h_t = a_t * h_{t-1} + b_t  with
       a_t = 1 - z_t,  b_t = z_t * tilde_h_t.
    For T=1024 we use a simple Python loop over time (elementwise on small d_state — cheap).
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(input_size, 2 * hidden_size, bias=True)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> Tensor:
        # x: [B, T, in]; returns h_all [B, T, H]
        B, T, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        out = self.proj(x)                                 # [B, T, 2H]
        z = torch.sigmoid(out[..., :self.hidden_size])     # [B, T, H]
        tilde = torch.sigmoid(out[..., self.hidden_size:]) # [B, T, H]
        a = 1.0 - z                                        # [B, T, H]
        b = z * tilde                                      # [B, T, H]
        # Sequential elementwise scan (cheap for small H).
        h_list = []
        h = h0
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            h_list.append(h)
        return torch.stack(h_list, dim=1)                  # [B, T, H]


# --- META-NETWORK v2 ---
class MetaNetwork(nn.Module):
    """SURGE-M v2: vectorized error compression, minGRU navigation, 4 output heads (u, v, g, a)."""
    def __init__(self, d_model, vocab_size, d_err, d_state, num_surge_layers,
                 gate_init_bias, forget_init_bias=2.3, uv_init_std=0.01):
        super().__init__()
        self.num_surge_layers = num_surge_layers
        self.d_state = d_state
        self.vocab_size = vocab_size
        self.d_err = d_err

        # Error projector: vocab -> d_err
        self.err_proj = nn.Linear(vocab_size, d_err, bias=False)
        self.err_ln = nn.LayerNorm(d_err)

        # minGRU on (h_lower [B,T,d_model], error [B,T,d_err]) -> s_all [B,T,d_state]
        self.mingru = MinGRU(input_size=d_model + d_err, hidden_size=d_state)

        # Four output heads per SURGE layer: u, v, g (update gate), a (forget gate).
        self.u_heads = nn.ModuleList([nn.Linear(d_state, d_model, bias=True) for _ in range(num_surge_layers)])
        self.v_heads = nn.ModuleList([nn.Linear(d_state, d_model, bias=True) for _ in range(num_surge_layers)])
        self.g_heads = nn.ModuleList([nn.Linear(d_state, 1, bias=True)       for _ in range(num_surge_layers)])
        self.a_heads = nn.ModuleList([nn.Linear(d_state, 1, bias=True)       for _ in range(num_surge_layers)])

        for head in list(self.u_heads) + list(self.v_heads):
            if uv_init_std > 0:
                nn.init.normal_(head.weight, mean=0.0, std=uv_init_std)
            else:
                nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for g in self.g_heads:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, gate_init_bias)         # sigmoid ~0.01
        for a in self.a_heads:
            nn.init.zeros_(a.weight)
            nn.init.constant_(a.bias, forget_init_bias)        # sigmoid ~0.9 (mostly retain)

    def compress_error_seq(self, logits_prev_seq: Tensor, token_ids_seq: Tensor) -> Tensor:
        """logits_prev_seq: [B, T, V] (detached); token_ids_seq: [B, T] -> [B, T, d_err]"""
        probs = F.softmax(logits_prev_seq.float(), dim=-1)
        one_hot = F.one_hot(token_ids_seq, num_classes=logits_prev_seq.shape[-1]).float()
        error = one_hot - probs                              # [B, T, V]
        return self.err_ln(self.err_proj(error))             # [B, T, d_err]

    def navigation_seq(self, h_lower: Tensor, e: Tensor) -> Tensor:
        """h_lower: [B, T, d_model]; e: [B, T, d_err] -> s_all [B, T, d_state]"""
        gru_in = torch.cat([h_lower.float(), e.float()], dim=-1)  # [B, T, d_model+d_err]
        return self.mingru(gru_in)                                # [B, T, d_state]

    def get_chunk_updates(self, s_all: Tensor, chunk_size: int, layer_idx: int) -> dict:
        """
        s_all: [B, T, d_state]. Sample state at end of each chunk.
        Returns dict with u [B, C, d_model], v [B, C, d_model], g [B, C, 1], a [B, C, 1].
        """
        T = s_all.shape[1]
        # End-of-chunk indices: chunk_size-1, 2*chunk_size-1, ...
        boundary = torch.arange(chunk_size - 1, T, chunk_size, device=s_all.device)
        s_c = s_all[:, boundary]                              # [B, C, d_state]
        return {
            'u': self.u_heads[layer_idx](s_c),                # [B, C, d_model]
            'v': self.v_heads[layer_idx](s_c),                # [B, C, d_model]
            'g': torch.sigmoid(self.g_heads[layer_idx](s_c)), # [B, C, 1]
            'a': torch.sigmoid(self.a_heads[layer_idx](s_c)), # [B, C, 1]
        }


# --- WY CORRECTION ---
def wy_correction(attn_pre_wo: Tensor, updates: dict, chunk_size: int, W_0: Tensor) -> Tensor:
    """
    Computes the SURGE correction to add to the residual stream contribution of layer's W_O output.

    For each chunk c:
      delta_c = z_c @ A_c^T  where  A_c = a_c * A_{c-1} + g_{c-1} * u_{c-1} ⊗ v_{c-1}
    A is represented in WY form as A_c = W_mat @ Y_mat^T.

    attn_pre_wo: [B, T, d]   — attention output before W_O is applied
    updates: dict with u [B, C, d], v [B, C, d], g [B, C, 1], a [B, C, 1]
    W_0: [d, d] — base W_O weight (frozen for this calculation, gradient still flows
        through u/v/g/a via the einsum chain)

    Returns: correction [B, T, d] — to be added to z_c (= attn_pre_wo @ W_0^T) at each chunk.
    """
    B, T, d = attn_pre_wo.shape
    C = T // chunk_size
    u = updates['u']   # [B, C, d]
    v = updates['v']   # [B, C, d]
    g = updates['g']   # [B, C, 1]
    a = updates['a']   # [B, C, 1]

    corrections: list[Tensor] = []
    W_cols: list[Tensor] = []   # each entry: [B, d]
    Y_cols: list[Tensor] = []   # each entry: [B, d]

    # Cast W_0 once
    W_0_dt = W_0.to(attn_pre_wo.dtype)

    for c in range(C):
        cs = c * chunk_size
        ce = cs + chunk_size
        # Standard z_c using base W_0 (frozen during this calc)
        z_c = F.linear(attn_pre_wo[:, cs:ce], W_0_dt)         # [B, chunk, d]

        if len(W_cols) == 0:
            corrections.append(torch.zeros_like(z_c))
        else:
            W_mat = torch.stack(W_cols, dim=-1)                # [B, d, c]
            Y_mat = torch.stack(Y_cols, dim=-1)                # [B, d, c]
            ZY = torch.einsum('btd,bdc->btc', z_c, Y_mat)      # [B, chunk, c]
            corr = torch.einsum('btc,bdc->btd', ZY, W_mat)     # [B, chunk, d]
            corrections.append(corr)

        # Update WY factors with chunk c's update
        u_c = u[:, c]   # [B, d]
        v_c = v[:, c]   # [B, d]
        g_c = g[:, c]   # [B, 1]
        a_c = a[:, c]   # [B, 1]

        # New W column: M_c @ u_c = u_c + sum_{k<c} (Y_k^T u_c) * W_k
        if len(W_cols) == 0:
            Mu_c = u_c
        else:
            W_mat = torch.stack(W_cols, dim=-1)                # [B, d, c]
            Y_mat = torch.stack(Y_cols, dim=-1)                # [B, d, c]
            Yu = torch.einsum('bdc,bd->bc', Y_mat, u_c)        # [B, c]
            Mu_c = u_c + torch.einsum('bdc,bc->bd', W_mat, Yu) # [B, d]

        gv_c = g_c * v_c                                       # [B, d] — new Y col
        # Apply forget gate to existing columns
        if len(W_cols) > 0:
            W_cols = [a_c * w for w in W_cols]
            Y_cols = [a_c * y for y in Y_cols]

        W_cols.append(Mu_c)
        Y_cols.append(gv_c)

    return torch.cat(corrections, dim=1)                       # [B, T, d]


# --- MULTIPLICATIVE UPDATE ---
def apply_multiplicative_update(W_prev: Tensor, u: Tensor, v: Tensor, gate: Tensor,
                                W_0: Tensor, max_drift_fraction: float = 0.1):
    """
    W_t = (I + gate * outer(u, v)) @ W_prev
        = W_prev + gate * outer(u, v^T @ W_prev)   [efficient form]

    u, v: [B, d]; gate: [B, 1]. We average over the batch (shared W for the batch).
    W_prev and W_0 are [d_out, d_in] (for W_O: d_out==d_in==d_model).
    Returns: W_new of shape same as W_prev. Detached from W_prev (FOMAML).
    """
    u_mean = u.mean(0).float()          # [d]
    v_mean = v.mean(0).float()          # [d]
    gate_mean = gate.mean(0).float()    # [1]

    W_prev_f = W_prev.float()
    # W_O shape is [d_out, d_in]; standard "y = F.linear(x, W)" means y = x @ W.T.
    # The multiplicative update composes on the left: W_new = (I + g u v^T) W_prev.
    # So in matrix form: W_new[i, j] = W_prev[i, j] + g * u[i] * (v^T @ W_prev)[j]
    v_W = v_mean @ W_prev_f              # [d_in]
    delta = gate_mean * torch.outer(u_mean, v_W)   # [d_out, d_in]
    W_new = W_prev_f + delta

    # Elastic anchor: clip drift to a fraction of ||W_0||
    W_0_f = W_0.float()
    drift = (W_new - W_0_f).norm(p="fro")
    max_drift = max_drift_fraction * W_0_f.norm(p="fro")
    if (drift > max_drift) and (max_drift > 0):
        excess = drift / max_drift
        W_new = W_0_f + (W_new - W_0_f) / excess

    return W_new.to(dtype=W_prev.dtype).detach()


# --- SURGE-M MODEL ---
class SurgeMModel(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init,
                 chunk_size, d_err, d_state, max_drift_fraction, gate_init_bias,
                 surge_layers, uv_init_std=0.0):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.max_drift_fraction = max_drift_fraction
        self.surge_layers = list(surge_layers)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

        self.meta_net = MetaNetwork(
            d_model=model_dim, vocab_size=vocab_size,
            d_err=d_err, d_state=d_state,
            num_surge_layers=len(self.surge_layers),
            gate_init_bias=gate_init_bias,
            uv_init_std=uv_init_std,
        )

        # Diagnostics captured from the last forward pass.
        self._diag = {}

        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_block(self, layer_idx: int, x: Tensor, x0: Tensor, W_O_override: Tensor | None) -> Tensor:
        return self.blocks[layer_idx](x, x0, W_O_override=W_O_override)

    def _block_forward_partial(self, layer_idx: int, x: Tensor, x0: Tensor, W_O_override: Tensor | None) -> Tensor:
        """Forward through a block returning post-layer activations. Used uniformly with/without override."""
        return self.blocks[layer_idx](x, x0, W_O_override=W_O_override)

    def forward(self, input_ids, target_ids):
        """
        SURGE-M v2 TWO-PASS forward:
          1. First pass: full base transformer forward (W_0 everywhere).
             Capture h_lower (before SURGE layer), attn_pre_wo at SURGE, x_after_surge (output of SURGE block).
          2. Compute base logits, derive errors (frozen base approximation, spec §3.7).
          3. minGRU parallel scan → chunk-boundary states → (u, v, g, a).
          4. WY correction [B, S, d] to residual stream at SURGE layer output.
          5. Second pass: run layers after SURGE with (x_after_surge + correction) + remaining skip connections.
          6. Final logits from second pass. Loss = CE(logits_corrected, target_ids).

        This is ~1.5x baseline compute (one full forward + re-running upper layers).
        Correction now propagates through MLP/attention/lm_head of all layers after SURGE.
        """
        B, S = input_ids.shape
        device = input_ids.device
        chunk_size = self.chunk_size
        assert len(self.surge_layers) == 1, "SURGE-M v2 supports a single SURGE layer"
        surge_layer = self.surge_layers[0]
        num_chunks = S // chunk_size

        # ---- PASS 1: full base transformer forward with W_0 ----
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        h_lower_collected: Tensor | None = None
        attn_pre_wo: Tensor | None = None
        x_after_surge: Tensor | None = None

        # Encoder half
        for i in range(self.num_encoder_layers):
            if i == surge_layer:
                h_lower_collected = x.detach()
                x, attn_pre_wo = self.blocks[i](x, x0, W_O_override=None, return_pre_wo=True)
                x_after_surge = x
            else:
                x = self.blocks[i](x, x0)
            skips.append(x)

        # Save the skip-connection state: we'll need to replay the decoder half from surge_layer+1.
        # Check if SURGE is in the encoder or decoder half.
        surge_in_encoder = surge_layer < self.num_encoder_layers

        # Decoder half (first pass)
        for j in range(self.num_decoder_layers):
            li = self.num_encoder_layers + j
            if skips:
                x = x + self.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
            if li == surge_layer:
                h_lower_collected = x.detach()
                x, attn_pre_wo = self.blocks[li](x, x0, W_O_override=None, return_pre_wo=True)
                x_after_surge = x
            else:
                x = self.blocks[li](x, x0)

        # Base logits from first pass (frozen base approximation)
        x_norm = self.final_norm(x)
        if self.tie_embeddings:
            logits_base = F.linear(x_norm, self.tok_emb.weight)
        else:
            logits_base = self.lm_head(x_norm)
        logits_base = self.logit_softcap * torch.tanh(logits_base / self.logit_softcap)

        # ---- Error computation (detached from grad) ----
        leading = torch.zeros(B, 1, self.vocab_size, device=device, dtype=logits_base.dtype)
        shifted = torch.cat([leading, logits_base[:, :-1]], dim=1).detach()
        e = self.meta_net.compress_error_seq(shifted, input_ids)

        # ---- minGRU scan + chunk updates + WY correction ----
        s_all = self.meta_net.navigation_seq(h_lower_collected, e)
        updates = self.meta_net.get_chunk_updates(s_all, chunk_size, 0)
        W_0 = self.blocks[surge_layer].attn.proj.weight.detach()
        T_used = num_chunks * chunk_size
        attn_used = attn_pre_wo[:, :T_used] if attn_pre_wo.size(1) > T_used else attn_pre_wo
        correction = wy_correction(
            attn_pre_wo=attn_used, updates=updates,
            chunk_size=chunk_size, W_0=W_0,
        )
        if correction.size(1) < S:
            pad = torch.zeros(B, S - correction.size(1), correction.size(2),
                              device=device, dtype=correction.dtype)
            correction = torch.cat([correction, pad], dim=1)

        # ---- PASS 2: rerun upper layers with corrected residual stream ----
        # Add correction to x_after_surge. x_after_surge came from first pass; we need to detach base-model
        # contribution to prevent gradient double-flow, but keep correction gradient live.
        # Simpler approach: keep x_after_surge attached (gradient flows both through first-pass base and
        # through correction → M). This increases memory but is cleaner.
        x = x_after_surge + self.blocks[surge_layer].attn_scale.to(dtype=x_after_surge.dtype)[None, None, :] * correction.to(x_after_surge.dtype)

        # Determine which layers to re-run: surge_layer+1 ... num_layers-1
        # Replay skips from this point: we consumed all pre-surge skips into x via the first pass's decoder loop,
        # but in first pass we used SURGE-layer's actual output (with W_0). The correction changes what the
        # downstream skip additions would have seen. Rather than properly replaying, approximate: re-run from
        # the layer AFTER surge_layer with the corrected x as input.
        #
        # IMPORTANT: first pass already consumed the encoder skips in order. We cannot cleanly replay the
        # decoder skip additions for the re-run because they were computed on uncorrected x. So we approximate
        # by running subsequent layers WITHOUT re-adding skips (only attn + MLP forward).
        for k in range(surge_layer + 1, self.num_layers):
            x = self.blocks[k](x, x0)

        # Final logits from second pass
        x_norm = self.final_norm(x)
        if self.tie_embeddings:
            logits_total = F.linear(x_norm, self.tok_emb.weight)
        else:
            logits_total = self.lm_head(x_norm)
        logits_total = self.logit_softcap * torch.tanh(logits_total / self.logit_softcap)

        # Loss
        loss = F.cross_entropy(logits_total.reshape(-1, logits_total.size(-1)).float(),
                               target_ids.reshape(-1))
        num_tokens = torch.tensor(target_ids.numel(), device=device, dtype=torch.float32)

        # Diagnostics
        with torch.no_grad():
            self._diag = {
                "u_norm_mean": [updates['u'].norm(dim=-1).mean().item()],
                "v_norm_mean": [updates['v'].norm(dim=-1).mean().item()],
                "gate_mean": [updates['g'].mean().item()],
                "forget_mean": [updates['a'].mean().item()],
                "error_magnitude": e.detach().float().norm(dim=-1).mean().item(),
                "correction_rms": correction.detach().float().pow(2).mean().sqrt().item(),
                "surge_layers": list(self.surge_layers),
            }

        return loss, num_tokens.detach()


# --- MAIN ---
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
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)

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

    surge_layers = [int(x) for x in args.surge_layers_str.split(",") if x.strip() != ""]
    log0(f"=== SURGE-M EXPERIMENT ===")
    log0(f"model_dim:{args.model_dim} num_layers:{args.num_layers} num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(f"surge_layers:{surge_layers} chunk_size:{args.chunk_size} d_err:{args.d_err} d_state:{args.d_state}")
    log0(f"max_drift_fraction:{args.max_drift_fraction} gate_init_bias:{args.gate_init_bias} lr_meta:{args.lr_meta}")
    log0(code, console=False)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = SurgeMModel(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        chunk_size=args.chunk_size, d_err=args.d_err, d_state=args.d_state,
        max_drift_fraction=args.max_drift_fraction, gate_init_bias=args.gate_init_bias,
        surge_layers=surge_layers, uv_init_std=args.uv_init_std,
    ).to(device).bfloat16()

    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    # Meta-network (GRU + projections) runs in float32 for numerical stability.
    base_model.meta_net.float()
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

    n_params = sum(p.numel() for p in base_model.parameters())
    meta_params_count = sum(p.numel() for p in base_model.meta_net.parameters())
    log0(f"total_params:{n_params} meta_params:{meta_params_count}")

    # Note: cannot use torch.compile -- dynamic per-chunk graph.
    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    # --- Optimizer setup ---
    # Meta-net params: AdamW (excluded from Muon)
    meta_param_ids = set(id(p) for p in base_model.meta_net.parameters())

    # Base block params (excluding meta_net)
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for name, p in block_named_params
                     if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for name, p in block_named_params
                     if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)

    # Filter to exclude any accidental meta params (shouldn't happen since meta_net is its own module).
    matrix_params = [p for p in matrix_params if id(p) not in meta_param_ids]
    scalar_params = [p for p in scalar_params if id(p) not in meta_param_ids]

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups: group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_meta = torch.optim.AdamW(
        [{"params": list(base_model.meta_net.parameters()), "lr": args.lr_meta, "base_lr": args.lr_meta}],
        betas=(args.beta1, 0.999), eps=args.adam_eps, weight_decay=args.meta_weight_decay,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_meta]

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    def zero_grad_all():
        for opt in optimizers: opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0: return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup (re-initialize afterward)
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl, _ = model(x, y)
                (wl * grad_scale).backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=args.grad_clip)
            for opt in optimizers: opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True): opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training loop
    training_time_ms = 0.0
    stop_after_step = None
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
                 f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize(); t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss, _ = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Gradient clipping on all params
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=args.grad_clip)

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * scale
        for opt in optimizers: opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            diag = base_model._diag if hasattr(base_model, "_diag") else {}
            diag_str = ""
            if diag:
                gl = diag.get("gate_mean", [])
                un = diag.get("u_norm_mean", [])
                vn = diag.get("v_norm_mean", [])
                fg = diag.get("forget_mean", [])
                em = diag.get("error_magnitude", 0.0)
                cr = diag.get("correction_rms", 0.0)
                lyrs = diag.get("surge_layers", [])
                parts = []
                for idx, li in enumerate(lyrs):
                    g = gl[idx] if idx < len(gl) else 0.0
                    u = un[idx] if idx < len(un) else 0.0
                    v = vn[idx] if idx < len(vn) else 0.0
                    f = fg[idx] if idx < len(fg) else 0.0
                    parts.append(f"L{li}_gate:{g:.4f} L{li}_u:{u:.4f} L{li}_v:{v:.4f} L{li}_forget:{f:.4f}")
                diag_str = " " + " ".join(parts) + f" err_mag:{em:.4f} corr_rms:{cr:.6f}"
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                 f"{diag_str}")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # Quantize and roundtrip
    if master_process:
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    if master_process:
        with open("final_model_surge_m.int8.ptz", "wb") as f: f.write(quant_blob)
        log0(f"Total submission size int8+zlib: "
             f"{os.path.getsize('final_model_surge_m.int8.ptz') + len(code.encode('utf-8'))} bytes")

    with open("final_model_surge_m.int8.ptz", "rb") as f: quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    if distributed: dist.destroy_process_group()


if __name__ == "__main__":
    main()
