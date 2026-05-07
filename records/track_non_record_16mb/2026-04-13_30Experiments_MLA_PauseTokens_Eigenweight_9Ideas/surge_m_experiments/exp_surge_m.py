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

    def forward(self, x, W_O_override: Tensor | None = None):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        if W_O_override is not None:
            return F.linear(y, W_O_override.to(y.dtype))
        return self.proj(y)


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

    def forward(self, x, x0, W_O_override: Tensor | None = None):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(self.attn_norm(x), W_O_override=W_O_override)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# --- META-NETWORK ---
class MetaNetwork(nn.Module):
    """Produces per-chunk multiplicative update factors (u, v, gate) for each SURGE layer."""
    def __init__(self, d_model, vocab_size, d_err, d_state, num_surge_layers, gate_init_bias, uv_init_std=0.0):
        super().__init__()
        self.num_surge_layers = num_surge_layers
        self.d_state = d_state
        self.vocab_size = vocab_size

        # Error projector: vocab -> d_err
        self.err_proj = nn.Linear(vocab_size, d_err, bias=False)
        self.err_ln = nn.LayerNorm(d_err)

        # GRU taking [h_lower (d_model), error (d_err)] -> d_state.
        # Use vectorized nn.GRU (processes whole chunk in parallel) for speed.
        self.gru = nn.GRU(input_size=d_model + d_err, hidden_size=d_state, batch_first=True)

        # Per-layer u/v/gate heads.
        # uv_init_std=0 means zero-init (default, but causes chicken-egg).
        # uv_init_std>0 (e.g. 0.01) breaks the symmetry so M can start learning immediately.
        self.u_heads = nn.ModuleList([nn.Linear(d_state, d_model, bias=False) for _ in range(num_surge_layers)])
        self.v_heads = nn.ModuleList([nn.Linear(d_state, d_model, bias=False) for _ in range(num_surge_layers)])
        self.gates   = nn.ModuleList([nn.Linear(d_state, 1, bias=True)       for _ in range(num_surge_layers)])

        for head in list(self.u_heads) + list(self.v_heads):
            if uv_init_std > 0:
                nn.init.normal_(head.weight, mean=0.0, std=uv_init_std)
            else:
                nn.init.zeros_(head.weight)
        for g in self.gates:
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, gate_init_bias)

    def compress_error(self, logits_prev, token_ids):
        """logits_prev: [B, vocab], token_ids: [B] -> [B, d_err]"""
        probs = F.softmax(logits_prev.float(), dim=-1)
        one_hot = F.one_hot(token_ids, num_classes=logits_prev.shape[-1]).float()
        error = one_hot - probs
        return self.err_ln(self.err_proj(error))

    def update_state(self, h_lower_t, e_t, s_prev):
        """Single-step GRU (kept for backwards compatibility, used in eval).
        h_lower_t: [B, d_model], e_t: [B, d_err], s_prev: [B, d_state]
        """
        gru_in = torch.cat([h_lower_t.float(), e_t.float()], dim=-1).unsqueeze(1)  # [B, 1, in]
        s_prev_3d = s_prev.unsqueeze(0).contiguous()  # [1, B, d_state]
        _, s_new = self.gru(gru_in, s_prev_3d)
        return s_new.squeeze(0)  # [B, d_state]

    def update_state_chunk(self, h_lower_chunk, e_chunk, s_prev):
        """Vectorized GRU over an entire chunk in parallel.
        h_lower_chunk: [B, T, d_model], e_chunk: [B, T, d_err], s_prev: [B, d_state]
        Returns: s_new [B, d_state] (final state after the chunk)
        """
        gru_in = torch.cat([h_lower_chunk.float(), e_chunk.float()], dim=-1)  # [B, T, in]
        s_prev_3d = s_prev.unsqueeze(0).contiguous()  # [1, B, d_state]
        _, s_new = self.gru(gru_in, s_prev_3d)
        return s_new.squeeze(0)  # [B, d_state]

    def get_update(self, s, layer_idx):
        """s: [B, d_state] -> (u, v, gate) with u,v: [B, d_model], gate: [B, 1]"""
        u = self.u_heads[layer_idx](s)
        v = self.v_heads[layer_idx](s)
        gate = torch.sigmoid(self.gates[layer_idx](s))
        return u, v, gate


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
        input_ids: (B, S), target_ids: (B, S)
        Returns: (loss, num_tokens)
        """
        B, S = input_ids.shape
        device = input_ids.device
        chunk_size = self.chunk_size

        # Base weights for SURGE layers (detached references for anchoring).
        W_base = [self.blocks[li].attn.proj.weight for li in self.surge_layers]
        W_0 = [W.detach().clone() for W in W_base]
        W_cur = [W.detach().clone() for W in W_base]  # current evolving weights (detached from base graph initially)

        # GRU navigation state
        s = torch.zeros(B, self.meta_net.d_state, device=device, dtype=torch.float32)

        # Bookkeeping for errors
        num_chunks = (S + chunk_size - 1) // chunk_size
        prev_last_logit = None  # [B, vocab]

        all_logits = []

        # Diagnostics accumulators (cheap, cpu at end)
        drift_layer_sum = [0.0 for _ in self.surge_layers]
        drift_layer_count = 0
        gate_sum = [0.0 for _ in self.surge_layers]
        u_norm_sum = [0.0 for _ in self.surge_layers]
        v_norm_sum = [0.0 for _ in self.surge_layers]
        gate_count = 0
        error_mag_sum = 0.0
        error_mag_count = 0

        # Per-chunk forward
        for c in range(num_chunks):
            cs = c * chunk_size
            ce = min(cs + chunk_size, S)
            ids_chunk = input_ids[:, cs:ce]

            # STEP 1: Apply multiplicative update based on current GRU state s.
            # For chunk 0, s == 0 -> u,v zero -> delta zero. Still functionally correct.
            W_override_list = []
            for li_idx, _ in enumerate(self.surge_layers):
                u, v, gate = self.meta_net.get_update(s, li_idx)
                # Efficient rank-1 update computation with gradient flowing into u/v/gate.
                u_mean = u.mean(0)                       # [d]
                v_mean = v.mean(0)                       # [d]
                gate_mean = gate.mean(0)                 # [1]
                W_prev = W_cur[li_idx]
                v_W = v_mean @ W_prev.float()            # [d]
                delta = gate_mean * torch.outer(u_mean, v_W)   # [d, d], gradient flows through u/v/gate
                W_new = W_prev.float() + delta
                # Elastic anchor
                drift = (W_new - W_0[li_idx].float()).norm(p="fro")
                max_drift = self.max_drift_fraction * W_0[li_idx].float().norm(p="fro")
                if (drift > max_drift) and (max_drift.item() > 0):
                    excess = drift / max_drift
                    W_new = W_0[li_idx].float() + (W_new - W_0[li_idx].float()) / excess
                # Use W_new (with grad) for the current chunk forward,
                # then store detached version for next chunk's W_prev (FOMAML).
                W_override_list.append(W_new)
                W_cur[li_idx] = W_new.detach()

                # Diagnostics
                gate_sum[li_idx] += float(gate_mean.detach().float().item())
                u_norm_sum[li_idx] += float(u_mean.detach().float().norm().item())
                v_norm_sum[li_idx] += float(v_mean.detach().float().norm().item())
                drift_layer_sum[li_idx] += float((W_new.detach() - W_0[li_idx].float()).norm().item()
                                                 / (W_0[li_idx].float().norm().item() + 1e-12))
            gate_count += 1
            drift_layer_count += 1

            # STEP 2: Forward pass through all layers for the chunk (parallel over sequence).
            x_chunk = self.tok_emb(ids_chunk)
            x_chunk = F.rms_norm(x_chunk, (x_chunk.size(-1),))
            x0 = x_chunk
            # We support the baseline U-Net skip-connection layout (encoder/decoder halves).
            skips = []
            h_lower = None
            surge_set = set(self.surge_layers)
            # Build a map layer_idx -> override
            override_by_layer = {li: W_override_list[idx] for idx, li in enumerate(self.surge_layers)}

            x = x_chunk
            # Encoder half
            for i in range(self.num_encoder_layers):
                over = override_by_layer.get(i)
                x = self._run_block(i, x, x0, over)
                # Capture the lower-stack output just before the first SURGE layer
                if h_lower is None and (i + 1) in surge_set:
                    h_lower = x
                skips.append(x)
            # Decoder half
            for j in range(self.num_decoder_layers):
                li = self.num_encoder_layers + j
                if skips:
                    x = x + self.skip_weights[j].to(dtype=x.dtype)[None, None, :] * skips.pop()
                over = override_by_layer.get(li)
                x = self._run_block(li, x, x0, over)
                if h_lower is None and (li + 1) in surge_set:
                    h_lower = x
            if h_lower is None:
                # Fallback: use the pre-SURGE input representation as GRU input signal.
                h_lower = x_chunk

            x = self.final_norm(x)
            if self.tie_embeddings:
                chunk_logits = F.linear(x, self.tok_emb.weight)
            else:
                chunk_logits = self.lm_head(x)
            chunk_logits = self.logit_softcap * torch.tanh(chunk_logits / self.logit_softcap)  # [B, L, V]
            all_logits.append(chunk_logits)

            # STEP 3: Compute prediction errors for each token (shifted).
            # error_t = one_hot(token_t) - softmax(logits_{t-1}).
            # For the first token of chunk 0, use zeros; for other chunk-starts, use prev_last_logit.
            L = chunk_logits.size(1)
            if L == 0:
                continue
            if c == 0:
                leading = torch.zeros(B, 1, self.vocab_size, device=device, dtype=chunk_logits.dtype)
            else:
                leading = prev_last_logit.unsqueeze(1)
            shifted = torch.cat([leading, chunk_logits[:, :-1]], dim=1)  # [B, L, V]

            # STEP 4: Vectorized GRU update over the entire chunk in parallel.
            # Compress all errors at once: shifted [B,L,V] + ids [B,L] -> e_chunk [B,L,d_err]
            shifted_flat = shifted.reshape(-1, shifted.size(-1))                # [B*L, V]
            ids_flat = ids_chunk.reshape(-1)                                     # [B*L]
            e_chunk_flat = self.meta_net.compress_error(shifted_flat, ids_flat)  # [B*L, d_err]
            e_chunk = e_chunk_flat.reshape(B, L, -1)                              # [B, L, d_err]
            s = self.meta_net.update_state_chunk(h_lower, e_chunk, s)
            error_mag_sum += float(e_chunk.detach().float().norm(dim=-1).mean().item())
            error_mag_count += 1

            prev_last_logit = chunk_logits[:, -1].detach()

        logits = torch.cat(all_logits, dim=1)  # [B, S, V]

        # Loss on all positions using target_ids.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1))
        num_tokens = torch.tensor(target_ids.numel(), device=device, dtype=torch.float32)

        # Store diagnostics for external logging
        if gate_count > 0:
            self._diag = {
                "gate_mean": [g / gate_count for g in gate_sum],
                "u_norm_mean": [u / gate_count for u in u_norm_sum],
                "v_norm_mean": [v / gate_count for v in v_norm_sum],
                "drift_fraction": [d / max(drift_layer_count, 1) for d in drift_layer_sum],
                "error_magnitude": (error_mag_sum / max(error_mag_count, 1)),
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
                dr = diag.get("drift_fraction", [])
                em = diag.get("error_magnitude", 0.0)
                lyrs = diag.get("surge_layers", [])
                parts = []
                for idx, li in enumerate(lyrs):
                    parts.append(
                        f"L{li}_gate:{gl[idx]:.4f} L{li}_u:{un[idx]:.4f} L{li}_v:{vn[idx]:.4f} L{li}_drift:{dr[idx]:.4f}"
                    )
                diag_str = " " + " ".join(parts) + f" err_mag:{em:.4f}"
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
