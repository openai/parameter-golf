"""
SURGE-M Ablation D: Memoryless meta-network (no GRU).

Identical to SURGE-M (section 12 of SURGE_M_architecture.md) EXCEPT the
MetaNetwork M is a 2-layer MLP with no hidden state:
    (u, v, gate) = MLP( cat( mean_h_chunk_{c-1}, mean_e_chunk_{c-1} ) )

For chunk c the update is computed from the chunk-mean of (h_lower, error)
of the *previous* chunk c-1. For c=0 the input is zeros (which combined with
zero-init heads => sigmoid gate ~= 0.01, i.e. essentially no update).

Everything else is identical to the main SURGE-M experiment:
  - 9 layers, d_model=256, n_heads=4, n_kv_heads=4, d_ff=512, SP vocab=1024
  - SURGE layers [3, 4] with multiplicative W_O update W = (I + g*u v^T) W
  - chunk_size=64, MAX_DRIFT_FRACTION=0.1, gate bias init -4.6
  - FOMAML detach on W_t, elastic anchor on drift
  - Muon for 2D base params, Adam for embedding/scalars, AdamW for meta_net (lr=3e-4)
  - FineWeb SP1024 data, BPB eval, int8+zlib serialization, torchrun, 600s cap
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
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
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 400))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # === Architecture (SURGE-M spec) ===
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 256))
    num_heads = int(os.environ.get("NUM_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))

    # === SURGE-M specifics ===
    surge_layer_indices = tuple(int(x) for x in os.environ.get("SURGE_LAYERS", "3,4").split(","))
    chunk_size = int(os.environ.get("CHUNK_SIZE", 64))
    d_err = int(os.environ.get("D_ERR", 64))
    d_hidden = int(os.environ.get("META_D_HIDDEN", 128))
    max_drift_fraction = float(os.environ.get("MAX_DRIFT_FRACTION", 0.1))
    gate_init_bias = float(os.environ.get("GATE_INIT_BIAS", -4.6))

    # === Optimizers ===
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 1e-3))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    meta_lr = float(os.environ.get("META_LR", 3e-4))
    meta_wd = float(os.environ.get("META_WD", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


# --- Muon optimizer ---
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
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
            total = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total, device=params[0].device, dtype=torch.bfloat16)
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


# --- Tokenizer-aware BPB LUTs ---
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


def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files for {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

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


# --- Quantization (int8 per-row + zlib) ---
CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight", "skip_weights")


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
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
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


# --- Transformer building blocks ---
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
    """
    Standard attention. Accepts an optional explicit W_O override so we can
    use the evolving W_O for SURGE layers instead of the stored nn.Parameter.
    """
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, kv_dim, bias=False)
        self.c_v = nn.Linear(dim, kv_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)  # W_O — evolves for SURGE layers
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

        # Standard init
        nn.init.normal_(self.c_q.weight, std=0.02)
        nn.init.normal_(self.c_k.weight, std=0.02)
        nn.init.normal_(self.c_v.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, W_O_override: Tensor = None):
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
        self.fc = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x, W_O_override: Tensor = None):
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(F.rms_norm(x, (x.size(-1),)), W_O_override=W_O_override)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


# --- Memoryless Meta-Network (Ablation D) ---
class MetaNetworkMemoryless(nn.Module):
    """
    Memoryless: 2-layer MLP (no GRU, no recurrent state).
    Input:  concat( mean_h_lower_prev_chunk, mean_error_prev_chunk )  ->  [B, d_model + d_err]
    Output: (u3, v3, g3, u4, v4, g4)  — per-SURGE-layer rank-1 update factors
    """
    def __init__(self, d_model=256, vocab_size=1024, d_err=64, d_hidden=128, gate_init_bias=-4.6):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.d_err = d_err
        self.d_hidden = d_hidden

        # Compress vocab-space prediction error with LayerNorm
        self.err_proj = nn.Sequential(
            nn.Linear(vocab_size, d_err, bias=False),
            nn.LayerNorm(d_err),
        )

        # 2-layer MLP (memoryless)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_err, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

        # Per-layer u/v/gate heads (zero init; gate bias -> sigmoid ~= 0.01)
        self.u_head_3 = nn.Linear(d_hidden, d_model, bias=False)
        self.v_head_3 = nn.Linear(d_hidden, d_model, bias=False)
        self.gate_3 = nn.Linear(d_hidden, 1, bias=True)
        self.u_head_4 = nn.Linear(d_hidden, d_model, bias=False)
        self.v_head_4 = nn.Linear(d_hidden, d_model, bias=False)
        self.gate_4 = nn.Linear(d_hidden, 1, bias=True)
        for head in (self.u_head_3, self.v_head_3, self.u_head_4, self.v_head_4):
            nn.init.zeros_(head.weight)
        nn.init.constant_(self.gate_3.bias, gate_init_bias)
        nn.init.constant_(self.gate_4.bias, gate_init_bias)

    def compress_error(self, logits_prev: Tensor, token_ids: Tensor) -> Tensor:
        """
        logits_prev: [B, vocab]   - logits BEFORE seeing token_ids
        token_ids:   [B]
        Returns:     [B, d_err]
        """
        probs = F.softmax(logits_prev.float(), dim=-1)
        one_hot = F.one_hot(token_ids, num_classes=logits_prev.shape[-1]).float()
        error = one_hot - probs
        return self.err_proj(error)

    def get_update(self, mean_input: Tensor):
        """
        mean_input: [B, d_model + d_err]
        Returns: u3, v3, g3, u4, v4, g4 (all [B, d_model] or [B, 1])
        """
        h = self.mlp(mean_input)
        u3 = self.u_head_3(h)
        v3 = self.v_head_3(h)
        g3 = torch.sigmoid(self.gate_3(h))
        u4 = self.u_head_4(h)
        v4 = self.v_head_4(h)
        g4 = torch.sigmoid(self.gate_4(h))
        return u3, v3, g3, u4, v4, g4


# --- Multiplicative update with elastic anchor + FOMAML detach ---
def apply_multiplicative_update(W_prev: Tensor, u: Tensor, v: Tensor, gate: Tensor,
                                W_0: Tensor, max_drift_fraction: float = 0.1) -> Tensor:
    """
    W_t = (I + gate * u v^T) @ W_{t-1}
    Efficient: delta = gate * outer(u, v^T @ W_{t-1})
    u, v:    [B, d] - averaged across batch
    gate:    [B, 1]
    W_prev:  [d_out, d_in]   (for W_O: d_out == d_in == d_model)
    """
    u_mean = u.mean(0)           # [d]
    v_mean = v.mean(0)           # [d]
    gate_mean = gate.mean(0).squeeze().to(W_prev.dtype)  # scalar tensor

    # v^T @ W_prev  (W_prev has shape [d_out, d_in]; v should match d_out for
    # W_new = (I + g u v^T) W_prev with u,v \in R^{d_out}.)
    v_W = v_mean.to(W_prev.dtype) @ W_prev            # [d_in]
    delta = gate_mean * torch.outer(u_mean.to(W_prev.dtype), v_W)  # [d_out, d_in]
    W_new = W_prev + delta

    # Elastic anchor — constrain Frobenius drift from base
    W0_norm = W_0.norm()
    drift = (W_new - W_0).norm()
    max_drift = max_drift_fraction * W0_norm
    if drift > max_drift and drift > 0:
        W_new = W_0 + (W_new - W_0) * (max_drift / drift)

    # FOMAML approximation: detach so grad doesn't flow through the W-chain
    return W_new.detach()


# --- SURGE-M Model (Ablation D: memoryless meta-network) ---
class SurgeMModel(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.model_dim = args.model_dim
        self.chunk_size = args.chunk_size
        self.surge_layer_indices = tuple(args.surge_layer_indices)
        self.max_drift_fraction = args.max_drift_fraction

        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)

        self.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads,
                  args.mlp_mult, args.rope_base, args.qk_gain_init)
            for _ in range(args.num_layers)
        ])

        self.logit_softcap = args.logit_softcap

        # Memoryless meta-network (NO GRU)
        self.meta_net = MetaNetworkMemoryless(
            d_model=args.model_dim,
            vocab_size=args.vocab_size,
            d_err=args.d_err,
            d_hidden=args.d_hidden,
            gate_init_bias=args.gate_init_bias,
        )

    def _lower_forward(self, h):
        for i in range(self.surge_layer_indices[0]):
            h = self.blocks[i](h)
        return h

    def _upper_forward(self, h):
        upper_start = self.surge_layer_indices[-1] + 1
        for i in range(upper_start, len(self.blocks)):
            h = self.blocks[i](h)
        return h

    def _final_logits(self, h):
        h = F.rms_norm(h, (h.size(-1),))
        logits_proj = F.linear(h, self.tok_emb.weight)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids: Tensor, target_ids: Tensor):
        """
        Chunked forward with per-chunk W_O evolution on SURGE layers.

        Update rule (Ablation D — memoryless):
          For chunk c:
            update_input_c = mean over chunk c-1 of cat(h_lower, error)
            (u, v, g)_c    = MLP(update_input_c)
            W_{c}          = (I + g*u v^T) W_{c-1}
          For c=0: update_input is zeros  =>  zero-init heads yield no-op update.
        """
        B, T = input_ids.shape
        device = input_ids.device
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size

        # Snapshot base W_O for SURGE layers (retain grads into W_0)
        W_3_idx, W_4_idx = self.surge_layer_indices[0], self.surge_layer_indices[1]
        W_0_3 = self.blocks[W_3_idx].attn.proj.weight
        W_0_4 = self.blocks[W_4_idx].attn.proj.weight

        # Evolving weights (start at base); use override-path in forward
        W_3 = W_0_3  # first chunk uses the base (zero update)
        W_4 = W_0_4

        # State between chunks: last chunk's logits (for shifted error) and its input_ids
        prev_last_logit = None  # [B, 1, vocab]

        # Memoryless meta input from prev chunk: mean(h_lower, error) -> [B, d+d_err]
        # Initialize to zeros for c=0 so the zero-init heads yield gate ~= 0.01 and u=v=0.
        d_input = self.model_dim + self.args.d_err
        next_meta_input = torch.zeros(B, d_input, device=device)

        # Embed once
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))

        all_logits = []

        for c in range(num_chunks):
            cs = c * chunk_size
            ce = min(cs + chunk_size, T)
            chunk_len = ce - cs
            x_chunk = x[:, cs:ce]
            ids_chunk = input_ids[:, cs:ce]

            # === Step 1: compute update from previous chunk's mean signal ===
            u3, v3, g3, u4, v4, g4 = self.meta_net.get_update(next_meta_input)
            W_3 = apply_multiplicative_update(W_3, u3, v3, g3, W_0_3, self.max_drift_fraction)
            W_4 = apply_multiplicative_update(W_4, u4, v4, g4, W_0_4, self.max_drift_fraction)

            # === Step 2: forward pass for this chunk ===
            # Note: `x` is already embedded+normed at the sequence level; for
            # chunk-local attention we treat each chunk as an independent
            # causal segment. This matches the spec's "forward pass through
            # model in parallel for this chunk" description.
            h = x_chunk
            # Lower layers (fixed)
            h_lower = self._lower_forward(h)

            # SURGE layer 3 (explicit W_O)
            h = self.blocks[W_3_idx](h_lower, W_O_override=W_3)
            # Layers strictly between the two SURGE layers (if any)
            for i in range(W_3_idx + 1, W_4_idx):
                h = self.blocks[i](h)
            # SURGE layer 4
            h = self.blocks[W_4_idx](h, W_O_override=W_4)

            # Upper layers
            h = self._upper_forward(h)

            chunk_logits = self._final_logits(h)  # [B, chunk_len, V]
            all_logits.append(chunk_logits)

            # === Step 3: compute per-token prediction errors for this chunk ===
            # logit_{t-1} predicts token_t; for t=0 in chunk use previous chunk's last logit.
            if c == 0:
                first_logit = torch.zeros(B, 1, self.vocab_size, device=device, dtype=chunk_logits.dtype)
            else:
                first_logit = prev_last_logit
            shifted_logits = torch.cat([first_logit, chunk_logits[:, :-1]], dim=1)  # [B, chunk_len, V]

            # Compute compressed errors for each token position in chunk — vectorized
            probs = F.softmax(shifted_logits.float(), dim=-1)
            one_hot = F.one_hot(ids_chunk, num_classes=self.vocab_size).float()
            errors = one_hot - probs  # [B, chunk_len, V]
            # Project: flatten batch*time for Linear
            e_flat = self.meta_net.err_proj(errors.reshape(-1, self.vocab_size))
            e_chunk = e_flat.reshape(B, chunk_len, self.args.d_err)

            # === Step 4 (memoryless): form next chunk's MLP input = mean of (h_lower, e) ===
            mean_h = h_lower.float().mean(dim=1)   # [B, d_model]
            mean_e = e_chunk.float().mean(dim=1)   # [B, d_err]
            next_meta_input = torch.cat([mean_h, mean_e], dim=-1)  # [B, d_model + d_err]

            prev_last_logit = chunk_logits[:, -1:].detach()

        logits = torch.cat(all_logits, dim=1)  # [B, T, V]
        targets = target_ids.reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size).float(), targets, reduction="mean")
        return loss


# --- Validation (BPB) ---
def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens,
             base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_seqs = max(local_batch_tokens // args.train_seq_len, 1)
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
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# --- MAIN ---
def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if 8 % max(world_size, 1) != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // max(world_size, 1)
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
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
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

    log0(f"=== SURGE-M Ablation D: MEMORYLESS (no GRU) === d_model={args.model_dim} "
         f"surge_layers={args.surge_layer_indices} chunk_size={args.chunk_size} "
         f"d_err={args.d_err} d_hidden={args.d_hidden} max_drift={args.max_drift_fraction}")
    log0(code, console=False)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")

    base_model = SurgeMModel(args).to(device)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_meta = sum(p.numel() for p in base_model.meta_net.parameters())
    log0(f"model_params:{n_params} meta_params:{n_meta} base_params:{n_params - n_meta}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")

    model = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    # === Parameter groups ===
    # - Muon: 2D base weights (not meta_net, not tok_emb)
    # - Adam: tok_emb
    # - Adam: base scalars (<2D) / control tensors (attn_scale, mlp_scale, q_gain)
    # - AdamW: meta_net params
    meta_params = list(base_model.meta_net.parameters())
    meta_param_ids = {id(p) for p in meta_params}

    muon_params, scalar_params = [], []
    for name, p in base_model.named_parameters():
        if id(p) in meta_param_ids:
            continue
        if name == "tok_emb.weight":
            continue
        if "rotary" in name or "inv_freq" in name:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            scalar_params.append(p)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_muon = Muon(muon_params, lr=args.matrix_lr,
                          momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True,
    )
    optimizer_meta = torch.optim.AdamW(
        [{"params": meta_params, "lr": args.meta_lr, "base_lr": args.meta_lr}],
        betas=(0.9, 0.999), eps=args.adam_eps, weight_decay=args.meta_wd,
    )
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_meta]

    log0(f"matrix_lr:{args.matrix_lr} token_lr:{token_lr} scalar_lr:{args.scalar_lr} meta_lr:{args.meta_lr}")
    log0(f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
         f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
         f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}")
    log0(f"seed:{args.seed}")

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
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # --- Warmup (prime kernels, restore state) ---
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    wl = model(x, y)
                (wl * grad_scale).backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # --- Main training loop ---
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
            torch.cuda.synchronize()
            t0 = time.perf_counter()
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
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        for group in optimizer_muon.param_groups:
            group["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0)
        if should_log_train:
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                 f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms")

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(f"peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # --- Serialization + int8+zlib + roundtrip validation ---
    if master_process:
        code_bytes = len(code.encode("utf-8"))
        log0(f"Code size: {code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
    fname = f"final_model_{args.run_id}.int8.ptz"
    if master_process:
        with open(fname, "wb") as f:
            f.write(quant_blob)
        log0(f"Total submission size int8+zlib: {os.path.getsize(fname) + len(code.encode('utf-8'))} bytes")
        log0(f"quant_stats: {quant_stats}")

    if distributed:
        dist.barrier()

    # Roundtrip eval
    if master_process:
        with open(fname, "rb") as f:
            quant_blob_disk = f.read()
        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    if distributed:
        # Broadcast roundtripped weights
        for p in base_model.parameters():
            dist.broadcast(p.data, src=0)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps,
                                     val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
         f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
