"""
Looped Transformer with Per-Layer LoRA + 6-bit Export for OpenAI Parameter Golf.

Key innovations over baseline:
1. Looped architecture: 5 unique blocks cycled to create 30 virtual layers
2. Per-virtual-layer LoRA adapters on Q,V projections (rank=4)
3. 6-bit quantized export with fp16 embedding passthrough
4. Sliding window evaluation for improved BPB scoring

Target: artifact < 16MB, maximize quality per stored byte.
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

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 10))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape — looped architecture
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    model_dim = int(os.environ.get("MODEL_DIM", 768))
    unique_layers = int(os.environ.get("UNIQUE_LAYERS", 5))
    virtual_depth = int(os.environ.get("VIRTUAL_DEPTH", 30))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    num_heads = int(os.environ.get("NUM_HEADS", 12))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # LoRA config for per-virtual-layer adaptation
    lora_rank = int(os.environ.get("LORA_RANK", 4))

    # Export format
    export_bits = int(os.environ.get("EXPORT_BITS", 6))

    # Sliding window evaluation
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 4096))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))

    # Optimizer hyperparameters
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.03))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    lora_lr = float(os.environ.get("LORA_LR", 0.01))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    swa_checkpoints = int(os.environ.get("SWA_CHECKPOINTS", 7))


# -----------------------------
# LORA MODULE
# -----------------------------

class LoRA(nn.Module):
    """Low-rank adapter for per-virtual-layer differentiation."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * (1.0 / math.sqrt(in_dim)))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype)


class LoRALinear(nn.Module):
    """Linear layer that applies a base weight + LoRA delta selected by layer index."""
    def __init__(self, base_linear: nn.Module, lora_adapters: nn.ModuleList):
        super().__init__()
        self.base = base_linear
        self.adapters = lora_adapters
        self._current_layer_idx = 0

    def set_layer(self, idx: int):
        self._current_layer_idx = idx

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        adapter = self.adapters[self._current_layer_idx]
        return base_out + adapter(x)


# -----------------------------
# NORMUON OPTIMIZER (row-normalized Newton-Schulz)
# -----------------------------

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Per-row normalization (NorMuon): ensures each row contributes equally
    X = X / X.norm(dim=1, keepdim=True).clamp_min(eps)
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
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
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
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
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


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION (unchanged from baseline)
# -----------------------------

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


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
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
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
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


# -----------------------------
# 4-BIT QUANTIZATION
# -----------------------------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
INT6_KEEP_FLOAT_MAX_NUMEL = 65_536
INT6_CLIP_PERCENTILE = 99.99


def quantize_tensor_int6(t):
    """Quantize a float tensor to int6 range [-31, 31], stored as int8."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT6_CLIP_PERCENTILE / 100.0, dim=1).clamp_min(1e-8)
        scale = (clip_abs / 31.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale[:, None].float()), -31, 31).to(torch.int8)
    else:
        clip_abs = torch.quantile(t32.abs().flatten(), INT6_CLIP_PERCENTILE / 100.0).clamp_min(1e-8).item()
        scale = torch.tensor(clip_abs / 31.0, dtype=torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()), -31, 31).to(torch.int8)
    return q, scale


def quantize_state_dict_int6(state_dict):
    """Quantize state dict: int6 for large weights, fp16 passthrough for embeddings/LoRA/small tensors."""
    quantized = {}
    scales = {}
    shapes = {}
    passthrough = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        # Keep tok_emb.weight in fp16 (most quantization-sensitive)
        if "tok_emb.weight" in name:
            passthrough[name] = t.to(torch.float16).contiguous()
            continue
        # Keep LoRA parameters in fp16 (small and sensitive)
        if "lora_q." in name or "lora_v." in name:
            passthrough[name] = t.to(torch.float16).contiguous()
            continue
        if not t.is_floating_point() or t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                passthrough[name] = t.float().contiguous()
            else:
                passthrough[name] = t.to(torch.float16).contiguous()
            continue
        q, s = quantize_tensor_int6(t)
        shapes[name] = list(q.shape)
        quantized[name] = q
        scales[name] = s.contiguous()
    return {"__quant_format__": "int6_v1", "quantized": quantized, "scales": scales, "shapes": shapes, "passthrough": passthrough}


def dequantize_state_dict_int6(obj):
    """Dequantize int6 state dict back to bfloat16."""
    out = {}
    for name, q in obj["quantized"].items():
        shape = obj["shapes"][name]
        s = obj["scales"][name]
        if s.ndim > 0:
            dequant = (q.float() * s.float().view(shape[0], *([1] * (len(shape) - 1)))).to(torch.bfloat16)
        else:
            dequant = (q.float() * s.float().item()).to(torch.bfloat16)
        out[name] = dequant
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out


# -----------------------------
# DATA LOADING (unchanged from baseline)
# -----------------------------

class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
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
        self.rank = rank
        self.world_size = world_size
        self.device = device
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


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

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
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = freqs[None, None, :, :]
        # Use register_buffer to avoid inference-mode tensor issues with torch.compile
        self._cos_cached = nn.Parameter(emb.cos(), requires_grad=False)
        self._sin_cached = nn.Parameter(emb.sin(), requires_grad=False)
        self._seq_len_cached = seq_len

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            self._build_cache(seq_len, device)
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        self.fc = CastedLinear(dim, mlp_mult * dim, bias=False)
        self.proj = CastedLinear(mlp_mult * dim, dim, bias=False)
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

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class LoopedGPT(nn.Module):
    """GPT with looped/cycled transformer blocks, per-layer LoRA, and encoder-decoder skip connections."""
    def __init__(self, vocab_size, model_dim, unique_layers, virtual_depth,
                 num_heads, num_kv_heads, mlp_mult, tie_embeddings,
                 tied_embed_init_std, logit_softcap, rope_base, qk_gain_init, lora_rank):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.virtual_depth = virtual_depth
        self.unique_layers = unique_layers

        # Encoder-decoder split
        self.num_encoder_layers = virtual_depth // 2
        self.num_decoder_layers = virtual_depth - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)

        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(unique_layers)
        ])

        # Per-virtual-layer LoRA adapters on Q and V projections
        kv_dim = num_kv_heads * (model_dim // num_heads)
        self.lora_q = nn.ModuleList([LoRA(model_dim, model_dim, lora_rank) for _ in range(virtual_depth)])
        self.lora_v = nn.ModuleList([LoRA(model_dim, kv_dim, lora_rank) for _ in range(virtual_depth)])

        # Per-virtual-layer scale (learnable, cheap)
        self.layer_scales = nn.Parameter(torch.ones(virtual_depth, dtype=torch.float32))

        # Skip connections from encoder to decoder (learned weights)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

        if tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def _run_layer(self, x, x0, i):
        """Run one virtual layer: resid_mix + LoRA attention + MLP."""
        block = self.blocks[i % self.unique_layers]

        # Residual mixing (blend hidden state with original embedding)
        mix = block.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # LoRA-augmented attention
        norm_x = block.attn_norm(x)
        q = block.attn.c_q(norm_x) + self.lora_q[i](norm_x)
        k = block.attn.c_k(norm_x)
        v = block.attn.c_v(norm_x) + self.lora_v[i](norm_x)

        bsz, seqlen, dim = x.shape
        hd = block.attn.head_dim
        nh = block.attn.num_heads
        nkv = block.attn.num_kv_heads

        q = q.reshape(bsz, seqlen, nh, hd).transpose(1, 2)
        k = k.reshape(bsz, seqlen, nkv, hd).transpose(1, 2)
        v = v.reshape(bsz, seqlen, nkv, hd).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = block.attn.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * block.attn.q_gain.to(dtype=q.dtype)[None, :, None, None]

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(nkv != nh))
        attn_out = block.attn.proj(attn_out.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim))

        # Scale and residual
        scale = self.layer_scales[i].to(dtype=x.dtype)
        x = x + scale * block.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + scale * block.mlp_scale.to(dtype=x.dtype)[None, None, :] * block.mlp(block.mlp_norm(x))
        return x

    def _looped_pass(self, x, x0):
        """Encoder-decoder forward with skip connections."""
        skips = []
        # Encoder half: store skip tensors
        for i in range(self.num_encoder_layers):
            x = self._run_layer(x, x0, i)
            skips.append(x)
        # Decoder half: add skip connections in reverse
        for i in range(self.num_encoder_layers, self.virtual_depth):
            dec_idx = i - self.num_encoder_layers
            if dec_idx < self.num_skip_weights and skips:
                x = x + self.skip_weights[dec_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self._run_layer(x, x0, i)
        return x

    def forward(self, input_ids, target_ids):
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        x = self._looped_pass(x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids):
        """Return logits (B, T, V) without computing loss. Used by sliding window eval."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        x = self._looped_pass(x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# -----------------------------
# SLIDING WINDOW EVALUATION
# -----------------------------

def eval_sliding_window(model, val_tokens, device, base_bytes_lut, has_leading_space_lut,
                        is_boundary_token_lut, eval_seq_len, eval_stride):
    """Sliding-window evaluation: each token scored with nearly full eval_seq_len context."""
    # Unwrap DDP / torch.compile to get the raw model
    raw_model = model
    while hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod

    raw_model.eval()
    total_nll = 0.0
    total_bytes = 0.0
    total_tokens_scored = 0
    n_tokens = val_tokens.numel()

    with torch.inference_mode():
        for start in range(0, n_tokens - eval_seq_len, eval_stride):
            window = val_tokens[start : start + eval_seq_len].to(device=device, dtype=torch.int64).unsqueeze(0)
            input_ids = window[:, :-1]
            target_ids = window[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = raw_model.forward_logits(input_ids)

            # Only score the rightmost eval_stride tokens
            score_start = eval_seq_len - 1 - eval_stride
            logits_slice = logits[:, score_start:, :].reshape(-1, logits.size(-1))
            targets_slice = target_ids[:, score_start:].reshape(-1)

            nll = F.cross_entropy(logits_slice.float(), targets_slice, reduction="sum").item()
            total_nll += nll
            total_tokens_scored += targets_slice.numel()

            # Byte counting for BPB
            prev_ids = input_ids[:, score_start:].reshape(-1)
            tgt_ids = targets_slice
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.float64)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.float64)
            total_bytes += token_bytes.sum().item()

    avg_loss = total_nll / total_tokens_scored
    bits_per_token = avg_loss / math.log(2.0)
    tokens_per_byte = total_tokens_scored / total_bytes
    bpb = bits_per_token * tokens_per_byte
    raw_model.train()
    return avg_loss, bpb


# -----------------------------
# TRAINING (adapted from baseline)
# -----------------------------

def main():
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

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
    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"

    def log0(msg, console=True):
        if not master_process:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0(f"Running PyTorch {torch.__version__}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Build looped model
    base_model = LoopedGPT(
        vocab_size=args.vocab_size, model_dim=args.model_dim,
        unique_layers=args.unique_layers, virtual_depth=args.virtual_depth,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, lora_rank=args.lora_rank,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    for name, p in base_model.named_parameters():
        if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            p.data = p.data.float()

    # Pre-build rotary cache before compile to avoid inference tensor issues
    dummy_seq_len = args.train_seq_len
    for block in base_model.blocks:
        block.attn.rotary._build_cache(dummy_seq_len, device)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer setup: base blocks use Muon, LoRA + scalars use Adam
    block_matrix_params = [p for n, p in base_model.blocks.named_parameters() if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    block_scalar_params = [p for n, p in base_model.blocks.named_parameters() if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    lora_params = list(base_model.lora_q.parameters()) + list(base_model.lora_v.parameters())
    block_scalar_params.append(base_model.layer_scales)
    block_scalar_params.append(base_model.skip_weights)

    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam([{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(block_matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in optimizer_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam([{"params": block_scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_lora = torch.optim.Adam([{"params": lora_params, "lr": args.lora_lr, "base_lr": args.lora_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [optimizer_tok, optimizer_muon, optimizer_scalar, optimizer_lora]
    if not args.tie_embeddings and base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam([{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}], betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        optimizers.append(optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_base_params = sum(p.numel() for p in base_model.blocks.parameters()) + base_model.tok_emb.weight.numel()
    n_lora_params = sum(p.numel() for p in lora_params)
    log0(f"total_params:{n_params} base_params:{n_base_params} lora_params:{n_lora_params}")
    log0(f"unique_layers:{args.unique_layers} virtual_depth:{args.virtual_depth} model_dim:{args.model_dim}")
    log0(f"export_bits:{args.export_bits}")

    # Wallclock-based warmdown scheduling
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        """Compute LR multiplier with warmup + wallclock-aware warmdown."""
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is not None and step > 0:
            step_ms = elapsed_ms / step
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            if remaining_ms <= warmdown_ms:
                return max(remaining_ms / warmdown_ms, 0.0)
        else:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            if step >= warmdown_start:
                return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
        return 1.0

    # Training loop
    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    training_time_ms = 0.0
    tokens_processed = 0
    swa_snapshots = []
    swa_started = False

    for step in range(args.iterations + 1):
        # LR scheduling (wallclock-aware)
        lr_frac = lr_mul(step, training_time_ms)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", g["lr"]) * lr_frac

        # Muon momentum warmup
        if step < args.muon_momentum_warmup_steps:
            frac = step / max(args.muon_momentum_warmup_steps, 1)
            mom = args.muon_momentum_warmup_start + frac * (args.muon_momentum - args.muon_momentum_warmup_start)
            for g in optimizer_muon.param_groups:
                g["momentum"] = mom

        # Validation
        if args.val_loss_every > 0 and (step % args.val_loss_every == 0 or step == args.iterations):
            val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log0(f"step:{step} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")

        if step == args.iterations:
            break

        # Wallclock cap
        if max_wallclock_ms is not None and training_time_ms >= max_wallclock_ms:
            log0(f"Wallclock cap hit at step {step}, {training_time_ms/1000:.1f}s")
            break

        t0 = time.perf_counter()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        for micro in range(grad_accum_steps):
            # Only sync gradients on last micro-step (critical for multi-GPU perf)
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            (loss * grad_scale).backward()
            tokens_processed += x.numel()

        # Gradient clipping
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()

        dt = time.perf_counter() - t0
        training_time_ms += dt * 1000.0

        # SWA: collect snapshots during warmdown
        if lr_frac < 1.0 and args.swa_checkpoints > 0:
            if not swa_started:
                swa_started = True
                swa_interval = max(args.warmdown_iters // args.swa_checkpoints, 1)
                swa_step_counter = 0
            swa_step_counter += 1
            if swa_step_counter % swa_interval == 0:
                swa_snapshots.append({k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()})
                log0(f"step:{step} SWA snapshot {len(swa_snapshots)} captured")

        if step > 0 and step % args.train_log_every == 0:
            log0(f"step:{step} loss:{loss.item():.4f} dt:{dt*1000:.1f}ms lr:{lr_frac:.3f} tokens:{tokens_processed} time:{training_time_ms/1000:.1f}s")

    # Capture final snapshot for SWA if in warmdown
    if swa_started and args.swa_checkpoints > 0:
        swa_snapshots.append({k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()})
        log0(f"SWA final snapshot captured (total: {len(swa_snapshots)})")

    # Apply SWA averaging
    if len(swa_snapshots) > 1:
        avg_sd = {}
        for key in swa_snapshots[0]:
            stacked = torch.stack([s[key].float() for s in swa_snapshots])
            avg_sd[key] = stacked.mean(dim=0).to(swa_snapshots[0][key].dtype)
        base_model.load_state_dict(avg_sd, strict=True)
        log0(f"SWA: averaged {len(swa_snapshots)} checkpoints")
        del swa_snapshots  # free memory

    # Final validation (standard, for training-loop consistency)
    val_loss, val_bpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log0(f"FINAL val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
    log0(f"Train time: {training_time_ms/1000:.1f}s, Tokens: {tokens_processed}")

    # Export
    if master_process:
        sd = {k: v for k, v in base_model.state_dict().items()}
        if args.export_bits == 6:
            quant_obj = quantize_state_dict_int6(sd)
            format_tag = "int6"
        else:
            raise ValueError(f"Unsupported export_bits={args.export_bits}")

        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        raw = buf.getvalue()
        compressed = zlib.compress(raw, level=9)
        model_path = f"logs/{args.run_id}_final_model.{format_tag}.ptz"
        with open(model_path, "wb") as f:
            f.write(compressed)

        code_bytes = len(code.encode("utf-8"))
        model_bytes = len(compressed)
        total_bytes = code_bytes + model_bytes
        log0(f"Serialized model {format_tag}+zlib: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {total_bytes} bytes")

        # Roundtrip verification with sliding window eval
        loaded = torch.load(io.BytesIO(zlib.decompress(compressed)), weights_only=False)
        rt_sd = dequantize_state_dict_int6(loaded)
        rt_model = LoopedGPT(
            vocab_size=args.vocab_size, model_dim=args.model_dim,
            unique_layers=args.unique_layers, virtual_depth=args.virtual_depth,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult, tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
            rope_base=args.rope_base, qk_gain_init=args.qk_gain_init, lora_rank=args.lora_rank,
        ).to(device).bfloat16()
        rt_model.load_state_dict(rt_sd, strict=False)

        rt_val_loss, rt_val_bpb = eval_sliding_window(
            rt_model, val_tokens, device,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            eval_seq_len=args.eval_seq_len, eval_stride=args.eval_stride,
        )
        log0(f"final_{format_tag}_zlib_roundtrip_sliding val_bpb: {rt_val_bpb:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
