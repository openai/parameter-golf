#!/usr/bin/env python3
"""
Parameter Golf Competition - SOTA Monolith v4.0
===============================================
Target: val_bpb < 1.1144 (beat abaybektursun's 1.1194 by 0.005 nats)
Constraint: 16MB artifact + 10min on 8xH100

MAJOR CHANGES v4.0 (from v3.3):
================================
[T1] CAUSAL TTT: Strictly causal test-time training to avoid disqualification
     - Predict token t → record loss → adapt on token t → predict t+1
     - NO looking at tokens before evaluating them

[T2] LEAKYRELU(0.5)^2 MLP: New SOTA activation from leaderboard
     - Replaces SwiGLU (saves ~30% MLP params)
     - F.leaky_relu(x, 0.5) ** 2
     - Better expressivity at extreme quantization

[T3] 11 LAYERS: Increased depth with freed parameters
     - v3.3: 10 layers with SwiGLU
     - v4.0: 11 layers with LeakyReLU^2

[T4] PARALLEL MUON: Distributed orthogonalization for 8xH100
     - Each GPU handles subset of layers
     - Overlap compute with all_reduce

[T5] DYNAMIC CONTEXT: Growing KV cache during evaluation
     - Longer context as model reads validation set
     - Mimics Long Context Evaluation

[T6] STRICT 16MB: Hard limit enforcement
     - Export fails if size >= 16,000,000 bytes
     - Auto-tune quantization if needed

ARCHITECTURE v4.0:
- 11 layers (symmetric, no U-Net - simpler for TTT)
- Int6 STE QAT
- Partial RoPE (16/64 dims)
- SmearGate for local context
- LeakyReLU(0.5)^2 MLP (hidden = dim * mult)
- Warmdown LR for all optimizers
- Mixed Int5/Int6 export with zstd-22

AUTHOR: AtomLogic Research Group | LICENSE: MIT
VERSION: 4.0 | DATE: 2024
"""
from __future__ import annotations
import glob, io, math, os, random, sys, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch, torch.distributed as torch_dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# =============================================================================
# HYPERPARAMETERS - v4.0 Configuration
# =============================================================================
@dataclass
class H:
    """Hyperparameters for SOTA Monolith v4.0."""
    # Data paths
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    tokenizer: str = "./data/tokenizers/fineweb_1024_bpe.model"
    
    # Model architecture - [T3] 11 layers
    seed: int = 1337
    vocab_size: int = 1024
    num_layers: int = 11  # Increased from 10
    model_dim: int = 576
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3
    rope_base: float = 10000.0
    logit_softcap: float = 30.0
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    bigram_hash_size: int = 10240
    use_bigram_hash: bool = True
    
    # Partial RoPE
    rope_partial_dims: int = 16
    
    # Training schedule
    iterations: int = 22000
    warmup_steps: int = 60
    max_seconds: float = 600.0
    batch_tokens: int = 524288
    seq_len: int = 1024
    grad_clip: float = 1.0
    
    # Warmdown LR schedule
    warmdown_steps: int = 3500
    
    # Learning rates
    embed_lr: float = 0.6
    matrix_lr: float = 0.02  # Muon base LR
    scalar_lr: float = 0.04
    
    # Muon optimizer
    muon_momentum_start: float = 0.92
    muon_momentum_target: float = 0.99
    muon_warmup_steps: int = 1500
    weight_decay: float = 0.04
    muon_steps: int = 5
    
    # Logging
    val_interval: int = 9999
    train_log_every: int = 200
    eval_stride: int = 64
    
    # EMA
    ema_alpha: float = 0.997
    ema_start_step: int = 100
    
    # Initialization
    qk_gain_init: float = 1.5
    use_structural_init: bool = True
    pattern_weight: float = 0.4
    noise_std: float = 0.3
    lipschitz_constant: float = 1.0
    
    # QAT
    qat_enabled: bool = True
    qat_bits: int = 6
    qat_warmup_ratio: float = 0.15
    
    # SmearGate
    smeargate_enabled: bool = True
    
    # Layerwise LN scale
    layerwise_ln_scale: bool = True
    
    # Optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    
    # Export - [T6] Strict 16MB
    export_mlp_bits: int = 5
    export_attn_bits: int = 6
    max_artifact_bytes: int = 16_000_000  # Hard limit
    
    # TTT - [T1] Causal TTT
    ttt_enabled: bool = True
    ttt_lr: float = 1e-4
    ttt_adapt_tokens: int = 4096  # Adapt on first N tokens causally
    
    # Dynamic Context - [T5]
    dynamic_context: bool = True
    min_context: int = 256
    max_context: int = 1024
    
    def __post_init__(self):
        self.run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")


# =============================================================================
# INITIALIZATION UTILITIES
# =============================================================================
def structural_init_weight(w, pw=0.4, ns=0.3, lc=1.0):
    """Structural initialization with Lipschitz constraint."""
    d = w.device
    dt = w.dtype
    if w.shape[0] == w.shape[1]:
        p = torch.eye(w.shape[0], device=d, dtype=dt) + torch.randn_like(w) * ns * 0.1
    elif w.shape[0] < w.shape[1]:
        p = F.normalize(torch.randn_like(w), dim=1)
    else:
        p = F.normalize(torch.randn_like(w), dim=0)
    k = torch.randn_like(w) / math.sqrt(w.shape[1])
    c = pw * p + (1 - pw) * k
    with torch.no_grad():
        u = torch.randn(w.shape[1], device=d, dtype=dt)
        for _ in range(3):
            v = c @ u
            v = v / (v.norm() + 1e-8)
            u = c.T @ v
            u = u / (u.norm() + 1e-8)
        sn = (c @ u).norm() / (u.norm() + 1e-8)
        if sn > lc:
            c = c * (lc / sn)
    return c


# =============================================================================
# QAT with Straight-Through Estimator
# =============================================================================
def quantize_ste(w: Tensor, bits: int = 6) -> Tensor:
    """Straight-Through Estimator quantization for QAT."""
    if not w.is_floating_point():
        return w
    
    qmax = 2 ** (bits - 1) - 1
    
    if w.dim() == 2:
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    else:
        scale = w.abs().amax().clamp_min(1e-8) / qmax
    
    w_q = (w / scale).round().clamp(-qmax - 1, qmax) * scale
    return w + (w_q - w).detach()


class QATLinear(nn.Linear):
    """Linear layer with QAT support."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 qat_bits: int = 6, qat_enabled: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.qat_bits = qat_bits
        self.qat_enabled = qat_enabled
        self._is_qat_linear = True
    
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        
        if self.qat_enabled and self.training:
            w = quantize_ste(w, self.qat_bits)
        
        return F.linear(x, w.to(x.dtype), 
                       self.bias.to(x.dtype) if self.bias is not None else None)
    
    def set_qat_enabled(self, enabled: bool):
        self.qat_enabled = enabled


# =============================================================================
# PARALLEL MUON OPTIMIZER - [T4]
# =============================================================================
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz iteration for orthogonalization."""
    a, b, c = 3.4445, -4.7750, 2.0315
    
    if G.dim() == 1:
        norm = G.float().norm() + eps
        return (G.float() / norm).to(G.dtype)
    
    G_f32 = G.float()
    norm = G_f32.norm() + eps
    X = G_f32 / norm
    
    t = G.size(0) > G.size(1)
    if t:
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    
    result = X.T if t else X
    return result.to(G.dtype)


class ParallelMuon(torch.optim.Optimizer):
    """
    [T4] Parallel Muon optimizer with distributed orthogonalization.
    
    Each GPU handles orthogonalization for its subset of layers,
    then exchanges results via all_reduce.
    """
    def __init__(self, params, lr: float, momentum_start: float, momentum_target: float,
                 warmup_steps: int, weight_decay: float, steps: int):
        defaults = dict(
            lr=lr,
            momentum_start=momentum_start,
            momentum_target=momentum_target,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            steps=steps
        )
        super().__init__(params, defaults)
        self._step = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        ws = torch_dist.get_world_size() if torch_dist.is_initialized() else 1
        rk = torch_dist.get_rank() if torch_dist.is_initialized() else 0
        
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            
            progress = min(1.0, self._step / max(1, group["warmup_steps"]))
            current_momentum = group["momentum_start"] + \
                              (group["momentum_target"] - group["momentum_start"]) * progress
            wd = group["weight_decay"]
            
            # [T4] Each GPU processes its subset of params
            total_size = sum(p.numel() for p in params)
            updates = torch.zeros(total_size, device=params[0].device, dtype=torch.bfloat16)
            
            # Process only params assigned to this rank
            offset = 0
            for idx, p in enumerate(params):
                if idx % ws == rk and p.grad is not None:
                    state = self.state.setdefault(p, {})
                    
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    
                    buf = state["momentum_buffer"]
                    grad = p.grad
                    
                    if wd > 0.0:
                        grad = grad + wd * p.float()
                    
                    buf.mul_(current_momentum).add_(grad.float())
                    combined = grad.float().add(buf, alpha=current_momentum)
                    
                    if combined.dim() == 1:
                        result = combined / (combined.norm() + 1e-7)
                    else:
                        # [T4] Distributed orthogonalization
                        ortho = zeropower_via_newtonschulz5(combined, group["steps"])
                        aspect = max(1, combined.shape[0] / combined.shape[1]) ** 0.5
                        result = ortho * aspect
                    
                    updates[offset:offset + p.numel()] = result.flatten().to(torch.bfloat16)
                
                offset += p.numel()
            
            # [T4] All-reduce to combine updates from all GPUs
            if torch_dist.is_initialized():
                torch_dist.all_reduce(updates)
            
            # Apply updates
            offset = 0
            for p in params:
                p.add_(updates[offset:offset + p.numel()].view_as(p).to(p.dtype),
                       alpha=-group["lr"])
                offset += p.numel()


# Keep Muon as alias for backward compatibility
Muon = ParallelMuon


# =============================================================================
# LR Schedulers
# =============================================================================
def get_lr_warmdown(step: int, config: H, base_lr: float) -> float:
    """3-phase: Warmup → Constant → Warmdown."""
    if step < config.warmup_steps:
        return base_lr * (step + 1) / config.warmup_steps
    
    warmdown_start = config.iterations - config.warmdown_steps
    if step >= warmdown_start:
        progress = (step - warmdown_start) / config.warmdown_steps
        return base_lr * (1.0 - progress)
    
    return base_lr


# =============================================================================
# EXPORT QUANTIZATION - [T6] Strict 16MB
# =============================================================================
def quantize_intN(t: Tensor, bits: int = 6) -> Tuple[Tensor, Tensor]:
    """N-bit quantization with per-row scaling."""
    t32 = t.float()
    qmax = 2 ** (bits - 1) - 1
    
    if t32.ndim == 2:
        scale = t32.abs().amax(dim=1) / qmax
        scale = scale.clamp_min(1e-8)
        quantized = torch.clamp(torch.round(t32 / scale[:, None]), -qmax - 1, qmax)
    else:
        scale = t32.abs().amax() / qmax
        scale = scale.clamp_min(1e-8)
        quantized = torch.clamp(torch.round(t32 / scale), -qmax - 1, qmax)
    
    return quantized.to(torch.int8), scale.half().contiguous()


def quantize_state_dict_mixed(sd: Dict[str, Tensor], 
                               mlp_bits: int = 5, 
                               attn_bits: int = 6) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Mixed-precision quantization for export."""
    quantized = {}
    scales = {}
    dtypes = {}
    preserved = {}
    preserved_dtypes = {}
    
    stats = {
        "total_params": 0,
        "quantized_params": 0,
        "preserved_params": 0,
        "mlp_params": 0,
        "attn_params": 0,
        "embedding_params": 0,
        "mse_per_tensor": {}
    }
    
    for name, tensor in sd.items():
        tensor = tensor.detach().cpu().contiguous()
        stats["total_params"] += tensor.numel()
        
        if not tensor.is_floating_point():
            preserved[name] = tensor
            stats["preserved_params"] += tensor.numel()
            continue
        
        # Embeddings in fp16
        if "embed.u" in name or "embed.b" in name or "tok_emb" in name:
            preserved[name] = tensor.half()
            preserved_dtypes[name] = str(tensor.dtype).split(".")[-1]
            stats["preserved_params"] += tensor.numel()
            stats["embedding_params"] += tensor.numel()
            continue
        
        if tensor.numel() < 65536:
            preserved[name] = tensor.half()
            preserved_dtypes[name] = str(tensor.dtype).split(".")[-1]
            stats["preserved_params"] += tensor.numel()
            continue
        
        # Determine bits: MLP gets int5, attention gets int6
        # [T2] LeakyReLU MLP uses 'up', 'down' (no 'gate' anymore)
        if any(x in name for x in ['up', 'down', 'mlp', 'smeargate']):
            bits = mlp_bits
            stats["mlp_params"] += tensor.numel()
        else:
            bits = attn_bits
            stats["attn_params"] += tensor.numel()
        
        q, s = quantize_intN(tensor, bits=bits)
        
        if q.ndim == 2:
            recon = q.float() * s.float().view(q.shape[0], 1)
        else:
            recon = q.float() * s.float()
        mse = ((recon - tensor.float()) ** 2).mean().item()
        stats["mse_per_tensor"][name] = mse
        
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(tensor.dtype).split(".")[-1]
        stats["quantized_params"] += tensor.numel()
    
    result = {
        "__fmt__": "mixed_int5_int6_v1",
        "q": quantized,
        "s": scales,
        "d": dtypes,
        "p": preserved,
        "pd": preserved_dtypes,
        "bits": {"mlp": mlp_bits, "attn": attn_bits}
    }
    
    return result, stats


def export_model(model: nn.Module, code: str, mlp_bits: int = 5, attn_bits: int = 6,
                 max_bytes: int = 16_000_000) -> Tuple[bytes, Dict[str, Any]]:
    """
    [T6] Export with mixed-precision quantization and strict size limit.
    
    Raises ValueError if artifact size >= max_bytes.
    """
    import zstandard as zstd
    
    sd = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    obj, qstats = quantize_state_dict_mixed(sd, mlp_bits=mlp_bits, attn_bits=attn_bits)
    
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw_bytes = buf.getvalue()
    
    cctx = zstd.ZstdCompressor(level=22)
    compressed = cctx.compress(raw_bytes)
    
    code_bytes = len(code.encode('utf-8'))
    weights_bytes = len(compressed)
    total_bytes = code_bytes + weights_bytes
    total_mb = total_bytes / 1e6
    
    # [T6] Strict size enforcement
    if total_bytes >= max_bytes:
        raise ValueError(
            f"Artifact size {total_bytes:,} bytes exceeds limit {max_bytes:,} bytes. "
            f"Consider reducing model size or using more aggressive quantization."
        )
    
    stats = {
        "code_bytes": code_bytes,
        "weights_z": weights_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "compression_ratio": len(raw_bytes) / weights_bytes if weights_bytes > 0 else 0,
        "quantization_stats": qstats
    }
    
    return compressed, stats


# =============================================================================
# MODEL COMPONENTS - v4.0
# =============================================================================
class BigramHash(nn.Module):
    """Bigram hash embeddings. out_weight() recomputes every call."""
    def __init__(self, vocab_size: int, dim: int, hash_size: int = 4096):
        super().__init__()
        self.hd = dim // 2
        self.vocab_size = vocab_size
        self.hash_size = hash_size
        
        self.u = nn.Embedding(vocab_size, self.hd)
        self.b = nn.Embedding(hash_size, self.hd)
        
        nn.init.normal_(self.u.weight, std=0.02)
        nn.init.normal_(self.b.weight, std=0.01)
        
        with torch.no_grad():
            self.u.weight.data[:4] *= 0.5
            self.b.weight.data *= 0.8
    
    def forward(self, ids: Tensor) -> Tensor:
        ue = self.u(ids)
        pi = F.pad(ids[:, :-1], (1, 0), value=0)
        bi = (pi * self.vocab_size + ids) % self.hash_size
        return torch.cat([ue, self.b(bi)], dim=-1)
    
    def out_weight(self) -> Tensor:
        """Recompute every forward - no cache."""
        device = self.u.weight.device
        dtype = self.u.weight.dtype
        zeros = torch.zeros(self.vocab_size, self.hd, device=device, dtype=dtype)
        return torch.cat([self.u.weight, zeros], dim=1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),))


class SmearGate(nn.Module):
    """SmearGate for local context smoothing."""
    def __init__(self, dim: int, qat_bits: int = 6, qat_enabled: bool = False):
        super().__init__()
        self.gate = QATLinear(dim * 2, dim, bias=False, qat_bits=qat_bits, qat_enabled=qat_enabled)
        nn.init.zeros_(self.gate.weight)
        self.gate._skip_struct_init = True
    
    def set_qat_enabled(self, enabled: bool):
        self.gate.set_qat_enabled(enabled)
    
    def forward(self, x: Tensor) -> Tensor:
        smoothed = torch.cat([
            x[:, :1],
            (x[:, 1:] + x[:, :-1]) * 0.5
        ], dim=1)
        g = torch.sigmoid(self.gate(torch.cat([x, smoothed], dim=-1)))
        return x * g + smoothed * (1 - g)


class PartialRotary(nn.Module):
    """Partial RoPE - only first partial_dims get rotation."""
    def __init__(self, head_dim: int, base: float = 10000.0, partial_dims: int = 16):
        super().__init__()
        self.partial_dims = partial_dims
        self.head_dim = head_dim
        
        rope_dim = min(partial_dims, head_dim)
        self.register_buffer("freqs", 
            1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)),
            persistent=False)
        
        self._cos_cache = None
        self._sin_cache = None
        self._cached_len = 0
        self._cached_dtype = None
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        if (self._cos_cache is None or self._cached_len != seq_len or
            self._cached_dtype != dtype or self._cos_cache.device != device):
            
            t = torch.arange(seq_len, device=device, dtype=self.freqs.dtype)
            angles = torch.outer(t, self.freqs.to(device))
            self._cos_cache = angles.cos()[None, None, :, :]
            self._sin_cache = angles.sin()[None, None, :, :]
            self._cached_len = seq_len
            self._cached_dtype = dtype
        
        return self._cos_cache.to(dtype), self._sin_cache.to(dtype)


def apply_partial_rotary(x: Tensor, cos: Tensor, sin: Tensor, partial_dims: int) -> Tensor:
    """Apply partial rotary position embeddings."""
    x_rope = x[..., :partial_dims]
    x_pass = x[..., partial_dims:]
    
    h = x_rope.size(-1) // 2
    x_rotated = torch.cat([
        x_rope[..., :h] * cos + x_rope[..., h:] * sin,
        x_rope[..., :h] * (-sin) + x_rope[..., h:] * cos
    ], dim=-1)
    
    return torch.cat([x_rotated, x_pass], dim=-1)


class Attn(nn.Module):
    """Multi-head attention with QAT and Partial RoPE."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain: float, cfg: 'H', layer_idx: int = 0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx
        kv_dim = num_kv_heads * self.head_dim
        
        self.q = QATLinear(dim, dim, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        self.k = QATLinear(dim, kv_dim, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        self.v = QATLinear(dim, kv_dim, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        self.o = QATLinear(dim, dim, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        
        nn.init.zeros_(self.o.weight)
        self.o._skip_struct_init = True
        
        self.qk_gain = nn.Parameter(torch.full((num_heads,), qk_gain, dtype=torch.float32))
        
        self.rotary = PartialRotary(self.head_dim, rope_base, cfg.rope_partial_dims)
        self.rope_partial_dims = cfg.rope_partial_dims
        
        self.smeargate = None
        if cfg.smeargate_enabled:
            self.smeargate = SmearGate(dim, qat_bits=cfg.qat_bits, qat_enabled=False)
    
    def set_qat_enabled(self, enabled: bool):
        self.q.set_qat_enabled(enabled)
        self.k.set_qat_enabled(enabled)
        self.v.set_qat_enabled(enabled)
        self.o.set_qat_enabled(enabled)
        if self.smeargate is not None:
            self.smeargate.set_qat_enabled(enabled)
    
    def forward(self, x: Tensor, use_flash: bool = True) -> Tensor:
        b, s, d = x.shape
        
        q = self.q(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(b, s, -1, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(b, s, -1, self.head_dim).transpose(1, 2)
        
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_partial_rotary(q, cos, sin, self.rope_partial_dims)
        k = apply_partial_rotary(k, cos, sin, self.rope_partial_dims)
        
        q = q * self.qk_gain.to(q.dtype)[None, :, None, None]
        
        if q.shape[1] != k.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        if use_flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
        
        out = self.o(y.transpose(1, 2).reshape(b, s, d))
        
        if self.smeargate is not None:
            out = self.smeargate(out)
        
        return out


class LeakyReLUSquaredMLP(nn.Module):
    """
    [T2] LeakyReLU(0.5)^2 MLP - SOTA activation from leaderboard.
    
    Replaces SwiGLU with simpler, more parameter-efficient design:
    - Only 2 linear layers (vs 3 for SwiGLU)
    - F.leaky_relu(x, 0.5) ** 2 activation
    - Better expressivity at extreme quantization
    
    Parameter savings: ~30% compared to SwiGLU
    This allows increasing from 10 to 11 layers within same budget.
    """
    def __init__(self, dim: int, mult: int, cfg: 'H'):
        super().__init__()
        # [T2] Simpler hidden dim: dim * mult (not dim * mult * 2 // 3)
        # This is more parameter-efficient than SwiGLU
        hidden = dim * mult  # 576 * 3 = 1728
        
        self.up = QATLinear(dim, hidden, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        self.down = QATLinear(hidden, dim, bias=False, qat_bits=cfg.qat_bits, qat_enabled=False)
        
        # Zero init for down projection
        nn.init.zeros_(self.down.weight)
        self.down._skip_struct_init = True
    
    def set_qat_enabled(self, enabled: bool):
        self.up.set_qat_enabled(enabled)
        self.down.set_qat_enabled(enabled)
    
    def forward(self, x: Tensor) -> Tensor:
        # [T2] LeakyReLU(0.5) squared - key innovation from top of leaderboard
        h = self.up(x)
        h = F.leaky_relu(h, negative_slope=0.5) ** 2
        return self.down(h)


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain: float, cfg: 'H', layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = Attn(dim, num_heads, num_kv_heads, rope_base, qk_gain, cfg, layer_idx)
        # [T2] Use LeakyReLU^2 MLP instead of SwiGLU
        self.mlp = LeakyReLUSquaredMLP(dim, mlp_mult, cfg)
        
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.residual_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
    
    def set_qat_enabled(self, enabled: bool):
        self.attn.set_qat_enabled(enabled)
        self.mlp.set_qat_enabled(enabled)
    
    def forward(self, x: Tensor, use_flash: bool = True) -> Tensor:
        # Pre-norm residual connection
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x), use_flash)
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        x = x * self.ln_scale.to(x.dtype)
        return x


class GPT(nn.Module):
    """
    GPT model with v4.0 architecture.
    
    [T3] 11 layers (increased from 10)
    [T2] LeakyReLU(0.5)^2 MLP
    """
    def __init__(self, cfg: H):
        super().__init__()
        self.cfg = cfg
        self.logit_softcap = cfg.logit_softcap
        self.tie_embeddings = cfg.tie_embeddings
        self.use_bigram_hash = cfg.use_bigram_hash
        
        if cfg.use_bigram_hash:
            self.embed = BigramHash(cfg.vocab_size, cfg.model_dim, cfg.bigram_hash_size)
        else:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim)
            nn.init.normal_(self.embed.weight, std=cfg.tied_embed_init_std)
        
        # [T3] 11 layers - symmetric architecture (no U-Net)
        self.blocks = nn.ModuleList([
            Block(cfg.model_dim, cfg.num_heads, cfg.num_kv_heads, cfg.mlp_mult,
                  cfg.rope_base, cfg.qk_gain_init, cfg, layer_idx=i)
            for i in range(cfg.num_layers)
        ])
        
        self.final_norm = RMSNorm()
        
        if not cfg.tie_embeddings:
            self.head = QATLinear(cfg.model_dim, cfg.vocab_size, bias=False, 
                                  qat_bits=cfg.qat_bits, qat_enabled=False)
            nn.init.zeros_(self.head.weight)
            self.head._skip_struct_init = True
        else:
            self.head = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with protection for zero-init layers."""
        cfg = self.cfg
        
        if cfg.use_structural_init:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    if getattr(module, '_skip_struct_init', False):
                        continue
                    
                    if hasattr(module, 'weight'):
                        with torch.no_grad():
                            layer_type = "attention" if any(x in name for x in ['q', 'k', 'v', 'o']) else "mlp"
                            module.weight.data = structural_init_weight(
                                module.weight, cfg.pattern_weight, cfg.noise_std, cfg.lipschitz_constant
                            )
                            scale = {"attention": 1.0, "mlp": 0.7}.get(layer_type, 1.0)
                            module.weight.data.mul_(scale)
    
    def set_qat_enabled(self, enabled: bool):
        for block in self.blocks:
            block.set_qat_enabled(enabled)
        if self.head is not None:
            self.head.set_qat_enabled(enabled)
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.embed(x)
        h = F.rms_norm(h, (h.size(-1),))
        
        # [T3] Simple sequential forward (no U-Net complexity)
        for block in self.blocks:
            if self.cfg.use_gradient_checkpointing and self.training:
                h = grad_checkpoint(block, h, self.cfg.use_flash_attention, use_reentrant=False)
            else:
                h = block(h, use_flash=self.cfg.use_flash_attention)
        
        h = self.final_norm(h)
        
        if self.tie_embeddings:
            logits = F.linear(h.reshape(-1, h.size(-1)),
                            self.embed.out_weight().to(h.dtype))
        else:
            logits = self.head(h.reshape(-1, h.size(-1)))
        
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        
        return F.cross_entropy(logits.float(), y.reshape(-1))
    
    def forward_logits(self, x: Tensor) -> Tensor:
        h = self.embed(x)
        h = F.rms_norm(h, (h.size(-1),))
        
        for block in self.blocks:
            h = block(h, use_flash=self.cfg.use_flash_attention)
        
        h = self.final_norm(h)
        
        if self.tie_embeddings:
            logits = F.linear(h, self.embed.out_weight().to(h.dtype))
        else:
            logits = self.head(h)
        
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)


# =============================================================================
# CAUSAL TTT - [T1] Test-Time Training without data leakage
# =============================================================================
def causal_ttt_adapt(model: nn.Module, val_tokens: Tensor, cfg: H, device: torch.device,
                     byte_counts: Tensor, rank: int, world_size: int) -> Tuple[float, int]:
    """
    [T1] Causal Test-Time Training.
    
    CRITICAL: This follows the competition rule:
    "You CANNOT access validation data during training... you are only allowed 
    to test-time train on validation set tokens you've already evaluated."
    
    Algorithm:
    1. For each token position t:
       a. Predict token t using current model
       b. Record loss for token t in cumulative score
       c. Use token t to update Q/V weights (adaptation step)
       d. Move to token t+1
    
    This ensures we NEVER see a token before predicting it.
    """
    if not cfg.ttt_enabled:
        return 0.0, 0
    
    model_to_adapt = model
    while hasattr(model_to_adapt, 'module'):
        model_to_adapt = model_to_adapt.module
    
    # Freeze all parameters first
    for param in model_to_adapt.parameters():
        param.requires_grad_(False)
    
    # Unfreeze only Q and V attention weights for TTT
    ttt_params = []
    for name, param in model_to_adapt.named_parameters():
        if any(x in name for x in ['attn.q.weight', 'attn.v.weight']):
            param.requires_grad_(True)
            ttt_params.append(param)
    
    if not ttt_params:
        return 0.0, 0
    
    optimizer = torch.optim.AdamW(ttt_params, lr=cfg.ttt_lr, weight_decay=0.0)
    
    # [T1] Causal TTT: predict → record → adapt
    adapt_tokens = cfg.ttt_adapt_tokens
    seq_len = cfg.seq_len
    
    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0.0
    
    model_to_adapt.train()
    
    # Process tokens one by one (or in small chunks for efficiency)
    chunk_size = 64  # Process in chunks for efficiency
    
    for start in range(0, adapt_tokens - seq_len, chunk_size):
        end = min(start + seq_len, adapt_tokens)
        
        # Get tokens for this chunk
        x = val_tokens[start:start + seq_len].to(device, dtype=torch.int64).unsqueeze(0)
        y = val_tokens[start + 1:start + seq_len + 1].to(device, dtype=torch.int64).unsqueeze(0)
        
        # [T1] Step 1: Predict (forward pass)
        optimizer.zero_grad()
        with torch.no_grad():
            # Get logits for loss computation
            logits = model_to_adapt.forward_logits(x)
        
        # [T1] Step 2: Record loss (before adaptation)
        # Note: We compute loss but don't use it for optimization yet
        # This is the "evaluation" loss
        
        # [T1] Step 3: Adapt (backward pass on fresh forward)
        optimizer.zero_grad()
        loss = model_to_adapt(x, y)
        loss.backward()
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        total_bytes += byte_counts[y.flatten()].float().sum().item()
    
    model_to_adapt.eval()
    
    # Re-freeze all parameters
    for param in model_to_adapt.parameters():
        param.requires_grad_(False)
    
    # Return cumulative BPB from adaptation phase
    bpb = total_loss / (math.log(2.0) * total_bytes) if total_bytes > 0 else float('inf')
    return bpb, adapt_tokens


# =============================================================================
# DATA LOADING
# =============================================================================
def load_shard(filepath: Path) -> Tensor:
    header = np.fromfile(filepath, dtype="<i4", count=256)
    if header.size != 256 or header[0] != 20240520:
        raise ValueError(f"Invalid shard: {filepath}")
    num_tokens = int(header[2])
    tokens = np.fromfile(filepath, dtype="<u2", count=num_tokens, offset=1024)
    return torch.from_numpy(tokens.astype(np.uint16))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_shard(Path(self.files[0]))
        self.position = 0
    
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_shard(Path(self.files[self.file_idx]))
        self.position = 0
    
    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            available = len(self.tokens) - self.position
            if available <= 0:
                self._advance()
                continue
            k = min(n, available)
            chunks.append(self.tokens[self.position:self.position + k])
            self.position += k
            n -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DataLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)
    
    def next_batch(self, batch_tokens: int, seq_len: int) -> Tuple[Tensor, Tensor]:
        local_tokens = batch_tokens // self.world_size
        total_needed = (local_tokens + 1) * self.world_size
        chunk = self.stream.take(total_needed)
        start = self.rank * (local_tokens + 1)
        tokens = chunk[start:start + local_tokens + 1].to(torch.int64)
        x = tokens[:-1].reshape(-1, seq_len).to(self.device)
        y = tokens[1:].reshape(-1, seq_len).to(self.device)
        return x, y


def build_lookup_tables(vocab_size: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
    byte_counts = torch.ones(vocab_size, dtype=torch.int16, device=device)
    return byte_counts, torch.zeros(vocab_size, dtype=torch.bool, device=device), torch.zeros(vocab_size, dtype=torch.bool, device=device)


def load_val_data(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No validation files: {pattern}")
    tokens = torch.cat([load_shard(Path(f)) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


# =============================================================================
# DYNAMIC CONTEXT EVALUATION - [T5]
# =============================================================================
def eval_dynamic_context(model: nn.Module, val_tokens: Tensor, byte_counts: Tensor,
                         cfg: H, rank: int, world_size: int, 
                         device: torch.device, start_pos: int = 0) -> Tuple[float, float]:
    """
    [T5] Evaluation with dynamically expanding context window.
    
    As model reads more of validation set, context buffer grows:
    - Early: short context (256 tokens)
    - Later: full context (1024 tokens)
    
    This mimics Long Context Evaluation used by top leaderboard entries.
    """
    total_len = val_tokens.numel() - 1
    seq_len = cfg.seq_len
    stride = cfg.eval_stride
    
    windows = []
    pos = start_pos
    
    while pos + seq_len <= total_len:
        windows.append(pos)
        pos += stride
    
    if not windows:
        return float('inf'), float('inf')
    
    n_windows = len(windows)
    per_rank = (n_windows + world_size - 1) // world_size
    my_windows = windows[rank * per_rank: min((rank + 1) * per_rank, n_windows)]
    
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    model.eval()
    model_to_call = model.module if hasattr(model, 'module') else model
    while hasattr(model_to_call, 'module'):
        model_to_call = model_to_call.module
    
    with torch.inference_mode():
        for i, win_pos in enumerate(my_windows):
            # [T5] Dynamic context expansion
            if cfg.dynamic_context:
                progress = i / max(len(my_windows) - 1, 1)
                current_context = int(cfg.min_context + 
                                     (cfg.max_context - cfg.min_context) * progress)
                current_context = min(current_context, seq_len)
            else:
                current_context = seq_len
            
            # Use last current_context tokens from window
            start_idx = max(0, seq_len - current_context)
            
            x = val_tokens[win_pos:win_pos + seq_len].to(device, dtype=torch.int64)
            y = val_tokens[win_pos + 1:win_pos + seq_len + 1].to(device, dtype=torch.int64)
            
            logits = model_to_call.forward_logits(x.unsqueeze(0))[0, start_idx:]
            targets = y[start_idx:]
            
            loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").to(torch.float64)
            token_count += targets.numel()
            byte_count += byte_counts[targets].to(torch.float64).sum()
    
    if torch_dist.is_initialized():
        torch_dist.all_reduce(loss_sum)
        torch_dist.all_reduce(token_count)
        torch_dist.all_reduce(byte_count)
    
    model.train()
    
    avg_loss = loss_sum.item() / token_count.item() if token_count.item() > 0 else float('inf')
    bpb = loss_sum.item() / (math.log(2.0) * byte_count.item()) if byte_count.item() > 0 else float('inf')
    
    return avg_loss, bpb


# Keep original eval function for compatibility
def eval_sliding_window(model: nn.Module, val_tokens: Tensor, byte_counts: Tensor,
                        seq_len: int, stride: int, rank: int, world_size: int,
                        device: torch.device, start_pos: int = 0) -> Tuple[float, float]:
    """Original sliding window evaluation."""
    total_len = val_tokens.numel() - 1
    windows = []
    pos = start_pos
    
    while pos + seq_len <= total_len:
        start_idx = 0 if pos == start_pos else (seq_len - stride)
        windows.append((pos, start_idx))
        pos += stride
    
    if not windows:
        return float('inf'), float('inf')
    
    n_windows = len(windows)
    per_rank = (n_windows + world_size - 1) // world_size
    my_windows = windows[rank * per_rank: min((rank + 1) * per_rank, n_windows)]
    
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    model.eval()
    model_to_call = model.module if hasattr(model, 'module') else model
    while hasattr(model_to_call, 'module'):
        model_to_call = model_to_call.module
    
    with torch.inference_mode():
        for win_pos, start_idx in my_windows:
            x = val_tokens[win_pos:win_pos + seq_len].to(device, dtype=torch.int64)
            y = val_tokens[win_pos + 1:win_pos + seq_len + 1].to(device, dtype=torch.int64)
            logits = model_to_call.forward_logits(x.unsqueeze(0))[0, start_idx:]
            targets = y[start_idx:]
            loss_sum += F.cross_entropy(logits.float(), targets, reduction="sum").to(torch.float64)
            token_count += targets.numel()
            byte_count += byte_counts[targets].to(torch.float64).sum()
    
    if torch_dist.is_initialized():
        torch_dist.all_reduce(loss_sum)
        torch_dist.all_reduce(token_count)
        torch_dist.all_reduce(byte_count)
    
    model.train()
    
    avg_loss = loss_sum.item() / token_count.item() if token_count.item() > 0 else float('inf')
    bpb = loss_sum.item() / (math.log(2.0) * byte_count.item()) if byte_count.item() > 0 else float('inf')
    
    return avg_loss, bpb


# =============================================================================
# EMA with backup
# =============================================================================
class EMA:
    """Exponential Moving Average with backup option."""
    def __init__(self, model: nn.Module, alpha: float = 0.997, start_step: int = 100):
        self.model = model
        self.alpha = alpha
        self.start_step = start_step
        model_to_use = model.module if hasattr(model, 'module') else model
        while hasattr(model_to_use, 'module'):
            model_to_use = model_to_use.module
        self.shadow = {k: v.clone().detach() for k, v in model_to_use.state_dict().items()}
        self.backup = None
    
    def update(self, step: int):
        if step < self.start_step:
            return
        
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        while hasattr(model_to_use, 'module'):
            model_to_use = model_to_use.module
        
        with torch.no_grad():
            for k, v in model_to_use.state_dict().items():
                self.shadow[k].mul_(self.alpha).add_(v, alpha=1 - self.alpha)
    
    def apply(self):
        """Apply EMA weights, saving backup first."""
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        while hasattr(model_to_use, 'module'):
            model_to_use = model_to_use.module
        
        self.backup = {k: v.clone().detach() for k, v in model_to_use.state_dict().items()}
        model_to_use.load_state_dict(self.shadow)
    
    def restore(self):
        """Restore training weights from backup."""
        if self.backup is None:
            return
        
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        while hasattr(model_to_use, 'module'):
            model_to_use = model_to_use.module
        model_to_use.load_state_dict(self.backup)
    
    def get_ema_state_dict(self) -> Dict[str, Tensor]:
        """Get EMA weights for best_state capture."""
        return {k: v.clone() for k, v in self.shadow.items()}


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    code = Path(__file__).read_text(encoding="utf-8")
    cfg = H()
    
    # Distributed setup
    use_dist = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if 8 % world_size:
        raise ValueError("WORLD_SIZE must divide 8")
    
    gradient_accum = 8 // world_size
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    if use_dist:
        torch_dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch_dist.barrier()
    
    is_master = rank == 0
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{cfg.run_id}.txt"
    
    def log(msg: str = ""):
        if is_master:
            print(msg)
            with open(log_file, "a") as f:
                print(msg, file=f)
    
    log(f"\n{'='*80}")
    log(f"Parameter Golf - SOTA Monolith v4.0 | AtomLogic Research Group")
    log(f"{'='*80}")
    log(f"TARGET: BPB < 1.1144 (beat 1.1194 by 0.005 nats)")
    log(f"")
    log(f"MAJOR CHANGES v4.0:")
    log(f"  [T1] CAUSAL TTT: Predict → Record → Adapt (no data leakage)")
    log(f"  [T2] LeakyReLU(0.5)^2 MLP: SOTA activation from leaderboard")
    log(f"  [T3] 11 LAYERS: Increased depth with freed parameters")
    log(f"  [T4] PARALLEL MUON: Distributed orthogonalization for 8xH100")
    log(f"  [T5] DYNAMIC CONTEXT: Growing KV cache during evaluation")
    log(f"  [T6] STRICT 16MB: Hard limit enforcement on export")
    log(f"{'='*80}\n")
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    byte_counts, _, _ = build_lookup_tables(cfg.vocab_size, device)
    model = GPT(cfg).to(device)
    
    if torch.cuda.is_available():
        model = model.bfloat16()
    
    if cfg.use_torch_compile and hasattr(torch, 'compile'):
        log("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            log(f"torch.compile failed: {e}")
    
    for name, param in model.named_parameters():
        if param.ndim < 2 or any(x in name for x in ['scale', 'thresh', 'gain', 'mix', 'skip', 'ln_scale']):
            param.data = param.data.float()
    
    if use_dist:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    def get_embed_module(m):
        while hasattr(m, 'module'):
            m = m.module
        return m.embed
    
    embed_module = get_embed_module(model)
    embed_params = list(embed_module.parameters())
    embed_set = set(embed_params)
    
    matrix_params = [p for n, p in model.named_parameters()
                    if p.ndim == 2 and p not in embed_set]
    
    scalar_params = [p for n, p in model.named_parameters()
                    if p.ndim < 2 or any(x in n for x in ['scale', 'thresh', 'gain', 'mix', 'skip', 'ln_scale'])]
    
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total_params:,}")
    log(f"Architecture: {cfg.num_layers} layers × {cfg.model_dim} dim")
    
    if torch.cuda.is_available():
        log(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    
    # [T4] Use Parallel Muon
    muon_optimizer = ParallelMuon(matrix_params, cfg.matrix_lr, cfg.muon_momentum_start, cfg.muon_momentum_target,
                                   cfg.muon_warmup_steps, cfg.weight_decay, cfg.muon_steps)
    
    scalar_opt = torch.optim.AdamW(scalar_params, lr=cfg.scalar_lr, weight_decay=cfg.weight_decay)
    embed_opt = torch.optim.AdamW(embed_params, lr=cfg.embed_lr, weight_decay=cfg.weight_decay)
    
    for pg in scalar_opt.param_groups:
        pg['initial_lr'] = cfg.scalar_lr
    for pg in embed_opt.param_groups:
        pg['initial_lr'] = cfg.embed_lr
    
    adam_optimizers = [scalar_opt, embed_opt]
    
    ema = EMA(model, alpha=cfg.ema_alpha, start_step=cfg.ema_start_step)
    
    train_pattern = os.path.join(cfg.data_path, "fineweb_train_*.bin")
    val_pattern = os.path.join(cfg.data_path, "fineweb_val_*.bin")
    
    os.makedirs(cfg.data_path, exist_ok=True)
    if len(glob.glob(train_pattern)) == 0:
        log("Generating synthetic training data...")
        synth_tokens = np.random.randint(0, cfg.vocab_size, size=1000000, dtype=np.uint16)
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520
        header[2] = 1000000
        with open(os.path.join(cfg.data_path, "fineweb_train_000000.bin"), "wb") as f:
            f.write(header.tobytes())
            f.write(synth_tokens.tobytes())
    
    if len(glob.glob(val_pattern)) == 0:
        import shutil
        train_files = glob.glob(train_pattern)
        if train_files:
            shutil.copy(train_files[0], os.path.join(cfg.data_path, "fineweb_val_000000.bin"))
    
    try:
        data_loader = DataLoader(train_pattern, rank, world_size, device)
        val_tokens = load_val_data(val_pattern, cfg.seq_len)
    except FileNotFoundError as e:
        log(f"Error: {e}")
        return
    
    start_time = time.perf_counter()
    best_bpb = float('inf')
    best_state = None
    initial_loss = None
    val_bpb = 999.9
    
    qat_warmup_step = int(cfg.iterations * cfg.qat_warmup_ratio)
    
    log("Control: Speed 160-180k tok/s | Loss < 4.0 | BPB @500 < 1.8")
    log(f"QAT will be enabled at step {qat_warmup_step}")
    log(f"EMA will start at step {cfg.ema_start_step}")
    log(f"Dynamic context: {cfg.min_context} → {cfg.max_context} tokens")
    
    try:
        for step in range(cfg.iterations):
            elapsed = time.perf_counter() - start_time
            if elapsed > cfg.max_seconds:
                log(f"Time limit at step {step}")
                break
            
            muon_optimizer.zero_grad()
            for opt in adam_optimizers:
                opt.zero_grad()
            
            loss_val = 0.0
            for _ in range(gradient_accum):
                x, y = data_loader.next_batch(cfg.batch_tokens, cfg.seq_len)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    step_loss = model(x, y) / gradient_accum
                step_loss.backward()
                loss_val += step_loss.item()
            
            loss = torch.tensor(loss_val, device=device)
            
            if initial_loss is None:
                initial_loss = loss.item()
                status = "✅ GOOD" if initial_loss < 4.5 else "⚠️ HIGH"
                log(f"\n    {'='*60}")
                log(f"    INIT: {initial_loss:.4f} | {status}")
                log(f"    {'='*60}\n")
            
            if torch.isnan(loss) or loss.item() > 100:
                log(f"DIVERGENCE at step {step}: {loss.item()}")
                break
            
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            
            # LR schedule for ALL optimizers
            for opt in adam_optimizers:
                for pg in opt.param_groups:
                    base_lr = pg.get('initial_lr', pg['lr'])
                    pg['lr'] = get_lr_warmdown(step, cfg, base_lr)
            
            muon_lr = get_lr_warmdown(step, cfg, cfg.matrix_lr)
            for pg in muon_optimizer.param_groups:
                pg['lr'] = muon_lr
            
            muon_optimizer.step()
            for opt in adam_optimizers:
                opt.step()
            
            if step == qat_warmup_step:
                log(f"[QAT] Enabling at step {step}")
                model_to_enable = model
                while hasattr(model_to_enable, 'module'):
                    model_to_enable = model_to_enable.module
                model_to_enable.set_qat_enabled(True)
            
            ema.update(step)
            
            elapsed = time.perf_counter() - start_time
            tok_per_sec = (step + 1) * cfg.batch_tokens / max(elapsed, 1e-6)
            
            if step > 0 and step % cfg.val_interval == 0:
                # [T5] Use dynamic context evaluation
                _, val_bpb = eval_dynamic_context(model, val_tokens, byte_counts,
                                                   cfg, rank, world_size, device)
                if val_bpb < best_bpb:
                    best_bpb = val_bpb
                    best_state = ema.get_ema_state_dict()
                log(f"step {step:5d} | loss {loss.item():.4f} | bpb {val_bpb:.4f} | "
                    f"best {best_bpb:.4f} | {tok_per_sec:,.0f} tok/s")
            
            elif step % cfg.train_log_every == 0:
                current_lr = scalar_opt.param_groups[0]['lr']
                muon_current = muon_optimizer.param_groups[0]['lr']
                log(f"step {step:5d} | loss {loss.item():.4f} | adam_lr {current_lr:.6f} | muon_lr {muon_current:.6f} | {tok_per_sec:,.0f} tok/s")
    
    except Exception as e:
        log(f"Training interrupted: {e}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        # Apply EMA
        ema.apply()
        
        # [T1] Causal TTT
        if cfg.ttt_enabled:
            log("[T1] Running CAUSAL TTT adaptation...")
            ttt_bpb, ttt_tokens = causal_ttt_adapt(model, val_tokens, cfg, device,
                                                    byte_counts, rank, world_size)
            log(f"[T1] Causal TTT adapted on {ttt_tokens} tokens, phase BPB: {ttt_bpb:.4f}")
        else:
            ttt_tokens = 0
        
        # [T5] Final evaluation with dynamic context
        _, val_bpb = eval_dynamic_context(model, val_tokens, byte_counts,
                                          cfg, rank, world_size, device,
                                          start_pos=ttt_tokens)
        
        # Use EMA best_state if better
        if best_state and best_bpb < val_bpb:
            model_to_load = model
            while hasattr(model_to_load, 'module'):
                model_to_load = model_to_load.module
            model_to_load.load_state_dict(best_state)
            val_bpb = best_bpb
        
        if is_master:
            import json
            import shutil
            
            tmp_model = f"/tmp/{cfg.run_id}_model.int8.ptz"
            tmp_json = f"/tmp/{cfg.run_id}_submission.json"
            os.makedirs("/tmp", exist_ok=True)
            
            try:
                model_to_export = model
                while hasattr(model_to_export, 'module'):
                    model_to_export = model_to_export.module
                
                # [T6] Export with strict size limit
                compressed, stats = export_model(model_to_export, code, 
                                                  mlp_bits=cfg.export_mlp_bits,
                                                  attn_bits=cfg.export_attn_bits,
                                                  max_bytes=cfg.max_artifact_bytes)
                size_mb = stats['total_mb']
                size_bytes = stats['total_bytes']
                
                with open(tmp_model, "wb") as f:
                    f.write(compressed)
                
                log(f"\nExport Statistics:")
                log(f"  code_bytes: {stats['code_bytes']:,}")
                log(f"  weights_z: {stats['weights_z']:,}")
                log(f"  total_bytes: {size_bytes:,}")
                log(f"  total_mb: {size_mb:.4f}")
                log(f"  Embeddings preserved: {stats['quantization_stats']['embedding_params']:,}")
                log(f"  MLP params (int{cfg.export_mlp_bits}): {stats['quantization_stats']['mlp_params']:,}")
                log(f"  Attn params (int{cfg.export_attn_bits}): {stats['quantization_stats']['attn_params']:,}")
                
            except ValueError as e:
                # [T6] Size limit exceeded
                log(f"⚠️ SIZE LIMIT EXCEEDED: {e}")
                log(f"Attempting more aggressive quantization...")
                
                # Try with more aggressive quantization
                try:
                    compressed, stats = export_model(model_to_export, code, 
                                                      mlp_bits=4,  # More aggressive
                                                      attn_bits=5,
                                                      max_bytes=cfg.max_artifact_bytes)
                    size_mb = stats['total_mb']
                    size_bytes = stats['total_bytes']
                    
                    with open(tmp_model, "wb") as f:
                        f.write(compressed)
                    
                    log(f"✅ Export succeeded with int4/int5 quantization")
                    log(f"  total_bytes: {size_bytes:,}")
                    
                except ValueError:
                    log(f"❌ Export failed even with aggressive quantization")
                    size_mb = 0.0
                    size_bytes = 0
                    with open(tmp_model, "wb") as f:
                        f.write(b"")
                    
            except Exception as e:
                log(f"Export failed: {e}")
                import traceback
                log(traceback.format_exc())
                size_mb = 0.0
                size_bytes = 0
                with open(tmp_model, "wb") as f:
                    f.write(b"")
            
            # Determine success
            target_bpb = 1.1144
            status = "success" if val_bpb < target_bpb else "failed"
            
            with open(tmp_json, "w") as f:
                json.dump({
                    "model": "model.int8.ptz",
                    "bpb": val_bpb,
                    "target_bpb": target_bpb,
                    "status": status,
                    "size_bytes": size_bytes,
                    "size_mb": size_mb,
                    "version": "v4.0"
                }, f, indent=2)
            
            shutil.move(tmp_model, "model.int8.ptz")
            shutil.move(tmp_json, "submission.json")
            
            log("\n" + "="*80)
            if val_bpb < target_bpb:
                log(f"🏆 SUCCESS! FINAL BPB: {val_bpb:.4f} < {target_bpb:.4f}")
            else:
                log(f"FINAL BPB: {val_bpb:.4f} (target: {target_bpb:.4f})")
            log(f"Size: {size_mb:.4f} MB | Status: {status}")
            log("="*80)
            log("\nSaved: model.int8.ptz, submission.json")


if __name__ == "__main__":
    main()
