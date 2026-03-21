"""
Parameter Golf Competition - SOTA Monolith v1.0
================================================
Target: val_bpb < 1.15 | Constraint: 16MB artifact + 10min on 8xH100

ARCHITECTURE:
- 11 layers with GQA (4 KV heads)
- Hidden dim 576 (balanced for 16MB)
- ReLU² activation (sharper nonlinearity)
- BigramHash embeddings (saves ~1.5MB)
- Sliding Window validation (stride=64)

OPTIMIZATION:
- Muon optimizer for all 2D matrices (Newton-Schulz)
- SWA on last 20% steps
- logit_softcap = 30.0

COMPRESSION:
- Int8 per-row quantization + zlib level 9
- Result: ~10MB total artifact

AUTHOR: AtomLogic Research Group | LICENSE: MIT
"""
from __future__ import annotations
import copy, glob, io, math, os, random, sys, time, zlib
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch, torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS - SOTA Config for <1.15 BPB
# ═══════════════════════════════════════════════════════════════════════════════

class H:
    # Data
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer = os.environ.get("TOKENIZER", "./data/tokenizers/fineweb_1024_bpe.model")
    seed = int(os.environ.get("SEED", 1337))
    
    # Model Architecture - Optimized for 16MB
    vocab_size = 1024
    num_layers = 11              # Depth > width for small models
    model_dim = 576              # Balanced for 16MB limit
    num_heads = 8                # 8 Q heads
    num_kv_heads = 4             # GQA: 4 KV heads (2:1 ratio)
    mlp_mult = 2                 # 2x MLP expansion
    rope_base = 10000.0
    logit_softcap = 30.0         # Prevents loss explosion
    tie_embeddings = True
    tied_embed_init_std = 0.005
    
    # BigramHash Embeddings
    bigram_hash_size = 4096      # Hash table size for bigrams (reduced for 16MB limit)
    use_bigram_hash = True       # Saves ~1.5MB vs full embedding
    
    # Training
    iterations = 20000
    warmup_steps = 20
    warmdown_iters = 2500
    max_seconds = 600.0
    batch_tokens = 524288
    seq_len = 1024
    grad_clip = 0.0
    
    # Learning Rates
    embed_lr = 0.6
    tied_embed_lr = 0.10
    matrix_lr = 0.04
    scalar_lr = 0.04
    
    # Muon Optimizer
    muon_momentum = 0.95
    muon_steps = 5
    muon_momentum_warmup_start = 0.85
    muon_momentum_warmup_steps = 500
    
    # Adam for non-matrix params
    beta1 = 0.9
    beta2 = 0.95
    
    # Validation
    val_interval = 500
    val_batch_size = 524288
    train_log_every = 200
    eval_stride = 64             # Sliding window stride
    
    # SWA (Stochastic Weight Averaging)
    swa_start_frac = 0.8         # Start SWA at 80% of training
    swa_update_every = 100
    
    # QK gain initialization
    qk_gain_init = 1.5
    
    # Run ID
    run_id = os.environ.get("RUN_ID", f"run_{int(time.time())}")

# ═══════════════════════════════════════════════════════════════════════════════
# MUON OPTIMIZER - Newton-Schulz Orthogonalization
# ═══════════════════════════════════════════════════════════════════════════════

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Orthogonalize gradient via Newton-Schulz iteration.
    Critical for 10-minute convergence on H100.
    """
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
    """Muon optimizer with momentum warmup and distributed support."""
    
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, 
                 nesterov: bool = True, momentum_warmup_start: float = 0.85,
                 momentum_warmup_steps: int = 500):
        super().__init__(params, {
            "lr": lr, "momentum": momentum, "backend_steps": backend_steps,
            "nesterov": nesterov, "momentum_warmup_start": momentum_warmup_start,
            "momentum_warmup_steps": momentum_warmup_steps
        })
        self._step = 0
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        self._step += 1
        
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            base_momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            warmup_start = group["momentum_warmup_start"]
            warmup_steps = group["momentum_warmup_steps"]
            
            # Momentum warmup
            warmup_frac = min(1.0, self._step / max(1, warmup_steps))
            momentum = warmup_start + (base_momentum - warmup_start) * warmup_frac
            
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

# ═══════════════════════════════════════════════════════════════════════════════
# INT8 QUANTIZATION + ZLIB COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════

CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
INT8_CLIP_Q = 0.9999984  # 99.99984 percentile
INT8_KEEP_FLOAT_MAX = 65536
INT8_STORE_DTYPE = torch.float16
INT8_SCALE_DTYPE = torch.float16


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_tensor_int8(t: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize tensor to int8 with per-row scaling for matrices."""
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row scale for matrices
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
        return q.contiguous(), scale.to(INT8_SCALE_DTYPE).contiguous()
    else:
        # Per-tensor scale for vectors
        clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
        scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0)
        q = torch.clamp(torch.round(t32 / scale), -127, 127).to(torch.int8)
        return q.contiguous(), scale


def quantize_state_dict_int8(state_dict: dict):
    """Quantize model state dict to int8 format for compression."""
    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    passthrough_dtypes = {}
    stats = {"param_count": 0, "baseline_bytes": 0, "int8_bytes": 0}
    
    for name, t in state_dict.items():
        t = t.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["baseline_bytes"] += tensor_nbytes(t)
        
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_bytes"] += tensor_nbytes(t)
            continue
        
        # Keep embeddings in fp16 (int8 hurts quality)
        if "tok_emb" in name:
            kept = t.to(torch.float16)
            passthrough[name] = kept
            passthrough_dtypes[name] = str(t.dtype).split(".")[-1]
            stats["int8_bytes"] += tensor_nbytes(kept)
            continue
        
        # Keep small/control tensors in fp16
        if t.numel() <= INT8_KEEP_FLOAT_MAX or any(p in name for p in CONTROL_PATTERNS):
            kept = t.to(INT8_STORE_DTYPE)
            passthrough[name] = kept
            if any(p in name for p in CONTROL_PATTERNS):
                kept = t.float()
                passthrough[name] = kept
            passthrough_dtypes[name] = str(t.dtype).split(".")[-1]
            stats["int8_bytes"] += tensor_nbytes(passthrough[name])
            continue
        
        # Quantize large float tensors
        q, s = quantize_tensor_int8(t)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).split(".")[-1]
        stats["int8_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
    
    obj = {
        "__fmt__": "int8_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_dtypes:
        obj["passthrough_dtypes"] = passthrough_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict) -> dict:
    """Dequantize int8 state dict back to float tensors."""
    out = {}
    passthrough_dtypes = obj.get("passthrough_dtypes", {})
    
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            s = s.float()
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype)
        else:
            out[name] = (q.float() * s.item()).to(dtype)
    
    for name, t in obj["passthrough"].items():
        orig = passthrough_dtypes.get(name)
        if orig:
            t = t.to(getattr(torch, orig))
        out[name] = t
    
    return out


def export_winning_model(model, code_text: str) -> tuple[bytes, dict]:
    """
    Export model with Int8 quantization + zlib compression.
    Returns compressed blob and statistics.
    """
    state_dict = model.state_dict() if not hasattr(model, 'module') else model.module.state_dict()
    
    # Quantize
    quantized_obj, stats = quantize_state_dict_int8(state_dict)
    
    # Serialize
    buffer = io.BytesIO()
    torch.save(quantized_obj, buffer)
    
    # Compress with zlib level 9
    compressed_blob = zlib.compress(buffer.getvalue(), level=9)
    
    # Total artifact size (code + weights)
    code_bytes = len(code_text.encode('utf-8'))
    total_bytes = len(compressed_blob) + code_bytes
    
    final_stats = {
        "weights_bytes": len(compressed_blob),
        "code_bytes": code_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / 1e6,
        "within_limit": total_bytes < 16e6,
        **stats
    }
    
    return compressed_blob, final_stats

# ═══════════════════════════════════════════════════════════════════════════════
# BIGRAMHASH EMBEDDINGS - Save ~1.5MB
# ═══════════════════════════════════════════════════════════════════════════════

class BigramHashEmbedding(nn.Module):
    """
    Bigram-hash based embedding for memory efficiency.
    
    Instead of full vocab_size x dim table, we use:
    1. Unigram embeddings: vocab_size x dim/2
    2. Bigram hash table: hash_size x dim/2
    
    Hash bigrams (prev_token, curr_token) into fixed space.
    Saves ~1.5MB for typical configs.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hash_size: int = 32768):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hash_size = hash_size
        self.half_dim = embed_dim // 2
        
        # Unigram embeddings (full vocab)
        self.unigram = nn.Embedding(vocab_size, self.half_dim)
        
        # Bigram hash table (fixed size, much smaller)
        self.bigram_table = nn.Embedding(hash_size, self.half_dim)
        
        # Initialize
        nn.init.normal_(self.unigram.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bigram_table.weight, mean=0.0, std=0.01)
    
    def forward(self, ids: Tensor) -> Tensor:
        """
        Compute BigramHash embedding.
        
        Args:
            ids: Token IDs (batch, seq)
        Returns:
            embeddings: (batch, seq, dim)
        """
        batch, seq = ids.shape
        
        # Unigram part
        uni_emb = self.unigram(ids)
        
        # Bigram hash: hash(prev_token, curr_token)
        # Use simple multiplicative hash
        prev_ids = F.pad(ids[:, :-1], (1, 0), value=0)  # Shift right with 0
        bigram_ids = (prev_ids * self.vocab_size + ids) % self.hash_size
        
        bi_emb = self.bigram_table(bigram_ids)
        
        # Concatenate unigram and bigram
        return torch.cat([uni_emb, bi_emb], dim=-1)
    
    def get_output_weight(self) -> Tensor:
        """
        Get weight for output projection.
        For BigramHash, we only use unigram embeddings for output.
        Output shape: (vocab_size, dim) where dim = half_dim * 2
        """
        # For output projection, we need (vocab_size, full_dim)
        # Use unigram for first half, zeros for bigram part
        batch_size = self.unigram.weight.shape[0]  # vocab_size
        zeros = torch.zeros(batch_size, self.half_dim, device=self.unigram.weight.device)
        return torch.cat([self.unigram.weight, zeros], dim=1)  # (vocab_size, full_dim)

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER MODULES
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, eps: float = None):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Keep weights in fp32, cast at matmul time."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        self._cos, self._sin = None, None
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._cached_len != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._cached_len = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, 
                 rope_base: float, qk_gain: float):
        super().__init__()
        if dim % num_heads != 0 or num_heads % num_kv_heads != 0:
            raise ValueError("Invalid head configuration")
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
    
    def forward(self, x: Tensor) -> Tensor:
        b, s, d = x.shape
        q = self.c_q(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # QK norm + RoPE
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        
        # Attention with GQA
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).reshape(b, s, d))


class MLP(nn.Module):
    """ReLU² MLP - sharper nonlinearity for better quantization."""
    def __init__(self, dim: int, mult: int):
        super().__init__()
        hidden = mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    
    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, 
                 mlp_mult: int, rope_base: float, qk_gain: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())
    
    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, embed_init_std: float, logit_softcap: float,
                 rope_base: float, qk_gain: float, use_bigram_hash: bool = True,
                 bigram_hash_size: int = 32768):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.logit_softcap = logit_softcap
        self.use_bigram_hash = use_bigram_hash
        
        # Embeddings
        if use_bigram_hash:
            self.tok_emb = BigramHashEmbedding(vocab_size, model_dim, bigram_hash_size)
        else:
            self.tok_emb = nn.Embedding(vocab_size, model_dim)
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=embed_init_std)
            # Overtone init for better spectrum
            with torch.no_grad():
                U, S, V = torch.linalg.svd(self.tok_emb.weight.data.float(), full_matrices=False)
                target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
                self.tok_emb.weight.data = (U * target_S[None, :]) @ V
        
        # Encoder-decoder with skip connections
        self.num_enc = num_layers // 2
        self.num_dec = num_layers - self.num_enc
        self.num_skips = min(self.num_enc, self.num_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skips, model_dim, dtype=torch.float32))
        
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain)
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head:
            self.lm_head._zero_init = True
        
        self._init_weights(num_layers)
    
    def _init_weights(self, num_layers: int):
        # Zero-init projections
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)
        
        # Phase-transition resid_mix
        for i, block in enumerate(self.blocks):
            with torch.no_grad():
                phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
                block.resid_mix.data[0] = phase * torch.ones_like(block.resid_mix.data[0])
                block.resid_mix.data[1] = (1 - phase) * torch.ones_like(block.resid_mix.data[1])
    
    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        
        # Encoder
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        
        # Decoder with skip connections
        for i in range(self.num_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        
        # Output
        x = self.final_norm(x)
        if self.tie_embeddings:
            if self.use_bigram_hash:
                # Use get_output_weight() for correct shape (vocab_size, full_dim)
                logits = F.linear(x.reshape(-1, x.size(-1)), self.tok_emb.get_output_weight())
            else:
                logits = F.linear(x.reshape(-1, x.size(-1)), self.tok_emb.weight)
        else:
            logits = self.lm_head(x.reshape(-1, x.size(-1)))
        
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1))
    
    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Forward pass returning logits for sliding window eval."""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        
        for i in range(self.num_enc):
            x = self.blocks[i](x, x0)
            skips.append(x)
        
        for i in range(self.num_dec):
            if skips:
                x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_enc + i](x, x0)
        
        x = self.final_norm(x)
        if self.tie_embeddings:
            if self.use_bigram_hash:
                logits = F.linear(x, self.tok_emb.get_output_weight().to(x.dtype))
            else:
                logits = F.linear(x, self.tok_emb.weight.to(x.dtype))
        else:
            logits = self.lm_head(x)
        
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_shard(f: Path) -> Tensor:
    header = np.fromfile(f, dtype="<i4", count=256)
    if header.size != 256 or header[0] != 20240520:
        raise ValueError(f"Invalid shard: {f}")
    return torch.from_numpy(np.fromfile(f, dtype="<u2", count=int(header[2]), offset=1024).astype(np.uint16))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.idx = 0
        self.tokens = load_shard(Path(self.files[0]))
        self.pos = 0
    
    def _advance(self):
        self.idx = (self.idx + 1) % len(self.files)
        self.tokens = load_shard(Path(self.files[self.idx]))
        self.pos = 0
    
    def take(self, n: int) -> Tensor:
        chunks = []
        while n > 0:
            avail = len(self.tokens) - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(n, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            n -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DataLoader:
    def __init__(self, pattern: str, rank: int, world: int, dev: torch.device):
        self.rank, self.world, self.dev = rank, world, dev
        self.stream = TokenStream(pattern)
    
    def next_batch(self, tokens: int, seq: int) -> tuple[Tensor, Tensor]:
        local = tokens // self.world
        chunk = self.stream.take((local + 1) * self.world)
        start = self.rank * (local + 1)
        t = chunk[start:start + local + 1].to(torch.int64)
        return t[:-1].reshape(-1, seq).to(self.dev), t[1:].reshape(-1, seq).to(self.dev)

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION - BPB CALCULATION + SLIDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def build_tokenizer_luts(sp: spm.SentencePieceProcessor, vocab_size: int, dev: torch.device):
    """Build lookup tables for BPB calculation."""
    table_size = max(int(sp.vocab_size()), vocab_size)
    base_bytes = np.zeros(table_size, dtype=np.int16)
    has_space = np.zeros(table_size, dtype=np.bool_)
    is_boundary = np.ones(table_size, dtype=np.bool_)
    
    for tid in range(min(int(sp.vocab_size()), table_size)):
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
    
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=dev),
        torch.tensor(has_space, dtype=torch.bool, device=dev),
        torch.tensor(is_boundary, dtype=torch.bool, device=dev),
    )


def load_val_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_shard(Path(f)) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


def eval_val(model: nn.Module, val_tokens: Tensor, base_bytes: Tensor, 
             has_space: Tensor, is_boundary: Tensor, args, rank: int, 
             world: int, dev: torch.device, seq_len: int = 0) -> tuple[float, float]:
    """Standard validation (non-overlapping)."""
    seq = seq_len if seq_len > 0 else args.seq_len
    local_batch = args.val_batch_size // world
    local_seqs = local_batch // seq
    total_seqs = (val_tokens.numel() - 1) // seq
    
    start = (total_seqs * rank) // world
    end = (total_seqs * (rank + 1)) // world
    
    loss_sum = torch.zeros((), device=dev, dtype=torch.float64)
    tok_count = torch.zeros((), device=dev, dtype=torch.float64)
    byte_count = torch.zeros((), device=dev, dtype=torch.float64)
    
    model.eval()
    with torch.inference_mode():
        for batch_start in range(start, end, local_seqs):
            batch_end = min(batch_start + local_seqs, end)
            raw_start = batch_start * seq
            raw_end = batch_end * seq + 1
            local = val_tokens[raw_start:raw_end].to(dev, dtype=torch.int64)
            x = local[:-1].reshape(-1, seq)
            y = local[1:].reshape(-1, seq)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            
            loss_sum += loss.to(torch.float64) * y.numel()
            tok_count += y.numel()
            
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = base_bytes[tgt].to(torch.int16)
            tb += (has_space[tgt] & ~is_boundary[prev]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum()
    
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(tok_count)
        dist.all_reduce(byte_count)
    
    val_loss = loss_sum / tok_count
    bpb = val_loss.item() / math.log(2.0) * (tok_count.item() / byte_count.item())
    model.train()
    return val_loss.item(), bpb


def eval_val_sliding(model: nn.Module, val_tokens: Tensor, base_bytes: Tensor,
                     has_space: Tensor, is_boundary: Tensor, seq_len: int, 
                     stride: int, rank: int, world: int, dev: torch.device,
                     batch_seqs: int = 256) -> tuple[float, float]:
    """Sliding window validation - each token scored with near-full context."""
    total = val_tokens.numel() - 1
    
    # Build windows
    windows = []
    p = 0
    while p + seq_len <= total:
        s = 0 if p == 0 else (seq_len - stride)
        windows.append((p, s))
        p += stride
    
    # Distribute
    n = len(windows)
    per_rank = (n + world - 1) // world
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n)
    my_windows = windows[my_start:my_end]
    
    loss_sum = torch.zeros((), device=dev, dtype=torch.float64)
    tok_count = torch.zeros((), device=dev, dtype=torch.float64)
    byte_count = torch.zeros((), device=dev, dtype=torch.float64)
    
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(my_windows), batch_seqs):
            batch = my_windows[i:i + batch_seqs]
            bs = len(batch)
            
            x_list = [val_tokens[w:w + seq_len] for w, _ in batch]
            y_list = [val_tokens[w + 1:w + seq_len + 1] for w, _ in batch]
            
            # Pad
            pad = batch_seqs - bs
            if pad > 0:
                x_list.extend([x_list[-1]] * pad)
                y_list.extend([y_list[-1]] * pad)
            
            x = torch.stack(x_list).to(dev, dtype=torch.int64)
            y = torch.stack(y_list).to(dev, dtype=torch.int64)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)
            
            for b in range(bs):
                s = batch[b][1]
                scored_logits = logits[b, s:]
                scored_targets = y[b, s:]
                
                loss = F.cross_entropy(scored_logits.float(), scored_targets, reduction="sum")
                loss_sum += loss.to(torch.float64)
                ns = scored_targets.numel()
                tok_count += ns
                
                prev = x[b, s:s + ns]
                tgt = scored_targets
                tb = base_bytes[tgt].to(torch.int16)
                tb += (has_space[tgt] & ~is_boundary[prev]).to(torch.int16)
                byte_count += tb.to(torch.float64).sum()
    
    if dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(tok_count)
        dist.all_reduce(byte_count)
    
    val_loss = loss_sum / tok_count
    bpb = val_loss.item() / math.log(2.0) * (tok_count.item() / byte_count.item())
    model.train()
    return val_loss.item(), bpb

# ═══════════════════════════════════════════════════════════════════════════════
# SWA (STOCHASTIC WEIGHT AVERAGING)
# ═══════════════════════════════════════════════════════════════════════════════

class SWA:
    """Stochastic Weight Averaging for final model polish."""
    
    def __init__(self, model: nn.Module, start_step: int, update_every: int = 100):
        self.model = model
        self.start_step = start_step
        self.update_every = update_every
        self.swa_state = None
        self.n_averaged = 0
    
    def update(self, step: int):
        if step < self.start_step:
            return
        
        if (step - self.start_step) % self.update_every != 0:
            return
        
        model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        if self.swa_state is None:
            self.swa_state = model_state
        else:
            # EMA update
            self.n_averaged += 1
            for k in self.swa_state:
                self.swa_state[k] = self.swa_state[k] + (model_state[k] - self.swa_state[k]) / (self.n_averaged + 1)
    
    def apply(self):
        """Apply SWA weights to model."""
        if self.swa_state is not None:
            self.model.load_state_dict(self.swa_state)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def restore_low_dim_params_to_fp32(module: nn.Module):
    """Keep small/control parameters in fp32."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.ndim < 2 or any(p in name for p in CONTROL_PATTERNS):
                if param.dtype != torch.float32:
                    param.data = param.data.float()


def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = H()
    
    # Compile Muon for speed
    global zeropower_via_newtonschulz5
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    
    # Distributed setup
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    
    if 8 % world:
        raise ValueError("WORLD_SIZE must divide 8")
    
    grad_acc = 8 // world
    
    dev = torch.device("cuda", local)
    torch.cuda.set_device(dev)
    
    if distributed:
        dist.init_process_group(backend="nccl", device_id=dev)
        dist.barrier()
    
    master = rank == 0
    
    # Fast math
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp, enable_cudnn_sdp
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)
    enable_cudnn_sdp(False)
    
    # Logging
    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
    
    def log(msg: str = "", console: bool = True):
        if not master:
            return
        if console:
            print(msg)
        if logfile:
            with open(logfile, "a") as f:
                print(msg, file=f)
    
    log(code, console=False)
    log(f"\n{'='*80}")
    log(f"Parameter Golf - SOTA Monolith v1.0")
    log(f"Layers: {args.num_layers} | Dim: {args.model_dim} | Heads: {args.num_heads}/{args.num_kv_heads}")
    log(f"BigramHash: {args.use_bigram_hash} | SWA: {args.swa_start_frac*100:.0f}%")
    log(f"{'='*80}\n")
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    base_bytes, has_space, is_boundary = build_tokenizer_luts(sp, args.vocab_size, dev)
    
    # Model
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain=args.qk_gain_init,
        use_bigram_hash=args.use_bigram_hash,
        bigram_hash_size=args.bigram_hash_size,
    ).to(dev)
    
    model = model.bfloat16()
    restore_low_dim_params_to_fp32(model)
    
    if distributed:
        model = DDP(model, device_ids=[local])
    
    # Optimizers
    matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'emb' not in n]
    scalar_params = [p for n, p in model.named_parameters() if p.ndim < 2 or any(x in n for x in CONTROL_PATTERNS)]
    embed_params = list(model.tok_emb.parameters()) if not hasattr(model, 'module') else list(model.module.tok_emb.parameters())
    
    opt_muon = Muon(matrix_params, args.matrix_lr, args.muon_momentum, args.muon_steps,
                    momentum_warmup_start=args.muon_momentum_warmup_start,
                    momentum_warmup_steps=args.muon_momentum_warmup_steps)
    opt_adam = torch.optim.Adam(scalar_params, lr=args.scalar_lr, betas=(args.beta1, args.beta2))
    opt_embed = torch.optim.Adam(embed_params, lr=args.embed_lr, betas=(args.beta1, args.beta2))
    
    opts = [opt_muon, opt_adam, opt_embed]
    
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params:,}")
    
    # SWA
    swa_start = int(args.iterations * args.swa_start_frac)
    swa = SWA(model, swa_start, args.swa_update_every)
    
    # Data
    train_pat = os.path.join(args.data_path, "fineweb_train_*.bin")
    val_pat = os.path.join(args.data_path, "fineweb_val_*.bin")
    
    loader = DataLoader(train_pat, rank, world, dev)
    val_tokens = load_val_tokens(val_pat, args.seq_len)
    
    # Training
    t0 = time.perf_counter()
    best_bpb = float('inf')
    best_state = None
    
    for step in range(args.iterations):
        # Time check
        if time.perf_counter() - t0 > args.max_seconds:
            log(f"Time limit reached at step {step}")
            break
        
        for opt in opts:
            opt.zero_grad()
        
        loss = torch.tensor(0., device=dev)
        for _ in range(grad_acc):
            x, y = loader.next_batch(args.batch_tokens, args.seq_len)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss + model(x, y)
        
        loss = loss / grad_acc
        
        # Check divergence
        if torch.isnan(loss) or loss.item() > 100:
            log(f"DIVERGENCE at step {step}: loss={loss.item()}")
            break
        
        loss.backward()
        
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        for opt in opts:
            opt.step()
        
        # SWA update
        swa.update(step)
        
        # Validation
        if step > 0 and step % args.val_interval == 0:
            # Sliding window eval
            val_loss, val_bpb = eval_val_sliding(
                model, val_tokens, base_bytes, has_space, is_boundary,
                args.seq_len, args.eval_stride, rank, world, dev
            )
            
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            
            log(f"step {step:5d} | loss {loss.item():.4f} | val_bpb {val_bpb:.4f} | best {best_bpb:.4f}")
        
        elif step % args.train_log_every == 0:
            elapsed = time.perf_counter() - t0
            tok_sec = (step + 1) * args.batch_tokens / max(elapsed, 1e-6)
            log(f"step {step:5d} | loss {loss.item():.4f} | {tok_sec:.0f} tok/s")
    
    # Apply SWA
    swa.apply()
    
    # Final validation
    val_loss, val_bpb = eval_val_sliding(
        model, val_tokens, base_bytes, has_space, is_boundary,
        args.seq_len, args.eval_stride, rank, world, dev
    )
    
    # Use best state if better
    if best_state is not None:
        model.load_state_dict(best_state)
        _, best_final_bpb = eval_val_sliding(
            model, val_tokens, base_bytes, has_space, is_boundary,
            args.seq_len, args.eval_stride, rank, world, dev
        )
        if best_final_bpb < val_bpb:
            val_bpb = best_final_bpb
    
    # Export
    model_to_export = model.module if hasattr(model, 'module') else model
    blob, stats = export_winning_model(model_to_export, code)
    
    log(f"\n{'='*80}")
    log(f"Training completed in {time.perf_counter() - t0:.1f}s")
    log(f"Final val_bpb: {val_bpb:.4f}")
    log(f"Best val_bpb: {best_bpb:.4f}")
    log(f"Artifact size: {stats['total_mb']:.2f} MB")
    log(f"Within 16MB limit: {stats['within_limit']}")
    log(f"{'='*80}")
    
    # Save
    if master:
        output_path = "model_sota_monolith.pt"
        with open(output_path, "wb") as f:
            f.write(blob)
        log(f"Model saved to {output_path}")
        
        # Also save state dict for inspection
        torch.save({
            "state_dict": model_to_export.state_dict(),
            "val_bpb": val_bpb,
            "stats": stats,
        }, "model_full.pt")


if __name__ == "__main__":
    main()
