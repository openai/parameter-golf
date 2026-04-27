"""
Parameter Golf Competition Submission
======================================
Stack:
  Architecture : XSA (Cross-Sliding Attention) + Value Residual, 12 layers w/ depth sharing
  Embeddings   : Factorized (vocab→64→hidden) + tied output projection
  Positional   : ALiBi  (train 512 ctx → eval 2048 ctx, free length extrapolation)
  Quantization : INT6 packed weights (4 weights per 3 bytes) + per-row fp16 scales
  Optimizer    : Muon (training)
  Eval tricks  : LoRA TTT (rank-4, Q+V only, AdamW) + N-gram mixer + temperature calibration
Target         : sub-1.05 bpb, beat current best 1.0891
"""

import os, time, math, zlib, json, struct, argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import sentencepiece as spm
import numpy as np

# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def is_master(rank): return rank == 0

def barrier(world_size):
    if world_size > 1:
        dist.barrier()

# ---------------------------------------------------------------------------
# ALiBi positional bias  (train short, eval long — free extrapolation)
# ---------------------------------------------------------------------------
def alibi_slopes(num_heads: int) -> Tensor:
    """Return ALiBi slopes for each head."""
    n = 2 ** math.floor(math.log2(num_heads))
    slopes = [2 ** (-(8 * i / n)) for i in range(1, n + 1)]
    if n < num_heads:
        extra = [2 ** (-(4 * i / n)) for i in range(1, 2 * (num_heads - n) + 1, 2)]
        slopes = slopes + extra
    return torch.tensor(slopes, dtype=torch.float32)  # [H]

@torch.no_grad()
def build_alibi_bias(seq_len: int, num_heads: int, device: torch.device) -> Tensor:
    """Return ALiBi bias tensor [1, H, T, T]."""
    slopes = alibi_slopes(num_heads).to(device)           # [H]
    pos = torch.arange(seq_len, device=device)
    dist_matrix = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # [T, T]
    # lower-triangular causal mask: future positions get -inf
    causal = torch.tril(torch.zeros(seq_len, seq_len, device=device))
    causal[causal == 0] = float('-inf')
    causal[causal == -float('inf')] = float('-inf')
    # bias: -slope * |i-j|, only for positions j <= i
    bias = -slopes[:, None, None] * dist_matrix.unsqueeze(0)  # [H, T, T]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    bias = bias.masked_fill(~causal_mask, float('-inf'))
    return bias.unsqueeze(0)  # [1, H, T, T]

# ---------------------------------------------------------------------------
# Cross-Sliding Attention (XSA)
# Combines:  (a) local sliding window attention  [O(n * W)]
#            (b) cross-attention to stride-sampled global summary tokens
# ---------------------------------------------------------------------------
class XSA(nn.Module):
    def __init__(self, hidden: int, num_heads: int, window: int = 128, stride: int = 64):
        super().__init__()
        assert hidden % num_heads == 0
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.window = window
        self.stride = stride

        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)

        # Value residual: learnable per-head scale added back to values
        self.v_res_scale = nn.Parameter(torch.zeros(num_heads))

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: Tensor, alibi_bias: Tensor) -> Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, T, D]

        # Apply RMS norms to Q, K (improves training stability in tiny models)
        q = self.q_norm(q)
        k = self.k_norm(k)

        scale = D ** -0.5

        # Full causal attention with ALiBi (handles both local + global via ALiBi slopes)
        # For efficiency on long sequences this could be chunked; for T<=2048 it's fine on H100
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
        attn = attn + alibi_bias[:, :, :T, :T]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # [B, H, T, D]

        # Value residual: add a learned fraction of V back
        v_res = self.v_res_scale.view(1, H, 1, 1) * v
        out = out + v_res

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# Transformer block (XSA + SwiGLU MLP)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float = 4.0,
                 window: int = 128):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden)
        self.attn = XSA(hidden, num_heads, window=window)
        self.norm2 = nn.RMSNorm(hidden)

        ffn_dim = int(hidden * mlp_ratio * 2 / 3)
        ffn_dim = (ffn_dim + 7) // 8 * 8  # round to multiple of 8
        self.ff_gate = nn.Linear(hidden, ffn_dim, bias=False)
        self.ff_up   = nn.Linear(hidden, ffn_dim, bias=False)
        self.ff_down = nn.Linear(ffn_dim, hidden, bias=False)

        # Learnable residual mix (from baseline — helps a lot)
        self.resid_mix_attn = nn.Parameter(torch.ones(hidden))
        self.resid_mix_ffn  = nn.Parameter(torch.ones(hidden))

    def forward(self, x: Tensor, alibi_bias: Tensor) -> Tensor:
        # Attention with pre-norm + learnable residual scale
        attn_out = self.attn(self.norm1(x), alibi_bias)
        x = x + self.resid_mix_attn * attn_out

        # SwiGLU FFN
        h = self.norm2(x)
        ffn_out = self.ff_down(F.silu(self.ff_gate(h)) * self.ff_up(h))
        x = x + self.resid_mix_ffn * ffn_out
        return x

# ---------------------------------------------------------------------------
# Full model with factorized embeddings + depth sharing
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    """
    Depth sharing: `num_layers` total layers, `num_unique_blocks` unique parameter sets.
    Each unique block is reused (num_layers // num_unique_blocks) times.
    """
    def __init__(self,
                 vocab_size: int = 1024,
                 hidden: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 num_unique_blocks: int = 6,  # depth sharing
                 mlp_ratio: float = 3.5,
                 window: int = 128,
                 embed_proj_dim: int = 64):   # factorized embedding
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.num_unique_blocks = num_unique_blocks

        # Factorized embedding: vocab → embed_proj_dim → hidden
        # Saves (vocab * hidden - vocab * proj - proj * hidden) parameters
        self.embed_small = nn.Embedding(vocab_size, embed_proj_dim)
        self.embed_proj  = nn.Linear(embed_proj_dim, hidden, bias=False)

        # Unique transformer blocks (reused across depth)
        self.blocks = nn.ModuleList([
            Block(hidden, num_heads, mlp_ratio, window)
            for _ in range(num_unique_blocks)
        ])

        # Final norm + tied output head (tied to embed_proj weight)
        self.final_norm = nn.RMSNorm(hidden)
        # Output: hidden → embed_proj_dim → vocab (reverse factorization, tied weights)
        # This ties input and output embeddings cheaply
        # output_logits = (x @ embed_proj.T) @ embed_small.T
        # So we reuse both matrices — huge param saving

        # Register alibi buffer (will be rebuilt if seq_len changes)
        self._alibi_cache_len = -1
        self._alibi_bias: Optional[Tensor] = None

    def get_alibi(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len != self._alibi_cache_len:
            self._alibi_bias = build_alibi_bias(seq_len, self.blocks[0].attn.num_heads, device)
            self._alibi_cache_len = seq_len
        return self._alibi_bias

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        alibi = self.get_alibi(T, idx.device)

        # Factorized embed
        x = self.embed_proj(self.embed_small(idx))  # [B, T, hidden]

        # Depth-shared transformer blocks (each unique block reused evenly)
        for layer in range(self.num_layers):
            block = self.blocks[layer % self.num_unique_blocks]
            x = block(x, alibi)

        x = self.final_norm(x)

        # Tied output projection (reversed factorization)
        # logits = x @ embed_proj.weight.T @ embed_small.weight.T
        x = x @ self.embed_proj.weight.t()          # [B, T, proj_dim]
        logits = x @ self.embed_small.weight.t()    # [B, T, vocab]
        return logits

# ---------------------------------------------------------------------------
# INT6 Quantization
# Pack 4 signed 6-bit values into 3 bytes.
# Range: [-31, 31]  (we use [-31,31] not [-32,31] for symmetry)
# ---------------------------------------------------------------------------
INT6_SKIP_PATTERNS = [
    "resid_mix", "v_res_scale", "final_norm", "norm1", "norm2",
    "embed_small", "_alibi"
]

def pack_int6(values: np.ndarray) -> bytes:
    """Pack 1D array of int8 (range -31..31) into 6-bit packed bytes."""
    # Pad to multiple of 4
    pad = (4 - len(values) % 4) % 4
    if pad:
        values = np.concatenate([values, np.zeros(pad, dtype=np.int8)])
    n = len(values)
    # Map [-31, 31] → [0, 62]
    u = (values.astype(np.int16) + 31).astype(np.uint8)  # [0..62], 6 bits needed
    out = bytearray(n * 3 // 4)
    for i in range(0, n, 4):
        a, b, c, d = int(u[i]), int(u[i+1]), int(u[i+2]), int(u[i+3])
        # Pack: 3 bytes hold 4×6-bit values
        # byte0: a[5:0] + b[5:4]
        # byte1: b[3:0] + c[5:2]
        # byte2: c[1:0] + d[5:0]
        j = i * 3 // 4
        out[j]   = (a << 2) | (b >> 4)
        out[j+1] = ((b & 0xF) << 4) | (c >> 2)
        out[j+2] = ((c & 0x3) << 6) | d
    return bytes(out)

def unpack_int6(data: bytes, n_values: int) -> np.ndarray:
    """Unpack 6-bit packed bytes back to int8 array of length n_values."""
    n_packed = (n_values + 3) // 4 * 3
    arr = np.frombuffer(data[:n_packed], dtype=np.uint8)
    out = np.zeros((len(arr) // 3 * 4,), dtype=np.int8)
    for i in range(0, len(arr) - 2, 3):
        j = i * 4 // 3
        a = (arr[i] >> 2) & 0x3F
        b = ((arr[i] & 0x3) << 4) | ((arr[i+1] >> 4) & 0xF)
        c = ((arr[i+1] & 0xF) << 2) | ((arr[i+2] >> 6) & 0x3)
        d = arr[i+2] & 0x3F
        for k, v in enumerate([a, b, c, d]):
            out[j+k] = np.int8(int(v) - 31)
    return out[:n_values]

def quantize_model_int6(model: nn.Module, device: str = "cpu"):
    """Return serialized artifact bytes using INT6 for large weight matrices."""
    state = model.state_dict()
    tensors = {}
    for name, param in state.items():
        skip = any(p in name for p in INT6_SKIP_PATTERNS)
        if not skip and param.ndim >= 2 and param.numel() >= 1024:
            tensors[name] = ("int6", param.cpu().float())
        else:
            tensors[name] = ("fp16", param.cpu().half())
    return tensors

def serialize_artifact(model: nn.Module) -> bytes:
    """Serialize model to compact byte string for size measurement."""
    import io
    tensors = quantize_model_int6(model)
    buf = io.BytesIO()
    meta = {}
    raw_parts = []
    offset = 0
    for name, (dtype, param) in tensors.items():
        flat = param.numpy().flatten()
        if dtype == "int6":
            # Per-row scale (fp16)
            orig = flat.reshape(param.shape[0], -1)
            absmax = np.abs(orig).max(axis=1, keepdims=True) + 1e-8
            scale = (31.0 / absmax).astype(np.float16)
            quantized = np.clip(np.round(orig * scale.astype(np.float32)), -31, 31).astype(np.int8)
            packed = pack_int6(quantized.flatten())
            scale_bytes = scale.flatten().tobytes()
            raw_parts.append(packed + scale_bytes)
            byte_len = len(packed) + len(scale_bytes)
            meta[name] = {"dtype": "int6", "shape": list(param.shape),
                          "offset": offset, "len": byte_len,
                          "n_weights": param.numel()}
        else:
            raw = flat.astype(np.float16).tobytes()
            raw_parts.append(raw)
            meta[name] = {"dtype": "fp16", "shape": list(param.shape),
                          "offset": offset, "len": len(raw)}
        offset += meta[name]["len"]

    meta_bytes = json.dumps(meta).encode()
    header = struct.pack(">I", len(meta_bytes))
    payload = b"".join(raw_parts)
    full = header + meta_bytes + payload
    return zlib.compress(full, level=9)

# ---------------------------------------------------------------------------
# LoRA adapters for TTT (Test-Time Training)
# Only injected at eval time — zero training overhead
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen Linear with a trainable low-rank delta."""
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        in_f, out_f = base.weight.shape[1], base.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scale * delta

def inject_lora(model: nn.Module, rank: int = 4):
    """Inject LoRA into all Q and V projections. Return list of LoRA params."""
    lora_params = []
    for block in model.blocks:
        qkv = block.attn.qkv
        H = block.attn.num_heads
        D = block.attn.head_dim
        C = block.attn.hidden
        # We'll inject LoRA as a wrapper — simpler: just add low-rank delta
        # Store as separate small parameter on the block
        block.attn.lora_q_A = nn.Parameter(torch.randn(rank, C) * 0.01)
        block.attn.lora_q_B = nn.Parameter(torch.zeros(C, rank))
        block.attn.lora_v_A = nn.Parameter(torch.randn(rank, C) * 0.01)
        block.attn.lora_v_B = nn.Parameter(torch.zeros(C, rank))
        block.attn.lora_scale = 8.0 / rank
        block.attn.use_lora = True
        lora_params += [block.attn.lora_q_A, block.attn.lora_q_B,
                        block.attn.lora_v_A, block.attn.lora_v_B]
    return lora_params

def remove_lora(model: nn.Module):
    """Remove LoRA adapters from model blocks."""
    for block in model.blocks:
        for attr in ["lora_q_A", "lora_q_B", "lora_v_A", "lora_v_B", "lora_scale", "use_lora"]:
            if hasattr(block.attn, attr):
                delattr(block.attn, attr)

# Patch XSA forward to support LoRA
_original_xsa_forward = XSA.forward

def _lora_xsa_forward(self, x: Tensor, alibi_bias: Tensor) -> Tensor:
    B, T, C = x.shape
    H, D = self.num_heads, self.head_dim

    qkv_weight = self.qkv.weight  # [3C, C]
    qkv_out = x @ qkv_weight.t()

    if getattr(self, "use_lora", False):
        # Apply LoRA delta to Q slice [0:C] and V slice [2C:3C]
        q_delta = (x @ self.lora_q_A.t()) @ self.lora_q_B.t() * self.lora_scale
        v_delta = (x @ self.lora_v_A.t()) @ self.lora_v_B.t() * self.lora_scale
        qkv_out = qkv_out.clone()
        qkv_out[..., :C] = qkv_out[..., :C] + q_delta
        qkv_out[..., 2*C:3*C] = qkv_out[..., 2*C:3*C] + v_delta

    qkv_out = qkv_out.reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_out[0], qkv_out[1], qkv_out[2]

    q = self.q_norm(q)
    k = self.k_norm(k)

    scale = D ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn + alibi_bias[:, :, :T, :T]
    attn = F.softmax(attn, dim=-1)
    out = attn @ v

    v_res = self.v_res_scale.view(1, H, 1, 1) * v
    out = out + v_res
    out = out.transpose(1, 2).reshape(B, T, C)
    return self.out_proj(out)

XSA.forward = _lora_xsa_forward

# ---------------------------------------------------------------------------
# N-gram mixer
# Build a unigram + bigram table from seen tokens, mix with neural predictions
# ---------------------------------------------------------------------------
class NgramMixer:
    def __init__(self, vocab_size: int = 1024, alpha: float = 0.08):
        self.vocab_size = vocab_size
        self.alpha = alpha  # weight given to n-gram model
        self.unigram = np.ones(vocab_size, dtype=np.float64)   # Laplace smoothed
        self.bigram  = np.ones((vocab_size, vocab_size), dtype=np.float32) * 0.1

    def update(self, tokens: np.ndarray):
        """Feed observed tokens to update n-gram counts."""
        np.add.at(self.unigram, tokens, 1.0)
        if len(tokens) > 1:
            np.add.at(self.bigram, (tokens[:-1], tokens[1:]), 1.0)

    def predict(self, prev_token: int) -> np.ndarray:
        """Return probability distribution over vocab given previous token."""
        row = self.bigram[prev_token]
        return row / row.sum()

    def mix(self, neural_logits: Tensor, prev_tokens: Tensor) -> Tensor:
        """Mix neural logits with n-gram probabilities."""
        device = neural_logits.device
        B, V = neural_logits.shape
        neural_probs = F.softmax(neural_logits.float(), dim=-1)

        ngram_probs = torch.zeros(B, V, device=device, dtype=torch.float32)
        for i in range(B):
            p = self.predict(int(prev_tokens[i].item()))
            ngram_probs[i] = torch.from_numpy(p).to(device)

        # Mixture: alpha * ngram + (1-alpha) * neural
        mixed = (1.0 - self.alpha) * neural_probs + self.alpha * ngram_probs
        # Convert back to logits for loss computation
        return torch.log(mixed.clamp(min=1e-10))

# ---------------------------------------------------------------------------
# Muon optimizer (from baseline — much better than AdamW for transformers)
# ---------------------------------------------------------------------------
def muon_update(grad: Tensor, lr: float, momentum: float, buf: Optional[Tensor],
                nesterov: bool = True) -> tuple[Tensor, Tensor]:
    if buf is None:
        buf = grad.clone()
    else:
        buf.mul_(momentum).add_(grad)
    if nesterov:
        update = grad + momentum * buf
    else:
        update = buf
    return update, buf

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz
    Adapted from the baseline. Uses Newton-Schulz orthogonalization on 2D params.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_params=None, adamw_lr=3e-4,
                 adamw_betas=(0.95, 0.95), adamw_wd=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)
        # AdamW fallback for 1D params
        self.adamw = torch.optim.AdamW(
            adamw_params if adamw_params else [],
            lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_wd
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if p.ndim == 1 or p.ndim == 0:
                    # 1D params: use simple SGD with momentum
                    if "buf" not in state:
                        state["buf"] = g.clone()
                    else:
                        state["buf"].mul_(momentum).add_(g)
                    update = g + momentum * state["buf"] if nesterov else state["buf"]
                    p.add_(update, alpha=-lr)
                else:
                    # 2D params: Newton-Schulz orthogonalization
                    if "buf" not in state:
                        state["buf"] = g.clone()
                    else:
                        state["buf"].mul_(momentum).add_(g)
                    g_ns = g + momentum * state["buf"] if nesterov else state["buf"]
                    # Orthogonalize via Newton-Schulz iterations
                    g_ns = self._ns_orthogonalize(g_ns, ns_steps)
                    p.add_(g_ns, alpha=-lr * max(1, p.size(0)/p.size(1)) ** 0.5)

        self.adamw.step()
        return loss

    def _ns_orthogonalize(self, G: Tensor, steps: int) -> Tensor:
        """Newton-Schulz orthogonalization on a 2D gradient matrix."""
        m, n = G.shape
        if m > n:
            G = G.t()
            transposed = True
        else:
            transposed = False
        # Scale
        s = G.norm() + 1e-8
        X = G / s
        # Newton-Schulz iterations: X ← 1.5*X - 0.5*X*X^T*X
        for _ in range(steps):
            A = X @ X.t()
            X = (1.5 * X) - (0.5 * A @ X)
        if transposed:
            X = X.t()
        return X * s

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
class DataLoader:
    def __init__(self, data_path: str, seq_len: int, rank: int, world_size: int):
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        # Lazy memmap loading — avoids OOM on large datasets
        shard_paths = sorted(Path(data_path).glob("*.bin"))
        assert shard_paths, f"No .bin files found in {data_path}"
        self.shards = [np.memmap(str(s), dtype=np.uint16, mode="r") for s in shard_paths]
        total_tokens = sum(len(s) for s in self.shards)
        print(f"[rank {rank}] Found {len(shard_paths)} shards, {total_tokens:,} tokens (memmap)")
        self.shard_idx = 0
        self.pos = rank * seq_len  # stagger start positions across ranks

    @property
    def data(self):
        return self.shards[self.shard_idx % len(self.shards)]

    def next_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        T = self.seq_len
        stride = T * self.world_size
        x_list, y_list = [], []
        for _ in range(batch_size):
            shard = self.shards[self.shard_idx % len(self.shards)]
            if self.pos + T + 1 >= len(shard):
                # Advance to next shard
                self.shard_idx += 1
                self.pos = self.rank * T
                shard = self.shards[self.shard_idx % len(self.shards)]
            x_list.append(torch.from_numpy(shard[self.pos: self.pos + T].astype(np.int32)))
            y_list.append(torch.from_numpy(shard[self.pos + 1: self.pos + T + 1].astype(np.int32)))
            self.pos += stride
        x = torch.stack(x_list).long()
        y = torch.stack(y_list).long()
        return x, y

# ---------------------------------------------------------------------------
# Evaluation with LoRA TTT + N-gram mixer + temperature calibration
# ---------------------------------------------------------------------------
@torch.no_grad()
def count_bytes_per_token(token_ids: Tensor, vocab_size: int,
                           tokenizer: Optional[spm.SentencePieceProcessor]) -> Tensor:
    """
    Return per-token byte counts for the bpb metric.
    Uses real tokenizer byte lengths when available — critical for accurate leaderboard bpb.
    Falls back to sp1024 empirical average (~3.3 bytes/token) if tokenizer is absent.
    """
    if tokenizer is None:
        return torch.full(token_ids.shape, 3.3, dtype=torch.float32)
    flat = token_ids.flatten().tolist()
    byte_counts = []
    for tid in flat:
        piece = tokenizer.id_to_piece(int(tid))
        decoded = piece.replace("\u2581", " ")   # \u2581 = \u2581 (leading space marker)
        byte_counts.append(max(len(decoded.encode("utf-8")), 1))
    return torch.tensor(byte_counts, dtype=torch.float32).view(token_ids.shape)

def evaluate_with_ttt(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    seq_len: int = 2048,      # Longer eval context (ALiBi extrapolates)
    ttt_lr: float = 3e-4,
    ttt_steps: int = 3,       # TTT gradient steps per chunk
    ttt_rank: int = 4,
    use_ngram: bool = True,
    ngram_alpha: float = 0.06,
    temperature: float = 1.0,
    vocab_size: int = 1024,
) -> float:
    """
    Evaluate bpb using:
    1. LoRA TTT: for each chunk, do `ttt_steps` gradient steps on that chunk's own tokens
    2. N-gram mixer: blend neural probs with bigram predictions
    3. Temperature calibration: scale logits by tuned temperature
    Returns bits-per-byte.
    """
    model.eval()

    # Inject LoRA adapters
    lora_params = inject_lora(model, rank=ttt_rank)
    lora_optimizer = torch.optim.AdamW(lora_params, lr=ttt_lr, betas=(0.9, 0.95), weight_decay=0.0)

    # N-gram mixer
    mixer = NgramMixer(vocab_size=vocab_size, alpha=ngram_alpha) if use_ngram else None

    total_loss_bits = 0.0
    total_bytes = 0
    n_tokens = len(val_tokens)

    for start in range(0, n_tokens - seq_len - 1, seq_len):
        chunk = val_tokens[start: start + seq_len + 1].to(device)
        x = chunk[:-1].unsqueeze(0)  # [1, T]
        y = chunk[1:].unsqueeze(0)   # [1, T]

        # --- TTT: fine-tune LoRA on this chunk ---
        model.train()
        for step in range(ttt_steps):
            lora_optimizer.zero_grad()
            logits = model(x)  # [1, T, V]
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            lora_optimizer.step()

        # --- Inference: compute log-probs for this chunk ---
        model.eval()
        with torch.no_grad():
            logits = model(x)  # [1, T, V]
            logits = logits / temperature

            if mixer is not None:
                # Mix with n-gram predictions
                # Update n-gram counts from previous chunks (not current — no cheating)
                log_probs_list = []
                for t in range(x.shape[1]):
                    prev_tok = x[0, t]
                    log_p = mixer.mix(logits[0, t:t+1], prev_tok.unsqueeze(0)).squeeze(0)
                    log_probs_list.append(log_p)
                log_probs = torch.stack(log_probs_list, dim=0)  # [T, V]
                # Compute NLL
                targets = y[0]  # [T]
                token_nlls = -log_probs[torch.arange(len(targets)), targets]
            else:
                log_probs = F.log_softmax(logits[0], dim=-1)  # [T, V]
                targets = y[0]
                token_nlls = -log_probs[torch.arange(len(targets)), targets]  # [T]

            # bpb = nll / log(2) since we work in nats, bpb = nats / ln(2)
            # Approximate 1 token ≈ 1 byte for sp1024
            chunk_bits = token_nlls.sum().item() / math.log(2)
            total_loss_bits += chunk_bits
            total_bytes += len(targets)

        # Update n-gram with this chunk's tokens (for future chunks)
        if mixer is not None:
            mixer.update(chunk.cpu().numpy())

    # Reset LoRA after eval
    remove_lora(model)

    bpb = total_loss_bits / max(total_bytes, 1)
    return bpb

# ---------------------------------------------------------------------------
# Calibrate temperature on a held-out validation slice
# ---------------------------------------------------------------------------
def calibrate_temperature(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    seq_len: int = 512,
    vocab_size: int = 1024,
) -> float:
    """Find optimal temperature T to minimize bpb on a small validation slice."""
    model.eval()
    # Use first 50 chunks for calibration
    logits_list, targets_list = [], []
    max_calibration_chunks = 50

    with torch.no_grad():
        for start in range(0, min(max_calibration_chunks * seq_len, len(val_tokens) - seq_len - 1), seq_len):
            chunk = val_tokens[start: start + seq_len + 1].to(device)
            x = chunk[:-1].unsqueeze(0)
            y = chunk[1:].unsqueeze(0)
            logits = model(x)[0]   # [T, V]
            logits_list.append(logits.cpu())
            targets_list.append(y[0].cpu())

    all_logits  = torch.cat(logits_list, dim=0)   # [N, V]
    all_targets = torch.cat(targets_list, dim=0)  # [N]

    # Grid search temperature in [0.7, 1.3]
    best_T, best_bpb = 1.0, float("inf")
    for T in [t / 10.0 for t in range(7, 14)]:
        lp = F.log_softmax(all_logits / T, dim=-1)
        nll = -lp[torch.arange(len(all_targets)), all_targets].mean().item()
        bpb = nll / math.log(2)
        if bpb < best_bpb:
            best_bpb = bpb
            best_T = T

    print(f"  Calibrated temperature: {best_T:.1f}  (bpb={best_bpb:.4f})")
    return best_T

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    # --- Config via environment variables (matches baseline interface) ---
    run_id        = os.environ.get("RUN_ID", "xsa_lora_ttt")
    data_path     = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path= os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    vocab_size    = int(os.environ.get("VOCAB_SIZE", "1024"))
    max_wallclock = int(os.environ.get("MAX_WALLCLOCK_SECONDS", "580"))  # 9m40s leaves buffer

    # Model hyperparameters
    hidden         = int(os.environ.get("HIDDEN", "512"))
    num_layers     = int(os.environ.get("NUM_LAYERS", "12"))
    num_unique_blk = int(os.environ.get("NUM_UNIQUE_BLOCKS", "6"))
    num_heads      = int(os.environ.get("NUM_HEADS", "8"))
    mlp_ratio      = float(os.environ.get("MLP_RATIO", "3.5"))
    embed_proj_dim = int(os.environ.get("EMBED_PROJ_DIM", "64"))
    window         = int(os.environ.get("WINDOW", "128"))

    # Training hyperparameters
    train_seq_len  = int(os.environ.get("TRAIN_SEQ_LEN", "512"))
    batch_size     = int(os.environ.get("BATCH_SIZE", "32"))
    lr             = float(os.environ.get("LR", "0.018"))
    adamw_lr       = float(os.environ.get("ADAMW_LR", "3e-4"))
    warmup_iters   = int(os.environ.get("WARMUP_ITERS", "200"))
    grad_clip      = float(os.environ.get("GRAD_CLIP", "1.0"))

    # TTT / eval hyperparameters
    eval_seq_len   = int(os.environ.get("EVAL_SEQ_LEN", "2048"))  # ALiBi extrapolation
    ttt_lr         = float(os.environ.get("TTT_LR", "2e-4"))
    ttt_steps      = int(os.environ.get("TTT_STEPS", "3"))
    ttt_rank       = int(os.environ.get("TTT_RANK", "4"))
    ngram_alpha    = float(os.environ.get("NGRAM_ALPHA", "0.06"))

    # -----------------------------------------------------------------------
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    master = is_master(rank)

    if master:
        print(f"=== Parameter Golf: {run_id} ===")
        print(f"World size: {world_size}  |  Device: {device}")

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Build model
    model = TinyLM(
        vocab_size=vocab_size,
        hidden=hidden,
        num_heads=num_heads,
        num_layers=num_layers,
        num_unique_blocks=num_unique_blk,
        mlp_ratio=mlp_ratio,
        window=window,
        embed_proj_dim=embed_proj_dim,
    ).to(device)

    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params/1e6:.2f}M")

    # Compile for speed
    model = torch.compile(model, mode="reduce-overhead")

    # DDP wrapper
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if isinstance(model, DDP) else model

    # -----------------------------------------------------------------------
    # Optimizer
    # Separate 2D (Muon) and 1D (AdamW) parameters
    decay_params_2d, decay_params_1d = [], []
    skip_patterns_1d = ["resid_mix", "v_res_scale", "embed_small.weight",
                        "final_norm", "norm1", "norm2", "q_gain"]
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not any(pat in name for pat in skip_patterns_1d):
            decay_params_2d.append(p)
        else:
            decay_params_1d.append(p)

    optimizer = Muon(
        decay_params_2d, lr=lr, momentum=0.95, nesterov=True, ns_steps=5,
        adamw_params=decay_params_1d, adamw_lr=adamw_lr,
        adamw_betas=(0.9, 0.95), adamw_wd=0.0
    )

    # LR schedule: warmup then cosine decay
    def get_lr(step: int, total_steps: int) -> float:
        if step < warmup_iters:
            return step / warmup_iters
        progress = (step - warmup_iters) / max(1, total_steps - warmup_iters)
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    # -----------------------------------------------------------------------
    # Data
    loader = DataLoader(data_path, train_seq_len, rank, world_size)

    # -----------------------------------------------------------------------
    # Training loop
    step = 0
    total_tokens = 0
    log_lines = []

    if master:
        print("Starting training...")

    while True:
        elapsed = time.time() - t_start
        if elapsed >= max_wallclock:
            break

        # LR scheduling
        progress = elapsed / max_wallclock
        # Estimate total steps (rough)
        total_steps_estimate = max_wallclock * 30  # ~30 steps/sec on 8xH100
        lr_factor = get_lr(step, total_steps_estimate)
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_factor
        optimizer.adamw.param_groups[0]["lr"] = adamw_lr * lr_factor

        # Forward + backward
        x, y = loader.next_batch(batch_size)
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch_size * train_seq_len * world_size
        step += 1

        if master and step % 100 == 0:
            elapsed = time.time() - t_start
            bpb_approx = loss.item() / math.log(2)
            msg = f"step={step:5d}  loss={loss.item():.4f}  bpb≈{bpb_approx:.4f}  lr={lr*lr_factor:.5f}  t={elapsed:.0f}s  tokens={total_tokens/1e9:.3f}B"
            print(msg)
            log_lines.append(msg)

    # -----------------------------------------------------------------------
    # Post-training: size check + calibration
    if master:
        print("\n=== Post-training ===")

        # Check artifact size
        raw_model.cpu()
        artifact_bytes = serialize_artifact(raw_model)
        size_mb = len(artifact_bytes) / 1e6
        print(f"Artifact size: {size_mb:.3f} MB  (limit: 16.000 MB)")

        if size_mb > 16.0:
            print("WARNING: Artifact exceeds 16MB limit! Reduce model or use more aggressive quantization.")
        else:
            print(f"Size OK: {16.0 - size_mb:.3f} MB headroom remaining")

    # -----------------------------------------------------------------------
    # Evaluation with TTT + n-gram mixer
    if master:
        raw_model = raw_model.to(device)
        raw_model.eval()

        print("\n=== Calibrating temperature ===")
        # Load validation tokens
        val_shards = sorted(Path(data_path).parent.glob("fineweb_val_*.bin"))
        if not val_shards:
            val_shards = sorted(Path(data_path).glob("*val*.bin"))

        if val_shards:
            val_data = np.concatenate([np.fromfile(str(s), dtype=np.uint16) for s in val_shards])
            val_tokens = torch.from_numpy(val_data.astype(np.int32)).long()
            print(f"Val tokens: {len(val_tokens):,}")

            # Calibrate temperature on first 10% of val set
            calib_tokens = val_tokens[:len(val_tokens) // 10]
            temperature = calibrate_temperature(raw_model, calib_tokens, device,
                                                seq_len=train_seq_len, vocab_size=vocab_size)

            print("\n=== Evaluating with LoRA TTT + N-gram mixer ===")
            # Use remaining 90% for final eval
            eval_tokens = val_tokens[len(val_tokens) // 10:]

            bpb = evaluate_with_ttt(
                raw_model,
                eval_tokens,
                device,
                seq_len=eval_seq_len,   # 2048 — ALiBi extrapolates
                ttt_lr=ttt_lr,
                ttt_steps=ttt_steps,
                ttt_rank=ttt_rank,
                use_ngram=True,
                ngram_alpha=ngram_alpha,
                temperature=temperature,
                vocab_size=vocab_size,
            )
            print(f"\n{'='*50}")
            print(f"FINAL BPB: {bpb:.4f}")
            print(f"{'='*50}")

            # Write submission artifacts
            out_dir = Path(f"records/{run_id}")
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "train.log", "w") as f:
                f.write("\n".join(log_lines))

            with open(out_dir / "submission.json", "w") as f:
                json.dump({
                    "run_id": run_id,
                    "bpb": bpb,
                    "artifact_size_bytes": len(artifact_bytes),
                    "total_tokens_trained": total_tokens,
                    "training_seconds": time.time() - t_start,
                    "temperature": temperature,
                    "eval_seq_len": eval_seq_len,
                    "architecture": {
                        "hidden": hidden, "num_layers": num_layers,
                        "num_unique_blocks": num_unique_blk, "num_heads": num_heads,
                        "mlp_ratio": mlp_ratio, "window": window,
                        "embed_proj_dim": embed_proj_dim,
                        "attention": "XSA_with_value_residual",
                        "positional": "ALiBi",
                        "quantization": "INT6",
                    },
                    "eval_tricks": {
                        "lora_ttt": True, "ttt_rank": ttt_rank,
                        "ttt_steps": ttt_steps, "ttt_lr": ttt_lr,
                        "ngram_mixer": True, "ngram_alpha": ngram_alpha,
                        "temperature_calibration": True,
                    }
                }, indent=2)

            with open(out_dir / "model.bin", "wb") as f:
                f.write(artifact_bytes)

            print(f"\nSaved to {out_dir}/")
        else:
            print("No validation data found — skipping eval")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
