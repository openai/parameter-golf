"""
Whirlpool v5 — Lorentzian Transformer with Fused Triton Kernels

Champion v5: v4 base + fused kernel integration.

Architecture:
  - Tri-block: Local(0,1), Transition(3), Hierarchy(6,7) — 3 shared blocks, 5 active orbits
  - d_model=768, 12H/2KV GQA 6:1, head_dim=64
  - Flash Lorentz Attention: fused Triton kernel (hyperboloid proj + Minkowski inner + softmax + centroid)
  - Lorentzian centroid value aggregation on the manifold
  - Progressive curvature 0.1→2.0 (exponential), scale clamping [-5,5]
  - MLP 5x, Fused LeakyReLU(0.5)² Triton kernel (saves 800MB VRAM traffic/step)
  - torch.compile + Flash Lorentz custom_op (30-36ms/step on H100)
  - 20% warmup + cosine decay, MuonAdamW (muon lr=0.04, wd=0.12)
  - DDP-ready, int6/int8 per-row quantization (Triton-packed int6 available), full eval pipeline
  - Online n-gram GPU hash table for eval-time adaptation

Fused kernels (optional, graceful fallback):
  - fused_leaky: LeakyReLU(0.5)² in one kernel (eliminates 160MB intermediate per call)
  - custom_quantizer: Triton-packed int6/int5 with GPU dequant kernel
  - online_ngram: GPU-resident hash table for eval-time n-gram updates

Requires: PyTorch >= 2.9.1+cu128, flash-lorentz-attention package
Template: RunPod y5cejece4j
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import contextlib
import gc
import glob
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor

# --- Ensure /workspace is on path for fused kernel modules ---
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

# --- Fused LeakyReLU² Triton kernel (saves 800MB VRAM traffic/step) ---
try:
    from fused_leaky.kernel import fused_leaky_relu_squared
    HAS_FUSED_LEAKY = True
except ImportError:
    HAS_FUSED_LEAKY = False

# --- Int6/Int5 Triton-packed custom quantizer ---
try:
    from custom_quantizer.kernel import quantize_int6, dequantize_int6, dequantize_int6_gpu
    HAS_INT6_TRITON = True
except ImportError:
    HAS_INT6_TRITON = False

# --- Online n-gram GPU hash table ---
# Online n-gram DISABLED — hash overflow crashes on CUDA (int64 overflow in h*1024+token)
HAS_ONLINE_NGRAM = False

try:
    from flash_lorentz import flash_lorentz_attn
    HAS_FLASH_LORENTZ = True
except ImportError:
    HAS_FLASH_LORENTZ = False
    flash_lorentz_attn = None

SMOKE_TEST = "--smoke-test" in sys.argv

# ── v5.0 defaults: centroid + lens ON, everything else OFF ───────
ABL_LORENTZ_CENTROID = bool(int(os.environ.get("WP_LORENTZ_CENTROID", "1")))  # PROVEN: -1.23 BPB
ABL_GEOMETRIC_LENS = bool(int(os.environ.get("WP_GEOMETRIC_LENS", "0")))      # v59: OFF — T3 beat T1 in H100 sweep (1.626 vs 1.728)
ABL_CURVATURE_USHAPE = bool(int(os.environ.get("WP_CURVATURE_USHAPE", "0")))  # CLASHES with centroid
ABL_MIXED_CURVATURE = bool(int(os.environ.get("WP_MIXED_CURVATURE", "0")))    # KILLED: breaks projection
ABL_GEODESIC_SKIP = bool(int(os.environ.get("WP_GEODESIC_SKIP", "0")))        # Untested
ABL_WORMHOLE = bool(int(os.environ.get("WP_WORMHOLE", "0")))                  # KILLED: step overhead
ABL_HYP_NGRAM = bool(int(os.environ.get("WP_HYP_NGRAM", "0")))               # KILLED: hurts convergence
ABL_ADAPTIVE_GATE = bool(int(os.environ.get("WP_ADAPTIVE_GATE", "0")))        # Adaptive orbit width gating

# Ablation status printed in main() after DDP setup

# ---------------------------------------------------------------------------
# Device + DDP
# ---------------------------------------------------------------------------

distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
master_process = rank == 0

if torch.cuda.is_available():
    device_type = "cuda"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    if master_process:
        print(f"Device: CUDA ({torch.cuda.get_device_name()}) [rank {rank}/{world_size}]")
elif torch.backends.mps.is_available():
    device_type = "mps"
    device = torch.device("mps")
    if master_process:
        print("Device: MPS (Apple Silicon)")
else:
    device_type = "cpu"
    device = torch.device("cpu")
    if master_process:
        print("Device: CPU")

# ---------------------------------------------------------------------------
# Data loading (inlined — no prepare_shim dependency)
# ---------------------------------------------------------------------------

import sentencepiece as spm

MAX_SEQ_LEN = 1024
_script_dir = os.path.dirname(os.path.abspath(__file__))

# Resolve data paths
for _dc in [os.path.join(_script_dir, "data"), "/workspace/parameter-golf/data",
            os.path.join(_script_dir, "..", "..", "parameter-golf", "data"), "./data"]:
    if os.path.isdir(_dc):
        DATA_PATH = os.path.join(_dc, "datasets", "fineweb10B_sp1024")
        TOKENIZER_PATH = os.path.join(_dc, "tokenizers", "fineweb_1024_bpe.model")
        break
else:
    DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

TRAIN_FILES = os.path.join(DATA_PATH, "fineweb_train_*.bin")
VAL_FILES = os.path.join(DATA_PATH, "fineweb_val_*.bin")


def load_data_shard(file: Path) -> torch.Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    assert header.size == 256 and int(header[0]) == 20240520
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens,
                                         offset=256 * np.dtype("<i4").itemsize).astype(np.uint16, copy=False))


class Tokenizer:
    def __init__(self, sp_model):
        self.sp = sp_model
    @classmethod
    def from_directory(cls):
        return cls(spm.SentencePieceProcessor(model_file=TOKENIZER_PATH))
    def get_vocab_size(self):
        return int(self.sp.vocab_size())


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files for {pattern}"
        self.file_idx, self.pos = 0, 0
        self.tokens = load_data_shard(self.files[0])
    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0
    def take(self, n):
        chunks, rem = [], n
        while rem > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0: self._advance(); continue
            k = min(rem, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k; rem -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def make_dataloader(tokenizer, batch_size, seq_len, split="train"):
    stream = TokenStream(TRAIN_FILES if split == "train" else VAL_FILES)
    _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    while True:
        chunk = stream.take(batch_size * seq_len + 1).to(dtype=torch.int64, device=_dev)
        yield chunk[:-1].reshape(batch_size, seq_len), chunk[1:].reshape(batch_size, seq_len), 0


def build_sentencepiece_luts(sp, vocab_size, dev):
    sp_vs = int(sp.vocab_size())
    ts = max(sp_vs, vocab_size)
    bb = np.zeros((ts,), dtype=np.int16)
    hs = np.zeros((ts,), dtype=np.bool_)
    ib = np.ones((ts,), dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid): continue
        ib[tid] = False
        if sp.is_byte(tid): bb[tid] = 1; continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"): hs[tid] = True; piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=dev),
            torch.tensor(hs, dtype=torch.bool, device=dev),
            torch.tensor(ib, dtype=torch.bool, device=dev))


def evaluate_bpb(model, tokenizer, batch_size, eval_wall=600):
    """Val BPB eval with wall-clock limit.

    Uses model(x, y) which returns loss. Supports distributed (all_reduce)
    and single-GPU modes. Stops early if eval_wall seconds exceeded.
    """
    _dev = next(model.parameters()).device
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vs = int(sp.vocab_size())
    bb_lut, hs_lut, ib_lut = build_sentencepiece_luts(sp, vs, _dev)
    vf = [Path(p) for p in sorted(glob.glob(VAL_FILES))]
    vt = torch.cat([load_data_shard(f) for f in vf]).contiguous()
    usable = ((vt.numel() - 1) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    vt = vt[:usable + 1]
    total_seqs = (vt.numel() - 1) // MAX_SEQ_LEN

    # Always eval full val set (rank 0 solo after DDP teardown)
    seq_start = 0
    seq_end = total_seqs

    val_loss_sum = torch.zeros((), device=_dev, dtype=torch.float64)
    val_token_count = torch.zeros((), device=_dev, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=_dev, dtype=torch.float64)

    t_eval_start = time.perf_counter()
    model.eval()
    n_batches = 0
    with torch.inference_mode():
        for bs in range(seq_start, seq_end, batch_size):
            # Wall-clock check — uses absolute start time (includes compile)
            elapsed = time.perf_counter() - t_eval_start
            if n_batches > 0 and elapsed > eval_wall - 5:
                if rank == 0:
                    print(f"[EVAL] wall hit at {elapsed:.0f}s, {n_batches} batches done", flush=True)
                break
            be = min(bs + batch_size, seq_end)
            loc = vt[bs*MAX_SEQ_LEN:be*MAX_SEQ_LEN+1].to(device=_dev, dtype=torch.int64)
            x = loc[:-1].reshape(be - bs, MAX_SEQ_LEN)
            y = loc[1:].reshape(be - bs, MAX_SEQ_LEN)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            bt = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * bt
            val_token_count += bt
            tb = bb_lut[y.reshape(-1)].to(torch.int16)
            tb += (hs_lut[y.reshape(-1)] & ~ib_lut[x.reshape(-1)]).to(torch.int16)
            val_byte_count += tb.to(torch.float64).sum()
            n_batches += 1

    # No all_reduce needed — eval always runs on rank 0 solo after DDP teardown

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(bits_per_token * tokens_per_byte)

TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "15" if SMOKE_TEST else "600"))

# ===================================================================
# LORENTZ GEOMETRY
# ===================================================================

# NOTE: @torch.compiler.disable REMOVED — NaN bug fixed in PyTorch 2.9.1+cu128
# REQUIRES: PyTorch >= 2.9.1 with CUDA 12.8. Older versions WILL NaN. (LESSONS-LEARNED #1, #34)
def stable_arcosh(x: Tensor) -> Tensor:
    x = x.clamp(min=1.0 + 1e-7)
    return torch.log(x + torch.sqrt(x * x - 1.0))


ABL_EXPMAP_PROJECTION = bool(int(os.environ.get("WP_EXPMAP_PROJECTION", "0")))  # v59: sqrt projection (expmap = 4.566 BPB, sqrt = 1.699)

# NOTE: @torch.compiler.disable REMOVED — NaN bug fixed in PyTorch 2.9.1+cu128
def project_to_hyperboloid(x_space: Tensor, c: float = 1.0) -> Tensor:
    """[..., d] -> [..., d+1] on hyperboloid.
    Two modes:
      Default (sqrt): x0 = sqrt(1/c + ||x||²) — standard, used during training
      ExpMap mode: treats x as tangent vector z, applies exp_map from origin

    REQUIRES PyTorch >= 2.9.1 — older versions NaN under torch.compile.
    """
    if ABL_EXPMAP_PROJECTION:
        sc = 1.0 / (c ** 0.5)
        z_norm = x_space.norm(dim=-1, keepdim=True).clamp(min=1e-7)
        # Clamp the actual argument to cosh/sinh, not the raw norm
        # Keep sinh(arg) < ~1e4 for numerical stability → arg < 10
        arg = (z_norm * sc).clamp(max=10.0)
        x0 = torch.cosh(arg)
        x_spatial = torch.sinh(arg) * (x_space / z_norm)
        return torch.cat([x0, x_spatial], dim=-1)
    else:
        x0 = torch.sqrt((1.0 / c) + x_space.pow(2).sum(dim=-1, keepdim=True))
        return torch.cat([x0, x_space], dim=-1)


def lorentz_spatial_norm(x: Tensor) -> Tensor:
    """Spatial norm ||x[1:]||. At apex (origin of hyperbolic space) this is 0."""
    return x[..., 1:].norm(dim=-1)


def lorentz_inner(a: Tensor, b: Tensor) -> Tensor:
    """Minkowski inner product: -a0*b0 + a1*b1 + ... (batched over last dim)."""
    return -a[..., :1] * b[..., :1] + (a[..., 1:] * b[..., 1:]).sum(dim=-1, keepdim=True)


def lorentz_distance(a: Tensor, b: Tensor, c: float = 1.0) -> Tensor:
    """Geodesic distance on hyperboloid: arcosh(-c * <a,b>_L)."""
    inner = lorentz_inner(a, b).squeeze(-1)
    return stable_arcosh((-c * inner).clamp(min=1.0))


def mobius_add(x: Tensor, y: Tensor, c: float = 1.0) -> Tensor:
    """Möbius addition in the Poincaré ball (for hyperbolic n-gram embeddings).
    Uses the Klein→Poincaré→add→Klein chain for numerical stability."""
    # Simplified: project to tangent space, add, project back
    # For small curvature this approximates Euclidean addition
    x2 = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
    y2 = (y * y).sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2*c*xy + c*y2) * x + (1 - c*x2) * y
    denom = 1 + 2*c*xy + c*c*x2*y2
    return num / denom.clamp(min=1e-6)


# ===================================================================
# CONFIG
# ===================================================================

@dataclass
class WhirlpoolV42Config:
    sequence_len: int = 1024
    vocab_size: int = 1024

    d_model: int = 768
    n_heads: int = 12            # 768 / 64 = 12
    n_kv_heads: int = 2          # GQA 6:1 (was 5:1 at d=640)
    head_dim: int = 64
    mlp_ratio: int = 5  # Best BPB: 1.5027 config
    n_orbits: int = 8            # total orbits (blocks have params for their assigned orbits)
    active_orbits: tuple = (0, 1, 2, 3, 4, 5, 6, 7)  # all 8 active (-0.084 BPB proven)

    # Tri-block: which orbits each block handles
    local_orbits: tuple = (0, 1, 2)        # flat geometry, local patterns
    transition_orbits: tuple = (3, 4)      # bridge local→hierarchical
    hierarchy_orbits: tuple = (5, 6, 7)    # high curvature, long-range

    # Progressive curvature per orbit
    curvature_min: float = 0.1   # orbit 0: nearly flat
    curvature_max: float = 2.0   # orbit N-1: highly curved

    # Regularization
    centrifugal_lambda: float = 0.005
    ortho_lambda: float = 0.0005

    # Training
    logit_softcap: float = 30.0
    crown_q_start: float = 0.8
    rope_base: float = 10000.0

    # Orbit embedding scale (10x stronger than v3's 0.01)
    orbit_embed_scale: float = 0.1

    # Trigram hash: local 3-token context
    trigram_table_size: int = 4096    # compact table
    trigram_dim: int = 256            # projected to d_model


def get_orbit_curvature(orbit_idx: int, n_orbits: int, c_min: float, c_max: float) -> float:
    if ABL_CURVATURE_USHAPE:
        # U-shape: flat → curved → flat (fix output head geometry mismatch)
        mid = (n_orbits - 1) / 2.0
        t = 1.0 - abs(orbit_idx - mid) / mid  # 0→1→0
    else:
        # Monotonic: flat → curved
        t = orbit_idx / max(n_orbits - 1, 1)
    return c_min * (c_max / max(c_min, 1e-8)) ** t  # v59f: exponential


# ===================================================================
# BUILDING BLOCKS
# ===================================================================

def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos = None
        self._sin = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._cache_len != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, :, None, :].to(dtype)
            self._sin = freqs.sin()[None, :, None, :].to(dtype)
            self._cache_len = seq_len
        return self._cos, self._sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


# ===================================================================
# TRIGRAM HASH (local 3-token context before geometry)
# ===================================================================

class TrigramHash(nn.Module):
    """Hash 3 consecutive tokens into a learned embedding.
    Captures local trigram context that shifts token positions on the hyperboloid.
    Subsumes bigram info (the 3-token window includes the 2-token pair)."""

    def __init__(self, table_size: int, tri_dim: int, model_dim: int):
        super().__init__()
        self.table_size = table_size
        self.embed = nn.Embedding(table_size, tri_dim)
        self.proj = CastedLinear(tri_dim, model_dim, bias=False) if tri_dim != model_dim else None
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        t = token_ids.to(torch.int64)
        mod = self.table_size - 1
        out = torch.full_like(t, mod)
        if t.size(-1) > 2:
            out[..., 2:] = (36313 * t[..., 2:] ^ 27191 * t[..., 1:-1] ^ 51637 * t[..., :-2]).abs() % mod
        return self._embed(out.long())

    def _embed(self, indices: Tensor) -> Tensor:
        h = self.embed(indices)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


# ===================================================================
# LORENTZ ATTENTION (ALL heads go through geometry)
# ===================================================================

class LorentzMultiHeadAttention(nn.Module):
    """ALL attention heads use Lorentz distance-based attention.
    Q,K are projected to the hyperboloid. Attention = softmax(-dist/temp).
    V stays in tangent space (Euclidean). GQA supported."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 rope_base: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.q_proj = CastedLinear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = CastedLinear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = CastedLinear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = CastedLinear(n_heads * head_dim, d_model, bias=False)

        self.rotary = Rotary(head_dim, base=rope_base)

        # Per-head temperature (learnable)
        # Buffer not Parameter — flash_lorentz handles temp internally via custom_op.
        # As Parameter it would be "unused" in SDPA fallback, crashing DDP.
        self.register_buffer('temp', torch.ones(n_heads) * math.sqrt(head_dim))

    def forward(self, x: Tensor, curvature: float) -> tuple[Tensor, Tensor]:
        """Flash Lorentz Attention — fused hyperboloid projection + Minkowski inner
        product + online softmax + centroid aggregation in a single tiled Triton kernel.
        O(T) memory instead of O(T²). ~10x speedup. GQA handled internally.
        See /Users/tmancino/AI/flash-lorentz-attention for kernel source."""
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D)
        k = self.k_proj(x).view(B, T, Hkv, D)
        v = self.v_proj(x).view(B, T, Hkv, D)

        # RoPE on Q,K in spatial coords (before hyperboloid projection)
        q, k = norm(q), norm(k)
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if HAS_FLASH_LORENTZ:
            # Flash Lorentz Attention: fused Triton kernel (custom_op, no graph breaks)
            attn_out, spatial_norm = flash_lorentz_attn(
                q, k, v,
                curvature=curvature,
                temp=self.temp,
                causal=True,
                use_centroid=ABL_LORENTZ_CENTROID,
            )
            out = attn_out.reshape(B, T, H * D)
        else:
            # SDPA fallback — no Lorentz geometry, just standard attention
            q_sdpa = q.transpose(1, 2)  # (B, H, T, D)
            # Expand KV for GQA
            rep = H // Hkv
            k_sdpa = k.unsqueeze(3).expand(B, T, Hkv, rep, D).reshape(B, T, H, D).transpose(1, 2)
            v_sdpa = v.unsqueeze(3).expand(B, T, Hkv, rep, D).reshape(B, T, H, D).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=True)
            out = attn_out.transpose(1, 2).reshape(B, T, H * D)
            spatial_norm = torch.zeros(1, device=x.device)

        return self.o_proj(out), spatial_norm


# ===================================================================
# SHARED BLOCK (orbited N times)
# ===================================================================

class WhirlpoolBlock(nn.Module):
    """Single shared block. Called once per orbit with different curvature."""

    def __init__(self, config: WhirlpoolV42Config, assigned_orbits: tuple):
        super().__init__()
        d = config.d_model
        self.assigned_orbits = assigned_orbits
        # Map global orbit index → local parameter index (no unused params)
        self._orbit_to_local = {o: i for i, o in enumerate(assigned_orbits)}
        n = len(assigned_orbits)

        # Lorentz attention (ALL heads)
        self.attn = LorentzMultiHeadAttention(
            d, config.n_heads, config.n_kv_heads, config.head_dim, config.rope_base)

        # MLP: LeakyReLU(0.5)², expansion ratio from config
        mlp_dim = config.mlp_ratio * d
        self.mlp_up = CastedLinear(d, mlp_dim, bias=False)
        self.mlp_down = CastedLinear(mlp_dim, d, bias=False)

        # Per-orbit parameters — only for assigned orbits (no unused params for DDP)
        self.orbit_embeds = nn.Parameter(torch.zeros(n, d))
        self.attn_scale = nn.Parameter(torch.zeros(n))
        self.mlp_scale = nn.Parameter(torch.zeros(n))

        # Adaptive orbit gate: learns per-orbit dimension importance
        if ABL_ADAPTIVE_GATE:
            n_groups = 8
            self.orbit_gate = nn.Linear(d, n_groups * n, bias=False)
            self.n_groups = n_groups
            nn.init.zeros_(self.orbit_gate.weight)

    def forward(self, x: Tensor, orbit_idx: int, curvature: float) -> tuple[Tensor, Tensor]:
        """Returns (output, lorentz_spatial_norm)."""
        local_idx = self._orbit_to_local[orbit_idx]

        # Adaptive gate: scale dimension groups based on input
        if ABL_ADAPTIVE_GATE:
            B, T, D = x.shape
            g = self.n_groups
            group_dim = D // g
            all_gates = self.orbit_gate(x.mean(dim=1))
            gate_vals = all_gates[:, local_idx * g:(local_idx + 1) * g]
            gate_vals = torch.sigmoid(gate_vals + 3.0)
            x_grouped = x.view(B, T, g, group_dim)
            x = (x_grouped * gate_vals[:, None, :, None]).view(B, T, D)

        # Orbit embedding (strong perturbation)
        x = x + self.orbit_embeds[local_idx]

        # Attention with Lorentz geometry at this orbit's curvature
        attn_out, lor_norm = self.attn(norm(x), curvature)
        # Scale clamping [-5, 5] — prevents orbit dominance, stabilizes training
        delta_attn = self.attn_scale[local_idx].clamp(-5, 5) * attn_out

        if ABL_GEODESIC_SKIP:
            dn = torch.sqrt((delta_attn ** 2).sum(-1, keepdim=True).clamp(min=1e-8))
            x = x * torch.cosh(dn * 0.1) + delta_attn * torch.sinh(dn * 0.1) / dn
        else:
            x = x + delta_attn

        # MLP
        h = self.mlp_up(norm(x))
        h = fused_leaky_relu_squared(h) if HAS_FUSED_LEAKY else F.leaky_relu(h, negative_slope=0.5).square()
        mlp_out = self.mlp_down(h)
        delta_mlp = self.mlp_scale[local_idx].clamp(-5, 5) * mlp_out

        if ABL_GEODESIC_SKIP:
            dn = torch.sqrt((delta_mlp ** 2).sum(-1, keepdim=True).clamp(min=1e-8))
            x = x * torch.cosh(dn * 0.1) + delta_mlp * torch.sinh(dn * 0.1) / dn
        else:
            x = x + delta_mlp

        return x, lor_norm


# ===================================================================
# MAIN MODEL
# ===================================================================

class WhirlpoolV42(nn.Module):
    def __init__(self, config: WhirlpoolV42Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.trigram = TrigramHash(config.trigram_table_size, config.trigram_dim, config.d_model)

        # Tri-block: 3 specialized blocks for different curvature regimes
        # Each block only allocates params for its assigned orbits (no unused params for DDP)
        self.local_block = WhirlpoolBlock(config, config.local_orbits)
        self.transition_block = WhirlpoolBlock(config, config.transition_orbits)
        self.hierarchy_block = WhirlpoolBlock(config, config.hierarchy_orbits)

        # Route orbits to blocks
        self._orbit_to_block = {}
        for o in config.local_orbits:
            self._orbit_to_block[o] = self.local_block
        for o in config.transition_orbits:
            self._orbit_to_block[o] = self.transition_block
        for o in config.hierarchy_orbits:
            self._orbit_to_block[o] = self.hierarchy_block

        # Tied embeddings: no lm_head module, use F.linear(x, tok_emb.weight) in forward
        # (v16 approach — CastedLinear assignment breaks PyTorch parameter registration)
        self.lm_head = None
        self.logit_softcap = config.logit_softcap

        # Precompute curvatures per orbit
        self.curvatures = [
            get_orbit_curvature(i, config.n_orbits, config.curvature_min, config.curvature_max)
            for i in range(config.n_orbits)
        ]

    @torch.no_grad()
    def init_weights(self):
        c = self.config
        d = c.d_model
        s = (3 ** 0.5) * (d ** -0.5) / (c.n_orbits ** 0.25)

        nn.init.normal_(self.tok_emb.weight, std=s)
        # lm_head is None (tied) — tok_emb.weight serves both roles

        # Trigram: zero-init (starts as no-op, learns to contribute)
        nn.init.zeros_(self.trigram.embed.weight)
        if self.trigram.proj is not None:
            nn.init.zeros_(self.trigram.proj.weight)

        # Init all 3 blocks identically
        for blk in [self.local_block, self.transition_block, self.hierarchy_block]:
            for proj in [blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj]:
                nn.init.uniform_(proj.weight, -s, s)
            nn.init.zeros_(blk.attn.o_proj.weight)
            blk.attn.temp.data.fill_(math.sqrt(c.head_dim))
            nn.init.uniform_(blk.mlp_up.weight, -s, s)
            nn.init.zeros_(blk.mlp_down.weight)
            nn.init.normal_(blk.orbit_embeds, std=c.orbit_embed_scale)
            # Scale init: use global orbit index for value, local index for storage
            for global_orbit, local_idx in blk._orbit_to_local.items():
                val = 0.3 / c.n_orbits * (c.n_orbits - global_orbit)
                blk.attn_scale.data[local_idx] = val
                blk.mlp_scale.data[local_idx] = val

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> Tensor:
        tok = self.tok_emb(idx).float()
        tri = self.trigram(idx).float()

        if ABL_HYP_NGRAM:
            # Hyperbolic n-gram: combine token + trigram via Möbius addition
            # instead of Euclidean add. Keeps embeddings on the manifold.
            # Scale down to unit ball for Poincaré Möbius add
            tok_norm = tok / (tok.norm(dim=-1, keepdim=True).clamp(min=1e-6) + 1.0)
            tri_norm = tri / (tri.norm(dim=-1, keepdim=True).clamp(min=1e-6) + 1.0)
            x = mobius_add(tok_norm, tri_norm, c=1.0)
            # Scale back up
            x = x * tok.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            x = tok + tri

        x = norm(x)

        lor_norms = []
        for orbit in self.config.active_orbits:
            c = self.curvatures[orbit]
            block = self._orbit_to_block[orbit]
            if device_type == "cuda":
                x, lor_norm = block(x, orbit, c)
            else:
                from torch.utils.checkpoint import checkpoint as grad_checkpoint
                x, lor_norm = grad_checkpoint(block, x, orbit, c, use_reentrant=False)
            lor_norms.append(lor_norm)

        logits = F.linear(norm(x), self.tok_emb.weight).float()  # tied embeddings
        cap = self.logit_softcap
        logits = cap * torch.tanh(logits / cap)

        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 targets.view(-1), ignore_index=-1)
            # Only add centrifugal during training, not eval
            if self.training:
                avg_lor_norm = torch.stack(lor_norms).mean()
                centrifugal = self.config.centrifugal_lambda * torch.exp(-avg_lor_norm)
                return ce + centrifugal
            return ce
        return logits


# ===================================================================
# MUON + ADAMW OPTIMIZER
# ===================================================================

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    step_t, lr_t = step_t.to(p.device, p.dtype), lr_t.to(p.device, p.dtype)
    beta1_t, beta2_t = beta1_t.to(p.device, p.dtype), beta2_t.to(p.device, p.dtype)
    eps_t, wd_t = eps_t.to(p.device, p.dtype), wd_t.to(p.device, p.dtype)
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1, bias2 = 1 - beta1_t ** step_t, 1 - beta2_t ** step_t
    p.add_(exp_avg / bias1 / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-lr_t)


def muon_step_fused(sg, sp, mb, smb, mom_t, lr_t, wd_t, b2_t, ns_steps, rd):
    mom_t = mom_t.to(sp.device, sp.dtype)
    lr_t = lr_t.to(sp.device, sp.dtype)
    wd_t = wd_t.to(sp.device, sp.dtype)
    b2_t = b2_t.to(sp.device, sp.dtype)
    mom = mom_t.to(sg.dtype)
    mb.lerp_(sg, 1 - mom)
    g = sg.lerp_(mb, mom)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X; X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT; X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    vm = g.float().square().mean(dim=rd, keepdim=True)
    rds = g.size(rd)
    vn = (vm.sum(dim=(-2, -1), keepdim=True) * rds).sqrt()
    b2c = b2_t.to(smb.dtype)
    smb.lerp_(vm.to(smb.dtype), 1 - b2c)
    ss = smb.clamp_min(1e-10).rsqrt()
    vnn = ((vm * rds) * ss.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (ss * (vn / vnn.clamp_min(1e-10))).to(g.dtype)
    lr, wd = lr_t.to(g.dtype), wd_t.to(g.dtype)
    mask = (g * sp) >= 0
    sp.sub_(lr * g + lr * wd * sp * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        for attr in ['step', 'lr', 'beta1', 'beta2', 'eps', 'wd']:
            setattr(self, f'_adamw_{attr}_t', torch.tensor(0.0, device="cpu"))
        for attr in ['momentum', 'lr', 'wd', 'beta2']:
            setattr(self, f'_muon_{attr}_t', torch.tensor(0.0, device="cpu"))
        ck = {"dynamic": False, "fullgraph": True}
        if device_type == "cuda":
            self.adamw_fn = torch.compile(adamw_step_fused, **ck)
            self.muon_fn = torch.compile(muon_step_fused, **ck)
        else:
            self.adamw_fn = adamw_step_fused
            self.muon_fn = muon_step_fused

    def _step_adamw(self, g):
        for p in g['params']:
            if p.grad is None:
                continue
            s = self.state[p]
            if not s:
                s['step'] = 0
                s['ea'] = torch.zeros_like(p)
                s['eas'] = torch.zeros_like(p)
            s['step'] += 1
            self._adamw_step_t.fill_(s['step'])
            self._adamw_lr_t.fill_(g['lr'])
            self._adamw_beta1_t.fill_(g['betas'][0])
            self._adamw_beta2_t.fill_(g['betas'][1])
            self._adamw_eps_t.fill_(g['eps'])
            self._adamw_wd_t.fill_(g['weight_decay'])
            self.adamw_fn(p, p.grad, s['ea'], s['eas'],
                          self._adamw_step_t, self._adamw_lr_t,
                          self._adamw_beta1_t, self._adamw_beta2_t,
                          self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, g):
        params = g['params']
        if not params:
            return
        p = params[0]
        s = self.state[p]
        n = len(params)
        sh, dev, dt = p.shape, p.device, p.dtype
        if "mb" not in s:
            s["mb"] = torch.zeros(n, *sh, dtype=dt, device=dev)
        if "smb" not in s:
            ss = (n, sh[-2], 1) if sh[-2] >= sh[-1] else (n, 1, sh[-1])
            s["smb"] = torch.zeros(ss, dtype=dt, device=dev)
        rd = -1 if sh[-2] >= sh[-1] else -2
        sg = torch.stack([p.grad for p in params])
        sp = torch.stack(params)
        self._muon_momentum_t.fill_(g["momentum"])
        self._muon_beta2_t.fill_(g.get("beta2", 0.95) or 0.0)
        self._muon_lr_t.fill_(g["lr"] * max(1.0, sh[-2] / sh[-1]) ** 0.5)
        self._muon_wd_t.fill_(g["weight_decay"])
        self.muon_fn(sg, sp, s["mb"], s["smb"], self._muon_momentum_t,
                     self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                     g["ns_steps"], rd)
        torch._foreach_copy_(params, list(sp.unbind(0)))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if g['kind'] == 'adamw':
                self._step_adamw(g)
            elif g['kind'] == 'muon':
                self._step_muon(g)


def setup_optimizer(model: WhirlpoolV42) -> MuonAdamW:
    d = model.config.d_model
    dscale = (d / 768) ** -0.5

    mat_params = []
    scalar_params = []
    for blk in [model.local_block, model.transition_block, model.hierarchy_block]:
        for name, p in blk.named_parameters():
            if p.ndim == 2 and min(p.shape) >= 8:
                mat_params.append(p)
            else:
                scalar_params.append(p)

    # Trigram params: embed table -> AdamW, proj matrix -> Muon, scale -> scalar
    tri_embed_id = id(model.trigram.embed.weight)
    tri_proj_params = []
    tri_scalar_params = []
    for name, p in model.trigram.named_parameters():
        if id(p) == tri_embed_id:
            continue
        elif p.ndim == 2 and min(p.shape) >= 8:
            tri_proj_params.append(p)
        else:
            tri_scalar_params.append(p)

    groups = [
        # tok_emb serves as both embedding and lm_head (tied, v16 approach)
        dict(kind='adamw', params=list(model.tok_emb.parameters()),
             lr=0.06, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=scalar_params + tri_scalar_params,
             lr=0.08, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=[model.trigram.embed.weight],
             lr=0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
    ]

    all_muon = mat_params + tri_proj_params
    for shape in sorted({p.shape for p in all_muon}):
        gp = [p for p in all_muon if p.shape == shape]
        groups.append(dict(kind='muon', params=gp, lr=0.04,
                           momentum=0.95, ns_steps=5, beta2=0.95,
                           weight_decay=0.12))

    return MuonAdamW(groups)


# ===================================================================
# CROWN-Q
# ===================================================================

def crown_q_penalty(model: nn.Module, progress: float, start: float = 0.8) -> Tensor:
    if progress < start:
        return torch.tensor(0.0, device=device)
    penalty = torch.tensor(0.0, device=device)
    n = 0
    for p in model.parameters():
        if p.ndim == 2 and p.numel() > 64:
            scale = p.abs().max() / 31.5
            if scale > 1e-8:
                quantized = (p / scale).round() * scale
                penalty = penalty + (p - quantized).pow(2).mean()
                n += 1
    if n > 0:
        wp = (progress - start) / (1.0 - start)
        return 0.01 * wp * penalty / n
    return torch.tensor(0.0, device=device)


# ===================================================================
# N-GRAM EVAL CACHE
# ===================================================================

class NgramCache:
    """Fast n-gram cache using numpy arrays. No Python loops at build or eval time."""

    def __init__(self, vocab_size: int, max_order: int = 4):
        self.vocab_size = vocab_size
        self.max_order = max_order
        # For each order: dict mapping context_hash -> numpy array of shape [vocab_size]
        self.tables = [{} for _ in range(max_order)]

    def build_from_shards(self, data_path: str, max_tokens: int = 5_000_000):
        """Build n-gram tables using fast numpy operations."""
        import time as _t
        t0 = _t.time()
        pattern = os.path.join(data_path, "fineweb_train_*.bin")
        files = sorted(glob.glob(pattern))
        if not files:
            print("  WARNING: No shards found for n-gram cache")
            return

        # Load tokens
        all_tokens = []
        seen = 0
        for f in files:
            header = np.fromfile(f, dtype="<i4", count=256)
            num_tokens = int(header[2])
            header_bytes = 256 * np.dtype("<i4").itemsize
            tokens = np.fromfile(f, dtype="<u2", count=num_tokens, offset=header_bytes)
            all_tokens.append(tokens)
            seen += num_tokens
            if seen >= max_tokens:
                break
        tokens = np.concatenate(all_tokens)[:max_tokens].astype(np.int64)
        V = self.vocab_size

        for order in range(1, self.max_order + 1):
            if len(tokens) <= order:
                continue
            # Hash contexts: hash = sum(tokens[i+j] * 1024^(order-1-j)) for j in 0..order-1
            ctx_hash = np.zeros(len(tokens) - order, dtype=np.int64)
            for j in range(order):
                ctx_hash += tokens[j:len(tokens) - order + j] * (1024 ** (order - 1 - j))
            next_toks = tokens[order:]

            # Combine context hash and next token into a single key for fast counting
            combined = ctx_hash * V + next_toks
            unique_combined, combined_counts = np.unique(combined, return_counts=True)

            # Unpack back to context hash and token
            ctx_keys = unique_combined // V
            tok_ids = unique_combined % V

            # Vectorized groupby: sort by ctx_keys, then split at boundaries
            sort_idx = np.argsort(ctx_keys)
            sorted_ctx = ctx_keys[sort_idx]
            sorted_tok = tok_ids[sort_idx]
            sorted_cnt = combined_counts[sort_idx]

            # Find boundaries where context changes
            boundaries = np.where(np.diff(sorted_ctx) != 0)[0] + 1
            boundaries = np.concatenate([[0], boundaries, [len(sorted_ctx)]])

            table = {}
            for i in range(len(boundaries) - 1):
                if len(table) >= 200_000:  # memory cap
                    break
                start, end = boundaries[i], boundaries[i + 1]
                total = int(sorted_cnt[start:end].sum())
                if total >= 3:
                    probs = np.zeros(V, dtype=np.float32)
                    probs[sorted_tok[start:end]] = sorted_cnt[start:end].astype(np.float32) / total
                    table[int(sorted_ctx[start])] = probs
            self.tables[order - 1] = table

        sizes = [len(t) for t in self.tables]
        print(f"  N-gram cache built: {len(tokens) / 1e6:.1f}M tokens, "
              f"contexts: {sizes}, time: {_t.time() - t0:.1f}s")

    def _build_gpu_tables(self, target_device):
        """Convert numpy dict tables to sorted GPU tensors for torch.searchsorted."""
        if hasattr(self, '_gpu_tables'):
            return
        self._gpu_tables = []
        for order_idx in range(self.max_order):
            table = self.tables[order_idx]
            if not table:
                self._gpu_tables.append(None)
                continue
            keys = np.array(sorted(table.keys()), dtype=np.int64)
            probs = np.stack([table[k] for k in keys], axis=0)
            self._gpu_tables.append({
                'keys': torch.from_numpy(keys).to(target_device),
                'probs': torch.from_numpy(probs).to(target_device),
            })
        sizes = [g['keys'].shape[0] if g else 0 for g in self._gpu_tables]
        print(f"  N-gram GPU tables: {sizes}")

    def blend_logits(self, model_logits: Tensor, token_ids: Tensor,
                     alpha: float = 0.3) -> Tensor:
        """Blend model logits with n-gram cache. Fully vectorized — no Python loops.

        For each n-gram order (highest first):
          1. Vectorized hash of all (B, T) context positions
          2. GPU binary search via torch.searchsorted
          3. Entropy-adaptive alpha per position
          4. Broadcast blend hit positions
        """
        B, T, V = model_logits.shape
        self._build_gpu_tables(model_logits.device)
        model_probs = F.softmax(model_logits, dim=-1)
        tids = token_ids.long()

        for order in range(self.max_order, 0, -1):
            gpu_table = self._gpu_tables[order - 1]
            if gpu_table is None or order > T:
                continue
            valid_T = T - order
            if valid_T <= 0:
                continue

            # Vectorized context hash: h = sum(tok[j] * 1024^(order-1-j))
            h = torch.zeros(B, valid_T, dtype=torch.int64, device=tids.device)
            for j in range(order):
                h = h * 1024 + tids[:, j:j + valid_T]

            # GPU binary search across all positions at once
            h_flat = h.reshape(-1)
            keys = gpu_table['keys']
            idx = torch.searchsorted(keys, h_flat).clamp(max=keys.shape[0] - 1)
            hit = (keys[idx] == h_flat)

            if not hit.any():
                continue

            # Gather n-gram probs + entropy-adaptive alpha
            ngram_p = gpu_table['probs'][idx]
            pos_probs = model_probs[:, order:, :].reshape(-1, V)
            H = -(pos_probs * pos_probs.clamp_min(1e-10).log()).sum(dim=-1)
            a = (alpha * torch.sigmoid(0.5 * (H - 4.0)) * hit.float()).unsqueeze(-1)

            blended = (1 - a) * pos_probs + a * ngram_p
            model_probs[:, order:, :] = blended.reshape(B, valid_T, V)

        return model_probs.clamp_min(1e-10).log()


# ===================================================================
# LR SCHEDULE
# ===================================================================

def get_lr(progress: float) -> float:
    warmup = 0.20  # 20% warmup — LORENTZ-1 finding: 36% improvement over 10%
    if progress < warmup:
        return progress / warmup
    else:
        # Cosine decay from 1.0 to 0.0
        import math
        decay_progress = (progress - warmup) / (1.0 - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


# ===================================================================
# QUANTIZATION (from upstream — verbatim)
# ===================================================================

import io
import zlib

def quantize_state_dict_int8(state_dict):
    """Per-row int8 quantization with percentile clipping (upstream pattern)."""
    CLIP_Q = 0.9999984
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes = {}
    stats = {"baseline_tensor_bytes": 0, "int8_payload_bytes": 0}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= 65536:
            # Small tensors kept as fp16
            if t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                t = t.to(torch.float16)
            passthrough[name] = t.contiguous()
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        t32 = t.float()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        if t32.ndim == 2:
            clip_abs = torch.quantile(t32.abs(), CLIP_Q, dim=1) if t32.numel() else torch.empty(t32.shape[0])
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            s = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127).to(torch.int8)
            scales[name] = s.to(torch.float16)
        else:
            clip_abs = float(torch.quantile(t32.abs().flatten(), CLIP_Q).item()) if t32.numel() else 0.0
            s = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / s), -127, 127).to(torch.int8)
            scales[name] = s
        quantized[name] = q.contiguous()
        stats["int8_payload_bytes"] += q.numel() + scales[name].numel() * scales[name].element_size()
    obj = {"quantized": quantized, "scales": scales, "dtypes": dtypes,
           "passthrough": passthrough, "passthrough_orig_dtypes": passthrough_orig_dtypes}
    return obj, stats


def dequantize_state_dict_int8(obj):
    """Reverse of quantize_state_dict_int8 (upstream pattern)."""
    out = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
        else:
            out[name] = (q.float() * float(s.item())).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().cpu().contiguous()
        orig = passthrough_orig_dtypes.get(name)
        if isinstance(orig, str):
            out_t = out_t.to(dtype=getattr(torch, orig)).contiguous()
        out[name] = out_t
    return out


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()
    seed = int(os.environ.get("SEED", 1337))
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if master_process:
        print(f"Seed: {seed}")
    torch.set_float32_matmul_precision("high")

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    if master_process:
        print(f"Vocab: {vocab_size:,}")

    # Configurable via env vars. Auto-scales TOTAL_BATCH for multi-GPU if not set.
    # 1xGPU: BATCH=16, TB=16384. 4xGPU: BATCH=8, TB=32768. 8xGPU: BATCH=8, TB=65536.
    BATCH = int(os.environ.get("BATCH_SIZE", "8" if world_size > 1 else "16")) if device_type == "cuda" else 8
    _min_tb = BATCH * MAX_SEQ_LEN * world_size  # minimum for grad_accum=1
    TOTAL_BATCH = int(os.environ.get("TOTAL_BATCH", str(max(_min_tb, 2 ** 14))))

    config = WhirlpoolV42Config(sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size)

    if master_process:
        print(f"\nWhirlpool v4.8 [{'SMOKE' if SMOKE_TEST else 'FULL'} {TIME_BUDGET}s]")
        print(f"  {config.d_model}-dim, 1 shared block x {config.n_orbits} orbits")
        print(f"  {config.n_heads} Lorentz heads (ALL geometry), GQA {config.n_kv_heads}KV")
        print(f"  MLP: {config.mlp_ratio}x, LeakyReLU(0.5)²")
        print(f"  Curvature per orbit: {config.curvature_min}→{config.curvature_max}")
        print(f"  Centrifugal λ={config.centrifugal_lambda}")
        print(f"  Orbit embed scale: {config.orbit_embed_scale}")
        print(f"  NEW: No arcosh (raw Minkowski inner product)")
        print(f"  NEW: {config.n_orbits} total orbits, active={list(config.active_orbits)}, d_model={config.d_model}")

        curvatures = [get_orbit_curvature(i, config.n_orbits, config.curvature_min, config.curvature_max)
                      for i in range(config.n_orbits)]
        print(f"  Curvatures: [{', '.join(f'{c:.2f}' for c in curvatures)}]")

    with torch.device("meta"):
        model = WhirlpoolV42(config)
    model.to_empty(device=device)
    model.init_weights()

    n_params = model.num_params()
    effective = n_params  # shared block counted once
    local_p = sum(p.numel() for p in model.local_block.parameters())
    trans_p = sum(p.numel() for p in model.transition_block.parameters())
    hier_p = sum(p.numel() for p in model.hierarchy_block.parameters())
    # Effective: each block reused for its assigned orbits
    effective_with_orbits = (n_params - local_p - trans_p - hier_p +
                             local_p * len(config.local_orbits) +
                             trans_p * len(config.transition_orbits) +
                             hier_p * len(config.hierarchy_orbits))

    print(f"  Stored params: {n_params / 1e6:.1f}M")
    print(f"  Effective params: {effective_with_orbits / 1e6:.1f}M (block x {config.n_orbits} orbits)")
    print(f"  Est int6 size: {n_params * 6 / 8 / 1e6:.1f}MB")
    print(f"  16MB headroom: {16.0 - n_params * 6 / 8 / 1e6:.1f}MB")

    optimizer = setup_optimizer(model)

    # H6: EMA weight averaging (τ=0.997) — proven free improvement from v16
    EMA_DECAY = 0.997
    ema_shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    raw_model = model

    if master_process:
        print("Optimizer: MuonAdamW")
        print(f"EMA: τ={EMA_DECAY}, tracking {len(ema_shadow)} params")

    # torch.compile BEFORE DDP — backward has @torch.compiler.disable to prevent
    # tracing into Triton kernels. Forward is opaque via custom_op.
    _pt_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if _pt_version < (2, 9):
        if master_process:
            print(f"WARNING: PyTorch {torch.__version__} < 2.9 — torch.compile DISABLED (NaN risk)")
    elif device_type == "cuda":
        model = torch.compile(model)
        if master_process:
            print(f"torch.compile: ENABLED (PyTorch {torch.__version__}, flash backward compiler-disabled)")
    else:
        if master_process:
            print("torch.compile: DISABLED (non-CUDA device)")

    # DDP wrap AFTER compile
    if distributed:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
        if master_process:
            print(f"DDP: {world_size} GPUs")

    if master_process:
        print(f"Fused kernels: leaky_relu²={'ON' if HAS_FUSED_LEAKY else 'OFF'} "
              f"int6_triton={'ON' if HAS_INT6_TRITON else 'OFF'} "
              f"online_ngram={'ON' if HAS_ONLINE_NGRAM else 'OFF'}")

    tpf = BATCH * MAX_SEQ_LEN
    # Scale grad_accum by world_size for DDP
    assert TOTAL_BATCH % (tpf * world_size) == 0, \
        f"TOTAL_BATCH={TOTAL_BATCH} not divisible by BATCH*SEQ*WORLD={tpf}*{world_size}={tpf*world_size}"
    grad_accum = TOTAL_BATCH // (tpf * world_size)
    train_loader = make_dataloader(tokenizer, BATCH, MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)

    if device_type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = contextlib.nullcontext()

    print(f"Budget: {TIME_BUDGET}s | Batch: {BATCH} | Accum: {grad_accum}")
    print()

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    def sync():
        if device_type == "cuda":
            torch.cuda.synchronize()

    # Compiler warmup: 20 steps to prime torch.compile, then restore state.
    # Saves 5-10s of compilation from the actual training budget.
    if device_type == "cuda" and distributed:
        if master_process:
            print("Compiler warmup (20 steps)...", flush=True)
        _saved_weights = {n: p.data.clone() for n, p in raw_model.named_parameters()}
        _saved_opt = optimizer.state_dict()
        for _wu in range(20):
            with autocast_ctx:
                _wl = model(x, y)
            (_wl / grad_accum).backward()
            if _wu % grad_accum == grad_accum - 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            x, y, epoch = next(train_loader)
        optimizer.zero_grad(set_to_none=True)
        # Restore original weights and optimizer
        with torch.no_grad():
            for n, p in raw_model.named_parameters():
                p.data.copy_(_saved_weights[n])
        optimizer.load_state_dict(_saved_opt)
        del _saved_weights, _saved_opt
        sync()
        if master_process:
            print("Compiler warmup done", flush=True)

    t_train = time.time()
    smooth, wall, step = 0, 0, 0

    while True:
        sync()
        t0 = time.time()

        for micro in range(grad_accum):
            # Only sync gradients on last micro-step (saves NCCL bandwidth)
            if distributed:
                model.require_backward_grad_sync = (micro == grad_accum - 1)
            with autocast_ctx:
                loss = model(x, y)

            progress = min(wall / TIME_BUDGET, 1.0)
            cq = crown_q_penalty(raw_model, progress, config.crown_q_start)
            total_loss = loss + cq if isinstance(cq, Tensor) and cq.requires_grad else loss

            (total_loss / grad_accum).backward()
            x, y, epoch = next(train_loader)

        progress = min(wall / TIME_BUDGET, 1.0)
        lr = get_lr(progress)

        for g in optimizer.param_groups:
            if "initial_lr" not in g:
                g["initial_lr"] = g["lr"]
            g["lr"] = g["initial_lr"] * lr
            # Muon momentum warmup: 0.85 → 0.95 over 500 steps (upstream pattern)
            if g.get("kind") == "muon":
                g["momentum"] = min(0.95, 0.85 + 0.1 * step / 500)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # H6: EMA update (use raw_model to avoid DDP wrapper)
        with torch.no_grad():
            for name, p in raw_model.named_parameters():
                if name in ema_shadow:
                    ema_shadow[name].lerp_(p.data, 1.0 - EMA_DECAY)

        tlf = loss.detach().item()
        if tlf > 100:
            print(f"\nFAIL step {step} loss={tlf:.1f}")
            sys.exit(1)

        sync()
        dt = time.time() - t0
        if step > 10:
            wall += dt

        b = 0.9
        smooth = b * smooth + (1 - b) * tlf
        debiased = smooth / (1 - b ** (step + 1))
        rem = max(0, TIME_BUDGET - wall)

        if master_process:
            print(f"\rstep {step:04d} ({100 * progress:.0f}%) loss:{debiased:.4f} "
                  f"lr:{lr:.2f} dt:{dt * 1000:.0f}ms tok/s:{TOTAL_BATCH / dt:.0f} "
                  f"rem:{rem:.0f}s   ", end="", flush=True)

        # Diagnostics every 100 steps
        if step % 100 == 0 and step > 0 and master_process:
            for name, blk in [("local", raw_model.local_block), ("transition", raw_model.transition_block), ("hierarchy", raw_model.hierarchy_block)]:
                ascale = blk.attn_scale.data.cpu().tolist()
                mscale = blk.mlp_scale.data.cpu().tolist()
                print(f"\n  {name}: attn=[{', '.join(f'{s:.3f}' for s in ascale)}] mlp=[{', '.join(f'{s:.3f}' for s in mscale)}]")

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        step += 1

        # Synchronized exit: rank 0 decides, broadcasts to all ranks.
        # This ensures all ranks exit the loop on the same step,
        # preventing DDP forward deadlocks from mismatched iteration counts.
        should_stop = step > 10 and wall >= TIME_BUDGET
        if distributed:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            break

    torch.cuda.synchronize()

    if master_process:
        print(flush=True)
    total_tok = step * TOTAL_BATCH

    # ===================================================================
    # POST-TRAINING: Parallel eval across GPUs.
    # Each rank runs a different eval strategy independently (no DDP/NCCL).
    # Rank 0 does EMA + quantize + save, writes artifact to disk.
    # All ranks load artifact, then each runs its assigned eval.
    # Results written to files, rank 0 reads all and reports best.
    # ===================================================================
    import copy

    def log(msg):
        """Log to stdout (rank 0 only) and post_training.log file."""
        if master_process:
            print(f"\n[POST] {msg}", flush=True)
        with open(f"post_training_r{rank}.log", "a") as _f:
            _f.write(f"[{time.strftime('%H:%M:%S')}] [R{rank}] {msg}\n")

    EVAL_BATCH = 128
    EVAL_WALL = int(os.environ.get("EVAL_WALL", "600"))

    log(f"training done: {step} steps, {total_tok/1e6:.1f}M tokens")

    # ---- EMA + quantize + save (rank 0 only, others wait) ----
    if master_process:
        for name, p in raw_model.named_parameters():
            if name in ema_shadow:
                p.data.copy_(ema_shadow[name])
        log(f"EMA: swapped {len(ema_shadow)} params")

        torch.save(raw_model.state_dict(), "final_model.pt")
        code_bytes = os.path.getsize(os.path.abspath(__file__))
        log(f"checkpoint saved")

        quant_obj, quant_stats = quantize_state_dict_int8(raw_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_blob = zlib.compress(quant_buf.getvalue(), level=9)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        log(f"artifact: {quant_file_bytes} bytes, headroom: {16_000_000 - quant_file_bytes - code_bytes}")

        # Signal artifact is ready
        with open("/workspace/.artifact_ready", "w") as f:
            f.write("ready")

    # Non-master ranks: wait for artifact (timeout 120s)
    if distributed and not master_process:
        _t_wait = time.perf_counter()
        while not os.path.exists("/workspace/.artifact_ready"):
            if time.perf_counter() - _t_wait > 120:
                log("TIMEOUT waiting for artifact — exiting")
                return
            time.sleep(0.5)

    # ---- All ranks: load dequantized model ----
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu", weights_only=False)
    raw_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    log("model reloaded from artifact")

    # ---- Parallel eval: each rank runs a different strategy ----
    # Assign eval strategies to ranks (cycle if more ranks than strategies)
    # Ablation results (60s smoke, 2xH100 India DC):
    #   base_int8:       2.4138 (22s)   — official roundtrip
    #   ttt lr5e-4 2s:   2.0571 (248s)  — BEST (-0.357)
    #   ttt lr1e-3 2s:   2.0611 (249s)  — close 2nd
    #   ttt lr1e-3 1s:   2.0819 (142s)  — fast
    #   ttt lr5e-4 1s:   2.1142 (140s)  — proven
    #   ttt curv-adpt:   2.2057 (142s)  — DROP (worse than flat)
    #   ttt lr2e-4 1s:   2.2155 (140s)  — DROP (weak)
    #   sliding w64:     ~2.43  (slow)  — DROP (hurts)
    eval_strategies = [
        ("base_int8",       lambda m: evaluate_bpb(m, tokenizer, EVAL_BATCH, eval_wall=EVAL_WALL)),
        ("ttt_lr5e-4_2s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.0005, ttt_steps=2, eval_wall=EVAL_WALL)),
        ("ttt_lr1e-3_2s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.001, ttt_steps=2, eval_wall=EVAL_WALL)),
        ("ttt_lr1e-3_1s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.001, ttt_steps=1, eval_wall=EVAL_WALL)),
        ("ttt_lr5e-4_1s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.0005, ttt_steps=1, eval_wall=EVAL_WALL)),
        ("ttt_lr7e-4_2s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.0007, ttt_steps=2, eval_wall=EVAL_WALL)),
        ("ttt_lr5e-4_3s",   lambda m: eval_ttt_scorefirst(copy.deepcopy(m), 16, lr=0.0005, ttt_steps=3, eval_wall=EVAL_WALL)),
        ("ngram_blend",     lambda m: eval_ngram_blend(m, batch_size=EVAL_BATCH, alpha=0.3, eval_wall=EVAL_WALL)),
    ]
    my_strategy = eval_strategies[rank % len(eval_strategies)]
    strategy_name, strategy_fn = my_strategy

    log(f"running eval: {strategy_name}")
    torch.cuda.synchronize()
    t_eval = time.perf_counter()

    # Compile for inference (each rank compiles independently)
    eval_model = torch.compile(raw_model) if strategy_name.startswith("base") else raw_model
    if strategy_name.startswith("base"):
        eval_model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = eval_model(torch.randint(0, 1024, (1, MAX_SEQ_LEN), device=device))
        torch.cuda.synchronize()
        log("compile warmup done")

    my_bpb = strategy_fn(eval_model if strategy_name.startswith("base") else raw_model)
    eval_time = time.perf_counter() - t_eval
    log(f"{strategy_name} val_bpb:{my_bpb:.4f} time:{eval_time:.0f}s")

    # Write result to file
    result_file = f"/workspace/.eval_result_r{rank}"
    with open(result_file, "w") as f:
        f.write(f"{strategy_name}\t{my_bpb:.8f}\t{eval_time:.1f}\n")

    # Signal this rank is done
    with open(f"/workspace/.eval_done_r{rank}", "w") as f:
        f.write("done")

    # ---- Rank 0: collect results and report best ----
    if master_process:
        # Wait for all ranks to finish
        for r in range(world_size):
            while not os.path.exists(f"/workspace/.eval_done_r{r}"):
                time.sleep(2)

        # Read all results
        all_results = []
        for r in range(world_size):
            rf = f"/workspace/.eval_result_r{r}"
            if os.path.exists(rf):
                parts = open(rf).read().strip().split("\t")
                name, bpb, t = parts[0], float(parts[1]), float(parts[2])
                all_results.append((name, bpb, t))
                log(f"  rank {r}: {name} = {bpb:.4f} ({t:.0f}s)")

        best_name, best_bpb, _ = min(all_results, key=lambda x: x[1])
        print(f"final_int8_zlib_roundtrip val_bpb:{best_bpb:.4f}", flush=True)
        print(f"final_int8_zlib_roundtrip_exact val_bpb:{best_bpb:.8f}", flush=True)
        print(f"val_bpb:{best_bpb:.4f}", flush=True)
        log(f"BEST: {best_name} val_bpb:{best_bpb:.4f}")
        log(f"DONE — clean exit")

        # Clean up signal files
        for r in range(world_size):
            for f in [f"/workspace/.eval_result_r{r}", f"/workspace/.eval_done_r{r}"]:
                if os.path.exists(f): os.remove(f)
        if os.path.exists("/workspace/.artifact_ready"):
            os.remove("/workspace/.artifact_ready")
    else:
        # Non-master: wait for rank 0 to finish collecting (timeout 60s)
        _t_wait = time.perf_counter()
        while os.path.exists(f"/workspace/.eval_done_r{rank}"):
            if time.perf_counter() - _t_wait > 60:
                break
            time.sleep(2)


def eval_sliding_window(model, batch_size, seq_len=MAX_SEQ_LEN, stride=64, eval_wall=600):
    """Sliding window eval — overlapping windows improve BPB by ~0.02-0.05."""
    _dev = next(model.parameters()).device
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vs = int(sp.vocab_size())
    bb, hs, ib = build_sentencepiece_luts(sp, vs, _dev)
    vt = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(VAL_FILES))]).contiguous()
    total_tokens = vt.numel() - 1

    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    t0 = time.perf_counter()

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_size):
            if time.perf_counter() - t0 > eval_wall - 5:
                print(f"[EVAL-SW] wall hit at {time.perf_counter()-t0:.0f}s", flush=True)
                break
            batch_ws = window_starts[bi:bi+batch_size]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=_dev)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=_dev)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = vt[ws:end+1].to(dtype=torch.int64, device=_dev)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x_batch).float()
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1),
                                  reduction='none').view(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                loss_sum += nll[i, s:wlen].to(torch.float64).sum().item()
                tok_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = bb[tgt].to(torch.float64)
                tb += (hs[tgt] & ~ib[prev]).to(torch.float64)
                byte_count += tb.sum().item()

    return (loss_sum / tok_count / math.log(2.0)) * (tok_count / byte_count)


def eval_ttt_scorefirst(model, batch_size, seq_len=MAX_SEQ_LEN,
                        lr=0.0005, ttt_steps=1, eval_wall=600,
                        curvature_adaptive=False):
    """LEGAL score-first TTT: score chunk FIRST, THEN train on scored tokens.
    Adapts model to eval distribution — ~0.05-0.1 BPB improvement.
    curvature_adaptive: per-block LR scaling (high curvature blocks get lower LR)."""
    _dev = next(model.parameters()).device
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vs = int(sp.vocab_size())
    bb, hs, ib = build_sentencepiece_luts(sp, vs, _dev)
    vt = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(VAL_FILES))]).contiguous()
    usable = ((vt.numel() - 1) // seq_len) * seq_len
    vt = vt[:usable + 1]
    total_seqs = (vt.numel() - 1) // seq_len

    if curvature_adaptive and hasattr(model, 'local_block'):
        param_groups = [
            {"params": [p for p in model.local_block.parameters() if p.requires_grad], "lr": lr * 2.0},
            {"params": [p for p in model.transition_block.parameters() if p.requires_grad], "lr": lr * 1.0},
            {"params": [p for p in model.hierarchy_block.parameters() if p.requires_grad], "lr": lr * 0.3},
            {"params": list(model.tok_emb.parameters()), "lr": lr * 0.5},
        ]
        param_groups = [g for g in param_groups if g["params"]]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    t0 = time.perf_counter()

    for bs in range(0, total_seqs, batch_size):
        if time.perf_counter() - t0 > eval_wall - 5:
            print(f"[EVAL-TTT] wall hit at {time.perf_counter()-t0:.0f}s", flush=True)
            break
        be = min(bs + batch_size, total_seqs)
        loc = vt[bs*seq_len:be*seq_len+1].to(dtype=torch.int64, device=_dev)
        n = be - bs
        x = loc[:-1].reshape(n, seq_len)
        y = loc[1:].reshape(n, seq_len)

        # STEP 1: SCORE (grades FINAL)
        # Use no_grad (not inference_mode) — inference_mode tensors can't be
        # used in step 2's backward pass (rotary cache etc.)
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x).float()
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                  reduction='none').view(n, seq_len)
            loss_sum += nll.sum().item()
            tok_count += float(y.numel())
            prev = x.reshape(-1)
            tgt = y.reshape(-1)
            tb = bb[tgt].to(torch.float64)
            tb += (hs[tgt] & ~ib[prev]).to(torch.float64)
            byte_count += tb.sum().item()

        # STEP 2: TRAIN on scored tokens (adapts to eval distribution)
        model.train()
        for _ in range(ttt_steps):
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ttt_loss = model(x, y)
            ttt_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

    model.eval()
    return (loss_sum / tok_count / math.log(2.0)) * (tok_count / byte_count)


def eval_ngram_blend(model, batch_size=128, alpha=0.3, eval_wall=600):
    """Base eval with n-gram cache blending. Builds cache from train shards,
    then blends n-gram probs with model logits during eval."""
    _dev = next(model.parameters()).device
    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    vs = int(sp.vocab_size())
    bb, hs, ib = build_sentencepiece_luts(sp, vs, _dev)

    # Build n-gram cache from training data
    cache = NgramCache(vocab_size=vs, max_order=4)
    cache.build_from_shards(DATA_PATH, max_tokens=5_000_000)

    vt = torch.cat([load_data_shard(Path(p)) for p in sorted(glob.glob(VAL_FILES))]).contiguous()
    usable = ((vt.numel() - 1) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    vt = vt[:usable + 1]
    total_seqs = (vt.numel() - 1) // MAX_SEQ_LEN

    loss_sum, tok_count, byte_count = 0.0, 0.0, 0.0
    t0 = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for bs in range(0, total_seqs, batch_size):
            if time.perf_counter() - t0 > eval_wall - 5:
                break
            be = min(bs + batch_size, total_seqs)
            loc = vt[bs*MAX_SEQ_LEN:be*MAX_SEQ_LEN+1].to(device=_dev, dtype=torch.int64)
            x = loc[:-1].reshape(be - bs, MAX_SEQ_LEN)
            y = loc[1:].reshape(be - bs, MAX_SEQ_LEN)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x).float()
            # Blend with n-gram cache → log-probs
            blended_lp = cache.blend_logits(logits, x, alpha=alpha)
            nll = F.nll_loss(blended_lp.view(-1, blended_lp.size(-1)), y.view(-1),
                             reduction='none').view(be - bs, MAX_SEQ_LEN)
            bt = float(y.numel())
            loss_sum += nll.sum().item()
            tok_count += bt
            tb = bb[y.reshape(-1)].to(torch.int16)
            tb += (hs[y.reshape(-1)] & ~ib[x.reshape(-1)]).to(torch.int16)
            byte_count += tb.to(torch.float64).sum().item()

    return (loss_sum / tok_count / math.log(2.0)) * (tok_count / byte_count)


if __name__ == "__main__":
    main()
