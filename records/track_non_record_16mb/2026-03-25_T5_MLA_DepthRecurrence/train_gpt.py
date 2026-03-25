"""
T5 Parameter Golf Entry: MLA + DeepNorm + Mixed-Precision Depth.

Variants (VARIANT env var):
  looped40 — 8 unique blocks × 4 phases × 5 reps = 40 depth at d=512 MLP3x
             Phase-based recurrence: [AB]×5 [CD]×5 [EF]×5 [GH]×5
             24.3M params, FP4(phase1-2)+Int6(phase3-4), ~13 MB
  deep20   — 20L x 384d x MLP3x, FP4(16)+Int6(4) graduated (33.6M, ~15.7 MB)
  test     — 4L x 384d x MLP2x, Int6 (for local smoke tests)

Key innovations:
  - MLA: K/V low-rank compression, 20% fewer attention params per layer
  - Graduated precision: FP4 for early layers, Int6 for final layers
  - DeepNorm init: stable training at 20-40 layer depth

Based on the OpenAI Parameter Golf baseline. Requires FineWeb data shards.

Usage:
  # Local smoke test (Mac/CPU)
  VARIANT=test DEVICE=cpu ITERATIONS=10 VAL_LOSS_EVERY=0 python3 train_gpt.py

  # Full H100 run
  VARIANT=deep20 torchrun --standalone --nproc_per_node=8 train_gpt.py
  VARIANT=ultra40 torchrun --standalone --nproc_per_node=8 train_gpt.py
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import struct
import time
import uuid
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

class Hyperparameters:
    variant = os.environ.get("VARIANT", "competitive")

    # Data
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4())[:8])
    seed = int(os.environ.get("SEED", 1337))

    # Architecture — set by variant
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    seq_len = int(os.environ.get("SEQ_LEN", 1024))

    if variant == "looped40":
        # 8 unique × 4 phases × 5 reps = 40 depth at FULL d=512
        # Phase-based: [AB]×5 [CD]×5 [EF]×5 [GH]×5
        # 24.3M unique params, ~13 MB with graduated precision
        n_layer = int(os.environ.get("N_LAYER", 40))
        n_embd = int(os.environ.get("N_EMBD", 512))
        n_head = int(os.environ.get("N_HEAD", 8))
        kv_lora_rank = int(os.environ.get("KV_LORA_RANK", 64))
        mlp_mult = int(os.environ.get("MLP_MULT", 3))
        n_unique_blocks = int(os.environ.get("N_UNIQUE", 8))
        n_phases = int(os.environ.get("N_PHASES", 4))
        # Graduated: Phase 1-2 FP4 (coarse), Phase 3-4 Int6 (fine)
        precision_schedule = [(4, 4), (6, 4)]  # per UNIQUE block
    elif variant == "deep20":
        # 20L x 384d x MLP3x — graduated FP4(16)+Int6(4)
        n_layer = int(os.environ.get("N_LAYER", 20))
        n_embd = int(os.environ.get("N_EMBD", 384))
        n_head = int(os.environ.get("N_HEAD", 6))
        kv_lora_rank = int(os.environ.get("KV_LORA_RANK", 48))
        mlp_mult = int(os.environ.get("MLP_MULT", 3))
        n_unique_blocks = n_layer
        n_phases = 1
        precision_schedule = [(4, 16), (6, 4)]
    else:  # test (local smoke test)
        n_layer = int(os.environ.get("N_LAYER", 4))
        n_embd = int(os.environ.get("N_EMBD", 384))
        n_head = int(os.environ.get("N_HEAD", 6))
        kv_lora_rank = int(os.environ.get("KV_LORA_RANK", 48))
        mlp_mult = int(os.environ.get("MLP_MULT", 2))
        n_unique_blocks = n_layer
        n_phases = 1
        precision_schedule = [(6, 4)]

    head_dim = n_embd // n_head
    qk_rope_dim = max(16, head_dim // 4)
    qk_nope_dim = head_dim - qk_rope_dim
    v_head_dim = head_dim
    d_ff = n_embd * mlp_mult

    # Training
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmup_iters = int(os.environ.get("WARMUP_ITERS", 256))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3072))
    max_wallclock = int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600))

    # Optimizer
    muon_lr = float(os.environ.get("MUON_LR", 0.02))
    adamw_lr = float(os.environ.get("ADAMW_LR", 3e-4))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.0))

    # QAT + EMA
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.15))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.95))
    ema_start_step = int(os.environ.get("EMA_START_STEP", 500))

    # Eval
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    max_eval_tokens = int(os.environ.get("MAX_EVAL_TOKENS", 0))  # 0 = all (for submission)

    # Stability (from main model)
    z_loss_weight = float(os.environ.get("Z_LOSS_WEIGHT", 1e-4))  # PaLM/Gemini style
    logit_clamp = float(os.environ.get("LOGIT_CLAMP", 30.0))
    qk_clip_tau = float(os.environ.get("QK_CLIP_TAU", 100.0))

    # Quant bits for serialization (default from precision_schedule)
    quant_bits = int(os.environ.get("QUANT_BITS", precision_schedule[0][0]))

H = Hyperparameters

# =============================================================================
# DISTRIBUTED SETUP (deferred to main)
# =============================================================================

def setup_distributed():
    global ddp, rank, local_rank, world_size, device, master
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        force_device = os.environ.get("DEVICE", "")
        if force_device:
            device = torch.device(force_device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        master = True

    if master:
        print(f"[T5 Entry] variant={H.variant} n_layer={H.n_layer} n_embd={H.n_embd} "
              f"n_head={H.n_head} kv_rank={H.kv_lora_rank} mlp_mult={H.mlp_mult}")
        print(f"[T5 Entry] device={device} world_size={world_size}")

# Module-level defaults for import
ddp, rank, local_rank, world_size, master = False, 0, 0, 1, True
device = torch.device("cpu")

# =============================================================================
# DATA LOADING
# =============================================================================

def load_shard_tokens(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")

class DataLoader:
    def __init__(self, pattern: str, batch_tokens: int, seq_len: int, rank: int, world_size: int):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"No files found for pattern: {pattern}"
        self.batch_tokens = batch_tokens
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_tokens // seq_len
        assert self.batch_size > 0
        self._shard_idx = 0
        self._pos = rank * self.batch_size * seq_len
        self._data = load_shard_tokens(self.files[0])

    def next_batch(self) -> tuple[Tensor, Tensor]:
        B, T = self.batch_size, self.seq_len
        needed = B * T + 1  # +1 for targets

        # Advance shard if needed
        while self._pos + needed > len(self._data):
            self._shard_idx = (self._shard_idx + 1) % len(self.files)
            self._data = load_shard_tokens(self.files[self._shard_idx])
            self._pos = self.rank * needed

        buf = torch.from_numpy(self._data[self._pos : self._pos + needed].astype(np.int64))
        buf = buf.clamp(max=H.vocab_size - 1)  # OOV tokens → last vocab entry
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self._pos += needed * self.world_size
        return x.to(device), y.to(device)

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_len: int, theta: float = 500000.0) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to last dimension of x. x: (..., rope_dim)."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MLAttention(nn.Module):
    """Multi-Head Latent Attention with decoupled RoPE (DeepSeek-V2 style).

    Compresses K/V through a low-rank bottleneck:
      x → kv_a_proj → [latent, k_rope_raw]
      latent → kv_a_norm → kv_b_proj → [k_nope, v]
      k_rope_raw → apply_rope → k_rope

    Then attention: score = [q_nope, q_rope] @ [k_nope, k_rope]^T
    """

    def __init__(self):
        super().__init__()
        d = H.n_embd
        nh = H.n_head
        r = H.kv_lora_rank
        rope_d = H.qk_rope_dim
        nope_d = H.qk_nope_dim
        v_d = H.v_head_dim

        # Q projection → n_head * head_dim
        self.q_proj = nn.Linear(d, nh * (nope_d + rope_d), bias=False)
        self.q_norm = RMSNorm(nope_d + rope_d)

        # KV compression: d → rank + rope_dim
        self.kv_a_proj = nn.Linear(d, r + rope_d, bias=False)
        self.kv_a_norm = RMSNorm(r)

        # KV expansion: rank → n_head * (nope_dim + v_dim)
        self.kv_b_proj = nn.Linear(r, nh * (nope_d + v_d), bias=False)

        # Output projection
        self.o_proj = nn.Linear(nh * v_d, d, bias=False)

        self.nh = nh
        self.rope_d = rope_d
        self.nope_d = nope_d
        self.v_d = v_d
        self.r = r

        # QK-Clip: capture max attention logits per head for post-step rescaling
        # Only on CUDA (materializing full attention matrix is too slow on MPS)
        self._capture_max_logits = False  # enabled at runtime by trainer
        self._current_max_logits: torch.Tensor | None = None

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape

        # Q: project and split into nope + rope
        q = self.q_proj(x).view(B, T, self.nh, self.nope_d + self.rope_d)
        q = self.q_norm(q)
        q_nope, q_rope = q.split([self.nope_d, self.rope_d], dim=-1)

        # KV: compress → normalize → expand
        kv_a = self.kv_a_proj(x)
        kv_latent, k_rope_raw = kv_a.split([self.r, self.rope_d], dim=-1)
        kv_latent = self.kv_a_norm(kv_latent)

        kv_b = self.kv_b_proj(kv_latent).view(B, T, self.nh, self.nope_d + self.v_d)
        k_nope, v = kv_b.split([self.nope_d, self.v_d], dim=-1)

        # RoPE on rope parts (k_rope shared across heads)
        k_rope = k_rope_raw.unsqueeze(2).expand(-1, -1, self.nh, -1)
        cos_t = cos[:T].unsqueeze(0).unsqueeze(2)  # (1, T, 1, rope_d//2)
        sin_t = sin[:T].unsqueeze(0).unsqueeze(2)
        q_rope = apply_rope(q_rope, cos_t, sin_t)
        k_rope = apply_rope(k_rope, cos_t, sin_t)

        # Concatenate nope + rope for standard dot-product attention
        q_full = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)  # (B, nh, T, hd)
        k_full = torch.cat([k_nope, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)  # (B, nh, T, v_d)

        # Scaled dot-product attention (uses Flash Attention when available)
        # QK-Clip: capture max attention logits before softmax for post-step rescaling
        if self._capture_max_logits and self.training:
            with torch.no_grad():
                scale = (self.nope_d + self.rope_d) ** -0.5
                raw_scores = (q_full @ k_full.transpose(-2, -1)) * scale
                self._current_max_logits = raw_scores.detach().amax(dim=(-2, -1))  # (B, nh) → (nh,) via mean over B
                self._current_max_logits = self._current_max_logits.mean(dim=0)  # (nh,)

        out = F.scaled_dot_product_attention(q_full, k_full, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh * self.v_d)
        return self.o_proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self):
        super().__init__()
        d, d_ff = H.n_embd, H.d_ff
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(H.n_embd)
        self.attn = MLAttention()
        self.ln2 = RMSNorm(H.n_embd)
        self.mlp = SwiGLUMLP()

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Transformer with phase-based depth recurrence.

    Instead of n_layer unique blocks, uses n_unique_blocks blocks arranged
    in phases. Each phase has a pair of unique blocks repeated multiple times.
    This allows deep processing (40+ effective layers) with few parameters.

    Example (n_unique=8, n_layer=40, n_phases=4):
      Phase 1 (lexical):    [A B] × 5 = 10 effective layers
      Phase 2 (syntactic):  [C D] × 5 = 10 effective layers
      Phase 3 (semantic):   [E F] × 5 = 10 effective layers
      Phase 4 (prediction): [G H] × 5 = 10 effective layers
      Total: 8 unique blocks, 40 effective depth

    Weight sharing benefits:
      - Each weight set gets gradient signal from multiple positions
      - Recurrence within a phase = iterative refinement
      - Massive parameter savings → wider model or better precision
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(H.vocab_size, H.n_embd)

        # Unique blocks (the actual trainable parameters)
        n_unique = getattr(H, "n_unique_blocks", H.n_layer)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(n_unique)])

        # Build the execution schedule: which block index at each depth position
        n_phases = getattr(H, "n_phases", 1)
        self.schedule = self._build_schedule(n_unique, H.n_layer, n_phases)

        self.ln_f = RMSNorm(H.n_embd)

        # Precompute RoPE
        cos, sin = precompute_rope(H.qk_rope_dim, H.seq_len + 64)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    @staticmethod
    def _build_schedule(n_unique: int, n_depth: int, n_phases: int) -> list[int]:
        """Build execution schedule: block index for each depth position.

        Phase-based: divide unique blocks into phases, repeat within each phase.
        E.g., 8 unique, 40 depth, 4 phases → [0,1,0,1,...] [2,3,2,3,...] ...
        """
        if n_phases <= 1 or n_unique == n_depth:
            # No recurrence: each position uses its own unique block
            return list(range(min(n_unique, n_depth)))

        blocks_per_phase = n_unique // n_phases
        depth_per_phase = n_depth // n_phases

        schedule = []
        for phase in range(n_phases):
            phase_block_start = phase * blocks_per_phase
            phase_blocks = list(range(phase_block_start, phase_block_start + blocks_per_phase))
            # Repeat the phase blocks to fill the depth
            for i in range(depth_per_phase):
                schedule.append(phase_blocks[i % blocks_per_phase])

        # Handle remainder (if n_depth not divisible by n_phases)
        while len(schedule) < n_depth:
            schedule.append(schedule[-1])

        return schedule

    def _init_weights(self):
        # DeepNorm beta based on EFFECTIVE depth (not unique blocks)
        beta = (8.0 * H.n_layer) ** -0.25
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        for block in self.blocks:
            block.attn.o_proj.weight.data.mul_(beta)
            block.mlp.w_down.weight.data.mul_(beta)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        # Execute schedule: each depth position uses its assigned block
        for block_idx in self.schedule:
            x = self.blocks[block_idx](x, cos, sin)
        x = self.ln_f(x)
        return F.linear(x, self.embed.weight)

# =============================================================================
# MUON OPTIMIZER (Momentum + Newton-Schulz orthogonalization for 2D params)
# =============================================================================

@torch.no_grad()
def newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate matrix sign/polar decomposition via Newton-Schulz."""
    a, b, c = 3.4445, -4.7750, 2.0315
    # Use float32 on MPS (no bfloat16 support), bfloat16 on CUDA
    dtype = torch.bfloat16 if G.device.type == "cuda" else torch.float32
    X = G.to(dtype)
    nrm = X.norm() + eps
    X = X / nrm
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2D weight matrices, AdamW for everything else."""

    def __init__(self, muon_params, adamw_params, lr=0.02, momentum=0.95,
                 adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.0):
        defaults = dict(lr=lr, momentum=momentum)
        # Separate param groups
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "lr": lr, "momentum": momentum, "is_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "lr": adamw_lr, "is_muon": False})
        super().__init__(groups, defaults)
        self.adamw_betas = adamw_betas
        self.adamw_wd = adamw_wd

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            is_muon = group.get("is_muon", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if is_muon:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                if is_muon:
                    # Newton-Schulz orthogonalization + momentum
                    buf = state["momentum_buffer"]
                    g_orth = newton_schulz_5(g.view(g.shape[0], -1)).view_as(g)
                    buf.mul_(group["momentum"]).add_(g_orth)
                    p.add_(buf, alpha=-lr)
                else:
                    # AdamW
                    b1, b2 = self.adamw_betas
                    wd = self.adamw_wd
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    t = state["step"]

                    if wd > 0:
                        p.mul_(1.0 - lr * wd)
                    exp_avg.mul_(b1).add_(g, alpha=1 - b1)
                    exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1.0 - b1 ** t
                    bc2 = 1.0 - b2 ** t
                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(1e-8)
                    p.addcdiv_(exp_avg, denom, value=-step_size)

# =============================================================================
# QK-CLIP: Post-optimizer attention score clipping (Kimi K2, Section 3.3)
# =============================================================================

@torch.no_grad()
def qk_clip_step(model: nn.Module, tau: float = 100.0, alpha: float = 0.5) -> int:
    """Rescale Q/K weights for MLA heads where attention scores exceed tau.

    Called after optimizer.step(). Per-head rescaling:
      gamma_h = min(1, tau / max_logit_h)
      Q_nope *= gamma^alpha,  Q_rope *= gamma
      K_nope *= gamma^(1-alpha),  V: untouched
    """
    total_clipped = 0
    for module in model.modules():
        if not isinstance(module, MLAttention):
            continue
        if module._current_max_logits is None:
            continue

        max_logits = module._current_max_logits
        gamma = torch.clamp(tau / max_logits, max=1.0)
        clip_mask = gamma < 1.0

        if not clip_mask.any():
            module._current_max_logits = None
            continue

        nope_d = module.nope_d
        rope_d = module.rope_d
        v_d = module.v_d
        hd = nope_d + rope_d
        kv_per_head = nope_d + v_d

        for h in range(module.nh):
            if not clip_mask[h]:
                continue
            g = gamma[h]
            # Q rescaling
            q_start = h * hd
            module.q_proj.weight.data[q_start:q_start + nope_d] *= g ** alpha
            module.q_proj.weight.data[q_start + nope_d:q_start + hd] *= g
            # K rescaling (in kv_b_proj)
            kv_start = h * kv_per_head
            module.kv_b_proj.weight.data[kv_start:kv_start + nope_d] *= g ** (1.0 - alpha)
            # V: untouched
            total_clipped += 1

        module._current_max_logits = None
    return total_clipped

# =============================================================================
# FP8 TRAINING: All persistent state in FP8, stochastic rounding
# =============================================================================

@torch.no_grad()
def stochastic_round_fp8(
    x: Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> Tensor:
    """Stochastic rounding to FP8 via dither noise injection.

    Adds uniform noise scaled to the FP8 ULP (unit in last place)
    before deterministic rounding. This is an unbiased estimator
    that converges to the true value over many updates.

    E4M3: 3 mantissa bits → ULP ≈ |x| * 2^(-3) = |x| / 8
    E5M2: 2 mantissa bits → ULP ≈ |x| * 2^(-2) = |x| / 4
    """
    mantissa_bits = 3 if "e4m3" in str(dtype) else 2
    step = x.abs().clamp(min=1e-12) * (2.0 ** (-mantissa_bits))
    noise = (torch.rand_like(x) - 0.5) * step
    return (x + noise).to(dtype)


class FP8TrainingState:
    """Pure FP8 training state — nothing above FP8 persists.

    All master weights stored as FP8 E4M3 (1 byte/param).
    All optimizer momentum stored as FP8 E5M2 (1 byte/param).
    Transient FP16 casts only during optimizer math (not persisted).

    Before each forward: dequantize FP8 → model params (BF16).
    After each optimizer step: stochastic round → FP8.
    """

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        if not self.enabled:
            return

        self.weight_fp8: dict[str, Tensor] = {}
        for name, p in model.named_parameters():
            self.weight_fp8[name] = p.data.to(torch.float8_e4m3fn)

    def load_weights(self, model: nn.Module) -> None:
        """Before forward: dequantize FP8 weights into model params."""
        if not self.enabled:
            return
        for name, p in model.named_parameters():
            if name in self.weight_fp8:
                p.data.copy_(self.weight_fp8[name].to(p.dtype))

    def save_weights(self, model: nn.Module) -> None:
        """After optimizer step: stochastic round model params → FP8."""
        if not self.enabled:
            return
        for name, p in model.named_parameters():
            if name in self.weight_fp8:
                self.weight_fp8[name] = stochastic_round_fp8(
                    p.data, torch.float8_e4m3fn
                )

    def save_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        """Stochastic round optimizer state (momentum, variance) to FP8."""
        if not self.enabled:
            return
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                if "momentum_buffer" in state:
                    buf = state["momentum_buffer"]
                    fp8 = stochastic_round_fp8(buf, torch.float8_e5m2)
                    state["momentum_buffer"] = fp8.to(buf.dtype)
                if "exp_avg" in state:
                    ea = state["exp_avg"]
                    fp8 = stochastic_round_fp8(ea, torch.float8_e5m2)
                    state["exp_avg"] = fp8.to(ea.dtype)
                if "exp_avg_sq" in state:
                    eas = state["exp_avg_sq"]
                    fp8 = stochastic_round_fp8(eas, torch.float8_e5m2)
                    state["exp_avg_sq"] = fp8.to(eas.dtype)


# =============================================================================
# INT/FP4 QAT (Fake Quantization with Straight-Through Estimator)
# =============================================================================

def fake_quantize(w: Tensor, bits: int = 6) -> Tensor:
    """Simulate N-bit symmetric quantization during forward pass (STE)."""
    qmax = (1 << (bits - 1)) - 1  # 31 for int6, 7 for int4
    scale = w.detach().abs().amax() / max(qmax, 1)
    if scale == 0:
        return w
    w_q = (w / scale).round().clamp(-qmax - 1, qmax)
    return (w_q * scale - w).detach() + w


def get_layer_bits(layer_idx: int, schedule: list) -> int:
    """Get quantization bits for a layer based on precision schedule.

    schedule: [(bits, count), ...] e.g. [(4, 30), (6, 10)]
    """
    offset = 0
    for bits, count in schedule:
        if layer_idx < offset + count:
            return bits
        offset += count
    return schedule[-1][0]  # fallback to last


class QATWrapper(nn.Module):
    """Wraps a model to apply per-layer fake quantization during forward.

    Supports graduated precision: different bits for different layers,
    controlled by the precision_schedule in Hyperparameters.
    """

    def __init__(self, model: nn.Module, schedule: list):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.enabled = False
        self._handles = []

    def enable(self):
        if self.enabled:
            return
        self.enabled = True
        # Per-layer quantization hooks
        for block_idx, block in enumerate(self.model.blocks):
            bits = get_layer_bits(block_idx, self.schedule)
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    h = m.register_forward_pre_hook(
                        self._make_hook(bits)
                    )
                    self._handles.append(h)

    def disable(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.enabled = False

    @staticmethod
    def _make_hook(bits: int):
        def hook(module, _input):
            module.weight.data = fake_quantize(module.weight.data, bits)
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.95):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(d).add_(v, alpha=1.0 - d)

    def state_dict(self):
        return self.shadow

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)

# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, val_files: list[str], tokenizer_path: str,
             batch_tokens: int, seq_len: int,
             max_eval_tokens: int = 0) -> tuple[float, float]:
    """Compute validation loss and bits per byte (BPB).

    Args:
        max_eval_tokens: If > 0, evaluate only this many tokens (for fast local testing).
                         If 0, evaluate on ALL validation data (for final submission).
    """
    model.eval()
    sp = spm.SentencePieceProcessor(tokenizer_path)
    total_loss = 0.0
    total_tokens = 0
    B = max(1, batch_tokens // seq_len)

    for vf in sorted(glob.glob(val_files) if isinstance(val_files, str) else val_files):
        data = load_shard_tokens(vf)
        n_tokens = (len(data) - 1) // (B * seq_len) * (B * seq_len)
        if n_tokens == 0:
            continue

        for start in range(0, n_tokens, B * seq_len):
            end = start + B * seq_len + 1
            if end > len(data):
                break
            buf = torch.from_numpy(data[start:end].astype(np.int64)).to(device)
            buf = buf.clamp(max=H.vocab_size - 1)
            x = buf[:-1].view(B, seq_len)
            y = buf[1:].view(B, seq_len)

            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, H.vocab_size), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += B * seq_len

            if max_eval_tokens > 0 and total_tokens >= max_eval_tokens:
                break
        if max_eval_tokens > 0 and total_tokens >= max_eval_tokens:
            break

    if total_tokens == 0:
        model.train()
        return 0.0, 0.0

    avg_loss = total_loss / total_tokens  # nats per token

    # Compute bytes: decode tokens back to bytes to get exact byte count
    # Approximation: use average bytes per token from tokenizer
    # More precise: total_bytes = sum(len(sp.decode(tok)) for tok in all_tokens)
    # For speed, use the ratio from a sample
    sample_ids = list(range(min(1000, H.vocab_size)))
    sample_text = sp.decode(sample_ids)
    bytes_per_token = len(sample_text.encode("utf-8")) / max(len(sample_ids), 1)
    # Fallback: if tokenizer has byte coverage info
    if bytes_per_token < 0.5:
        bytes_per_token = 1.0

    bpb = avg_loss / bytes_per_token / math.log(2)
    model.train()
    return avg_loss, bpb

# =============================================================================
# SERIALIZATION (Quantize + Compress)
# =============================================================================

def quantize_state_dict(state_dict: dict, schedule: list, n_layers: int) -> dict:
    """Quantize weights with per-layer precision schedule.

    Embeddings and norms kept in fp16. Layer weights quantized per schedule.
    """
    q_dict = {}
    for k, v in state_dict.items():
        if "embed" in k or "ln" in k or "norm" in k:
            q_dict[k] = v.half()
        elif v.ndim >= 2:
            # Determine bits from layer index
            bits = schedule[0][0]  # default
            for i in range(n_layers):
                if f"blocks.{i}." in k:
                    bits = get_layer_bits(i, schedule)
                    break

            qmax = (1 << (bits - 1)) - 1
            scale = v.abs().amax() / max(qmax, 1)
            q_vals = (v / scale).round().clamp(-qmax - 1, qmax).to(torch.int8)
            q_dict[k] = {"quantized": q_vals, "scale": scale.half(), "bits": bits}
        else:
            q_dict[k] = v.half()
    return q_dict


def serialize_compressed(state_dict: dict) -> bytes:
    """Serialize quantized state dict to compressed bytes."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    return zlib.compress(raw, level=9)


def deserialize_compressed(data: bytes) -> dict:
    """Deserialize compressed state dict."""
    raw = zlib.decompress(data)
    buf = io.BytesIO(raw)
    return torch.load(buf, map_location="cpu", weights_only=False)


def dequantize_state_dict(q_dict: dict) -> dict:
    """Restore full-precision state dict from quantized."""
    state_dict = {}
    for k, v in q_dict.items():
        if isinstance(v, dict) and "quantized" in v:
            state_dict[k] = v["quantized"].float() * v["scale"].float()
        else:
            state_dict[k] = v.float()
    return state_dict

# =============================================================================
# LR SCHEDULE
# =============================================================================

def get_lr(step: int, warmup: int, total: int, warmdown: int, peak_lr: float) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    if step >= total - warmdown:
        # Linear warmdown to 0
        remaining = total - step
        return peak_lr * remaining / warmdown
    # Cosine decay between warmup and warmdown
    progress = (step - warmup) / max(1, total - warmup - warmdown)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

# =============================================================================
# TRAINING
# =============================================================================

def main():
    setup_distributed()
    torch.manual_seed(H.seed + rank)
    np.random.seed(H.seed + rank)

    # Model
    raw_model = GPT().to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    n_params_unique = n_params - raw_model.embed.weight.numel()  # tied embedding
    if master:
        print(f"[T5 Entry] Parameters: {n_params_unique / 1e6:.2f}M (unique, excl. tied embed)")
        print(f"[T5 Entry] Parameters: {n_params / 1e6:.2f}M (total incl. tied)")

    # Enable QK-Clip capture only on CUDA (too slow on MPS/CPU)
    if device.type == "cuda":
        for m in raw_model.modules():
            if isinstance(m, MLAttention):
                m._capture_max_logits = True
        if master:
            print("[T5 Entry] QK-Clip capture enabled (CUDA)")

    # QAT wrapper
    qat = QATWrapper(raw_model, schedule=H.precision_schedule)
    model = qat

    # DDP
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(qat, device_ids=[local_rank])

    # Split params for Muon (2D weights) vs AdamW (rest)
    muon_params, adamw_params = [], []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name and "ln" not in name and "norm" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if master:
        print(f"[T5 Entry] Muon params: {sum(p.numel() for p in muon_params) / 1e6:.2f}M, "
              f"AdamW params: {sum(p.numel() for p in adamw_params) / 1e6:.2f}M")

    optimizer = MuonAdamW(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=H.muon_lr,
        momentum=H.muon_momentum,
        adamw_lr=H.adamw_lr,
        adamw_wd=H.weight_decay,
    )

    # FP8 Training State (CUDA only — all persistent state in FP8)
    fp8 = FP8TrainingState(raw_model, enabled=(device.type == "cuda"))
    if fp8.enabled and master:
        print("[T5 Entry] FP8 training enabled: master weights=E4M3, momentum=E5M2, stochastic rounding")

    # EMA
    ema = EMA(raw_model, decay=H.ema_decay)

    # Data
    train_loader = DataLoader(H.train_files, H.train_batch_tokens, H.seq_len, rank, world_size)
    val_pattern = H.val_files

    # Training loop
    start_time = time.time()
    qat_enabled = False
    best_val_bpb = float("inf")

    if master:
        print(f"[T5 Entry] Starting training: {H.iterations} iterations, "
              f"{H.max_wallclock}s wallclock limit")

    raw_model.train()
    for step in range(H.iterations):
        t0 = time.time()

        # Enable QAT after warmup fraction
        if not qat_enabled and step >= int(H.iterations * H.qat_start_frac):
            qat.enable()
            qat_enabled = True
            if master:
                print(f"[T5 Entry] QAT enabled at step {step}")

        # LR schedule
        muon_lr = get_lr(step, H.warmup_iters, H.iterations, H.warmdown_iters, H.muon_lr)
        adamw_lr = get_lr(step, H.warmup_iters, H.iterations, H.warmdown_iters, H.adamw_lr)
        for group in optimizer.param_groups:
            if group.get("is_muon", False):
                group["lr"] = muon_lr
            else:
                group["lr"] = adamw_lr

        # FP8: load quantized weights into model before forward
        fp8.load_weights(raw_model)

        # Forward + backward
        x, y = train_loader.next_batch()
        logits = model(x)

        # Logit clamping (prevents -log(0)=Inf in cross-entropy)
        if H.logit_clamp > 0:
            logits = torch.clamp(logits, -H.logit_clamp, H.logit_clamp)

        loss = F.cross_entropy(logits.view(-1, H.vocab_size), y.view(-1))

        # Z-Loss: penalize large logit magnitudes (PaLM/Gemini stability)
        z_loss_val = 0.0
        if H.z_loss_weight > 0:
            z_loss = H.z_loss_weight * torch.logsumexp(logits, dim=-1).pow(2).mean()
            loss = loss + z_loss
            z_loss_val = z_loss.item()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)

        optimizer.step()

        # FP8: stochastic round weights + optimizer state back to FP8
        fp8.save_weights(raw_model)
        fp8.save_optimizer_state(optimizer)

        # QK-Clip: post-step attention score rescaling (Kimi K2)
        qk_clipped = qk_clip_step(raw_model, tau=H.qk_clip_tau)

        optimizer.zero_grad(set_to_none=True)

        # EMA update
        if step >= H.ema_start_step:
            ema.update(raw_model)

        # Logging
        dt = time.time() - t0
        tokens_per_sec = H.train_batch_tokens / dt
        log_every = 100 if H.iterations > 200 else max(1, H.iterations // 10)
        if master and step % log_every == 0:
            elapsed = time.time() - start_time
            extra = ""
            if qat_enabled:
                extra += " [QAT]"
            if qk_clipped > 0:
                extra += f" [QK-Clip:{qk_clipped}]"
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"lr_muon {muon_lr:.5f} | lr_adamw {adamw_lr:.6f} | "
                  f"tok/s {tokens_per_sec:.0f} | elapsed {elapsed:.0f}s{extra}")

        # Validation
        if master and H.val_loss_every > 0 and (step + 1) % H.val_loss_every == 0:
            val_loss, val_bpb = evaluate(
                raw_model, val_pattern, H.tokenizer_path, H.val_batch_size, H.seq_len,
                max_eval_tokens=H.max_eval_tokens,
            )
            print(f"  val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
            raw_model.train()

        # Wallclock limit
        elapsed = time.time() - start_time
        if elapsed > H.max_wallclock and H.max_wallclock > 0:
            if master:
                print(f"[T5 Entry] Wallclock limit reached at step {step} ({elapsed:.0f}s)")
            break

    # === Final evaluation with EMA weights ===
    if master:
        print("\n[T5 Entry] Applying EMA weights for final evaluation...")
        # Save current weights
        orig_state = copy.deepcopy(raw_model.state_dict())

        # Apply EMA
        ema.apply_to(raw_model)
        raw_model.eval()

        val_loss, val_bpb = evaluate(
            raw_model, val_pattern, H.tokenizer_path, H.val_batch_size, H.seq_len
        )
        print(f"  [EMA] val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

        # === Serialize and measure artifact size ===
        q_state = quantize_state_dict(raw_model.state_dict(), H.precision_schedule, H.n_layer)
        compressed = serialize_compressed(q_state)
        model_bytes = len(compressed)

        # Code size
        code_path = os.path.abspath(__file__)
        code_bytes = os.path.getsize(code_path) if os.path.exists(code_path) else 0

        total_artifact = model_bytes + code_bytes
        fits = total_artifact <= 16_000_000

        print(f"\n{'=' * 60}")
        print(f"  FINAL RESULTS ({H.variant})")
        print(f"{'=' * 60}")
        print(f"  val_loss (EMA):  {val_loss:.4f}")
        print(f"  val_bpb (EMA):   {val_bpb:.4f}")
        print(f"  model bytes:     {model_bytes:,} ({model_bytes / 1e6:.2f} MB)")
        print(f"  code bytes:      {code_bytes:,}")
        print(f"  total artifact:  {total_artifact:,} ({total_artifact / 1e6:.2f} MB)")
        print(f"  fits 16MB:       {'YES' if fits else 'NO'}")
        print(f"  architecture:    {H.n_layer}L x {H.n_embd}d, MLA(r={H.kv_lora_rank}), "
              f"MLP{H.mlp_mult}x, vocab={H.vocab_size}")
        print(f"  params:          {n_params_unique / 1e6:.2f}M unique")
        print(f"{'=' * 60}")

        # Save artifact
        out_dir = f"runs/{H.run_id}"
        os.makedirs(out_dir, exist_ok=True)
        artifact_path = os.path.join(out_dir, "model.bin")
        with open(artifact_path, "wb") as f:
            f.write(compressed)
        print(f"  Artifact saved:  {artifact_path}")

        # Roundtrip test: decompress → dequantize → evaluate
        print("\n[T5 Entry] Roundtrip verification...")
        q_restored = deserialize_compressed(compressed)
        sd_restored = dequantize_state_dict(q_restored)
        raw_model.load_state_dict(sd_restored)
        rt_loss, rt_bpb = evaluate(
            raw_model, val_pattern, H.tokenizer_path, H.val_batch_size, H.seq_len,
            max_eval_tokens=H.max_eval_tokens,
        )
        print(f"  [Roundtrip] val_loss={rt_loss:.4f} val_bpb={rt_bpb:.4f}")
        print(f"  Quantization gap: {abs(rt_bpb - val_bpb):.4f} BPB")

    # Cleanup
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
