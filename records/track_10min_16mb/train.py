"""
model_custom.py — Parameter Golf v37 architecture
Compatible with: train_gpt.py, torch.compile, DDP, bfloat16

Interface:
    loss = model(x, y)                          # training
    loss, logits = model(x, y, return_logits=True)  # eval / logging

Plug-in to train_gpt.py:
    from model_custom import ModelArgs, CustomGPT as GPT
    # or drop in as:
    model = CustomGPT(ModelArgs())
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────
# CONFIG DATACLASS  (mirrors train_gpt.py's GPTConfig pattern)
# ──────────────────────────────────────────────────────────────────
@dataclass
class ModelArgs:
    vocab_size:   int   = 1024
    d_model:      int   = 384
    n_heads:      int   = 6
    kv_heads:     int   = 2          # GQA: 6Q / 2KV
    num_layers:   int   = 14
    mlp_mult:     float = 3.0        # SwiGLU hidden = d_model * mlp_mult (rounded to 64)
    rope_dims:    int   = 32         # partial RoPE (only first 32 dims rotated)
    bigram_vocab: int   = 4096       # trigram hash table size
    bigram_dim:   int   = 64         # trigram embedding dim (projected → d_model)
    mem_tokens:   int   = 4          # global memory tokens prepended during refine
    refine_steps: int   = 2          # number of shared-weight refine passes
    # Refine is triggered when > refine_density fraction of tokens exceed gate_thresh
    gate_thresh:  float = 0.25
    refine_density: float = 0.08

# ──────────────────────────────────────────────────────────────────
# ROPE  (cached, device-aware)
# ──────────────────────────────────────────────────────────────────
_rope_cache: dict = {}

def _get_rope(T: int, rope_dims: int, dev: torch.device):
    key = (T, rope_dims, str(dev))
    if key not in _rope_cache:
        half  = rope_dims // 2
        theta = 1.0 / (10000 ** (torch.arange(half, device=dev).float() / half))
        freqs = torch.outer(torch.arange(T, device=dev).float(), theta)
        _rope_cache[key] = (freqs.cos()[None, None], freqs.sin()[None, None])
    return _rope_cache[key]

def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                rope_dims: int) -> torch.Tensor:
    xr, xp = x[..., :rope_dims], x[..., rope_dims:]
    x1, x2 = xr[..., ::2], xr[..., 1::2]
    # cast cos/sin to match x dtype (important for bfloat16)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    xr  = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], -1).flatten(-2)
    return torch.cat([xr, xp], -1)

# ──────────────────────────────────────────────────────────────────
# HYBRID ATTENTION MASK  (memory=bidirectional, sequence=causal)
# ──────────────────────────────────────────────────────────────────
_mask_cache: dict = {}

def _get_hybrid_mask(M: int, T: int, dev: torch.device) -> torch.Tensor:
    key = (M, T, str(dev))
    if key not in _mask_cache:
        L    = M + T
        mask = torch.ones(L, L, dtype=torch.bool, device=dev)
        mask[M:, M:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
        _mask_cache[key] = mask[None, None]   # (1, 1, L, L)
    return _mask_cache[key]

# ──────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Depth-scaled RMSNorm. Deeper layers start with smaller initial scale."""
    def __init__(self, d: int, depth: int = 0):
        super().__init__()
        self.w     = nn.Parameter(torch.ones(d))
        self.scale = 1.0 / math.sqrt(depth + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.w * x / (x.pow(2).mean(-1, True) + 1e-6).sqrt()


class GQAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with:
    - partial RoPE (first rope_dims dimensions)
    - per-head learned temperature (replaces fixed 1/√d scaling)
    - flash attention via F.scaled_dot_product_attention
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads  = args.n_heads
        self.kv_heads = args.kv_heads
        self.head_dim = args.d_model // args.n_heads
        self.gqa_rep  = args.n_heads // args.kv_heads
        self.rope_dims = args.rope_dims

        self.q = nn.Linear(args.d_model, args.n_heads  * self.head_dim, bias=False)
        self.k = nn.Linear(args.d_model, args.kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(args.d_model, args.kv_heads * self.head_dim, bias=False)
        self.o = nn.Linear(args.d_model, args.d_model,                   bias=False)
        # Learned per-head temperature initialised near 1/√head_dim
        self.temp = nn.Parameter(torch.ones(args.n_heads) / math.sqrt(self.head_dim))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, Dh = self.n_heads, self.kv_heads, self.head_dim

        q = self.q(x).view(B, T, H,   Dh).transpose(1, 2)
        k = self.k(x).view(B, T, Hkv, Dh).transpose(1, 2)
        v = self.v(x).view(B, T, Hkv, Dh).transpose(1, 2)

        k = k.repeat_interleave(self.gqa_rep, 1)
        v = v.repeat_interleave(self.gqa_rep, 1)

        q = _apply_rope(q, cos, sin, self.rope_dims)
        k = _apply_rope(k, cos, sin, self.rope_dims)
        q = q * self.temp.to(q.dtype).view(1, -1, 1, 1)

        if attn_mask is not None:
            # Convert bool mask to additive float mask for SDPA
            add = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(~attn_mask, float('-inf'))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=add)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return self.o(out.transpose(1, 2).reshape(B, T, H * Dh))


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x)). Better gradient flow than ReLU²."""
    def __init__(self, d_model: int, hidden: int | None = None, mult: float = 3.0):
        super().__init__()
        h = hidden or int(d_model * mult / 64) * 64
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up   = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Standard pre-norm transformer block."""
    def __init__(self, args: ModelArgs, depth: int):
        super().__init__()
        self.norm_attn = RMSNorm(args.d_model, depth)
        self.norm_mlp  = RMSNorm(args.d_model, depth)
        self.attn      = GQAttention(args)
        self.mlp       = SwiGLU(args.d_model, mult=args.mlp_mult)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.mlp(self.norm_mlp(x))
        return x

# ──────────────────────────────────────────────────────────────────
# REFINE MODULES
# Single shared RefineBlock reused refine_steps times.
# HyperNorm provides distinct learned modulation per pass via
# step index, so weight sharing does not collapse passes.
# ──────────────────────────────────────────────────────────────────
class HyperNorm(nn.Module):
    """Step-conditioned RMSNorm: unique gamma/beta per refine pass."""
    def __init__(self, d: int, refine_steps: int):
        super().__init__()
        self.norm  = RMSNorm(d)
        self.hyper = nn.Parameter(torch.zeros(refine_steps, d * 2))

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.hyper[step].to(h.dtype).chunk(2, -1)
        return h * (1 + gamma) + beta


class RefineBlock(nn.Module):
    """
    Shared-weight refine block for iterative token refinement.
    Runs with hybrid attention mask (memory=bidirectional, seq=causal).
    Uses a wider FFN (REFINE_MLP > standard hidden) for more capacity per pass.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        refine_hidden = int(args.d_model * 2.5 / 64) * 64   # 960 for d=384
        self.norm_attn = HyperNorm(args.d_model, args.refine_steps)
        self.norm_mlp  = HyperNorm(args.d_model, args.refine_steps)
        self.attn      = GQAttention(args)
        self.mlp       = SwiGLU(args.d_model, hidden=refine_hidden)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor, step: int) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x, step), cos, sin, attn_mask=mask)
        x = x + self.mlp(self.norm_mlp(x, step))
        return x

# ──────────────────────────────────────────────────────────────────
# MAIN MODEL
# ──────────────────────────────────────────────────────────────────
class CustomGPT(nn.Module):
    """
    14-layer U-Net transformer with:
      - GQA (6Q/2KV heads), partial RoPE, per-head temperature
      - Trigram hash embeddings (additive context correction)
      - Gated U-Net skip connections (7 encoder + 7 decoder layers)
      - Adaptive entropy-gated refinement (shared 2-pass block)
      - Hard per-token binary blending (refined vs backbone output)
      - Weight tying: head = mem_head = embed

    Forward signature (train_gpt.py compatible):
        loss = model(x, y)
        loss, logits = model(x, y, return_logits=True)
    """
    def __init__(self, args: ModelArgs = ModelArgs()):
        super().__init__()
        self.args = args
        d = args.d_model

        # ── Embeddings ──────────────────────────────────────────
        self.embed       = nn.Embedding(args.vocab_size, d)
        self.bigram      = nn.Embedding(args.bigram_vocab, args.bigram_dim)
        self.bigram_proj = nn.Linear(args.bigram_dim, d, bias=False)

        # ── Memory tokens (prepended during refine pass) ────────
        self.mem     = nn.Parameter(torch.randn(1, args.mem_tokens, d) * 0.02)
        self.mem_pos = nn.Parameter(torch.randn(args.mem_tokens, d) * 0.01)

        # ── U-Net backbone ──────────────────────────────────────
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.num_layers)])
        # Skip gate init 0.0 → sigmoid(0)=0.5: balanced early gradient flow
        self.skip_gates = nn.Parameter(torch.zeros(args.num_layers // 2))

        # ── Adaptive refinement (single shared block) ───────────
        self.refine = RefineBlock(args)

        # ── Entropy gate parameters ─────────────────────────────
        self.entropy_bias = nn.Parameter(torch.tensor(0.3))
        # sharpness annealed externally by training loop (2.0 → 12.0)
        self.register_buffer("sharpness", torch.tensor(6.0))

        # ── Output heads ────────────────────────────────────────
        self.norm     = RMSNorm(d)
        self.head     = nn.Linear(d, args.vocab_size, bias=False)
        self.mem_head = nn.Linear(d, args.vocab_size, bias=False)
        # Weight tying — both heads share the embed table
        self.head.weight     = self.embed.weight
        self.mem_head.weight = self.embed.weight

    # ── state_dict overrides: exclude tied heads to avoid duplication ──
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        # Remove tied aliases — they equal embed.weight and waste space
        sd.pop("head.weight",     None)
        sd.pop("mem_head.weight", None)
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Restore tied heads before loading
        state_dict["head.weight"]     = state_dict["embed.weight"]
        state_dict["mem_head.weight"] = state_dict["embed.weight"]
        return super().load_state_dict(state_dict, strict=strict)

    # ── Forward ────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, y: torch.Tensor,
                return_logits: bool = False):
        B, T = x.shape
        args  = self.args

        # Embedding + trigram hash context correction
        tok  = self.embed(x)
        prev  = F.pad(x[:, :-1], (1, 0))
        prev2 = F.pad(x[:, :-2], (2, 0))
        bg    = (prev2 * 131 + prev * 31 + x) % args.bigram_vocab
        h     = tok + self.bigram_proj(self.bigram(bg))

        # RoPE for backbone
        cos, sin = _get_rope(T, args.rope_dims, x.device)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        # U-Net backbone with gated skip connections
        half  = args.num_layers // 2
        skips = []
        for i, block in enumerate(self.blocks):
            if i < half:
                h = block(h, cos, sin)
                skips.append(h)
            else:
                k = half - 1 - (i - half)
                h = h + torch.sigmoid(self.skip_gates[k]) * skips[k]
                h = block(h, cos, sin)

        # Entropy gate — always computed (DDP-safe, no random branching)
        # Running under no_grad keeps it cheap; no AMP issues since no .float() cast
        with torch.no_grad():
            logits_probe = self.head(self.norm(h))
            p            = F.softmax(logits_probe, dim=-1)
            entropy      = -(p * p.clamp_min(1e-9).log()).sum(-1, True)
        gate = torch.sigmoid(self.sharpness * (entropy - self.entropy_bias))

        # Hard per-token binary mask: 1 = refine, 0 = keep backbone output
        mask_hard   = (gate > args.gate_thresh).to(h.dtype)        # (B, T, 1)
        need_refine = (mask_hard.mean() > args.refine_density).item()

        # Adaptive refine pass (shared block, REFINE_STEPS iterations)
        if need_refine:
            mem   = self.mem.expand(B, -1, -1) + self.mem_pos
            h_ext = torch.cat([mem, h], dim=1)
            L_ext = h_ext.size(1)
            cos_r, sin_r = _get_rope(L_ext, args.rope_dims, x.device)
            cos_r = cos_r.to(h.dtype)
            sin_r = sin_r.to(h.dtype)
            attn_mask = _get_hybrid_mask(args.mem_tokens, T, x.device)
            for s in range(args.refine_steps):
                h_ext = self.refine(h_ext, cos_r, sin_r, attn_mask, s)
            mem_out = h_ext[:, :args.mem_tokens]
            seq_ref = h_ext[:, args.mem_tokens:]
            # Token-level blend: only hard-gated tokens take refined output
            h = mask_hard * seq_ref + (1 - mask_hard) * h
        else:
            mem_out = self.mem.expand(B, -1, -1)

        # Main logits
        logits = self.head(self.norm(h))

        # Loss
        main_loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        # Auxiliary memory loss (summary task: predict last mem_tokens of sequence)
        aux_target = y[:, -args.mem_tokens:].reshape(-1)
        aux_logits = self.mem_head(self.norm(mem_out))
        aux_loss   = F.cross_entropy(aux_logits.view(-1, args.vocab_size), aux_target)
        # Aux weight decays during training via sharpness buffer proxy;
        # fixed at 0.05 here since train_gpt.py controls the step externally
        loss = main_loss + 0.05 * aux_loss

        if return_logits:
            return loss, logits
        return loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n
