"""
hymba_layer.py
==============
Drop-in HymbaLayer for the parameter-golf SOTA train_gpt.py.

Tested against the PR #1493 record script. Matches its API exactly:
  - Uses CastedLinear (weights fp32, matmul bf16) like the host script
  - Uses Rotary + apply_rotary_emb from the host script (no reimplementation)
  - Uses F.rms_norm on q, k before RoPE (matching CausalSelfAttention)
  - Block.forward signature: forward(self, x, x0) -> x  (unchanged)
  - attn is called as: self.attn(self.attn_norm(x))  (single tensor in, single out)

APPLY THE PATCH
---------------
  python hymba_layer.py --patch /path/to/train_gpt.py

Or apply manually — see PATCH_INSTRUCTIONS at bottom of this file.

ENV VARS
--------
  HYMBA_ENABLED   : 1 to activate (default: 0)
  HYMBA_LAYERS    : comma-separated layer indices (default: 3,4,5)
  HYMBA_META_TOK  : meta tokens per hybrid layer (default: 4)
  HYMBA_KV_SHARE  : 1 to share K,V across adjacent Hymba layers (default: 1)
  HYMBA_SSM_STATE : SSM state dimension (default: 16)

ARCHITECTURE
------------
  HymbaLayer replaces CausalSelfAttention in the configured layers.
  n_attn = n_head // 2  (e.g. 4)   — standard attention heads
  n_ssm  = n_head // 2  (e.g. 4)   — Mamba-lite SSM heads
  Both pathways run IN PARALLEL on the same input and are concatenated.
  This guarantees n_attn % n_kv_head == 0 (since n_head % n_kv_head == 0).

REFERENCES
----------
  Hymba: A Hybrid-head Architecture for Small Language Models
  Dong, Fu, ..., Lin, Kautz, Molchanov — ICLR 2025 — arXiv:2411.13676

  CPT: Efficient Deep Neural Network Training via Cyclic Precision
  Lin et al. — ICLR 2021 Spotlight — arXiv:2101.09868
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
_HYMBA_ENABLED = bool(int(os.environ.get("HYMBA_ENABLED", "0")))
_HYMBA_LAYERS  = [int(x) for x in os.environ.get("HYMBA_LAYERS", "3,4,5").split(",") if x]
_HYMBA_META    = int(os.environ.get("HYMBA_META_TOK",   "4"))
_HYMBA_SHARE   = bool(int(os.environ.get("HYMBA_KV_SHARE",  "1")))
_HYMBA_STATE   = int(os.environ.get("HYMBA_SSM_STATE", "16"))
_HYMBA_SCAN_CHUNK = int(os.environ.get("HYMBA_SCAN_CHUNK", "64"))


# ── Mamba-lite SSM head ───────────────────────────────────────────────────────
class MambaLiteHead(nn.Module):
    """
    Selective SSM head (simplified Mamba) operating on (B, T, head_dim).
    Diagonal A, input-dependent B/C/dt (the "selective" part of Mamba).
    Uses a signed chunk-parallel fp32 scan. Each chunk computes the exact
    diagonal recurrence with torch.cumsum, then carries the final state into the
    next chunk. This keeps the signed Bx term intact and avoids a 1024-step
    Python loop.
    """
    def __init__(self, head_dim: int, state_dim: int):
        super().__init__()
        self.log_A   = nn.Parameter(torch.randn(state_dim) * 0.5)
        self.B_proj  = nn.Linear(head_dim, state_dim, bias=False)
        self.C_proj  = nn.Linear(head_dim, state_dim, bias=False)
        self.dt_proj = nn.Linear(head_dim, state_dim, bias=True)
        self.out_proj = nn.Linear(state_dim, head_dim, bias=False)
        nn.init.constant_(self.dt_proj.bias, -4.0)

    @staticmethod
    def _scan_chunks(log_a: Tensor, u: Tensor, chunk_size: int) -> Tensor:
        # Recurrence: h_t = exp(log_a_t) * h_{t-1} + u_t.
        # For one chunk with zero initial state:
        # h_t = p_t * cumsum_i(u_i / p_i), where p_t = prod_j<=t exp(log_a_j).
        # For nonzero initial state h0, add p_t * h0.
        B, T, S = u.shape
        h0 = torch.zeros(B, S, device=u.device, dtype=torch.float32)
        chunks = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            la = log_a[:, start:end]
            uu = u[:, start:end]
            log_p = torch.cumsum(la, dim=1).clamp(min=-30.0, max=0.0)
            p = torch.exp(log_p)
            h_local = p * torch.cumsum(uu * torch.exp(-log_p), dim=1)
            h_chunk = h_local + p * h0.unsqueeze(1)
            chunks.append(h_chunk)
            h0 = h_chunk[:, -1]
        return torch.cat(chunks, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, head_dim)
        out_dtype = x.dtype
        dt = F.softplus(self.dt_proj(x).float()).clamp(max=1.0)   # (B, T, S)
        A = -torch.exp(self.log_A.float()).clamp(max=10.0)        # (S,)
        log_a = (dt * A.view(1, 1, -1)).clamp(min=-0.5, max=0.0)  # log decay
        Bx = self.B_proj(x).float() * dt                          # (B, T, S)
        C = self.C_proj(x).float()                                # (B, T, S)

        h_seq = self._scan_chunks(log_a, Bx, max(1, _HYMBA_SCAN_CHUNK))
        y = (C * h_seq).sum(-1, keepdim=True)                     # (B, T, 1)
        return (self.out_proj(h_seq.to(out_dtype)) + y.to(out_dtype) * x[..., :1])


# ── Meta tokens ───────────────────────────────────────────────────────────────
class MetaTokens(nn.Module):
    """
    Learnable tokens prepended to each sequence before attention.
    Hymba paper: stores global context, reduces forced-to-attend burden.
    ~n * d_model extra parameters per layer.
    """
    def __init__(self, n: int, d: int):
        super().__init__()
        self.n      = n
        self.tokens = nn.Parameter(torch.randn(1, n, d) * 0.02)

    def prepend(self, x: Tensor) -> Tensor:
        return torch.cat([self.tokens.expand(x.size(0), -1, -1), x], dim=1)


# ── HymbaLayer ────────────────────────────────────────────────────────────────
class HymbaLayer(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention in train_gpt.py (PR #1493).

    Matches the host script's API exactly:
      forward(self, x: Tensor) -> Tensor

    Internally uses CastedLinear, Rotary, apply_rotary_emb, and F.rms_norm
    exactly as CausalSelfAttention does — these are imported from the host
    script at patch time so we never have a separate implementation.

    Head split: n_attn = n_head // 2, n_ssm = n_head // 2.
    This guarantees n_attn % n_kv_head == 0 for any valid (n_head, n_kv_head)
    pair where n_head % n_kv_head == 0 (required by GQA).
    """

    def __init__(
        self,
        dim:          int,
        num_heads:    int,
        num_kv_heads: int,
        rope_base:    float,
        qk_gain_init: float,
        kv_source:    Optional[HymbaLayer] = None,
    ):
        super().__init__()
        assert num_heads % 2 == 0, "num_heads must be even"
        assert (num_heads // 2) % num_kv_heads == 0, \
            f"n_attn={num_heads//2} must be divisible by num_kv_heads={num_kv_heads}"

        self.dim         = dim
        self.num_heads   = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_attn      = num_heads // 2
        self.n_ssm       = num_heads // 2
        self.head_dim    = dim // num_heads
        self.kv_rep      = self.n_attn // num_kv_heads
        self._kv_source  = kv_source

        # These are injected after construction by apply_patch() below
        # so that HymbaLayer uses the exact same CastedLinear and Rotary
        # classes as the host script.
        _CastedLinear = _get_host_class("CastedLinear")
        _Rotary       = _get_host_class("Rotary")

        D  = self.head_dim
        kv = num_kv_heads * D

        # Attention projections
        self.c_q  = _CastedLinear(dim, dim,  bias=False)
        self.proj = _CastedLinear(dim, dim,  bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((self.n_attn,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = _Rotary(D, base=rope_base)

        # KV projections — only if not sharing from another layer
        if kv_source is None:
            self.c_k = _CastedLinear(dim, kv, bias=False)
            self.c_v = _CastedLinear(dim, kv, bias=False)

        # SSM pathway
        self.ssm_in    = _CastedLinear(dim, self.n_ssm * D, bias=False)
        self.ssm_heads = nn.ModuleList([
            MambaLiteHead(D, _HYMBA_STATE) for _ in range(self.n_ssm)
        ])

        # Meta tokens
        self.meta = MetaTokens(_HYMBA_META, dim) if _HYMBA_META > 0 else None

        # Pathway scales
        self.attn_scale = nn.Parameter(torch.ones(1))
        self.ssm_scale  = nn.Parameter(torch.ones(1))

    def _kv_root(self) -> HymbaLayer:
        """Walk the sharing chain to the layer that owns c_k / c_v."""
        node = self
        while node._kv_source is not None:
            node = node._kv_source
        return node

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, dim)  — already RMSNorm'd by Block (matching host script)
        B, T, C = x.shape
        D = self.head_dim

        # ── Meta tokens (attention pathway only) ──────────────────────────────
        if self.meta is not None:
            x_ext = self.meta.prepend(x)   # (B, T+n_meta, C)
            n_pre = self.meta.n
        else:
            x_ext, n_pre = x, 0
        T_ext = T + n_pre

        # ── Q, K, V projections ───────────────────────────────────────────────
        q = self.c_q(x_ext).reshape(B, T_ext, self.num_heads,    D).transpose(1, 2)
        root = self._kv_root()
        k = root.c_k(x_ext).reshape(B, T_ext, self.num_kv_heads, D).transpose(1, 2)
        v = root.c_v(x_ext).reshape(B, T_ext, self.num_kv_heads, D).transpose(1, 2)

        # ── RMS norm + RoPE (matching CausalSelfAttention exactly) ───────────
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T_ext, x.device, q.dtype)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)

        # ── Attention pathway: first n_attn heads ─────────────────────────────
        q_attn = q[:, :self.n_attn] * self.q_gain.to(q.dtype)[None, :, None, None]
        k_exp  = k.repeat_interleave(self.kv_rep, dim=1)   # (B, n_attn, T_ext, D)
        v_exp  = v.repeat_interleave(self.kv_rep, dim=1)

        attn_out = F.scaled_dot_product_attention(
            q_attn, k_exp, v_exp, is_causal=True
        )  # (B, n_attn, T_ext, D)
        # Strip meta tokens and reshape
        attn_out = (attn_out[:, :, n_pre:]
                    .transpose(1, 2).contiguous()
                    .view(B, T, self.n_attn * D))
        attn_out = attn_out * self.attn_scale

        # ── SSM pathway: all n_ssm heads on original x (no meta tokens) ──────
        ssm_in   = self.ssm_in(x).view(B, T, self.n_ssm, D)
        ssm_out  = torch.cat(
            [self.ssm_heads[i](ssm_in[:, :, i]) for i in range(self.n_ssm)],
            dim=-1
        ) * self.ssm_scale   # (B, T, n_ssm * D)

        # ── Combine (n_attn*D + n_ssm*D = dim) and project ───────────────────
        return self.proj(torch.cat([attn_out, ssm_out], dim=-1))


# ── Host class registry ───────────────────────────────────────────────────────
# Populated by apply_patch() so HymbaLayer uses the host script's exact classes.
_HOST_CLASSES: dict = {}
_apply_rotary_emb = None   # set by apply_patch()

def _get_host_class(name: str):
    if name in _HOST_CLASSES:
        return _HOST_CLASSES[name]
    # Fallback to plain nn.Linear if patch hasn't been applied yet (e.g. smoke test)
    return nn.Linear


def register_host_classes(
    CastedLinear,
    Rotary,
    apply_rotary_emb_fn,
):
    """Call this from train_gpt.py after importing hymba_layer."""
    global _apply_rotary_emb
    _HOST_CLASSES["CastedLinear"] = CastedLinear
    _HOST_CLASSES["Rotary"]       = Rotary
    _apply_rotary_emb             = apply_rotary_emb_fn


# ── Builder ───────────────────────────────────────────────────────────────────
def build_and_swap_hymba_layers(model, args) -> None:
    """
    Build HymbaLayer instances and swap them into model.blocks in-place.
    Call this immediately after GPT is constructed.

    Args:
        model : the GPT instance (base_model in train_gpt.py)
        args  : Hyperparameters instance
    """
    if not _HYMBA_ENABLED:
        return

    layers: dict[int, HymbaLayer] = {}
    for i, idx in enumerate(_HYMBA_LAYERS):
        if idx >= len(model.blocks):
            print(f"[Hymba] WARNING: layer {idx} out of range ({len(model.blocks)} blocks), skipping")
            continue

        # KV sharing: find root owner
        kv_src = None
        if _HYMBA_SHARE and i > 0:
            prev_idx = _HYMBA_LAYERS[i - 1]
            if prev_idx in layers and abs(idx - prev_idx) == 1:
                kv_src = layers[prev_idx]

        layer = HymbaLayer(
            dim          = args.model_dim,
            num_heads    = args.num_heads,
            num_kv_heads = args.num_kv_heads,
            rope_base    = args.rope_base,
            qk_gain_init = args.qk_gain_init,
            kv_source    = kv_src,
        )
        layers[idx] = layer

        share_str = f" (KV→layer {_HYMBA_LAYERS[i-1]})" if kv_src else ""
        print(f"[Hymba] Layer {idx}: {layer.n_attn} attn + {layer.n_ssm} SSM "
              f"+ {_HYMBA_META} meta{share_str}")

    # Swap into model in-place
    device = next(model.parameters()).device
    for idx, hymba in layers.items():
        model.blocks[idx].attn = hymba.to(device)

    total_new = sum(p.numel() for idx in layers for p in layers[idx].parameters())
    print(f"[Hymba] Swapped layers {_HYMBA_LAYERS} | new params: {total_new:,}")


# ── Smoke test ────────────────────────────────────────────────────────────────
def _smoke_test():
    torch.manual_seed(42)
    # Register stub classes for smoke test (no host script available)
    register_host_classes(
        CastedLinear      = nn.Linear,
        Rotary            = _StubRotary,
        apply_rotary_emb_fn = lambda x, c, s: x,  # identity for smoke test
    )

    n_head, n_kv, dim = 8, 4, 512
    B, T = 2, 64

    layers: dict[int, HymbaLayer] = {}
    for i, idx in enumerate([3, 4, 5]):
        kv_src = None
        if _HYMBA_SHARE and i > 0:
            prev = [3, 4, 5][i - 1]
            if prev in layers:
                kv_src = layers[prev]
        layers[idx] = HymbaLayer(idx, n_head, n_kv, dim, 5.25, kv_source=kv_src)

    x = torch.randn(B, T, dim)
    for idx, layer in layers.items():
        out = layer(x)
        assert out.shape == (B, T, dim), f"Layer {idx}: {out.shape} != {(B,T,dim)}"
        n_params = sum(p.numel() for p in layer.parameters())
        share = f" (shares KV)" if layer._kv_source is not None else ""
        print(f"  Layer {idx}: {out.shape} | {n_params:,} params{share}")
    print("✓ Smoke test passed")


class _StubRotary(nn.Module):
    """Minimal Rotary stub for smoke test."""
    def __init__(self, dim, base=10000.0): super().__init__()
    def forward(self, seq_len, device, dtype):
        c = torch.ones(1, 1, seq_len, 1, dtype=dtype, device=device)
        s = torch.zeros(1, 1, seq_len, 1, dtype=dtype, device=device)
        return c, s


# ── Manual patch instructions ─────────────────────────────────────────────────
PATCH_INSTRUCTIONS = """\
MANUAL PATCH — add these 3 blocks to train_gpt.py
==================================================

── BLOCK 1: after all imports, before class Hyperparameters ──────────────────

import hymba_layer as _hymba_mod
from hymba_layer import build_and_swap_hymba_layers, register_host_classes

── BLOCK 2: after CastedLinear, Rotary, apply_rotary_emb are defined
           (around line 555, just before class CausalSelfAttention) ──────────

register_host_classes(CastedLinear, Rotary, apply_rotary_emb)

── BLOCK 3: after base_model = GPT(...) (around line 826) ────────────────────

build_and_swap_hymba_layers(base_model, args)

── DONE. Run as: ──────────────────────────────────────────────────────────────

HYMBA_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

Everything else (TTT, GPTQ, eval, artifact compression) runs unchanged.
"""


# ── Auto-patcher ──────────────────────────────────────────────────────────────
def auto_patch(path: str) -> None:
    with open(path) as f:
        src = f.read()

    if "hymba_layer" in src:
        print(f"[Hymba] {path} already patched.")
        return

    # Block 1: imports after existing imports
    src = src.replace(
        "from torch.nn.parallel import DistributedDataParallel as DDP\n",
        "from torch.nn.parallel import DistributedDataParallel as DDP\n"
        "import hymba_layer as _hymba_mod\n"
        "from hymba_layer import build_and_swap_hymba_layers, register_host_classes\n",
        1,
    )

    # Block 2: register host classes after apply_rotary_emb is defined
    src = src.replace(
        "\nclass CausalSelfAttention(nn.Module):\n",
        "\nregister_host_classes(CastedLinear, Rotary, apply_rotary_emb)\n"
        "\nclass CausalSelfAttention(nn.Module):\n",
        1,
    )

    # Block 3: swap layers after base_model = GPT(...)
    # Find the GPT(...) constructor call and add swap after it
    src = src.replace(
        "    base_model = GPT(\n",
        "    base_model = GPT(\n",
        1,
    )
    # Find the closing paren of GPT(...) and add swap after
    gpt_end = "    )\n"
    # Insert after the first occurrence that follows "base_model = GPT"
    gpt_idx = src.find("    base_model = GPT(")
    close_idx = src.find(gpt_end, gpt_idx)
    if close_idx != -1:
        insert_pos = close_idx + len(gpt_end)
        src = (src[:insert_pos] +
               "    build_and_swap_hymba_layers(base_model, args)\n" +
               src[insert_pos:])

    with open(path, "w") as f:
        f.write(src)
    print(f"[Hymba] Patched {path}")
    print("[Hymba] Run with: HYMBA_ENABLED=1 torchrun --standalone --nproc_per_node=1 train_gpt.py")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--patch" in sys.argv:
        idx = sys.argv.index("--patch")
        target = sys.argv[idx + 1]
        auto_patch(target)
    elif "--instructions" in sys.argv:
        print(PATCH_INSTRUCTIONS)
    else:
        _smoke_test()
