"""
Mamba-2/SSD + sparse attention hybrid autoregressive LM.

Simplifications vs canonical Mamba-2:
- Scalar A per head (not diagonal matrix) — enables simple linear recurrence
- Constant dt per head (learned bias + softplus), not input-dependent projection
- Sequential scan within chunks (chunk_size=64), no custom CUDA kernels
- Standard Conv1d (not fused with scan)
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Quantization control patterns — must match train.py's CONTROL_TENSOR_NAME_PATTERNS
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "mixer_scale,attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,skip_weights,A_log,dt_bias,D_param",
    ).split(",")
    if pattern
)

# ---------------------------------------------------------------------------
# Reused components from baseline (identical logic)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


# ---------------------------------------------------------------------------
# Mamba-2 / SSD selective scan (pure PyTorch, no custom CUDA)
# ---------------------------------------------------------------------------


def selective_scan(
    x: Tensor,      # [B, L, n_heads, head_dim]
    A: Tensor,       # [n_heads] — negative log-space decay rates
    B: Tensor,       # [B, L, n_heads, d_state]
    C: Tensor,       # [B, L, n_heads, d_state]
    dt: Tensor,      # [n_heads] — timestep (after softplus)
    chunk_size: int,
) -> Tensor:
    """Fully vectorized SSD-style chunked scan — no Python loops.

    Following the Mamba-2 reference (ssd_minimal.py) adapted for constant A/dt:
    Step 1: Intra-chunk causal matmul (diagonal blocks)
    Step 2: Compute per-chunk states via matmul
    Step 3: Inter-chunk state propagation via matmul (vectorized, no loop)
    Step 4: State-to-output correction via matmul
    """
    B_batch, L, n_heads, head_dim = x.shape
    d_state = B.shape[-1]

    # Scalar decay per head: exp(A * dt), in (0, 1)
    decay = torch.exp(A * dt)  # [n_heads]

    # Pad L to multiple of chunk_size (pad=0 is a no-op when already aligned)
    n_chunks = (L + chunk_size - 1) // chunk_size
    L_padded = n_chunks * chunk_size
    pad = L_padded - L
    x = F.pad(x, (0, 0, 0, 0, 0, pad))
    B = F.pad(B, (0, 0, 0, 0, 0, pad))
    C = F.pad(C, (0, 0, 0, 0, 0, pad))

    # Reshape into chunks: [B, nc, cs, nh, ...]
    x_c = x.reshape(B_batch, n_chunks, chunk_size, n_heads, head_dim)
    B_c = B.reshape(B_batch, n_chunks, chunk_size, n_heads, d_state)
    C_c = C.reshape(B_batch, n_chunks, chunk_size, n_heads, d_state)

    # Permute to [B, nc, nh, cs, dim] for efficient batched matmul
    C_p = C_c.permute(0, 1, 3, 2, 4)  # [B, nc, nh, cs, ds]
    B_p = B_c.permute(0, 1, 3, 2, 4)  # [B, nc, nh, cs, ds]
    x_p = x_c.permute(0, 1, 3, 2, 4)  # [B, nc, nh, cs, hd]

    # Scale x by dt
    x_dt = x_p * dt[None, None, :, None, None]  # [B, nc, nh, cs, hd]

    # ===== Step 1: Intra-chunk (diagonal blocks) =====
    # y_diag[i] = sum_{j<=i} decay^(i-j) * dt * (C[i] . B[j]) * x[j]
    # = (L_mat * (C @ B^T)) @ (x * dt)

    idx = torch.arange(chunk_size, device=x.device, dtype=x.dtype)
    diff = idx[:, None] - idx[None, :]  # [cs, cs]
    # Causal decay matrix: L[i,j] = decay^(i-j) if i>=j, else 0
    # Shape: [nh, cs, cs]
    L_mat = torch.where(
        (diff >= 0).unsqueeze(0),
        decay[:, None, None] ** diff.clamp(min=0).unsqueeze(0),
        torch.zeros(1, device=x.device, dtype=x.dtype),
    )

    # CB: [B, nc, nh, cs_i, cs_j]
    CB = torch.matmul(C_p, B_p.transpose(-1, -2))
    CB_L = CB * L_mat[None, None]  # apply causal decay

    # y_diag: [B, nc, nh, cs, hd]
    y_diag = torch.matmul(CB_L, x_dt)

    # ===== Step 2: Per-chunk states =====
    # s[c] = sum_{j=0}^{cs-1} decay^(cs-1-j) * dt * (B[c,j] outer x[c,j])
    # = B_weighted^T @ x_dt,  where B_weighted[j] = decay^(cs-1-j) * B[j]

    rev_idx = torch.arange(chunk_size - 1, -1, -1, device=x.device, dtype=x.dtype)
    decay_rev = decay[None, :] ** rev_idx[:, None]  # [cs, nh] -> broadcast to [nh, cs]
    # B_weighted: [B, nc, nh, cs, ds] scaled by decay_rev
    B_w = B_p * decay_rev.permute(1, 0)[None, None, :, :, None]
    # states: [B, nc, nh, ds, hd] = B_w^T @ x_dt
    states = torch.matmul(B_w.transpose(-1, -2), x_dt)  # [B, nc, nh, ds, hd]

    # ===== Step 3: Inter-chunk state propagation (vectorized) =====
    # h_carry[c] = sum_{c'<c} (decay^cs)^(c-1-c') * s[c']
    # This is a strictly lower-triangular Toeplitz matmul.

    chunk_idx = torch.arange(n_chunks, device=x.device, dtype=x.dtype)
    shift = chunk_idx[:, None] - chunk_idx[None, :] - 1  # [nc, nc]
    # M[c, c'] = (decay^cs)^shift if shift >= 0, else 0
    # Shape: [nh, nc, nc]
    chunk_decay = decay ** chunk_size  # [nh]
    M = torch.where(
        (shift >= 0).unsqueeze(0),
        chunk_decay[:, None, None] ** shift.clamp(min=0).unsqueeze(0),
        torch.zeros(1, device=x.device, dtype=x.dtype),
    )

    # h_carry: [B, nc, nh, ds, hd] via matmul over chunk dimension
    # states: [B, nc, nh, ds, hd] -> reshape for matmul
    # We need: h_carry[b, c, h, d, s] = sum_{c'} M[h, c, c'] * states[b, c', h, d, s]
    # Reshape states to [B*nh*ds*hd, nc] for batched matmul, or use einsum
    # More efficient: work in [B, nh, nc, ds*hd] layout
    states_r = states.permute(0, 2, 1, 3, 4)  # [B, nh, nc, ds, hd]
    states_flat = states_r.reshape(B_batch * n_heads, n_chunks, d_state * head_dim)
    M_exp = M.unsqueeze(0).expand(B_batch, -1, -1, -1).reshape(B_batch * n_heads, n_chunks, n_chunks)
    h_carry_flat = torch.matmul(M_exp, states_flat)  # [B*nh, nc, ds*hd]
    h_carry = h_carry_flat.reshape(B_batch, n_heads, n_chunks, d_state, head_dim)
    # -> [B, nc, nh, ds, hd]
    h_carry = h_carry.permute(0, 2, 1, 3, 4)

    # ===== Step 4: State-to-output correction =====
    # y_off[c, i] = C[c,i] . (decay^(i+1) * h_carry[c])
    # = decay^(i+1) * C[c,i] @ h_carry[c]

    # decay_out: [nh, cs]
    decay_out = decay[:, None] ** (idx[None, :] + 1)

    # C_p @ h_carry: [B, nc, nh, cs, ds] @ [B, nc, nh, ds, hd] = [B, nc, nh, cs, hd]
    y_off = torch.matmul(C_p, h_carry) * decay_out[None, None, :, :, None]

    # ===== Combine =====
    y = y_diag + y_off  # [B, nc, nh, cs, hd]

    # Reshape: [B, nc, nh, cs, hd] -> [B, nc, cs, nh, hd] -> [B, L, nh, hd]
    y = y.permute(0, 1, 3, 2, 4).reshape(B_batch, L_padded, n_heads, head_dim)

    return y[:, :L]


# ---------------------------------------------------------------------------
# Mamba-2 mixer block
# ---------------------------------------------------------------------------


class Mamba2Mixer(nn.Module):
    """Mamba-2/SSD-style selective state space mixer.

    Simplifications vs canonical Mamba-2:
    - Scalar A per head (not diagonal matrix)
    - Constant dt per head (learned bias + softplus), not input-dependent
    - Sequential scan (no custom CUDA kernels)
    - Standard Conv1d (not fused with scan)
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_state: int,
        conv_kernel: int,
        chunk_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.d_state = d_state
        self.head_dim = d_inner // n_heads
        self.chunk_size = chunk_size

        # Input projection: value + gate
        self.in_proj = CastedLinear(d_model, 2 * d_inner, bias=False)

        # Causal depthwise conv on value path
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=conv_kernel,
            groups=d_inner,
            padding=conv_kernel - 1,
        )

        # B and C projections from pre-norm input
        self.B_proj = CastedLinear(d_model, n_heads * d_state, bias=False)
        self.C_proj = CastedLinear(d_model, n_heads * d_state, bias=False)

        # Learnable per-head parameters
        # dt_bias: init so that softplus(dt_bias) ~ Uniform(0.001, 0.1)
        dt_init = torch.empty(n_heads).uniform_(0.001, 0.1)
        self.dt_bias = nn.Parameter(self._inverse_softplus(dt_init))

        # A_log: init as log(1..n_heads) — decay rates
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_heads + 1, dtype=torch.float32)))

        # D: skip connection weight per head
        self.D_param = nn.Parameter(torch.ones(n_heads))

        # Output projection (zero-init for residual)
        self.out_proj = CastedLinear(d_inner, d_model, bias=False)
        self.out_proj._zero_init = True

    @staticmethod
    def _inverse_softplus(x: Tensor) -> Tensor:
        return x + torch.log(-torch.expm1(-x))

    def forward(self, x: Tensor, x_prenorm: Tensor) -> Tensor:
        """
        Args:
            x: post-norm input for value/gate projections [B, L, d_model]
            x_prenorm: pre-norm input for B, C projections [B, L, d_model]
        """
        B_batch, L, _ = x.shape

        # 1. Value + gate from post-norm input
        vg = self.in_proj(x)  # [B, L, 2*d_inner]
        value, gate = vg.chunk(2, dim=-1)  # each [B, L, d_inner]

        # 2. Causal depthwise conv on value, then SiLU
        # Conv1d expects [B, C, L]
        value = value.transpose(1, 2)
        value = self.conv1d(value)[:, :, :L]  # causal: trim future padding
        value = value.transpose(1, 2)
        value = F.silu(value)

        # 3. B and C from pre-norm input
        B_proj = self.B_proj(x_prenorm).reshape(B_batch, L, self.n_heads, self.d_state)
        C_proj = self.C_proj(x_prenorm).reshape(B_batch, L, self.n_heads, self.d_state)

        # 4. Per-head timestep and decay
        dt = F.softplus(self.dt_bias)  # [n_heads]
        A = -torch.exp(self.A_log)     # [n_heads], negative

        # 5. Reshape value for scan: [B, L, n_heads, head_dim]
        value_heads = value.reshape(B_batch, L, self.n_heads, self.head_dim)

        # 6. Selective scan
        y = selective_scan(value_heads, A, B_proj, C_proj, dt, self.chunk_size)

        # 7. Skip connection with D
        y = y + self.D_param[None, None, :, None] * value_heads

        # 8. Reshape and gate
        y = y.reshape(B_batch, L, self.d_inner) * F.silu(gate)

        # 9. Output projection
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Block wrappers
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Mamba-2 mixer with pre-norm, residual scaling, and residual mixing."""

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_heads: int,
        d_state: int,
        conv_kernel: int,
        chunk_size: int,
    ):
        super().__init__()
        self.norm = RMSNorm()
        self.mixer = Mamba2Mixer(d_model, d_inner, n_heads, d_state, conv_kernel, chunk_size)
        self.mixer_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(d_model), torch.zeros(d_model))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_mixed = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.norm(x_mixed)
        return x_mixed + self.mixer_scale.to(dtype=x.dtype)[None, None, :] * self.mixer(n, n)


class AttentionBlock(nn.Module):
    """Attention + MLP block (same as baseline Block, without LoRA hooks)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self.attn(n)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Top-level hybrid model
# ---------------------------------------------------------------------------


class MambaHybrid(nn.Module):
    """Mamba-2/SSD + sparse attention hybrid autoregressive LM.

    Architecture:
    - Most layers are MambaBlock (SSM)
    - A few layers (specified by attn_layer_indices) are AttentionBlock
    - U-Net skip connections between encoder/decoder halves
    - Tied embeddings, logit softcap
    """

    def __init__(self, config):
        super().__init__()
        if config.logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {config.logit_softcap}")

        self.tie_embeddings = config.tie_embeddings
        self.tied_embed_init_std = config.tied_embed_init_std
        self.logit_softcap = config.logit_softcap
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)

        # U-Net skip connection setup
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, config.model_dim, dtype=torch.float32)
        )

        # Build blocks: Mamba for most, Attention at specified indices
        attn_indices = set(config.attn_layer_indices)
        d_inner = config.model_dim * config.ssm_expansion
        blocks = []
        for i in range(config.num_layers):
            if i in attn_indices:
                blocks.append(
                    AttentionBlock(
                        dim=config.model_dim,
                        num_heads=config.num_attn_heads,
                        num_kv_heads=config.num_kv_heads,
                        mlp_mult=config.mlp_mult,
                        rope_base=config.rope_base,
                        qk_gain_init=config.qk_gain_init,
                    )
                )
            else:
                blocks.append(
                    MambaBlock(
                        d_model=config.model_dim,
                        d_inner=d_inner,
                        n_heads=config.ssm_num_heads,
                        d_state=config.ssm_state_dim,
                        conv_kernel=config.ssm_conv_kernel,
                        chunk_size=config.ssm_chunk_size,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm()
        self.lm_head = None if config.tie_embeddings else CastedLinear(config.model_dim, config.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # Encoder half: store skip connections
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)

        # Decoder half: consume skip connections in reverse
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[bi](x, x0)

        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction="mean",
        )
