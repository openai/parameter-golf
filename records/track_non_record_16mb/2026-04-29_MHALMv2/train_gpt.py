"""MHALM — Multi-Head Atlas Language Model.

Option D-multi — 2-pass SSM+Attention on z (Stäckel-native) variant.

Both SSM and attention operate on concatenated encoder z (Stäckel coordinates).
A second SSM+Attention pass refines the temporal representation before combining
with kernel readout for final logits.

Architecture:
  encoders → z [128] → cat(z0,z1,z2) [384] → SSM(z_cat) → H_ssm [256]  ─┐
                                              → Attn(z_cat) → H_attn [256] ─┤→ gate → H [256]
                                                                             │
                              ┌─ proj_back [384] + residual ←───────────────┘
                              │→ SSM2(z2) → H_ssm2 [256]  ─┐
                              │→ Attn2(z2) → H_attn2 [256] ─┤→ gate2 → H_refined [256]
                 → kernels → Φ@W → mixed [1024]              │
                                           readout(H_refined, mixed) → output

Hybrid causal kernel VP architecture with 5 heads:
  Head 0 (Nyström Spherical): Causal token-to-token kernel with Gegenbauer mixture.
  Head 1 (Gabor): Fixed learned anchors, Gaussian window × cosine oscillation.
  Head 2 (Laplacian): Fixed learned anchors, learnable RBF mixture.
  Head 3 (Tucker GL): Gabor ⊙ Laplacian cross-product.
  Head 4 (Linear): Raw encoder output as design matrix.

Usage:
    python train_gpt_D_multi.py --mode golf --data-dir ../../data/fineweb_sp1024
    python train_gpt_D_multi.py --mode smoke  # quick test with synthetic data
"""

import argparse
import io
import math
import os
import time
import zstandard as zstd
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    import sentencepiece as spm
except ImportError:
    spm = None 


# ============================================================================
# === CONFIG =================================================================
# ============================================================================


@dataclass
class HybridConfig:
    V: int = 1024
    d_emb: int = 256
    H: int = 1024
    n_encoder_hidden: int = 2
    d_max: int = 64
    R: int = 512                  # anchors for Gabor/Laplacian heads
    L: int = 2
    R_s: int = 128
    B: int = 64
    T: int = 1024
    total_steps: int = 0
    max_wallclock: float = 600.0
    lr_muon: float = 0.02
    lr_encoder: float = 3e-4
    lr_basis: float = 1e-4
    lr_ssm: float = 1e-3
    lr_other: float = 3e-4
    wd_encoder: float = 0.1
    wd_basis: float = 0.01
    wd_ssm: float = 0.0
    wd_other: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    log_every: int = 50
    val_every: int = 200
    # Temporal
    use_ssm: bool = True
    attn_heads: int = 4
    use_attn: bool = True
    n_attn_layers: int = 2
    # Penalties
    staekel_beta: float = 0.02
    # AMP
    use_amp: bool = True
    # Per-head d_eff for Gabor/Laplacian truncation
    d_eff_gabor: int = 8
    d_eff_laplacian: int = 12
    # Asymmetric encoder widths (0 = use global H)
    H_gabor: int = 0
    H_laplacian: int = 0
    # Nyström approximation for causal spherical head
    R_nystrom: int = 256
    # Logit soft-capping
    logit_cap: float = 30.0
    use_resid_mix: bool = True
    use_residual_scale: bool = True
    use_unet_skip: bool = True
    q_gain_init: float = 1.5
    # Learnable kernel shapes
    use_learnable_gegenbauer: bool = True
    use_learnable_rbf: bool = True
    # SmearGate
    use_smear_gate: bool = True
    # BigramHash embedding
    bigram_vocab_size: int = 0    # 0=disabled, 4096=recommended
    bigram_dim: int = 128         # embedding dim before projection to d_emb
    # Orthogonal init
    use_ortho_init: bool = True
    muon_wd: float = 0.02
    muon_scale_cap: float = 2.0
    muon_momentum_start: float = 0.92
    muon_momentum_end: float = 0.99
    muon_momentum_warmup: int = 1500
    eval_stride: int = 64
    # Nyström normalisation variants
    use_nystrom_softmax: bool = False   # N2: softmax with learnable temperature
    use_nystrom_rowsum: bool = True     # N1: row-sum normalisation (−0.20 nats on FineWeb)
    use_laplacian_rowsum: bool = False  # L1: Laplacian row-sum normalisation
    use_gabor_envelope_norm: bool = False  # G1: Gabor envelope normalisation
    # SWA
    swa_enabled: bool = True
    swa_every: int = 50
    swa_start_frac: float = 0.75
    # Tucker GL cross-head
    use_tucker_gl: bool = True
    # Temporal modulation
    use_temporal_bandwidth: bool = True
    use_temporal_decay: bool = True
    # Dual-resolution Nyström
    use_dual_nystrom: bool = True
    # Linear kernel head
    use_linear_kernel_head: bool = True
    # Post-VP MLP: nonlinear correction after HeadScaler mixing
    use_post_vp_mlp: bool = False
    post_vp_hidden: int = 256
    # Asymmetric dual encoder for Nyström head (D1 from Stäckel theory)
    use_dual_encoder_nystrom: bool = False
    H_dual_nystrom: int = 256
    # Per-head kernel output normalization (data-dependent, from IGL)
    phi_norm_gabor: str = "none"       # "none" | "softmax" | "l2" | "rowsum"
    phi_norm_laplacian: str = "none"   # "none" | "softmax" | "l2" | "rowsum"
    # V1/V2a attention QK variants (from Stäckel theory)
    attn_qk_mode: str = "full"         # "full" | "diagonal" | "per_head_proj"
    attn_d_proj: int = 32              # d_proj for per_head_proj mode
    # S-series: Stiefel manifold enforcement on encoder w_out
    use_stiefel_enforcement: bool = False
    stiefel_steps: int = 3
    use_stiefel_scale: bool = False


def golf_config():
    """Golf submission config — final validated configuration."""
    return HybridConfig(
        V=1024, d_emb=512, H=512, n_encoder_hidden=4, d_max=160,  # H=512: symmetric Tensor Core aligned
        R=256, L=2, R_s=128, B=64, T=1024,
        total_steps=0, max_wallclock=600.0, warmup_steps=50,
        attn_heads=8, n_attn_layers=2,
        staekel_beta=0.02,
        d_eff_gabor=160, d_eff_laplacian=160,
        R_nystrom=256,
        logit_cap=30.0,
        use_resid_mix=True,
        use_residual_scale=True,
        use_unet_skip=True,
        q_gain_init=1.5,
        use_learnable_gegenbauer=True,
        use_learnable_rbf=True,
        use_amp=True,
        use_tucker_gl=True,
        use_temporal_bandwidth=True,
        use_temporal_decay=True,
        use_dual_nystrom=True,
        H_gabor=0, H_laplacian=0,  # 0 = use global H (symmetric: all 512)
        use_linear_kernel_head=True,
        use_smear_gate=True,
        bigram_vocab_size=16384, bigram_dim=160,  # K1: bigram hyper-scaler
        use_ortho_init=True,
        muon_wd=0.02,
        muon_scale_cap=2.0,
        eval_stride=0,  # disabled for golf — NCCL timeout risk; competition eval is authoritative
        swa_enabled=True, swa_every=15, swa_start_frac=0.60,
    )


# ============================================================================
# === DATA ===================================================================
# ============================================================================


def load_fineweb_shard(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        magic_bytes = f.read(4)
        magic = np.frombuffer(magic_bytes, dtype=np.int32)[0]
        if magic == 20240520:
            f.seek(0)
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            ntokens = header[2]
            tokens = np.frombuffer(f.read(ntokens * 2), dtype=np.uint16)
        else:
            f.seek(0)
            tokens = np.fromfile(f, dtype=np.int16)
    return torch.from_numpy(tokens.astype(np.int64))


def discover_fineweb_shards(data_dir: str) -> tuple[list[str], list[str]]:
    data_path = Path(data_dir)
    all_bins = sorted(data_path.glob("*.bin"))
    if not all_bins:
        raise FileNotFoundError(f"No .bin shards found in {data_dir}")
    train_shards, val_shards = [], []
    for p in all_bins:
        (val_shards if "val" in p.name else train_shards).append(str(p))
    if not val_shards and len(train_shards) > 1:
        val_shards = [train_shards.pop(0)]
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"FineWeb: {len(train_shards)} train shards, {len(val_shards)} val shards")
    return train_shards, val_shards


class BatchIterator:
    def __init__(self, data, B, T, device="cpu", V_clamp=None):
        self.data = data if V_clamp is None else data % V_clamp
        self.B, self.T, self.device = B, T, device
        self._offs = torch.arange(T + 1)

    def __iter__(self):
        return self

    def __next__(self):
        ix = torch.randint(0, len(self.data) - self.T, (self.B,))
        w = self.data[ix[:, None] + self._offs[None, :]]  # [B, T+1]
        return w[:, :-1].to(self.device, non_blocking=True), w[:, 1:].to(self.device, non_blocking=True)


class ShardedBatchIterator:
    def __init__(self, shard_paths: list[str], B: int, T: int, device: str = "cpu",
                 V_clamp=None, rank: int = 0, world_size: int = 1):
        # Round-robin assign shards to ranks; fallback to all shards if fewer than world_size
        if len(shard_paths) >= world_size:
            self.shard_paths = shard_paths[rank::world_size]
        else:
            self.shard_paths = shard_paths
        self.B, self.T, self.device = B, T, device
        self.V_clamp = V_clamp
        self.shard_idx = 0
        self.tokens_consumed = 0
        self._offs = torch.arange(T + 1)
        self._load_shard(0)

    def _load_shard(self, idx: int):
        self.shard_idx = idx % len(self.shard_paths)
        data = load_fineweb_shard(self.shard_paths[self.shard_idx])
        self.data = data if self.V_clamp is None else data % self.V_clamp
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pos + self.B * self.T + 1 >= len(self.data):
            self._load_shard(self.shard_idx + 1)
        ix = torch.randint(0, len(self.data) - self.T, (self.B,))
        w = self.data[ix[:, None] + self._offs[None, :]]  # [B, T+1]
        self.pos += self.B * self.T
        self.tokens_consumed += self.B * self.T
        return w[:, :-1].to(self.device, non_blocking=True), w[:, 1:].to(self.device, non_blocking=True)


# ============================================================================
# === EMBEDDING ==============================================================
# ============================================================================


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        g = torch.sigmoid(self.gate)[None, None, :]
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    """Hash consecutive token pairs into a learned embedding table.

    Provides explicit bigram context. Zero-initialised: starts as no-op.
    """

    def __init__(self, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = nn.Linear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale


# ============================================================================
# === ENCODER ================================================================
# ============================================================================


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


@torch.no_grad()
def _newton_schulz_orthogonalize(W, steps=3, power_iters=3):
    """Project W onto Stiefel manifold via Newton-Schulz iteration.
    Uses power iteration for spectral norm (no SVD, no host-device sync)."""
    X = W.data.float()
    # Fast spectral norm via power iteration (GPU-friendly, no cuSOLVER)
    v = torch.ones(X.size(1), 1, device=X.device, dtype=X.dtype)
    for _ in range(power_iters):
        u = X @ v
        u = u / (u.norm() + 1e-7)
        v = X.T @ u
        v = v / (v.norm() + 1e-7)
    spectral_norm = (u.T @ X @ v).squeeze()
    X = X / (spectral_norm + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = (15/8) * X - (10/8) * A @ X + (3/8) * A @ A @ X
    W.data.copy_(X.to(W.dtype))


class ChartEncoder(nn.Module):
    """Ψ: R^{d_in} → [-1,1]^{d_max}. N pre-norm residual layers → RMSNorm → tanh."""

    def __init__(self, d_in, H, d_max, n_hidden=2, use_stiefel_scale=False):
        super().__init__()
        self.linear0 = nn.Linear(d_in, H, bias=False)
        self.proj0 = nn.Linear(d_in, H, bias=False) if d_in != H else nn.Identity()
        self.hidden_lns = nn.ModuleList([nn.LayerNorm(H) for _ in range(n_hidden)])
        self.hidden_ws = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(n_hidden)])
        self.w_out = nn.Linear(H, d_max, bias=False)
        self.coord_temp = nn.Parameter(torch.ones(d_max))
        if use_stiefel_scale:
            self.stiefel_scale = nn.Parameter(torch.ones(d_max))
        else:
            self.stiefel_scale = None

    def forward(self, x):
        h = self.linear0(x)
        h = self.proj0(x) + F.silu(self.hidden_ws[0](self.hidden_lns[0](h)))
        for ln, w in zip(self.hidden_lns[1:], self.hidden_ws[1:]):
            h = h + F.silu(w(ln(h)))
        h_norm = F.rms_norm(h, (h.size(-1),))
        z_raw = self.w_out(h_norm)
        z = torch.tanh(z_raw / (F.softplus(self.coord_temp) + 1e-4))
        if self.stiefel_scale is not None:
            z = z * self.stiefel_scale
        return z


# ============================================================================
# === SPECTRAL BASES =========================================================
# ============================================================================


def _generate_anchor_grid(R, d_max):
    if d_max <= 3:
        ppd = max(2, int(R ** (1.0 / d_max)) + 1)
        coords = [torch.linspace(-0.9, 0.9, ppd) for _ in range(d_max)]
        grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1).reshape(-1, d_max)
        mu = grid[:R] if len(grid) >= R else torch.cat([grid, torch.rand(R - len(grid), d_max) * 1.8 - 0.9])
    else:
        mu = torch.rand(R, d_max) * 1.8 - 0.9
    sigma_base = math.sqrt(d_max / 3.0)
    mu = (mu + (torch.rand_like(mu) - 0.5) * sigma_base).clamp(-0.95, 0.95)
    return mu


class NystromCausalBasis(nn.Module):
    """Nyström approximation of causal token-to-token kernel.

    Selects R uniformly-spaced landmark positions. For each query position i,
    computes cosine similarity only to landmarks j < i. Optionally uses a
    learnable Gegenbauer mixture and temporal decay.
    """

    landmarks: torch.Tensor
    temporal_dist: torch.Tensor

    def __init__(self, R, T, learnable_gegenbauer=False,
                 use_temporal_decay=False, use_dual_nystrom=False,
                 use_softmax=False, use_rowsum=False, d_max=64):
        super().__init__()
        self.R = R
        self.T = T
        if use_dual_nystrom:
            R_global = R // 2
            R_local = R - R_global
            local_start = max(0, T - 192)
            global_lm = torch.linspace(0, T - 1, R_global).long()
            local_lm = torch.linspace(local_start, T - 1, R_local).long()
            landmarks = torch.cat([global_lm, local_lm]).sort().values
        else:
            landmarks = torch.linspace(0, T - 1, R).long()
        self.register_buffer("landmarks", landmarks)
        # Precompute causal mask: landmark j available to position i iff j < i
        positions = torch.arange(T).unsqueeze(1)       # [T, 1]
        landmark_pos = landmarks.unsqueeze(0)            # [1, R]
        self.register_buffer("causal_mask_bool", landmark_pos < positions)  # [T, R]
        self.use_softmax = use_softmax
        self.use_rowsum = use_rowsum
        if use_softmax:
            self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / math.sqrt(d_max))))
            self.gegenbauer_weights = None  # softmax replaces Gegenbauer
        elif learnable_gegenbauer:
            self.gegenbauer_weights = nn.Parameter(torch.tensor([3.0, 0.0, 0.0]))
        else:
            self.gegenbauer_weights = None
        self.use_temporal_decay = use_temporal_decay
        if use_temporal_decay:
            self.temporal_alpha = nn.Parameter(torch.tensor(-10.0))
            positions = torch.arange(T).float().unsqueeze(1)
            landmark_pos = landmarks.float().unsqueeze(0)
            temporal_dist = ((positions - landmark_pos) / T).clamp(min=0.0)
            self.register_buffer("temporal_dist", temporal_dist)

    def _phi_from_similarity(self, C, B, T, device, dtype):
        """Given cosine similarity C [B,T,R], apply masking, Gegenbauer, decay, norm."""
        causal_mask = self.causal_mask_bool.to(dtype)  # [T, R]

        if self.use_softmax:
            tau = torch.exp(self.log_tau)
            C_scaled = C * tau
            neg_inf_mask = ~self.causal_mask_bool
            C_scaled = C_scaled.masked_fill(neg_inf_mask.unsqueeze(0), float('-inf'))
            Phi = torch.softmax(C_scaled, dim=-1)
            Phi = torch.nan_to_num(Phi, nan=0.0)
        else:
            if self.gegenbauer_weights is not None:
                alpha = F.softmax(self.gegenbauer_weights, dim=0)
                P1 = (1.0 + C) * 0.5
                P2 = (3.0 * C * C - 1.0) * 0.5
                P3 = (5.0 * C * C * C - 3.0 * C) * 0.5
                Phi = alpha[0] * P1 + alpha[1] * P2 + alpha[2] * P3
            else:
                Phi = (1.0 + C) * 0.5

            # Apply causal mask AFTER polynomials (kills ghost bias from P(0)≠0)
            Phi = Phi * causal_mask.unsqueeze(0)

            if self.use_temporal_decay:
                alpha_td = F.softplus(self.temporal_alpha)
                decay = torch.exp(-alpha_td * self.temporal_dist[:T, :self.R])
                Phi = Phi * decay.unsqueeze(0)

            if self.use_rowsum:
                row_sum = Phi.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                Phi = Phi / row_sum

        Phi = torch.cat([Phi, torch.ones(B, T, 1, device=device, dtype=dtype)], dim=-1)
        return Phi

    def forward(self, z):
        B, T, d = z.shape
        Q = F.normalize(z, dim=-1)
        K = Q[:, self.landmarks[:self.R], :]
        C = torch.bmm(Q, K.transpose(1, 2))
        return self._phi_from_similarity(C, B, T, z.device, z.dtype)

    def forward_asymmetric(self, z_query, z_key):
        """Asymmetric Nyström: queries from Ψ_Q, landmarks from Ψ_K (D1 architecture)."""
        B, T, d = z_query.shape
        Q = F.normalize(z_query, dim=-1)
        K = F.normalize(z_key[:, self.landmarks[:self.R], :], dim=-1)
        C = torch.bmm(Q, K.transpose(1, 2))
        return self._phi_from_similarity(C, B, T, z_query.device, z_query.dtype)


class GaborBasis(nn.Module):
    """φ(z) = exp(-‖z-μ‖²/2σ²) · cos(k^T z + φ). Fixed learned anchors."""

    def __init__(self, R, d_max, d_eff=None, envelope_norm=False):
        super().__init__()
        self.R = R
        self.envelope_norm = envelope_norm
        d = d_eff if d_eff is not None else d_max
        self.d_eff = d
        self.mu = nn.Parameter(_generate_anchor_grid(R, d))
        self.K = nn.Parameter(torch.randn(R, d) * 0.5)
        sigma_base = math.sqrt(d / 3.0)
        self.log_sigma = nn.Parameter(math.log(sigma_base) + 0.2 * torch.randn(R))
        self.phi = nn.Parameter(torch.rand(R) * 2 * math.pi)

    def forward(self, z):
        z_used = z[..., :self.d_eff]
        sigma = torch.exp(self.log_sigma)
        z_sq = z_used.pow(2).sum(-1, keepdim=True)
        mu_sq = self.mu.pow(2).sum(-1)
        cross = torch.einsum("btd,rd->btr", z_used, self.mu)
        D_sq = (z_sq + mu_sq.unsqueeze(0).unsqueeze(0) - 2 * cross).clamp_min(0.0)
        sigma_sq = sigma.pow(2).unsqueeze(0).unsqueeze(0) + 1e-8
        window = torch.exp(-D_sq / (2 * sigma_sq))
        if self.envelope_norm:
            window = window / window.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        phase = torch.einsum("btd,rd->btr", z_used, self.K) + self.phi
        return window * torch.cos(phase)


class LaplacianBasis(nn.Module):
    """φ(z) = exp(-‖z-μ‖²/2σ²). Fixed learned anchors with learnable RBF mixture
    and temporal bandwidth modulation."""

    pos_frac: torch.Tensor

    def __init__(self, R, d_max, learnable_rbf=False, d_eff=None,
                 use_temporal_bandwidth=False, T_max=1024, rowsum_norm=False):
        super().__init__()
        self.R = R
        self.T_max = T_max
        d = d_eff if d_eff is not None else d_max
        self.d_eff = d
        self.mu = nn.Parameter(_generate_anchor_grid(R, d))
        sigma_base = math.sqrt(d / 3.0)
        self.log_sigma = nn.Parameter(math.log(sigma_base) + 0.2 * torch.randn(R))
        if learnable_rbf:
            self.rbf_weights = nn.Parameter(torch.tensor([3.0, 0.0, 0.0]))
        else:
            self.rbf_weights = None
        self.rowsum_norm = rowsum_norm
        self.use_temporal_bandwidth = use_temporal_bandwidth
        if use_temporal_bandwidth:
            self.temporal_gamma = nn.Parameter(torch.tensor(0.0))
            self.register_buffer("pos_frac", torch.arange(T_max, dtype=torch.float32) / max(T_max, 1))

    def forward(self, z):
        z_used = z[..., :self.d_eff]
        B, T, _ = z_used.shape
        sigma = torch.exp(self.log_sigma)
        z_sq = z_used.pow(2).sum(-1, keepdim=True)
        mu_sq = self.mu.pow(2).sum(-1)
        cross = torch.einsum("btd,rd->btr", z_used, self.mu)
        D_sq = (z_sq + mu_sq.unsqueeze(0).unsqueeze(0) - 2 * cross).clamp_min(0.0)
        sigma_sq = sigma.pow(2).unsqueeze(0).unsqueeze(0) + 1e-8
        if self.use_temporal_bandwidth:
            gamma = F.softplus(self.temporal_gamma)
            temporal_scale = 1.0 + gamma * self.pos_frac[:T]
            sigma_sq = sigma_sq * temporal_scale.view(1, T, 1)

        if self.rbf_weights is not None:
            alpha = F.softmax(self.rbf_weights, dim=0)
            K_gaussian = torch.exp(-D_sq / (2 * sigma_sq))
            D_abs = torch.sqrt(D_sq + 1e-8)
            sigma_expand = sigma.unsqueeze(0).unsqueeze(0) + 1e-8
            K_laplacian = torch.exp(-D_abs / sigma_expand)
            K_matern = (1.0 + math.sqrt(3.0) * D_abs / sigma_expand) * torch.exp(-math.sqrt(3.0) * D_abs / sigma_expand)
            phi = alpha[0] * K_gaussian + alpha[1] * K_laplacian + alpha[2] * K_matern
        else:
            phi = torch.exp(-D_sq / (2 * sigma_sq))
        if self.rowsum_norm:
            phi = phi / phi.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return phi


# ============================================================================
# === HEAD MIXING ============================================================
# ============================================================================


class HeadScaler(nn.Module):
    """Per-head scalar weights (softmax-normalized)."""

    def __init__(self, M):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(M))


# ============================================================================
# === SSM ====================================================================
# ============================================================================


def parallel_scan_real(a_r, a_i, x_r, x_i, h0_r=None, h0_i=None):
    # Inject carry-in state: h[0] = a[0]*h0 + x[0] instead of h[0] = x[0]
    if h0_r is not None:
        x_r = x_r.clone()
        x_i = x_i.clone()
        # Complex multiply: a[0]*h0 = (a_r*h0_r - a_i*h0_i) + i*(a_r*h0_i + a_i*h0_r)
        x_r[:, 0] = x_r[:, 0] + a_r[:, 0] * h0_r - a_i[:, 0] * h0_i
        x_i[:, 0] = x_i[:, 0] + a_r[:, 0] * h0_i + a_i[:, 0] * h0_r
    hr, hi = x_r, x_i
    ar, ai = a_r, a_i
    T = x_r.shape[1]
    stride = 1
    while stride < T:
        hr_head, hi_head = hr[:, :stride], hi[:, :stride]
        ar_head, ai_head = ar[:, :stride], ai[:, :stride]
        hr_p, hi_p = hr[:, :-stride], hi[:, :-stride]
        ar_c, ai_c = ar[:, stride:], ai[:, stride:]
        ar_p, ai_p = ar[:, :-stride], ai[:, :-stride]
        hr = torch.cat([hr_head, ar_c * hr_p - ai_c * hi_p + hr[:, stride:]], dim=1)
        hi = torch.cat([hi_head, ar_c * hi_p + ai_c * hr_p + hi[:, stride:]], dim=1)
        ar = torch.cat([ar_head, ar_c * ar_p - ai_c * ai_p], dim=1)
        ai = torch.cat([ai_head, ar_c * ai_p + ai_c * ar_p], dim=1)
        stride *= 2
    return hr, hi


class ComplexSSM(nn.Module):
    def __init__(self, d_in, R_s, T_max=1024):
        super().__init__()
        self.R_s = R_s
        self.lambda_raw = nn.Parameter(torch.zeros(R_s))
        log_min, log_max = math.log(1.0 / T_max), math.log(math.pi)
        # Spectral init: blend Weyl's law (Stäckel FP eigenvalues) with log-spaced
        n = torch.arange(1, R_s + 1, dtype=torch.float32)
        weyl_freqs = n * math.pi / R_s
        log_freqs = torch.exp(torch.linspace(log_min, log_max, R_s))
        self.omega = nn.Parameter(0.5 * weyl_freqs + 0.5 * log_freqs)
        self.W_proj = nn.Linear(d_in, R_s, bias=False)
        self.W_res = nn.Linear(d_in, 2 * R_s, bias=False)
        self.ln = nn.LayerNorm(2 * R_s)

    def forward(self, u, h0_r=None, h0_i=None):
        B, T, _ = u.shape
        lam = -torch.sigmoid(self.lambda_raw) * 5.0

        # 1. Time vector
        t_arange = torch.arange(T, device=u.device, dtype=torch.float32)

        # 2. Impulse response f_t = a^t (where a = mag * e^{i*omega})
        decay = torch.exp(lam.unsqueeze(1) * t_arange.unsqueeze(0))  # [R_s, T]
        phase = self.omega.unsqueeze(1) * t_arange.unsqueeze(0)       # [R_s, T]
        f_r = decay * torch.cos(phase)
        f_i = decay * torch.sin(phase)

        # 3. FFT causal convolution (float32 for numerical safety)
        u_proj = self.W_proj(u)
        u_f32 = u_proj.transpose(1, 2).to(torch.float32)  # [B, R_s, T]
        f_r_f32 = f_r.to(torch.float32).unsqueeze(0)       # [1, R_s, T]
        f_i_f32 = f_i.to(torch.float32).unsqueeze(0)       # [1, R_s, T]

        n_fft = 2 * T  # pad to 2T to prevent circular wrap-around
        U_f = torch.fft.rfft(u_f32, n=n_fft)
        F_r_f = torch.fft.rfft(f_r_f32, n=n_fft)
        F_i_f = torch.fft.rfft(f_i_f32, n=n_fft)

        h_r_f32 = torch.fft.irfft(U_f * F_r_f, n=n_fft)[..., :T]
        h_i_f32 = torch.fft.irfft(U_f * F_i_f, n=n_fft)[..., :T]

        # 4. Zero-input response: h0 decays as a^(t+1)
        if h0_r is not None and h0_i is not None:
            t_zir = t_arange + 1.0
            decay_zir = torch.exp(lam.unsqueeze(1) * t_zir.unsqueeze(0)).to(torch.float32)
            phase_zir = (self.omega.unsqueeze(1) * t_zir.unsqueeze(0)).to(torch.float32)
            f_r_zir = decay_zir * torch.cos(phase_zir)
            f_i_zir = decay_zir * torch.sin(phase_zir)
            h0_r_f32 = h0_r.to(torch.float32).unsqueeze(-1)
            h0_i_f32 = h0_i.to(torch.float32).unsqueeze(-1)
            h_r_f32 = h_r_f32 + h0_r_f32 * f_r_zir.unsqueeze(0) - h0_i_f32 * f_i_zir.unsqueeze(0)
            h_i_f32 = h_i_f32 + h0_r_f32 * f_i_zir.unsqueeze(0) + h0_i_f32 * f_r_zir.unsqueeze(0)

        # 5. Back to original dtype
        h_r = h_r_f32.transpose(1, 2).to(u.dtype)  # [B, T, R_s]
        h_i = h_i_f32.transpose(1, 2).to(u.dtype)  # [B, T, R_s]

        ssm_out = torch.cat([h_r, h_i], dim=-1)
        final_state = (h_r[:, -1].detach(), h_i[:, -1].detach())
        return self.ln(ssm_out + self.W_res(u)), final_state


# ============================================================================
# === CAUSAL SELF-ATTENTION ==================================================
# ============================================================================


def _build_rope_cache(T_max, d_head, device="cpu"):
    pos = torch.arange(T_max, device=device).unsqueeze(1)
    dim = torch.arange(0, d_head, 2, device=device).float()
    freqs = 1.0 / (10000.0 ** (dim / d_head))
    angles = pos * freqs
    return torch.cos(angles), torch.sin(angles)


def _apply_rope(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos_t = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)


class CausalSelfAttention(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, d_model, n_heads=4, T_max=1024, q_gain_init=0.0,
                 qk_mode="full", d_proj=32):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qk_mode = qk_mode

        if qk_mode == "diagonal":
            # V1: per-head diagonal weight vectors for Q/K
            self.w_q = nn.Parameter(torch.ones(n_heads, self.d_head))
            self.w_k = nn.Parameter(torch.ones(n_heads, self.d_head))
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.qkv = None
        elif qk_mode == "per_head_proj":
            # V2a: per-head linear projections for Q/K
            self.d_proj = d_proj
            self.q_projs = nn.Parameter(torch.randn(n_heads, d_model, d_proj) * (d_model ** -0.5))
            self.k_projs = nn.Parameter(torch.randn(n_heads, d_model, d_proj) * (d_model ** -0.5))
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.qkv = None
        else:  # "full"
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

        rope_d = d_proj if qk_mode == "per_head_proj" else self.d_head
        cos, sin = _build_rope_cache(T_max, rope_d)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        if q_gain_init > 0:
            self.q_gain = nn.Parameter(torch.full((n_heads,), q_gain_init))
        else:
            self.q_gain = None

    def forward(self, x):
        B, T, C = x.shape

        if self.qk_mode == "diagonal":
            # V1: diagonal QK — reshape input into heads, scale element-wise
            x_heads = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q = x_heads * self.w_q[None, :, None, :]
            k = x_heads * self.w_k[None, :, None, :]
            v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)
        elif self.qk_mode == "per_head_proj":
            # V2a: per-head linear projections
            q = torch.einsum('btd,hdo->bhto', x, self.q_projs)
            k = torch.einsum('btd,hdo->bhto', x, self.k_projs)
            v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)
        else:
            q, k, v = self.qkv(x).split(C, dim=2)
            q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            q = _apply_rope(q, self.rope_cos, self.rope_sin)
            k = _apply_rope(k, self.rope_cos, self.rope_sin)

        if self.q_gain is not None:
            q = q * self.q_gain[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # B4: XSA — subtract self-value alignment (orthogonal projection)
        v_norm = F.normalize(v, dim=-1)
        self_align = (y * v_norm).sum(dim=-1, keepdim=True) * v_norm
        y = y - self_align
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ============================================================================
# === MODEL ==================================================================
# ============================================================================


def _soft_cap(x, cap):
    if cap <= 0:
        return x.clamp(-30, 30)
    return cap * torch.tanh(x / cap)


def _apply_phi_norm(phi, mode):
    """Data-dependent normalization of kernel design matrix rows (from IGL)."""
    if mode == "none":
        return phi
    elif mode == "softmax":
        return F.softmax(phi, dim=-1)
    elif mode == "l2":
        return F.normalize(phi, p=2, dim=-1)
    elif mode == "rowsum":
        return phi / phi.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown phi_norm mode: {mode}")


class HybridAtlasBlock(nn.Module):
    """Hybrid block with 5 VP heads: Nyström spherical, Gabor, Laplacian,
    Tucker GL cross-product, and linear kernel."""

    def __init__(self, d_in, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in

        # 3 independent encoders (asymmetric widths)
        H_sph = cfg.H
        H_gab = cfg.H_gabor if cfg.H_gabor > 0 else cfg.H
        H_lap = cfg.H_laplacian if cfg.H_laplacian > 0 else cfg.H
        self.encoders = nn.ModuleList([
            ChartEncoder(d_in, H_sph, cfg.d_max, n_hidden=cfg.n_encoder_hidden, use_stiefel_scale=cfg.use_stiefel_scale),
            ChartEncoder(d_in, H_gab, cfg.d_max, n_hidden=cfg.n_encoder_hidden, use_stiefel_scale=cfg.use_stiefel_scale),
            ChartEncoder(d_in, H_lap, cfg.d_max, n_hidden=cfg.n_encoder_hidden, use_stiefel_scale=cfg.use_stiefel_scale),
        ])

        # Dual encoder for Nyström head (Ψ_K, D1 from Stäckel theory)
        if cfg.use_dual_encoder_nystrom:
            H_dual = cfg.H_dual_nystrom if cfg.H_dual_nystrom > 0 else cfg.H
            self.encoder_nystrom_key = ChartEncoder(
                d_in, H_dual, cfg.d_max, n_hidden=cfg.n_encoder_hidden,
                use_stiefel_scale=cfg.use_stiefel_scale)
        else:
            self.encoder_nystrom_key = None

        # resid_mix: per-encoder learnable blend
        n_resid = 4 if cfg.use_dual_encoder_nystrom else 3
        if cfg.use_resid_mix:
            self.resid_mix = nn.ParameterList([
                nn.Parameter(torch.stack([
                    torch.full((d_in,), 4.0),
                    torch.full((d_in,), -4.0),
                ]))
                for _ in range(n_resid)
            ])

        # Head 0: Nyström causal spherical
        self.causal_basis = NystromCausalBasis(
            cfg.R_nystrom, cfg.T,
            learnable_gegenbauer=cfg.use_learnable_gegenbauer,
            use_temporal_decay=cfg.use_temporal_decay,
            use_dual_nystrom=cfg.use_dual_nystrom,
            use_softmax=cfg.use_nystrom_softmax,
            use_rowsum=cfg.use_nystrom_rowsum,
            d_max=cfg.d_max)
        R_causal = cfg.R_nystrom + 1  # +1 for bias column

        # V1: Learned readout weights
        self.W_causal = nn.Parameter(torch.zeros(R_causal, cfg.V))

        # Head 1: Gabor (fixed anchors)
        self.gabor_basis = GaborBasis(cfg.R, cfg.d_max, d_eff=cfg.d_eff_gabor,
                                      envelope_norm=cfg.use_gabor_envelope_norm)
        self.W_gabor = nn.Parameter(torch.zeros(cfg.R, cfg.V))

        # Head 2: Laplacian (fixed anchors)
        self.laplacian_basis = LaplacianBasis(
            cfg.R, cfg.d_max, learnable_rbf=cfg.use_learnable_rbf,
            d_eff=cfg.d_eff_laplacian, use_temporal_bandwidth=cfg.use_temporal_bandwidth,
            T_max=cfg.T, rowsum_norm=cfg.use_laplacian_rowsum)
        R_laplacian = cfg.R
        self.W_laplacian = nn.Parameter(torch.zeros(R_laplacian, cfg.V))

        # Head 3: Tucker GL (Gabor ⊙ Laplacian)
        n_heads = 3
        if cfg.use_tucker_gl:
            self.R_tucker_gl = min(cfg.R, R_laplacian)
            self.W_tucker_gl = nn.Parameter(torch.zeros(self.R_tucker_gl, cfg.V))
            n_heads += 1

        # Head 4: Linear kernel (raw encoder output)
        if cfg.use_linear_kernel_head:
            self.W_linear = nn.Parameter(torch.zeros(cfg.d_max, cfg.V))
            n_heads += 1

        self.head_scaler = HeadScaler(n_heads)

        # Post-VP MLP: nonlinear correction after HeadScaler mixing
        if cfg.use_post_vp_mlp:
            self.post_vp_mlp = nn.Sequential(
                nn.Linear(cfg.V, cfg.post_vp_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.post_vp_hidden, cfg.V, bias=False),
            )
            nn.init.zeros_(self.post_vp_mlp[2].weight)  # identity at step 0
        else:
            self.post_vp_mlp = None

        # Option D: SSM + Attention on concatenated z (Stäckel coordinates)
        d_z_cat = 3 * cfg.d_max  # 384
        d_temporal = 2 * cfg.R_s  # 256

        # SSM on concatenated z (Stäckel coordinates)
        if cfg.use_ssm:
            self.ssm = ComplexSSM(d_z_cat, cfg.R_s, T_max=cfg.T)  # 384 → 256
        else:
            self.ssm = None
            self.proj_in = nn.Linear(d_z_cat, d_temporal, bias=False)
            self.proj_ln = nn.LayerNorm(d_temporal)

        # Attention on concatenated z (projection always needed for Option D)
        self.attn_proj_in = nn.Linear(d_z_cat, d_temporal, bias=False)
        self.attn_proj_ln = nn.LayerNorm(d_temporal)
        if cfg.use_attn:
            self.attn_norms = nn.ModuleList([nn.LayerNorm(d_temporal) for _ in range(cfg.n_attn_layers)])
            self.attns = nn.ModuleList([
                CausalSelfAttention(d_temporal, cfg.attn_heads, T_max=cfg.T,
                                    q_gain_init=cfg.q_gain_init,
                                    qk_mode=cfg.attn_qk_mode, d_proj=cfg.attn_d_proj)
                for _ in range(cfg.n_attn_layers)
            ])
        else:
            self.attns = None

        # Gate to combine SSM + Attention
        self.parallel_gate = nn.Parameter(torch.tensor(0.5))

        # Project temporal output only (mixed routes directly to logits via geom_scale)
        self.final_proj = nn.Linear(d_temporal, d_temporal, bias=False)
        self.final_ln = nn.LayerNorm(d_temporal)

        # Per-dimension residual scaling
        if cfg.use_residual_scale:
            self.ssm_scale = nn.Parameter(torch.ones(d_temporal))
            self.attn_scales = nn.ParameterList([
                nn.Parameter(torch.ones(d_temporal)) for _ in range(cfg.n_attn_layers)
            ])

        # Second SSM+Attn pass: project back to z-dim, run again
        d_z_cat = 3 * cfg.d_max  # 384
        self.temporal_proj_back = nn.Linear(d_temporal, d_z_cat, bias=False)
        self.temporal_proj_back_ln = nn.LayerNorm(d_z_cat)

        # Second SSM on z (if SSM enabled)
        if cfg.use_ssm:
            self.ssm2 = ComplexSSM(d_z_cat, cfg.R_s, T_max=cfg.T)
        else:
            self.ssm2 = None
            self.proj_in2 = nn.Linear(d_z_cat, d_temporal, bias=False)
            self.proj_ln2 = nn.LayerNorm(d_temporal)

        # Second attention on z
        if cfg.use_attn:
            self.attn_proj_in2 = nn.Linear(d_z_cat, d_temporal, bias=False)
            self.attn_proj_ln2 = nn.LayerNorm(d_temporal)
            self.attn_norms2 = nn.ModuleList([nn.LayerNorm(d_temporal) for _ in range(cfg.n_attn_layers)])
            self.attns2 = nn.ModuleList([
                CausalSelfAttention(d_temporal, cfg.attn_heads, T_max=cfg.T,
                                    q_gain_init=cfg.q_gain_init,
                                    qk_mode=cfg.attn_qk_mode, d_proj=cfg.attn_d_proj)
                for _ in range(cfg.n_attn_layers)
            ])
        else:
            self.attn_proj_in2 = nn.Linear(d_z_cat, d_temporal, bias=False)
            self.attn_proj_ln2 = nn.LayerNorm(d_temporal)
            self.attns2 = None

        # Second gate
        self.parallel_gate2 = nn.Parameter(torch.tensor(0.5))

        # Per-dimension residual scaling for second pass
        if cfg.use_residual_scale:
            self.ssm_scale2 = nn.Parameter(torch.ones(d_temporal))
            self.attn_scales2 = nn.ParameterList([
                nn.Parameter(torch.ones(d_temporal)) for _ in range(cfg.n_attn_layers)
            ])

    def _compute_all_phi(self, x, x0=None):
        d_max = self.cfg.d_max

        def _encoder_input(m, x_cur):
            if self.cfg.use_resid_mix and x0 is not None and x0.shape[-1] == x_cur.shape[-1]:
                ab = torch.sigmoid(self.resid_mix[m])
                return ab[0] * x_cur + ab[1] * x0
            return x_cur

        # Head 0: causal spherical (with optional asymmetric dual encoder)
        z0 = self.encoders[0](_encoder_input(0, x))
        z_extra = []
        if self.encoder_nystrom_key is not None:
            z0_key = self.encoder_nystrom_key(_encoder_input(3, x))
            phi0 = self.causal_basis.forward_asymmetric(z0, z0_key)
            z_extra.append(z0_key)
        else:
            phi0 = self.causal_basis(z0)

        # Head 1: Gabor (with data-dependent normalization)
        z1 = self.encoders[1](_encoder_input(1, x))
        phi1 = self.gabor_basis(z1)
        phi1 = _apply_phi_norm(phi1, self.cfg.phi_norm_gabor)

        # Head 2: Laplacian (with data-dependent normalization)
        z2 = self.encoders[2](_encoder_input(2, x))
        phi2 = self.laplacian_basis(z2)
        phi2 = _apply_phi_norm(phi2, self.cfg.phi_norm_laplacian)

        return [z0, z1, z2], [phi0, phi1, phi2], z_extra

    def _readout_and_temporal(self, z_list, phis, ssm_state=None):
        """Option D: SSM + Attention on z, kernel readout separate, then combine."""
        cap = self.cfg.logit_cap
        alpha = F.softmax(self.head_scaler.log_weights, dim=0)

        # Kernel readout (same as before)
        phi_parts = [phis[0], phis[1], phis[2]]
        w_parts = [alpha[0] * self.W_causal, alpha[1] * self.W_gabor,
                   alpha[2] * self.W_laplacian]
        idx = 3
        if self.cfg.use_tucker_gl:
            R_gl = self.R_tucker_gl
            phi_parts.append(phis[1][:, :, :R_gl] * phis[2][:, :, :R_gl])
            w_parts.append(alpha[idx] * self.W_tucker_gl)
            idx += 1
        if self.cfg.use_linear_kernel_head:
            phi_parts.append(z_list[0])
            w_parts.append(alpha[idx] * self.W_linear)
        Phi_all = torch.cat(phi_parts, dim=-1)
        W_all = torch.cat(w_parts, dim=0)
        mixed = _soft_cap(Phi_all @ W_all, cap)

        if self.post_vp_mlp is not None:
            mixed = mixed + self.post_vp_mlp(mixed)

        # Concatenate encoder outputs (Stäckel coordinates)
        z_cat = torch.cat(z_list, dim=-1)  # [B, T, 384]

        # Branch 1: SSM on z (long-range temporal in Stäckel space)
        new_ssm_state = None
        if self.ssm is not None:
            h0_r, h0_i = ssm_state if ssm_state is not None else (None, None)
            H_ssm, new_ssm_state = self.ssm(z_cat, h0_r=h0_r, h0_i=h0_i)
            if self.cfg.use_residual_scale:
                H_ssm = self.ssm_scale * H_ssm
        else:
            H_ssm = self.proj_ln(self.proj_in(z_cat))

        # Branch 2: Attention on z (local precision in Stäckel space)
        H_attn = self.attn_proj_ln(self.attn_proj_in(z_cat))
        if self.attns is not None:
            for idx_a, (norm, attn) in enumerate(zip(self.attn_norms, self.attns)):
                attn_out = attn(norm(H_attn))
                if self.cfg.use_residual_scale:
                    attn_out = self.attn_scales[idx_a] * attn_out
                H_attn = H_attn + attn_out

        # Combine SSM + Attention with learned gate (first pass)
        gate = torch.sigmoid(self.parallel_gate)
        H_temporal = gate * H_ssm + (1 - gate) * H_attn  # [B, T, 256]

        # === Second SSM+Attn pass ===
        # Project back to z-dim
        z_cat2 = self.temporal_proj_back_ln(self.temporal_proj_back(H_temporal)) + z_cat  # residual

        # Branch 1: SSM2
        if self.ssm2 is not None:
            H_ssm2, _ = self.ssm2(z_cat2)
            if self.cfg.use_residual_scale:
                H_ssm2 = self.ssm_scale2 * H_ssm2
        else:
            H_ssm2 = self.proj_ln2(self.proj_in2(z_cat2))

        # Branch 2: Attn2
        H_attn2 = self.attn_proj_ln2(self.attn_proj_in2(z_cat2))
        if hasattr(self, 'attns2') and self.attns2 is not None:
            for idx_a, (norm, attn) in enumerate(zip(self.attn_norms2, self.attns2)):
                attn_out = attn(norm(H_attn2))
                if self.cfg.use_residual_scale:
                    attn_out = self.attn_scales2[idx_a] * attn_out
                H_attn2 = H_attn2 + attn_out

        gate2 = torch.sigmoid(self.parallel_gate2)
        H_temporal = gate2 * H_ssm2 + (1 - gate2) * H_attn2  # refined [B, T, 256]

        # Combine temporal output with kernel readout
        H_out = self.final_ln(self.final_proj(H_temporal))  # mixed routes to logits only

        return H_out, mixed, {
            "z_list": z_list,
            "alpha": alpha.detach(),
            "ssm_state": new_ssm_state,
        }

    def forward_phase_b(self, x, x0=None, ssm_state=None):
        z_list, phis, z_extra = self._compute_all_phi(x, x0=x0)
        H_out, mixed, info = self._readout_and_temporal(z_list, phis, ssm_state=ssm_state)
        info["z_extra"] = z_extra
        return H_out, mixed, info

    def forward_phase_b_from_cached(self, z_list, phis, ssm_state=None):
        return self._readout_and_temporal(z_list, phis, ssm_state=ssm_state)


class HybridAtlasLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.V, cfg.d_emb)
        nn.init.normal_(self.embedding.weight, 0, 0.005)  # Crushed init for high embedding LR

        self.smear = SmearGate(cfg.d_emb) if cfg.use_smear_gate else None
        self.bigram = BigramHashEmbedding(cfg.bigram_vocab_size, cfg.bigram_dim, cfg.d_emb) \
            if cfg.bigram_vocab_size > 0 else None

        self.blocks = nn.ModuleList()
        for i in range(cfg.L):
            self.blocks.append(HybridAtlasBlock(cfg.d_emb if i == 0 else 2 * cfg.R_s, cfg))
        self.W_out = nn.Linear(2 * cfg.R_s, cfg.d_emb, bias=False) if 2 * cfg.R_s != cfg.d_emb else nn.Identity()

        # U-Net skip: project Block 1 encoder z's → Block 2 input dim
        if cfg.use_unet_skip and cfg.L >= 2:
            skip_in = 3 * cfg.d_max
            skip_out = 2 * cfg.R_s
            self.skip_proj = nn.Linear(skip_in, skip_out, bias=False)
            nn.init.zeros_(self.skip_proj.weight)
        else:
            self.skip_proj = None

        self.geom_scale = nn.Parameter(torch.zeros(1))

        if cfg.use_ortho_init:
            self._apply_ortho_init()
            # Re-zero skip_proj (ortho init overwrites the intentional zero-init)
            if self.skip_proj is not None:
                nn.init.zeros_(self.skip_proj.weight)

    def _apply_ortho_init(self):
        num_layers = self.cfg.L * 2
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and module.weight.ndim == 2:
                if min(module.weight.shape) >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj" in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * max(num_layers, 1)))

    def _embed(self, tokens):
        x = self.embedding(tokens)
        if self.bigram is not None:
            x = x + self.bigram(tokens)
        x = F.rms_norm(x, (x.size(-1),))
        if self.smear is not None:
            x = self.smear(x)
        return x

    def forward(self, tokens, ssm_states=None):
        x0 = self._embed(tokens)
        x = x0
        all_info = {}
        new_ssm_states = {}
        accum_logits = 0.0
        for i, block in enumerate(self.blocks):
            block_ssm = ssm_states[i] if ssm_states is not None and i in ssm_states else None
            H_out, mixed, info = block.forward_phase_b(x, x0=x0, ssm_state=block_ssm)
            accum_logits = accum_logits + mixed
            all_info[f"block_{i}"] = info
            new_ssm_states[i] = info.get("ssm_state")
            x = H_out
            if i == 0 and self.skip_proj is not None:
                block1_z = torch.cat(info["z_list"], dim=-1).detach()
                x = x + self.skip_proj(block1_z)
        temporal_logits = self.W_out(x) @ self.embedding.weight.T
        logits = _soft_cap(temporal_logits + self.geom_scale * accum_logits, self.cfg.logit_cap)
        return logits, all_info, new_ssm_states

    def compile_submodules(self):
        """Compile pure-compute submodules."""
        if self.bigram is not None:
            self.bigram = torch.compile(self.bigram)
        if self.smear is not None:
            self.smear = torch.compile(self.smear)
        for block in self.blocks:
            for i, enc in enumerate(block.encoders):
                block.encoders[i] = torch.compile(enc)
            block.causal_basis = torch.compile(block.causal_basis)
            block.gabor_basis = torch.compile(block.gabor_basis)
            block.laplacian_basis = torch.compile(block.laplacian_basis)
            if block.encoder_nystrom_key is not None:
                block.encoder_nystrom_key = torch.compile(block.encoder_nystrom_key)
            if block.ssm is not None:
                block.ssm = torch.compile(block.ssm)
            if block.attns is not None:
                for j, attn in enumerate(block.attns):
                    block.attns[j] = torch.compile(attn)
            if hasattr(block, 'ssm2') and block.ssm2 is not None:
                block.ssm2 = torch.compile(block.ssm2)
            if hasattr(block, 'attns2') and block.attns2 is not None:
                for j, attn in enumerate(block.attns2):
                    block.attns2[j] = torch.compile(attn)
            if block.post_vp_mlp is not None:
                block.post_vp_mlp = torch.compile(block.post_vp_mlp)

    def stored_params(self):
        yield from self.parameters()

    def count_stored_params(self):
        return sum(p.numel() for p in self.stored_params())


# ============================================================================
# === QUANTIZATION ===========================================================
# ============================================================================


def quantize_to_int8(state_dict):
    q, scales = {}, {}
    for name, t in state_dict.items():
        if t.dtype in (torch.int32, torch.int64, torch.long):
            q[name] = t.to(torch.int16)
            scales[name] = 1.0
        elif t.numel() == 0:
            q[name] = t
            scales[name] = 1.0
        elif t.numel() <= 512:
            q[name] = t.float().half()
            scales[name] = 1.0
        else:
            t = t.float()
            mx = t.abs().max().item()
            s = mx / 127.0 if mx > 0 else 1.0
            q[name] = (t / s).round().clamp(-127, 127).to(torch.int8)
            scales[name] = s
    return {"tensors": q, "scales": scales}


def dequantize(q_data):
    sd = {}
    for name, qt in q_data["tensors"].items():
        s = q_data["scales"][name]
        if qt.dtype == torch.int16:
            sd[name] = qt.to(torch.long)
        elif qt.dtype == torch.float16:
            sd[name] = qt.float()
        else:
            sd[name] = qt.float() * s
    return sd


def save_artifact(model, path):
    sd = {}
    full_sd = model.state_dict()
    for name, t in full_sd.items():
        if "rope_cos" in name or "rope_sin" in name:
            continue
        # Strip _orig_mod. prefix from compiled submodules
        clean_name = name.replace("_orig_mod.", "")
        sd[clean_name] = t.detach().cpu()
    q = quantize_to_int8(sd)
    buf = io.BytesIO()
    torch.save(q, buf)
    compressed = zstd.ZstdCompressor(level=22).compress(buf.getvalue())
    Path(path).write_bytes(compressed)
    return len(compressed)


def load_artifact(path, model):
    compressed = Path(path).read_bytes()
    q = torch.load(io.BytesIO(zstd.ZstdDecompressor().decompress(compressed)), weights_only=False)
    model.load_state_dict(dequantize(q), strict=False)
    return model


# ============================================================================
# === TRAINING ===============================================================
# ============================================================================


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _newton_schulz(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T
    X = G / (G.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0, scale_cap=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, scale_cap=scale_cap)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns = group["ns_steps"]
            wd = group["weight_decay"]
            cap = group["scale_cap"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                if group["nesterov"]:
                    update = g + mu * buf
                else:
                    update = buf
                if update.ndim == 2:
                    update = _newton_schulz(update, steps=ns)
                    raw_scale = max(1, update.size(0) / update.size(1)) ** 0.5
                    scale = min(raw_scale, cap) if cap > 0 else raw_scale
                    update = update * scale
                if wd > 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)


def build_optimizer(model, cfg):
    muon_params = []
    enc_non_matrix_p, basis_p, ssm_p, gate_p, embed_p, other_p = [], [], [], [], [], []
    gate_keywords = ("head_scaler", "q_gain", "ssm_scale", "attn_scale",
                     "smear", "resid_mix", "gegenbauer", "rbf_weights")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Surgical Muon: encoders + attention/SSM 2D matrices (not readouts/embeddings)
        is_encoder_2d = "encoder" in name and p.ndim == 2
        is_attn_2d = ("attn" in name or "qkv" in name or "proj" in name) and p.ndim == 2
        is_ssm_proj_2d = ("W_proj" in name or "W_res" in name) and p.ndim == 2
        is_readout = name.startswith("W_") or "W_causal" in name or "W_gabor" in name or "W_laplacian" in name or "W_tucker" in name or "W_linear" in name or "W_out" in name
        is_embedding = "embedding" in name

        if is_encoder_2d or ((is_attn_2d or is_ssm_proj_2d) and not is_readout and not is_embedding):
            muon_params.append(p)
        elif "encoder" in name:
            enc_non_matrix_p.append(p)
        elif "gabor_basis" in name or "laplacian_basis" in name:
            basis_p.append(p)
        elif "ssm" in name or "lambda_raw" in name or "omega" in name:
            ssm_p.append(p)
        elif any(k in name for k in gate_keywords):
            gate_p.append(p)
        elif "embedding" in name:
            embed_p.append(p)
        else:
            other_p.append(p)

    adam_groups = [
        {"params": enc_non_matrix_p, "lr": cfg.lr_encoder, "weight_decay": cfg.wd_encoder},
        {"params": basis_p, "lr": cfg.lr_basis, "weight_decay": cfg.wd_basis},
        {"params": ssm_p, "lr": cfg.lr_ssm, "weight_decay": cfg.wd_ssm},
        {"params": gate_p, "lr": cfg.lr_other, "weight_decay": 0.0},
        {"params": other_p, "lr": cfg.lr_other, "weight_decay": cfg.wd_other},
        {"params": embed_p, "lr": 0.05, "weight_decay": 0.0},  # Embedding turbocharger
    ]
    adam_groups = [g for g in adam_groups if g["params"]]

    muon_opt = Muon(muon_params, lr=cfg.lr_muon, momentum=cfg.muon_momentum_start,
                    weight_decay=cfg.muon_wd, scale_cap=cfg.muon_scale_cap) if muon_params else None
    adam_opt = torch.optim.AdamW(adam_groups) if adam_groups else None
    return muon_opt, adam_opt


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


@torch.inference_mode()
def evaluate(model, val_iter, n_batches=10):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for _ in range(n_batches):
        x, y = next(val_iter)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            logits, _, _ = model(x)
        total_loss += F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item() * y.numel()
        total_tok += y.numel()
    model.train()
    avg = total_loss / total_tok
    return {"val_loss": avg, "val_bpb": avg / math.log(2)}


@torch.inference_mode()
def evaluate_sliding(model, val_data, cfg, device, stride=256,
                     rank: int = 0, world_size: int = 1, eval_batch: int = 64):
    """Batched sliding window evaluation with distributed support.

    Batches multiple windows for efficient GPU utilization.
    SSM carry-over is disabled when batched (windows processed independently).
    """
    model.eval()
    T = cfg.T
    total_tokens = len(val_data) - 1
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tok = torch.zeros((), device=device, dtype=torch.float64)

    # Compute all window starts, partition across ranks
    all_windows = list(range(0, total_tokens, stride))
    my_windows = all_windows[rank::world_size]

    # Process in batches (pad last batch to eval_batch to avoid recompilation)
    for batch_start in range(0, len(my_windows), eval_batch):
        batch_ws = my_windows[batch_start:batch_start + eval_batch]
        B_cur = len(batch_ws)

        x_batch = torch.zeros(eval_batch, T, dtype=val_data.dtype)
        y_batch = torch.zeros(eval_batch, T, dtype=val_data.dtype)
        wlens = []

        for i, ws in enumerate(batch_ws):
            we = min(ws + T, total_tokens)
            wlen = we - ws
            wlens.append(wlen)
            x_batch[i, :wlen] = val_data[ws:we]
            y_batch[i, :wlen] = val_data[ws + 1:we + 1]

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            logits, _, _ = model(x_batch)
        # Per-token NLL: [eval_batch, T] — only first B_cur rows are real
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                              y_batch.reshape(-1), reduction='none').reshape(eval_batch, T)

        for i, ws in enumerate(batch_ws):
            wlen = wlens[i]
            score_start = 0 if ws == 0 else max(wlen - stride, 0)
            scored = nll[i, score_start:wlen]
            total_loss += scored.sum()
            total_tok += scored.numel()

    # All-reduce across ranks
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tok, op=dist.ReduceOp.SUM)

    model.train()
    avg = (total_loss / total_tok.clamp(min=1)).item()
    return {"val_loss": avg, "val_bpb": avg / math.log(2)}


# ============================================================================
# === TOKENIZER-AGNOSTIC METRIC (competition bpb) ===========================
# ============================================================================


def build_sentencepiece_luts(sp, vocab_size, device):
    """Build byte-count lookup tables from SentencePiece model.

    Returns (base_bytes, has_leading_space, is_boundary_token) tensors.
    Copied from competition reference code (train_gpt.py baseline).
    """
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


class EvalWrapper(nn.Module):
    """Wraps MHALM model to match the competition's model(x, y) → loss API.

    Handles V_clamp internally: receives original token IDs (for byte counting
    in eval_val), applies modulo remapping before model forward.
    """
    def __init__(self, model, V_clamp, V):
        super().__init__()
        self.model = model
        self.V_clamp = V_clamp
        self.V = V

    def forward(self, x, y):
        x_m = x % self.V_clamp if self.V_clamp else x
        y_m = y % self.V_clamp if self.V_clamp else y
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            logits, _, _ = self.model(x_m)
        return F.cross_entropy(logits.float().reshape(-1, self.V), y_m.reshape(-1))


def eval_val(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens,
    base_bytes_lut,
    has_leading_space_lut,
    is_boundary_token_lut,
) -> tuple:
    """Competition evaluation — copied verbatim from baseline train_gpt.py.

    Computes two metrics:
    - val_loss: token cross-entropy (natural log)
    - val_bpb: tokenizer-agnostic bits-per-byte (competition metric)
    """
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
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


def train(model, train_iter, val_iter, cfg, device):
    raw = model.module if hasattr(model, "module") else model
    muon_opt, adam_opt = build_optimizer(raw, cfg)
    model.train()

    use_time_progress = cfg.max_wallclock > 0 and cfg.total_steps == 0
    if cfg.total_steps > 0:
        est_total = cfg.total_steps
    elif use_time_progress:
        # Hardcode from validated H100 steady-state speed (100ms/step for d_max=128).
        # Avoids compile-warmup contamination 
        est_total = int(cfg.max_wallclock * 0.99 / 0.100)
    else:
        est_total = 50000
    step = 0
    t_start = time.time()
    t_log = time.time()
    best_val_bpb = float("inf")

    swa_state = None
    swa_count = 0

    is_main = not dist.is_initialized() or dist.get_rank() == 0

    stored_param_count = raw.count_stored_params()
    if is_main:
        print(f"model_params:{stored_param_count}")
        print(f"world_size:{1 if not dist.is_initialized() else dist.get_world_size()} "
              f"train_batch_tokens:{cfg.B * cfg.T * (dist.get_world_size() if dist.is_initialized() else 1)} "
              f"train_seq_len:{cfg.T} max_wallclock_seconds:{cfg.max_wallclock:.3f}")
        print(f"architecture:MHALM L={cfg.L} H={cfg.H} R={cfg.R} R_s={cfg.R_s} "
              f"bigram_vocab={cfg.bigram_vocab_size}")

    # AMP context
    if cfg.use_amp and device.startswith('cuda'):
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    while True:
        elapsed = time.time() - t_start
        if cfg.max_wallclock > 0 and elapsed > cfg.max_wallclock * 0.99:
            if is_main:
                print(f"\nWallclock limit approaching ({elapsed:.0f}s). Stopping.")
            break
        if cfg.total_steps > 0 and step >= cfg.total_steps:
            break

        # Refine est_total at step 400 using measured steady-state speed
        # Can only increase est_total (if actual speed is faster than 100ms assumption)
        if use_time_progress and step == 400 and step > 0:
            elapsed_so_far = time.time() - t_start
            remaining_time = cfg.max_wallclock * 0.99 - elapsed_so_far
            # Use last 200 steps to measure speed (avoids compile warmup)
            if hasattr(train, '_t_step_200'):
                ms_per_step = (time.time() - train._t_step_200) / 200 * 1000
                refined = step + int(remaining_time / (ms_per_step / 1000))
                if refined > est_total:
                    est_total = refined
                if is_main:
                    print(f"  Refined: {ms_per_step:.0f} ms/step → est_total={est_total}")
        if use_time_progress and step == 200:
            train._t_step_200 = time.time()

        # V1: Single forward-backward pass (no Phase A, no RLS)
        x, y = next(train_iter)
        if muon_opt:
            muon_opt.zero_grad()
        if adam_opt:
            adam_opt.zero_grad()

        with amp_ctx:
            logits, info, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Stäckel penalty — no .item() calls, keep as GPU tensors
        d_active = [cfg.d_max, cfg.d_eff_gabor, cfg.d_eff_laplacian]
        d_max_pairs = cfg.d_max * (cfg.d_max - 1)
        staekel_loss_t = torch.zeros((), device=device)
        if cfg.staekel_beta > 0 and d_max_pairs > 0 and info is not None:
            all_z = [z_h for bk in sorted(info.keys()) for z_h in info[bk]["z_list"]]
            # Include dual encoder outputs in penalty
            all_z += [z_h for bk in sorted(info.keys()) for z_h in info[bk].get("z_extra", [])]
            for idx, z_h in enumerate(all_z):
                head_idx = idx % 3 if idx < len(info) * 3 else -1  # -1 = dual encoder
                d_used = d_active[head_idx] if 0 <= head_idx < len(d_active) else cfg.d_max
                if d_used < 2:
                    continue  # Bug 4 guard: need ≥2 dims for off-diagonal
                z_flat = z_h[..., :d_used].reshape(-1, d_used)
                # Subsample for efficiency: 4096 tokens gives equivalent covariance estimate
                if z_flat.shape[0] > 4096:
                    z_flat = z_flat[torch.randint(z_flat.shape[0], (4096,), device=z_flat.device)]
                z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
                cov = (z_c.T @ z_c) / z_c.shape[0]
                d_used_pairs = d_used * (d_used - 1)
                off_diag_sq = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d_used_pairs
                beta_eff = cfg.staekel_beta * d_used_pairs / d_max_pairs
                loss = loss + beta_eff * off_diag_sq
                staekel_loss_t = staekel_loss_t + off_diag_sq.detach()

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None], cfg.grad_clip)

        # LR schedule
        if use_time_progress:
            time_frac = (time.time() - t_start) / (cfg.max_wallclock * 0.99)
            lr_step = int(time_frac * est_total)
        else:
            lr_step = step
        if adam_opt:
            for g in adam_opt.param_groups:
                if "initial_lr" not in g:
                    g["initial_lr"] = g["lr"]
                g["lr"] = cosine_lr(lr_step, est_total, cfg.warmup_steps, g["initial_lr"])
        if muon_opt:
            for g in muon_opt.param_groups:
                if "initial_lr" not in g:
                    g["initial_lr"] = g["lr"]
                g["lr"] = cosine_lr(lr_step, est_total, cfg.warmup_steps, g["initial_lr"])
            # Momentum warmup: ramp from start to end over warmup steps
            if cfg.muon_momentum_warmup > 0 and step < cfg.muon_momentum_warmup:
                cur_mu = cfg.muon_momentum_start + (cfg.muon_momentum_end - cfg.muon_momentum_start) * step / cfg.muon_momentum_warmup
            else:
                cur_mu = cfg.muon_momentum_end
            for g in muon_opt.param_groups:
                g["momentum"] = cur_mu

        if muon_opt:
            muon_opt.step()
        if adam_opt:
            adam_opt.step()

        # S-series: Stiefel enforcement on encoder w_out
        if cfg.use_stiefel_enforcement:
            for block in raw.blocks:
                for enc in block.encoders:
                    _newton_schulz_orthogonalize(enc.w_out.weight, cfg.stiefel_steps)
                if block.encoder_nystrom_key is not None:
                    _newton_schulz_orthogonalize(block.encoder_nystrom_key.w_out.weight, cfg.stiefel_steps)

        step += 1

        # SWA — rank 0 only, parameters only (not buffers)
        if cfg.swa_enabled and step % cfg.swa_every == 0 and is_main:
            if use_time_progress:
                progress = (time.time() - t_start) / (cfg.max_wallclock * 0.99)
            else:
                progress = step / est_total if est_total > 0 else 1.0
            if progress >= cfg.swa_start_frac:
                if swa_state is None:
                    swa_state = {name: p.detach().cpu().clone()
                                 for name, p in raw.named_parameters()}
                    swa_count = 1
                else:
                    for name, p in raw.named_parameters():
                        if name in swa_state:
                            swa_state[name] += p.detach().cpu()
                    swa_count += 1

        # Logging — .item() calls only here, not every step
        if step % cfg.log_every == 0 and is_main:
            loss_value = loss.item()
            train_time_ms = int((time.time() - t_start) * 1000)
            cur_step_avg = train_time_ms / max(step, 1)
            print(f"step:{step}/{est_total} train_loss:{loss_value:.4f} "
                  f"train_time:{train_time_ms}ms step_avg:{cur_step_avg:.2f}ms")
            t_log = time.time()

        if step % cfg.val_every == 0:
            vm = evaluate(model, val_iter)
            if dist.is_initialized():
                val_loss_t = torch.tensor(vm["val_loss"], device=device)
                dist.all_reduce(val_loss_t, op=dist.ReduceOp.AVG)
                vm["val_loss"] = val_loss_t.item()
                vm["val_bpb"] = vm["val_loss"] / math.log(2)
            if vm["val_bpb"] < best_val_bpb:
                best_val_bpb = vm["val_bpb"]
            if is_main:
                train_time_ms = int((time.time() - t_start) * 1000)
                cur_step_avg = train_time_ms / max(step, 1)
                # Estimate competition bpb: bits/token ÷ avg_bytes_per_token
                est_comp_bpb = vm['val_loss'] / (math.log(2) * 2.44)
                print(f"step:{step}/{est_total} val_loss:{vm['val_loss']:.4f} "
                      f"val_bpb:{vm['val_bpb']:.4f} est_bpb:{est_comp_bpb:.4f} "
                      f"train_time:{train_time_ms}ms step_avg:{cur_step_avg:.2f}ms")

    total_time = time.time() - t_start
    step_avg_ms = total_time / max(step, 1) * 1000
    if is_main:
        print(f"stopping_early: wallclock_cap train_time:{int(total_time*1000)}ms step:{step}/{est_total}")

    # Apply SWA
    if cfg.swa_enabled and swa_state is not None and swa_count > 1:
        if is_main:
            print(f"SWA: applying average of {swa_count} checkpoints")
        current_sd = raw.state_dict()
        for name, tensor in swa_state.items():
            if name in current_sd:
                current_sd[name] = (tensor / swa_count).to(dtype=current_sd[name].dtype, device=current_sd[name].device)
        raw.load_state_dict(current_sd, strict=True)

    return step, best_val_bpb, step_avg_ms, stored_param_count


# ============================================================================
# === MAIN ===================================================================
# ============================================================================


def setup_ddp():
    if "RANK" in os.environ:
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def main():
    parser = argparse.ArgumentParser(description="MHALM — Multi-Head Atlas Language Model")
    parser.add_argument("--mode", choices=["golf", "smoke"], default="golf")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--wallclock", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default="data/fineweb")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--R", type=int, default=None, help="Override R (anchor count)")
    parser.add_argument("--L", type=int, default=None, help="Override L (number of blocks)")
    parser.add_argument("--H", type=int, default=None, help="Override H (encoder hidden)")
    parser.add_argument("--H-gabor", type=int, default=None, help="Override H_gabor")
    parser.add_argument("--H-laplacian", type=int, default=None, help="Override H_laplacian")
    parser.add_argument("--no-tucker-gl", action="store_true")
    parser.add_argument("--no-linear-head", action="store_true")
    parser.add_argument("--no-temporal-bw", action="store_true")
    parser.add_argument("--no-staekel", action="store_true")
    parser.add_argument("--nystrom-softmax", action="store_true", help="N2: softmax normalisation")
    parser.add_argument("--nystrom-rowsum", action="store_true", help="N1: row-sum normalisation")
    parser.add_argument("--laplacian-rowsum", action="store_true", help="L1: Laplacian row-sum")
    parser.add_argument("--gabor-envelope-norm", action="store_true", help="G1: Gabor envelope norm")
    parser.add_argument("--symmetric-enc", action="store_true")
    parser.add_argument("--no-bigram", action="store_true", help="Disable BigramHash embedding")
    parser.add_argument("--bigram-vocab", type=int, default=None, help="BigramHash vocab size")
    parser.add_argument("--n-attn-layers", type=int, default=None, help="Override n_attn_layers")
    parser.add_argument("--d-eff-gabor", type=int, default=None, help="Override d_eff_gabor")
    parser.add_argument("--d-eff-laplacian", type=int, default=None, help="Override d_eff_laplacian")
    parser.add_argument("--d-max", type=int, default=None, help="Override d_max (encoder output dim)")
    parser.add_argument("--post-vp-mlp", action="store_true", help="Enable post-VP MLP")
    parser.add_argument("--post-vp-hidden", type=int, default=None, help="Post-VP MLP hidden dim")
    parser.add_argument("--tokenizer-path", type=str,
                        default="./data/tokenizers/fineweb_1024_bpe.model",
                        help="Path to SentencePiece model for competition bpb metric")
    parser.add_argument("--eval-sliding", action="store_true",
                        help="Enable slow sliding-window eval (default: off)")
    parser.add_argument("--no-dual-encoder", action="store_true",
                        help="Disable asymmetric dual encoder for Nyström head")
    parser.add_argument("--dual-encoder", action="store_true",
                        help="Enable asymmetric dual encoder for Nyström head")
    parser.add_argument("--attn-qk-mode", type=str, default=None,
                        choices=["full", "diagonal", "per_head_proj"],
                        help="Attention QK mode: full (default), diagonal (V1), per_head_proj (V2a)")
    parser.add_argument("--attn-d-proj", type=int, default=None,
                        help="d_proj for per_head_proj mode")
    parser.add_argument("--stiefel", action="store_true",
                        help="Enable Stiefel enforcement on encoder w_out")
    parser.add_argument("--stiefel-scale", action="store_true",
                        help="Enable learnable per-dim scale after Stiefel orthogonalization")
    parser.add_argument("--n-encoder-hidden", type=int, default=None,
                        help="Override n_encoder_hidden (encoder depth)")
    parser.add_argument("--phi-norm-gabor", type=str, default=None,
                        choices=["none", "softmax", "l2", "rowsum"],
                        help="Gabor kernel output normalization")
    parser.add_argument("--phi-norm-laplacian", type=str, default=None,
                        choices=["none", "softmax", "l2", "rowsum"],
                        help="Laplacian kernel output normalization")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = f"cuda:{local_rank}" if dist.is_initialized() else get_device()
    is_main = rank == 0

    # CUDA performance knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_flash_sdp, enable_mem_efficient_sdp
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)

    cfg = golf_config()
    if args.mode == "smoke":
        cfg.total_steps = 50
        cfg.log_every = 25
        cfg.val_every = 50
        cfg.max_wallclock = 0
        cfg.eval_stride = 0  # skip sliding eval in smoke mode
    if args.wallclock is not None:
        cfg.max_wallclock = args.wallclock
    if args.steps is not None:
        cfg.total_steps = args.steps
    if args.R is not None:
        cfg.R = args.R
    if args.L is not None:
        cfg.L = args.L
    if args.H is not None:
        cfg.H = args.H
    if args.H_gabor is not None:
        cfg.H_gabor = args.H_gabor
    if args.H_laplacian is not None:
        cfg.H_laplacian = args.H_laplacian
    if args.no_tucker_gl:
        cfg.use_tucker_gl = False
    if args.no_linear_head:
        cfg.use_linear_kernel_head = False
    if args.no_temporal_bw:
        cfg.use_temporal_bandwidth = False
    if args.no_staekel:
        cfg.staekel_beta = 0.0
    if args.symmetric_enc:
        cfg.H_gabor = 0
        cfg.H_laplacian = 0
    if args.nystrom_softmax:
        cfg.use_nystrom_softmax = True
        cfg.use_learnable_gegenbauer = False  # softmax replaces Gegenbauer
    if args.nystrom_rowsum:
        cfg.use_nystrom_rowsum = True
    if args.laplacian_rowsum:
        cfg.use_laplacian_rowsum = True
    if args.gabor_envelope_norm:
        cfg.use_gabor_envelope_norm = True
    if args.no_bigram:
        cfg.bigram_vocab_size = 0
    if args.bigram_vocab is not None:
        cfg.bigram_vocab_size = args.bigram_vocab
    if args.n_attn_layers is not None:
        cfg.n_attn_layers = args.n_attn_layers
    if args.d_max is not None:
        cfg.d_max = args.d_max
    if args.d_eff_gabor is not None:
        cfg.d_eff_gabor = args.d_eff_gabor
    if args.d_eff_laplacian is not None:
        cfg.d_eff_laplacian = args.d_eff_laplacian
    if args.post_vp_mlp:
        cfg.use_post_vp_mlp = True
    if args.post_vp_hidden is not None:
        cfg.post_vp_hidden = args.post_vp_hidden
    if args.no_dual_encoder:
        cfg.use_dual_encoder_nystrom = False
    if args.dual_encoder:
        cfg.use_dual_encoder_nystrom = True
    if args.phi_norm_gabor is not None:
        cfg.phi_norm_gabor = args.phi_norm_gabor
    if args.phi_norm_laplacian is not None:
        cfg.phi_norm_laplacian = args.phi_norm_laplacian
    if args.attn_qk_mode is not None:
        cfg.attn_qk_mode = args.attn_qk_mode
    if args.attn_d_proj is not None:
        cfg.attn_d_proj = args.attn_d_proj
    if args.stiefel:
        cfg.use_stiefel_enforcement = True
    if args.stiefel_scale:
        cfg.use_stiefel_scale = True
    if args.n_encoder_hidden is not None:
        cfg.n_encoder_hidden = args.n_encoder_hidden

    # Load FineWeb data
    data_dir = Path(args.data_dir)
    if data_dir.exists() and list(data_dir.glob("*.bin")):
        train_shards, val_shards = discover_fineweb_shards(str(data_dir))
        sample = load_fineweb_shard(train_shards[0])
        V_data = int(sample.max().item()) + 1
        V_clamp = cfg.V if V_data > cfg.V else None
        if V_clamp and is_main:
            print(f"Data vocab {V_data} > config vocab {cfg.V}: remapping via modulo")
        train_iter = ShardedBatchIterator(
            train_shards, cfg.B, cfg.T, device, V_clamp=V_clamp,
            rank=rank, world_size=world_size)
        val_tokens = torch.cat([load_fineweb_shard(s) for s in val_shards])
        val_iter = BatchIterator(val_tokens, cfg.B, cfg.T, device, V_clamp=V_clamp)
        val_raw = val_tokens if V_clamp is None else val_tokens % V_clamp
        if is_main:
            print(f"train_loader:dataset:fineweb10B_sp1024 train_shards:{len(train_shards)}")
            print(f"val_loader:tokens:{len(val_tokens)}")
    elif args.mode == "smoke":
        if is_main:
            print("Smoke mode: using synthetic data")
        synth = torch.randint(0, cfg.V, (cfg.B * cfg.T * 100,))
        train_iter = BatchIterator(synth, cfg.B, cfg.T, device)
        val_iter = BatchIterator(synth, cfg.B, cfg.T, device)
        val_raw = synth
        val_tokens = synth
        V_clamp = None
    else:
        raise FileNotFoundError(f"No FineWeb data at {data_dir}. Use --data-dir to specify.")

    # Load tokenizer for competition bpb metric
    bpb_luts = None
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        # Try relative to data dir
        tokenizer_path = data_dir.parent / "tokenizers" / "fineweb_1024_bpe.model"
    if tokenizer_path.exists() and spm is not None:
        sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        bpb_luts = build_sentencepiece_luts(sp, sp.vocab_size(), device)
        if is_main:
            print(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={tokenizer_path}")
    elif is_main:
        print("WARNING: No tokenizer found — competition bpb metric unavailable")
    if args.eval_sliding:
        cfg.eval_stride = 64

    model = HybridAtlasLM(cfg).to(device)
    if torch.cuda.is_available() and not args.no_compile:
        model = torch.compile(model)  # Whole-model compile (cross-module fusion)
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False, static_graph=True)
    raw_model = model.module if hasattr(model, "module") else model
    steps, best_bpb, step_avg_ms, raw_model_param_count = train(model, train_iter, val_iter, cfg, device)

    # === POST-TRAINING: save artifact first (rank 0), then distributed eval ===

    # Save artifact FIRST on rank 0 — before any eval
    artifact_path = Path("artifact.bin")
    if is_main:
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
            print(f"peak memory allocated: {peak_alloc:.0f} MiB reserved: {peak_reserved:.0f} MiB")
        artifact_bytes = save_artifact(raw_model, artifact_path)
        code_bytes = Path(__file__).stat().st_size
        total_bytes = artifact_bytes + code_bytes
        print(f"Serialized model: {raw_model_param_count * 2} bytes")
        print(f"Code size: {code_bytes} bytes")
        print(f"Total submission size: {total_bytes} bytes")
        print(f"Serialized model int8+zstd: {artifact_bytes} bytes")
        print(f"Total submission size int8+zstd: {total_bytes} bytes")

    # Sync all ranks before eval (rank 0 must finish saving artifact first)
    if dist.is_initialized():
        dist.barrier()

    # Competition bpb eval — load fresh uncompiled model from artifact (avoids torch.compile retrace)
    comp_val_loss, comp_val_bpb = None, None
    if bpb_luts is not None:
        from types import SimpleNamespace
        eval_args = SimpleNamespace(val_batch_size=cfg.B * cfg.T, train_seq_len=cfg.T)
        eval_fresh = HybridAtlasLM(cfg).to(device)  # uncompiled
        load_artifact(artifact_path, eval_fresh)
        eval_model = EvalWrapper(eval_fresh, V_clamp, cfg.V)
        t_eval = time.time()
        comp_val_loss, comp_val_bpb = eval_val(
            eval_args, eval_model, rank, world_size, device, 1,
            val_tokens, *bpb_luts)
        del eval_fresh  # free memory
        if is_main:
            eval_ms = int((time.time() - t_eval) * 1000)
            print(f"final val_loss:{comp_val_loss:.4f} val_bpb:{comp_val_bpb:.6f} eval_time:{eval_ms}ms")

    # Roundtrip — rank 0 only, skip in DDP
    if is_main:
        if not dist.is_initialized() or world_size == 1:
            model_q = HybridAtlasLM(cfg).to(device)
            load_artifact(artifact_path, model_q)
            rt_eval_model = EvalWrapper(model_q, V_clamp, cfg.V)
            if bpb_luts is not None:
                t_rt = time.time()
                rt_loss, rt_bpb = eval_val(
                    eval_args, rt_eval_model, 0, 1, device, 1,
                    val_tokens, *bpb_luts)
                rt_ms = int((time.time() - t_rt) * 1000)
                print(f"final_int8_zstd_roundtrip val_loss:{rt_loss:.4f} val_bpb:{rt_bpb:.6f} eval_time:{rt_ms}ms")
            else:
                rt = evaluate(model_q, val_iter, n_batches=20)
                print(f"final_int8_zstd_roundtrip val_loss:{rt['val_loss']:.4f} val_bpb:{rt['val_bpb']:.6f}")
        else:
            print("roundtrip: skipped (DDP mode)")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
