"""
H-Net (Path B): single-stage, attention-only.

Reference: Hwang, Wang, Gu (2025), "Dynamic Chunking for End-to-End Hierarchical
Sequence Modeling", arxiv:2507.07955. Reference impl: github.com/goombalab/hnet.

This is Path B from plan.md §2 — Mamba-2 layers replaced with vanilla pre-norm
Transformer blocks so we can isolate the dynamic-chunking contribution and ship
on Runpod without mamba_ssm CUDA-kernel surgery.

Differentiability follows paper §2.2.2:
- EMA smoothing on the upsampled (decoder-side) sequence weighted by the chunk's
  boundary probability — primary gradient pathway, ablated as ESSENTIAL in §3.3.
- Asymmetric confidence c_t = b·p + (1-b)·(1-p) with STE in the upsampler — paper
  Eq. 6/7, additional optimisation stabiliser.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DynamicChunking(nn.Module):
    """
    Routing + ratio loss + downsample. No STE here — STE lives in the upsampler.

    forward(x) -> (compressed, comp_mask, b_hard, p, p_compressed, ratio_loss)
        x:             (B, L, D)
        compressed:    (B, L_max, D)        chunk-aggregated, padded
        comp_mask:     (B, L_max) bool      True at real chunk positions
        b_hard:        (B, L)               binary boundary indicator (no STE)
        p:             (B, L)               soft boundary probabilities
        p_compressed:  (B, L_max)           p values at boundary positions only
        ratio_loss:    scalar
    """

    def __init__(self, dim: int, target_ratio: float = 1.0 / 6.0):
        super().__init__()
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.target_ratio = target_ratio
        # Identity init: at step 0 the router computes raw cos-sim of adjacent
        # hidden states — a meaningful starting point (per goombalab/hnet dc.py).
        with torch.no_grad():
            self.W_q.weight.copy_(torch.eye(dim))
            self.W_k.weight.copy_(torch.eye(dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        q = F.normalize(self.W_q(x), dim=-1)
        k = F.normalize(self.W_k(x), dim=-1)
        k_prev = F.pad(k[:, :-1, :], (0, 0, 1, 0), value=0.0)
        cos_sim = (q * k_prev).sum(dim=-1)

        p = (0.5 * (1.0 - cos_sim)).clamp(0.0, 1.0)
        first = torch.ones_like(p[:, :1])
        p = torch.cat([first, p[:, 1:]], dim=1)

        b_hard = (p >= 0.5).float()

        N = 1.0 / self.target_ratio
        F_actual = b_hard.mean()
        G_actual = p.mean()
        ratio_loss = (N / (N - 1.0)) * (
            (N - 1.0) * F_actual * G_actual + (1.0 - F_actual) * (1.0 - G_actual)
        )

        compressed, comp_mask = _downsample(x, b_hard)
        p_compressed, _ = _downsample(p.unsqueeze(-1), b_hard)
        p_compressed = p_compressed.squeeze(-1)
        return compressed, comp_mask, b_hard, p, p_compressed, ratio_loss


def _downsample(x: Tensor, b_hard: Tensor) -> tuple[Tensor, Tensor]:
    B, _, D = x.shape
    counts = b_hard.long().sum(dim=1)
    L_max = int(counts.max().item()) if B > 0 else 0
    out = x.new_zeros(B, L_max, D)
    mask = torch.zeros(B, L_max, dtype=torch.bool, device=x.device)
    for i in range(B):
        sel = x[i][b_hard[i].bool()]
        n = sel.shape[0]
        out[i, :n] = sel
        mask[i, :n] = True
    return out, mask


def upsample_with_ema(
    z_compressed: Tensor,
    b_hard: Tensor,
    p: Tensor,
    p_compressed: Tensor,
) -> Tensor:
    """Chunk-level EMA → gather to fine → asymmetric STE multiplier.

    Matches goombalab/hnet dc.py DeChunkLayer:
      EMA over the COMPRESSED sequence:
        z_bar[c] = p_c · z_compressed[c] + (1 - p_c) · z_bar[c-1]
      Plug back to fine level:
        out[t] = z_bar[chunk_idx[t]]
      Asymmetric confidence + STE (paper Eq. 6/7):
        c_t = b_hard·p + (1-b_hard)·(1-p),  STE(c_t) = c_t + stopgradient(1-c_t)
        out[t] = STE(c_t) · out[t]

    Loop runs L_max ≈ L · target_ratio times — ~6× shorter than fine-level EMA
    at default target_ratio=1/6. Vectorise via parallel scan or mamba_ssm kernel
    if Path A.
    """
    L_max = z_compressed.shape[1]
    p_c = p_compressed.clamp(1e-4, 1.0 - 1e-4)

    z_bar_steps = [z_compressed[:, 0]]
    for c in range(1, L_max):
        pc = p_c[:, c : c + 1]
        z_bar_steps.append(pc * z_compressed[:, c] + (1.0 - pc) * z_bar_steps[-1])
    z_bar_compressed = torch.stack(z_bar_steps, dim=1)

    chunk_idx = (b_hard.long().cumsum(dim=1) - 1).clamp(min=0)
    D = z_compressed.shape[-1]
    z_bar = torch.gather(z_bar_compressed, 1, chunk_idx.unsqueeze(-1).expand(-1, -1, D))

    c_score = b_hard * p + (1.0 - b_hard) * (1.0 - p)
    c_ste = c_score + (1.0 - c_score).detach()
    return c_ste.unsqueeze(-1) * z_bar


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_mult * dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_mult * dim, dim, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = True,
    ) -> Tensor:
        h = self.norm1(x)
        attn_mask = None
        if is_causal:
            L = h.shape[1]
            attn_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1
            )
        attn_out, _ = self.attn(
            h, h, h,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HNet(nn.Module):
    """Single-stage H-Net (Path B). See plan.md §3 for the dataflow."""

    def __init__(
        self,
        vocab_size: int = 260,
        d_enc: int = 128,
        d_main: int = 256,
        n_enc: int = 3,
        n_main: int = 6,
        n_dec: int = 3,
        n_heads: int = 4,
        target_ratio: float = 1.0 / 6.0,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_enc)
        self.encoder = nn.ModuleList(TransformerBlock(d_enc, n_heads) for _ in range(n_enc))
        self.dc = DynamicChunking(d_enc, target_ratio=target_ratio)
        self.proj_up = nn.Linear(d_enc, d_main, bias=False)
        self.main = nn.ModuleList(TransformerBlock(d_main, n_heads) for _ in range(n_main))
        self.proj_down = nn.Linear(d_main, d_enc, bias=False)
        self.decoder = nn.ModuleList(TransformerBlock(d_enc, n_heads) for _ in range(n_dec))
        self.norm_out = nn.LayerNorm(d_enc)
        self.head = nn.Linear(d_enc, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_enc ** -0.5)
        # Encoder→decoder residual stays near identity at init; the chunked path
        # only contributes once it has learned something useful (paper §2.3).
        nn.init.normal_(self.proj_down.weight, mean=0.0, std=1e-4)

    def forward(
        self, byte_ids: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        x = self.embed(byte_ids)
        for blk in self.encoder:
            x = blk(x)

        compressed, comp_mask, b_hard, p, p_compressed, ratio_loss = self.dc(x)

        z = self.proj_up(compressed)
        pad_mask = ~comp_mask
        for blk in self.main:
            z = blk(z, key_padding_mask=pad_mask)
        z = self.proj_down(z)

        x = x + upsample_with_ema(z, b_hard, p, p_compressed)
        for blk in self.decoder:
            x = blk(x)
        x = self.norm_out(x)
        logits = self.head(x)

        if targets is None:
            return logits, ratio_loss
        ar_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        return ar_loss, ratio_loss
