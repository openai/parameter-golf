from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SequenceMixerSpec:
    name: str
    hidden_dim: int
    num_heads: int = 8
    state_dim: int = 64


class SequenceMixer(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class AttentionBaselineMixer(SequenceMixer):
    def __init__(self, spec: SequenceMixerSpec):
        super().__init__()
        self.attn = nn.MultiheadAttention(spec.hidden_dim, spec.num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.attn(x, x, x, need_weights=False, is_causal=True)
        return y


class ChunkwiseRetentionMixer(SequenceMixer):
    def __init__(self, spec: SequenceMixerSpec):
        super().__init__()
        self.state_proj = nn.Linear(spec.hidden_dim, spec.state_dim, bias=False)
        self.out_proj = nn.Linear(spec.state_dim, spec.hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        state = torch.cumsum(self.state_proj(x), dim=1)
        return self.out_proj(F.rms_norm(state, (state.size(-1),)))


class HybridHeadMixer(SequenceMixer):
    def __init__(self, spec: SequenceMixerSpec):
        super().__init__()
        self.attn = AttentionBaselineMixer(spec)
        self.retention = ChunkwiseRetentionMixer(spec)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        mix = torch.softmax(self.mix, dim=0).to(dtype=x.dtype)
        return mix[0] * self.attn(x) + mix[1] * self.retention(x)


def build_sequence_mixer(spec: SequenceMixerSpec, *, kind: str) -> SequenceMixer:
    if kind == "attention_baseline":
        return AttentionBaselineMixer(spec)
    if kind == "retnet_chunkwise":
        return ChunkwiseRetentionMixer(spec)
    if kind in {"hybrid_hymba", "delta_hybrid"}:
        return HybridHeadMixer(spec)
    raise ValueError(f"Unknown sequence mixer kind: {kind}")

