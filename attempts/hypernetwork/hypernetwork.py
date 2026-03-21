"""
HyperNetwork for Parameter Golf

The idea: instead of shipping the full GPT weights in the artifact, we ship a
smaller "hypernetwork" whose weights, when executed, *generate* the full GPT
model's weight tensors. The artifact only needs to contain:
  1. This code
  2. The hypernetwork weights (much smaller than the target GPT)

At load time we run the hypernetwork forward pass to produce all target weights,
load them into the GPT, and evaluate normally.

The hypernetwork is a simple MLP that takes a per-layer/per-tensor conditioning
vector and outputs the flattened weight chunk. We chunk large weight matrices
into manageable pieces and generate each chunk from a shared trunk + per-chunk
conditioning.

Target: the generated GPT should be BIGGER than what would normally fit in 16MB,
because the hypernetwork compresses the weight space.
"""

from __future__ import annotations

import io
import math
import os
import zlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Target GPT config (what we want to generate) — bigger than baseline
# ---------------------------------------------------------------------------

@dataclass
class TargetGPTConfig:
    vocab_size: int = 1024
    num_layers: int = 11          # competitive: 11 layers
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: int = 3             # competitive: 3x MLP
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5


# ---------------------------------------------------------------------------
# Weight manifest — describes every tensor the target GPT needs
# ---------------------------------------------------------------------------

@dataclass
class WeightSpec:
    name: str
    shape: tuple[int, ...]
    generate: bool = True   # False for small control tensors we store directly


def build_weight_manifest(cfg: TargetGPTConfig) -> list[WeightSpec]:
    """Enumerate every parameter in the target GPT architecture."""
    specs: list[WeightSpec] = []
    dim = cfg.model_dim
    kv_dim = cfg.num_kv_heads * (dim // cfg.num_heads)
    mlp_hidden = cfg.mlp_mult * dim

    # Token embedding (tied with lm_head)
    specs.append(WeightSpec("tok_emb.weight", (cfg.vocab_size, dim)))

    # Skip weights — small, store directly
    num_enc = cfg.num_layers // 2
    num_dec = cfg.num_layers - num_enc
    num_skip = min(num_enc, num_dec)
    specs.append(WeightSpec("skip_weights", (num_skip, dim), generate=False))

    for i in range(cfg.num_layers):
        prefix = f"blocks.{i}"

        # Attention projections
        specs.append(WeightSpec(f"{prefix}.attn.c_q.weight", (dim, dim)))
        specs.append(WeightSpec(f"{prefix}.attn.c_k.weight", (kv_dim, dim)))
        specs.append(WeightSpec(f"{prefix}.attn.c_v.weight", (kv_dim, dim)))
        specs.append(WeightSpec(f"{prefix}.attn.proj.weight", (dim, dim)))

        # MLP
        specs.append(WeightSpec(f"{prefix}.mlp.fc.weight", (mlp_hidden, dim)))
        specs.append(WeightSpec(f"{prefix}.mlp.proj.weight", (dim, mlp_hidden)))

        # Small per-layer control tensors — store directly, don't generate
        specs.append(WeightSpec(f"{prefix}.attn.q_gain", (cfg.num_heads,), generate=False))
        specs.append(WeightSpec(f"{prefix}.attn_scale", (dim,), generate=False))
        specs.append(WeightSpec(f"{prefix}.mlp_scale", (dim,), generate=False))
        specs.append(WeightSpec(f"{prefix}.resid_mix", (2, dim), generate=False))

    return specs


# ---------------------------------------------------------------------------
# HyperNetwork
# ---------------------------------------------------------------------------

@dataclass
class HyperNetConfig:
    """Configuration for the hypernetwork."""
    # Shared trunk
    cond_dim: int = 64          # conditioning vector size per chunk
    trunk_hidden: int = 512     # trunk MLP hidden dim
    trunk_layers: int = 3       # number of trunk layers

    # Chunking: large weight matrices are split into chunks of this size
    chunk_size: int = 4096      # each generated chunk is this many floats

    # Target
    target_cfg: TargetGPTConfig = field(default_factory=TargetGPTConfig)


class HyperNetTrunk(nn.Module):
    """Shared MLP trunk that maps conditioning → weight chunk."""

    def __init__(self, cond_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = cond_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond)


class HyperNetwork(nn.Module):
    """
    Generates all weight tensors for the target GPT.

    For each weight tensor that needs generating:
    - Split into chunks of `chunk_size` floats
    - Each chunk has a learned conditioning vector
    - The shared trunk maps conditioning → chunk values
    - Reassemble chunks into the full weight tensor

    Small control tensors (scales, gains, etc.) are stored directly as
    parameters of the hypernetwork — no generation needed.
    """

    def __init__(self, config: HyperNetConfig):
        super().__init__()
        self.config = config
        self.manifest = build_weight_manifest(config.target_cfg)

        chunk_size = config.chunk_size
        cond_dim = config.cond_dim

        # Shared generation trunk
        self.trunk = HyperNetTrunk(cond_dim, config.trunk_hidden, config.trunk_layers, chunk_size)

        # Per-chunk conditioning embeddings for generated tensors
        # and direct storage for small tensors
        self.chunk_conds = nn.ParameterDict()
        self.direct_params = nn.ParameterDict()
        self._gen_info: dict[str, dict] = {}  # name -> {num_chunks, total_numel, shape}

        for spec in self.manifest:
            safe_name = spec.name.replace(".", "_")
            numel = 1
            for s in spec.shape:
                numel *= s

            if not spec.generate:
                # Store directly
                self.direct_params[safe_name] = nn.Parameter(torch.zeros(*spec.shape))
                continue

            # Compute chunking
            num_chunks = math.ceil(numel / chunk_size)
            # One conditioning vector per chunk
            self.chunk_conds[safe_name] = nn.Parameter(
                torch.randn(num_chunks, cond_dim) * 0.02
            )
            self._gen_info[safe_name] = {
                "num_chunks": num_chunks,
                "total_numel": numel,
                "shape": spec.shape,
                "name": spec.name,
            }

        self._init_direct_params()

    def _init_direct_params(self):
        """Initialize direct params with reasonable defaults."""
        cfg = self.config.target_cfg
        with torch.no_grad():
            for spec in self.manifest:
                if spec.generate:
                    continue
                safe_name = spec.name.replace(".", "_")
                p = self.direct_params[safe_name]
                if "skip_weight" in spec.name:
                    p.fill_(1.0)
                elif "q_gain" in spec.name:
                    p.fill_(cfg.qk_gain_init)
                elif "attn_scale" in spec.name or "mlp_scale" in spec.name:
                    p.fill_(1.0)
                elif "resid_mix" in spec.name:
                    p[0].fill_(1.0)
                    p[1].fill_(0.0)

    def generate_weights(self) -> dict[str, Tensor]:
        """Run the hypernetwork to produce all target GPT weights."""
        result: dict[str, Tensor] = {}
        chunk_size = self.config.chunk_size

        for spec in self.manifest:
            safe_name = spec.name.replace(".", "_")

            if not spec.generate:
                result[spec.name] = self.direct_params[safe_name]
                continue

            info = self._gen_info[safe_name]
            conds = self.chunk_conds[safe_name]  # (num_chunks, cond_dim)

            # Generate all chunks in one batched forward pass
            chunks = self.trunk(conds)  # (num_chunks, chunk_size)
            flat = chunks.reshape(-1)[:info["total_numel"]]
            result[spec.name] = flat.reshape(info["shape"])

        return result

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by category."""
        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        cond_params = sum(p.numel() for p in self.chunk_conds.values())
        direct_params = sum(p.numel() for p in self.direct_params.values())
        total = trunk_params + cond_params + direct_params
        return {
            "trunk": trunk_params,
            "conditioning": cond_params,
            "direct": direct_params,
            "total": total,
        }

    def count_target_parameters(self) -> int:
        """Count how many parameters the generated GPT would have."""
        total = 0
        for spec in self.manifest:
            numel = 1
            for s in spec.shape:
                numel *= s
            total += numel
        return total


# ---------------------------------------------------------------------------
# Utility: load hypernetwork weights → build full GPT state dict
# ---------------------------------------------------------------------------

def hypernet_to_gpt_state_dict(hypernet: HyperNetwork) -> dict[str, Tensor]:
    """Generate the full GPT state dict from a trained hypernetwork."""
    with torch.no_grad():
        return {k: v.detach().clone() for k, v in hypernet.generate_weights().items()}


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("HyperNetwork for Parameter Golf — Architecture Summary")
    print("=" * 60)

    config = HyperNetConfig()
    hnet = HyperNetwork(config)

    param_counts = hnet.count_parameters()
    target_params = hnet.count_target_parameters()

    print(f"\nTarget GPT config:")
    print(f"  Layers: {config.target_cfg.num_layers}")
    print(f"  Dim: {config.target_cfg.model_dim}")
    print(f"  MLP mult: {config.target_cfg.mlp_mult}")
    print(f"  Vocab: {config.target_cfg.vocab_size}")
    print(f"  Target params: {target_params:,}")

    print(f"\nHyperNetwork config:")
    print(f"  Cond dim: {config.cond_dim}")
    print(f"  Trunk hidden: {config.trunk_hidden}")
    print(f"  Trunk layers: {config.trunk_layers}")
    print(f"  Chunk size: {config.chunk_size}")

    print(f"\nHyperNetwork parameters:")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    print(f"\nCompression ratio: {target_params / param_counts['total']:.2f}x")
    print(f"  (generating {target_params:,} params from {param_counts['total']:,})")

    # Test generation
    print("\nGenerating target weights...")
    weights = hnet.generate_weights()
    print(f"Generated {len(weights)} tensors:")
    for name, tensor in sorted(weights.items()):
        print(f"  {name}: {tuple(tensor.shape)}")

    print("\nDone.")
