"""
Shepherd Embryo v0.0002 — True Probe Topology
Parameter Golf Entry — Non-Record Novel Architecture

WHAT CHANGED FROM v0.0001:
  - Real 5 parallel probe states: (B, P, S, D) not just (B, S, D)
  - Probe scoring: cosine similarity to seed + diversity bonus
  - Fold/merge: top-K probe selection, weighted merge, re-expansion
  - Regulator: drift detection against seed anchor, adaptive damping
  - True topology: expand → probe → score → fold → regulate → repeat

Pipeline:
  Seed → expand to 5 probes → [probe → score → fold → regulate] × 3 → Micro Core → logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# SHARED COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.to(x.dtype),
                        self.bias.to(x.dtype) if self.bias is not None else None)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cache is None or self._cache[0] != seq_len or self._cache[1].device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len,
                           freqs.cos()[None, None, :, :],
                           freqs.sin()[None, None, :, :])
        return self._cache[1].to(dtype=dtype), self._cache[2].to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


# =============================================================================
# MODULE 1: SEED GENERATOR
# Creates initial representation + expands to 5 distinct probe states.
# Each probe starts from a different learned perturbation of the seed.
# =============================================================================

class SeedGenerator(nn.Module):
    def __init__(self, vocab_size: int, model_dim: int, num_probes: int = 5, rank: int = 64):
        super().__init__()
        self.num_probes = num_probes
        self.model_dim = model_dim

        # Factored embedding
        self.embed_low = nn.Embedding(vocab_size, rank)
        self.expand = CastedLinear(rank, model_dim, bias=False)

        # Probe diversification: learned perturbation directions
        # Each probe gets a distinct starting bias via low-rank offset
        self.probe_directions = nn.Parameter(
            torch.randn(num_probes, model_dim) * 0.02
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_low.weight, std=0.01)
        nn.init.normal_(self.expand.weight, std=0.02)

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns:
          seed_anchor: (B, S, D) — the base representation for drift detection
          probes: (B, P, S, D) — 5 distinct probe starting states
        """
        z = self.embed_low(token_ids)       # (B, S, rank)
        x = self.expand(z)                  # (B, S, D)
        seed_anchor = F.rms_norm(x, (x.size(-1),))

        # Expand to P probes with learned directional perturbations
        # Each probe = seed + learned direction (distinct starting trajectories)
        B, S, D = seed_anchor.shape
        probes = seed_anchor.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        probes = probes + self.probe_directions[None, :, None, :]  # (B, P, S, D)

        return seed_anchor, probes


# =============================================================================
# MODULE 2: PROBE ENGINE
# Shared-weight attention block that processes each probe independently.
# Weight sharing across probes keeps parameter count bounded.
# The probes diverge through different starting states, not different weights.
# =============================================================================

class ProbeBlock(nn.Module):
    """Single attention + MLP block. Applied independently to each probe."""
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()

        self.c_q = CastedLinear(model_dim, model_dim, bias=False)
        self.c_k = CastedLinear(model_dim, kv_dim, bias=False)
        self.c_v = CastedLinear(model_dim, kv_dim, bias=False)
        self.proj = CastedLinear(model_dim, model_dim, bias=False)
        self.proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), 1.5, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

        self.mlp_fc = CastedLinear(model_dim, mlp_mult * model_dim, bias=False)
        self.mlp_proj = CastedLinear(mlp_mult * model_dim, model_dim, bias=False)
        self.mlp_proj._zero_init = True

        self.attn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))

    def _attn(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        q = self.c_q(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            _rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(_rep, dim=1)
            v = v.repeat_interleave(_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           )
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))

    def _mlp(self, x: Tensor) -> Tensor:
        return self.mlp_proj(torch.relu(self.mlp_fc(x)).square())

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S, D) — single probe state"""
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * self._attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self._mlp(self.mlp_norm(x))
        return x


class ProbeEngine(nn.Module):
    """
    Runs shared-weight ProbeBlock on all P probes in parallel.
    Weight sharing keeps parameters bounded — probes diverge through
    different starting states, not different weights.
    """
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float):
        super().__init__()
        # Single shared block — all probes use same weights
        self.block = ProbeBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base)

    def forward(self, probes: Tensor) -> Tensor:
        """
        probes: (B, P, S, D) — all probe states
        returns: (B, P, S, D) — updated probe states
        """
        B, P, S, D = probes.shape
        # Vectorized: reshape to (B*P, S, D), run block, restore
        flat = probes.reshape(B * P, S, D)
        flat = self.block(flat)
        return flat.reshape(B, P, S, D)


# =============================================================================
# PROBE SCORER
# Scores each probe based on:
#   1. Coherence with seed anchor (cosine similarity)
#   2. Diversity bonus (distance from other probes)
# Returns per-probe scores for fold selection.
# =============================================================================

class ProbeScorer(nn.Module):
    def __init__(self, similarity_weight: float = 0.5, coherence_weight: float = 0.3,
                 diversity_weight: float = 0.2):
        super().__init__()
        self.w_sim = similarity_weight
        self.w_coh = coherence_weight
        self.w_div = diversity_weight

    def forward(self, probes: Tensor, seed_anchor: Tensor) -> Tensor:
        """
        probes: (B, P, S, D)
        seed_anchor: (B, S, D)
        returns: scores (B, P) — higher is better
        """
        B, P, S, D = probes.shape

        # Pool across sequence for scoring (mean pool)
        probe_pooled = probes.mean(dim=2)       # (B, P, D)
        anchor_pooled = seed_anchor.mean(dim=1)  # (B, D)

        # 1. Similarity to seed anchor (coherence with origin)
        probe_norm = F.normalize(probe_pooled, dim=-1)
        anchor_norm = F.normalize(anchor_pooled, dim=-1)
        similarity = torch.bmm(probe_norm, anchor_norm.unsqueeze(-1)).squeeze(-1)  # (B, P)

        # 2. Self-coherence: internal consistency of each probe (inverse variance)
        probe_var = probes.var(dim=2).mean(dim=-1)  # (B, P)
        coherence = 1.0 / (probe_var + 1e-6)
        coherence = coherence / (coherence.max(dim=1, keepdim=True).values + 1e-6)  # normalize

        # 3. Diversity: average distance from other probes (exploration bonus)
        pairwise = torch.cdist(probe_pooled, probe_pooled)  # (B, P, P)
        diversity = pairwise.sum(dim=-1) / (P - 1)  # (B, P)
        diversity = diversity / (diversity.max(dim=1, keepdim=True).values + 1e-6)  # normalize

        scores = self.w_sim * similarity + self.w_coh * coherence + self.w_div * diversity
        return scores


# =============================================================================
# FOLD / MERGE
# Selects top-K probes, merges them, re-expands to P probes.
# This implements: explore → converge → explore
# =============================================================================

class ProbeFold(nn.Module):
    def __init__(self, model_dim: int, num_probes: int = 5, top_k: int = 2):
        super().__init__()
        self.num_probes = num_probes
        self.top_k = top_k

        # Re-expansion: from merged state back to P probe states
        self.re_expand = nn.Parameter(
            torch.randn(num_probes, model_dim) * 0.02
        )

    def forward(self, probes: Tensor, scores: Tensor) -> Tensor:
        """
        probes: (B, P, S, D)
        scores: (B, P)
        returns: (B, P, S, D) — new probe states from merged top-K
        """
        B, P, S, D = probes.shape

        # Select top-K probes
        _, top_indices = scores.topk(self.top_k, dim=1)  # (B, top_k)

        # Gather top probes
        top_indices_expanded = top_indices.unsqueeze(-1).unsqueeze(-1).expand(B, self.top_k, S, D)
        top_probes = torch.gather(probes, 1, top_indices_expanded)  # (B, top_k, S, D)

        # Weight top probes by their scores
        top_scores = torch.gather(scores, 1, top_indices)  # (B, top_k)
        weights = torch.softmax(top_scores, dim=1)  # (B, top_k)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (B, top_k, 1, 1)

        # Weighted merge
        merged = (top_probes * weights).sum(dim=1)  # (B, S, D)

        # Re-expand to P probes with fresh perturbations
        new_probes = merged.unsqueeze(1).expand(B, self.num_probes, S, D).clone()
        new_probes = new_probes + self.re_expand[None, :, None, :]

        return new_probes


# =============================================================================
# MODULE 3: REGULATOR
# Drift-aware damping that actually uses seed_anchor.
# Computes cosine drift, applies adaptive contraction,
# and blends toward anchor when drift exceeds threshold.
# =============================================================================

class Regulator(nn.Module):
    def __init__(self, model_dim: int, num_probes: int = 5, num_depths: int = 3,
                 drift_threshold: float = 0.6):
        super().__init__()
        self.drift_threshold = drift_threshold

        # Per-depth contraction strength (learned)
        self.contraction = nn.Parameter(
            torch.full((num_depths,), 0.9, dtype=torch.float32)
        )
        # Anchor blend strength when drift exceeds threshold
        self.anchor_blend = nn.Parameter(
            torch.full((num_depths,), 0.1, dtype=torch.float32)
        )

    def forward(self, probes: Tensor, depth: int, seed_anchor: Tensor) -> Tensor:
        """
        probes: (B, P, S, D)
        depth: current probe cycle index
        seed_anchor: (B, S, D) — original seed for drift detection
        returns: (B, P, S, D) — regulated probe states
        """
        B, P, S, D = probes.shape

        # Measure drift: cosine distance between each probe and seed anchor
        probe_flat = probes.reshape(B * P, S, D).mean(dim=1)  # (B*P, D)
        anchor_flat = seed_anchor.mean(dim=1)                  # (B, D)
        anchor_expanded = anchor_flat.unsqueeze(1).expand(B, P, D).reshape(B * P, D)

        drift = 1.0 - F.cosine_similarity(probe_flat, anchor_expanded, dim=-1)  # (B*P,)
        drift = drift.reshape(B, P)  # (B, P) — higher means more drifted

        # Apply contraction
        alpha = self.contraction[depth].to(dtype=probes.dtype)
        probes = probes * alpha

        # Blend toward anchor for probes that drifted too far
        anchor_3d = seed_anchor.unsqueeze(1).expand(B, P, S, D)
        drift_mask = (drift > self.drift_threshold).float()  # (B, P)
        blend = self.anchor_blend[depth].to(dtype=probes.dtype)
        blend_weight = (drift_mask * blend).unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)

        probes = probes * (1 - blend_weight) + anchor_3d * blend_weight

        return probes


# =============================================================================
# MODULE 4: MICRO CORE
# Small causal transformer — synthesis only. No probes, no branching.
# Takes the folded (B, S, D) representation and predicts tokens.
# =============================================================================

class MicroCoreBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.c_q = CastedLinear(model_dim, model_dim, bias=False)
        self.c_k = CastedLinear(model_dim, kv_dim, bias=False)
        self.c_v = CastedLinear(model_dim, kv_dim, bias=False)
        self.proj = CastedLinear(model_dim, model_dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), 1.5, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.mlp_fc = CastedLinear(model_dim, mlp_mult * model_dim, bias=False)
        self.mlp_proj = CastedLinear(mlp_mult * model_dim, model_dim, bias=False)
        self.mlp_proj._zero_init = True
        self.attn_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(model_dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape
        # Attention
        xn = self.attn_norm(x)
        q = self.c_q(xn).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(xn).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(S, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            _rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(_rep, dim=1)
            v = v.repeat_interleave(_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           )
        attn_out = self.proj(y.transpose(1, 2).contiguous().reshape(B, S, D))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        # MLP
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp_proj(
            torch.relu(self.mlp_fc(self.mlp_norm(x))).square()
        )
        return x


# =============================================================================
# SHEPHERD EMBRYO v0.0002 — TRUE PROBE TOPOLOGY
# =============================================================================

class ShepherdEmbryo(nn.Module):
    """
    Topology-first language model with real probe branching.

    Pipeline:
      Seed → 5 probes → [ProbeEngine → Score → Fold → Regulate] × depth → collapse → MicroCore → logits

    Shape flow:
      (B, S) → seed → (B, 5, S, D) → probe loop → fold to (B, S, D) → core → logits
    """
    def __init__(
        self,
        vocab_size: int = 1024,
        model_dim: int = 384,        # smaller than baseline to fit probe overhead
        seed_rank: int = 48,
        num_probes: int = 5,
        num_probe_depths: int = 3,
        fold_top_k: int = 2,
        num_core_layers: int = 3,
        num_heads: int = 6,
        num_kv_heads: int = 3,
        mlp_mult: int = 2,
        rope_base: float = 10000.0,
        logit_softcap: float = 30.0,
        tie_embeddings: bool = True,
        tied_embed_init_std: float = 0.005,
        drift_threshold: float = 0.6,
        **kwargs,
    ):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tie_embeddings = tie_embeddings
        self.num_probe_depths = num_probe_depths
        self.num_probes = num_probes

        # Module 1: Seed Generator — creates anchor + 5 probe states
        self.seed = SeedGenerator(vocab_size, model_dim, num_probes, seed_rank)

        # Module 2: Probe Engine — shared-weight block for parallel probes
        # One block per depth (different weights at each depth level)
        self.probe_engines = nn.ModuleList([
            ProbeEngine(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base)
            for _ in range(num_probe_depths)
        ])

        # Probe Scorer
        self.scorer = ProbeScorer()

        # Fold/Merge — one per depth
        self.folds = nn.ModuleList([
            ProbeFold(model_dim, num_probes, fold_top_k)
            for _ in range(num_probe_depths)
        ])

        # Module 3: Regulator — drift-aware damping
        self.regulator = Regulator(model_dim, num_probes, num_probe_depths, drift_threshold)

        # Module 4: Micro Core — small causal transformer
        self.core = nn.ModuleList([
            MicroCoreBlock(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base)
            for _ in range(num_core_layers)
        ])

        # Output
        self.final_norm = RMSNorm()
        if tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(model_dim, vocab_size, bias=False)
            self.lm_head._zero_init = True

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        # ---- STAGE 1: SEED ----
        seed_anchor, probes = self.seed(input_ids)
        # seed_anchor: (B, S, D)
        # probes: (B, P, S, D) — 5 distinct starting trajectories

        # ---- STAGE 2: PROBE LOOP (explore → score → fold → regulate) × depth ----
        for depth in range(self.num_probe_depths):
            # Run all probes through shared-weight block
            probes = self.probe_engines[depth](probes)   # (B, P, S, D)

            # Score probes against seed anchor
            scores = self.scorer(probes, seed_anchor)     # (B, P)

            # Fold: merge top-K, re-expand to P probes
            probes = self.folds[depth](probes, scores)    # (B, P, S, D)

            # Regulate: drift detection + adaptive damping
            probes = self.regulator(probes, depth, seed_anchor)  # (B, P, S, D)

        # ---- STAGE 3: COLLAPSE to single representation ----
        # Final scoring to select best probe state
        final_scores = self.scorer(probes, seed_anchor)   # (B, P)
        weights = torch.softmax(final_scores, dim=1)      # (B, P)
        x = (probes * weights[:, :, None, None]).sum(dim=1)  # (B, S, D)

        # ---- STAGE 4: MICRO CORE ----
        for block in self.core:
            x = block(x)

        # ---- STAGE 5: OUTPUT ----
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.seed.expand.weight.T.to(x.dtype))
            logits_proj = F.linear(logits_proj, self.seed.embed_low.weight.to(x.dtype))
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# =============================================================================
# PARAMETER COUNT + TOPOLOGY VERIFICATION
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    counts = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        counts[name] = n
    total = sum(p.numel() for p in model.parameters())
    counts['TOTAL'] = total
    return counts


if __name__ == "__main__":
    print("=" * 60)
    print("  SHEPHERD EMBRYO v0.0002 — True Probe Topology")
    print("=" * 60)

    model = ShepherdEmbryo(
        vocab_size=1024,
        model_dim=384,
        seed_rank=48,
        num_probes=5,
        num_probe_depths=3,
        fold_top_k=2,
        num_core_layers=3,
        num_heads=6,
        num_kv_heads=3,
        mlp_mult=2,
    )

    # Parameter budget
    counts = count_parameters(model)
    print("\nParameter Budget:")
    for name, count in counts.items():
        pct = count / counts['TOTAL'] * 100 if name != 'TOTAL' else 100
        print(f"  {name:20s}: {count:>10,d} ({pct:5.1f}%)")

    total_params = counts['TOTAL']
    estimated_bytes = total_params
    print(f"\n  Estimated artifact: {estimated_bytes / 1e6:.2f} MB (int8)")
    print(f"  Budget:            16.00 MB")
    print(f"  Margin:            {16e6 - estimated_bytes:,.0f} bytes")

    # Topology verification
    print("\nTopology Verification:")
    print(f"  Probes:     {model.num_probes}")
    print(f"  Depths:     {model.num_probe_depths}")
    print(f"  Fold top-K: {model.folds[0].top_k}")
    print(f"  Core layers:{len(model.core)}")
    print(f"  Drift threshold: {model.regulator.drift_threshold}")

    # Forward pass
    print("\nForward pass test...")
    B, S = 2, 128
    input_ids = torch.randint(0, 1024, (B, S))
    target_ids = torch.randint(0, 1024, (B, S))

    with torch.no_grad():
        loss = model(input_ids, target_ids)

    print(f"  Input shape:  ({B}, {S})")
    print(f"  Probe shape:  ({B}, {model.num_probes}, {S}, {model.seed.model_dim})")
    print(f"  Loss:         {loss.item():.4f}")
    print(f"  Expected:     ~{math.log(1024):.4f} (ln(vocab))")
    print(f"  Status:       {'ALIVE — TOPOLOGY ACTIVE' if abs(loss.item() - math.log(1024)) < 1.0 else 'CHECK'}")

    # Verify probe divergence
    print("\nProbe Divergence Test...")
    with torch.no_grad():
        seed_anchor, probes = model.seed(input_ids)
        probe_pooled = probes.mean(dim=2)  # (B, P, D)
        pairwise = torch.cdist(probe_pooled[0:1], probe_pooled[0:1]).squeeze()
        print(f"  Pairwise distances (batch 0):")
        for i in range(model.num_probes):
            dists = [f"{pairwise[i, j].item():.3f}" for j in range(model.num_probes)]
            print(f"    Probe {i}: [{', '.join(dists)}]")

    print("\n" + "=" * 60)
    print("  The embryo has topology.")
    print("=" * 60)
