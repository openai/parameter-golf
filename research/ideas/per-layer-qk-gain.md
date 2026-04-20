# Idea — Per-layer QK_GAIN initialization (port from #1648)

**Status:** 📝 CANDIDATE, deferred to spec 012 (was briefly bundled into draft 011 then dropped — QK is the highest-regression-risk lever and attribution would be muddled if bundled).
**Source:** PR #1648 (mikeapedia, non-record, out of compute).

## Core

#1736 initializes `q_gain = torch.full((num_heads,), 5.0)` uniformly across all 11 layers (multiplies Q before Q·Kᵀ; sharper attention for higher values).

#1648's convergence loop (train with learnable scalar → post-EMA value → hardcode → retrain new seed → average until stable) lands at per-layer values in the **2.0–3.0 range**. Author quote:
> "The model consistently prefers much softer attention than the 5.0 default."

## Evidence

Qualitative — author has per-seed converged values but no 8×H100 isolated ablation. The *consistency* of convergence across seeds is the main evidence; the gap from 5.0 → 2.5 is too large to be noise.

## Why it might help on #1736

If attention is over-sharpened at init, early training wastes steps fighting the sharpness before the model can learn meaningful attention patterns. Softening init to 2.5 is closer to the natural operating point the model converges to → faster effective learning in the fixed 600s budget.

This is the single largest architecture-side claim in the PR scan. If it lands, it's a basic init being wrong across most of the PG submission pool.

## Why it might not

- #1648's convergence runs were against #1586's stack (no CaseOps, no QuantGate, different TTT). The optimal per-layer values on #1736 could differ.
- #1736's phased TTT may already compensate for over-sharp init by redistributing at eval time. If so, softening init is redundant.
- Author did NOT run isolated 8×H100 confirmation. Confidence in the exact values is low; confidence in the *direction* (softer is better) is higher.

## Implementation plan

Two env vars:

1. `QK_GAIN_INIT=2.5` — scalar override (cheapest first pass; test "uniform softer" before per-layer).
2. `QK_GAIN_PER_LAYER="2.5,2.5,2.8,3.0,3.0,2.5,2.2,2.0,2.0,2.2,2.5"` — optional comma-sep per-layer list (11 values for 11 layers). If set, overrides the scalar.

After `self.blocks = nn.ModuleList(...)` construction, loop over blocks and overwrite `block.attn.q_gain` with per-layer value if list is set.

Default behavior (env vars absent): byte-identical to #1736.

## Testing strategy

Phase 1 (spec 012 — queued after spec 011 lands): test `QK_GAIN_INIT=2.5` (uniform) as the sole change vs spec 011's winning config. Cheapest test of the direction. Isolated so regression is easy to attribute.

Phase 2 (if spec 012 shows the direction is right): run a convergence loop on #1736's stack to find the actual per-layer optimum.

## Cross-references

- Companion: `research/ideas/gradpower-muon.md`.
- Not in spec 011 (training-bundle). Scheduled for spec 012.
- Source PR also proposes xIELU activation + symmetric resid_mix (deferred further).
