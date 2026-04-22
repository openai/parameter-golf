# Idea: V-Gate — per-head gated V-projection

**Source:** PR #1770 (liujshi, 2026-04-22)
**Claimed result on #1493 base:** 1.07960 (3-seed mean); watch-level vs our 1.06549 floor.

## What the lever is

A single learned per-head scalar that simultaneously:
1. Gates the information entering the V projection (input selector)
2. Scales that head's contribution to the output stream (per-head output scale)

This is mechanically distinct from `AttnOutGate` in our #1736 baseline:
- **AttnOutGate** gates the post-SDPA attention output (after softmax × V).
- **V-Gate** gates earlier — it controls what enters V _and_ how much that head contributes.

The two gates operate at different points in the attention pipeline and should be stackable.

## Author credibility

liujshi — not a previously tracked author; first record-track submission seen. #1770 builds
on #1493 directly (not our #1736 base), so the 1.0796 result lacks CaseOps, QuantGate, and
phased TTT. The V-Gate delta cannot be read directly from this PR.

## Estimated Δ-bucket

Unknown without ablation on #1736 base. 0.013 above our floor on the weaker stack; delta
likely shrinks substantially on a stronger base. Low confidence — treat as speculative.

## Implementation

Small code change: add a learned per-head gate weight in the attention module (roughly
5-10 lines in the SDPA block). Can be added on top of existing AttnOutGate without conflict.

## Feasibility

Moderate — needs a proper mini run to measure delta on our base. Not trivial like MLPClip12.

## Risks

- Delta may compress to noise on the #1736 base.
- AttnOutGate already covers part of the gating surface; marginal gain may be small.
- Not validated on a stack comparable to ours.

## Next step

Hold as watch; re-evaluate if someone posts V-Gate results on a CaseOps + phased TTT base.
If no such PR appears by 2026-04-28, consider a cheap proxy run after higher-priority specs.
