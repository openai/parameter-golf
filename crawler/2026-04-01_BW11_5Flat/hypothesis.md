# BW11_5Flat — Hypothesis

**Parent:** BW8_Tap (BW5 + TAP_DIM=32 shared)
**One variable:** `NUM_FLAT_LAYERS=5` (5F+1C vs baseline 4F+1C)

## Background

In the original Bandit_Wagon proxy ablation (junkyard/experiments/archive/Bandit_Wagon/):
- 5F+1C scored **1.54404** proxy BPB — best among all width/depth variants at mlp=6.0
- Beat dim=576 (1.60382) and dim=640 (1.63303) with similar parameter count
- Was labelled "BW-03: recommended winner for BW_II investigation"
- Research instead pivoted to pyramid choke (BW_III), which failed: +0.020 BPB regression

5F+1C was **never validated at full run on BW5-quality baseline**.

## CL1 failure context

An earlier 5F+1C test (Crawler_Leg_1 lineage) failed, but that was on:
- loops=4, mlp=4.0 (much weaker baseline than current)
- No TAP, no battery tuning, no COMPILE_FULLGRAPH

Not informative for the BW8 baseline.

## Architecture logic

Current config: 4 flat → 1 crawler (3 loops) → 4 flat (U-Net skip)
Test config:    5 flat → 1 crawler (3 loops) → 5 flat (U-Net skip)

One extra flat layer on each side of the crawler. The crawler sits at the same
bottleneck. The flat layers surrounding it get more capacity. U-Net skip connections
still align because both sides gain symmetrically.

Parameter impact: +1 flat block each side ≈ +~2M params. Artifact will be slightly
larger. Must stay under 16MB.

## Expected outcome

If the proxy signal was real:
- Raw BPB improvement (more flat capacity around the bottleneck)
- int6_sw should improve proportionally (more params → larger artifact, but better weights)

Worst case: flat layers are already at capacity for this model dim. No improvement,
slight size increase.

## Gate target (4×GPU SDPA, 2000 steps)

Control = BW8 baseline (NUM_FLAT_LAYERS=4)
Test = BW8 + NUM_FLAT_LAYERS=5

Pass: BWFF-01 int6_sw < BWFF-00
Fail: regression or step_avg inflates >5ms (architecture change should be parameter-neutral on step time)

Historical reference: Bandit_Wagon proxy +0.006 BPB advantage over depth alternatives.
Gate-scale signal expected. Full run gap may be larger or smaller.
