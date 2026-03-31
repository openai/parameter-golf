# bandit_wagon_choke_descend (BWCD) — Descending Battery on Pyramid-512

## Background

BWCB established that ascending battery (1,2,4) beats pyramid-512 alone by -0.00210 at 4
shards (Run B). The mega ablation (BWB series, flat MLP) showed descending (9,3,1) has
near-zero quant_gap (+0.0001) vs ascending 1,3,9 (+0.0028). Hypothesis: wide-first is the
natural refinement order for the crawler — loop 0 establishes context basin on the cleanest
residual, loops 1+2 refine.

BWCD tests descending + bracket configurations on pyramid-512 (BWCS winner).

## References

| Run | Config | INT6_SW_BPB | Quant Gap |
|-----|--------|-------------|-----------|
| BWCS-00 | flat ctrl (1 shard) | 1.45761 | +0.0013 |
| BWCS-02 | pyramid-512 (1 shard) | 1.44724 | -0.0001 |
| BWB-04 | flat 9,3,1 (80 shards) | 1.44156 | +0.0001 |
| BWB-01 | flat 1,2,4 (80 shards) | 1.43769 | -0.0010 |
| BWCB-00 | pyramid + 1,2,4 (4 shards) | 1.44515 | +0.0009 |

## Arms

| ID | Scales | Shape | Purpose |
|----|--------|-------|---------|
| BWCD-00 | 9,3,1 | descending | Mirror of 1,3,9 — does wide-first help 9× spread? |
| BWCD-01 | 4,2,1 | gentle descending | Mirror of 1,2,4 — gentler spread descending |
| BWCD-02 | 9,1,1 | first wide only | Loop 0 wide, loops 1+2 identical local |
| BWCD-03 | 9,3,9 | wide-med-wide bracket | Loops 0+2 share global scale, loop 1 refines |

## Results — 1 shard (seed=444, 500 steps)

| ID | Scales | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCS-02 |
|----|--------|---------|-------------|-----------|------------|
| BWCS-02 ref | — | 1.4473 | 1.44724 | -0.0001 | — |
| BWCD-00 | 9,3,1 | 1.4354 | 1.43779 | +0.0024 | -0.00945 |
| BWCD-01 | 4,2,1 | 1.4356 | 1.43749 | +0.0019 | -0.00975 |
| **BWCD-02** | **9,1,1** | **1.4352** | **1.43531** | **+0.0001** | **-0.01193** |
| BWCD-03 | 9,3,9 | 1.4363 | 1.44248 | +0.0062 | -0.00476 |

## Verdict: 9,1,1 Wins — Identical Trailing Loops is the Mechanism

**BWCD-02 (pyramid-512 + 9,1,1) beats pyramid alone by -0.01193 at 1 shard.**

This works at 1 shard (unlike BWCB-00 which needed 4 shards), because 9,1,1 doesn't require
training diversity to specialize — it's a structural advantage.

### The Core Principle

All four arms have nearly identical raw_bpb (~1.435). The battery does not change how much
the model learns — only how cleanly it quantizes. All differentiation is in the quant gap,
and quant gap tracks directly with **how many distinct scales the final loops use**:

- 9,1,1 → loops 1 and 2 **identical** (both scale=1) → quant gap **+0.0001**
- 4,2,1 → 3 distinct scales, gentle spread → quant gap +0.0019
- 9,3,1 → 3 distinct scales, aggressive spread → quant gap +0.0024
- 9,3,9 → loops 0 and 2 share scale=9, but different residuals → quant gap large (TBD)

The int8 quantizer uses a single per-row scale covering all three loops. When loops 1 and 2
share scale=1, they produce nearly identical activation distributions. The quantizer only
bridges two populations (loop 0's wide-context features vs loops 1+2's local features)
instead of three-way divergence.

### Why 9,3,9 Fails Despite Sharing Scale=9

Loops 0 and 2 share RoPE scale=9, but loop 2 runs on a doubly-processed residual. Same scale,
different input → different distribution. Structural symmetry ≠ distribution symmetry.
The bimodal framing doesn't hold when the input histories diverge.

### Why 9,1,1 Works at 1 Shard

Unlike ascending 1,2,4 which requires training diversity for each loop to specialize at its
causal horizon, 9,1,1 is a structural win:
- Loop 0 reads wide on the freshest signal (straight from encoder)
- Loops 1 and 2 are identical scale — no coordination needed, no specialization required
- The pyramid's shared stage1 anchors the universal representation before branching

### Training Stability

All 4 BWCD arms show identical early-step curves (step 2 spike ~9.68, monotonic recovery,
step 10 floor ~6.27). The battery configuration has zero effect on training dynamics.
**The pyramid choke's shared stage1 is the stabilizer** — it is the gradient bottleneck
that prevents loop divergence during warmup, regardless of RoPE scales.

This stability has implications for scaling: stable crawler = tighter LR tolerances,
less sensitivity to multi-node gradient variance on 8×H100.

## Follow-On

**BWCE** (or similar): validate 9,1,1 + pyramid at 4+ shards (match BWCB-00 shard count).
If 9,1,1 improves further with training diversity, it becomes the canonical battery config.

Also: compare to BWT-05 (tap=32 deep per-loop, 1.43004 at 80 shards) in a head-to-head
at matched shard count. BWCD-02 at 1.43531 (1 shard) may close or beat that gap at scale.
