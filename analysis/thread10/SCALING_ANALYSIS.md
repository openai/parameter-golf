# Parameter Golf Scaling Law Analysis

**Date**: 2026-03-18  
**Data**: 26 experiments (001-026) at 2K steps on 1×H100 + official baseline 13.7K steps on 8×H100  
**Goal**: Find the single best config to submit to the competition (beat val_bpb 1.2244)

---

## Executive Summary

After analyzing all 26 experiments, the data overwhelmingly shows:

**THROUGHPUT IS KING.** The single most important factor is how many training steps fit in the 600-second wallclock budget. The baseline architecture (9 unique blocks × 1 loop, dim=512) is the optimal config because:

1. It's the **fastest** config at ~43.5ms/step on 8×H100 → 13,780 steps
2. No 16MB-fitting alternative beats it on per-step quality
3. The warmdown zone (final 1,200 steps) provides 0.019 BPB of "free" improvement that slower configs can't access

**The one exception**: MoE (exp026) runs 1.54× FASTER than baseline on 1×H100. If this scales to 8×H100 AND maintains quality per step, it could achieve ~21K steps → potentially 1.19-1.21 BPB. This is the highest-upside bet but needs val_bpb validation.

---

## Data Tables

### All Experiments — Step 500 val_bpb (sorted best → worst)

| Rank | Config | Unique×Loops | Eff.Depth | Dim | Params | BPB@500 | Δ Baseline | Fits 16MB? |
|------|--------|-------------|-----------|-----|--------|---------|------------|------------|
| 1 | 7×2@672 | 7×2 | 14 | 672 | 22.9M | 1.4499 | -0.031 | ❌ |
| 2 | 6×2@688 | 6×2 | 12 | 688 | 17.9M | 1.4581 | -0.022 | ❌ |
| 3 | 9×1+sc15+eps | 9×1 | 9 | 512 | 17.1M | 1.4753 | -0.005 | ✅ |
| 4 | 6×2@576 | 6×2 | 12 | 576 | 14.6M | 1.4796 | -0.001 | ✅ |
| 5 | **9×1@512 (BASE)** | 9×1 | 9 | 512 | 17.1M | **1.4805** | 0.000 | ✅ |
| 6 | 7×2@544 | 7×2 | 14 | 544 | 15.1M | ~1.490 | +0.010 | ✅ |
| 7 | 10L+MLA64 | 10×1 | 10 | 512 | 16.0M | 1.5074 | +0.027 | ✅ |
| 8 | 3×3@720 | 3×3 | 9 | 720 | 11.7M | 1.5096 | +0.029 | ✅ |
| 9 | 5×2@704 | 5×2 | 10 | 704 | 19.0M | ~1.510 | +0.030 | ✅ |
| 10 | 9L+MLA64 | 9×1 | 9 | 512 | 16.5M | 1.5187 | +0.038 | ✅ |
| 11 | MTP only | 9×1 | 9 | 512 | 17.1M | ~1.550 | +0.070 | ✅ |
| 12 | MTP+sc15+eps | 9×1 | 9 | 512 | 17.1M | 1.6963 | +0.216 | ✅ |

### All Experiments — 2K Step Post-Quant val_bpb (where available)

| Rank | Config | PQ BPB | Δ Baseline | ms/step | Artifact | Fits? |
|------|--------|--------|------------|---------|----------|-------|
| 1 | 7×2@672 | 1.2797 | -0.018 | 1196 | 19.5MB | ❌ |
| 2 | 6×2@688 | 1.2912 | -0.007 | 1109 | 16.5MB | ❌ |
| 3 | **9×1@512** | **1.2978** | **0.000** | **464** | **15.0MB** | **✅** |
| 4 | 9×1+sc15+eps | 1.3001 | +0.002 | 480 | 15.0MB | ✅ |
| 5 | MTP only | 1.3016 | +0.004 | 480 | 15.0MB | ✅ |
| 6 | MTP+sc15+eps | 1.3059 | +0.008 | 480 | 15.0MB | ✅ |
| 7 | 5×2@704 | 1.3094 | +0.012 | 1750 | 15.5MB | ✅ |
| 8 | 3×3@720 | 1.3426 | +0.045 | 700 | 11.0MB | ✅ |

### Extrapolation to 8×H100 Full Run

| Config | 1GPU ms | Est 8GPU ms | Est Steps | Baseline BPB@N | Est Post-Q | Fits? |
|--------|---------|-------------|-----------|----------------|------------|-------|
| **Baseline 9×1** | **464** | **43.5** | **13,780** | **1.2172** | **1.2244** | **✅** |
| 9×1+sc15+eps | 480 | 45.0 | 13,325 | 1.2255 | ~1.234 | ✅ |
| 6×2@576 | 691 | 64.8 | 9,256 | 1.2500 | ~1.263 | ✅ |
| 7×2@544 | 992 | 93.1 | 6,447 | 1.2625 | ~1.275 | ✅ |
| 5×2@704 | 1750 | 164.2 | 3,654 | 1.2865 | ~1.295 | ✅ |
| **MoE 2-expert** | **302** | **28.3** | **~21,178** | **1.2172** | **~1.21?** | **✅** |

---

## Analysis 1: Depth vs Width

**Finding**: At the 16MB artifact budget (~17M params with int8+zlib), width and depth are zero-sum. The data shows:

- Wider models (dim≥672) with recurrence beat baseline per step — BUT exceed 16MB
- At budget-fitting dims (≤576), recurrence provides ZERO quality advantage
- The baseline's 9 unique blocks at dim=512 is at or near the Pareto frontier

**The recurrence "illusion"**: 7×2@672 looks great (-0.018 BPB) but uses 22.9M params → 19.5MB artifact. The quality comes from having MORE PARAMETERS, not from the recurrence itself.

---

## Analysis 2: Unique Blocks vs Loops

**Finding**: At iso-parameters within 16MB, 9 unique blocks × 1 loop ≥ N unique × K loops.

The apparent "7 > 6 > 5 > 3 unique blocks" trend is CONFOUNDED with wider dimensions. When we control for parameter budget:
- 7×2@544 (15.1M params): step 1000 BPB = 1.3814 (baseline = 1.3760, **0.005 WORSE**)
- 6×2@576 (14.6M params): step 500 BPB = 1.4796 (baseline = 1.4805, **tied**)
- 5×2@704 (19M params): 2K PQ BPB = 1.3094 (baseline = 1.2978, **0.012 WORSE**)

Weight sharing adds compute (2-4× slower) with zero quality benefit at budget-fitting sizes.

---

## Analysis 3: Compute-Quality Tradeoff

**Finding**: Within 16MB, no config has positive quality-per-compute. The baseline wins by being the fastest.

| Config | Quality/Compute Score |
|--------|----------------------|
| 7×2@672 | +0.0070 (over budget) |
| 6×2@688 | +0.0028 (over budget) |
| Baseline | 0.0000 (reference) |
| All others | NEGATIVE |

---

## Analysis 4: The Warmdown Effect

The official baseline curve reveals a critical dynamic:
- Steps 0-12,580: BPB improves from 4.098 → 1.240 (2.858 improvement over 12.6K steps)
- Steps 12,580-13,780: BPB improves from 1.236 → 1.217 (0.019 improvement in just 1.2K steps!)

The warmdown provides **0.019 BPB in 8.7% of training**. This is enormously valuable and only accessible to configs that can train for 12,500+ steps. At ~43.5ms/step, only the baseline (or faster configs) reach this zone.

---

## Analysis 5: MoE — The Dark Horse

MoE (exp026) data is sparse but tantalizing:
- **Speed**: 302ms/step on 1×H100 = 1.54× FASTER than baseline
- **Quality signal**: train_loss 3.19 at step 100 (baseline: 3.32 = 0.13 better)
- **Same params**: 17.1M params, ~15MB artifact
- **Projected 8×H100**: ~21K steps (54% more than baseline!)

IF MoE quality matches baseline per step, the throughput advantage alone projects to ~1.21 BPB.

**Critical caveat**: The "2 experts, top-2 routing" design means every token hits every expert. This is mathematically equivalent to a different MLP parameterization, not true sparse MoE. The speed gain may come from better parallelism of the two smaller MLPs rather than algorithmic advantage.

---

## RECOMMENDATION

### Primary Submission: Baseline (9×1@512) + softcap=15 + eps=1e-10
- **Expected**: ~1.22 BPB (matching baseline)
- **Risk**: Low — it's the proven config
- **Upside**: marginal (softcap/eps improvements wash out at 2K but might help at 13K)

### Contingency: MoE (if val_bpb validates)
- **Threshold**: If MoE val_bpb ≤ 1.48 at step 500, switch to MoE
- **Expected**: ~1.19-1.21 BPB (beating baseline by 0.01-0.03)
- **Risk**: Medium — no val_bpb data yet

### Immediate Next Steps
1. **GET MoE val_bpb at step 500** — this is the single highest-value experiment
2. Run baseline on proper 8×H100 NVLink node (not Thunder PCIe)
3. Explore training loop throughput optimizations (async prefetch, etc.)
4. Consider larger vocab (2048) as untested potential improvement
5. Investigate test-time compute tricks (eval-time depth recurrence, longer context)

### What NOT to Do
- ❌ More recurrence experiments (dead end at 16MB budget)
- ❌ MLA at dim=512 (too aggressive compression)
- ❌ MTP without fundamental rethinking (gradient inflation unsolved)
- ❌ BitNet/ternary quantization (no training speed benefit on H100)
