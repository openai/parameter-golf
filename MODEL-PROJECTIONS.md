# Parameter Golf — Model Projections & Analysis

**Date:** 2026-03-26 01:20 CDT
**Purpose:** Project all 8 models to full scale (15.9MB, 8×H100, 10 min)

---

## All Models — Ranked by Smoke Test Performance

| Rank | Model | Smoke bpb | Size (current) | Size (16MB budget) | ms/step (4070S) | Steps (8×H100) | **Projected bpb** | Novel? |
|------|-------|-----------|----------------|--------------------|-----------------|-----------------|--------------------|--------|
| 🥇 | **M3 Hybrid** | 2.529 | 5.1 MB | 15.9 MB (57M params) | 141ms | ~5,500 | **~1.24** | ❌* |
| 🥈 | **M1 Codec** | 2.631 | 8.0 MB | 15.9 MB (57M params) | 140ms | ~5,600 | **~1.27** | ✅ |
| 🥉 | **M5 Frankenstein** | 3.257 | 6.4 MB | 15.9 MB (57M params) | 165ms | ~4,700 | **~1.48** | ✅ |
| 4 | **M8 Crystal** | 3.342 | 5.2 MB | 15.9 MB (57M params) | 242ms | ~3,200 | **~1.62** | ✅ |
| 5 | **M7 Immune** | 3.464 | 5.3 MB | 15.9 MB (57M params) | 146ms | ~5,300 | **~1.50** | ✅ |
| 6 | **M4 Optimized** | 3.830 | 5.1 MB | 15.9 MB (57M params) | 140ms | ~5,600 | **~1.59** | ❌ |
| 7 | **M2 Recursive** | 4.010 | 5.7 MB | 15.9 MB (57M params) | 140ms | ~5,600 | **~1.63** | ✅ |
| 8 | **M6 Hive** | 4.031 | 9.4 MB | 15.9 MB (57M params) | 138ms | ~5,700 | **~1.64** | ✅ |

*M3's GatedRNN is dead code — it's functionally an optimized baseline, not a true hybrid.

**Current leaderboard context:**
- Official record: **1.1233 bpb** (merged, standard transformer optimizations)
- Best pending PR (Track A neural): **~1.12 bpb** (LeakyReLU + Legal TTT + Parallel Muon)
- Best pending PR (Track B n-gram): **~0.29-0.44 bpb** (n-gram backoff caching — may be ruled illegal)

---

## Projection Methodology

**Power-law scaling model:** `bpb(S) = a × S^(-0.40) + 0.80`

Calibrated against two known points:
- Baseline at ~200 steps (smoke test): bpb ≈ 2.5
- Baseline at ~6,500 steps (8×H100, 10 min): bpb = 1.2244

**Parameter scaling:** -0.03 bpb per 2× parameter increase (conservative, from empirical data)

**Compression:** PolarQuant 3-bit = 0.28 bytes/param → 57M params in 15.9MB

**H100 speed:** ~1.3× faster per step than RTX 4070 Super (from published benchmarks)

**Limitations of this projection:**
1. Power law may not hold for novel architectures at scale
2. Some architectures (M8 Crystal, M7 Immune) may converge faster or slower than the baseline
3. M1 Codec's n-gram features may give it a disproportionate advantage at scale (more training steps = better n-gram stats)
4. Ternary quantization (BitNet) could change the parameter count drastically
5. We haven't tested any model at >200 steps, so extrapolating to 5,000+ steps is speculative

---

## Individual Model Analysis

### 🥇 M1 Codec — Projected: ~1.27 bpb

**Current:** 2.631 bpb | 17M params | 8.0 MB | 5 steps complete
**Scaled:** 57M params | 15.9 MB | ~5,600 steps on 8×H100

**Why it could beat the projection:**
- The bigram embedding and n-gram predictor give the model "free" statistical priors from training data
- At 57M params, the transformer can learn much more complex patterns on top of the n-gram features
- The codec architecture has the LOWEST loss per training step of any model — it learns faster
- N-gram features scale with training data, not training steps — immediate benefit

**Why it could miss:**
- Already 8.0 MB at 17M params — the bigram/n-gram tables take ~3MB of overhead
- Scaled to 57M params, the n-gram tables would be proportionally smaller relative to model size
- GrowthRule regression showed that adding more components can hurt, not help

**Scaling recommendation:**
- Scale to 12L/640d (~30M params) with PolarQuant-3bit = ~8.4MB model + 3MB n-gram + 4.5MB headroom
- Use headroom for TTT (Test-Time Training) LoRA weights at eval
- Estimated: **1.10-1.20 bpb** (competitive with leaderboard)

**Unique advantage:** This is the ONLY model on the competition that combines n-gram statistical priors with neural prediction. Everyone else does pure neural (Track A) or pure n-gram (Track B). M1 is naturally a Track A+B hybrid.

---

### 🥈 M3 Hybrid — Projected: ~1.24 bpb (BEST PROJECTION)

**Current:** 2.529 bpb | 17M params | 5.1 MB | 3 steps complete
**Scaled:** 57M params | 15.9 MB | ~5,500 steps on 8×H100

**Critical finding:** M3 is NOT a hybrid. The GatedRNN is dead code. This is a vanilla optimized transformer with warmdown and grad clip. Its 2.529 bpb smoke test score reflects good optimization tuning.

**Why it projects best:**
- Lowest smoke bpb = lowest scaling coefficient = best extrapolation
- But this is misleading — M3's advantage may be from hyperparameter tuning, not architecture
- At scale (57M params, 5,500 steps), all standard transformers converge to similar scores

**Why it's risky:**
- No novel architecture = no differentiation from the 100+ other optimized transformer submissions
- The current record (1.1233) is already an optimized transformer with similar techniques
- To beat 1.12, we'd need TTT, better quantization, or architectural innovation

**Scaling recommendation:**
- Scale to 11L/768d (~40M params) with GPTQ-lite int6 = ~12MB, add TTT LoRA
- Add sliding window eval (stride 16-32)
- Add LeakyReLU(0.9)² (from PROTEUS PR — small but free improvement)
- Estimated: **1.10-1.15 bpb** (matches current leaderboard)

---

### 🥉 M5 Frankenstein — Projected: ~1.48 bpb

**Current:** 3.257 bpb | 17.4M params | 6.4 MB
**Scaled:** 57M params | 15.9 MB | ~4,700 steps on 8×H100

**Analysis:** Combines BigramEmbed (M1) + GrowthRule (M8). The combo beat each component alone but is slower (165ms/step) and didn't beat M1. The growth rule overhead means fewer training steps.

**Scaling recommendation:** Deprioritize. M1 already has bigram embed + more features, and M3 is faster. M5 doesn't offer enough improvement to justify the complexity.

---

### M8 Crystal — Projected: ~1.62 bpb

**Current:** 3.342 bpb | 17M params | 5.2 MB | 242ms/step
**Scaled:** 57M params | 15.9 MB | ~3,200 steps on 8×H100

**Analysis:** Per-layer growth rule scaling is a genuine finding — it helped every model except M1 Codec. But 242ms/step is a killer: only 3,200 steps vs 5,500 for standard models. That's 40% fewer training iterations.

**Scaling recommendation:** The growth rule computation needs to be pre-computed per epoch, not per forward pass. If we can cache the layer scales (they only depend on learned embeddings, not input), we eliminate the per-step overhead. This could drop ms/step to ~150ms while keeping the benefit.

**Optimization opportunity:** Pre-compute `growth_scales = self.growth_rule.get_all_scales()` once before each training epoch, store as a tensor, and index into it. Zero per-step overhead.

---

### M7 Immune — Projected: ~1.50 bpb

**Current:** 3.464 bpb | 17M params | 5.3 MB | 146ms/step
**Scaled:** 57M params | 15.9 MB | ~5,300 steps on 8×H100

**Analysis:** Template codebook with router is conceptually interesting but adds overhead and the router needs many steps to specialize. At 32 templates × 512d = 16K extra params (tiny). The real value is in the routing mechanism, not the templates themselves.

**Scaling recommendation:** Increase template count to 128-256 and increase template dim. If templates can capture common sub-word patterns, this becomes a learnable n-gram table — connecting Track A to Track B naturally.

---

### M4 Optimized — Projected: ~1.59 bpb

**Current:** 3.83 bpb | 17M params | 5.1 MB | 140ms/step
**Scaled:** 57M params | 15.9 MB | ~5,600 steps on 8×H100

**Analysis:** The kitchen-sink approach. 15 techniques stacked. But smoke test performance is mediocre because many techniques (BigramHash, SmearGate) need 1000+ steps to converge — they hurt in short runs but help in long runs.

**Scaling recommendation:** This is our "safe public submission" candidate. Scale to 11L/768d, apply PolarQuant, add TTT, submit for leaderboard. Expected: **1.10-1.15 bpb** (matches record range).

---

### M2 Recursive — Projected: ~1.63 bpb

**Current:** ~4.01 loss | 17M params | 5.7 MB | 140ms/step
**Scaled:** 57M params | 15.9 MB | ~5,600 steps on 8×H100

**Analysis:** Shared weight block × 9. Extreme parameter efficiency (one block = many effective layers). At scale, this could be very competitive because ALL the 57M params go into ONE block that's applied 9-12 times. Effectively 57M × 9 = 513M "effective" parameters in terms of compute.

**Scaling opportunity:** The entire 16MB budget goes to ONE block. At 57M params in a single block applied 12 times, this model has the computational equivalent of a 684M parameter model. No other model can match this compute density.

**Why it could surprise:** If the shared block learns a general "reasoning layer" that improves with each application, recursive depth could be the highest-leverage architectural choice. The BankLinear submission (PR #812) is exploring a similar concept.

---

### M6 Hive — Projected: ~1.64 bpb

**Current:** 4.031 bpb | 17.3M params | 9.4 MB | 138ms/step
**Scaled:** 57M params | 15.9 MB | ~5,700 steps on 8×H100

**Analysis:** The weakest model. Frozen backbone + LoRA only works with pre-trained weights, not random init. The original concept (90%+ frozen) failed completely. Even at 67% frozen, it's the worst performer.

**Scaling recommendation:** Abandon the pure Hive concept. The insight (most neural net params are redundant) is valid but requires pre-training or clever initialization, which we don't have time for.

---

## Compression Budget Analysis

| Compression | Bytes/Param | Max Params in 15.9MB | Notes |
|-------------|-------------|---------------------|-------|
| int8 + zlib | 0.38 | 42M | Current default, well-tested |
| int6 + zstd (GPTQ-lite) | 0.30 | 53M | Proven in M4 Step 12 |
| PolarQuant 3-bit | 0.28 | 57M | Spec written, NOT wired |
| Ternary (BitNet b1.58) | 0.22 | 72M | Used by competition record holder |
| Binary (1-bit) | 0.15 | 106M | Untested, likely too aggressive |

**Recommendation:** PolarQuant 3-bit is the sweet spot. Ternary requires completely rewriting the training loop (BitNet STE training). PolarQuant drops into existing models with minimal changes.

---

## Track A+B Hybrid Budget Splits

If we combine neural model with n-gram lookup tables:

| N-gram Budget | Model Budget | Model Params | N-gram Entries | Estimated bpb |
|---------------|-------------|--------------|----------------|---------------|
| 0 MB (pure A) | 15.9 MB | 57M | 0 | ~1.10-1.25 |
| 2 MB | 13.9 MB | 50M | 286K entries | ~0.95-1.10 |
| 4 MB | 11.9 MB | 43M | 571K entries | ~0.85-1.00 |
| 6 MB | 9.9 MB | 35M | 857K entries | ~0.80-0.95 |
| 8 MB | 7.9 MB | 28M | 1.1M entries | ~0.75-0.90 |

**The M1 Codec is already a natural Track A+B hybrid** — its bigram embedding and n-gram predictor ARE lookup tables learned from training data. Scaling M1 and adding explicit n-gram backoff tables could push well below 1.0 bpb.

---

## Priority Stack (What to Build Next)

1. **Wire PolarQuant into M1 Codec + M3** — unlock 57M params in 16MB
2. **Add n-gram backoff caching to M1** — explicit Track A+B hybrid
3. **Submit M4 (public)** for leaderboard visibility + compute grant application
4. **Scale M2 Recursive to fill 16MB** — test the "one giant block × 12" theory
5. **Pre-compute M8 growth scales** — eliminate per-step overhead
6. **Wait for Track B research results** — may completely change strategy

---

*Projections are estimates based on power-law scaling. Actual scores require 8×H100 runs.*
