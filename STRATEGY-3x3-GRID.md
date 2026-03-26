# Parameter Golf — 3×3 Strategy Grid

**Date:** 2026-03-26 02:00 CDT
**Purpose:** Master plan for 9 models across 3 tracks + 2 immediate submissions

---

## Immediate Priority: Get on the Leaderboard

| Submission | Base | Status | Goal |
|-----------|------|--------|------|
| **Sub-1: M4 Optimized (scaled)** | Incremental from baseline | Building now (Step 1 passed) | ~1.15-1.20 bpb |
| **Sub-2: M1 Codec (scaled)** | Incremental from M1 Step 5 | After Sub-1 validated | ~1.10-1.25 bpb |

These get us on the leaderboard and qualify for compute grant. Pure Track A, no risk.

---

## Track A — Pure Neural (maximize 16MB with model params only)

| Model | Architecture | Compression | Projected bpb | Risk |
|-------|-------------|-------------|---------------|------|
| **A1** | 11L/768d optimized transformer + LeakyReLU(0.9)² + all training opts | PolarQuant 3-bit (57M params) | 1.08-1.15 | Low |
| **A2** | M1 Codec scaled (bigram embed + n-gram predictor + transformer) | GPTQ-lite int6 (40M+ params) | 1.05-1.15 | Low |
| **A3** | Ternary BitNet b1.58 (72M params, following competition record approach) | Ternary + base-3 LZMA | 1.10-1.20 | Medium (new training method) |

**Key:** These are rule-safe no matter what. If Track B gets rejected, these are our submissions.

---

## Track B — N-gram Focused (maximize eval-time statistical memory)

| Model | Architecture | Eval-Time Technique | Projected bpb | Risk |
|-------|-------------|-------------------|---------------|------|
| **B1** | Baseline transformer + BackoffNgramMixer | 5MB n-gram cache, backoff orders 2-10 | 0.60-0.75 | Medium (rule risk) |
| **B2** | Complementary-trained transformer + score-first TTT | 5MB n-gram + TTT LoRA + entropy gating | 0.35-0.55 | Medium-High (eval time + rule risk) |
| **B3** | Hunter Alpha's Hive Mind (8 specialist transformers) | Per-GPU specialization + auction routing | 0.27-0.40 | High (untested, complex) |

**Key:** These push for the best absolute score. Rule risk exists but community consensus says they're legal.

---

## Track C — Hybrid (best of Track A + Track B)

| Model | Architecture | Hybrid Approach | Projected bpb | Risk |
|-------|-------------|----------------|---------------|------|
| **C1** | PolarQuant neural (7-11MB) + n-gram cache (4-6MB) + entropy gating | Chimera/Dominator (consensus strategy from 6 think tanks) | 0.35-0.55 | Medium |
| **C2** | M1 Codec (bigram+n-gram IN architecture) + eval-time n-gram backoff | Natural hybrid — n-gram inside AND outside the model | 0.40-0.60 | Medium |
| **C3** | Recursive shared block (0.5-7MB) + massive n-gram cache (8-10MB) | Oracle — minimum neural, maximum statistical memory | 0.55-0.90 | High |

**Key:** These are the most likely winners. Neural handles what n-grams can't; n-grams handle what neural is slow to learn.

---

## Build Order

### Phase 1: Leaderboard Entry (NOW → next 24h)
1. Finish incremental build of Sub-1 (M4 scaled)
2. Smoke test on 4070S → 1×H100 → 8×H100
3. Submit for leaderboard score
4. Repeat for Sub-2 (M1 Codec scaled)
5. Apply for compute grant with scores

### Phase 2: Cross-Synthesis (next 24h)
1. Full cross-analysis of 6 think tank documents
2. Write FINAL-STRATEGY.md with unified recommendations
3. Identify the #1 model from each track to build first

### Phase 3: 3×3 Grid Build (Days 3-15)
1. Build one model at a time, incrementally
2. Track A models first (safest)
3. Track C hybrid next (highest expected value)
4. Track B pure n-gram last (most novel, most risk)
5. Each model: spec → build → smoke test → fix → H100 test → score

### Phase 4: Optimization (Days 15-28)
1. Take best model from each track
2. Optimize aggressively (PolarQuant, TTT, n-gram table optimization)
3. Multi-seed validation (42, 1337, 2025)
4. Final submission prep

### Phase 5: Submission (Days 28-30, April 28-30)
1. Submit best Track A model (safe, public)
2. Submit best Track C hybrid (competitive, may be public or private)
3. Hold best Track B or novel model for last-minute drop

---

## Think Tank Documents (Source Material)

| Document | Author | Size | Key Contribution |
|----------|--------|------|-----------------|
| `PARAMETER-GOLF-STRATEGY-MASTER-HANDOFF.md` | Daedalus | ~15KB | 3 strategies, component ranking, implementation order |
| `MASTER-HANDOFF.md` | Healer Alpha | ~29KB+ | 12 sections, 30-day calendar, 1024-vocab insight |
| `HYPERION-COMBINATORIAL-ANALYSIS.md` | Hyperion | ~10KB | Build foundation once, branch variants |
| `MASTER-STRATEGY-HANDOFF-2026-03-26.md` | Artemis | ~29KB | Eval time risk, Grok Special (n-gram as training prior) |
| `HERMES-COMPREHENSIVE-HANDOFF.md` | Hermes/Grok | ~15KB | Achilles aggression, Track B beast |
| `THREE-MEGA-STRATEGIES.md` | Hunter Alpha | ~58KB | Hive Mind (8 GPU specialists), Crystal Brain |
| `MASTER-HANDOFF.md` (Hunter) | Hunter Alpha | ~20KB | Full competition landscape + 881-line reference |

**Total think tank output: ~175KB+ of analysis from 6 independent LLMs + their sub-agents**

---

## Key Research Files

| File | Content |
|------|---------|
| `TRACK-B-RESEARCH.md` | Deep technical analysis of n-gram approaches (Prometheus) |
| `TRACK-B-SOCIAL-INTEL.md` | Social intel on Track B legality (Hermes) |
| `TURBOQUANT-SPEC.md` | PolarQuant implementation spec |
| `MODEL-PROJECTIONS.md` | All 8 models projected to 16MB/8×H100 |
| `MODEL-SPECS.md` | Definitive specs for all 8 original models |
| `MODEL-BUILD-LOG.md` | Error history and lessons for all models |
| `PROGRESS.md` | Build status chart |
| `COMBINATORIAL-ANALYSIS-HANDOFF.md` | Original handoff doc given to all 6 think tanks |

---

*This is the master plan. Build incrementally. Test everything. Don't lose context.*
