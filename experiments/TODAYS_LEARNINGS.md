# Today's Learnings — March 20, 2026

## Executive Summary
Ran 9 experiments (080-088) testing layer depth, hparams, activations, positional encoding, FP16 stripping, and INT5 quantization. **Best submission-ready: exp084 at 1.1427 BPP sliding (9L, leaky_relu(0.5)²). Best BPP overall: exp087 at 1.1391 (10L, 252KB over budget).** Critical finding: INT5 QAT during training is catastrophic; INT5 must be post-training only. Largest remaining bottleneck: our serialization produces 3MB bigger artifacts than PR198 for identical params.

---

## Experiment Results Table

| Exp | Change (one variable) | Standard BPP | Sliding BPP | Artifact | Fits? | Verdict | Why |
|-----|----------------------|-------------|-------------|----------|-------|---------|-----|
| **080** | 10L + PR198 hparams (WD=0.04, LR=0.025) | — | 1.1412 | 16.89MB | ❌ | **HURT (slightly)** | Higher WD saved only 40KB; LR=0.025 needs more steps than 10L gets. 10L doesn't fit regardless of hparams. |
| **081** | 9L + PR198 hparams + BigramHash | 1.1653 | **1.1441** | 15.78MB | ✅ | **HELPED (+0.0007)** | WD=0.04 both + LR=0.025 + SCALAR_LR=0.025 + TIED_EMBED_LR=0.035 is a better hparam set. Marginal gain over exp075 (1.1448). |
| **082** | 10L + `.clone().contiguous()` fix | 1.1626 | — | 16.89MB | ❌ | **NEUTRAL (zero effect)** | Artifact gap is from different weight VALUES (entropy), not tensor storage format. zstd is deterministic — same bytes → same output. |
| **083** | abs² activation (replace relu²) | 1.1694 | 1.1480 | 15.78MB | ✅ | **HURT (−0.004 BPP)** | relu²'s zero-gating on negatives induces sparsity that helps both quality and weight regularity. abs² preserves all negatives, losing this beneficial sparsity mechanism. |
| **084** | leaky_relu(0.5)² activation | 1.1639 | **1.1427** | 15.76MB | ✅ | **HELPED (+0.0014) ⭐ NEW BEST** | Softer gating (shrink negatives 50% vs. zero) preserves gradient flow while maintaining partial sparsity. Best of both worlds between relu² and abs². |
| **085** | Partial RoPE 25% of head dims | 1.1701 | — | 15.73MB | ✅ | **HURT (−0.006 BPP)** | With 64-dim heads, 25% = only 16 dims get positional info. Catastrophically insufficient positional signal. Model needs RoPE on all head dims at this scale. |
| **087** | 10L + no FP16_KEEP + no BigramHash | 1.1602 | **1.1391** | 16.25MB | ❌ (252KB over) | **BEST BPP but over budget** | Removing FP16 passthrough saves ~470KB vs exp086 (16.72→16.25MB). 10L gives +0.0036 BPP over best 9L. INT5_MLP would close the 252KB gap. |
| **088** | INT5_MLP with int5 QAT during training | 1.1679 | 1.1468 | 16.61MB | ❌ | **HURT BADLY (−0.008 BPP, +360KB)** | Int5 fake quant during training constrains MLP to [-16,15], losing optimization capacity. andrewgcodes uses int6 QAT during train + int5 only at post-quant. Training doesn't need to "know" about int5. |

### Exp086 (referenced, no dedicated file)
- Config: 10L + BigramHash → 16.72MB ❌
- BigramHash adds ~470KB artifact cost on 10L

---

## Activation Function Sweep Summary

| Activation | Sliding BPP | Δ vs relu² | Mechanism |
|-----------|-------------|------------|-----------|
| relu² (baseline, exp081) | 1.1441 | — | Hard zero gate on negatives (full sparsity) |
| abs² (exp083) | 1.1480 | +0.004 (worse) | No gating at all (no sparsity) |
| **leaky_relu(0.5)² (exp084)** | **1.1427** | **−0.0014 (better)** | Soft gate (50% shrink, partial sparsity) |

**Conclusion**: Moderate sparsity > full sparsity > no sparsity. The sweet spot is preserving gradient flow for negative activations while still maintaining a sparsity-inducing bias. leaky_relu(0.5)² is strictly superior.

---

## Critical Discovery: 3MB Artifact Bloat (from PR198 Reproduction)

| Config (9L, same params) | PR198 Script | Our Script | Gap |
|--------------------------|-------------|------------|-----|
| Artifact size | **12.71MB** | **15.78MB** | **3.07MB** |

**Root causes identified:**
1. **FP16 passthrough** (~770KB): We keep tok_emb + 2 K projections in fp16; PR198 quantizes everything
2. **Extra metadata/structures**: Disabled features (outlier splitting, blockwise quant) still add overhead to state dict
3. **Different state dict structure**: Extra tensors from BigramHash/SmearGate when disabled

**Impact**: This is the #1 bottleneck. Fixing it makes 10L (currently 16.25MB) fit at ~13.2MB and 11L (currently 18.95MB) potentially fit at ~15.9MB.

---

## Critical Discovery: INT5 Must Be Post-Training Only

| Step | INT5 QAT (exp088) | INT6 QAT + INT5 post-quant (planned exp089) |
|------|-------------------|---------------------------------------------|
| During training | fake_quantize to [-16,15] for MLP | fake_quantize to [-32,31] for all |
| At serialization | int5 for MLP, int6 for attn | int5 for MLP, int6 for attn |
| Sliding BPP | 1.1468 (−0.008!) | ~1.1391 (match exp087) |
| Artifact | 16.61MB (+360KB!) | ~14.5MB (estimated) |

**Key insight**: The training optimizer needs full int6 dynamic range to find good weight values. Constraining MLP weights during training to 5-bit precision causes the model to converge to a worse optimum. The int5 quantization should only happen at the final serialization step, where the compression benefit (1.88× vs 1.51× for zstd) is reaped without quality cost.

---

## Key Insights from Competitive Analysis

### andrewgcodes PR#4 (1.1385 BPP, 15.87MB) — Current Target to Beat

Their improvement stack (cumulative):
| Technique | BPP Improvement | Implementation Effort |
|-----------|----------------|----------------------|
| Sliding window eval | ~0.03 | Already have |
| Int6 quant + STE QAT | ~0.04 | Already have |
| TTT (5 epochs SGD post-quant) | ~0.003-0.006 | LOW (50 lines) |
| WD=0.04→0.08 | ~0.002 | Trivial |
| BigBigram 16K→16384×64 | ~0.006 | Medium |
| SWA (start=30%, every=20 steps) | ~0.002 | LOW |
| 10L + Int5 MLP post-quant | enables depth | HIGH priority |
| 2% magnitude pruning | compression only | TRIVIAL (6 lines) |
| Attention gate (per-head sigmoid) | unknown | LOW (10 lines) |

**Critical**: Their STE is DISABLED in the winning config (INT6_STE=0). The int5/int6 quantization is purely post-training. They use zstd-22, standard torch.save, no custom serialization.

---

## What Works vs. What Doesn't (Cumulative Knowledge)

### ✅ CONFIRMED WINNERS
| Technique | BPP Impact | Artifact Impact | Source |
|-----------|-----------|----------------|--------|
| leaky_relu(0.5)² | +0.0014 | Neutral | exp084 |
| NorMuon optimizer | +0.004 | Neutral | exp058 |
| Int6 STE QAT (during train) | +0.002 | Neutral | exp060 |
| WD=0.04 both optimizers | +0.003 | −0.01MB | exp072/081 |
| PR198 LRs (LR=0.025, etc.) | +0.0007 | Neutral | exp081 |
| BigramHash (int6 quantized) | +0.001 | +400KB | exp075 |
| Removing FP16_KEEP | Neutral | −470KB | exp087 |
| 10 layers (vs 9) | +0.0036 | +480KB (int6) | exp087 vs 084 |
| Sliding window eval stride=64 | +0.02 | N/A | exp038 |

### ❌ CONFIRMED LOSERS
| Technique | BPP Impact | Why It Failed |
|-----------|-----------|---------------|
| abs² activation | −0.004 | No sparsity induction |
| Partial RoPE 25% | −0.006 | Insufficient positional signal at dim=64 |
| INT5 QAT during training | −0.008 | Constrains optimizer to worse minimum |
| .clone().contiguous() fix | 0.000 | Artifact gap is weight entropy, not storage |
| ROPE_BASE=200K | 0.000 | Neutral at this scale |
| SWA with WD≤0.04 | ≤0.000 | Not enough regularization for SWA to help |
| Bit-packing int6 | 0.000 BPP, +1MB | Removes zstd-exploitable redundancy |
| Outlier splitting | 0.000 BPP, +1MB | Extra tensor overhead |

### 🔬 UNTESTED BUT PROMISING (from competition analysis)
| Technique | Expected BPP | Risk | Priority |
|-----------|-------------|------|----------|
| INT5 MLP post-quant only | enables 10L (−2MB artifact) | Very low | **CRITICAL** |
| Fix artifact serialization bloat | enables 10-11L (−3MB) | Low | **CRITICAL** |
| TTT (5ep SGD lr=0.003) | +0.003-0.006 | Low | **HIGH** |
| 2% magnitude pruning | compression only | Very low | **HIGH** |
| WD=0.08 + SWA(30%/20) | +0.002 | Low | **HIGH** |
| BigBigram 16384×64 | +0.006 | Medium (artifact) | **MEDIUM** |
| Attention gate | unknown | Very low | **MEDIUM** |
| Muon momentum 0.99 | +0.001-0.002 | Very low | **MEDIUM** |
| 11 layers | +0.005-0.01 | Needs artifact fix | **MEDIUM** |

---

## Concrete Next Experiments (Priority Order)

### 🔴 CRITICAL PATH: Unlock 10+ Layers

| Exp # | Description | Expected Outcome | Rationale |
|-------|-------------|-----------------|-----------|
| **089** | 10L + int6 QAT train + INT5 MLP post-quant + TTT + 2% pruning (RUNNING) | 1.1391 BPP, ~14.5MB ✅ | The single most important experiment. Fixes exp088's mistake by separating train-time QAT (int6) from post-quant (int5). |
| **090** | Fix serialization to match PR198 (strip disabled code paths, simplify state dict) | −2-3MB artifact savings | Our code produces 3.07MB bigger artifacts for identical params. This alone could make 11L fit. |
| **091** | 10L + fixed serialization + INT5 post-quant + leaky2 | ~1.139 BPP, ~12-13MB ✅ | Combines the two critical fixes. Should have massive artifact headroom. |

### 🟡 HIGH PRIORITY: Stack Proven Winners

| Exp # | Description | Expected Outcome | Rationale |
|-------|-------------|-----------------|-----------|
| **092** | exp091 + BigramHash 16384×64 | ~1.133 BPP | andrewgcodes showed 0.006 BPP from bigger bigram. With artifact headroom, this fits. |
| **093** | exp092 + WD=0.08 + SWA(start=30%, every=20) | ~1.131 BPP | Their WD=0.08 is the key ingredient that makes SWA work. We only tested WD≤0.04. |
| **094** | exp093 + TTT 5ep SGD | ~1.127 BPP | Free eval-time improvement. Recovers quant gap + adapts to val distribution. |
| **095** | exp094 + attention gate + Muon momentum 0.99 | ~1.125 BPP | Small additional gains from proven techniques. |

### 🟢 STRETCH: If Time Permits

| Exp # | Description | Expected Outcome | Rationale |
|-------|-------------|-----------------|-----------|
| **096** | 11L version of best config | ~1.12 BPP | 11L was 0.009 better than 10L in earlier tests (exp076 vs exp079). With fixed serialization + INT5, should fit. |
| **097** | Partial RoPE 50% or 75% | Unknown | 25% was too aggressive; higher fractions may help at reduced cost. Low priority given 25% failure. |
| **098** | zstd-22 instead of zstd-19/zlib-9 | −100-300KB artifact | andrewgcodes uses zstd-22. Free compression improvement if we're not already using it. |
| **099** | Stride=256 eval (instead of 64) | +0.000-0.005 BPP | PR114 showed stride=256 was BETTER than 64. Free, no retraining. |

---

## Projected Path to Beating andrewgcodes (1.1385 BPP)

```
Current best (exp084):     1.1427 BPP, 15.76MB ✅
                           ──────────────────────────
+ Fix serialization:       1.1427 BPP, ~12.7MB ✅  (−3MB artifact)
+ 10 layers:               1.1391 BPP, ~13.2MB ✅  (+0.0036 BPP)
+ INT5 MLP post-quant:     1.1391 BPP, ~12.0MB ✅  (−1.2MB artifact)
+ BigBigram 16384×64:      ~1.1330 BPP, ~12.5MB ✅ (+0.006 BPP)
+ WD=0.08 + SWA:           ~1.1310 BPP, ~12.3MB ✅ (+0.002 BPP)
+ TTT 5 epochs:            ~1.1260 BPP, ~12.3MB ✅ (+0.005 BPP)
+ 11 layers (if fits):     ~1.1170 BPP, ~14.5MB ✅ (+0.009 BPP)
                           ──────────────────────────
Target:                    < 1.1385 BPP (beat andrewgcodes)
Projected:                 ~1.117-1.126 BPP ← SIGNIFICANT WIN
```

**The #1 blocker is artifact serialization bloat.** Once fixed, we have 3-4MB of headroom to add layers, bigger bigrams, and other parameter-hungry improvements. Everything else is additive.

---

## Meta-Learning: What We Learned About Methodology

1. **2K-step screening is reliable for architecture/activation changes** (exp083/084/085 gave clear signal) but unreliable for optimizer/schedule changes (exp004 showed early advantage that vanished).

2. **Always separate train-time and post-training techniques.** INT5 QAT (exp088) was a costly mistake because we assumed train-time awareness would help. andrewgcodes proved the opposite — the optimizer needs freedom, quantization should be imposed after.

3. **Artifact size is as important as BPP.** Our best BPP (exp076, 1.1304) is useless at 18.95MB. The competition is won at the intersection of quality and compression.

4. **Platform-specific compression differences are real but code-fixable.** The 3MB gap isn't hardware — it's our serialization code. PR198 proves the same hardware produces smaller artifacts.

5. **Competition PR analysis is extremely high-ROI.** Today's andrewgcodes analysis revealed INT5 post-quant, TTT, pruning, WD=0.08+SWA, and attention gates — all techniques we hadn't considered or had dismissed prematurely.
