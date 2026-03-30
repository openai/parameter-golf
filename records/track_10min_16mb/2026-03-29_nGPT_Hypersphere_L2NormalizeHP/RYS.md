# RYS (Repeat Your Self) — Layer Repetition Experiments

## Concept
Run a contiguous block of transformer layers twice during inference. Zero extra parameters — same weights, more compute. Based on [dnhkng's RYS blog](https://dnhkng.github.io/posts/rys/) which achieved top HuggingFace leaderboard results by repeating middle layers of Qwen2-72B.

---

## Interpretability Analysis

### 1. Logit Lens (per-layer prediction quality)

Project each layer's hidden state to vocabulary space. Shows when the model "knows" the next token.

| Layer | BPB | Δ BPB | Function |
|-------|-----|-------|----------|
| embed | 10.34 | — | Raw token embeddings |
| **0** | 5.46 | **-4.88** | Basic token patterns (biggest single drop) |
| 1 | 4.58 | -0.88 | Pattern matching |
| 2 | 3.82 | -0.76 | Best encoder output |
| 3 | 4.20 | +0.38 | U-Net transition — BPB INCREASES |
| 4 | 4.71 | +0.52 | Still restructuring |
| 5 | 5.29 | +0.57 | Worst point — decoder receiving skip |
| 6 | 4.63 | -0.66 | Recovery begins (skip connections) |
| **7** | **2.57** | **-2.05** | **Critical "thinking" layer — biggest useful drop** |
| 8 | 1.80 | -0.78 | Refining prediction |
| 9 | 1.37 | -0.43 | Nearly done |
| 10 | 1.19 | -0.18 | Final polish |

**Key findings:**
- Layer 7 is the "reasoning cortex" — drops BPB by 2.05, more than any layer except 0
- Layers 3-5 increase BPB — U-Net encoder→decoder transition restructures representation
- Layers 6-8 are the payoff zone — model goes from "confused" (4.63) to "nearly solved" (1.80)

### 2. Layer Knockout (importance when removed)

Zero out each layer's attn_scale + mlp_scale, measure BPB increase.

| Layer | Δ BPB when removed | Importance |
|-------|-------------------|------------|
| **0** | **+2.31** | Absolutely critical — basic patterns |
| 1 | +0.65 | Very important |
| 2 | +0.58 | Very important |
| 3 | +0.29 | Moderate |
| 4 | +0.25 | Moderate |
| **7** | **+0.19** | Most important decoder layer |
| 6 | +0.18 | Similar |
| 8 | +0.16 | Similar |
| 5 | +0.15 | Least important |
| 10 | +0.14 | Least important |
| 9 | +0.14 | Least important |

**Key findings:**
- Layers 0-2 are far more critical than any decoder layer
- Decoder layers (5-10) are roughly equally important (0.14-0.19 each)
- Layer 7 stands out slightly among decoder layers — aligns with logit lens finding
- The model is already quite efficient at layers 7-10 — knockout barely hurts

### 3. CKA Similarity (functional clusters)

Cosine similarity between layer hidden states reveals which layers work as circuits.

```
         embed  L0    L1    L2    L3    L4    L5    L6    L7    L8    L9    L10
embed    1.00  0.07  0.06  0.04  0.04  0.04  0.04  0.08  0.10  0.12  0.11  0.11
L0       0.07  1.00  0.83  0.57  0.35  0.24  0.16  0.15  0.30  0.38  0.35  0.20
L1       0.06  0.83  1.00  0.70  0.44  0.31  0.21  0.20  0.36  0.44  0.36  0.18
L2       0.04  0.57  0.70  1.00  0.71  0.53  0.39  0.36  0.55  0.55  0.44  0.23
L3       0.04  0.35  0.44  0.71  1.00  0.72  0.53  0.49  0.53  0.48  0.36  0.16
L4       0.04  0.24  0.31  0.53  0.72  1.00  0.72  0.55  0.51  0.45  0.32  0.13
L5       0.04  0.16  0.21  0.39  0.53  0.72  1.00  0.66  0.53  0.43  0.30  0.11
L6       0.08  0.15  0.20  0.36  0.49  0.55  0.66  1.00  0.78  0.65  0.49  0.27
L7       0.10  0.30  0.36  0.55  0.53  0.51  0.53  0.78  1.00  0.88  0.73  0.46
L8       0.12  0.38  0.44  0.55  0.48  0.45  0.43  0.65  0.88  1.00  0.88  0.59
L9       0.11  0.35  0.36  0.44  0.36  0.32  0.30  0.49  0.73  0.88  1.00  0.80
L10      0.11  0.20  0.18  0.23  0.16  0.13  0.11  0.27  0.46  0.59  0.80  1.00
```

**Layer-to-next similarity (consecutive transition smoothness):**

| Transition | Similarity | Interpretation |
|-----------|-----------|----------------|
| embed→0 | 0.07 | Huge transformation (encoding) |
| 0→1 | 0.83 | Very smooth (same circuit) |
| 1→2 | 0.70 | Smooth |
| 2→3 | 0.71 | Smooth |
| 3→4 | 0.72 | Smooth |
| 4→5 | 0.72 | Smooth |
| **5→6** | **0.66** | **Lowest — biggest representation shift in decoder** |
| 6→7 | 0.78 | Recovery |
| **7→8** | **0.88** | **Tightest bond — same circuit** |
| **8→9** | **0.88** | **Tightest bond — same circuit** |
| 9→10 | 0.80 | Tight |

**Three functional clusters:**
1. **Encoder circuit (L0-L2):** High mutual similarity (0.83). Pattern matching and basic encoding.
2. **Transition zone (L3-L6):** Gradual transformation. Each layer moderately similar to neighbors. Representation being restructured for decoder.
3. **Prediction circuit (L7-L10):** Tightest cluster (0.88 between L7↔L8 and L8↔L9). These layers work as a single unit refining the final prediction.

### Combined Analysis — Best RYS Target

| Evidence | Layers 3-6 | Layers 6-8 | Layers 7-9 |
|----------|-----------|-----------|-----------|
| Logit lens Δ | BPB goes UP (+0.38 to +0.57) | -0.66, **-2.05**, -0.78 | **-2.05**, -0.78, -0.43 |
| Knockout importance | 0.15-0.29 (moderate) | 0.18, **0.19**, 0.16 | **0.19**, 0.16, 0.14 |
| CKA circuit tightness | 0.66-0.72 (loose) | 0.66, 0.78 (moderate) | **0.88, 0.88** (tightest) |
| Blog's "middle" finding | ✓ middle encoder | ✓ mid-decoder | ✓ mid-decoder |

**Verdict: Layers 7-9 are the strongest RYS candidate.**
- Tightest functional circuit (0.88 similarity — they work as one unit)
- Layer 7 has the biggest logit lens improvement (-2.05)
- They form the "prediction refinement" pipeline
- A second pass lets the model iterate on its prediction before the final layer 10 polish

**Second choice: Layers 6-8** if we want to include the "recovery from confusion" step (layer 6's -0.66 after the U-Net transition).

---

## Implementation

RYS is controlled by env vars:
- `RYS_START`: first layer to repeat (inclusive)
- `RYS_END`: last layer to repeat (exclusive)
- `RYS_TRAIN_AFTER`: step to enable RYS during training (-1 = eval only, 0 = always, N = after step N)

The `_rys_pass` method is decorated with `@torch.compiler.disable` to avoid torch.compile graph breaks. When RYS is configured, `fullgraph=False` is used for the model compile.

---

## Experiments

### Eval-only sweep (uncalibrated)
Load trained model, test different RYS ranges without retraining.

| Config | Layers Repeated | BPB | Δ vs baseline | Status |
|--------|----------------|-----|---------------|--------|
| No RYS (baseline) | — | ~1.1195 | — | Known |
| L40S eval sweep | all configs | — | — | CANCELLED (too slow) |

### Quick eval on 3-6 calibrated model (20 sequences, not full sliding window)

Model trained with RYS 3-6 calibration for last 500 steps.

| Config | BPB | Δ vs baseline |
|--------|-----|---------------|
| No RYS (baseline) | 1.2835 | — |
| RYS 3-6 (calibrated range) | 4.6113 | +3.33 (destroyed) |
| RYS 6-8 (uncalibrated) | 4.0546 | +2.77 (destroyed) |
| RYS 7-9 (uncalibrated) | 2.0264 | +0.74 (bad) |
| **RYS layer 7 only** | **2.0957** | **+0.81** |
| RYS 6-7 | 3.5878 | +2.30 |
| RYS 7-8 | 2.2007 | +0.92 |

**Finding: RYS hurts massively at every layer range**, even the calibrated one. The model is too small — each layer's output is precisely tuned for the next layer's input, and repeating produces out-of-distribution hidden states.

Note: baseline BPB 1.2835 (quick eval) is worse than our real baseline 1.1195 (full sliding window) because quick eval uses only 20 sequences without stride overlap.

### Calibrated fork runs (full sliding window eval)
Trained to step 7700 (checkpoint), then forked with different RYS configs for 500 calibration steps. Each fork: load checkpoint → enable RYS → train 500 steps → GPTQ → full sliding window eval with RYS active. NO_COMPILE=1 (eager mode) to avoid torch.compile issues.

| Config | Layers Repeated | BPB | Δ vs baseline | Count |
|--------|----------------|-----|---------------|-------|
| **No RYS (baseline)** | **—** | **1.1177** | **—** | 0 |
| RYS layer 7 only | 7 | 1.1385 | +0.021 | 1 |
| RYS 7-9 | 7,8,9 | 1.1605 | +0.043 | 3 |
| RYS 6-8 | 6,7,8 | 1.1921 | +0.074 | 3 |
| RYS 3-6 | 3,4,5,6 | 1.7554 | +0.638 | 4 |

**Observations:**
1. Every RYS config hurts — no layer range improves over baseline
2. More layers repeated = more damage (linear relationship)
3. Layer 7 solo (+0.021) is the least harmful — aligns with logit lens showing it as the most important decoder layer
4. Transition zone layers 3-6 are catastrophic (+0.638) — confirms logit lens finding that these layers restructure representations
5. 500 steps of calibration is not sufficient to teach the model to use repeated layers

**Not yet tested:**
- Longer calibration (1000+ steps)
- Calibration with much lower LR
- RYS only at eval time (no calibration) on clean baseline model
- Logit ensemble approach (average logits from first and second pass)
- Dedicated refinement block (separate parameters, trained for iteration)
- Single layer with gated residual (α-scaled second pass)

### Technical Issues Encountered
1. **torch.compile cache_size_limit**: Enabling RYS mid-training changes forward graph → dynamo hits recompilation limit. Fix: `torch._dynamo.config.cache_size_limit = 64`
2. **torch.compile fullgraph + @compiler.disable**: PyTorch 2.6 can't handle `@torch.compiler.disable` at all — even with `fullgraph=False`. Fix: remove decorator, inline RYS logic.
3. **Must disable fullgraph on ALL compile calls**: `compiled_model`, `compiled_eval`, AND `compiled_logits` all need `fullgraph=False` when RYS is active.
4. **L40S too slow for eval**: Sliding window eval on L40S without compile takes 30+ min per config.

---

## Stochastic RYS Campaign (2026-03-26/27)

Standard RYS failed, so we developed **Stochastic RYS (SRYS)**: randomly repeat target layers during training with probability p, so layers learn to produce refinable representations. Full results in EXPERIMENT_RESULTS.md.

### Summary of approaches tested

| Approach | Architecture | Best RYS Δ | Outcome |
|----------|-------------|-----------|---------|
| Ungated SRYS (p=0.1-0.5) | 512×11 | -0.0005 | Layers learn near-identity (cos_sim=0.999) |
| Multi-repeat training (1-3) | 512×11 | -0.0004 | No compounding, ×3+ hurts |
| Gated SRYS | 512×11 | -0.0001 eval / -0.0028 base | Gate closes to 0.03-0.10; base improvement is regularization effect |
| Thin-deep (384×20, 352×24) | 384-352 dim | -0.0009 | More depth → better RYS stability, but width loss > RYS gain |
| Attn-only hybrid (9full+6attn) | 512×15 | -0.0002 | cos_sim dropped to 0.35 (big delta!) but gate closed to 0.06 |
| Ungated attn-only | 512×15 | worse | cos_sim=0.999 (self-regulated to identity) |

### Conclusion

**RYS is a scaling phenomenon.** At ~27M params / 11-24 layers:
- Layers are too specialized for their pipeline position to benefit from repetition
- The model consistently learns to minimize the repeat's contribution (gate→0, or cos_sim→1)
- Maximum improvement is -0.0005 to -0.0009 BPB — real but negligible
- The 72B model succeeded because it had functional redundancy in middle layers that doesn't exist at this scale

**Positive findings:**
- SRYS successfully teaches the contraction property (cos_sim 0.66→0.999)
- Gated SRYS acts as a useful training regularizer (-0.0028 base BPB) independent of eval-time RYS
- More depth genuinely improves RYS stability (352×24 plateaus at ×2-4 vs ×3 degradation at 11L)
- Attn-only blocks show dramatically different repeat dynamics (cos_sim=0.35) — the architecture matters

---

## Theoretical Notes

### Why RYS works at 72B but not 27M

The blog's Qwen2-72B has 80 layers. Middle layers (45-51) form a "reasoning circuit" with high mutual CKA similarity — they're functionally redundant, doing overlapping work. Repeating the circuit gives a second refinement pass on complex reasoning.

At 27M/11L, our CKA analysis shows:
- **Encoder (L0-2):** High mutual similarity (0.83) — pattern matching, one pass sufficient
- **Transition (L3-6):** Restructuring representations — repeating hurts
- **Prediction (L7-10):** Tightest cluster (0.88) — but only 4 layers, each maximally specialized

The critical difference: at 72B each layer does ~1/80th of the total work, leaving room for iterative refinement. At 27M each layer does ~1/11th — there's no slack for a second pass to exploit.

### The identity-or-reject dichotomy

Across all experiments, the model exhibits exactly two behaviors:
1. **Without gate:** Forces cos_sim→0.999 (repeat becomes identity, no effect)
2. **With gate:** Lets cos_sim drop freely but closes gate→0.03-0.10 (suppresses the delta)

Both strategies achieve the same goal: minimize the repeat's influence on the final output. The model has no use for the extra computation regardless of how it's offered.

### What would make RYS work at small scale?

Based on our findings, RYS at small scale would require:
1. **More total parameters** — functional redundancy needs parameter slack
2. **Fundamentally different architecture** — e.g., mixture-of-experts where some experts are naturally redundant
3. **Different training objective** — a loss that explicitly rewards multi-pass refinement (but this moves away from pure RYS toward recurrence)
