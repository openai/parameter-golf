# Stage 2 v2 Hypotheses — SOTA Stack Pivot

## Strategic Pivot

The seq4096 trunk is a dead end for catching the record (1.2014 vs SOTA 1.1574).
The SOTA's MLP 3x + int6 + sliding eval is fundamentally better.
All v2 hypotheses are built on the SOTA stack.

## Gap Decomposition (SOTA vs our trunk)

| Source | Estimated BPB contribution |
|--------|---------------------------|
| MLP 3x (1536 vs 1024 hidden), enabled by int6 | ~0.015-0.020 |
| int6 quantization (more params in 16MB) | baked into MLP 3x |
| fp16 tied-embedding export | ~0.002-0.005 |
| Late-K fp16 passthrough (last 2 layers) | ~0.001-0.003 |
| Sliding window eval stride=256 | ~0.015-0.020 |
| **Total gap** | **~0.044** |

## Screen v1 Results (all on seq4096 trunk — all lost)

| Slot | Mutation | post_quant BPB | delta vs B0 |
|------|----------|----------------|-------------|
| S2-B0 | baseline | 2.4024 | — |
| S2-E3 | quant clip+f32 scale | 2.4205 | +0.018 |
| S2-E2 | fp16 embed | 2.4530 | +0.051 |
| S2-E4 | fp16 embed + Muon WD | 2.4595 | +0.057 |
| S2-E6 | adaptive Muon | 2.6572 | +0.255 |

Conclusion: stop mutating the seq4096 trunk. Adopt the SOTA stack.

## Hypothesis Generation Process

### Initial 20 (pruned to 10)
Dropped: seq4096+MLP3x (memory), MLP 4x (size), alternate layer sharing (noise), logit_softcap/qk_gain/rope_base (minor knobs), gradient clipping (no signal), overtone init (needs code patch), 3/4 batch variant.

### 5 Gap-Filling Additions
- Training schedule interaction (warmdown + momentum combined)
- Aggressive small-batch test
- Muon Newton-Schulz steps=7
- Embed LR sweep
- Late-K passthrough layers=3

### Pruned to 8, then ranked to 6 + control

## Final 7 Slots (parallel 1-GPU screen, 150s each)

### Slot 0: R0 — SOTA Stack Control

**Env vars:** Exact SOTA replay (seq2048, batch 786K, MLP 1536, int6, fp16 embed, late-K 2, sliding eval stride=256)

**Why:** Every other hypothesis is measured relative to this. If this doesn't land near the SOTA proportionally, we have a code/data bug.

**Impact:** 0.000 (baseline)

---

### Slot 1: H5 — Smaller Batch, More Steps

**Delta from R0:** `TRAIN_BATCH_TOKENS=524288`

**Causal story:** SOTA uses 786K batch → ~7199 steps in 10 min. At 524K batch → ~10,800 steps. More optimizer updates = better in the Chinchilla regime. The SOTA may have over-batched. Each step at 524K still does 256 sequences (well above noise floor). This is the single biggest lever because it directly increases training updates by ~50%.

**Expected impact:** -0.010 to -0.020 BPB

**Validates:** More steps > bigger batch. **Kills:** Step time increase eats the gain, or noisy gradients destabilize.

---

### Slot 2: H1 — Higher Muon Momentum

**Delta from R0:** `MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500`

**Causal story:** Our trunk's momentum=0.99 was a key win (the TrainingOptSeq4096 recipe). SOTA uses default 0.95. Higher momentum smooths gradients, helps Muon's Newton-Schulz converge to better directions. Especially valuable with large batch where gradient noise is already low — momentum extends the effective gradient window.

**Expected impact:** -0.005 to -0.015 BPB

**Validates:** Momentum tuning transfers to the SOTA stack. **Kills:** SOTA's large batch already provides sufficient smoothing.

---

### Slot 3: H2 — Longer Warmdown

**Delta from R0:** `WARMDOWN_ITERS=3000`

**Causal story:** SOTA uses warmdown=1200. With ~7200 steps, warmdown covers last ~17%. At 3000, warmdown covers last ~42% — aggressive but finds flatter minima that are more robust to int6 quantization. Our trunk uses 3000 and has low quant gap. Since int6 has ~4x more quant noise than int8, longer warmdown is even more valuable here.

**Expected impact:** -0.003 to -0.008 BPB (primarily from reduced quant gap)

**Validates:** Longer warmdown improves int6 robustness. **Kills:** Pre-quant quality drops more than quant gap shrinks.

---

### Slot 4: H4 — Muon Weight Decay

**Delta from R0:** `MUON_WEIGHT_DECAY=0.02`

**Causal story:** From the 10L record. WD compresses weight norms → lower relative quantization error (smaller weights = smaller quant step). With int6's 31 quantization levels (vs int8's 127), any norm reduction is amplified. Also provides regularization against overfitting in the 10-min regime.

**Risk note:** S2-E4 tested this on our trunk and it was -0.057 worse. But that was a different stack (seq4096, int8, MLP 2x). The SOTA stack (int6, MLP 3x) may respond differently.

**Expected impact:** -0.002 to -0.005 BPB (or regression)

**Validates:** Muon WD + int6 is a good combination. **Kills:** Capacity loss outweighs quant benefit again.

---

### Slot 5: H6 — Denser Sliding Eval

**Delta from R0:** `EVAL_STRIDE=128`

**Causal story:** SOTA uses stride=256 (min context: 1792 tokens). Stride=128 gives min context 1920 tokens — 7% more context per position. Pure eval improvement, zero training cost. The 0.0215 BPB gain from sliding eval (SOTA plain→sliding) suggests diminishing returns may still have room.

**Expected impact:** -0.002 to -0.005 BPB

**Validates:** Denser sliding is still on the Pareto frontier. **Kills:** Diminishing returns, or eval time exceeds limit.

---

### Slot 6: H_combined — Batch + Momentum + Warmdown

**Delta from R0:** `TRAIN_BATCH_TOKENS=524288, MUON_MOMENTUM=0.99, MUON_MOMENTUM_WARMUP_START=0.92, MUON_MOMENTUM_WARMUP_STEPS=1500, WARMDOWN_ITERS=3000`

**Causal story:** Combines the three most mechanistically grounded mutations. They address orthogonal axes: batch → step count, momentum → per-step quality, warmdown → end-of-training convergence. In TrainingOptSeq4096, all three were combined and worked. Tests whether our trunk's optimizer tuning is a portable improvement over the SOTA's defaults.

**Expected impact:** -0.010 to -0.025 BPB if additive; regression risk if they interact.

**Validates:** Our optimizer tuning stacks on the SOTA architecture. **Kills:** Combined mutations destabilize, or SOTA defaults were already well-calibrated.

---

## Promotion Logic

- If any single-factor hypothesis beats R0: promote to full 8-GPU 10-min run
- If H_combined beats R0 AND beats the best single-factor: the combination is real
- If H_combined beats R0 but loses to a single-factor: the other factors hurt, decompose
- Top 2 survivors get full decision runs, then best composite
