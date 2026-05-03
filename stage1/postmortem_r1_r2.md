# Stage 1 Postmortem: R1 and R2 Family Runs

**Date:** 2026-03-19
**Hardware:** 1xA100-80GB proxy (10 min per family, ~812 steps)
**Target:** OpenAI Parameter Golf — best LM in 16MB artifact, 10 min on 8xH100

---

## Summary

Two family runs explored 15 mutations against their respective baselines. R1 tested original hypotheses; R2 tested public-evidence-updated mutations after a repo sync. Combined, we found 5 mutations that beat baseline and 3 that clearly hurt. The strongest single-slot signal is **seq2048 training** (R2 P1, -0.0078 vs baseline), followed by **INT8 clip percentile tuning** (R1 P7, -0.0157 vs its baseline).

---

## 1. What Worked (Winners)

### Tier A: Strong signal, high confidence

| Mutation | Family | Delta vs P0 | Notes |
|---|---|---|---|
| **quant_quality** (INT8_CLIP_PERCENTILE=99.99995, per-row float32 scale) | R1 P7 | **-0.0157** | Largest single-slot gain across both runs. Better quantization fidelity is cheap and clearly helps. |
| **seq2048 training** (TRAIN_SEQ_LEN=2048, lower LRs) | R2 P1 | **-0.0078** | Longer context during training pays off even at 812 steps. LR adjustment was necessary to stabilize. |
| **adaptive_muon** (backend=1, early=3, full=5, warmup=2000) | R1 P5 | **-0.0119** | Adaptive Muon schedule beat R1 baseline convincingly. However, the same config regressed in R2 (+0.0090), suggesting sensitivity to other hyperparams or repo state. |

### Tier B: Modest signal, worth stacking

| Mutation | Family | Delta vs P0 | Notes |
|---|---|---|---|
| **fp16 embed export** (INT8_KEEP_TOK_EMB_FP16=1, MLP_HIDDEN=992) | R2 P2 | **-0.0040** | Keeping token embeddings in fp16 preserves quality. MLP width bump uses freed param budget well. |
| **sliding eval stride=64** (EVAL_STRIDE=64) | R2 P4 | **-0.0033** | Pure eval-time improvement — no training cost. Should be stacked unconditionally. |
| **quant_bytes** (INT8_KEEP_FLOAT_MAX_NUMEL=8192) | R1 P6 | **-0.0025** | Small but consistent. Keeping small tensors in float avoids quantization noise on critical params. |

---

## 2. What Failed (Losers)

### Clear regressions

| Mutation | Family | Delta vs P0 | Analysis |
|---|---|---|---|
| **always-decay** (WARMDOWN_ITERS=20000, high LRs, GRAD_CLIP_NORM=1.0) | R2 P5 | **+0.3307** | Catastrophic. The aggressive LR + long warmdown schedule diverged badly in a short run. This config might only work at much longer horizons, but the magnitude of failure suggests fundamental miscalibration. |
| **kv2_width544** (NUM_LAYERS=9, MODEL_DIM=544, NUM_KV_HEADS=2) | R1 P2 | **+0.0254** | Trading depth for width with fewer KV heads did not pay off within the 16MB budget. The wider-shallower model underperformed. |
| **depth10_width480** (NUM_LAYERS=10, MODEL_DIM=480) | R1 P1 / R2 P6 | **+0.0239 / +0.0209** | Consistent loser in both families. 10 layers at 480 width is too narrow — the per-layer capacity is insufficient. Note: the public SOTA uses 10 layers but presumably with a different width and additional techniques (spectral init, Muon WD) that compensate. |
| **seq512** (TRAIN_SEQ_LEN=512) | R1 P4 | **+0.0034** | Shorter context slightly hurts, consistent with seq2048 winning in R2. |

### Ambiguous

| Mutation | Family | Delta vs P0 | Analysis |
|---|---|---|---|
| **alt_share** (ALTERNATE_LAYER_SHARING=1) | R1 P3 | **-0.0009** | Noise-level improvement. Layer sharing saves params but the freed budget was not reallocated, so no meaningful gain. |
| **AdamW partition** (TOKEN_OPTIMIZER=adamw, SCALAR_OPTIMIZER=adamw) | R2 P3 | **+0.0005** | Essentially flat. Replacing Muon with AdamW on token/scalar params neither helps nor hurts at this scale. |

---

## 3. Cross-Family Signals

### What looks stackable (orthogonal axes)

These mutations operate on independent axes and should compose well:

1. **quant_quality** (quantization fidelity) — pure artifact quality, no training interaction
2. **seq2048 training** (training sequence length) — training-time only
3. **sliding eval stride=64** (eval protocol) — eval-time only, zero training cost
4. **fp16 embed export** (param budget reallocation) — architecture/quantization hybrid

A stack of all four would address: quantization quality + training signal + eval protocol + param budget. No obvious interference.

### What conflicts

- **adaptive_muon**: Won in R1 (-0.0119) but lost in R2 (+0.0090). The R2 baseline included repo updates that may have changed Muon defaults. This mutation is **environment-sensitive** and needs careful re-testing in the exact 8xH100 environment before promotion.
- **depth10_width480**: Lost in both families. The public SOTA's 10-layer config works because of complementary techniques (spectral init, Muon WD, sliding window) that we did not stack with it. Depth alone is not the answer — it needs the full recipe.

### Baseline drift

R2 baseline (1.37392) was ~0.012 bpb better than R1 baseline (1.38551). This came from repo updates between runs. All delta comparisons are within-family only. Cross-family absolute values are not directly comparable.

---

## 4. Recommendations for 8xH100 Promotion

### Immediate stack (high confidence, promote now)

Combine the following into a single 8xH100 candidate:

1. **INT8_CLIP_PERCENTILE=99.99995** + **INT8_PER_ROW_SCALE_DTYPE=float32** (quant fidelity)
2. **TRAIN_SEQ_LEN=2048** with appropriately lowered LRs
3. **EVAL_STRIDE=64** (sliding window eval)
4. **INT8_KEEP_TOK_EMB_FP16=1** + **MLP_HIDDEN=992** (fp16 embed + budget reuse)
5. **INT8_KEEP_FLOAT_MAX_NUMEL=8192** (keep small tensors float)

Expected combined delta: roughly -0.02 to -0.03 bpb vs naive baseline (additive assumption with some diminishing returns).

### Conditional promotion (needs re-testing)

- **adaptive_muon**: Re-run on 8xH100 with the exact stacked config above. The R1/R2 disagreement must be resolved before including it in a submission.

### Do not promote

- **always-decay**: Catastrophic failure, do not revisit without fundamental redesign.
- **depth10_width480** (as tested): Only revisit if paired with spectral init + Muon WD to match the public SOTA recipe.
- **kv2_width544**: Strictly dominated.
- **seq512**: Strictly dominated by seq2048.

---

## 5. Open Questions

1. **Scaling behavior 812 steps vs 13,780 steps.** Our proxy runs are ~17x shorter than real 8xH100 runs. Mutations that help early (quant, seq length) may saturate; mutations that need longer to show signal (architecture, optimizer schedule) may be systematically undervalued. The always-decay catastrophe might even reverse at full length — but the magnitude makes this unlikely.

2. **Adaptive Muon instability.** Why did the same config win in R1 and lose in R2? Possible causes: (a) repo-level changes to Muon defaults between runs, (b) interaction with R2's different baseline hyperparams, (c) random seed sensitivity. A controlled A/B on identical code is needed.

3. **The public SOTA gap.** Our best single mutation (R1 P7, val_bpb=1.36981 on 1xA100) is still far from leaderboard SOTA (1.1574 on 8xH100). The gap is partly hardware (1x vs 8x = ~6x more steps) and partly technique (spectral init, Muon WD, int6 quantization, MLP 3x width — none of which we tested). The stacking question is whether our winners compose with those public techniques.

4. **INT6 quantization.** The public SOTA uses int6, which we have not explored. This is a major param-budget unlock (33% more effective parameters) and likely the single largest lever we are leaving on the table.

5. **Spectral / overtone init.** The second-place public entry uses "overtone init." We tested neither spectral nor overtone initialization. These are orthogonal to our current winners and could provide additional gains.

6. **MLP 3x width.** The SOTA uses 3x MLP hidden dim. Our R2 P2 only went to 992. There may be significant headroom in pushing MLP width further, especially if int6 frees param budget.

---

## Appendix: Raw Results

### R1 (h100_family_r1) — 812 steps, 739ms/step, 1xA100-80GB

| Rank | Slot | Mutation | val_bpb | vs P0 |
|---|---|---|---|---|
| 1 | P7 | quant_quality | 1.36981 | -0.0157 |
| 2 | P5 | adaptive_muon | 1.37363 | -0.0119 |
| 3 | P6 | quant_bytes | 1.38299 | -0.0025 |
| 4 | P3 | alt_share | 1.38462 | -0.0009 |
| 5 | P0 | baseline | 1.38551 | — |
| 6 | P4 | seq512 | 1.38892 | +0.0034 |
| 7 | P1 | depth10_width480 | 1.40940 | +0.0239 |
| 8 | P2 | kv2_width544 | 1.41094 | +0.0254 |

### R2 (frontier_family_r2) — 812 steps, 1xA100-80GB

| Rank | Slot | Mutation | val_bpb | vs P0 |
|---|---|---|---|---|
| 1 | P1 | seq2048 training | 1.36611 | -0.0078 |
| 2 | P2 | fp16 embed export | 1.36993 | -0.0040 |
| 3 | P4 | sliding eval stride=64 | 1.37064 | -0.0033 |
| 4 | P0 | baseline | 1.37392 | — |
| 5 | P3 | AdamW partition | 1.37447 | +0.0005 |
| 6 | P7 | adaptive Muon | 1.38296 | +0.0090 |
| 7 | P6 | depth10 width480 | 1.39479 | +0.0209 |
| 8 | P5 | always-decay | 1.70464 | +0.3307 |
