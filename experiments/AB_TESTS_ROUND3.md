# A/B Tests Round 3 — Post-1.1130 Optimization

> **IMPORTANT:** Before implementing anything from this document, use Nia to validate the claims, read the actual papers, and verify the techniques apply to our specific setup (14L, 512d, GQA 8/4, int6 GPTQ, 16MB budget, 10min train + 10min eval). Many techniques that sound good in papers fail at our scale or constraints. Cross-reference with the sources listed below. Also use Nia to search for NEW techniques not listed here that could be helpful — this document is a snapshot, not exhaustive. The competition and research landscape evolve daily.
>
> **MAINTENANCE:** This document MUST be updated:
>
> - **Before** each A/B test: add the experiment with status, hypothesis, expected gain
> - **After** each A/B test: record the result, update the completed results table
> - **When new research is found**: add it to the appropriate section with sources
> - **When priorities change**: reorder the priority list based on new evidence
> Keep this as the single source of truth for all optimization work.

## Current Best

**1.1130 val_bpb** | stride=76 | per-window SGD TTT | 14L | GPTQ int6 | EMA 0.997 | 575s eval | 15.87MB

## Best Clean PRs for Reference (UPDATED March 24)

**Our tier (architecture + standard TTT):**


| PR     | BPB        | Key technique                                  |
| ------ | ---------- | ---------------------------------------------- |
| **Us** | **1.1075** | **14L WD=0.05 QEP GPTQ + per-window SGD TTT**  |
| #595   | 1.1100     | SWA + BigramHash + AdamW TTT (10 epochs) — NEW |
| #609   | 1.1154     | Full GPTQ + XSA-all + NO TTT — NEW             |
| #593   | 1.1163     | Full GPTQ + LeakyReLU² — NEW no TTT            |
| #569   | 1.1175     | VRL + LeakyReLU² + Full GPTQ (no TTT)          |
| #638   | 1.1164     | 11L XSA-all + VR + GA + no TTT — NEW (single seed) |
| #633   | 1.1526     | 11L single-epoch LoRA TTT (legal) — NEW         |


**⚠️ LoRA TTT with score-every-epoch (0.5-0.8 BPP) RULED ILLEGAL. Multi-epoch cosine TTT (#672 1.0781, #518 1.0622) also ILLEGAL.**

**We are #1 in the legal 10-min leaderboard at 1.1127.** Next best: #609 at 1.1154 (no TTT, 3-seed). PR #525 at 1.1160 uses legal batched LoRA TTT (rank-8 Q/V/LM-head, Adam lr=0.01).

---

## WHERE WE ARE (March 24, end of day)

### Current Submission-Ready Best

**1.1075 BPB** — WD=0.05, QEP GPTQ, per-window SGD TTT @ stride=76, 551s eval

### Key Findings Today

1. **WD=0.05 is optimal** (U-shaped, confirmed 0.03-0.11 sweep). Was at 0.09, cost us 0.004 BPP.
2. **QEP GPTQ reduces quant gap** 0.015→0.012, but TTT absorbs most of the gain.
3. **Rescore@s64 after TTT** gives 1.1058 but exceeds 600s eval budget (642s).
4. **Freeze 8/14 layers during TTT** at stride=64: costs 0.001 BPP, saves ~214s → fits budget at ~440s.
5. **VRL neutral** at 14L. **Full QAT terrible**. **Late QAT worse** (dynamo reset disrupts training).
6. **EMA=0.997 confirmed optimal.** Warmdown 3500 confirmed optimal.
7. **LoRA TTT paradigm shift** in competition — sub-0.8 BPP but legality unclear.

### What to Do Next (Priority Order)

1. **Redo freeze sweep on WD=0.05 + QEP model** — exp205 showed freeze=8 doesn't transfer from WD=0.09
  - Need to: run exp203 config with the 201 script (which saves `final_model_pre_ttt.pt`), then eval-only freeze sweep on that saved model
  - Test freeze=4/6 at stride=64 — freeze=8 was too aggressive for WD=0.05 but freeze=4 might work
  - Goal: find the right freeze level to enable stride=64 under 600s

1. ~~**Switch SGD → AdamW + cosine LR for TTT**~~ — **TESTED, DOESN'T WORK for per-window TTT**. AdamW diverges at all LRs (0.0001→+0.015, 0.0005→+0.097, 0.001→+0.178). AdamW needs multiple steps to build moments; per-window gives only 1 step. PR #606 uses chunked TTT with 3 epochs — different paradigm.
2. **LQER-style low-rank residual correction** — SVD of GPTQ residuals, rank 4-8, ~900KB overhead. Published ICML 2024. Directly attacks our 0.012 quant gap. Could recover 0.003-0.005 BPP.
3. **Unfreeze norms + lm_head during TTT** — PR #606 unfreezes "last 2 blocks + norms + lm_head". We may only be unfreezing layer weights. Easy to test.
4. **Multi-seed verification** — run best config with seeds 42, 7 for submission
5. ~~**LoRA TTT**~~ — PR #610 confirmed LoRA doesn't work at 512d ("model too small for rank-16, 3% subspace"). Skip.
6. ~~**Per-layer LR for TTT**~~ — PR #537 tested, minimal gain. Skip.

### Freeze Sweep Results (WD=0.09 model, stride=64, eval-only)


| Freeze | BPB        | Delta       | Est time  |
| ------ | ---------- | ----------- | --------- |
| 2      | 1.1129     | —           | ~654s     |
| 4      | 1.1131     | +0.0002     | ~580s     |
| 6      | 1.1133     | +0.0004     | ~510s     |
| **8**  | **1.1139** | **+0.0010** | **~440s** |
| 10     | 1.1145     | +0.0016     | ~380s     |


**Freeze=8 is sweet spot**: 0.001 cost, ~214s savings. With WD=0.05 → ~1.1069 est in ~440s.

### Freeze Sweep Results (WD=0.05+QEP model, stride=64, eval-only, EATA on) — RUNNING

| Freeze | BPB | Delta vs f=0 | Eval time (EATA) |
|--------|------|-------------|-----------------|
| **0** | **1.1081** | **—** | **415s** |
| 4 | 1.1082 | +0.0001 | 401s |
| 6 | 1.1083 | +0.0002 | 397s |
| 8 | 1.1085 | +0.0004 | 391s |

**Conclusion:** Freeze cost is very small on WD=0.05 model (much less than WD=0.09). But none beat stride=76 results (1.1075-1.1078). EATA skips ~50% of adapt steps.

## ⚠️ CRITICAL BLOCKER: Artifact Size is 18.3MB (over 16MB limit!)

**Discovered March 24:** WD=0.05 artifact is 18.3MB — over 16MB. Entropy analysis shows brotli is already within 3.5% of theoretical minimum (4.30 vs 4.45 bits/value). **No compression trick can fix this — the entropy is fundamentally too high.** WD=0.09 is the only submittable WD (15.76MB). Bit-packing tested and confirmed useless (destroys byte patterns brotli relies on).

**Current submittable best: WD=0.09+QEP → 1.1127 BPP, 15.76MB, 551s.**

### Artifact Size Tests

| Config | Artifact | BPP (roundtrip) | BPP (TTT@s76) | Notes |
|--------|----------|-----------------|---------------|-------|
| WD=0.05, int6, QEP (exp206) | **18.3MB ❌** | 1.1332 | 1.1078 | OVER 16MB |
| WD=0.09, int6, no QEP (exp199) | **15.8MB ✓** | ~1.142 | 1.1130 | Old baseline, fits |
| WD=0.05, **INT5_MLP**, QEP (exp207) | **15.5MB ✓** | 1.1530 | **1.1213** | Fits but +0.0135 BPP — too costly |
| WD=0.05, int6, **prune=2%**, QEP | **18.3MB ❌** | 1.1337 | 1.1080 | Pruning saved ~0KB — useless at 2% |
| WD=0.05, int6, **no QEP** | **18.3MB ❌** | 1.1334 | 1.1078 | QEP is NOT the cause — same size without QEP |
| WD=0.07, int6, QEP | **16.9MB ❌** | 1.1369 | **1.1103** | 0.9MB over — need WD=0.08+ |
| WD=0.08, int6, QEP | **16.3MB ❌** | 1.1396 | **1.1113** | 294KB over even with 0 code bytes |
| WD=0.085, int6, QEP | **16.03MB ❌** | 1.1409 | running | Over by 25KB! So close. Need WD=0.087+ or RDO |
| **WD=0.09, int6, QEP** | **15.76MB ✓** | 1.1415 | **1.1127** | **FITS! First submittable result. QEP gained 0.0003 vs old 1.1130** |
| LQER rank=2 (WD=0.09) | 16.14MB ❌ | 1.1411 | — | Only +0.0004 roundtrip, over budget. Dead end. |
| **RDO lambda=0.02 (WD=0.05)** | **15.6MB ✓** | **1.2005** | running | **COMPRESSION WORKS! (-2.7MB) but +0.067 BPP — lambda too high** |
| RDO fine sweep (WD=0.05) | KILLED | — | — | Math shows RDO tradeoff is too steep — accuracy cost > WD benefit. Dead end. |
| Cosine SGD flat lr=0.002 | 1.1121 | 565s | baseline | Eval-only on pre-TTT model (no compression roundtrip) |
| Cosine SGD cosine lr=0.002 | 1.1122 | 563s | +0.0001 | **NEUTRAL** — cosine decay doesn't help at same LR |
| **Cosine SGD cosine lr=0.004** | **1.1118** | 564s | **-0.0003** | **Tiny gain from higher peak LR + cosine decay** |
| L1 Proximal λ=1e-5 (WD=0.05) | 18.04MB ❌ | 1.1351 | — | Only 260KB saved, +0.002 BPP. Same tradeoff wall as RDO. Dead end. |
| Recursive 7-unique (WD=0.05) | **9.69MB** | 1.1926 | **1.1628** | Compression great (-47%) but +0.05 BPP — too much capacity loss |
| WD warmup 0.05→0.09 (exp212) | **16.78MB ❌** | 1.1418 | **1.1119** | BPP better than WD=0.09 (+0.0008!) but 780KB over. Ramp not aggressive enough. |
| Recursive 7ub dim=640 (WD=0.05) | **13.98MB ✓** | 1.1664 | **1.1387** | Fits but 1.1387 (+0.026) and 703s eval (over budget). dim=640 too slow. |
| **WD warmup 0.05→0.12 (v2)** | **15.73MB ✓** | 1.1442 | **1.1158** | Fits but BPP worse than constant 0.09. WD=0.12 too aggressive. |
| WD warmup 0.05→0.10 (v3) | 16.40MB ❌ | 1.1404 | 1.1130 | 400KB over. BPP same as constant 0.09. **WD warmup dead end — crossover doesn't exist.** |

### WD Sweep Results (rescore@s64, full training runs)


| WD       | BPB           |
| -------- | ------------- |
| 0.03     | 1.1063        |
| 0.04     | 1.1060        |
| 0.045    | 1.1064        |
| **0.05** | **1.1058 🏆** |
| 0.06     | 1.1065        |
| 0.065    | 1.1072        |
| 0.07     | 1.1075        |
| 0.075    | 1.1080        |
| 0.09     | 1.1099        |
| 0.11     | 1.1124        |


### A/B Round 3 Results (full training, rescore@s64)


| Exp                   | Config              | TTT@s76 | Rescore@s64 | Delta vs baseline |
| --------------------- | ------------------- | ------- | ----------- | ----------------- |
| T2 baseline (WD=0.09) | —                   | 1.1130  | 1.1099      | —                 |
| **T5 WD=0.07**        | MUON_WD=0.07        | 1.1098  | **1.1075**  | **-0.0024**       |
| T6 WD=0.11            | MUON_WD=0.11        | 1.1165  | 1.1124      | +0.0025           |
| T8 BigDim=96          | BIGRAM_DIM=96       | 1.1128  | 1.1097      | -0.0002           |
| T9 EMA=0.995          | EMA_DECAY=0.995     | —       | 1.1104      | +0.0005           |
| T10 EMA=0.999         | EMA_DECAY=0.999     | —       | 1.1188      | +0.0089           |
| T11 warmdown=4000     | WARMDOWN_ITERS=4000 | —       | 1.1096      | -0.0003           |
| T12 warmdown=3000     | WARMDOWN_ITERS=3000 | —       | 1.1099      | 0.0000            |
| T3 Full QAT           | QAT_ENABLED=1       | 1.1269  | 1.1249      | +0.0150           |
| T4 Full QAT 0.5       | QAT_ENABLED=1       | 1.1265  | 1.1244      | +0.0145           |


### Other Experiments


| Exp               | Config                           | Result                 | Notes                                         |
| ----------------- | -------------------------------- | ---------------------- | --------------------------------------------- |
| VRL v3            | VRL_ENABLED=1, WD=0.09           | 1.1133 s76             | Neutral — U-Net skips sufficient at 14L       |
| Late QAT 0.15     | LATE_QAT_THRESHOLD=0.15, WD=0.05 | 1.1109 s64             | +0.005 WORSE — dynamo reset disrupts training |
| QEP GPTQ          | QEP_ENABLED=1, WD=0.05           | 1.1075 s76, 1.1060 s64 | Quant gap 0.015→0.012, TTT absorbs most       |
| Chunked AdamW 1ep | TTT_MODE=chunked, WD=0.09        | 1.1175                 | Fast (138s) but worse BPP                     |
| T=0.98            | TTT_TEMPERATURE=0.98, WD=0.09    | 1.1132 s76             | Worse — don't use                             |


### Stride Sweep Results (per-window SGD TTT, WD=0.09)


| Stride | BPB        | Time        |
| ------ | ---------- | ----------- |
| 64     | 1.1126     | 654s (over) |
| 68     | 1.1129     | 640s (over) |
| 72     | 1.1129     | 604s (over) |
| **76** | **1.1130** | **575s ✓**  |
| 80     | 1.1130     | 547s ✓      |
| 96     | 1.1131     | 458s ✓      |
| 128    | 1.1133     | 347s ✓      |


---

## exp205 RESULT: Combined config WORSE than exp203

- **1.1082 BPB at stride=64, 623s eval — OVER BUDGET and worse BPP**
- Freeze=8 hurts more on WD=0.05 model than on WD=0.09 model
- The freeze sweep was done on WD=0.09 — doesn't transfer to WD=0.05
- **exp203 remains best: 1.1075 BPB, stride=76, 551s, under budget**

## exp206 RESULT: Pre-TTT model saved, confirms exp203

- **1.1078 BPB at stride=76, 551s eval** — matches exp203 (1.1075) within noise
- Pre-quant roundtrip: 1.1332 BPB (quant gap = 0.012 after QEP GPTQ, then TTT recovers most)
- `final_model_pre_ttt.pt` saved (134.5MB) for freeze sweep
- **Freeze sweep RUNNING** on this model: freeze=0/4/6/8 at stride=64

---

## Experiments Queue

### 1. Value Residual Learning (VRL)

**Status: RUNNING (exp202_vrl)**

- Layer 0's V output blended into all subsequent layers via per-layer sigmoid gates
- Source: arxiv:2410.17897, PR #569 (1.1175 without TTT)
- Expected: -0.001 to -0.003 BPB
- Control: exp202_novrl (identical config, VRL_ENABLED=0)
- Extra params: 13 scalars (negligible)

### 2. QAT-Export Alignment

**Status: READY TO IMPLEMENT**

- Problem: STE fake-quant during training may use different clipping than GPTQ at export
- Fix: Match STE clip percentile to GPTQ's percentile (0.9995 from PR #569)
- Source: PR #569 explicitly aligns these
- Expected: -0.001 to -0.002 BPB (reduces quant gap)
- Risk: Low — one constant change

### 3. Soft-Round QAT

**Status: NEEDS IMPLEMENTATION**

- Replace hard `round()` in STE with temperature-controlled smooth approximation
- Source: PR #589 "Late Soft-Round QAT"
- Gives optimizer gradient signal near quantization bin boundaries
- Expected: -0.001 to -0.002 BPB
- Risk: Medium — new forward pass math, could destabilize training if temp schedule is wrong
- Implementation: `soft_round(x, temp) = x + (1/pi) * arctan(sin(2*pi*x) / temp)` or similar

### 4. Int5 GPTQ (fit bigger model)

**Status: NEEDS INVESTIGATION**

- Quantize to int5 (clip_range=15) instead of int6 (clip_range=31)
- Saves ~17% per weight → could fit 15th layer or wider MLP
- Source: PR #545 (33.6M params in 15.5MB with int5)
- Expected: depends on what we do with the space savings
- Risk: High — int5 quality loss might outweigh capacity gain at 14L
- A/B: int5 14L vs int6 14L (same architecture, just quant precision)

### 5. Backward-Looking Chunk TTT

**Status: IMPLEMENTED (exp201 chunked), already tested**

- Score chunk N, train on chunks 0..N-1 (not on N itself)
- Result: 1.1175 BPB in 138s (faster but 0.005 worse than per-window)
- Verdict: Per-window SGD is better for us. Could revisit with AdamW + more epochs.

### 6. EVAL_STRIDE=32

**Status: NOT VIABLE (time budget)**

- PR #545 uses stride=32 for finer overlap
- Our stride=76 at 575s leaves no room for stride=32 (~1300s est)
- Only viable if combined with chunked TTT (~138s base) + stride=32 scoring
- A/B: chunked TTT stride=32 vs per-window SGD stride=76

### 7. MHA 8/8 (drop GQA)

**Status: NEEDS TESTING**

- PR #545 uses full multi-head attention (8/8) vs our GQA (8/4)
- More attention capacity but more params per layer
- At 14L this might push artifact over 16MB unless combined with int5
- A/B: 14L 8/8 MHA vs 14L 8/4 GQA

### 8. Early QAT (threshold 0.5)

**Status: NEEDS TESTING**

- PR #545 starts QAT at 50% through warmdown (threshold=0.5)
- Our current: no QAT during training
- More QAT steps = model better adapted to quantization noise
- Source: PR #414 uses 0.15, PR #545 uses 0.5
- A/B: QAT threshold 0.5 vs 0.15 vs 0 (current)

### 9. Magnitude Pruning Post-GPTQ

**Status: NEEDS TESTING**

- PR #569 uses 2% magnitude pruning AFTER quantization for compression
- We disabled pruning (PRUNE_FRAC=0)
- Could help compression → smaller artifact → room for larger model
- A/B: PRUNE_FRAC=0.02 vs 0 (current)

### 10. Temperature Search (post-TTT)

**Status: TESTED — T=0.98 HURT (+0.0002)**

- Verdict: Not useful for our SGD TTT setup

---

## Completed Results


| Experiment                  | BPB        | Time       | Delta vs baseline  | Notes                                                                    |
| --------------------------- | ---------- | ---------- | ------------------ | ------------------------------------------------------------------------ |
| stride=64 per-window SGD    | 1.1126     | 654s       | —                  | Over time budget                                                         |
| stride=68                   | 1.1129     | 640s       | +0.0003            | Over budget                                                              |
| stride=72                   | 1.1129     | 604s       | +0.0003            | 4s over                                                                  |
| **stride=76**               | **1.1130** | **575s**   | **+0.0004**        | **Submission candidate**                                                 |
| stride=80                   | 1.1130     | 547s       | +0.0004            | Safe margin                                                              |
| stride=96                   | 1.1131     | 458s       | +0.0005            | Very safe                                                                |
| stride=128                  | 1.1133     | 347s       | +0.0007            | Very safe                                                                |
| stride=76 T=0.98            | 1.1132     | 566s       | +0.0006            | Temp hurts                                                               |
| chunked AdamW 1ep           | 1.1175     | 138s       | +0.0049            | Fast but worse                                                           |
| VRL v3 (exp202)             | 1.1133     | 577s       | +0.0003            | **NEUTRAL** — U-Net skips already carry early info at 14L                |
| no-VRL control              | skipped    | —          | —                  | VRL neutral, no need for control                                         |
| QEP GPTQ (exp203)           | 1.1075 s76 | 551s       | -0.003 roundtrip   | Quant gap reduced 0.015→0.012! But TTT undoes most of the gain           |
| Late QAT 0.15 (exp204)      | —          | 1.1109 s64 | +0.0051 WORSE      | QAT hurt — recompilation disrupts training, or clipping too aggressive   |
| TTT freeze sweep (WD=0.09)  | see table  | ~50s ea    | stride=64 recovery | eval-only on WD=0.09 model — results DON'T transfer to WD=0.05          |
| exp205 (freeze=8+s64, WD=0.05) | 1.1082  | 623s       | +0.0007 vs exp203  | WORSE + over budget — freeze=8 from WD=0.09 doesn't transfer             |
| exp206 (retrain for pre-TTT) | 1.1078    | 551s       | +0.0003 vs exp203  | Confirms exp203. Saved final_model_pre_ttt.pt for freeze sweep           |
| Freeze sweep WD=0.05 f=0    | 1.1081     | 415s(EATA) | —                  | Freeze cost tiny on WD=0.05: f4=+0.0001, f6=+0.0002, f8=+0.0004        |
| exp207 INT5_MLP             | **1.1213** | 551s       | +0.0135            | ❌ 15.5MB fits but +0.0135 BPP — too costly. Int5 quant gap too large.   |
| Size sweep (prune/noQEP/WD) | done       | —          | —                  | Only WD=0.09 fits under 16MB. WD=0.05-0.08 all over.                    |
| **WD=0.09+QEP (submittable!)** | **1.1127** | **551s**   | —                  | **15.76MB ✓ First submittable result!**                                  |
| AdamW TTT sweep (WD=0.09)  | all worse  | ~347s      | +0.015 to +0.178   | AdamW diverges for per-window TTT (needs multi-step). SGD is correct.   |
| LQER rank=2 (WD=0.09)      | 16.14MB ❌  | 1.1411     | —                  | Only 0.0004 roundtrip gain, artifact over budget. Dead end.             |
| **TTT@s76 + rescore@s64**   | **1.1099** | **~642s**  | **-0.0031!**       | **MASSIVE FIND — TTT adapts weights, rescore at s64 for precise BPP**    |
| T3 Full QAT (QAT_ENABLED=1) | 1.1269     | ~575s      | +0.0139            | **TERRIBLE** — full QAT hurts badly, need proper LATE QAT with threshold |
| T4 Full QAT 0.5             | 1.1265     | —          | +0.0145            | TERRIBLE — full QAT hurts                                                |
| **T5 WD=0.07**              | **1.1098** | **1.1075** | **-0.0024**        | **🏆 NEW BEST! WD=0.09 was too aggressive for 14L near Dcrit**           |
| T6 WD=0.11                  | 1.1165     | 1.1124     | +0.0025            | Worse — confirms WD=0.09 was already near limit                          |
| T8 BigDim=96                | 1.1128     | 1.1097     | -0.0002            | Marginal — need to verify artifact fits 16MB                             |
| T9 EMA=0.995                | —          | 1.1104     | +0.0005            | Slightly worse — 0.997 is near-optimal                                   |
| T10 EMA=0.999               | —          | 1.1188     | +0.0089            | Much worse — too slow for 5700 steps                                     |
| T11 warmdown=4000           | —          | 1.1096     | -0.0003            | Marginal — not significant                                               |
| T12 warmdown=3000           | —          | 1.1099     | 0.0000             | Same as baseline — 3500 is fine                                          |
| **WD=0.05**                 | —          | **1.1058** | **-0.0041**        | **🏆🏆 CONFIRMED BEST — U-shaped curve, optimal at 0.05**                |
| WD=0.045                    | —          | 1.1064     | -0.0035            | Worse than 0.05                                                          |
| WD=0.04                     | —          | 1.1060     | -0.0039            | Close but 0.05 wins                                                      |
| WD=0.03                     | —          | 1.1063     | -0.0036            | Too low, under-regularized                                               |
| WD=0.06                     | —          | 1.1065     | -0.0034            |                                                                          |
| WD=0.065                    | —          | 1.1072     | -0.0027            |                                                                          |
| WD=0.075                    | —          | 1.1080     | -0.0019            |                                                                          |
| WD=0.08                     | —          | pending    | —                  |                                                                          |


---

## NEW from research (March 24)

### 16. ~~AdamW + Cosine LR for TTT~~ — TESTED, DOESN'T WORK

**Status: TESTED — ALL WORSE.** AdamW diverges for per-window TTT (1 step per window isn't enough to build moment estimates). SGD lr=0.002 is correct for our setup.
- AdamW lr=0.0001: +0.015 BPP worse
- AdamW lr=0.0005: +0.097 BPP worse
- AdamW lr=0.001: +0.178 BPP worse
- Source: PR #606, PR #601, PR #615

### 17. Unfreeze Norms + LM Head During TTT

**Status: EASY TO TEST**

- PR #606 unfreezes "last 2 blocks + norms + lm_head" — we may only unfreeze block weights
- LayerNorm params and tied embedding/lm_head contain distribution info that benefits from adaptation
- Zero overhead, eval-only testable
- Expected: 0.001-0.002 BPP

### 18. LoRA TTT — Needs Our Own Testing (lower priority)

**Status: NOT YET TESTED** — PR #610 claimed "model too small for rank-16" at dim=512, but their implementation may have been bad (wrong LR, wrong projections, wrong rank). Don't trust other PRs' conclusions. Our full-param SGD TTT already gets 0.026 BPP improvement, so LoRA would need to beat that — possibly via faster per-step → finer stride, or better optimization landscape. Test after higher-priority items (LQER, compression).

### 19. Compression Improvements ⭐⭐ HIGHEST PRIORITY — Unlocks WD=0.05 (1.1078)

**Status: READY TO IMPLEMENT**

Our #1 blocker: WD=0.05 gives 1.1078 BPP but 18.3MB artifact. Need to save 2.5MB. These techniques are additive:

**A. ~~Bit-Packing int6 → 6 bits~~ — TESTED, DOESN'T HELP**
- Bit-packing saves 25% raw but brotli can't compress the packed byte patterns
- manual+brotli with bitpacking: 21MB (WORSE). torch+brotli without: 18.3MB (same as before)
- Brotli already exploits byte-level redundancy in int8 values. Bit-packing destroys those patterns.
- **The problem is ENTROPY of the quantized distribution, not bit-level waste.**

**A2. Rate-Aware GPTQ / AdaRound — Entropy-Optimal Quantization** ⭐ NEW
- Modify GPTQ to also minimize entropy of quantized values (not just reconstruction error)
- arxiv:2505.18758: adds quadratic rate penalty to GPTQ objective. 20-40% better compression at same accuracy.
- AdaRound (arxiv:2004.10568): optimally rounds up/down to reduce entropy while minimizing error
- Implementation: after GPTQ, second pass that nudges values toward more common bins when error cost is low
- **Expected: 15-30% compression improvement — could save 2-4MB**
- Risk: Medium — modifies quantization output, need to verify BPP is preserved

**B. Hadamard Rotation Before Quantization (QuIP#-style)**
- Multiply weights by random orthogonal (Hadamard) matrix before quantization
- Decorrelates weight components → more uniform distribution → less quant error AND better compression
- At inference: multiply input by inverse Hadamard (fast, O(n log n) via butterfly)
- Source: QuIP# (Cornell), KVLinC (arxiv:2510.05373), HALO (NeurIPS 2025)
- **Expected: 0.002-0.005 BPP improvement + 10-20% better compression**
- Risk: Medium — need fast Hadamard at inference, modifies forward pass
- Implementation: ~50 lines (Hadamard transform + modified quantize/dequantize)

**C. Entropy Coding (Huffman/ANS) After Quantization**
- Int6 has only 63 unique values. Custom Huffman tree is tiny (~126 bytes)
- More efficient than brotli for structured quantized data
- EntroLLM (arxiv:2505.02380): 7-11x better compression than raw with Huffman on quantized weights
- Can combine: bit-pack first, then Huffman, then brotli on top
- **Expected: 10-30% compression improvement**
- Risk: Low — no model changes

**D. Permutation-Based Weight Reordering (CVPR 2021)**
- Reorder rows/columns of weight matrices to cluster similar quantized values
- Functionally equivalent if you permute the corresponding layer too
- Better local redundancy → brotli compresses much better
- Source: "Permute, Quantize, and Fine-Tune" (CVPR 2021)
- **Expected: 5-15% compression improvement**
- Risk: Medium — need to find good permutations, modify inference to unpermute

**E. Quantization-Aware Regularizers (arxiv:2602.03614)** — NEW
- Drive weights to form clusters during TRAINING, making them quantization-friendly
- Quantization levels become learnable backprop parameters — first approach to embed quant params in training
- "Quantization-friendly" weights from the start → less quant error AND lower entropy
- Like QAT but as regularization, not fake quantization
- **Expected: better BPP + better compression simultaneously**
- Risk: Medium — modifies training, needs tuning of regularizer strength
- Source: arxiv:2602.03614

**F. Neural Weight Compression / NWC (arxiv:2510.11234)** — NEW
- Learned compression scheme optimized for neural network weights (not general-purpose like brotli)
- "SOTA accuracy-compression tradeoffs at 4-6 bit regime" without Hadamard transform
- Chunk-and-normalize preprocessing + importance-aware training objective
- Could outperform brotli on our quantized weights → unlock WD=0.05
- **Expected: potentially 10-30% better than brotli for quantized data**
- Risk: High — significant implementation, need to train the compressor
- Source: arxiv:2510.11234

**G. L1 Proximal Gradient During Training** ⭐ TOP PRIORITY — NEW
- NOT L1 loss penalty — use proximal operator after each step: `w ← sign(w) * max(|w| - α*λ, 0)`
- Creates EXACT zeros (not just small values) — fundamentally different from L2 weight decay
- Source: PROXGEN (NeurIPS 2021): adaptive proximal gradient methods for structured NNs
- 3-line change to training loop. Compatible with Muon optimizer.
- Unlike post-hoc 2% pruning (which failed because it removed important weights), L1 proximal lets the model LEARN which weights to zero out during training → organic sparsity with minimal accuracy cost
- If 10-20% of weights become zero → 1-3MB compression savings → could unlock WD=0.05
- **Expected: WD=0.05 quality (1.1078) with WD=0.09 compression (15.8MB)**
- Risk: Low — simple implementation, tune λ. Start with λ=1e-5, sweep up.
- **A/B test: WD=0.05 + L1_LAMBDA=1e-5/3e-5/1e-4 vs WD=0.05 baseline (18.3MB)**

**H. PROXQUANT — Align Weights to Quantization Grid During Training** — NEW
- Proximal operator that pushes weights toward nearest int6 codebook values during training
- Weights naturally cluster onto discrete levels → minimal quant error AND low entropy
- Source: ProxQuant (arxiv:1810.00861), PROXGEN (NeurIPS 2021)
- More complex than L1 proximal but directly attacks both accuracy and compression
- **Expected: better roundtrip BPP + lower entropy → better compression**
- Risk: Medium — need to define codebook proximal operator, tune strength schedule

**I. Relaxed Recursive Transformers — Weight Sharing Across Layers** — TESTED
- Use 7 unique layers × 2 repetitions = 14 effective layers, HALF the unique weights
- Source: arxiv:2410.20672 (ICLR 2025, Google DeepMind)
- **RESULT: 9.69MB artifact (great!) but 1.1628 BPP (terrible, +0.05 vs baseline)**
- 7×2 sharing loses too much capacity at our small scale (26.8M→13.4M params)
- The 102ms/step (faster) and 5843 steps (more) don't compensate for capacity loss
- Gentler sharing (10 or 12 unique) might work but diminishing returns on compression
- **Verdict: 7×2 at 512d not viable. BUT 9.69MB gives 6.3MB headroom — try wider (dim=640) or more unique blocks or lower WD**
- **Next test: 7 unique × 2 reps at dim=640, 10H/5KV, WD=0.05** — more capacity per layer, ~15MB estimated
- Could also try 10 unique × 2 = 20 effective layers, or WD=0.03 for even more headroom

**UPDATED Priority: G (L1 proximal) running now. If it fails, I (recursive transformer) is the biggest remaining opportunity — attacks compression at the architecture level. B (Hadamard) still worth trying. E/F are backup.**

**Key insight from entropy analysis:** Brotli is within 3.5% of theoretical entropy limit. The ONLY way to reduce artifact size is to reduce the entropy of the weight distribution itself — either during training (E, G) or during quantization (B Hadamard).

### 11. ~~Multi-Pass Score-First TTT~~ — ILLEGAL

**Status: RULED ILLEGAL** — min(NLL) across passes = selecting scores after seeing outcomes = same violation as score-every-epoch LoRA. PR #573 is invalid.

### 22. TrigramHash Embedding — TESTED, doesn't help at 14L

**Status: TESTED — 1.1146 BPP (+0.0019 worse). Artifact 15.78MB ✓ but no BPP gain.**

- Extends BigramHash (2-token context) to 3-token context via XOR hashing
- Formula: `xor(36313*t[i], 27191*t[i-1], 51497*t[i-2]) % (bucket_size - 1)`
- PR #486: -0.008 BPP gain in ablation (despite 22% fewer training steps due to overhead)
- We already have BigramHash(8192, 64) — TrigramHash adds a parallel 3-token embedding
- Size overhead: 8192×64 embedding = ~2MB before compression. Need to check if it fits at WD=0.09.
- **Expected: 0.005-0.008 BPP improvement**
- Risk: Low — simple extension of existing BigramHash
- Could push us from 1.1127 to 1.106-1.108

### 23. Gradient-Guided Adaptive Quantization — NEW from PR #486

**Status: INVESTIGATE**

- During last 10% of warmdown, accumulate per-tensor gradient sensitivity
- Quantize adaptively: top 10% sensitivity → Int7, middle 70% → Int6, bottom 20% → Int5
- Same average bits but much less quant error on sensitive layers + better compression on insensitive ones
- This is the mixed-precision approach done RIGHT — data-driven, not manual
- **Expected: better roundtrip BPP + potentially smaller artifact**
- Risk: Medium — need gradient tracking infrastructure

### 21. TTT Speed Optimization — Keep Everything on GPU

**Status: INVESTIGATE**

- `val_tokens` is on CPU — transferred to GPU every batch during TTT. ~200MB, easily fits on H100.
- `base_bytes_lut`, `has_leading_space_lut`, `is_boundary_token_lut` also on CPU.
- `base_model.eval()` / `base_model.train()` toggles every batch — may trigger torch.compile recompilation
- Moving all to GPU could save 10-20% eval time → finer stride → 0.0004+ BPP
- Low risk, easy implementation

### 20. TTT Research — New Findings (March 25)

**ZOA (Zeroth-Order Adaptation for Quantized NNs, arxiv:2508.02180):**
- Two forward passes instead of backprop — avoids vanishing gradient problem in quantized models
- 5% improvement over first-order adaptation on quantized ViT
- Potentially applicable to our GPTQ-quantized model — our SGD TTT does backprop through dequantized weights which may have gradient issues
- **Status: INVESTIGATE** — could explain why our TTT gains plateau

**LaCT (Large Chunk TTT, arxiv:2505.23884):**
- Uses 2K-1M token chunks instead of small windows → >5% GPU utilization → orders of magnitude faster
- Mentions **Muon as TTT optimizer** — we already use Muon for training, could use it for TTT too
- Our per-window TTT uses tiny batches (128 seqs × 2048 tokens) — we may be leaving perf on the table
- **Status: INVESTIGATE** — Muon TTT could be a breakthrough since we already have the optimizer

### 12. Full GPTQ (not lite)

**Status: NEEDS TESTING**

- PR #593 claims full GPTQ improves post-quant BPB by 0.0048 vs GPTQ-lite
- We already use GPTQ but need to verify it's the full Hessian version with Cholesky error compensation
- Check: does our GPTQ_ENABLED=1 do full Cholesky or simplified?

### 13. EMA Decay Tuning

**Status: NEEDS TESTING**

- Our 14L model is deeper than typical 11L — may need different EMA dynamics
- Test: EMA(0.995), EMA(0.9975), EMA(0.9985) vs current EMA(0.997)
- Quick eval-only A/B if we save pre-EMA checkpoints

### 14. Weight Sharing (Recursive Depth)

**Status: HIGH RISK, INVESTIGATE LATER**

- PR #579: 6 unique blocks × 2 loops = 12 effective depth, wider MLP
- GPTQ catastrophically fails at 3+ loops — only 2 loops viable
- Would require major architecture rework
- Potential: large gain but untested at our scale

### 15. Cosine-Annealed SGD TTT LR — UNTESTED, queued after RDO

**Status: QUEUED** — code already in 201 script (`TTT_COSINE_LR=1`), never tested with SGD on submission config.

- PR #581, #589: cosine LR during TTT (not flat) improves by 0.003+
- We use flat SGD lr=0.002 — cosine decay from 0.002→0 across eval window
- NOTE: We tested cosine with AdamW and it failed, but that's because AdamW itself failed. Cosine on SGD is untested.
- Can test with saved model (eval-only mode) on WD=0.09 pre-TTT model
- **Expected: 0.001-0.003 BPP improvement. Quick eval-only test.**

---

## Priority Order (Updated)

1. **VRL** (running) — architecture change, proven in #569
2. **QAT-export alignment** — low effort, directly attacks quant gap
3. **Cosine-annealed TTT LR** — easy eval-only test, proven in #581/#589
4. **Early QAT threshold 0.5** — proven in multiple PRs
5. **Soft-round QAT** — novel, higher effort but targets quant gap
6. **Full GPTQ verification** — make sure we're doing full Hessian, not lite
7. **Int5 + wider MLP** — risky but could unlock more capacity
8. **Multi-pass TTT** — need to verify legality and time budget
9. **EMA decay tuning** — quick eval-only if checkpoints available
10. **Pruning 2% post-GPTQ** — easy test, helps compression

---

## Deep Analysis: Where Our 14L Model Has Specific Inefficiencies

### Our unique situation

We're the ONLY clean submission at 14 layers. Everyone else is at 11L. Our depth advantage gives us ~0.012 BPB over 11L baselines, but it comes with costs:

- **~105ms/step vs ~85ms/step** (fewer training steps: 5700 vs 7000)
- **More params to quantize** (more quantization error compounding across layers)
- **Deeper gradient paths** (vanishing/exploding gradient risk)
- **Less room in 16MB** (14 layers of weights vs 11)

### Specific inefficiencies to target

**A. Quantization gap is our biggest loss (0.0149 BPB)**

- Pre-quant: 1.1268, Post-quant: 1.1417, gap = 0.0149
- Top PRs have gaps of 0.007-0.008 (half ours!)
- Our 14 layers mean more weight matrices to quantize → error compounds layer by layer
- GPTQ helps but we're still leaving ~0.007 BPP on the table vs best practices
- **Fix: QAT-export alignment + Early QAT + possibly Soft-Round QAT**
- **Fix: MLWQ-style per-layer bit allocation — give critical layers more bits**

**B. Attention may be under-utilizing our depth**

- 14 layers of identical attention with no cross-layer communication (except skip connections)
- VRL (testing now) adds one form of cross-layer info (V0 residual)
- **Differential Attention** (ICLR 2025): dual softmax branch, subtract noise → amplify signal. At 14 layers, attention noise compounds more than at 11L. Diff attention specifically helps deeper models.
- **Gated Attention** (arxiv 2505.06708): sigmoid gate after softmax, dynamic sparsity. Could help our deeper model prune irrelevant attention at later layers where patterns are more abstract.

**C. MLP activation may not be optimal for our depth**

- leaky_relu(0.5)² is proven at 11L but untested at 14L
- At 14 layers, the squared activation compounds gradient magnitudes more
- **PolyGLU** (arxiv 2603.13347): state-conditional activation routing — different activation per token based on hidden state. Zero extra params, just routes between existing activation functions.
- SwiGLU (PR #505) needs 3 weight matrices vs our 2 → doesn't fit at 14L without shrinking MLP

**D. BigramHash dim=64 may be undersized**

- We use BigramHash(8192, dim=64) then project to dim=512
- The projection from 64→512 is a bottleneck — 64 dims can't capture much
- PR #505 uses dim=128. Our earlier test (exp186) with dim=128 went over 16MB
- **Fix: Try dim=80 or dim=96 — halfway, might fit**

**E. Skip connections may need tuning for 14L**

- U-Net skip: 7 encoder layers, 7 decoder layers
- Skip weights are learned scalars per dim — initialized to 1.0
- At 14L, the encoder/decoder split is deeper than 11L's 5/6 split
- **Fix: Skip gating** (PR #569 uses learned gating on skips) — sigmoid gate instead of scalar multiply, adapts during training to route information more selectively

**F. Our RoPE base=50000 was tuned for earlier configs**

- Default 10000, we use 50000 — tuned for shorter training at an earlier layer count
- At 14L with ~5700 steps (fewer than 11L's ~7000), the RoPE dynamics differ
- **May be worth retesting** base=10000, 30000, 100000

---

## Novel Ideas Specific to Our 14L Architecture

### N1. Per-Layer Quantization Precision (MLWQ-inspired)

- Instead of uniform int6 everywhere, allocate bits per layer based on loss sensitivity
- First and last layers are most sensitive → give them int7 or int8
- Middle layers are more redundant → could handle int5
- Net: same average bits, much less quantization error where it matters
- **This is the #1 thing that could close our 0.015 quant gap**
- Implementation: run GPTQ with different clip_ranges per layer based on Fisher information or gradient magnitude

### N2. Differential Attention (lightweight version)

- Add a second, smaller attention branch (maybe 2 KV heads) that gets subtracted
- Paper shows it specifically helps deeper models by canceling accumulated noise
- Extra cost: ~2 extra KV heads per layer × 14 layers = modest param increase
- Could implement as: `y = y_main - lambda * y_noise` with learned lambda per layer
- **Particularly relevant for us** because 14L accumulates more attention noise than 11L

### N3. Layer-Wise LR Scaling for TTT

- Our per-window SGD TTT uses same LR for all unfrozen layers
- PR #545 uses per-layer LR groups: `lr * (0.5 + 0.5 * layer_idx / (num_layers - 1))`
- Later layers need more adaptation (closer to output), earlier layers less
- **Easy to implement** in the TTT optimizer setup, eval-only testable

### N4. Asymmetric U-Net (more encoder than decoder)

- Currently 7 encoder + 7 decoder (symmetric)
- Research suggests asymmetric splits (e.g., 9 encoder + 5 decoder) can be better
- Encoder layers are "cheaper" (no skip addition) → slightly faster per step
- More encoder layers = richer skip representations for decoder
- **Zero extra params, just changes the split point**

---

## Updated Priority Order (with novel ideas integrated)

### Tier 1: Highest confidence, lowest effort

1. **VRL** (running) — proven in #569
2. **QAT-export alignment** — one constant, attacks 0.015 quant gap
3. **Cosine-annealed TTT LR** — eval-only test, proven in #581/#589
4. **Per-layer TTT LR** — easy, specifically helps 14L
5. **Early QAT threshold 0.15-0.5** — proven in multiple PRs

### Tier 2: Medium effort, novel/high potential

1. **Per-layer quant precision (MLWQ)** — directly targets our #1 loss (quant gap)
2. **Soft-round QAT** — novel, bin-aware gradients
3. **Asymmetric U-Net split** — zero params, might unlock free BPB
4. **Skip gating (sigmoid)** — better than scalar for 14L depth

### Tier 3: Higher effort, uncertain

1. **Int5 + 15th layer** — risky, need int5 GPTQ quality to be good enough
2. **Differential attention (lightweight)** — novel, param cost
3. **BigramHash dim=80-96** — quick if artifact has room
4. **Full GPTQ verification** — verify Cholesky, not lite
5. **Multi-pass TTT** — legality + time budget questions

### Tier 4: Research directions

1. **PolyGLU activation routing** — zero params, novel
2. **RoPE base re-tuning** — may have drifted from 14L optimum
3. **EMA decay tuning for 14L** — deeper model may want different decay

---

## CRITICAL RESEARCH: What's Insanely Applicable to Our 14L Case

### The #1 Insight: Our quantization gap compounds across 14 layers

From EPTQ/HAWQ research: quantization error doesn't just add across layers — it **multiplies**. Each layer's output has small perturbations from quantization, and the next layer amplifies those perturbations through its nonlinear activations. With 14 layers vs 11, we have 27% more layers for error to compound through.

**This explains why our quant gap (0.015) is 2x the 11L PRs (0.007).**

It's not that our GPTQ is worse — it's that we have more layers for error to propagate. The fix isn't better GPTQ — it's reducing error propagation.

### R1. Hessian-Weighted Per-Layer Bit Allocation (HAWQ-style)

**Why it's perfect for us:**

- Our 14 layers have different sensitivities to quantization
- First layer (embedding projection) and last layer (pre-logit) are most sensitive
- Middle U-Net layers are least sensitive (redundancy from skip connections)
- **Concrete plan:** Run one forward pass with calibration data, compute per-layer Hessian trace (cheap via Hutchinson's estimator), then:
  - Layers with top 3 Hessian traces → int8 (256 levels)
  - Layers with bottom 3 Hessian traces → int5 (32 levels)
  - Rest stay int6 (64 levels)
  - Net: same average bits, dramatically less total quantization error
- **Expected gain: 0.003-0.007 BPB** (could halve our quant gap)
- **Effort: Medium** — need to add Hessian computation + mixed-bit GPTQ

### R2. Block-Wise Error Compensation Order

**Why it matters for 14L:**

- Standard GPTQ quantizes layers in order (0, 1, 2, ..., 13)
- Error from layer 0 propagates to layer 1's calibration data, corrupting it
- By layer 13, the calibration data has been corrupted 13 times
- **Fix:** Quantize in sensitivity order (most sensitive first, while calibration data is cleanest)
- Or: quantize from both ends inward (0, 13, 1, 12, 2, 11, ...)
- **Expected gain: 0.001-0.003 BPP**
- **Effort: Low** — just reorder the GPTQ loop

### R3. Activation Smoothing Before Quantization (SmoothQuant-style)

**Why it's perfect for deep models:**

- Deep models develop activation outliers that make quantization harder
- 14 layers = more outlier buildup than 11
- SmoothQuant migrates the quantization difficulty from activations to weights via per-channel scaling
- This makes GPTQ's job easier on every layer
- **Concrete plan:** Before GPTQ, compute per-channel activation scales from calibration data, fold them into weights
- **Expected gain: 0.002-0.004 BPB**
- **Effort: Medium** — add smoothing pass before GPTQ

### R4. Low-Rank Quantization Error Correction (LQER/ResQ) ⭐ NEW VALIDATED

**This is a real, published technique with open-source code:**

- **LQER** (ICML 2024, arxiv:2402.02446, github.com/ChengZhang-98/lqer): Low-Rank Quantization Error Reconstruction. After GPTQ, compute residual `R = W_float - W_quantized`, then SVD to get low-rank approximation. Store low-rank matrices (high precision) alongside quantized weights (low precision). Runs both GEMMs in parallel at inference.
- **ResQ** (ICML 2025, arxiv:2412.14363): Mixed-precision via PCA of activation variance. Identifies low-rank subspace (~1/8 hidden dim) where outliers concentrate, keeps that subspace at 8-bit while rest is 4-bit. "Up to 33% lower perplexity than SpinQuant."

**For our 14L model:**
- Compute `R = W_float - Q_int6(W)` for each weight matrix after QEP GPTQ
- SVD: `U, S, Vt = svd_lowrank(R, rank=8)`
- Store U@diag(S) and Vt in fp16 alongside int6 weights
- Size cost: per 512×512 matrix at rank=8: (512×8 + 8×512) × 2 bytes = ~16KB
- Total: ~56 matrices × 16KB = **~900KB** — easily fits in 16MB budget
- Expected recovery: **30-50% of quant error → 0.005-0.008 BPB**
- **Effort: Medium** — SVD after GPTQ, modify inference to add correction
- **This directly attacks our #1 problem (0.015 quant gap) with proven methods**

### R5. Knowledge Distillation from Pre-Quant Model During TTT

**Why it's uniquely applicable to our setup:**

- We save the pre-quant model state (EMA weights before GPTQ)
- During TTT, we could use the pre-quant model as a teacher
- TTT adaptation target: match pre-quant model's output distribution, not just next-token loss
- This directly reverses quantization damage during eval
- **Expected gain: 0.002-0.005 BPB** (directly attacks quant gap during TTT)
- **Effort: High** — need to load both models, compute KL divergence
- **Risk: May not fit in GPU memory** (two 14L models)

### TLDR: If we only do ONE thing

**R4 (LQER-style low-rank residual correction)** is now the top priority. It's a published, validated technique (ICML 2024) that directly attacks our 0.015 quant gap. SVD of GPTQ residuals with rank=8 costs only ~900KB but could recover 0.005-0.008 BPP. Combined with our existing QEP GPTQ, this could bring our quant gap from 0.012 down to 0.004-0.007 — matching the 11L baselines. R1 (mixed-precision bit allocation) is the second priority.

---

## Sources & References

### Quantization

- **HAWQ-V3** (Hessian-aware per-layer bit allocation): [https://assets.amazon.science/a5/a5/bc16183e477aabdb282bfbeea260/hawq-v3-dyadic-neural-network-quantization.pdf](https://assets.amazon.science/a5/a5/bc16183e477aabdb282bfbeea260/hawq-v3-dyadic-neural-network-quantization.pdf)
- **FlexQ** (INT6 with flexible group sizes): [https://arxiv.org/abs/2508.04405](https://arxiv.org/abs/2508.04405)
- **EPTQ** (Hessian-guided block reconstruction): [https://arxiv.org/abs/2309.11531](https://arxiv.org/abs/2309.11531), code: [https://github.com/ssi-research/eptq-sla](https://github.com/ssi-research/eptq-sla)
- **APTQ** (Attention-aware mixed precision): [https://arxiv.org/abs/2402.14866](https://arxiv.org/abs/2402.14866)
- **MLWQ** (Multi-level weight quantization for SLMs): [https://aclanthology.org/2025.emnlp-main.408](https://aclanthology.org/2025.emnlp-main.408)
- **SmoothQuant** (Activation smoothing before quant): [https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- **Soft-Round QAT** (PR #589 in parameter-golf): temperature-controlled smooth rounding surrogate
- **LQER** (Low-Rank Quantization Error Reconstruction, ICML 2024): [https://arxiv.org/abs/2402.02446](https://arxiv.org/abs/2402.02446), code: [https://github.com/ChengZhang-98/lqer](https://github.com/ChengZhang-98/lqer)
- **ResQ** (Mixed-Precision with Low-Rank Residuals, ICML 2025): [https://arxiv.org/abs/2412.14363](https://arxiv.org/abs/2412.14363)
- **Quantization-Aware Regularizers** (Cluster weights during training): [https://arxiv.org/abs/2602.03614](https://arxiv.org/abs/2602.03614)
- **NWC** (Neural Weight Compression, learned compressor for NN weights): [https://arxiv.org/abs/2510.11234](https://arxiv.org/abs/2510.11234)
- **Rate-Distortion PTQ** (Entropy-aware GPTQ rounding): [https://arxiv.org/abs/2505.18758](https://arxiv.org/abs/2505.18758)
- **R²** (Range Regularization for model compression, Apple): [https://machinelearning.apple.com/research/range-regularization](https://machinelearning.apple.com/research/range-regularization)
- **AdaRound** (Optimal up/down rounding for PTQ): [https://arxiv.org/abs/2004.10568](https://arxiv.org/abs/2004.10568)
- **PROXGEN** (Adaptive Proximal Gradient for structured NNs, NeurIPS 2021): [https://proceedings.neurips.cc/paper_files/paper/2021/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2021/file/cc3f5463bc4d26bc38eadc8bcffbc654-Paper.pdf)
- **ProxQuant** (Quantized NNs via Proximal Operators): [https://arxiv.org/abs/1810.00861](https://arxiv.org/abs/1810.00861)
- **Gradient L1 Regularization for Quantization** (ICLR 2020): [https://openreview.net/pdf?id=ryxK0JBtPr](https://openreview.net/pdf?id=ryxK0JBtPr)
- **Relaxed Recursive Transformers** (Weight sharing + per-layer LoRA, ICLR 2025): [https://arxiv.org/abs/2410.20672](https://arxiv.org/abs/2410.20672)
- **ZOA** (Zeroth-Order Adaptation for Quantized NNs): [https://arxiv.org/abs/2508.02180](https://arxiv.org/abs/2508.02180)
- **LaCT** (Large Chunk TTT with Muon optimizer): [https://arxiv.org/abs/2505.23884](https://arxiv.org/abs/2505.23884)
- **FP6-LLM** (6-bit bit-packing for inference): [https://arxiv.org/abs/2401.14112](https://arxiv.org/abs/2401.14112)
- **EntroLLM** (Tensor-level quant + Huffman coding): [https://arxiv.org/abs/2505.02380](https://arxiv.org/abs/2505.02380)
- **QuIP#** (Hadamard rotation for incoherence before quantization): [https://github.com/Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)
- **Permute, Quantize, Fine-Tune** (Weight reordering for compression, CVPR 2021): [https://openaccess.thecvf.com/content/CVPR2021/papers/Martinez_Permute_Quantize_and_Fine-Tune_Efficient_Compression_of_Neural_Networks_CVPR_2021_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Martinez_Permute_Quantize_and_Fine-Tune_Efficient_Compression_of_Neural_Networks_CVPR_2021_paper.pdf)
- **HALO** (Hadamard-Assisted Lower-Precision Optimization, NeurIPS 2025): [https://neurips.cc/virtual/2025/poster/118283](https://neurips.cc/virtual/2025/poster/118283)

### Architecture

- **VRL / ResFormer** (Value Residual Learning): [https://arxiv.org/abs/2410.17897](https://arxiv.org/abs/2410.17897)
- **Differential Attention**: [https://proceedings.iclr.cc/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html](https://proceedings.iclr.cc/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)
- **Gated Attention**: [https://arxiv.org/abs/2505.06708](https://arxiv.org/abs/2505.06708)
- **PolyGLU** (State-conditional activation routing): [https://arxiv.org/abs/2603.13347](https://arxiv.org/abs/2603.13347)
- **Autoregressive U-Net** (gated skip connections): [https://arxiv.org/abs/2506.14761](https://arxiv.org/abs/2506.14761)
- **Depth-Width Tradeoff**: [https://arxiv.org/abs/2503.01805](https://arxiv.org/abs/2503.01805)

### Training & Optimization

- **EMA dynamics in deep learning**: [https://arxiv.org/abs/2411.18704](https://arxiv.org/abs/2411.18704)
- **Muon optimizer**: [https://github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon), [https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)
- **Compute-optimal scaling** (OptiBERT): [https://aclanthology.org/2025.emnlp-main.1804](https://aclanthology.org/2025.emnlp-main.1804)

### Competition PRs (verified clean)

- **PR #569** (1.1175, VRL + GPTQ, no TTT): VRL + LeakyReLU² + Full GPTQ + QAT-export alignment
- **PR #545** (1.1179, int5 GPTQ): 33.6M params, int5 per-row GPTQ, Early QAT 0.5
- **PR #589** (1.1178, Soft-Round QAT): Late Soft-Round QAT + Backward-Looking TTT
- **PR #505** (1.1181, SwiGLU + VE128): SwiGLU + VE128 + no TTT
- **PR #414** (1.1233, EMA + GPTQ-lite): EMA + GPTQ-lite + [QAT@0.15](mailto:QAT@0.15)
- **PR #606** (1.1162, legal SOTA TTT): AdamW TTT lr=0.0001, cosine decay, last 2 blocks + norms + lm_head
- **PR #610** (1.1190, SGD TTT + GPTQ): Found SGD hurts GPTQ models, LoRA doesn't work at 512d
- **PR #615** (1.1169, grouped AdamW TTT): Grouped TTT with AdamW
- **PR #633** (1.1526, single-epoch LoRA TTT): Legal but weak — LoRA too small at 512d
- **PR #638** (1.1164, no TTT): 11L XSA-all + VR + GA, closest non-TTT competitor

---

## HYPERSPECIFIC RESEARCH: Problems Unique to Our 14L/512d/GQA/leaky²/GPTQ Model

### Problem 1: leaky_relu(0.5)² gradient compounding at 14 layers

**The issue:** Our activation is `leaky_relu(x, 0.5).square()`. The squared operation means gradients scale as `2 * leaky_relu(x)` in the backward pass. Over 14 layers of MLP, this creates a multiplicative gradient factor. At 11 layers this is manageable; at 14 layers the gradient magnitude compounds ~27% more.

**Research says** (arxiv:2402.03804, zdtech.substack.com): "Squared ReLU exacerbates vanishing gradients in deep networks because the derivative is proportional to 2x for x>0, leading to rapidly diminishing gradients across many layers."

**Our leaky(0.5) mitigation helps** — the 0.5 negative slope prevents dead neurons. But the squaring still compounds. At 14 layers we may be hitting the limit.

**Potential fixes:**

- **R6. Replace square with abs** — `leaky_relu(x, 0.5).abs()` has gradient magnitude 1 everywhere, no compounding. Zero extra params. But changes the activation landscape.
- **R7. Learnable activation power** — `leaky_relu(x, 0.5).pow(p)` where p is learned per-layer, initialized at 2.0. Lets later layers reduce squaring intensity. 14 extra scalar params.
- **R8. Layer-wise activation scaling** — multiply activation output by `1/sqrt(layer_idx+1)` (like LN Scale but for MLP). Prevents gradient explosion from squaring. We tested LN Scale on attention norms and it hurt (+0.028) — but on MLP activations it's different.
- Source: [https://arxiv.org/abs/2402.03804](https://arxiv.org/abs/2402.03804)

### Problem 2: GQA 8H/4KV wastes attention capacity at 14 layers

**The issue:** With GQA, each KV head serves 2 query heads. At dim=512, each head is 64d. With only 4 KV heads × 64d = 256d of unique key/value information, we're projecting 512d inputs to 256d keys/values — 50% information bottleneck in every layer.

**At 14 layers, this bottleneck compounds:** later layers receive increasingly processed representations but still squeeze them through the same 256d KV bottleneck. 11L models have fewer layers so the bottleneck matters less.

**Research says** (Cost-Optimal GQA, arxiv:2503.09579): "Decoupling total attention head dimensions from model hidden size to flexibly control inference FLOPs." Also: "Weighted Grouped-Query Attention introduces learnable weights for aggregating key and value heads."

**Potential fixes:**

- **R9. Increase KV heads from 4 to 6** — more unique KV information (384d vs 256d). Costs ~33% more attention params but may help deep layers. Need to check if artifact still fits 16MB.
- **R10. Weighted GQA** — add a learned scalar per (query_group, kv_head) pair that weights the KV aggregation. 8 extra params per layer × 14 layers = 112 params total. Near-zero overhead.
- Source: [https://arxiv.org/abs/2503.09579](https://arxiv.org/abs/2503.09579)

### Problem 3: Our RoPE base=50000 was tuned at fewer layers

**The issue:** We set ROPE_BASE=50000 during earlier experiments (likely at 9-12 layers). RoPE base frequency determines how quickly position information decays across sequence length. Higher base = slower decay = longer effective context. But at 14 layers, the model processes positions differently — deeper models build hierarchical position representations that interact with RoPE frequencies.

**Research says** (arxiv:2503.04355): "Layer-Specific Scaling of Positional Encodings for Superior Long-Context Modeling" — different layers should have different RoPE scaling. Early layers need fine-grained position info, deep layers need coarse position info.

**Potential fixes:**

- **R11. Per-layer RoPE base** — layers 0-4 use base=10000 (fine position), layers 5-9 use 50000 (medium), layers 10-13 use 200000 (coarse). Zero extra params, just changes the precomputed cos/sin tables.
- **R12. Sweep RoPE base** — just test 10000, 30000, 50000, 100000 as a hyperparameter. Quick A/B with eval-only if we save checkpoints at different bases (would need retraining).
- Source: [https://arxiv.org/abs/2503.04355](https://arxiv.org/abs/2503.04355)

### Problem 4: Entropy of our quantized weights could be lower for better compression

**The issue:** Our artifact is 15.83MB with brotli-11. We have only 170KB headroom. If we could improve compression by even 200KB, we could try BigramHash dim=96 or wider MLP.

**Research says** (EntroLLM, arxiv:2505.02380): Tensor-level quantization (one scale per entire tensor, not per-row) produces "spiky" weight distributions with much lower entropy → dramatically better compression. Per-row quantization spreads the distribution out.

**The tradeoff:** tensor-level quantization is less precise than per-row, so model quality drops. But if the compression savings let us fit a bigger model, net effect could be positive.

**Potential fixes:**

- **R13. Hybrid quantization** — use tensor-level quantization for the middle (least sensitive) layers and per-row for first/last layers. Better compression on middle layers, preserved quality on critical layers.
- **R14. Huffman coding after quantization** — replace brotli with Huffman coding on the int6 weight values. Since int6 has only 64 possible values, the Huffman tree is tiny and compression is near-optimal. May beat brotli on structured quantized data.
- Source: [https://arxiv.org/abs/2505.02380](https://arxiv.org/abs/2505.02380)

### Problem 5: Our TTT only updates last 12 layers (freeze 2), but research says update only last 25%

**The issue:** We freeze first 2 of 14 layers during TTT (85% unfrozen). Research on efficient TTT says updating only the last 25% of layers (i.e., last 3-4 of 14) maintains performance while being much faster.

**Research says** (test-time-training.github.io, NanoAdapt IJCAI 2024): "Constraining updates to a subset of parameters (e.g., only MLP weights in the last 25% of transformer blocks) maintains performance while lowering computation."

**Potential fixes:**

- **R15. Freeze first 10 layers** (only update layers 10-13). Cuts backward computation by ~70%. Each TTT step is 3x faster → either faster eval or more steps in same budget.
- **R16. Only update MLP weights** during TTT (freeze all attention). MLPs are where most of the model's distribution knowledge lives. Attention patterns are more structural and shouldn't change for domain adaptation.
- Source: [https://test-time-training.github.io](https://test-time-training.github.io), [https://www.ijcai.org/proceedings/2024/0616.pdf](https://www.ijcai.org/proceedings/2024/0616.pdf)

### Problem 6: We are RIGHT at the critical depth — "The Depth Delusion"

**CRITICAL FINDING** (arxiv:2601.20994, "The Depth Delusion"):

> "Transformer performance follows architecture-conditioned scaling laws with a critical depth Dcrit ∝ W^0.44. **At width 512, Dcrit ≈ 16 layers.** Beyond Dcrit, adding layers increases loss despite more parameters due to gradient starvation."

**We are at 14 of 16 critical layers.** This explains EVERYTHING:

- Why 15L was worse (exp189: 1.1171 vs 14L: 1.1155) — approaching Dcrit
- Why techniques that help 11L models fail for us — 11L is well below Dcrit, 14L is near it
- Why our quant gap is so big — gradient starvation in early layers means they're undertrained, making them more fragile to quantization
- Why XSA/LN Scale/Partial RoPE all hurt — they add perturbations to a model that's already at the edge of gradient stability

**The U-shaped loss curve:** Below Dcrit, more depth helps. At Dcrit, it's optimal. Above, loss INCREASES. We're 2 layers below the cliff.

**Implications for our strategy:**

- **DO NOT add a 15th layer.** We're near the edge.
- **Focus on making 14 layers more efficient, not deeper.**
- **Gradient health is critical.** Any technique that worsens gradient flow (squared activations, aggressive weight decay, large learning rates) is more dangerous at 14L than at 11L.
- **Our Muon WD=0.09 might be too aggressive** — high WD shrinks weights, which shrinks gradients through the square activation. At 14L near Dcrit, this could be starving early layers.
- Source: [https://arxiv.org/abs/2601.20994](https://arxiv.org/abs/2601.20994)

### Problem 7: QEP — Quantization Error Propagation is a solved problem we're not using

**CRITICAL FINDING** (arxiv:2504.09629, NeurIPS 2025):
The Quantization Error Propagation (QEP) framework explicitly addresses our exact problem: "quantization errors introduced during layer-wise PTQ accumulate across successive layers, leading to error growth proportional to network depth."

**Their solution:** Add a tunable propagation coefficient α_l to each layer's GPTQ optimization that compensates for upstream quantization errors. When quantizing layer L, they feed in the *actually quantized* outputs from layers 0..L-1 (not the original float outputs). This means each layer's GPTQ optimizes against realistic inputs, not ideal inputs.

**Our current GPTQ probably uses ideal float inputs** for calibration (standard GPTQ). This means layer 13's calibration data is based on perfect float outputs from layers 0-12, but at inference those outputs are quantized. The deeper we go, the bigger the mismatch.

**Fix R17: QEP-aware GPTQ** — During GPTQ calibration, propagate quantized outputs through already-quantized layers instead of float outputs. Each subsequent layer sees realistic (quantized) inputs.

- Expected gain: **0.003-0.007 BPB** (directly proportional to our excess quant gap)
- Effort: Medium — modify GPTQ calibration loop to use quantized intermediate outputs
- This is the single most impactful technique for our specific situation.
- Source: [https://arxiv.org/abs/2504.09629](https://arxiv.org/abs/2504.09629)

### Problem 8: Layer sensitivity is wildly unequal — SmolLM2 study

(arxiv:2603.19348): Study found a "critical core" in layers 8-11 of a 15L model where ablation causes +63,419% perplexity degradation. Also found "anti-layers" at specific depths where removal IMPROVES performance.

**For our 14L model:** Our critical core is probably layers 10-13 (proportionally scaled). This means:

- Layers 0-3 are likely less critical → safe to quantize more aggressively (int5)
- Layers 10-13 are critical → should get int8 or at minimum careful GPTQ
- There may be an "anti-layer" in our model whose weights are net-negative after quantization
- **R18: Profile layer sensitivity** — run GPTQ with each layer individually at int5 vs int6 and measure per-layer BPB impact. The result tells us exactly where to allocate bits.
- Source: [https://arxiv.org/abs/2603.19348](https://arxiv.org/abs/2603.19348)

