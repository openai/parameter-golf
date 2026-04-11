# Parameter Golf Technique Matrix

**Last Updated:** 2026-03-21 13:23 EDT (Auto-scan #4)

Comprehensive map of all techniques mentioned in top PRs, ranked by adoption rate and estimated impact.

---

## Legend

- ✅ **Widely Adopted** (60%+ of top-20 submissions)
- 🟡 **Moderately Adopted** (20-60%)
- 🔴 **Rarely Adopted** (<20%)
- ⚫ **Untried / Ruled Out**
- 🎯 **Our Current Stack** (from PR #198 SOTA base)

---

## Architecture Techniques

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **11 layers (vs 9)** | 🟡 50% | High (0.005) | Low | 🎯 Yes | Funded by Int6 savings |
| **12+ layers** | 🔴 5% | High (0.007) | Medium | ❌ No | Needs Int5 or VQ to fit budget |
| **MLP 3x (vs 2x)** | 🟡 50% | High (0.004) | Low | 🎯 Yes | Standard in SOTA configs |
| **MLP 4x** | 🔴 2% | Medium (0.002) | Low | ❌ No | Budget-constrained |
| **GQA (4 KV heads)** | ✅ 90% | Medium (0.003) | Low | 🎯 Yes | Almost universal |
| **ReLU² activation** | ✅ 80% | Low (0.001) | Low | 🎯 Yes | Better than GELU |
| **SwiGLU activation** | 🔴 5% | Medium (0.002) | Low | ❌ No | PR #250 MoE uses this |
| **U-Net skip connections** | 🔴 10% | Medium (0.003) | Medium | 🎯 Yes | Encoder→decoder residuals |
| **Backout (learned residual subtraction)** | 🔴 2% | **Low (0.002)** | Medium | ❌ No | **PR #295. λ·x_mid subtracted from final output** |
| **Looped Transformer (recurrent depth)** | 🔴 <1% | **Unknown (0.003?)** | High | ❌ No | **PR #325 (1.1462, non-tuned). Shared recurrent core. Risky, needs tuning.** |
| **Mixture of Experts (MoE)** | 🔴 <1% | High? (0.01?) | Very High | ❌ No | PR #250 only. Risky. |
| **Depth recurrence (layer sharing)** | 🔴 2% | Medium (0.003) | High | ❌ No | PR #167, #268 |

---

## Canon Layer Architecture (Zeyuan Allen-Zhu 2025)

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Canon ACD (K=3)** | 🔴 <1% | **High (0.006)** | Medium | ❌ No | **PR #312 (1.1668). Skips expensive B position on QKV concat. Positions: A=pre-attn, C=pre-MLP, D=in-MLP-hidden. arXiv:5240330** |
| **Canon ABCD (K=3)** | ⚫ 0% | Unknown | High | ❌ No | Full canon (includes QKV position B). More expensive. |

---

## Position Encoding

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **RoPE (base=10K)** | ✅ 95% | Baseline | Low | 🎯 Yes | Universal |
| **RoPE (base=50K)** | 🔴 5% | Low (0.001) | Low | ❌ No | For seq2048, smoother interpolation |
| **NTK-RoPE** | 🟡 20% | Low (0.001) | Low | 🎯 Yes | Same as base=10K in our config |
| **Partial RoPE (16 of 64 dims)** | 🔴 <1% | **Medium (0.003)** | Low | ❌ No | **PR #315 (1.1248). 25% of dims use RoPE, rest position-free. Zero params.** |
| **ALiBi** | ⚫ 0% | Unknown | Low | ❌ No | No one using it |

---

## Quantization

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Int6 uniform** | ✅ 70% | High (0.008) | Medium | 🎯 Yes | Range [-32,31], per-row scale |
| **Int5 MLP + Int6 attn** | 🔴 5% | **High (0.006)** | Medium | ❌ No | **PR #295 (1.1477). Saves ~1.9MB. With QAT: quant gap 0.016→0.005 BPB** |
| **Int8 uniform** | 🟡 20% | Medium (0.005) | Low | ❌ No | Worse than Int6 |
| **Int4 uniform** | 🔴 2% | Negative | Medium | ❌ No | 0.06 BPB quant gap |
| **Ternary VQ ({-1,0,1})** | 🔴 <1% | High? (0.01?) | Very High | ❌ No | PR #211 (1.1719 BPB) |
| **QAT (STE)** | 🟡 30% | Negative | High | ❌ No | Early QAT hurts (PR #281) |
| **Late STE QAT (85%+)** | 🔴 5% | **High (0.011)** | Medium | ❌ No | **PR #295, #297. Start QAT at 85% wallclock. Avoids Muon momentum corruption. Gap: 0.016→0.005** |
| **Adaptive Per-Layer Quant (CLASE)** | 🔴 <1% | **Medium (0.003)** | Medium | ❌ No | **PR #309 (1.1914). Boundary layers int8, middle int6, embeddings fp16. Inspired by CLASE (HDXspeed 2026). +15% size savings.** |
| **FP16 embeddings** | ✅ 60% | Low (0.001) | Low | 🎯 Yes | Keeps embedding precision |

---

## Compression

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **zstd-22** | ✅ 80% | High (0.005) | Low | 🎯 Yes | Better than zlib |
| **zlib-9** | 🟡 20% | Medium (0.003) | Low | ❌ No | Fallback when zstd unavailable |
| **LZMA** | 🔴 5% | High (0.007) | Medium | ❌ No | PR #262 (paid prefix, ruled out) |

---

## Attention Mechanisms

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **FlashAttention 2** | 🟡 40% | Medium (0.004) | Low | 🎯 Yes (need to verify) | Better precision than cuDNN |
| **FlashAttention 3** | 🔴 10% | Medium (0.004) | High | ❌ No | PR #198 (1.1318). Hopper-only. |
| **cuDNN SDPA** | 🟡 30% | Negative | Low | ❌ No | 40% faster but worse BPB (PR #281) |
| **xFormers SDPA** | 🟡 20% | Medium (0.003) | Low | ❌ No | Some use this |
| **Vanilla PyTorch attn** | 🔴 10% | Low | Low | ❌ No | Too slow |
| **Sigmoid attention gate** | 🔴 2% | Low? (0.001?) | Low | ❌ No | PR #mattqlf mention |
| **Exclusive Self-Attention (XSA)** | 🔴 5% | **High (0.005)** | Low | ❌ No | **PR #287 (1.1271). Removes self-value bias via orthogonal projection. Zero params. Last 4 layers.** |
| **Partial XSA (GQA-aware)** | 🔴 5% | **Medium (0.003)** | Low | ❌ No | **PR #290 (1.1354). Efficient for GQA. Last 3 layers. arXiv:2603.09078** |
| **XSA4 + EMA combo** | 🔴 <1% | **High (0.007)** | Medium | ❌ No | **PR #307 (1.1357). Best open PR. XSA (last 4) + EMA (decay=0.997). Synergistic effect.** |

---

## Embedding Features

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **SmearGate** | ✅ 70% | Medium (0.003) | Low | 🎯 Yes | Blend token with prev token |
| **BigramHash (2048 buckets)** | ✅ 70% | Medium (0.003) | Low | 🎯 Yes | XOR hash for token pairs |
| **BigramHash (4096 buckets)** | 🟡 20% | Low (0.001) | Low | ❌ No | More buckets, less compression gain |
| **BigramHash (10240 buckets)** | 🔴 2% | **Medium (0.003)** | Low | ❌ No | **PR #295 (1.1477). 5x larger than standard. Better expressiveness** |
| **Bigram dim=128** | ✅ 60% | Baseline | Low | 🎯 Yes | Standard size |
| **Bigram dim=256** | 🔴 5% | Low (0.001) | Low | ❌ No | Too expensive |
| **Memory Tokens (learnable prefix)** | 🔴 <1% | **Medium (0.014)** | Low | ❌ No | **PR #352 (1.1659). 64 learnable embeddings overwrite first K positions of every input sequence. Global context scratchpad. A/B tested: -0.014 BPB improvement.** |

---

## Initialization

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Orthogonal init** | ✅ 70% | Medium (0.003) | Low | 🎯 Yes | Better convergence |
| **muP scaling** | ✅ 60% | Medium (0.002) | Medium | 🎯 Yes | Output layer scaling |
| **Xavier/Kaiming** | 🟡 30% | Baseline | Low | ❌ No | Vanilla init |
| **Overtone init** | 🔴 5% | **Low (0.002)** | Medium | ❌ No | **PR #297. Combined with orthogonal init** |
| **SVD Embedding Init** | 🔴 2% | **Low (0.001)** | Medium | ❌ No | **PR #295. Spectral decay 1/√k for tied embeddings** |
| **Spectral init** | 🔴 5% | Low (0.001) | Medium | ❌ No | We tried this (minimal gain) |

---

## Optimizer & Training

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Muon optimizer** | ✅ 90% | High (0.005) | Medium | 🎯 Yes | Newton-Schulz preconditioner |
| **AdamW** | 🟡 20% | Baseline | Low | ❌ No | Fallback optimizer |
| **Lion** | 🔴 2% | Medium? (0.002?) | Low | ❌ No | Some mention it |
| **Momentum=0.99** | ✅ 70% | Low (0.001) | Low | 🎯 Yes | High momentum is standard |
| **Momentum=0.995** | 🔴 5% | Low (0.001) | Low | ❌ No | LAWA-EMA uses this |
| **Weight decay=0.04** | ✅ 60% | Medium (0.003) | Low | 🎯 Yes | Was 0.01 in baseline |
| **Weight decay=0.042** | 🔴 5% | Low (0.001) | Low | ❌ No | PR #281 optimal for 15.5MB artifact |
| **Gradient clipping=0.3** | ✅ 70% | Low (0.001) | Low | 🎯 Yes | Stability |
| **Warmup steps=1500** | 🟡 40% | Low (0.001) | Low | 🎯 Yes | PR #198 uses this |
| **Warmdown iters=3000** | 🟡 50% | Low (0.001) | Low | 🎯 Yes | Longer than baseline (1200) |
| **LN Scale (layer dampening)** | 🔴 <1% | **Low (0.002)** | Low | ❌ No | **PR #315 (1.1248). RMSNorm output scaled by 1/sqrt(layer_idx+1). Zero params.** |
| **Warmdown iters=15000** | 🔴 <1% | **Medium (0.003)** | Low | ❌ No | **PR #310 (1.1787). 5x longer than standard. Tighter weights → reduced quant penalty.** |
| **Ramping weight decay (0.02→0.08)** | 🔴 <1% | **Low (0.002)** | Low | ❌ No | **PR #309. Cosine schedule during warmdown. Compresses weight dist for cleaner PTQ.** |
| **Cosine warm restarts (SGDR)** | 🔴 2% | Medium? (0.003?) | Medium | ❌ No | PR #250 (MoE) |

---

## Averaging / Ensembling

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **SWA (periodic)** | ✅ 80% | Medium (0.002) | Low | 🎯 Yes | Every 50-200 steps during warmdown |
| **SWA (every 50 steps)** | 🟡 40% | Baseline | Low | 🎯 Yes | Our current config |
| **SWA (every 200 steps)** | 🔴 10% | Low (0.001) | Low | ❌ No | PR #281 uses this (fewer, later checkpoints) |
| **EMA (continuous)** | 🔴 5% | **Medium (0.005)** | Low | ❌ No | **PR #287 (1.1271). Every-step EMA decay=0.997. Better than SWA for compression & generalization** |
| **LAWA-EMA (continuous)** | 🔴 2% | Low (0.001) | Low | ❌ No | Every-step EMA, decay=0.995 |
| **Model soup (ensemble)** | ⚫ 0% | Unknown | Medium | ❌ No | Not allowed? (single artifact rule) |

---

## Training Data & Schedule

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Train seq=2048** | ✅ 70% | High (0.005) | Low | 🎯 Yes | 2x context vs baseline (1024) |
| **Train seq=1024** | 🟡 30% | Baseline | Low | ❌ No | Baseline config |
| **Context-length curriculum** | 🔴 5% | Medium? (0.003?) | Medium | ❌ No | Start 1024, grow to 2048. PR #0xjaishy |
| **Batch tokens=524288** | 🟡 40% | Low (0.002) | Low | ❌ No | More gradient updates per second |
| **Batch tokens=786432** | 🟡 30% | Baseline | Low | 🎯 Yes | Our current (larger batch) |
| **Max wallclock=600s** | ✅ 95% | Constraint | Low | 🎯 Yes | Competition limit |

---

## Evaluation Techniques

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Sliding window (stride=64)** | ✅ 80% | Medium (0.003) | Low | 🎯 Yes | 4x more context overlap |
| **Sliding window (stride=128)** | 🟡 20% | Low (0.001) | Low | ❌ No | Baseline |
| **Non-overlapping eval** | 🔴 5% | Baseline | Low | ❌ No | Worst performance |
| **TTT (LoRA)** | 🟡 30% | Low (0.002) | Medium | 🎯 Yes | Our current TTT |
| **TTT (full-weight SGD)** | 🔴 5% | **Medium (0.007)** | Medium | ❌ No | **PR #317 (1.1419). 3 epochs, lr=0.002, freeze first 2 blocks. Better than LoRA.** |
| **TTT (2 epochs)** | 🔴 5% | Baseline | Low | ❌ No | PR #264 |
| **TTT (3 epochs)** | 🔴 2% | Low (0.001) | Low | ❌ No | PR #281, #317 |
| **TTT (5 epochs)** | ⚫ 0% | Unknown | Low | ❌ No | Unexplored |
| **Causal TTT** | 🔴 <1% | Unknown | High | ❌ No | **PR #322. Each token scored before model sees it. Not yet tested.** |
| **Neural Cache (cross-window KV)** | 🔴 <1% | Unknown | Medium | ❌ No | **PR #318. Cache K/V across sliding windows → 50K+ context. Untested (bug).** |
| **Paid prefix** | ⚫ Banned | N/A | Medium | ❌ No | PR #262, #275 ruled out-of-scope |

---

## Code & Submission

| Technique | Adoption | Impact | Complexity | In Our Stack? | Notes |
|-----------|----------|--------|------------|---------------|-------|
| **Code <1500 lines** | ✅ 100% | Constraint | Low | 🎯 Yes | Hard rule |
| **Code minification** | 🔴 2% | Low (0.003) | Medium | ❌ No | 69KB→40KB frees ~29KB for params |
| **Script submission** | ✅ 95% | Baseline | Low | 🎯 Yes | Standard format |
| **Notebook submission** | 🔴 5% | Baseline | Low | ❌ No | Some use Jupyter |

---

## Summary: Gaps & Opportunities (Updated 2026-03-20 21:23 EDT)

### **🔥 New High-Impact Targets (from latest PRs — Updated 2026-03-21 03:23 EDT):**
1. **Partial RoPE + LN Scale + Late QAT** (<1% adoption, **0.005 BPB total**) — **PR #315 (1.1248 — NEW RECORD)**. All three are zero-param additions. **PRIORITY #1 (NEW)**
2. **Full-model SGD TTT (3 epochs)** (<1% adoption, **0.007 BPB**) — PR #317 (1.1419). Freeze first 2 blocks, lr=0.002, momentum=0.9. **PRIORITY #2 (NEW)**
3. **Canon ACD (K=3)** (<1% adoption, **0.006 BPB**) — PR #312 (1.1668). Skips expensive B position. **PRIORITY #3**
4. **XSA4 + EMA combo** (<1% adoption, **0.007 BPB**) — PR #307 (1.1357), confirmed in #317. **PRIORITY #4**
5. **Late STE QAT (85%+)** (5% adoption, **0.011 BPB**) — PR #295, #297, #315. Closes quant gap 0.016→0.005 without Muon corruption. **PRIORITY #5**
6. **Int5 MLP + Int6 attn** (5% adoption, **0.006 BPB**) — PR #295. With Late QAT: massive quant gap reduction. **PRIORITY #6**

### **High-Impact, Low-Adoption (Secondary Targets):**
7. **Warmdown iters=15000** (<1% adoption, **0.003 BPB**) — PR #310 (1.1787). 5x longer than standard.
8. **Adaptive Per-Layer Quant (CLASE)** (<1% adoption, **0.003 BPB**) — PR #309 (1.1914). +15% size savings.
9. **TTT with full-weight SGD** (5% adoption, 0.005 BPB) — Replace LoRA (confirmed in #290, #297)
10. **Partial XSA (GQA-aware)** (5% adoption, 0.003 BPB) — PR #290 (1.1354). Efficient version of XSA
11. **RoPE base 50K** (5% adoption, 0.001 BPB) — Better positional encoding (PR #290)
12. **BigramHash(10240)** (2% adoption, 0.003 BPB) — 5x larger buckets (PR #295)

### **High-Risk, High-Reward (Moonshots):**
13. **Mixture of Experts** (<1% adoption, 0.01 BPB?) — Sparse computation
14. **Ternary VQ** (<1% adoption, 0.01 BPB?) — Extreme compression
15. **12+ layers** (5% adoption, 0.007 BPB) — Needs Int5 to fit budget

### **Already Saturated (Don't Waste Time):**
- SmearGate, BigramHash(2048), SWA, Muon, Int6, zstd — everyone uses these
- Train seq=2048, sliding window stride=64 — standard now
- 11 layers, MLP 3x, GQA — table stakes for SOTA

---

## Recommended Next Experiments (Updated 2026-03-21 01:23 EDT)

**🔥 Priority Stack (Next Run):**
1. **PR #315 Stack + Full-model SGD TTT** — Best current combo
   - Config: Partial RoPE (16/64) + LN Scale + Late QAT (85%) + XSA4 + EMA + Full-model SGD TTT (3 epochs, freeze first 2 blocks)
   - Expected: ~1.11-1.12 BPB (PR #315 base 1.1248 + TTT gain 0.007 = **~1.118**)
   - Risk: Low (all proven techniques from top 2 open PRs)
   - **PC1 ready:** ❌ Script needs authoring

**Quick wins (1 run each):**
2. **XSA4 + EMA + Late QAT (85%)** — Best combo from open PRs
3. Int5 MLP + Int6 attn + Late QAT (85%) — PR #295 stack
4. Warmdown 15000 + Ramping WD (0.02→0.08) — PR #310 + #309 technique
5. Adaptive Per-Layer Quant (CLASE) — PR #309 solo test

**Medium effort (2-3 runs):**
6. Canon ACD + XSA + EMA + Int5/Int6 + attempt 12 layers
7. Code minification, fund wider model (d=528 or 544)

**Moonshot (if compute allows):**
8. Simple MoE (4 experts, learned routing) on SOTA base
9. Context-length curriculum (1024→2048 over first 50% of training)

---

**End of Matrix**  
**Use this to pick experiments that differentiate from the pack.**
ack.**
