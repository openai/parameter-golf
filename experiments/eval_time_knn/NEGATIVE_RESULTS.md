# Experiment Results — 2026-03-29

**Hardware:** 4×H100 PCIe (vast.ai spot, UK), id:32306178
**Baseline model:** PR #1019 (AR Self-Gen GPTQ + XSA-all + BigramHash 3072×112)
**Baseline BPB:** 1.11525 (4xH100 PCIe, seed 314, 6789 steps, sequential data pipeline)
**PR reference BPB:** 1.11508 (8xH100 SXM, seed 314, 6927 steps)

## Summary

8 experiments tested. **1 positive, 7 negative.**

| # | Experiment | Type | Delta BPB | Verdict |
|---|------------|------|-----------|---------|
| 1 | **Memmap multi-shard data pipeline** | Training | **−0.0033** | **POSITIVE** |
| 2 | Single-layer kNN-LM | Eval-time | +0.0026 | Negative |
| 3 | Multi-layer kNN-LM (11 layers) | Eval-time | +0.0031 | Negative |
| 4 | Sliding window logit averaging | Eval-time | +0.0237 | Catastrophically negative |
| 5 | SelfExtend / extended context 4096 | Eval-time | +0.48 | Catastrophically negative |
| 6 | N-gram log-linear blend (single-pass) | Eval-time | −0.0003 | Marginal, too slow for 10-min budget |
| 7 | Mixed-precision GPTQ (int4 attn) | Post-training | +0.047 | Catastrophically negative |
| 8 | Loss truncation (95th percentile) | Training | +0.081 | Catastrophically negative |

---

## 1. Single-Layer kNN-LM (GPU 0)

**Config:** k=64, λ=0.015, T=1.0, L2 distance, no normalization, 2M store
**Key:** Final hidden states (post-final_norm, pre-lm_head), 512-dim
**Method:** Score each position, then add hidden+target to store. Strict causality.

**Partial results at 14.4% completion (window 140K/969K):**

| Window | Store Size | kNN BPB | Base BPB | Delta |
|--------|-----------|---------|----------|-------|
| 0 | 16K | 1.1204 | 1.1198 | +0.0006 |
| 4K | 2M (full) | 1.1130 | 1.1112 | +0.0018 |
| 48K | 2M | 1.1214 | 1.1189 | +0.0025 |
| 100K | 2M | 1.1233 | 1.1207 | +0.0026 |
| 140K | 2M | 1.1213 | 1.1187 | +0.0026 |

**Analysis:** The kNN interpolation consistently hurts. The delta worsens as the store fills and stabilizes at +0.0026 BPB. The L2 retrieval in raw hidden space adds noise — nearest neighbors in activation space are not necessarily semantically useful for next-token prediction. The model's own predictions (via XSA on all 11 layers) already capture local context patterns.

---

## 2. Multi-Layer kNN-LM (GPU 1)

**Config:** k=64, λ=0.015, T=1.0, cosine similarity (L2-normalized keys), 2M store
**Key:** Concatenated hidden states from all 11 layers, 5632-dim
**Attempt 1:** L2 distance in fp16 → NaN (overflow in 5632-dim squared norms)
**Attempt 2:** L2 distance in fp32 → OOM (2M × 5632 × 4B = 45GB temporary tensor)
**Attempt 3 (final):** L2-normalize keys at insertion, cosine similarity via dot product in fp16

**Partial results at 1.4% completion (window 14K/969K):**

| Window | Store Size | kNN BPB | Base BPB | Delta |
|--------|-----------|---------|----------|-------|
| 0 | 8K | 1.1068 | 1.1064 | +0.0004 |
| 2K | 2M | 1.1299 | 1.1274 | +0.0025 |
| 8K | 2M | 1.1215 | 1.1185 | +0.0030 |
| 14K | 2M | 1.1163 | 1.1132 | +0.0031 |

**Analysis:** Multi-layer concatenation is *worse* than single-layer (+0.0031 vs +0.0026). Early layers (0-4) add noise to the retrieval key — they capture low-level features that don't correlate with next-token prediction. The normalization also loses magnitude information that might be relevant. The 5632-dim keys make each kNN search 11× slower and consume 22.5GB VRAM for the store alone.

---

## 3. Sliding Window Logit Averaging (GPU 2)

**Config:** stride=64, seq_len=2048, geometric mean of probabilities across all covering windows
**Method:** For each position, compute NLL from ALL 32 overlapping windows (not just max-context). Average NLLs (= geometric mean of probabilities).

**Result:**

| Metric | Standard (max-context) | Averaged (all 32 windows) | Delta |
|--------|----------------------|--------------------------|-------|
| val_loss | 1.88306102 | 1.92300541 | +0.03994 |
| val_bpb | 1.11525769 | 1.13891205 | **+0.02365** |

**Coverage:** 100% (62M positions), mean 32.0 windows/position
**Eval time:** 993s (same as standard — no extra model calls needed)

**Analysis:** Catastrophically negative. The issue: in a stride-64 window, positions near the start of the window have only 0-63 tokens of context, while positions at the end have 1984-2047 tokens. Averaging the NLL of a position scored with 64 tokens of context together with one scored with 2048 tokens drags the quality down. The max-context approach is correct — each token should be scored with maximum available context.

A variant that might work: weighted averaging by context length, or only averaging windows where the position has at least N tokens of context. But the expected gain is tiny since the max-context window already provides the best single prediction.

---

## 4. Extended Context / SelfExtend (GPU 3)

### 4a. SelfExtend (position remapping)
**Config:** eval_seq_len=4096, group_window=1024, group_size=4
**Method:** Remap positions beyond 1024 to floor(pos/4) to keep all RoPE embeddings in-distribution.

**Result:** 2.35 BPB — catastrophically broken. The model uses dynamic NTK-aware RoPE scaling (positions are already rescaled when seq_len > train_seq_len=1024). The SelfExtend remapping produces position patterns the model has never seen during training.

### 4b. Plain Extended Context (NTK scaling)
**Config:** eval_seq_len=4096, using model's built-in NTK RoPE scaling

**Result:** 1.59 BPB at window 1600 — still very bad (+0.48 BPB). The model was trained with seq_len=2048 and the dynamic NTK scaling only applies to the 16 RoPE dimensions (of 64 total head_dim). Extending to 4096 produces out-of-distribution attention patterns in those dimensions.

---

---

## 5. N-gram Log-Linear Blend (single-pass, legal)

**Config:** Interpolated n-gram (order 6, hash-based counts), log-linear blend with α=0.02
**Method:** Single-pass sliding window: at each scored position, compute neural logits AND n-gram distribution, blend as `log P = (1-α)*log P_neural + α*log P_ngram`, normalize, score. N-gram cache updated strictly after scoring.

**Result (1% test, 620K tokens):**
- α=0.005: −0.00012 BPB
- α=0.01: −0.00022 BPB
- **α=0.02: −0.00032 BPB** (best)
- α=0.05: +0.00014 BPP
- α=0.10: +0.00357 BPB

**Full eval (3M tokens before killed):** −0.00027 BPB, consistent with 1% test.

**Problem:** Pure Python n-gram blending runs at ~9K tokens/sec (slowing as cache grows). Full 62M tokens would take ~2 hours. The 10-minute eval budget requires ~100K+ tok/s. Would need a C extension to be practical. The −0.0003 gain is real but not worth the engineering.

---

## 6. Mixed-Precision GPTQ

**Hypothesis (from NEXT_STEPS.md):** MLP accounts for 80% of quantization damage. Steal bits from attention (low SVD utilization), give to MLP_down. Expected +0.003-0.005 BPB.

**Method:** Dequantize int6 model to FP32, re-quantize with different bit allocations, compress, evaluate.

| Config | Attn | MLP_up | MLP_down | Size (MiB) | BPB | Delta |
|--------|------|--------|----------|-----------|-----|-------|
| uniform_int6 | 6 | 6 | 6 | 14.87 | 1.11773 | baseline |
| attn4_mlpD8 | 4 | 6 | 8 | 12.84 | 1.16472 | **+0.047** |
| attn5_mlpD7 | 5 | 6 | 7 | 13.88 | 1.12998 | **+0.012** |

**Analysis:** The NEXT_STEPS.md analysis was wrong about attention being insensitive to quantization. SVD utilization (72.6%) does not predict quantization robustness. Int4 attention destroys critical signal. Even int5 hurts. Uniform int6 is already near-optimal — the model needs every bit of precision across all weight types equally.

---

## 7. Loss Truncation (training-time)

**Config:** Clip per-token loss at 95th percentile during training. Tokens above threshold contribute zero gradient. Uses memmap pipeline.

**Result:**
- val_bpb at step 4000: **1.2772** (vs memmap baseline: **1.1959**)
- Final pre-quant val_bpb: **1.2233** (vs baseline: **1.1321**)
- **Delta: +0.081 BPB — catastrophically negative**

**Analysis:** The model learned to predict only "easy" tokens well and abandoned the hard ones. The top 5% of per-token losses included genuine learning signal — rare words, domain-specific terms, context-dependent predictions — not just noise. Truncating these gradients prevented the model from learning long-tail vocabulary and complex patterns. The epiplexity framework's prediction that truncation helps was wrong for this model scale: at 27M params, EVERY gradient signal matters. There's no capacity to waste.

---

## Conclusions

1. **The only positive result is better data sampling.** Memmap multi-shard pipeline (−0.0033) improves training by exposing the model to more diverse data. Everything else — eval-time interventions, quantization tricks, loss engineering — is negative.

2. **This model is tightly optimized.** Uniform int6 quantization, max-context sliding window scoring, and standard cross-entropy loss are all locally optimal. There's no free lunch from post-hoc interventions.

3. **Eval-time symbolic methods don't help.** kNN, n-gram blending, logit averaging, extended context — all negative. The model already captures the patterns these methods target (via XSA, BigramHash, attention).

4. **Quantization precision is uniform.** The SVD/sensitivity analysis predicted attention could tolerate fewer bits. It can't. All weight types need equal precision.

5. **Loss truncation hurts.** At 27M params, every training signal matters. Filtering "hard" tokens removes genuine learning signal, not noise.
