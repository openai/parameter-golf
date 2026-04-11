# Parameter Golf — Combinatorial Analysis & Strategy Handoff

**Date:** 2026-03-26 01:30 CDT
**Purpose:** Identify the optimal combination of techniques to fill 16MB and minimize bpb
**For:** Independent analysis by multiple LLMs (Achilles, Codex, DeepSeek, Grok, etc.)

---

## The Problem

We have a **16,000,000 byte** budget for a language model artifact (code + compressed weights + any auxiliary data). Train on 8×H100 SXM for 10 minutes. Eval on same hardware. Minimize bits-per-byte (bpb) on FineWeb 10B validation set with 1024-token BPE vocabulary.

**Our models are using 5-8 MB out of 16 MB.** We're wasting half the budget. The question is: **what is the optimal way to fill all 16MB?**

---

## Available Components (The Building Blocks)

### A. Architecture Components (from our 8 models)

| # | Component | Source | Size Cost | Speed Cost | Proven Impact | Compatible With |
|---|-----------|--------|-----------|------------|---------------|-----------------|
| A1 | **Standard Transformer Block** | Baseline | ~1.9MB per layer (512d) | ~15ms/layer | Baseline | Everything |
| A2 | **Bigram Embedding** (2048-bucket hash) | M1 Codec | ~1.0MB | +5ms/step | -0.10 bpb (improves learning speed) | A1, A3, A5, A6 |
| A3 | **N-gram Predictor** (unigram+bigram log-prob projection) | M1 Codec | ~4.2MB (1024² bigram table + proj) | +2ms/step | -0.05 bpb est. | A1, A2 |
| A4 | **Shared Weight Block** (1 block × N applications) | M2 Recursive | Same as 1 block | ~15ms × N apps | Extreme param efficiency | A1, C1-C4 |
| A5 | **GatedRNN Layers** (sequential recurrent) | M3 spec | ~2MB per layer | +140ms/layer (KILLER) | Untested (was dead code) | A1 but speed-prohibitive |
| A6 | **Growth Rule** (per-layer ±5% learned scaling) | M8 Crystal | ~0.04MB | +15ms/step | -0.08 bpb on vanilla, HURTS M1 | A1, A4, A7, A8 |
| A7 | **Template Codebook** (N templates + router) | M7 Immune | ~0.03MB (32×512d) | +6ms/step | -0.05 bpb est. | A1, A6 |
| A8 | **LoRA Adapters** (rank 8 on attn+MLP) | M6 Hive | ~0.24MB per set | +3ms/step | Only works with pre-trained weights | A1 |
| A9 | **U-Net Skip Connections** (encoder-decoder with learned skip weights) | Baseline | ~0.01MB per skip | 0ms | Built into baseline | A1, A4 |
| A10 | **BigramHash** (token pair hash features) | M4 Step 9 | ~0.02MB | +2ms/step | Needs 1000+ steps to converge | A1 |
| A11 | **SmearGate** (current/previous token blending) | M4 Step 10 | ~0.01MB | +1ms/step | Needs 1000+ steps | A1 |

### B. Compression Techniques

| # | Technique | Source | Bytes/Param | Max Params in 16MB | Proven? |
|---|-----------|--------|-------------|---------------------|---------|
| B1 | **int8 + zlib** | Baseline | 0.38 | 42M | ✅ Yes |
| B2 | **GPTQ-lite int6/int8 + zstd-22** | M4 Step 12 | 0.30 | 53M | ✅ Yes |
| B3 | **PolarQuant 3-bit** (Google TurboQuant paper) | Spec written | 0.28 | 57M | ❌ Code exists, NOT wired |
| B4 | **Ternary (BitNet b1.58)** | Competition record | 0.22 | 72M | ✅ By others |
| B5 | **Base-3 + LZMA** (ternary packing) | Ternary submission | 0.22 | 72M | ✅ By others |
| B6 | **FP8 QAT** (8-bit float for non-ternary params) | Ternary submission | 0.50 (FP params only) | N/A | ✅ By others |

### C. Training Optimizations

| # | Technique | Source | Cost | Impact |
|---|-----------|--------|------|--------|
| C1 | **Warmdown scheduling** (3500 iterations) | M4/Baseline | Free | -0.02 bpb |
| C2 | **Gradient clipping** (0.3) | M4 | Free | Stability |
| C3 | **EMA** (decay 0.997) | M4/Baseline | +1MB memory | -0.01 bpb |
| C4 | **Muon optimizer** (with WD 0.04) | M4 | Free | -0.03 bpb |
| C5 | **Sequence length 2048** | M4 Step 5 | More VRAM | -0.02 bpb |
| C6 | **LeakyReLU(0.9)²** | Competition PRs | Free (replaces relu²) | -0.024 bpb |
| C7 | **Late QAT** (quantization-aware training last 15%) | M4 | Free | Better roundtrip |
| C8 | **Orthogonal init** | M4 Step 8 | Free | -0.001 bpb |

### D. Eval-Time Techniques (Track B)

| # | Technique | Source | Size Cost | Speed Cost | Impact |
|---|-----------|--------|-----------|------------|--------|
| D1 | **N-gram backoff cache** (orders 2-15) | Competition PRs | 3-5MB | +300s eval | -0.45 bpb |
| D2 | **Score-first TTT** (LoRA adaptation at eval) | Competition PRs | ~0.5MB LoRA weights | +200s eval | -0.15 bpb |
| D3 | **Sliding window eval** (stride 16-32) | M4/Competition | 0MB | +100s eval | -0.025 bpb |
| D4 | **Complementary training** (train neural on hard tokens) | PR #811 | 0MB | Free (training change) | -0.23 bpb |
| D5 | **Entropy-adaptive gating** (blend n-gram vs neural by confidence) | PR #796 | ~0.01MB | +10s eval | -0.18 bpb |
| D6 | **Temperature scaling** (T=0.90 on logits) | Ternary submission | 0MB | Free | -0.005 bpb |

### E. Novel Concepts (Untested Combinations)

| # | Concept | Theory | Risk |
|---|---------|--------|------|
| E1 | **Recursive block + n-gram tables** | 57M param single block × 12 = 684M effective, plus 3MB n-gram | Unknown scaling of shared weights |
| E2 | **Codec + n-gram backoff** | M1's built-in n-gram + explicit backoff tables = triple hybrid | May be redundant |
| E3 | **Template codebook as learnable n-gram** | 256 templates = 256 common patterns, router = pattern matcher | Needs more templates to match explicit n-gram |
| E4 | **Growth rule + ternary** | Per-layer scaling on 72M ternary params | Growth rule may not help ternary |
| E5 | **PolarQuant + n-gram + TTT** | 57M neural (PolarQuant) + 3MB n-gram + 0.5MB TTT LoRA | Most components exist, needs assembly |

---

## Size Budget Analysis

**Total budget: 16,000,000 bytes**
- Code: ~55,000 bytes (fixed)
- Remaining for model + data: **15,945,000 bytes**

### Budget Allocation Options

| Config | Neural Model | Compression | Neural Params | N-gram Tables | TTT LoRA | Spare | Estimated bpb |
|--------|-------------|-------------|---------------|---------------|----------|-------|---------------|
| **Pure Neural Max** | 15.9 MB | PolarQuant 3-bit | 57M | 0 | 0 | 0 | ~1.10-1.25 |
| **Neural + N-gram** | 11.9 MB | PolarQuant 3-bit | 43M | 3.5 MB | 0.5 MB | 0 | ~0.65-0.85 |
| **Neural + Big N-gram** | 7.9 MB | PolarQuant 3-bit | 28M | 7.0 MB | 1.0 MB | 0 | ~0.50-0.70 |
| **Ternary + N-gram** | 13.0 MB | Ternary | 59M | 2.5 MB | 0.4 MB | 0 | ~0.55-0.75 |
| **Ternary Max** | 15.9 MB | Ternary | 72M | 0 | 0 | 0 | ~1.10-1.20 |
| **Recursive + N-gram** | 10.0 MB | PolarQuant 3-bit | 36M (×12 apps) | 5.0 MB | 0.9 MB | 0 | ~0.55-0.80 |
| **Codec + N-gram Backoff** | 10.0 MB | int6+zstd | 33M | 5.0 MB | 0.9 MB | 0 | ~0.50-0.70 |

---

## Compatibility Matrix

Which components work well together? (✅ = proven compatible, ⚠️ = untested, ❌ = known conflict)

| | A1 Trans. | A2 Bigram | A3 N-gram | A4 Shared | A6 Growth | A7 Template | B3 Polar | D1 N-gram Cache | D2 TTT |
|---|---|---|---|---|---|---|---|---|---|
| **A1 Transformer** | — | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **A2 Bigram Embed** | ✅ | — | ✅ | ⚠️ | ❌ hurt M1 | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **A3 N-gram Pred** | ✅ | ✅ | — | ⚠️ | ❌ hurt M1 | ⚠️ | ⚠️ | ⚠️ redundant? | ⚠️ |
| **A4 Shared Block** | ✅ | ⚠️ | ⚠️ | — | ⚠️ | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **A6 Growth Rule** | ✅ | ❌ | ❌ | ⚠️ | — | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **A7 Templates** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — | ⚠️ | ✅ | ⚠️ |
| **B3 PolarQuant** | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | — | ✅ | ⚠️ |
| **D1 N-gram Cache** | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| **D2 Score-first TTT** | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ✅ | — |

---

## Smoke Test Results (Our Data)

| Model | Architecture | Smoke bpb | Size | ms/step | Steps in 30s | Key Technique |
|-------|-------------|-----------|------|---------|--------------|---------------|
| M3 | Optimized baseline | **2.529** | 5.1 MB | 141 | 213 | Warmdown + grad_clip (GatedRNN dead) |
| M1 | Codec (bigram+ngram+transformer) | **2.631** | 8.0 MB | 140 | ~200 | N-gram statistical priors |
| M5 | BigramEmbed + GrowthRule | **3.257** | 6.4 MB | 165 | 183 | Technique combo |
| M8 | Growth rule per-layer scaling | **3.342** | 5.2 MB | 242 | 125 | Per-layer learned scaling |
| M7 | 32 template codebook + router | **3.464** | 5.3 MB | 146 | 206 | Template mixing + diversity reg |
| M4 | 15 stacked optimizations | **3.830** | 5.1 MB | 140 | ~200 | Kitchen sink |
| M2 | Single block × 9 (shared weights) | **~4.01** | 5.7 MB | 140 | ~200 | Extreme param efficiency |
| M6 | 67% frozen + LoRA | **4.031** | 9.4 MB | 138 | 217 | Frozen backbone (failed concept) |

---

## Competition Leaderboard Context

**Track A (pure neural, merged records):**
- 1.1194 bpb — abaybektursun (LeakyReLU + TTT + Muon)
- 1.1228 bpb — signalrush (EMA + GPTQ)
- 1.1233 bpb — 11L GPTQ-lite warmdown QAT
- 1.1570 bpb — 74M ternary BitNet b1.58 (latest merged)

**Track B (n-gram hybrid, pending PRs):**
- 0.295 bpb — Chunk-based 9-gram + score-first TTT
- 0.437 bpb — Complementary training + backoff n-gram mixer
- 0.437 bpb — Distributed prefill + 15-gram + EBLS
- 0.667 bpb — BackoffNgramMixer (clean reference implementation)
- 0.849 bpb — PROTEUS+STYX (5-gram eval cache)

**Google TurboQuant/PolarQuant** (our "Track C" — compression breakthrough):
- 3-bit weights with zero calibration overhead via Hadamard rotation
- 26% more params in same size budget vs standard int5
- Spec written (`TURBOQUANT-SPEC.md`), code partially exists, NOT wired into any model

---

## Your Task

Analyze all of the above and answer:

1. **What is the single best combination of components to fill 16MB and minimize bpb?** Consider architecture (A), compression (B), training (C), eval-time (D), and novel combinations (E). Show your math for the size budget.

2. **What are the top 3 alternative combinations?** Different philosophies (e.g., max neural vs hybrid vs recursive).

3. **Which components should be EXCLUDED?** What doesn't stack well or wastes space?

4. **What's your projected bpb for each combination?** Show your reasoning.

5. **What's the implementation order?** What do we build first for maximum impact with minimum risk?

6. **Are there combinations we haven't considered?** Novel architectures or techniques from your training data that could help?

7. **Which LLMs/tools would be best for implementing each component?** (Codex for code generation, DeepSeek for research, etc.)

---

## Constraints

- **Hard limit:** 16,000,000 bytes total artifact (code + model + data)
- **Training:** 10 minutes on 8×H100 SXM
- **Eval:** 10 minutes on 8×H100 SXM
- **Dataset:** FineWeb 10B tokens, 1024 BPE vocabulary
- **Framework:** PyTorch, single-file submission
- **No external data or network calls during eval**
- **3-seed validation required** (seeds 42, 1337, 2025)
- **Deadline:** April 30, 2026

---

*This document is designed to be self-contained. Any LLM should be able to analyze it independently and provide strategic recommendations.*
