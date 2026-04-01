# Parameter Golf — Complete Experiment Log

**Author:** Ciprian-Florin Ifrim
**Date:** March 2026

---

## Challenge Overview

Train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8×H100 SXM GPUs, evaluated by tokenizer-agnostic bits-per-byte (BPB) compression on the FineWeb validation set.

- **Baseline:** 1.2244 bpb (9L 512d int8+zlib, 1k vocab)
- **Our best (ternary, valid):** 1.1565 bpb sliding (P2, 10L 768d relu² 4×MLP fp8, EMBED_DIM=254, seed=42, 16.00MB)
- **Our best (binary, unconstrained):** 1.1239 bpb sliding (15L 768d binary relu² 4×MLP fp8, 50k steps / ~2h compute, 15.67MB)
- **Our best (quality, over budget):** 1.1771 bpb (F59, 12L 768d swiglu 3×MLP, 21.96MB)
- **Challenge period:** March 18 – April 30, 2026
- **Compute sponsor:** OpenAI ($1M in compute credits)

The challenge is framed as L(N) optimisation — minimising loss given fixed parameter count N, unconstrained by data, compute, steps, or architecture. Related challenges include NanoGPT Speedrun (L(T): lowest loss given constrained time) and NanoGPT Slowrun (L(D): lowest loss given constrained dataset).

---

## Run Numbering Convention

| Prefix | Description |
|--------|-------------|
| Plain (1–100) | Dev runs on RTX 5090, 100 steps |
| R prefix (R1...) | Record runs — 600s on 8×H100, leaderboard-targeted |
| S prefix (S1...) | Scaling runs — 1500 steps or 300s on 8×H100, controlled sweeps |
| SB prefix (SB1...) | Binary scaling runs |
| F prefix (F1...) | Final runs — 600s on 8×H100, official submissions |
| P prefix (P1...) | Pushed/submission runs — final config pushed to GitHub |

Additionally, 20 early architecture iterations were performed on MLX (Mac Studio M1 Ultra, 32GB unified memory) and 2 on MPS (MacBook Pro M1 Pro, 32GB unified memory) for rapid prototyping before GPU scaling.

> **Note:** This document covers ~85 named runs (F, S, R series). An additional ~165 dev runs (plain numbered 1–100, repeated sweeps, smoke tests) were conducted but are not individually listed. Key findings from those runs are incorporated into the sweep tables and decision rationale. Separate synthetic-data notebooks were used to isolate the behaviour of specific techniques (Tversky similarity, linear alternatives, grouped projections) before committing H100 compute.

---

## Hardware

| System | Spec | Notes |
|--------|------|-------|
| Dev | RTX 5090 32GB, single GPU | Triton smem ceiling 101KB/SM; blocks value embeddings and some kernels |
| Mac (MLX) | Mac Studio M1 Ultra 32GB | MLX early iteration, 20 runs |
| Mac (MPS) | MacBook Pro M1 Pro 32GB | MPS early iteration, 2 runs |
| Final | 8×H100 SXM 80GB | Primary training platform |

**Step times at 768d (12L):** relu² 2x: 89ms | relu² 3x: 99ms | relu² 4x: 91ms | swiglu 3x: 127ms | leaky relu 3x: 103ms

**Step times at 512d:** 26L baseline: 149ms → 136ms with FA3 → 127ms with FA3 + fusions + EMBED=256 at 25L

**FlashAttention-3** reduced step time by ~9% (~380 free training steps per 600s run).

**Kernel fusion optimisations** (fused QKV + fused SwiGLU + dataloader + softcap) saved a further ~7-10ms/step.

**Width vs depth discovery:** 12L 768d at 106ms/step gets ~5640 steps in 600s vs ~4720 steps for 25L 512d — 920 extra steps from the faster per-step time of wider/shallower models. Final 10L 768d 4×MLP at 91.8ms/step gets ~6530 steps.

---

## Architecture: Ternary U-Net Transformer

### Quantisation Scheme

BitNet b1.58 ternary quantisation — weights constrained to {−1, 0, +1} with per-group absmean scaling. Approximately 1.6 bits per parameter.

**Compression pipeline:** Base-3 packing (5 trits/byte) or bitmask packing → LZMA (preset=9). Best method auto-selected per run. Bitmask wins when zero fraction is high.

**Quantisation shrinkage fix:** When ternary Q contains zeros, `mean(|Q|) < 1.0`, causing scale mismatch on reload. Fix: inflate by `1/mean(|Q|)` during dequantisation. Eliminates all roundtrip gaps.

### U-Net Skip Connections

The model uses a U-Net style encoder/decoder structure with learned skip connections. The first `num_layers // 2` blocks (encoder) store their outputs; the second half (decoder) receives these via `x = x + skip_weight[i] * skips.pop()`. This allows the decoder to simultaneously access high-level semantic representations (from deep processing) and low-level token-level features (from early processing), without requiring the decoder to reconstruct low-level information from the compressed residual stream.

Additionally, each block receives `x0` (the original input embedding) via a learned residual mix: `x = mix[0] * x + mix[1] * x0`, giving every layer direct access to the raw token representation regardless of accumulated residual drift.

For odd layer counts, the decoder receives the larger half (e.g. 27L → 13 encoder + 14 decoder), which is the standard U-Net convention — more processing power applied after skip injection.

### Factored Embedding

With `EMBED_DIM=254`, token embedding is `[8192, 254]` instead of `[8192, 768]`, with learned projections `embed_proj` (254→768) and `embed_proj_rev` (768→254) for the tied output head.

**EMBED_DIM history:** Started at 128 (dev runs), upgraded to 256 after an optimizer coverage fix revealed that the projection layers had not been receiving gradients (−0.024 bpb improvement vs 128 once trained), then trimmed to 254 to fit artifact+code under the 16,000,000 byte budget (~0.0004 bpb cost, 0.00018/dim from 128→256 scaling data).

### Fused Operations

**Fused QKV:** Single `TernaryLinear(dim, dim + 2*kv_dim)`. **Fused SwiGLU/relu²:** Gate and up projections combined into single wide matrix. Combined saving: ~4-6ms/step.

### Z-Loss Regularisation

`1e-4 * logsumexp(logits)²` (from PaLM/Gemma) anchors logits near zero, keeping gradients sharp through the ternary STE.

---

## Compression Scheme

### Base-3 + LZMA (Primary)

5 trits per byte (1.585 bits/trit), lossless. LZMA at preset=9 achieves ~39% reduction over int8+zlib. Ternary distribution at convergence: ~20–29% zeros, ~35–40% each ±1. The skewed distribution (more zeros) is exploited by LZMA's entropy coding.

### Bitmask Compression (Alternative)

Encodes "is this weight zero?" and "if nonzero, is it +1?" as separate bitmasks. Both methods are tried and the smaller is selected automatically. In practice, bitmask and base-3+LZMA produce nearly identical artifact sizes — bitmask wins marginally in some runs (e.g. S72: 15.84MB vs 15.87MB). Zero fraction would need to drop below ~5% for bitmask to provide a clear advantage; our zero fraction ranges from 17–29% at convergence, making bitmask non-competitive.

### 3D Tensor Support

Conv1d weights (`[dim, dim, kernel]`) are reshaped to 2D before ternary quantisation and restored to original shape on load.

### FP8 QAT

Non-ternary parameters (embeddings, projections) stored at fp8 (e4m3) with Quantisation-Aware Training via STE. Halves fp_params storage (~5MB → ~2.5MB). Typical roundtrip gap: 0.001–0.002 bpb.

---

## Submission Runs (P prefix) — Ternary

Configuration: F88 (10L 768d relu² 4×MLP fp8, WD=0, EMBED_DIM=254, 599s wallclock, TEMP=0.90)

| Seed | Steps | val_bpb | RT bpb | Sliding bpb | Train Time | Eval Time | Artifact | Budget |
|------|-------|---------|--------|-------------|------------|-----------|----------|--------|
| 1337 | 6520 | 1.1825 | 1.1839 | **1.1568** | 599.1s | 428.7s | 15.92MB | 16.00/16.00MB |
| 42 | 6530 | 1.1816 | 1.1837 | **1.1565** | 599.7s | 429.3s | 15.92MB | 15.99/16.00MB |
| 7 | 6530 | 1.1823 | 1.1850 | **1.1578** | 599.6s | 429.0s | 15.92MB | 15.99/16.00MB |
| **Mean** | **6527** | **1.1821** | **1.1842** | **1.1570** | **599.5s** | **429.0s** | **15.92MB** | |
| **Std** | **5** | **0.0005** | **0.0007** | **0.0007** | **0.3s** | **0.3s** | **0.00MB** | |

All three seeds fit within the 16,000,000 byte budget. The standard deviation of 0.0007 bpb across seeds confirms high reproducibility. All runs achieve p < 0.001 improvement over the 1.2244 bpb baseline.

### Batch Size Sensitivity (Ternary, 599s wallclock)

| Batch Tokens | Steps | ms/step | val_bpb | Sliding bpb | Tokens Seen | Fits Budget |
|-------------|-------|---------|---------|-------------|-------------|-------------|
| 262,144 | 10,000 | 49 | 1.2413 | — | 2.6B | No |
| **524,288** | **6,530** | **92** | **1.1850** | **1.1578** | **3.4B** | **Yes** |
| 1,048,576 | 3,480 | 172 | 1.1925 | 1.1659 | 3.5B | No |

524k batch tokens is the optimal operating point. Halving the batch (262k) doubles the step count but degrades quality by 0.056 bpb due to noisier gradients interacting poorly with the ternary STE. Doubling it (1M) sees similar total tokens but fewer gradient updates, costing 0.008 bpb.

---

## Current Best Configuration

### Ternary: 10L 768d relu² 4×MLP fp8, WD=0, EMBED_DIM=254

```bash
NUM_LAYERS=10           MODEL_DIM=768          NUM_HEADS=8
NUM_KV_HEADS=4          MLP_MULT=4             VOCAB_SIZE=8192
ACTIVATION=relu2        LOGIT_SOFTCAP=10       SOFTCAP_TYPE=poly
QK_GAIN_INIT=2.25       ROPE_BASE=5000         ROPE_TYPE=yarn
YARN_MAX_LEN=2048       EMBED_DIM=254          TIE_EMBEDDINGS=1
BITNET_GROUP_SIZE=128   FP_STORAGE=FP8         MUON_WD=0.0
MATRIX_LR=0.04          SCALAR_LR=0.02         TIED_EMBED_LR=0.02
MUON_BACKEND_STEPS=3    MUON_MOMENTUM=0.95     WARMDOWN_FRACTION=0.2
MAX_WALLCLOCK_SECONDS=599
SLIDING_EVAL=1          SLIDING_EVAL_STRIDE=16 TEMP_SCALING=1
TRAIN_BATCH_TOKENS=524288
```

| Metric | Value |
|--------|-------|
| val_bpb (mean) | 1.1821 |
| RT bpb (mean) | 1.1842 |
| Sliding bpb (mean) | 1.1570 |
| Artifact + code | 15,992,753–15,995,705 / 16,000,000 bytes |
| Steps | 6520–6530 |
| ms/step | 91.8 |
| zero_frac | 0.335–0.336 |
| optimal_T | 0.90 |
| Params | 73,685,840 |

---

## Dev Runs (RTX 5090, 100–500 steps)

### Phase 0 — Ternary vs Binary (500 steps, 16L 512d, 1k vocab)

| Run | Config | val_bpb | RT bpb | Artifact | ms/step |
|-----|--------|---------|--------|----------|---------|
| 17 | Ternary baseline | 1.7110 | 1.7300 | 23.95MB | 1312 |
| 18 | Binary {−1,+1} | 1.7121 | 1.7316 | 23.93MB | 1309 |

Ternary wins by 0.0016 bpb. The zero state provides representational benefit.

---

### Phase 1 — Training Techniques (100 steps, 9L 512d, 1k vocab)

| Run | Config | val_bpb | RT bpb | Artifact | Notes |
|-----|--------|---------|--------|----------|-------|
| 19 | Ternary 16L 512d baseline | 2.3371 | 2.3793 | 7.33MB | |
| 20 | + Untie lm_head at 2/3 | 2.3569 | 2.3983 | 8.13MB | Deferred — needs wallclock fix |
| 21 | + Value embeddings | — | — | — | Blocked: RTX 5090 Triton smem |
| 22 | + Smear module | 2.3593 | 2.3985 | 7.33MB | Deferred — gate needs many steps |
| 23 | Baseline 9L 512d | 2.4483 | 2.4768 | 4.45MB | Switched from 16L |
| 24 | + Polynomial softcap | 2.3981 | 2.4438 | 4.45MB | **−0.033 rt** |
| 25 | + Seq length schedule | 2.4633 | 2.5106 | 4.45MB | Deferred — recompile cost |
| 26 | + NorMuon | 2.4018 | 2.4104 | 4.40MB | **−0.033 rt**, 5× smaller RT gap |
| 27 | + Grad accum delay | 2.6298 | 2.6571 | 4.40MB | Deferred — needs 2000+ steps |

---

### Vocabulary Sweep (100 steps, 9L 512d)

| Run | Vocab | val_bpb | RT bpb | Artifact | Notes |
|-----|-------|---------|--------|----------|-------|
| 23 | 1024 | 2.4483 | 2.4768 | 4.45MB | Baseline |
| 28 | 4096 | 2.0930 | 2.0974 | 6.68MB | −0.32 vs 1k |
| **29** | **8192** | **1.9946** | **1.9990** | **9.64MB** | **−0.42 vs 1k — largest single win** |

8192 vocab locked. The tokeniser merges ~1.57× more aggressively than 1k, directly reducing BPB. Val token count drops from 63.8M (sp1024) to 40.5M (sp8192) for the same 50k documents.

---

### Activation Sweep (100 steps, 9L 512d, 8k vocab)

| Run | Activation | val_bpb | RT bpb | Artifact | ms/step |
|-----|-----------|---------|--------|----------|---------|
| 29 | relu2 | 1.9946 | 1.9990 | 9.64MB | 838 |
| 30 | relu | 1.9846 | 1.9879 | 9.63MB | 830 |
| **31** | **SwiGLU** | **1.9704** | **1.9743** | **10.70MB** | **960** |
| 32 | SwiGLU + MTP(2) | 1.9627 | 1.9672 | 10.69MB | 1111 |

SwiGLU with MTP auxiliary loss gives −0.032 bpb but +16% slower. SwiGLU alone gives −0.025 bpb. MTP deferred.

---

### Embedding Factorization Sweep (100 steps, 9L 512d, 8k vocab)

| Run | EMBED_DIM | val_bpb | RT bpb | RT gap | Artifact |
|-----|-----------|---------|--------|--------|----------|
| 33a | 0 (=512) | 1.9931 | 1.9962 | 0.003 | 9.63MB |
| **33d** | **128** | **1.9656** | **1.9656** | **0.000** | **9.12MB** |
| 33c | 256 | 2.0538 | 2.1339 | 0.080 | 6.68MB |
| 33e | 64 | 2.0936 | 2.0968 | 0.003 | 4.49MB |
| 33f | 1024 | 2.0709 | 2.1845 | 0.114 | 15.60MB |

128 was optimal at dev scale. After an optimizer fix revealed the projection layers had not been training, 256 became optimal at full convergence — see EMBED_DIM Sweep at full convergence.

---

### Tversky Neural Network Investigation

Based on Doumbouya et al. (2025). Three-term Tversky similarity: `S = theta * f(A intersection B) - alpha * f(A - B) - beta * f(B - A)` with learned membership functions.

**Feature count sweep (FP16 features, ternary prototypes, 100 steps, 9L 512d):**

| Run | Features | val_bpb | RT bpb | RT gap | Artifact |
|-----|----------|---------|--------|--------|----------|
| — | No Tversky | 1.9751 | 1.9751 | 0.000 | 5.33MB |
| 38 | 16 | 1.9877 | 2.0186 | 0.031 | 5.46MB |
| 39 | 32 | 1.9843 | 2.0133 | 0.029 | 5.57MB |
| 40 | 64 | 1.9790 | 2.0097 | 0.031 | 5.79MB |
| **41** | **128** | **1.9427** | **1.9865** | **0.044** | **6.20MB** |
| 42 | 256 | 1.9737 | 2.0863 | 0.113 | 5.63MB |
| 43 | 512 | 2.0036 | 2.0965 | 0.093 | 5.90MB |
| 44 | 128 + shrinkage fix | 1.9425 | **1.9425** | **0.000** | 6.20MB |

Tversky showed genuine quality benefit (~-0.017 bpb) at dev scale with 128 features and fp16 prototype storage. However, subsequent investigation at full convergence (12L 768d) and with corrected prototype storage showed all Tversky variants within noise of the linear baseline. Additional experiments included full ternary prototypes, shared feature pools across layers, no-features mode, logit-head application, and different membership functions (sigmoid, poly, tanh). A synthetic-data notebook confirmed that Tversky's asymmetric similarity only helps on tasks with genuine directional feature relationships (hypernym/hyponym, cause/effect); next-token prediction on FineWeb web text is not such a task.

At the 768d architecture with relu², Tversky also incurred a 19ms/step overhead because the smaller MLP no longer masked the compute cost.

**Conclusion:** Tversky is quality-neutral on FineWeb language modelling regardless of configuration. Not a quantisation issue, not an optimizer issue — the task simply does not benefit from asymmetric similarity.

---

### Key Hyperparameter Sweeps (100 steps, 9L 512d, 8k vocab)

**QK_GAIN_INIT sweep:**

| Run | QK_GAIN | val_bpb | Delta |
|-----|---------|---------|-------|
| 75 | 1.0 | 2.0007 | +0.0076 |
| 73 | 1.5 | 1.9931 | baseline |
| 81 | 2.15 | 1.9913 | −0.0018 |
| **79** | **2.25** | **1.9898** | **−0.0033** |
| 77 | 2.5 | 1.9915 | −0.0016 |
| 80 | 2.75 | 1.9975 | +0.0044 |
| 78 | 3.0 | 2.0011 | +0.0080 |

Clear inverted-U response. **QK_GAIN_INIT=2.25 locked.**

**LOGIT_SOFTCAP sweep:**

| Run | SOFTCAP | val_bpb | Delta |
|-----|---------|---------|-------|
| 74 | 5 | 1.9942 | −0.0013 |
| **73** | **10** | **1.9931** | **−0.0024** |
| 72 | 20 | 1.9935 | −0.0020 |
| 71 | 50 | 1.9957 | +0.0003 |

**LOGIT_SOFTCAP=10 locked.**

**Softcap type (poly vs tanh):**

| Run | Type | val_bpb | Notes |
|-----|------|---------|-------|
| S23 | poly | 1.3680 | |
| S24 | tanh | 1.3693 | |
| S28/S29 | both at EMBED=1024 | 1.3460–1.3462 | Identical at convergence |

Zero effect. Polynomial retained as default.

**ROPE_BASE sweep:**

| Run | ROPE_BASE | val_bpb | Notes |
|-----|-----------|---------|-------|
| **70** | **5000** | **1.9959** | Best at short training |
| 73 | 10000 | 1.9931 | Close second |
| 69 | 20000 | 2.0008 | |
| 68 | 50000 | 2.0017 | |

**KV Heads:**

| Run | KV_HEADS | val_bpb | Artifact |
|-----|----------|---------|----------|
| **58** | **4 (GQA)** | **1.9955** | **7.75MB** |
| 66 | 8 (MHA) | 2.0148 | 8.46MB |

**MLP_MULT:**

| Run | MLP_MULT | val_bpb | Artifact |
|-----|----------|---------|----------|
| **58** | **2** | **1.9955** | **7.75MB** |
| 64 | 3 | 2.0004 | 9.09MB |
| 65 | 4 | 1.9992 | 10.39MB |

**Storage precision:**

| Run | Storage | val_bpb | RT bpb | RT gap | Artifact |
|-----|---------|---------|--------|--------|----------|
| **90** | **fp16** | **1.9656** | **1.9656** | **0.000** | **9.06MB** |
| 91 | fp8 | 1.9662 | 1.9702 | 0.004 | 7.83MB |
| 92 | fp4 | 1.9661 | 1.9955 | 0.029 | 7.11MB |

**TTT-LoRA sweep (100 steps, ROPE=5000):**

| Run | Rank | LR | TTT bpb | Delta |
|-----|------|-----|---------|-------|
| **85** | **8** | **0.01** | **1.9368** | **−0.0315** |
| 86 | 8 | 0.005 | 1.9378 | −0.0312 |
| 87 | 8 | 0.02 | 1.9644 | −0.0038 |
| **88** | **4** | **0.01** | **1.9371** | **−0.0285** |
| 89 | 16 | 0.01 | OOM | — |

TTT confirmed working at dev scale (−0.0315 bpb). Incompatible at convergence — see TTT investigation.

**EMBED_DIM sweep at 512d (12L, 100 steps):**

| Run | EMBED_DIM | Tversky feat | RT bpb | Artifact | bpb/MB efficiency |
|-----|-----------|-------------|--------|----------|-------------------|
| 95 | 64 | 128 | 2.1961 | 8.40MB | worst |
| 98 | 96 | 128 | 2.0356 | 8.74MB | |
| 97 | 128 | 128 | 1.9656 | 9.12MB | best |
| 99 | 192 | 128 | 2.0409 | 10.07MB | |
| 94 | 256 | 128 | 2.0703 | 10.93MB | |
| 100 | 256 | 256 | 2.0340 | 10.09MB | RT gap 0.021 |
| 96 | 512 (off) | 128 | 2.0642 | 13.50MB | |

128 confirmed optimal at dev scale.

---

### Architecture Sizing Table (Ternary, EMBED_DIM=128, standard proj)

| Config | Layers | Artifact | Under 16MB? | RT gap | Headroom |
|--------|--------|----------|-------------|--------|----------|
| fp16 | 20 | 14.23MB | Yes | 0.0001 | 1.77MB |
| **fp16** | **22** | **15.48MB** | **Yes** | **0.0001** | **0.52MB** |
| fp16 | 24 | 16.74MB | No | — | −0.74MB |
| fp8 QAT | 24 | 14.63MB | Yes | 0.028 | 1.37MB |
| fp8 QAT | 26 | 15.77MB | Yes | 0.066 | 0.23MB |
| **fp8 QAT** | **27** | **15.42MB** | **Yes** | **0.0025** | **0.58MB** |
| fp8 QAT | 28 | 15.92MB+code | Marginal | 0.0029 | ~0MB |
| fp8 QAT | 30 | 16.92MB | No | 0.0029 | −0.92MB |

---

## H100 Record Runs (R prefix)

**Hardware:** 8×H100 SXM 80GB | **Time limit:** 600 seconds

| Run | Config | Steps | val_bpb | RT bpb | Artifact | Notes |
|-----|--------|-------|---------|--------|---------|-------|
| R1 | 22L Tversky fp16 | 4299 | 1.2789 | 1.2792 | 15.80MB | |
| R2 | 26L standard fp16 | 3973 | 1.2649 | 1.2650 | 15.85MB | Pre-LR tuning best |
| R3 | 16L Tversky fp16 | 5949 | 1.2900 | 1.2904 | 11.95MB | Too shallow |
| R4 | 9L Tversky fp16 | 10112 | 1.3374 | 1.3394 | 7.48MB | Way too shallow |
| R5 | 30L fp8 | 2852 | 1.2689 | 1.2815 | 17.22MB | Over budget |
| R6 | 26L fp16, 2× LR | ~4003 | 1.2991 | — | ~15.85MB | LR overshot |
| **R7** | **26L fp16, LR=0.02** | **4008** | **1.2608** | **1.2610** | **15.83MB** | **Best pre-FA3** |
| R8 | 26L fp16, LR=0.01 | 4017 | 1.2853 | 1.2855 | 15.72MB | LR too low |
| R9 | 26L BigramHash | 4010 | 1.2804 | 1.2802 | 15.81MB | BigramHash negative |
| R10 | 26L untie@66% | 3706 | 1.2754 | 1.2753 | 23.15MB | Over budget |
| R11 | 26L tied, updated code | 4009 | 1.2806 | 1.2808 | 15.81MB | Code regression |

**LR sweep (R-series):**

| LR | val_bpb | Notes |
|----|---------|-------|
| 0.08 | 1.2991 | Overshoots — ternary STE amplifies gradient noise |
| **0.02** | **1.2608** | **Optimal** |
| 0.01 | 1.2853 | Too slow |

---

## Scaling Runs (S prefix)

**Hardware:** 8×H100 SXM 80GB | **Steps:** 1500 | **Timer:** disabled (MAX_WALLCLOCK_SECONDS=0)
**Base config:** 26L 512d, EMBED_DIM=128, ROPE=5000, QK_GAIN=2.25, SOFTCAP=10, LR=0.02 all, VOCAB=8192, SwiGLU, SEED=1337

---

### Warmdown Sweep

| Run | Fraction | val_bpb |
|-----|----------|---------|
| S3 | 10% | 1.3467 |
| **S1** | **20%** | **1.3438** |
| S2 | 30% | 1.3443 |
| S4 | 30% repeat | 1.3458 |
| S5 | 40% | 1.3501 |

S2 vs S4 (identical config): 0.0015 bpb spread — confirmed seed variance floor.

### Muon Backend Steps

| Run | Steps | ms/step | val_bpb |
|-----|-------|---------|---------|
| S8 | 3 | 144.87 | 1.3491 |
| S9 | 4 | 146.61 | 1.3448 |
| **S1** | **5** | **149.19** | **1.3438** |
| S7 | 8 | 164.31 | 1.3441 |
| S6 | 10 | 157.95 | 1.3456 |

At full convergence (F6 vs F1): 3 steps matches 5 due to +190 extra training steps. Locked at 3.

### Muon Momentum

| Run | Momentum | val_bpb | zero_frac | Artifact |
|-----|----------|---------|-----------|---------|
| S11 | 0.90 | 1.3680 | 0.179 | 15.39MB |
| **S1** | **0.95** | **1.3438** | **0.205** | **15.56MB** |
| S10 | 0.99 | 1.3505 | 0.259 | 15.78MB |

Higher momentum increases zero_frac, inflating artifact size.

### Architecture Experiments

| Run | Config | ms/step | val_bpb | Notes |
|-----|--------|---------|---------|-------|
| S12 | 20L 640d (80M params) | 160.58 | 1.6676 | 17.75MB — over budget |
| **S1** | **26L 512d baseline** | **149.19** | **1.3438** | **Reference** |
| S13 | 26L, TRAINING_DR=2 | 281.63 | 1.3727 | ~795 effective steps, OOM at DR=3 |

### Eval Depth Recurrence Sweep

| Run | EVAL_DR | val_bpb |
|-----|---------|---------|
| S15 | 0/1 | 1.3685–1.3690 |
| S16 | 2 | 1.3688 |
| S17 | 3 | 1.3681 |
| S18 | 4 | 1.3690 |
| S19 | 5 | 1.3683 |

Total range: 0.0009 bpb — pure noise.

### Weight Decay (1500 steps)

| Run | MUON_WD | val_bpb | zero_frac | Artifact |
|-----|---------|---------|-----------|---------|
| **S15** | **0.00** | **1.3685** | **0.179** | **15.39MB** |
| S20 | 0.04 | 1.3722 | 0.145 | 15.12MB |

WD hurts at 1500 steps but saves 0.27MB. Reversed at full convergence — see Final Ternary Record Runs.

### BigramHash

| Run | Config | Steps | val_bpb | Artifact |
|-----|--------|-------|---------|---------|
| S21 | 26L + BigramHash | 1500 | 1.3681 | 15.45MB |
| R9 | 26L + BigramHash | 4010 | 1.2804 | 15.81MB |

At full convergence: 0.020 bpb worse than R7. The 2.1MB fp16 cost of the bigram table displaces ternary layer depth at convergence. **Not viable within budget.**

### Tied Embedding / Correction Weight / Untie Investigation

| TIE_EMBEDDINGS | UNTIE_AT_FRACTION | LM_HEAD_RANK | Behaviour |
|---------------|-------------------|--------------|-----------|
| 0 | any | any | Untied from start — unstable, loss = log(8192) = 9.01 |
| 1 | 0.0 | 0 | Always tied — current best |
| 1 | 0.66 | 0 | Tied → full-rank untie at 66% of wallclock |
| 2 | 0.0 | 0 | Tied + correction weight residual on tok_emb |
| 2 | 0.66 | 0 | Tied + correction → full-rank untie at 66% |
| 2 | 0.66 | r | Tied + correction → SVD rank-r untie at 66% |

**1500-step results:**

| Run | TIE | UNTIE | RANK | val_bpb | Artifact |
|-----|-----|-------|------|---------|---------|
| S15 | 1 | 0.00 | 0 | 1.3685 | 15.39MB |
| S30 | 2 | 0.00 | 0 | 1.3678 | 15.39MB |
| S36 | 1 | 0.66 | 0 | 1.3648 | 22.83MB |
| **S37** | **2** | **0.66** | **0** | **1.3642** | **22.84MB** |
| S38 | 1 | 0.66 | 0 | 1.3667 | 22.84MB |
| S39 | 0 | 0.66 | 0 | 3.4890 | 10.88MB |

Untie gives +0.005 bpb gain but adds 7.3MB — over budget. **TIE=1, no untie locked.**

### LM Head Factorization (SVD-at-Untie)

| Run | RANK | val_bpb | Artifact | Delta vs baseline |
|-----|------|---------|---------|-------------------|
| S37 | 0 (full) | 1.3642 | 22.84MB | +0.004 — over budget |
| S43 | 32 | 1.4873 | 17.27MB | −0.119 |
| S41 | 64 | 1.4243 | 17.60MB | −0.056 |
| S42 | 128 | 1.3889 | 18.40MB | −0.020 |

SVD factorization does not recover within the remaining 34% of training. The model requires full-rank lm_head for 8192-class separability in 512-dimensional space.

### Tied Embed LR Sweep

| Run | TIED_EMBED_LR | MATRIX_LR | SCALAR_LR | val_bpb |
|-----|--------------|-----------|-----------|---------|
| S33 | 0.01 | 0.02 | 0.02 | 1.3723 |
| **S15** | **0.02** | **0.02** | **0.02** | **1.3685** |
| S34 | 0.03 | 0.02 | 0.02 | 1.3742 |

Symmetric degradation. **TIED_EMBED_LR=0.02 locked.**

### TTT-LoRA Investigation

Test-time training with per-document LoRA adapters. Confirmed working at dev scale (−0.0315 bpb). Incompatible at convergence across 6 diagnostic runs.

| Run | Config | val_bpb | TTT bpb | Notes |
|-----|--------|---------|---------|-------|
| S22 | TTT_LR=0.01 | 1.3690 | 1.5065 | TTT hurts |
| S23 | No lm_head_lora | 1.3690 | 1.4993 | Still hurts |
| S24 | tanh softcap | 1.3693 | 1.4982 | No improvement |
| S25 | Q/V loras only | 1.3692 | 1.5193 | Worse |
| S26 | EMBED_DIM=1024 | 1.3473 | 1.4746 | Bottleneck not cause |
| S27 | 9L (original depth) | 1.4039 | 1.5189 | Still incompatible at 9L |

**Root cause:** Every `TernaryLinear` applies RMSNorm to its input before the weight multiply. The LoRA adapter delta is computed on the pre-normalised representation, but injected into a forward pass where base weights operate on a differently-normalised space. At 100 steps the model is poorly calibrated and LoRA signal dominates. At convergence, the base model's representations are precisely calibrated to this normalised space, and any LoRA delta corrupts rather than adapts. This incompatibility is architectural. **TTT permanently disabled.**

### MTP (Multi-Token Prediction)

| Run | MTP_HEADS | ms/step | val_bpb | Notes |
|-----|-----------|---------|---------|-------|
| **S47** | **0** | **149** | **1.3693** | **Baseline** |
| S45 | 2 | 157 | 1.3704 | +0.0011 worse |
| S62 | 2 | 144 | 1.3727 | +0.0034 worse |

Confirmed at both 1500 steps and full convergence (post-fix retest: 0.006 bpb worse at both MTP=1 and MTP=2). A 60M+ parameter, 1.58-bit model does not have the parameter bandwidth for auxiliary future-planning objectives.

### Smear Module

| Run | SMEAR | val_bpb | ms/step |
|-----|-------|---------|---------|
| **S48** | **0** | **1.3687** | **149** |
| S49 | 1 | 1.3675 | 182 |

+22% slower, −0.0012 bpb at 1500 steps. At full 600s wallclock, smear costs ~740 fewer training steps. Not viable within the ternary 10-minute budget but explored further in the binary track.

### Sequence Length Schedule

| Run | Config | val_bpb | ms/step avg |
|-----|--------|---------|-------------|
| S48 | baseline | 1.3687 | 149 |
| S51 | smear + seq@33% | 1.3660 | ~240 |
| S52 | smear + seq@33% repeat | 1.3640 | ~221 |
| **S58** | **smear + seq@33% + YaRN** | **1.3628** | **~221** |

Real gain at 1500 steps but severe step penalty at full 600s. **Disabled for final runs.**

### Batch Size Schedule

| Run | Config | val_bpb |
|-----|--------|---------|
| S48 | baseline | 1.3687 |
| S50 | smear + batch | 1.3698 |
| S53 | smear + seq + batch | 1.3667 |

Noisier gradients interfere with ternary STE convergence. **Not viable.**

### YaRN Positional Encoding

| Run | Config | val_bpb |
|-----|--------|---------|
| S48 | RoPE baseline | 1.3687 |
| S54 | YaRN 4096 | 1.3705 |
| S55 | YaRN 2048 | 1.3679 |
| S56 | YaRN 2048 + seq@33% | 1.3672 |
| S57 | YaRN 2048 + seq@50% + smear | 1.3637 |
| **S58** | **YaRN 2048 + seq@33% + smear** | **1.3628** |

YaRN 4096 hurts (scale=0.25 too aggressive). YaRN 2048 marginally better. **YaRN 2048 retained; seq schedule disabled.**

ROPE_BASE with YaRN: S63 (10000) = 1.3692, **S61 (5000) = 1.3686**. ROPE_BASE=5000 locked.

### Sliding Window Evaluation

| Run | Stride | Sliding bpb | Eval time |
|-----|--------|-------------|-----------|
| S60 | 16 | 1.3452* | >600s |
| S67 | 24 | 1.3146 | 592s |
| **S61/S66** | **32** | **1.3139–1.3452*** | **~350s** |

*S60/S61 used incorrect momentum=0.90. At full convergence (F1): stride=32 gives 1.2312 sliding bpb in 280s.

### Temperature Scaling

Grid search over T in [0.80, 1.20] on 65,536 training tokens. 5-point grid. Optimal T was consistently 1.00 at convergence for the 512d SwiGLU architecture. At the 768d relu² architecture, T=0.90 was consistently optimal (relu² logits slightly underconfident). **TEMP_SCALING=1 in all final runs.**

### Group Size Sweep (S73–S76, 2000 steps, 27L)

| Run | Group Size | Layers | val_bpb | Artifact | Total |
|-----|-----------|--------|---------|----------|-------|
| S76 | 32 | 27 | 1.2739 | 17.64MB | 17.73MB |
| S75 | 64 | 27 | 1.2683 | 16.22MB | 16.31MB |
| **S73** | **128** | **27** | **1.2677** | **15.53MB** | **15.62MB** |
| S74 | 256 | 27 | 1.2699 | 15.19MB | 15.28MB |

128 wins on both quality and compression.

### Skip Weights Init — Zero vs Ones (S77)

| Run | Init | val_bpb | artifact |
|-----|------|---------|---------|
| S73 | ones | 1.2677 | 15.62MB |
| S77 | zeros | 1.2781 | 15.62MB |

Zero-init is **0.0104 bpb worse**. Decoder needs skip signal from step 0.

### FP8/FP4 Storage with QAT

**FP8 sweep:**

| Run | Config | val_bpb | RT bpb | RT gap | Sliding bpb | Artifact |
|-----|--------|---------|--------|--------|-------------|---------|
| S64 | 26L fp16 | 1.3390 | 1.3390 | 0.000 | 1.3150 | 15.58MB |
| S65 | 30L fp8, no QAT | 1.3346 | 1.3394 | 0.0048 | 1.3150 | 16.92MB |
| S66 | 30L fp8, QAT | 1.3351 | 1.3380 | 0.0029 | **1.3139** | 16.92MB |
| S71 | 27L fp8, QAT | 1.3380 | 1.3405 | 0.0025 | 1.3164 | 15.42MB |
| S72 | 28L fp8, QAT | 1.3377 | 1.3406 | 0.0029 | 1.3166 | 15.92MB |

QAT reduces fp8 RT gap from 0.0048 to 0.0029 (40% improvement). However at full convergence (F3), 28L fp8 QAT (1.2353 sliding) loses to 26L fp16 (1.2312 sliding).

**FP4 sweep:**

| Run | Config | val_bpb | RT bpb | RT gap | Sliding bpb | Artifact |
|-----|--------|---------|--------|--------|-------------|---------|
| S68 | 30L fp4 QAT | 1.3377 | 1.3643 | **0.0266** | 1.3404 | 16.49MB |
| S69 | 26L fp4 Tversky QAT | 1.3543 | 1.3835 | **0.0292** | 1.3606 | 15.01MB |
| S70 | 28L fp4 QAT | 1.3405 | 1.3666 | **0.0261** | 1.3424 | 15.43MB |

FP4 RT gap of ~0.026–0.029 even with QAT is unrecoverable. **FP4 not viable at any layer count.**

### EMBED_DIM Sweep (Full Convergence, 25L)

| Config | EMBED_DIM | Steps | val_bpb | sliding_bpb | artifact | Notes |
|--------|-----------|-------|---------|-------------|---------|-------|
| S80 | 0 (=512) | 4500 | 1.1902 | ~1.168 est | 19.78MB | OOM on sliding eval |
| **F22** | **256** | **4720** | **1.2012** | **1.1739 (s16)** | **16.21MB** | **Best 512d result** |
| F16-era | 128 | 4310 | 1.2245 | — | 16.19MB | Pre-fix baseline |

**EMBED_DIM=256 locked.** Budget impact: fp_params ~4.85MB vs ~2.48MB at 128 (+2.37MB).

---

## Final Ternary Record Runs (F prefix)

**Hardware:** 8×H100 SXM 80GB | **FlashAttention-3 enabled** | **Time limit:** 600 seconds

| Run | Config | Steps | val_bpb | RT bpb | Sliding bpb | Eval time | Artifact |
|-----|--------|-------|---------|--------|-------------|-----------|---------|
| **F1** | **26L fp16, no smear, no seq** | **4362** | **1.2560** | **1.2560** | **1.2312** | **280s** | **15.85MB** |
| F2 | 26L fp16, smear + seq@33% | 3044 | 1.2779 | 1.2778 | 1.2535 | 390s | 15.85MB |
| F3 | 28L fp8 QAT, no smear, no seq | 4019 | 1.2571 | 1.2601 | 1.2353 (s24) | 385s | 16.14MB |
| F4 | 26L fp16, EMA=1 | 4145 | 1.2589 | 2.3307 | — | — | 14.52MB |
| F5 | 26L fp16, EMA fix v1 (smoke) | 407 | 1.5483 | 2.3642 | — | — | 14.90MB |
| F6 | 26L fp16, MUON_BACKEND_STEPS=3 | 4552 | 1.2558 | 1.2558 | 1.2311 (s24) | 362s | 15.81MB |
| F7 | 26L fp16, WD=0.04, steps=3 | 4499 | 1.2552 | 1.2551 | 1.2302 (s24) | 362s | 15.60MB |
| F8 | 28L fp16, WD=0.04, steps=2, LR=0.02 | 4219 | 1.2799 | 1.2801 | 1.2558 (s16) | 577s | 15.92MB |
| F9 | 28L fp16, WD=0.04, steps=2, LR=0.03 | 4231 | 1.2673 | 1.2676 | 1.2431 (s16) | 577s | 16.00MB |
| F10 | 28L fp16, WD=0.04, steps=2, LR=0.04 | 4226 | 1.2636 | 1.2636 | 1.2391 (s16) | 578s | 16.01MB |
| F11 | 28L fp16, WD=0.04, steps=3, LR=0.04 | 4137 | 1.2489 | 1.2488 | — | — | 16.69MB |
| F12 | 28L fp16, WD=0.04, steps=4, LR=0.04 | 4047 | 1.2496 | 1.2500 | — | — | 16.71MB |
| F13 | 28L fp16, WD=0.04, steps=3, LR=0.05 | 4048 | 1.2512 | 1.2510 | — | — | 16.73MB |
| F14 | 28L fp16, WD=0.04, steps=3, LR=0.08 | 4036 | 1.2576 | 1.2574 | — | — | 16.75MB |
| F15 | 27L fp16, AdamW matrix, LR=0.01 | 4676 | 1.2943 | 1.2942 | — | — | 15.71MB |
| F16 | 27L fp16, Muon, LR=0.04, WD=0.04 | 4310 | 1.2245 | — | — | — | 16.19MB |
| **F22** | **25L fp16, EMBED=256, steps=3, WD=0.04** | **4720** | **1.2012** | **1.2011** | **1.1739 (s16)** | **493s** | **16.21MB** |

**Key findings:** F22 with EMBED_DIM=256 and corrected optimizer achieves 0.055 bpb improvement over F1 (the best pre-fix config). 28L extensively attempted (F8–F14) but artifact always over budget at competitive LR. AdamW for matrix params (F15) is clearly worse than Muon.

---

## Phase 2 — Post-Optimizer-Fix Experiments (25L 512d EMBED=256)

### EMA (Exponential Moving Average)

| Run | Config | Steps | val_bpb | RT bpb | Artifact |
|-----|--------|-------|---------|--------|----------|
| F4 | EMA=1, decay=0.999 | 4145 | 1.2589 | 2.3307 | 14.52MB |
| — | Full run with EMA | 4144 | 1.2584 | 1.3776 | 14.94MB |

**EMA is fundamentally incompatible with ternary quantization.** EMA averaging in fp32 produces smoother, more zero-centered weights. More latent weights near zero → more round to 0 in ternary → scale factor mismatch → 0.13 bpb RT gap. **Permanently disabled.**

### Muon Backend Steps — Full Convergence

| Run | Steps | step_avg | val_bpb | sliding_bpb | artifact |
|-----|-------|----------|---------|-------------|---------|
| F1 (steps=5) | 4362 | 137ms | 1.2560 | 1.2312 | 15.85MB |
| F6 (steps=3) | 4552 | 131ms | 1.2558 | 1.2311 | 15.81MB |

6ms/step saving → 190 extra steps → quality equivalent. **MUON_BACKEND_STEPS=3 locked.**

### Weight Decay — Full Convergence

| Run | WD | Steps | val_bpb | sliding_bpb | zero_frac | artifact |
|-----|-----|-------|---------|-------------|-----------|---------|
| F6 | 0.00 | 4552 | 1.2558 | 1.2311 | 0.294 | 15.81MB |
| F7 | 0.04 | 4499 | 1.2552 | 1.2302 | 0.221 | 15.60MB |

WD=0.04 wins at full convergence on the 26L architecture. However at 10L 4×MLP (Phase 4), WD=0.00 was better — wider MLP needs full weight freedom.

### MTP Retest (Post-Fix)

| Run | MTP_HEADS | Steps | step_avg | val_bpb | artifact |
|-----|-----------|-------|----------|---------|---------|
| F22 baseline | 0 | 4720 | 127ms | 1.2012 | 16.29MB |
| Run 26 | 1 | 4560 | 131ms | 1.2074 | 16.30MB |
| Run 27 | 2 | 4420 | 135ms | 1.2074 | 16.29MB |

**MTP confirmed not viable post-fix.** 0.006 bpb worse at both heads. **MTP_HEADS=0 permanently locked.**

### Tversky Phase 2 (Post-Fix, 12L 768d, fp16 Prototypes)

Comprehensive retest with corrected optimizer and fp16 prototype storage:

| Run | Config | Features | Pools | val_bpb | RT gap |
|-----|--------|----------|-------|---------|--------|
| 49 | No Tversky | — | — | **1.1888** | 0.0002 |
| 50 | Attn proj only | 128 | 1 | 1.1893 | 0.0000 |
| 51 | Attn proj only | 256 | 1 | 1.1894 | 0.0001 |
| 52 | Attn proj only | 32 | 1 | 1.1898 | 0.0001 |
| 53 | Attn + head | 128 | 1 | 1.1892 | — |
| 54 | Attn + head | 128 | 0 (local) | 1.1897 | +0.0006 |

All variants within 0.001–0.002 bpb of baseline — pure noise. Confirmed by synthetic-data analysis that Tversky's asymmetric similarity only helps on tasks with directional feature relationships, which next-token prediction on web text is not.

---

## Phase 3 — Architecture Exploration (Post-Optimizer-Fix)

### Width vs Depth

The central Phase 3 finding: wider models with fewer layers beat deeper models.

#### 768d Scaling Curve

| Run | Layers | Steps | step_avg | val_bpb | Artifact |
|-----|--------|-------|----------|---------|----------|
| 34 | 8 | 8110 | 74ms | 1.2894 | 12.94MB |
| 30 | 12 | 5640 | 106ms | 1.1893 | 17.50MB |
| 38 | 14 | 4900 | 122ms | 1.1870 | 19.79MB |
| 33/37 | 16 | 4320 | 139ms | 1.1825–37 | 22.08MB |
| 39 | 18 | 3870 | 155ms | 1.1801 | 24.39MB |
| 36 | 20 | 3510 | 171ms | 1.1854 | 26.67MB |

Peak at 18L, then step penalty dominates. 8L collapses (U-Net encoder too shallow). Seed variance: Run 33 vs 37 = 0.0012 bpb.

#### Cross-Architecture Comparison

| Config | Layers | Dim | Steps | val_bpb |
|--------|--------|-----|-------|---------|
| F22 | 25 | 512 | 4720 | 1.2012 |
| Run 30 | 12 | 768 | 5640 | 1.1893 |
| Run 40 | 8 | 1024 | 5870 | 1.1858 |
| Run 41 | 10 | 896 | 5400 | 1.1862 |
| Run 35 | 20 | 640 | 4170 | 1.1927 |
| Run 42 | 6 | 896 | 8510 | 1.2157 |

Width beats depth: 12L 768d (1.1893) beats 25L 512d (1.2012). Minimum viable depth: 768d ~10–12L, 896d ~10L, 1024d ~8L.

### FP8 at 768d

| Run | Layers | Storage | val_bpb | RT bpb | RT gap |
|-----|--------|---------|---------|--------|--------|
| 49 | 12 | fp16 | 1.1888 | 1.1886 | 0.0002 |
| 42 | 13 | fp8 | 1.1879 | 1.1900 | 0.0021 |

FP8 RT gap acceptable at 768d. Enables extra layers within budget.

### LM_HEAD_RANK Investigation (Post-Fix, 768d)

| Run | Config | val_bpb | RT bpb | Total | Notes |
|-----|--------|---------|--------|-------|-------|
| Run 49 | baseline | 1.1888 | 1.1886 | 17.50MB | Reference |
| Run 43 | TIE=2, rank=256, fp8 | 1.2021 | 1.2028 | 20.41MB | Artifact bloated |
| Run 44 | TIE=0, rank=512, untie=0.0 | 1.3196 | 1.3195 | 16.92MB | Random head, no learning |
| Run 45 | TIE=2, rank=512, fp16 | 1.2312 | 1.2317 | 26.87MB | Catastrophic artifact blowup |

Root cause: the SVD factors U and V require fp16/fp8 precision to maintain approximation quality. At any viable compression level, the two new matrices cost more storage than the original tied embedding saves. **Not viable.**

---

## Phase 4 — Final Architecture Search

### Activation Sweep (12L 768d 3×MLP, 600s)

| Run | Activation | MLP | ms/step | Steps | val_bpb | Artifact |
|-----|-----------|-----|---------|-------|---------|----------|
| F55 | relu | 2× | 88.7 | 6760 | 1.2284 | 14.49MB |
| **F56** | **relu²** | **2×** | **89.5** | **6700** | **1.2042** | **14.48MB** |
| F60 | leaky relu | 3× | 102.6 | 5840 | 1.2094 | 17.50MB |
| **F57** | **relu²** | **3×** | **101.5** | **5910** | **1.1878** | **17.51MB** |
| F58 | swiglu | 3× | 127.4 | 4700 | 1.1786 | 22.05MB |
| **F59** | **swiglu** | **3×** | **127.3** | **4710** | **1.1771** | **21.96MB** |

relu² beats relu by 0.024 bpb at no cost — strictly dominant. relu² locked for budget-constrained path.

### MLP Width Sweep (600s)

| Run | Activation | MLP | Layers | ms/step | Steps | val_bpb | Artifact |
|-----|-----------|-----|--------|---------|-------|---------|----------|
| F56 | relu² | 2× | 12 | 89.5 | 6700 | 1.2042 | 14.48MB |
| F64 | relu² | 3× | 12 | 99.4 | 6030 | 1.1873 | 17.50MB |
| F75 | relu² | 4× | 12 | 91.6 | 6550 | 1.1795 | 20.54MB |
| F82 | relu² | 4× | 10 | 91.6 | 6550 | 1.1861 | 16.04MB |

4× MLP at 10L beats 3× at 12L within similar budget.

### Layer Count vs MLP Width (fp8, 600s)

| Run | Config | Layers | ms/step | Steps | val_bpb | RT bpb | Artifact |
|-----|--------|--------|---------|-------|---------|--------|----------|
| F78 | relu² 3× fp8 | 12 | 99.3 | 6040 | 1.1884 | 1.1898 | 15.80MB |
| F77 | relu² 3× fp8 | 13 | 106.6 | 5630 | 1.2065 | 1.2077 | 16.96MB |
| F80 | relu² 2× fp8 | 15 | 106.9 | 5610 | 1.2120 | 1.2136 | 15.45MB |
| F81 | relu² 2× fp8 | 16 | 113.9 | 5270 | 1.1996 | 1.2009 | 16.33MB |
| F79 | relu² 3× fp8 | 11 | 91.5 | 6560 | 1.1920 | 1.1933 | 14.66MB |
| **F82** | **relu² 4× fp8** | **10** | **91.6** | **6550** | **1.1861** | **1.1877** | **16.04MB** |
| F83 | swiglu 3× fp8 | 10 | 105.5 | 5690 | 1.1842 | 1.1853 | 17.29MB |

### Weight Decay at 10L 4×MLP fp8

| Run | WD | val_bpb | RT bpb | Artifact |
|-----|-----|---------|--------|----------|
| F82 | 0.04 | 1.1861 | 1.1877 | 16.04MB |
| F84 | 0.08 | 1.1983 | 1.1998 | 16.04MB |
| **F85** | **0.00** | **1.1828** | **1.1844** | **16.02MB** |
| S87 | 0.00 | 1.1831 | 1.1843 | 16.01MB |
| **F88** | **0.00 (EMBED=254)** | **1.1820** | **1.1839** | **16.00MB — FITS** |

WD=0 optimal at 10L 4× — opposite to 26L result. Wider MLP needs full weight freedom.

---

## Binary Quantisation Track

### Motivation

Binary quantisation constrains weights to {-1, +1} with no zero state. At 1 bit/param vs ternary's 1.6 bits/param, binary packs approximately 60% more parameters per MB. The hypothesis was that additional depth could compensate for the loss of the zero state.

Starting point: the ternary best config (10L, 768d, 8h, 4kv, 4× relu², FP8, 524k batch, 599s) scoring 1.1578 sliding bpb.

### Binary Scaling Runs

| Run | Layers | MLP | FP | Other | Steps | ms/step | Sliding bpb | Artifact | Fits |
|-----|--------|-----|-----|-------|-------|---------|-------------|----------|------|
| F17 | 17 | 4× | FP8 | — | 4010 | 149 | 1.2022 | 17.45MB | No |
| **F1** | **14** | **4×** | **FP8** | **—** | **4820** | **124** | **1.1824** | **14.74MB** | **Yes** |
| F2 | 14 | 4× | FP8 | EMA | 4800 | 125 | 1.2110 | 14.56MB | Yes |
| S3 | 15 | 4× | FP8 | — | 1000 | 133 | 1.3114 | 15.65MB | Yes |
| S4 | 20 | 3× | FP8 | — | 1000 | 160 | 1.3077 | 16.90MB | No |
| S5 | 21 | 3× | FP4 | — | 1000 | 167 | 1.3676 | 16.64MB | No |
| S6 | 19 | 3× | FP8 | — | 1000 | 152 | 1.3130 | 16.16MB | No |
| S7 | 15 | 4× | FP8 | refiner | 1000 | 135 | 1.3123 | 15.89MB | Yes |
| S8 | 15 | 4× | FP8 | smear | 1000 | 155 | 1.3043 | 15.67MB | Yes |
| S9 | 15 | 4× | FP8 | tversky_attn | 1000 | 179 | 1.4016 | 15.74MB | Yes |

### Key Decisions from Binary Scaling

**MLP width (4× vs 3×):** 4× won even when 3× received 4–5 extra layers. S3 (15L 4×) outperformed S6 (19L 3×) at matched steps. Width matters more than depth past a minimum viable layer count.

**FP storage (FP8 vs FP4):** FP4 added a 0.06 bpb roundtrip penalty and was immediately ruled out. FP8 used for all non-binary tensors.

**Layer count:** 17L was the theoretical maximum at 4× FP8 but landed 1.45MB over budget. 15L at 15.65MB was the maximum that fit. 14L left 1.26MB headroom.

**EMA:** Mathematically sound for binary (no zero bucket means `mean(|Q|)=1.0` always, clean roundtrip). In practice, 0.03 bpb worse — the smoothed weights apparently hurt binary's learning dynamics despite the clean quantisation math.

**Smear:** 0.007 bpb gain at 1000 steps but added 22ms/step overhead (133→155ms). Retained for the extended binary run to test whether the gain survives the step penalty at longer training.

**Refiner (causal conv):** Neutral at 1000 steps, added 2ms/step. Not justified.

**Tversky attention projection:** 0.09 bpb worse. Completely incompatible with binary weights.

**Activation:** relu² inherited from ternary sweeps, not retested for binary. SwiGLU would cost ~4MB extra across 15 layers, eliminating the layer budget advantage.

### Extended Binary Run (Unconstrained Compute)

To measure the binary architecture's convergence ceiling without the 10-minute wallclock constraint, a single extended run was conducted at 50,000 steps (~2 hours on 8×H100).

**Configuration:** 15L 768d, 4× relu², FP8, smear, 524k batch tokens, seed=42, MUON_WD=0.0

```
step:50000/50000 val_loss:2.9692 val_bpb:1.1497 train_time:7763s
artifact:15.60MB binary:97320960(13685760B) fp:2542200(2585072B) code:70399
budget:15670651/16000000 (15.67/16.00MB) FITS
final_binary_roundtrip val_loss:2.9743 val_bpb:1.1516
temp_scaling optimal_T:0.90
final_sliding val_loss:2.9027 val_bpb:1.1239 (stride=16, T=0.90)
```

| Metric | Value |
|--------|-------|
| val_bpb | 1.1497 |
| RT bpb | 1.1516 |
| Sliding bpb | **1.1239** |
| Artifact | 15.60MB (15.67MB total) |
| Params | 97,320,960 |
| Steps | 50,000 |
| ms/step | 155.3 |
| Training time | ~2.15 hours |

The 1.1239 sliding bpb demonstrates that with sufficient compute the binary architecture reaches strong quality. This validates the compression approach — nearly 100M parameters in 15.67MB via 1-bit quantisation — though the 50k steps required far exceeds the competition's 10-minute budget.

### Binary vs Ternary at Equal Architecture (Dev Scale)

| Metric | Binary | Ternary | Delta |
|--------|--------|---------|-------|
| val_bpb | 1.8609 | 1.8113 | Ternary wins by 0.050 |
| Artifact | 9.14MB | 11.56MB | Binary saves 2.42MB |
| ms/step | 918 | 924 | Identical |
| RT gap | 0.000 | 0.000 | Both clean |

Ternary is better at equal architecture. Binary's only advantage is fitting more layers in the same budget.

### Binary Conclusion

Binary lost the depth-for-sparsity trade. The 5 extra layers (15L binary vs 10L ternary) could not overcome ternary's representational advantage from the zero state. The 0.0016 bpb gap measured at 500 dev steps significantly understated the true difference at convergence. Ternary at 1.1578 sliding bpb (10-minute budget) outperforms binary's best fitting run (F1: 1.1824 at 14L without smear) by 0.025 bpb. Even the over-budget 17L binary run (1.2022) could not match ternary.

The extended 50k-step binary run reaching 1.1239 sliding bpb shows that binary has a competitive convergence ceiling, but it requires approximately 8× more training steps to approach competitive quality — well beyond the competition constraints.

---

## Grouped MLP Investigation

Tested GroupedTernaryLinear: splits MLP into independent groups for parameter/speed savings.

### Real Model Results (relu² 3×, 768d, 600s)

| Run | Config | Layers | ms/step | Steps | val_bpb | Artifact |
|-----|--------|--------|---------|-------|---------|----------|
| F64 | standard | 12 | 99.4 | 6030 | 1.1873 | 17.50MB |
| F72 | g=2 | 12 | 87.4 | 6870 | 1.2180 | 12.97MB |
| F71 | g=4 | 12 | 83.5 | 7190 | 1.2429 | 10.74MB |
| F73 | g=2 | 16 | 114.2 | 5260 | 1.2037 | 16.04MB |
| F74 | swiglu g=2 | 12 | 113.3 | 5300 | 1.2084 | 15.24MB |

Cross-group isolation costs 0.031–0.056 bpb. Even with 4 extra layers (F73), only recovers 0.014 of the deficit. **Not viable for language modelling.**

---

## Differential Attention

Microsoft (2024): computes two attention maps from split Q/K and takes their difference.

| Run | Config | ms/step | Steps | val_bpb |
|-----|--------|---------|-------|---------|
| F64 | standard | 99.4 | 6030 | 1.1873 |
| F68 | diff_attn | 109.3 | 5480 | 1.2094 |

Splits 96-dim heads into 48-dim sub-heads — insufficient dimensionality for meaningful attention patterns at this model scale.

---

## Sequence Refiner (CausalConvRefiner)

| Run | Config | ms/step | Steps | val_bpb | Artifact |
|-----|--------|---------|-------|---------|----------|
| F64 | none | 99.4 | 6030 | 1.1873 | 17.50MB |
| F69 | k=3 | 102.2 | 5860 | 1.1885 | 19.92MB |
| F70 | k=5 | 103.0 | 5820 | 1.2018 | 18.13MB |

Noise-level quality improvement with storage bloat. 12 attention layers already saturate local pattern capture.

---

## ByteCNN Vocabulary Generator

Replaces `nn.Embedding(8192, 256)` with a CNN that generates the embedding matrix from byte spellings.

```
step:500 loss:9.0471 — step:2000 loss:9.0471 (flat, no learning)
```

All 8192 CNN-generated embeddings converge to near-identical vectors at initialisation. The CNN's inductive bias (byte-similar tokens → similar embeddings) destroys the initial diversity needed for gradient signal.

---

## Asymmetric Tokenizer Investigation

8k BPE input with 256-byte output to eliminate large output projection.

| Model | BPB | Notes |
|-------|-----|-------|
| Standard (tied, emb=256) | 3.10 | reference |
| Asymmetric parallel (emb=256) | 8.65 | byte independence assumption fails |
| Asymmetric autoregressive (emb=256) | 8.17 | tiny GRU insufficient capacity |

Multi-byte parallel heads assume conditional independence between bytes within a token — mathematically incorrect. Sequence-length mismatch (7 BPE tokens → 70 bytes) also incompatible with the evaluation framework.

---

## Linear Alternative Exploration

Systematic notebook testing of linear layer alternatives at real model dimensions (768d).

### Projection Benchmark (DIM → DIM, H100)

| Model | Params | ms | vs Linear |
|-------|--------|-----|-----------|
| Linear | 589,824 | 0.07ms | 1.00× |
| LowRank r=64 | 98,304 | 0.03ms | 0.44× |
| BlockDiag b=4 | 147,456 | 0.03ms | 0.40× |
| Grouped g=4 | 147,456 | 0.03ms | 0.40× |
| BD4 + mix32 | 196,608 | 0.07ms | 0.97× |
| Hash 65536 | 65,536 | 0.08ms | 1.13× |

BlockDiag/Grouped offer speed advantages but cross-group isolation degrades LM quality in practice.

---

## H100 Microbenchmark Results

Standalone kernel timing vs torch.compile behaviour (critical lesson: standalone microbenchmarks can mislead when torch.compile fuses operations).

### STE Speed

| Variant | ms/call |
|---------|---------|
| Current | 0.041 |
| Reciprocal | 0.043 |

No gain — 48 STE calls/step = ~2ms overhead (unavoidable).

### Contiguous Checks

Q and K are contiguous after RoPE. V is non-contiguous (view into fused QKV). V's `.contiguous()` costs 0.065ms/call = 0.78ms/step (necessary for flash_attn).

### RoPE Variants

Current (half-split + cat) is fastest at 0.52ms/call.

### Softcap: Poly5 vs Tanh

| Variant | ms/call |
|---------|---------|
| Poly5 (current) | 8.43 |
| Poly3 | 5.98 |
| Tanh | 2.12 |
| Hardtanh | 0.71 |

**Critical finding:** Tanh is 4× faster standalone due to H100 hardware transcendental units. However in the real training loop, torch.compile fuses poly5 with surrounding ops into a single kernel. **Switching to tanh broke fusion — F63 was 16ms/step slower.** Poly5 retained.

### CE + Z-Loss Fusion

| Variant | ms/call (fwd+bwd) |
|---------|-------------------|
| Separate (current) | 16.56 |
| Fused (shared LSE) | 12.33 |

**Same lesson:** 4.2ms saving standalone, but torch.compile already optimises `F.cross_entropy`. Manual gather+logsumexp prevents optimisation. Current approach retained.

---

## Efficiency Analysis

### BPB Gained Per Component

| Component | BPB gain | Source |
|-----------|----------|--------|
| relu → relu² | −0.024 | F55 vs F56 |
| MLP 2× → 3× (relu²) | −0.017 | F56 vs F64 |
| MLP 3× → 4× (relu²) | −0.008 | F64 vs F75 |
| relu² → swiglu (at 3×) | −0.010 | F64 vs F59 |
| +1 layer (average) | −0.0012 | scaling data |
| fp16 → fp8 (RT penalty) | +0.002 | run 42 vs 49 |
| Sliding eval stride=16 | −0.025 | F22 data |
| WD=0.04 vs WD=0 (at 26L) | −0.001 | F7 vs F6 |

### MB Cost Per Component

| Component | MB/layer |
|-----------|----------|
| relu² 2× layer | 0.767 |
| relu² 3× layer | 1.003 |
| relu² 4× layer | 1.220 |
| swiglu 3× layer | 1.357 |
| fp16 → fp8 (fixed saving) | −2.51 |

### Efficiency Ratio (BPB Gained Per MB Spent)

| Change | BPB gain | MB cost | BPB/MB |
|--------|----------|---------|--------|
| relu → relu² | −0.024 | 0.00 | infinite (free) |
| Sliding eval | −0.025 | 0.00 | infinite (free) |
| MLP 2× → 3× | −0.017 | +2.83 (12L) | −0.0060/MB |
| MLP 3× → 4× | −0.008 | +2.83 (12L) | −0.0028/MB |
| relu² → swiglu | −0.010 | +4.25 (12L) | −0.0024/MB |
| +1 layer (relu² 2×) | −0.0012 | +0.767 | −0.0016/MB |
| +1 layer (relu² 3×) | −0.0012 | +1.003 | −0.0012/MB |

MLP 2×→3× is the most efficient paid upgrade. relu² and sliding eval are free wins.

### Layer Budget at 768d

| Config | Max Layers | Est ms/step |
|--------|-----------|-------------|
| relu² 2× fp16 | 14L | ~95ms |
| relu² 2× fp8 | 17L | ~97ms |
| relu² 3× fp16 | 10L | ~99ms |
| relu² 3× fp8 | 13L | ~106ms |
| relu² 4× fp8 | 10L | ~92ms |
| swiglu 3× fp8 | 9L | ~105ms |

---

## Ternary-Incompatible Techniques

These are not merely unhelpful but structurally incompatible with 1.58-bit quantisation:

| Technique | Mechanism of failure |
|-----------|---------------------|
| **EMA** | Weight averaging → values cluster near zero → ternary rounds most to 0 → 0.12 bpb RT gap |
| **TTT-LoRA** | LoRA delta computed outside RMSNorm space that TernaryLinear normalises into. Corrupts calibrated representations at convergence |
| **Ternary prototypes + sigmoid** | Sigmoid membership needs continuous values. Ternary {-1,0,+1} collapses membership patterns → 0.077 RT gap |
| **LM head rank factorisation** | SVD factors U,V need fp16 precision. Storage exceeds original tied embedding |

---

## Software Optimisations

| Optimisation | Saving | Notes |
|---|---|---|
| Fused QKV (c_q+c_k+c_v → single matmul) | ~2ms/step | Safe: in_features divisible by all group sizes |
| Fused SwiGLU/relu² (gate+up → single wide matmul) | ~2-4ms/step | Same params, fewer kernel launches |
| Z-loss regularisation (1e-4 x logsumexp²) | quality | Anchors logits, keeps STE gradients sharp |
| DataLoader int16 transfer (pin then cast on GPU) | ~1ms/step | 4× less PCIe bandwidth |
| FlashAttention-3 | ~13ms/step | ~9% speedup, ~380 free training steps |
| TernaryLinear bf16 weights, cleaner STE | ~1ms/step | Eliminates fp32 roundtrip |
| DDP static_graph + gradient_as_bucket_view | ~1ms/step | Free when find_unused=False |
| Fused optimizer loop (LR set + step in one pass) | ~0.5ms/step | Fewer Python-level iterations |
| Removed CUBLAS determinism tax | ~1ms/step | Not required for competition |
| Temperature grid: 5 points instead of 21 | ~1s total | T=0.90 consistently with relu² |
| Temp scaling moved to eval phase | ~3 steps gained | No longer steals training time |
| `_e()` helper for Hyperparameters | -1.8KB code | Eliminates env var boilerplate |
| 3D tensor ternary quantisation | storage fix | Conv1d weights reshaped to 2D for ternary |

---

## Rejected Techniques (Summary)

| Technique | Reason |
|-----------|--------|
| Tversky (all variants) | Quality-neutral on FineWeb LM — confirmed via synthetic data analysis; speed penalty with relu² |
| Differential attention | Halved head_dim (96→48) degrades quality at this model scale |
| Grouped MLP (g=2, g=4) | Cross-group isolation costs 0.031–0.056 bpb; not recoverable with extra layers |
| CausalConvRefiner | Noise-level quality; storage bloat from Conv1d weights |
| ByteCNN vocabulary generator | Embedding collapse — CNN inductive bias destroys initial diversity |
| Asymmetric tokenizer | Byte independence assumption incorrect; sequence mismatch with eval framework |
| EMA | Incompatible with ternary — weight averaging causes 0.12 bpb RT gap |
| TTT-LoRA | Architectural incompatibility with RMSNorm space in TernaryLinear |
| LM head factorisation | SVD factors bloat artifact beyond budget; unrecoverable quality loss |
| MTP | 0.006 bpb worse — model capacity too limited for auxiliary objectives |
| BigramHash | 0.020 bpb worse at convergence; fp16 table displaces ternary layers |
| Seq/batch schedule | Recompile and step penalties dominate at 600s wallclock |
| SmearModule | +22% step cost for −0.001 gain within ternary 10-minute budget |
| Depth recurrence | Halves effective steps; OOM at DR=3 |
| AdamW for matrix params | Clearly inferior to Muon for ternary weights |
| FP4 storage | 0.026–0.029 RT gap even with QAT — unrecoverable |
| Tanh softcap | Faster standalone but breaks torch.compile kernel fusion |
| Fused CE+Z-loss | Same — breaks compile optimisation |
| 16 heads at 768d | 48-dim head_dim insufficient for meaningful attention |
| relu (plain) | Strictly dominated by relu² |
| leaky relu | Strictly dominated by relu² |
| Distillation (in-run) | Train-from-scratch teacher always worse than supervised |
| reduce-overhead compile | Rotary + embed_proj_rev incompatible with CUDA graphs |
| max-autotune compile | 30+ minute kernel search prohibitive for 600s runs |
| Skip weights zero-init | 0.010 bpb worse — decoder needs skip signal from step 0 |
| EMBED_DIM=0 (full 512) | 19.78MB artifact — 3.78MB over budget |
| Untie lm_head full-rank | 7.3MB budget overrun not justified by 0.005 bpb gain |

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| 8k vocabulary | −0.42 bpb, largest single win |
| relu² activation | −0.024 bpb vs relu, free (no cost) |
| 4×MLP width | Best BPB within budget at 10L; 0.008 better than 3× |
| 10L 768d | Minimum viable depth at 768d with maximum MLP width |
| WD=0.0 at 10L 4× | Opposite to deep models — wider MLP needs full weight freedom |
| fp8 storage | Halves fp_params (5MB→2.5MB), enables wider MLP within budget |
| EMBED_DIM=254 | 256-2 dims to fit artifact+code under 16,000,000 byte budget; ~0.0004 bpb cost |
| BITNET_GROUP_SIZE=128 | Same quality as 64; saves 0.69MB |
| 8 heads, 4 KV, 96-dim head_dim | 16h at 48-dim insufficient; MHA only +0.0012 at +1.5MB |
| Poly softcap | Fuses with torch.compile; tanh breaks fusion |
| ROPE_BASE=5000 + YaRN 2048 | Best frequency calibration |
| Muon optimizer | Newton-Schulz normalisation compensates for ternary STE gradient attenuation |
| MUON_BACKEND_STEPS=3 | Equivalent to 5 at convergence; +190 extra steps |
| MUON_MOMENTUM=0.95 | Both directions degrade; affects artifact via zero_frac |
| WARMDOWN=20% | Asymmetric — too little hurts more than too much |
| MATRIX_LR=0.04 | Higher LR compensates for ternary STE gradient attenuation |
| SCALAR_LR=0.02 | Optimal — scalars do not pass through STE |
| TIED_EMBED_LR=0.02 | Optimal |
| TRAIN_BATCH_TOKENS=524k | Optimal tradeoff between gradient quality and step count |
| Base-3 + LZMA | 39% reduction over int8+zlib |
| Shrinkage fix | Eliminates all RT gaps universally |
| Skip weights ones-init | Decoder needs skip signal from step 0; zeros costs 0.010 bpb |
| Tied embeddings | Untie costs 7.3MB; not justified |
| Standard attn projection | Tversky quality-neutral; grouped destroys quality |
| No EMA | Fundamentally incompatible with ternary |
| No TTT | RMSNorm space incompatibility confirmed across 6 runs |
| No MTP | Confirmed post-fix: 0.006 bpb worse |
| Temperature scaling T=0.90 | relu² logits slightly underconfident; auto-calibrated |
| Fused QKV + relu² | ~130-180 free training steps per run |
| Z-loss regularisation | Anchors logits; keeps STE gradients sharp |
| FlashAttention-3 | Free ~380 extra training steps per 600s run |
| Sliding eval stride=16 | Best quality when eval budget unconstrained |
| Optimizer coverage fix | embed_proj/embed_proj_rev now train; +0.055 bpb improvement |
| MAX_WALLCLOCK_SECONDS=599 | 1s leeway for safety margin |
| Binary 15L 768d 4× fp8 | 97M params in 15.67MB — maximum parameter density; convergence ceiling validated at 50k steps |
