# Parameter Golf Competition — Final State (May 4, 2026)
# Compiled from GitHub PRs post-competition close (April 30, 2026)

## Competition Timeline
- Start: March 18, 2026
- **End: April 30, 2026 5:00 PM Pacific** (CLOSED)
- Grace policy: PRs opened before cutoff, results added after, still count

## Official Leaderboard (Final, from PR #2146 audit)

| Rank | PR | BPB | Key Techniques |
|------|----|-----|----------------|
| **1** | **#2135** | **1.05651** | PR#2130 arch + GPTQ_CALIBRATION_BATCHES=32 + TTT |
| 2 | #2014 | 1.05759 | Progressive context + short-doc TTT |
| 3 | #1953 | 1.05855 | 2560 eval seqlen + no_qv TTT mask + QK_GAIN=5.25 |
| 4 | #1945 | 1.05943 | PR#1855 + AWQ-lite + AsymLogit Rescale |
| 5 | **#1855** (merged) | **1.06108** | SP8192 + LQER + Sparse Attn Gate + SmearGate (official) |
| 6 | #1868 (merged) | 1.06141 | SmearGate BOS fix compliance re-run |
| 7 | #1851 (merged) | 1.06128 | SmearGate BOS Fix + Phased TTT |
| — | Our v9 | 1.1194 | Batched Muon + Full GPTQ + random calib |

**Our gap from SOTA:** 0.063 BPB. Large, but understanding the techniques closes the gap.

## What PR #2130 Was (and Why It Was Invalidated)

PR #2130 was our base architecture. It was invalidated because:
> "PR #2018 and PR #2130: invalid due train/validation document overlap in the submitted CaseOps data construction."

The bug: their CaseOps data had overlapping documents between train and validation splits. Our v9 (which was based on PR #2130's techniques but using clean FineWeb data) is unaffected.

---

## The Winning Technique Stack (PR #2135 lineage)

### Layer 1: Base Architecture (PR #1855)
```
11L 512d transformer
8H/4KV GQA
U-Net skips
Parallel residuals (layers 8+)
Partial RoPE
Fused LeakyReLU² MLP
SP8192 (SparsePrecision? SuperPosition? 8192-dim vocab expansion?)
CaseOps tokenizer
```

### Layer 2: Training Innovations
```
Polar Express Newton-Schulz Muon optimizer
Phased TTT (score-first, 3 phases at doc boundaries 833/1666/2500)
Legal TTT: score chunk FIRST, then train on it
Progressive training context: TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000
```

### Layer 3: Quantization Stack (the compression magic)
```
GPTQ int6 weights
GPTQ int7 embeddings
LQER asymmetric rank-4 (Low-rank Quantization Error Recovery)
AWQ-lite (Activation-aware Weight Quantization, lightweight version)
Asymmetric Logit Rescale (softcap_pos ≠ softcap_neg)
GPTQ_CALIBRATION_BATCHES=32 (critical: 32 vs 16 gives ~0.004 BPB)
Per-row int8 attn-gate
```

### Layer 4: Compression (artifact fitting)
```
Per-group lrzip + brotli compression
lrzip -z -L 9 (ZPAQ context-mixing encoder)
Hot groups: L1 similarity sort before compressing
Result: ~280KB smaller than plain brotli
```

### Layer 5: Attention Architecture
```
Sparse attention head-output gate
SmearGate (per-token forward mixing: x[:, 1:] + g * x[:, :-1])
BOS fix: mask the mixing where current token is BOS
```

---

## Key Numerical Values for Final SOTA

| Metric | Value |
|--------|-------|
| Train steps (600s) | ~4994 steps |
| ms/step | ~120ms |
| Pre-quant BPB | ~1.061 |
| Post-quant BPB | ~1.069 |
| Post-TTT BPB | **1.057** |
| TTT BPB gain | ~0.012 |
| GPTQ 32 vs 16 batches gain | ~0.004 |
| Artifact size | 15.95MB |

---

## Critical Discoveries from Competition Analysis

### 1. GPTQ Calibration Batches: Diminishing Returns but Huge Initial Gain
- 16 batches: baseline
- 32 batches: **-0.004 BPB** (!)
- The more calibration data, the better the quantization
- Worth trying 64/128 batches (may diminish after 32)

### 2. TTT Is Worth 0.012 BPB
- Pre-TTT quant: ~1.069
- Post-TTT: ~1.057
- Score-first TTT gives ~0.012 BPB improvement
- Our v9 TTT was broken (torch.compile issue); this is a big loss

### 3. AWQ-lite vs GPTQ
- AWQ-lite (Activation-aware): weights scaled by activation magnitude before quant
- More robust than GPTQ for long-tail distributions
- LQER: low-rank adapter to absorb quantization error (rank-4 is enough)

### 4. CaseOps Tokenizer
- Case-preserving byte encoding
- More tokens per document than plain byte-level
- SP8192 likely means "softmax partition 8192" — enlarging effective vocab

### 5. Progressive Context Schedule
```
TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000
```
- Start with short context (1024) for first 10% of steps
- Scale to medium (2048) for most training
- Finish at 3072 for last 30%
- More efficient early training + better late generalization

### 6. Phased TTT Details
- 3 phases, triggered at document boundaries
- Prefix docs: gradient=0 (scoring only)
- Suffix docs: gradient=1 (learning)
- LoRA rank: 1-4 for K and V matrices
- Local LR multiplier: 0.75 works better than 1.0

### 7. What We Missed
- **SmearGate** (+0.003 BPB) — simple gating that mixes adjacent tokens
- **AWQ-lite** — better quant than our row-max approach  
- **LQER** — low-rank quant error correction
- **Progressive context** — cheap efficiency gain
- **CaseOps tokenizer** — the base data pipeline everyone converged on

---

## Comparison: Our v9 vs Final SOTA

| Component | Our v9 | Final SOTA (PR #2135) |
|-----------|--------|----------------------|
| Architecture | PR #2130 (invalidated, but same baseline) | PR #1855 lineage |
| BPB | 1.1194 | **1.05651** |
| Steps | ~4450 | ~4994 |
| ms/step | ~134ms | ~120ms |
| TTT | Broken (compile issue) | Working, +0.012 BPB |
| GPTQ batches | Random calib | 32 batches |
| Compression | lzma | lrzip+brotli per-group |
| Tokenizer | Standard | CaseOps |
| SmearGate | No | Yes |
| AWQ-lite | No | Yes |
| LQER | No | Yes |

**Speed gap:** 14ms/step. Extra 544 steps × learning = ~0.002 BPB.
**TTT gap:** 0.012 BPB  
**Quantization gap:** 0.008 BPB (AWQ + LQER + 32-batch GPTQ vs our random calib)
**Architecture gap (SmearGate, etc.):** ~0.005 BPB
**Tokenizer gap (CaseOps):** Unknown but substantial

---

## Kernel Findings Relevant to Competition

### Our Mega-Kernel's Role

If our fused kernels reduce step time from 120ms to ~117ms (2.5% speedup):
- 600s ÷ 117ms = 5128 steps vs 5000 baseline → +128 extra steps
- At the rate of final-phase learning (~0.00003 BPB/step): +0.004 BPB improvement
- That's equivalent to the GPTQ 32-vs-16 batch gain

**The mega-kernel could be worth up to 0.004 BPB if it achieves 2.5% speedup.**

### H100 Baseline Timings (from our autotune run, May 4 2026)
```
M=73728 (competition-realistic):
  MLP (K=512, N=1536):  0.760ms  ← cuBLAS reference
  QKV (K=512, N=Q+K+V):  0.272ms ← cuBLAS reference
```

### Expected Savings from Kernel Fusion
Eliminating 1 normed_x tensor (75MB write + 75MB read) at H100's 3.35 TB/s:
- Time saved: 0.150GB ÷ 3.35TB/s = 0.000045s = 0.045ms per layer
- 22 layers × fwd+bwd: ~2ms per step
- At 120ms/step: ~1.7% speedup → ~+85 steps in 600s
