# Parameter Golf — SOTA Analysis

Analysis of top 6-8 leaderboard submissions (March 31 — April 9, 2026).
Current SOTA: **1.0810 bpb** (8×H100, 600s).

---

## Top Submissions Leaderboard

| Date | Submission | val_bpb | Key Innovation |
|------|-----------|---------|----------------|
| 04-09 | SP8192_3LayerRecur_ParResid_QK525_LegalTTT | **1.0810** | 3-layer recurrence + TTT |
| 04-08 | SP8192_ParallelResid_ScoreFirstTTT | 1.0822 | Parallel residuals + TTT |
| 04-06 | SP8192_QK5_LegalTTT | 1.0828 | QK gain tuning + TTT |
| 04-06 | SP8192_HessianSDClip_ProgressiveRecurrence | 1.0835 | Hessian-aware clipping |
| 04-05 | SP8192_GPTQ-Embeddings_SDClip | 1.0856 | GPTQ baseline for SP8192 |
| 04-04 | SP4096_DepthRecurrence_ParallelResid | 1.0897 | SP4096 variant |
| 04-03 | MuonEqR_DepthRecurrence_WD090 | 1.0912 | Weight decay tuning |
| 03-31 | ParallelResiduals_MiniDepthRecurrence | 1.1063 | Parallel residuals |

---

## Universal Techniques (ALL top submissions use these)

### 1. SP8192 Tokenizer
- SentencePiece BPE with vocab=8192
- All top-3 use SP8192; SP4096 is ~0.02 bpb worse
- Requires generating custom tokenizer + retokenizing FineWeb

### 2. Architecture: 11L × 512d × 8H/4KV, MLP 4×
- 11 physical layers (not baseline's 9)
- mlp_mult=4 (not baseline's 2) — wider MLP stores more knowledge
- Same model_dim=512, num_heads=8, num_kv_heads=4
- Tied embeddings, logit_softcap=30

### 3. Depth Recurrence on Layers 4,5
- 11 physical layers → 13 virtual passes (layers 4,5 repeated once)
- Activated mid-training at ~35-50% of steps (not from step 0)
- Best submission uses 3-layer recurrence (layers 3,4,5)

### 4. GPTQ INT6 Quantization
- Full-Hessian GPTQ (not basic round-to-nearest INT8)
- INT6 for weight matrices (k=12.85 clip)
- INT8 for embeddings
- SDClip: clip = k × std(row) for principled rate-distortion
- Combined with Brotli-11 compression (not zlib)

### 5. MuonEq-R Optimizer
- Row-normalized variant of Muon
- AdamW only for embeddings and scalar parameters
- Weight decay 0.085-0.095 (much higher than default)

### 6. LeakyReLU(0.5)²
- Replace relu(x).square() with leaky_relu(x, 0.5).square()
- Allows small negative gradients through → better early convergence

### 7. Partial RoPE (16/64 dims)
- Only apply rotary embeddings to 16 of 64 head dimensions
- Remaining 48 dims have no position encoding
- Saves compute and reportedly improves quality

### 8. Brotli-11 Compression
- Replace zlib with Brotli at compression level 11
- Better compression ratio → more room in 16 MB budget

---

## Common Techniques (4+ of top 6)

### 9. Parallel Residuals (layer 7+)
- GPT-J style: attention and MLP read from same pre-residual input
- Activated from physical layer 7 onward
- Learned per-layer routing weights
- Consistent +0.002-0.004 bpb improvement

### 10. EMA Weights (decay ~0.997)
- Exponential moving average of weights during training
- Use EMA weights at save/eval time
- Almost free improvement (~0.002 bpb)

### 11. QK Gain 5.0-5.25
- Higher than baseline's 1.5, higher than our best (4.0)
- May need more training steps + other techniques to benefit
- Top submission uses 5.25

### 12. Warmdown ~72% of training
- Much longer warmdown than baseline's ~6% (1200/20000)
- Cosine decay over final 72% of steps

---

## Top 3 Only

### 13. Legal Score-First TTT
- Test-time training: adapt model on already-scored val tokens at eval time
- SGD lr=0.005, momentum=0.9, 3 epochs per 32K-token chunk
- Score all windows first (torch.no_grad), then train on scored tokens
- Adds ~0.002 bpb improvement
- Within 600s eval budget

---

## Our Current Best vs SOTA (updated April 18, 2026)

**Exp 21: 1.1963 INT8 val_bpb** (2×H100, 40min, 9511 steps)
**Projected 8×H100 10min: ~1.207** (estimated ~8000 steps with proper warmdown)

```
                    Our Best (Exp 21)    SOTA (8×H100)     Status
val_bpb             1.1963 (2×40min)    1.0810
                    ~1.207 (8×10min est)
Tokenizer           SP1024              SP8192             ❌ need SP8192 data pipeline
Layers              9                   11                 ❌ needs INT6 to fit in 16MB
MLP mult            2                   4                  ❌ needs INT6 to fit in 16MB
Recurrence          layers 3,4 staged   layers 3,4,5       ✅ implemented (staged costs 77s recompile)
Quantization        INT8 + zlib         INT6 GPTQ + Brotli ❌ major implementation effort
Optimizer           Muon                MuonEq-R           ❌ ~30 lines code change
Weight decay        default             0.090-0.095        ⏳ just env var
Activation          LeakyReLU(0.5)²     LeakyReLU(0.5)²    ✅ implemented
RoPE               partial (16/64)      partial (16/64)     ✅ implemented
Parallel residuals  yes (layer 7+)      yes (layer 7+)      ✅ implemented
EMA                 no                  yes (~0.997)        ⏳ ~20 lines (broken w/ INT8, needs GPTQ)
QK gain             4.0                 5.0-5.25            ✅ (4.0 optimal for our config)
TTT                 no                  yes                 ❌ ~50 lines code, eval-time only
seq_len             2048                2048                ✅ implemented

Implemented: 7/13     ✅✅✅✅✅✅✅ (added partial RoPE since last update)
Easy to add:  1/13    ⏳ (weight decay)
Medium:       1/13    ⏳ (EMA — needs GPTQ first to quantize properly)
Hard:         4/13    ❌❌❌❌ (SP8192, INT6/GPTQ, MuonEq-R, TTT)
```

### Progress Tracker

| Technique | Impact | Status | Notes |
|-----------|--------|--------|-------|
| seq_len=2048 | -0.015 | ✅ done | biggest single lever |
| LeakyReLU(0.5)² | ~-0.005 | ✅ done | stable structural win |
| Parallel residuals | ~-0.004 | ✅ done | layer 7+, GPT-J style |
| Depth recurrence | ~-0.004 | ✅ done | layers 3,4 |
| Staged recurrence | ~-0.002 | ✅ done | costs 77s recompile |
| q_gain=4.0 | ~-0.004 | ✅ done | optimal for our config |
| Partial RoPE 16/64 | ~-0.001 | ✅ done | hard to isolate, confirmed in Exp 21 |
| Weight decay 0.09 | ~-0.002 | ⏳ next | env var only |
| EMA weights | ~-0.003 | ⏳ blocked | needs GPTQ — EMA breaks INT8 |
| SP8192 tokenizer | ~-0.03-0.05 | ❌ blocked | needs data pipeline, biggest remaining lever |
| GPTQ INT6 | ~-0.02-0.04 | ❌ blocked | enables 11L+MLP4x, unlocks EMA |
| MuonEq-R | ~-0.005-0.01 | ❌ todo | row-normalized Muon |
| TTT | ~-0.002 | ❌ last | eval-time only |

### Current Status (April 18, 2026)

We're now running the SOTA code directly. Exp 24 matched SOTA pre-quant (1.0867 vs 1.0873).
Working from the SOTA codebase, not our own 17M model.

### Gap Analysis (from Exp 24 on 2×H100)

```
Exp 24 (no sliding/TTT):  1.0985
SOTA (with sliding+TTT):  1.0810
Gap:                       0.018

Easy gains:
  Enable sliding window:   ~-0.012 (free, just eval-time)
  Enable TTT:              ~-0.005 (free, just eval-time)
  Total:                   ~-0.017 → ~1.082 (match SOTA)
```

### Techniques from Other Submissions NOT in Current SOTA

Surveyed all records in `track_10min_16mb/2026-03-*` and `2026-04-*`.

#### Training-time techniques (need retraining)

| Technique | Source | Impact | Cost | What it does |
|---|---|---|---|---|
| Progressive recurrence | Apr 6 HessianSDClip (1.0835) | reduces loss spike | zero | Fractional activation instead of hard on/off at 35% |
| DISABLE_LAYER0_ATTN | Mar 31 ParallelResid (1.1063) | skip useless attn | zero | First-layer attention is mostly noise, skip it |
| REPEAT_UNTIE_MLP | Mar 31 ParallelResid (1.1063) | more params | more params | Untied MLP weights in recurring layers |
| BigramHash 3072x112 | Mar 25 ValCalib | proven +0.005 | +params, fits 16MB | Cheap n-gram feature embeddings |
| YaRN RoPE | Mar 24 Ternary UNet | better pos encoding | small code | Scaled RoPE for longer contexts |
| Layer-wise LR decay | LLM community | unknown | zero | Different learning rates per layer depth |
| Differential attention | Microsoft paper | unknown | small code | Learned attention subtraction |
| SWA + EMA combo | Mar 25 ValCalib | better averaging | zero | SWA every 50 steps on top of EMA |

#### Post-training techniques (testable on checkpoint, no retraining)

| Technique | Source | Impact | Cost | What it does |
|---|---|---|---|---|
| Hessian-aware SDClip (lambda=0.175) | Apr 6 HessianSDClip (1.0835) | better quant | zero | Use Hessian to weight clip importance per row |
| Per-group clip allocation | Apr 6 (suggested) | non-uniform quant | small | Important layers get more bits (INT8), others INT6 |
| AR Self-Gen GPTQ | Mar 25 ValCalib | better calibration | zero | Model generates own data for GPTQ calibration |

### Submission Plan

1. Enable sliding window + TTT → should match ~1.08
2. Try zero-cost improvements (progressive recurrence, Hessian SDClip)
3. Try BigramHash (proven +0.005)
4. Run on 8×H100 10min for valid submission
5. Target: beat merged SOTA (1.1147) by >0.005 nats — **already achieved** (1.0985 < 1.1097)
