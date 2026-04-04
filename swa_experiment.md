# Sliding Window Attention Experiment

## Hypothesis

Replace full causal attention with sliding window attention (window_size=256) on early layers (0-7), keeping full attention on deep layers (8-10). This should reduce per-step compute, giving more training steps in the 600-second budget, without significantly hurting model quality since early layers primarily learn local patterns.

## What was changed

**Base:** Current #1 submission (`2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`, val_bpb 1.1147)

**File:** `train_gpt_swa.py`

**Changes (6 total, all minimal):**

1. **Hyperparameters** — added `SWA_WINDOW_SIZE=256` and `SWA_FULL_ATTN_LAYERS=3`
2. **CausalSelfAttention.__init__** — added `self.window_size = (-1, -1)` default attribute
3. **CausalSelfAttention.forward** — pass `window_size=self.window_size` to `flash_attn_3_func`
4. **GPT.__init__** — set `window_size=(256, 256)` on layers 0-7, leave (-1,-1) on layers 8-10
5. **Logging** — print which layers use sliding window at startup
6. **Eval model** — disable sliding window on the eval model (full attention for eval)
7. **FA import** — added fallback chain (FA3 standalone → FA3 bundled → FA2)

**Zero changes to:** optimizer, quantization, GPTQ, parameter banks, Parallel Muon, or any other component.

## Environment

### Run 1: torch 2.6 + FA2 (coderkhatana)

- **Hardware:** 8x NVIDIA H100 80GB HBM3
- **PyTorch:** 2.6.0+cu126
- **flash-attn:** 2.8.3 (FA2 — FA3 not available via pip flash-attn with torch 2.6)
- **Attention backend:** FA2

### Run 2: latest torch + FA3 (itssshikhar)

- **Hardware:** 8x NVIDIA H100 80GB HBM3
- **PyTorch:** 2.11.0+cu128
- **flash-attn:** FA3 standalone via `flash_attn_3` wheel
- **Attention backend:** FA3

## Configuration

Only env var overrides (everything else uses code defaults which match #1 submission):

```bash
RUN_ID=swa_w256_full3
BIGRAM_VOCAB_SIZE=3072
BIGRAM_DIM=112
WARMDOWN_ITERS=4000
SWA_WINDOW_SIZE=256
SWA_FULL_ATTN_LAYERS=3
```

Effective config: layers 0-7 use window=256 sliding attention, layers 8-10 use full causal attention. All 11 layers have XSA enabled.

## Run 1 Results (torch 2.6 + FA2)

Training completed but eval timed out before producing final numbers.

| Metric | Value |
|---|---|
| Steps completed | 6889 / 20000 (wallclock capped) |
| step_avg | 87.11 ms |
| Post-EMA val_bpb | 1.1372 |
| Int6 roundtrip | (eval timed out) |
| Artifact size | 15,914,823 bytes |

## Run 2 Results (latest torch + FA3)

| Metric | Value |
|---|---|
| Steps completed | 7305 / 20000 (wallclock capped) |
| step_avg | **82.15 ms** |
| Peak GPU memory | 23,037 MiB / 81,559 MiB per GPU |
| model_params | 27,067,484 |

### Validation scores

| Eval stage | val_loss | val_bpb |
|---|---|---|
| step 0 (init) | 6.9301 | 4.1044 |
| step 4000 (mid-train) | 2.0490 | 1.2135 |
| step 7305 (wallclock stop) | 1.9198 | 1.1370 |
| Post-EMA | 1.9182 | 1.1361 |
| **Int6 roundtrip** | **4.2835** | **2.5369** |
| **Sliding window eval** | **6.0174** | **3.5639** |

### Training loss trajectory

| Step | train_loss | train_time | step_avg |
|---|---|---|---|
| 1 | 6.9307 | 216ms | 216.01ms |
| 500 | 2.3714 | 40,731ms | 81.46ms |
| 1000 | 2.2520 | 81,729ms | 81.73ms |
| 2000 | 2.0537 | 163,460ms | 81.73ms |
| 3000 | 2.1450 | 245,305ms | 81.77ms |
| 4000 | 1.9601 | 327,242ms | 81.81ms |
| 5000 | 2.0943 | 409,149ms | 81.83ms |
| 6000 | 1.9407 | 491,018ms | 81.84ms |
| 7000 | 1.7935 | 574,117ms | 82.02ms |
| 7305 | -- | 600,074ms | 82.15ms |

### Submission size

| Component | Size |
|---|---|
| Serialized model (raw) | 106,289,590 bytes |
| Code | 103,819 bytes |
| Model int6+lzma | 15,756,668 bytes |
| **Total submission** | **15,860,487 bytes** |

### GPTQ calibration

- AR self-gen: 64 seqs x 2048 tokens, temp=0.8, completed in 287.9s
- Hessians collected for 68 layers
- Selective pruning: not needed (already fits under 15.9MB)

## Comparison with current #1

| Metric | SWA (run 2) | Current #1 (no SWA) |
|---|---|---|
| step_avg | **82.15ms** | 86.6ms |
| Steps in 600s | **7305** | 6927 |
| Post-EMA val_bpb | 1.1361 | 1.1354 |
| **Int6 roundtrip val_bpb** | **2.5369 (BROKEN)** | ~1.15 |
| **Sliding eval val_bpb** | **3.5639 (BROKEN)** | **1.1147** |

## Analysis

### What went right

1. **step_avg improved significantly:** 82.15ms vs 86.6ms — **5.1% faster** per step. This is real: sliding window attention at window=256 on 8 of 11 layers saves compute.
2. **More training steps:** 7305 vs 6927 — **378 extra steps** in the same 600 seconds.
3. **Pre-quant quality is competitive:** Post-EMA 1.1361 vs 1.1354 — only 0.0007 BPB worse despite the architectural change. The model learns nearly as well with local attention on early layers.

### What went catastrophically wrong

**The quantization roundtrip is completely broken.** Pre-quant val_bpb (1.1361) → post-quant val_bpb (2.5369) is a **1.4 BPB gap**. Normal int6 quantization gap is ~0.01 BPB. This is not quantization error — this is a **bug**.

### Likely root cause

The eval model is created with the same `GPT(...)` constructor, which reads `SWA_WINDOW_SIZE` and `SWA_FULL_ATTN_LAYERS` from env vars and sets `window_size` on early layers' attention. We then override this:

```python
# line 2108-2110 in train_gpt_swa.py
for block in eval_model.blocks:
    block.attn.window_size = (-1, -1)
```

But the **Hessian collection model** (`_HessianAttn`) does NOT have a `window_size` attribute at all — it always uses full attention in its `flash_attn_3_func(q, k, v, causal=True)` call. This creates a **mismatch**: the GPTQ Hessians are collected with full attention, but the training model was trained with sliding window attention on layers 0-7. The Hessians don't represent the true activation distribution of the training model, so GPTQ optimizes for the wrong target.

Additionally, there could be a state dict key mismatch: if `window_size` is somehow being saved/restored in the state dict (it shouldn't be — it's a plain Python attribute, not an `nn.Parameter`), the unbank/rebank pipeline might be corrupted.

### Other possible causes

1. **torch.compile + window_size interaction:** `torch.compile(fullgraph=True)` traces the model once and bakes in the `window_size` value. If the eval compiled model was traced with sliding window but then we set `window_size=(-1,-1)` after compilation, the compiled graph still uses the old value.
2. **FA3 `window_size` behavior difference:** FA3's sliding window might interact differently with the causal mask than expected, causing the quantized model to produce different attention patterns.

### Conclusion

Sliding window attention **does** speed up training (82ms vs 86ms, 5% faster). Pre-quant model quality is nearly identical. But the quantization pipeline is broken with this change — the GPTQ calibration and/or the eval model loading doesn't handle the window_size attribute correctly.

**The experiment is promising for training speed but needs a bug fix in the quantization/eval pipeline before the results are meaningful.**

## Run 3: torch 2.9.1 + FA3 (same version as current #1)

To rule out torch version as the cause, re-ran with torch 2.9.1+cu128 — the exact version the current #1 submission was tested on.

| Metric | Value |
|---|---|
| Steps completed | 7915 / 20000 (wallclock capped) |
| step_avg | **75.82 ms** |
| Peak GPU memory | 22,845 MiB per GPU |

### Validation scores

| Eval stage | val_loss | val_bpb |
|---|---|---|
| step 4000 (mid-train) | 2.0618 | 1.2211 |
| step 7915 (wallclock stop) | 1.9166 | 1.1351 |
| Post-EMA | 1.9145 | **1.1339** |
| **Int6 roundtrip** | **4.1738** | **2.4720 (BROKEN)** |
| **Sliding window eval** | **5.9307** | **3.5125 (BROKEN)** |

### Training loss trajectory

| Step | train_loss | step_avg |
|---|---|---|
| 500 | 2.3715 | 73.40ms |
| 1000 | 2.2543 | 73.84ms |
| 2000 | 2.0496 | 74.16ms |
| 3000 | 2.1463 | 74.34ms |
| 4000 | 1.9760 | 74.40ms |
| 5000 | 2.1088 | 74.45ms |
| 6000 | 1.9593 | 74.47ms |
| 7000 | 1.8220 | 74.47ms |
| 7915 | — | 75.82ms |

### Comparison — all runs

| Run | Torch | step_avg | Steps | Post-EMA bpb | Roundtrip bpb |
|---|---|---|---|---|---|
| Run 1 (FA2) | 2.6.0 | 87.11ms | 6889 | 1.1372 | (timed out) |
| Run 2 (FA3) | 2.11.0 | 82.15ms | 7305 | 1.1361 | **2.5369 BROKEN** |
| **Run 3 (FA3)** | **2.9.1** | **75.82ms** | **7915** | **1.1339** | **2.4720 BROKEN** |
| Current #1 (no SWA) | 2.9.1 | 86.6ms | 6927 | 1.1354 | ~1.15 OK |

### Key findings

1. **Torch version is NOT the cause.** The bug persists on torch 2.9.1 — the same version the current #1 was tested on.

2. **step_avg improved dramatically.** 75.82ms vs 86.6ms — **12.5% faster.** Torch 2.9.1 + FA3 + sliding window gives the best per-step performance of any variant we've tested. 7915 steps vs 6927 — nearly 1000 extra steps.

3. **Pre-quant quality is excellent.** Post-EMA 1.1339 is actually **better** than the current #1's 1.1354 (0.0015 BPB better) thanks to the extra training steps.

4. **The bug is 100% in the GPTQ pipeline, not in training or sliding window.**

## Debugging: local roundtrip tests

### Test 1: Simple int8 roundtrip (debug_roundtrip.py)

Tested bank → int8 quantize → dequantize → eval on local 4050 GPU with 4-layer model. All three scenarios passed (gap < 0.01):

| Scenario | Gap |
|---|---|
| No SWA → No SWA | -0.0015 |
| SWA → SWA | +0.0110 |
| SWA → No SWA (override) | -0.0032 |

**Conclusion:** Int8 roundtrip works fine. Sliding window + override is not the issue.

### Test 2: Full int6 GPTQ-lite roundtrip (debug_roundtrip_full.py)

Tested bank → unbank → int6 GPTQ-lite → LZMA → dequant → rebank → eval + torch.compile on local 4050 GPU with 11-layer banked model. All three scenarios passed (gap < 0.01):

| Scenario | Gap |
|---|---|
| No SWA → No SWA | +0.0035 |
| SWA → SWA | +0.0078 |
| SWA → No SWA (override) | +0.0005 |

**Conclusion:** The full pipeline (including parameter banks, unbank/rebank, int6 quantization, LZMA, torch.compile) works perfectly.

### Root cause identified

The local tests used **GPTQ-lite** (percentile search, no Hessians). The H100 runs used **Full Hessian GPTQ** (Cholesky error compensation, column reordering, AR self-generated calibration).

The `_HessianGPT` model used for GPTQ calibration uses `_HessianAttn`, which always does full causal attention — it has no `window_size` parameter. But the training model used sliding window on layers 0-7. This means:

1. Training produces weights optimized for local attention patterns (window=256)
2. GPTQ collects Hessians H = X^T @ X using full attention
3. The Hessians reflect a different activation distribution than what the weights were trained for
4. Full Hessian GPTQ uses these wrong Hessians for Cholesky decomposition and cross-column error compensation
5. The quantization "compensates" errors based on wrong assumptions about weight importance
6. Result: catastrophically wrong quantized weights

GPTQ-lite doesn't have this problem because it doesn't use Hessians — it just searches for the best clipping percentile per row, which is independent of the attention pattern.

### Fix needed

Add `window_size` support to `_HessianAttn` so the GPTQ calibration model matches the training model's attention behavior. The Hessians must be collected under the same attention pattern that produced the weights.

```python
# In _HessianAttn.__init__:
self.window_size = (-1, -1)  # set by GPT constructor based on layer index

# In _HessianAttn.forward:
y = flash_attn_3_func(q, k, v, causal=True, window_size=self.window_size)
```

Then in the Hessian GPT constructor, set `window_size` on early layers to match training config.

## Debugging the roundtrip bug

### Fix attempt 1: Add window_size to _HessianAttn (FAILED)

Added `self.window_size` to `_HessianAttn` and `_HessianGPT`, matching the training model's sliding window config so Hessians are collected under the same attention pattern.

Result: **Still broken** (2.3935 BPB roundtrip). The Hessian mismatch was not the cause.

### Fix attempt 2: Skip Full Hessian GPTQ, use GPTQ-lite (FAILED)

Set `hessians=None` in `mixed_quantize_int6` to force GPTQ-lite (percentile search, no Hessians). This completely bypasses the Hessian-based quantization path.

Result: **Still broken** (2.3839 BPB roundtrip). The quantization METHOD is not the cause — both Full Hessian GPTQ and GPTQ-lite produce the same catastrophic failure.

Note: Had to verify with a `DEBUG:quantizing with hessians=None` log print that the correct file was deployed, because Modal's `add_local_file` caches aggressively.

### Fix attempt 3: Remove window_size override on eval model (FIXED)

The eval model was constructed with sliding window (from env vars), then we overrode `window_size=(-1,-1)` on all layers before `torch.compile`:

```python
# This was the bug:
for block in eval_model.blocks:
    block.attn.window_size = (-1, -1)
compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
```

Removed the override — let the eval model keep the same sliding window config as training.

Result: **FIXED.** Roundtrip val_bpb = 1.1411 (normal 0.008 gap from pre-quant 1.1332).

### Root cause

`torch.compile(dynamic=False, fullgraph=True)` on torch 2.9.1 captures the `window_size` attribute at model construction time, not at the time of the Python override. The compiled graph bakes in the constructor's `window_size=(256,256)` for layers 0-7. Setting `window_size=(-1,-1)` in Python after construction has no effect on the compiled graph.

This means the eval model was running with `window_size=(256,256)` on layers 0-7 (sliding window) despite us setting `(-1,-1)` (full attention). But the quantized weights were loaded correctly — they're from a model trained with sliding window, evaluated with sliding window. The mismatch was that we THOUGHT we were evaluating with full attention but were actually evaluating with sliding window through the compiled graph.

Wait — if the compiled graph was using sliding window (same as training), why was the roundtrip broken? Because the DIAGNOSTIC eval used the TRAINING compiled model (compiled during warmup, with sliding window baked in correctly). The roundtrip eval used a FRESHLY compiled eval model where `torch.compile` might have captured an inconsistent state — the model was constructed with window=(256,256), then overridden to (-1,-1), and `torch.compile` may have captured a partially-overridden state or triggered a different code path.

The real fix is simple: **don't override `window_size` after construction.** If the eval model should use the same attention as training (which it should — the weights were trained with it), just let it be.

## Run 5: Fixed — torch 2.9.1 + FA3 + no window_size override

### Environment

- **Hardware:** 8x NVIDIA H100 80GB HBM3
- **PyTorch:** 2.9.1+cu128
- **flash-attn:** FA3 standalone via `flash_attn_3` wheel
- **Quantization:** GPTQ-lite (hessians=None — debug mode, not optimal)

### Results

| Metric | Value |
|---|---|
| Steps completed | **7965** / 20000 (wallclock capped) |
| step_avg | **75.34 ms** |
| Peak GPU memory | ~22,800 MiB per GPU |

#### Validation scores

| Eval stage | val_loss | val_bpb |
|---|---|---|
| step 0 (init) | 6.9301 | 4.1044 |
| step 4000 (mid-train) | — | 1.2142 |
| step 7965 (wallclock stop) | 1.9149 | 1.1341 |
| Post-EMA | 1.9134 | **1.1332** |
| Int6 roundtrip (GPTQ-lite) | 1.9267 | **1.1411** |
| **Sliding window eval** | **1.8886** | **1.1186** |

### Comparison with current #1 and all previous runs

| Run | step_avg | Steps | Post-EMA bpb | Roundtrip bpb | Sliding bpb |
|---|---|---|---|---|---|
| Original #1 (no SWA) | 88.56ms | 6776 | 1.1355 | 1.1395 | **1.1159** |
| SWA Run 1 (FA2, torch 2.6) | 87.11ms | 6889 | 1.1372 | (timed out) | — |
| SWA Run 2 (FA3, torch 2.11) | 82.15ms | 7305 | 1.1361 | 2.5369 BROKEN | — |
| SWA Run 3 (FA3, torch 2.9.1) | 75.82ms | 7915 | 1.1339 | 2.4720 BROKEN | — |
| SWA + Hessian fix | 82.05ms | 7314 | 1.1357 | 2.3935 BROKEN | — |
| SWA + GPTQ-lite | 80.95ms | 7413 | 1.1352 | 2.3839 BROKEN | — |
| SWA + GPTQ-lite (force rebuild) | 80.95ms | 7413 | 1.1352 | 2.3971 BROKEN | — |
| **SWA + no override (FIXED)** | **75.34ms** | **7965** | **1.1332** | **1.1411** | **1.1186** |

### Key findings

1. **13% faster per step.** 75.34ms vs 88.56ms — sliding window on 8 of 11 layers saves significant compute with FA3 on H100.

2. **~1000 extra training steps.** 7965 vs 6776 in the same 600 seconds.

3. **Better pre-quant quality.** 1.1332 vs 1.1355 — 0.0023 BPB improvement, entirely from more training steps.

4. **Sliding eval within 0.0027 of current #1.** 1.1186 vs 1.1159 — and this is with GPTQ-lite (suboptimal quantization), not Full Hessian GPTQ.

5. **The roundtrip bug was `torch.compile` + attribute override.** Setting `window_size=(-1,-1)` after model construction doesn't affect `torch.compile(fullgraph=True)` graphs. Never override model attributes between construction and compilation.

## Run 6: 4-way diagnostic eval (definitive root cause)

To determine whether the roundtrip bug was caused by `torch.compile` or by full attention on sliding-window-trained weights, ran all 4 combinations on the same quantized model:

### Results

| Test | Attention | Execution | val_bpb | Status |
|------|-----------|-----------|---------|--------|
| Test 1 | SWA (window=256) | torch.compile | **1.1433** | **OK** |
| Test 4 | SWA (window=256) | eager | **1.1438** | **OK** |
| Test 3 | Full attn (-1,-1) | eager | **2.4014** | **BROKEN** |
| Test 2 | Full attn (-1,-1) | torch.compile | (crashed) | — |

Training: 7376 steps, 81.37ms/step, post-EMA val_bpb 1.1354.

### Definitive root cause

**It's NOT `torch.compile`.** Test 3 runs in pure eager mode (no compilation) with full attention and still produces 2.4014 BPB. `torch.compile` is irrelevant to the bug.

**The root cause is attention pattern mismatch.** Layers 0-7 trained with sliding window (window=256) — their Q, K, V weights learned to produce attention scores for tokens at most 256 positions apart. During training, these layers never computed attention between distant tokens. The weights optimized exclusively for local patterns.

When the eval model switches to full attention (window=(-1,-1)), layers 0-7 suddenly compute attention scores for all 2048 positions. The Q/K dot products for distant token pairs produce **untrained, essentially random scores**. On the fp32/bf16 model, these garbage attention scores might average out because the softmax dampens small random scores. But after int6 quantization, the rounding errors in Q/K weights shift these untrained scores enough to create spurious high-attention values. These propagate through the network and compound across layers, producing catastrophic output.

This explains:
- Why pre-quant full attention would probably work (untrained scores are small, softmax suppresses them)
- Why post-quant full attention breaks (int6 noise pushes some untrained scores above the suppression threshold)
- Why SWA eval works (same attention pattern as training, no untrained scores)
- Why the gap is ~1.3 BPB, not ~0.01 BPP (it's not quantization noise — it's a structural mismatch amplified by quantization)

### Correction to previous analysis

The earlier claim that "`torch.compile` bakes in constructor's `window_size` and ignores the override" was wrong. `torch.compile` correctly captures the overridden `window_size=(-1,-1)`. The problem is that using full attention on sliding-window-trained weights is fundamentally broken after quantization, regardless of eager vs compiled execution.

### Implication

**Sliding window attention is a one-way door.** Once you train with it, you must evaluate with it. You cannot switch to full attention at eval time — the weights don't generalize to the broader attention pattern, especially after quantization. This is different from standard training where eval-time attention changes (e.g., longer sequences via RoPE extrapolation) usually degrade gracefully.

The correct fix remains: **don't override `window_size` on the eval model.** Let it use the same sliding window config as training.

## Summary of all runs

| Run | step_avg | Steps | Post-EMA bpb | Roundtrip bpb | Sliding bpb | Notes |
|---|---|---|---|---|---|---|
| Original #1 (no SWA) | 88.56ms | 6776 | 1.1355 | 1.1395 | **1.1159** | Baseline |
| SWA Run 1 (FA2, torch 2.6) | 87.11ms | 6889 | 1.1372 | (timed out) | — | |
| SWA Run 2 (FA3, torch 2.11) | 82.15ms | 7305 | 1.1361 | 2.5369 | — | eval override bug |
| SWA Run 3 (FA3, torch 2.9.1) | 75.82ms | 7915 | 1.1339 | 2.4720 | — | eval override bug |
| SWA + Hessian fix | 82.05ms | 7314 | 1.1357 | 2.3935 | — | Hessian fix didn't help |
| SWA + GPTQ-lite | 80.95ms | 7413 | 1.1352 | 2.3971 | — | GPTQ method not the cause |
| **SWA + no override (FIXED)** | **75.34ms** | **7965** | **1.1332** | **1.1411** | **1.1186** | **Working** |
| SWA + 4-way diagnostic | 81.37ms | 7376 | 1.1354 | 1.1433 (SWA) / 2.4014 (full) | — | Confirmed: attention mismatch, not torch.compile |

### Remaining gaps to close

1. **Re-enable Full Hessian GPTQ** (change `hessians=None` back to `hessians=hessians`). GPTQ-lite has a larger quantization gap (~0.008 BPB) than Full Hessian GPTQ (~0.004 BPB). This alone could bring sliding eval from 1.1186 to ~1.1150.

2. **Multi-seed validation.** Need 3 seeds with statistical significance (p<0.01, delta > 0.005 nats).

3. **Window size tuning.** 256 was arbitrary. Try 128, 512. Also try different numbers of full-attention layers (2, 4, 5 instead of 3).

4. **HF upload for checkpoints.** Added upload code to push `final_model.pt` and `final_model.int6.ptz` to `Itssshikhar/parameter-golf-swa` for reuse without retraining.
