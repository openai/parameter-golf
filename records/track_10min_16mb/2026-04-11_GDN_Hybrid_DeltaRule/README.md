# Record: GDN-Hybrid — Gated DeltaNet + Sliding Window Attention

**val_bpb = 1.0283** (3-seed mean, cold cache) | **14.70 MB** | 8×H100 SXM

*First non-transformer architecture in the 10-min record track.*
*Beats current SOTA (1.0810) by **5.27 centiBPB** — largest single improvement in competition history.*
*No TTT. Fixed predictor. Sliding-window eval only.*

---

## Results

### 3-seed cold-cache runs (VAL_LOSS_EVERY=9999, fresh pods)

All three seeds run on separate fresh pods. Cold-start signature confirmed: step 1 at ~105s
(Triton JIT overhead), verifying no prior cache contamination.

| Seed | Steps | EMA BPB  | **Quantized BPB** | Artifact (bytes) |
|------|-------|----------|-------------------|-----------------|
| 42   | 1857  | 1.017970 | **1.027163**      | 15,188,240      |
| 1337 | 1858  | 1.018624 | **1.027614**      | 15,417,768      |
| 2024 | 1858  | 1.020559 | **1.030148**      | 15,314,099      |
| **Mean** | — | **1.019051** | **1.028308**  | —               |
| **Std**  | — | **0.001356** | **0.001610**  | —               |

Training: 590s on 8×H100 SXM per seed. No in-training validation evals (`VAL_LOSS_EVERY=9999`).
Cold-start overhead (~105s of Triton JIT) accounts for fewer steps (~1858) vs warm cache (~2247).

Merged SOTA (PR #1493): **1.0810 BPB**. 3-seed mean delta: **−0.0527 BPB**. Clears the 0.005 threshold.

### Additional Evidence (not part of the submitted claim)

Two supplemental data points included for completeness. Neither affects the submitted val_bpb.

**Eval-determinism audit — seed=42 artifact, independent reload:**

After the phase-2 seed=42 run, the quantized artifact was loaded from disk and re-evaluated
independently using `eval_rls.py` (separate process, no training state):

```
Model loaded from checkpoints/final_model_D_GDN_Hybrid_seed42.int6.ptz
Sliding BPB (stride=64, no XSA): 1.053155
(Expected from training: 1.053155)
```

Exact match. This confirms the artifact format is self-contained and the sliding-window eval
is deterministic — loading and rescoring the artifact yields the same BPB as the training
log's roundtrip validation.

**Warm-cache seed=1337 run:**

One earlier run of seed=1337 was conducted on the same pod after Triton kernels were already
compiled (warm cache). With ~52s less JIT overhead, training reached 2247 steps instead of
1858, producing a lower BPB:

| Seed | Steps | EMA BPB  | Quantized BPB | Artifact (bytes) | Cache |
|------|-------|----------|---------------|-----------------|-------|
| 1337 | 2247  | 1.007164 | 1.015890      | 15,802,210      | warm  |

This is included as a datapoint showing the architecture's capability under favorable
conditions. It is **not** part of the 3-seed submitted claim and is not reproducible on a
fresh pod without pre-compiling Triton kernels. The official submission is based solely on
the cold-cache runs above.

---

## What This Is

Every competitive submission since March 18 has been a transformer with incremental improvements:
depth recurrence, parallel residuals, SP8192, QK-gain tuning, TTT. This submission replaces the
entire transformer backbone with **Gated DeltaNet (GDN)** — a delta-rule linear recurrence model —
and achieves a new record without any of those stacked tricks.

**The core model** is a Griffin-style hybrid:

```
[GDN×5] → [SWA] → [GDN×5] → [SWA_shared]
```

- **GDN layers**: each layer maintains a recurrent key-value associative memory updated by the
  delta rule. At each token, the memory is queried (read) and updated (write) using learned
  per-token gates. This is fundamentally different from attention — there is no O(T²) computation,
  no KV cache that grows with sequence length, and the model "compresses" past context into a
  fixed-size hidden state that evolves online.
- **SWA layers**: two Sliding Window Attention layers (window=512) with shared weights provide
  local attention. The weight sharing (SWA_shared) reduces parameter count while maintaining
  expressiveness.

The result is a model that handles long-range context via its recurrent state and local context
via attention, similar in spirit to Griffin/Hawk but using GDN's stronger delta-rule memory update.

---

## Architecture

**Model D — GDN-Hybrid:**
- Layout: `[GDN×5] → [SWA] → [GDN×5] → [SWA_shared]` (12 layers total)
- Dimension: 512, MLP mult: 3×, GDN head_dim: 64
- SWA: window=512, 8 heads / 4 KV heads, weight-shared across both SWA layers
- QK-Gain: 5.0 (learnable per-head scaling, consistent with transformer SOTA finding)
- BigramHash(3072, 112) + trigram hash embeddings
- SmearGate on token embeddings
- Logit softcap at 30.0
- **Total parameters: 33,862,953**

**GDN implementation:** `fla.layers.GatedDeltaNet` from the Flash Linear Attention library.
Each layer uses `expand_v=1`, `head_dim=64`, `use_short_conv=True`. The delta rule update:
```
h_t = (I - β_t k_t k_t^T) h_{t-1} + β_t v_t k_t^T
```
where β_t is a learned forgetting gate, k_t/v_t are key/value projections of the current token.

---

## Training

- **Optimizer:** Muon (Newton-Schulz 5) for matrices, AdamW for embeddings/scalars
- **Steps:** 2247 in 590s (seed=1337, VAL_LOSS_EVERY=9999)
- **Batch:** 786,432 tokens (384 sequences × 2048)
- **LR schedule:** cosine warmup (100 steps) → constant (no warmdown reached within budget)
- **EMA:** decay 0.997, applied at end of training
- **`VAL_LOSS_EVERY=9999`:** In-training validation evals are disabled. Each eval consumes
  ~141–182s of the 590s budget. With `VAL_LOSS_EVERY=500`, two evals fire at steps 500 and
  1000, capping training at ~1390 steps. Setting `VAL_LOSS_EVERY=9999` gives the full 590s
  budget to training. Cold-cache pods reach ~1857–1858 steps (105s Triton JIT overhead);
  warm-cache pods reach ~2247 steps.
- **`torch._dynamo.config.recompile_limit = 64`:** Defensive guard for FLA's `layer_idx`
  integer attributes. No longer strictly needed (eval compile was removed) but harmless.

---

## Quantization

Full-Hessian GPTQ with int6 matrices + zstd-22 compression:

1. **Calibration data:** 64 autoregressive sequences × 2048 tokens, generated from the model
   itself (AR self-generated, same approach as PR #1019).
   - RoPE fix: generation starts from `init_len=16` tokens to avoid `T < num_heads`
     shape mismatch in `apply_rotary_emb` (heads=8, so T∈[2,8) causes broadcast failure).
2. **Hessian collection:** 29 linear layers instrumented; Hessians computed in bfloat16.
3. **Quantization:** percentile-based clipping search per layer, int6 packing.
4. **Compression:** zstd level 22.
5. **Degradation:** ~0.009–0.010 BPB across all seeds (EMA→quantized).

---

## Compliance

Fixed predictor — no eval-time adaptation of any kind.

Per Issue #1017 (Track A — fixed predictor):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored
  from prefix tokens only. GDN recurrent state is forward-only.
- **Condition 2 (Normalized distribution):** Standard softmax over full 1024-token vocabulary.
  No n-gram cache, no logit biasing, no TTT corrections.
- **Condition 3 (Score before update):** N/A — no eval-time parameter updates.
- **Condition 4 (Single pass):** Each validation token scored exactly once. No rescoring.

Additional:
- `TTT_ENABLED=0` (no test-time training)
- No SLOT, no RLS, no n-gram mixer at eval time
- No pre-quantization adaptation on validation data
- GPTQ calibration uses model-generated synthetic sequences only (no val data)
- All 3 artifacts < 16,000,000 bytes ✓ (14.48–14.70 MB)
- All 3 seeds: training 590s on 8×H100 SXM ✓

---

## Reproduction

```bash
# Install dependencies
pip install sentencepiece zstandard
pip install flash_attn_3 --no-deps  # for SWA layers
pip install flash-linear-attention  # FLA (GDN implementation)

# Download SP1024 dataset
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train + quantize (590s budget, ~23 min total including GPTQ)
SEED=1337 ARCH_MODE=D MAX_WALLCLOCK_SECONDS=590 ITERATIONS=9999 \
  TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
  QK_GAIN_INIT=5.0 GPTQ_ENABLED=1 VAL_LOSS_EVERY=9999 \
  torchrun --standalone --nproc_per_node=8 experiments/direction5_gdn/train_gpt.py
```

Expected output on a **cold pod** (fresh Triton cache, as submitted):
```
# seed=42
Training complete in 590s (1857 steps)
EMA BPB (no XSA): 1.017970
Quantized BPB (no XSA): 1.027163
Artifact: 15,417,768 bytes (14.70 MB)

# seed=1337
Training complete in 590s (1858 steps)
EMA BPB (no XSA): 1.018624
Quantized BPB (no XSA): 1.027614
Artifact: 15,314,099 bytes (14.60 MB)

# seed=2024
Training complete in 590s (1858 steps)
EMA BPB (no XSA): 1.020559
Quantized BPB (no XSA): 1.030148
Artifact: 15,188,240 bytes (14.48 MB)
```

Cold-start signature: step 1 at ~105s (Triton JIT), step 100 at ~131s.

On a **warm pod** (pre-compiled Triton cache), expect ~2247 steps and quantized BPB ~1.015–1.016.

Expected output (seed=42, warm):
```
Training complete in 590s (2247 steps)
EMA BPB (no XSA): ~1.008
Quantized BPB (no XSA): ~1.017
Artifact: 14,620,730 bytes (13.94 MB)
```

---

## Key Engineering Notes

### torch.compile recompile_limit
The single most impactful engineering fix. FLA's `GatedDeltaNet` stores `layer_idx` as
an integer `nn.Module` attribute. torch.compile treats this as a static guard, triggering
one full recompilation per unique `layer_idx`. With 10 GDN layers (indices 0–9) and the
default `recompile_limit=8`, layers 8 and 9 permanently fall back to eager mode at ~step 500,
reducing throughput from 3.8 steps/s to ~0.57 steps/s (7× slowdown).

Fix (line ~55 of train_gpt.py):
```python
torch._dynamo.config.recompile_limit = 64
```

### GPTQ RoPE shape mismatch
During autoregressive calibration generation, growing sequences from length 1 hits a shape
mismatch in `apply_rotary_emb`: the function uses `x.shape[-2]` (= `num_heads=8`) to slice
the cosine table, so when `T < num_heads`, `cos[:8]` clips to shape `[T, D//2]` which fails
to broadcast against `[B, T, 8, D//2]` at dimension 2. Fix: start generation with `init_len=16`
tokens, skipping the problematic `T ∈ [2, 8)` range.

---

## Credits

- **Christopher-Lee-McClendon** — GDN-Hybrid architecture, FLA integration, GPTQ pipeline,
  production training code (PR #1370, branch `submission/10L-gdn-puregdn-7k-legal-ttt`)
- **Abhishek8108** — Direction-5 adaptation: QK-Gain 5.0, BigramHash(3072×112)+trigram,
  torch.compile fix, GPTQ RoPE fix, 3-seed verification on 8×H100 SXM

---

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — training + quantization script
- `architectures.py` — GDN-Hybrid model definition
- `configs.py` — Model D configuration
- `train_seed42.log`
- `train_seed2024.log`
- `train_seed1337.log`
