# Non-Record Submission: GDN-Hybrid — Gated DeltaNet + Sliding Window Attention

**val_bpb = 1.209735** (3-artifact mean, stride=512 rescore) | **14.48–14.70 MB** | 8×H100 SXM

*A GDN-hybrid non-transformer submission.*
*No TTT. Fixed predictor. Sliding-window eval only.*

---

## Results

### Corrected BPB (post-hoc rescore, stride=512)

Artifacts from the original 3-seed training runs were rescored using the fixed BPB formula
(see **BPB Correction** below). Eval was run on a single H100 with `EVAL_STRIDE=512`.
Full results in `rescore_results.tsv`.

| Seed | Artifact (bytes) | Corrected BPB |
|------|-----------------|---------------|
| 42   | 15,188,240      | **1.208549**  |
| 1337 | 15,417,768      | **1.208902**  |
| 2024 | 15,314,099      | **1.211754**  |
| **Mean** | —           | **1.209735**  |

Training: 590s on 8×H100 SXM per seed. `VAL_LOSS_EVERY=9999` (no in-training val evals).

### Training logs (for reference)

The training logs report lower BPB values — these were computed with the buggy formula
and are not the correct BPB. They are included here only as provenance of the training runs.

| Seed | Steps | EMA BPB (log, buggy) | Quantized BPB (log, buggy) |
|------|-------|----------------------|---------------------------|
| 42   | 1857  | 1.017970             | 1.027163                  |
| 1337 | 1858  | 1.018624             | 1.027614                  |
| 2024 | 1858  | 1.020559             | 1.030148                  |

---

## BPB Correction

The original submission (PR #1545, now closed) reported val_bpb = 1.028, which was incorrect
due to a double-counting bug in `build_sentencepiece_luts`.

**The bug:** For tokens with a leading space (SentencePiece `▁` prefix), the function was
including the space byte in `base_bytes` AND then adding it again conditionally in the eval loop:

```python
# Buggy — adds +1 here...
base_bytes[i] = len(piece[1:].encode("utf-8")) + 1

# ...then adds it again here
tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
```

This inflated `byte_count` (the denominator in BPB), making the score appear ~15% better
than it actually was.

**The fix:** Remove the `+1` from `build_sentencepiece_luts`, matching the canonical
`train_gpt.py` implementation. The conditional `+1` in the eval loop correctly handles
leading-space bytes on its own.

```python
# Fixed
base_bytes[i] = len(piece[1:].encode("utf-8"))
```

The training itself was unaffected — the bug was evaluation-only.

---

## What This Is

Every competitive submission since March 18 has been a transformer with incremental improvements:
depth recurrence, parallel residuals, SP8192, QK-gain tuning, TTT. This submission replaces the
entire transformer backbone with **Gated DeltaNet (GDN)** — a delta-rule linear recurrence model —
combined with Sliding Window Attention layers.

**The core model** is a Griffin-style hybrid:

```
[GDN×5] → [SWA] → [GDN×5] → [SWA_shared]
```

- **GDN layers**: each layer maintains a recurrent key-value associative memory updated by the
  delta rule. At each token, the memory is queried (read) and updated (write) using learned
  per-token gates. No O(T²) computation, no KV cache that grows with sequence length — the model
  compresses past context into a fixed-size hidden state that evolves online.
- **SWA layers**: two Sliding Window Attention layers (window=512) with shared weights provide
  local attention. Weight sharing reduces parameter count while maintaining expressiveness.

The result is a model that handles long-range context via its recurrent state and local context
via attention, similar in spirit to Griffin/Hawk but using GDN's stronger delta-rule memory update.

---

## Architecture

**Model D — GDN-Hybrid:**
- Layout: `[GDN×5] → [SWA] → [GDN×5] → [SWA_shared]` (12 layers total)
- Dimension: 512, MLP mult: 3×, GDN head_dim: 64
- SWA: window=512, 8 heads / 4 KV heads, weight-shared across both SWA layers
- QK-Gain: 5.0 (learnable per-head scaling)
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
- **Steps:** ~1857–1858 in 590s (cold-cache mean, VAL_LOSS_EVERY=9999)
- **Batch:** 786,432 tokens (384 sequences × 2048)
- **LR schedule:** cosine warmup (100 steps) → constant
- **EMA:** decay 0.997, applied at end of training
- **`VAL_LOSS_EVERY=9999`:** In-training validation disabled to give full 590s budget to training

---

## Quantization

Full-Hessian GPTQ with int6 matrices + zstd-22 compression:

1. **Calibration data:** 64 autoregressive sequences × 2048 tokens, generated from the model itself
   (AR self-generated). RoPE fix: generation starts from `init_len=16` tokens to avoid shape
   mismatch in `apply_rotary_emb` when `T < num_heads`.
2. **Hessian collection:** 29 linear layers instrumented; Hessians computed in bfloat16.
3. **Quantization:** percentile-based clipping search per layer, int6 packing.
4. **Compression:** zstd level 22.

---

## Compliance

Fixed predictor — no eval-time adaptation of any kind.

Per Issue #1017 (Track A — fixed predictor):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. GDN recurrent state is forward-only.
- **Condition 2 (Normalized distribution):** Standard softmax over full 1024-token vocabulary. No n-gram cache, no logit biasing, no TTT corrections.
- **Condition 3 (Score before update):** N/A — no eval-time parameter updates.
- **Condition 4 (Single pass):** Each validation token scored exactly once.

Additional:
- `TTT_ENABLED=0`
- No SLOT, no RLS, no n-gram mixer at eval time
- No pre-quantization adaptation on validation data
- GPTQ calibration uses model-generated synthetic sequences only (no val data)
- All 3 artifacts < 16,000,000 bytes ✓ (14.48–14.70 MB)
- All 3 seeds: training 590s on 8×H100 SXM ✓

---

## Reproduction

```bash
pip install sentencepiece zstandard flash-linear-attention
pip install flash_attn_3 --no-deps

python3 data/cached_challenge_fineweb.py --variant sp1024

SEED=42 ARCH_MODE=D MAX_WALLCLOCK_SECONDS=590 ITERATIONS=9999 \
  TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
  QK_GAIN_INIT=5.0 GPTQ_ENABLED=1 VAL_LOSS_EVERY=9999 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-11_GDN_Hybrid_DeltaRule/train_gpt.py
```

---

## Key Engineering Notes

### torch.compile recompile_limit
FLA's `GatedDeltaNet` stores `layer_idx` as an integer `nn.Module` attribute. torch.compile
treats this as a static guard, triggering one full recompilation per unique `layer_idx`. With
10 GDN layers and the default `recompile_limit=8`, layers 8 and 9 fall back to eager mode,
causing a ~7× throughput drop from step ~500 onward.

Fix:
```python
torch._dynamo.config.recompile_limit = 64
```

### GPTQ RoPE shape mismatch
During autoregressive calibration, growing sequences from length 1 hits a shape mismatch in
`apply_rotary_emb` when `T < num_heads`. Fix: start generation with `init_len=16` tokens.

---

## Attribution

GDN-Hybrid architecture, FLA integration, and GPTQ pipeline originally developed by
**Christopher-Lee-McClendon** (PR #1370).

---

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — training + quantization script (BPB formula corrected)
- `architectures.py` — GDN-Hybrid model definition
- `configs.py` — Model D configuration
- `train_seed42.log`, `train_seed1337.log`, `train_seed2024.log` — training logs
- `rescore_results.tsv` — corrected BPB per artifact (stride=512)
