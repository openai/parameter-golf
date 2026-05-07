# Model 4: "The Optimized Transformer" — Build Spec

**Classification:** PUBLIC (safe to submit for leaderboard visibility)
**Target bpb:** 1.09-1.11 on 8×H100
**Approach:** Our own clean implementation of proven techniques, built incrementally

---

## Architecture

Based on understanding (not copying) the current #1, but written from scratch:

- **11 layers**, 512 dim, 8 heads (4 KV heads, GQA)
- **3x MLP expansion** (1536 hidden), ReLU² activation
- **Tied embeddings** (FP16 precision)
- **Muon optimizer** for matrix weights (lr=0.025, momentum warmup 0.92→0.99 over 1500 steps, WD=0.04)
- **AdamW** for embeddings (lr=0.035, WD=0.04)
- **Gradient clip:** 0.3
- **Sequence length:** 2048
- **Batch:** 786,432 tokens/step

## Techniques to Implement (ONE AT A TIME, verify each)

### Tier 1 — Core architecture (implement first, verify training converges)
1. Basic 11L transformer with GQA — verify loss drops normally
2. 3x MLP with ReLU² — verify no degradation
3. Muon optimizer — verify faster convergence than Adam

### Tier 2 — Training enhancements (add after Tier 1 works)
4. EMA weight averaging (decay=0.997) — verify eval improvement
5. Warmdown schedule at 3500 iterations
6. Late QAT (STE int6 fake-quantization when LR scale < 0.15)
7. OrthoInit for weight initialization

### Tier 3 — Architecture additions (add after Tier 2 works)
8. BigramHash (2048 buckets, dim=128)
9. SmearGate (current + previous token blending)
10. U-Net skip connections (encoder-decoder residuals)
11. Logit softcap at 30.0

### Tier 4 — Eval enhancements (add last)
12. Sliding window eval (stride=64)
13. GPTQ-lite quantization (5 clip percentiles per row)
14. Int6 per-row for MLP+attention, Int8 for embeddings
15. zstd level 22 compression

### Tier 5 — Our additions (what makes it ours)
16. Hard example mining — identify high-loss tokens, oversample them
17. Curriculum learning — order training data by difficulty
18. Test-Time Training with AdamW on already-evaluated tokens

## Build Instructions for Codex

**CRITICAL:** Build in tiers. After each tier, the code must:
- Parse with no syntax errors
- Run a 100-step smoke test without crashing
- Show loss decreasing

DO NOT implement all tiers at once. Start with Tier 1. Verify. Then Tier 2. Verify. Etc.

Reference files (READ but don't copy):
- `train_gpt.py` (baseline — understand the structure)
- `records/track_10min_16mb/2026-03-22_*/train_gpt.py` (current #1 — understand techniques)
- `records/track_10min_16mb/2026-03-20_*/train_gpt.py` (other top entries)

## Output
- `train_gpt_model4.py` — complete, tested, verified at each tier
