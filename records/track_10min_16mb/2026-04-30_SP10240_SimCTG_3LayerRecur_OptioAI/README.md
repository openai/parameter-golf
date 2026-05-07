# N9 SimCTG + 3-Layer Recurrence (Submission A — sliding-window baseline)

**val_bpb = 1.07502** (3-seed mean, std 0.00230) | artifact ~15.99 MB | 8×H100 SXM | brotli-quantized model + lzma-compressed code

## 3-Seed Results (sliding-window stride 64, no test-time training)

| Seed | sliding val_bpb | post-EMA | artifact (bytes) | fits cap |
|------|-----------------|----------|------------------|----------|
| 42   | **1.07766** | 1.07948 | 15,975,529 | ✅ |
| 1337 | **1.07400** | 1.07535 | 15,956,059 (with self-extracting code) | ✅ |
| 2025 | **1.07340** | 1.07497 | 15,999,989 | ✅ |
| **Mean** | **1.07502** | 1.07660 | | |
| **Std** | **0.00230** | | | |

Δ vs leaderboard sliding-window SOTA (1.0827, 2026-04-09 SP8192_3LayerRecur): **−0.00768 BPB** (7.7 mBPB better, 3-seed σ 2.3 mBPB).

## Architecture

11L × 512d × 8H / 4KV with: 3-Layer Recurrence (encoder loops layers 3-5), Parallel Residuals (from layer 7),
LeakyReLU(0.5)² SwiGLU, Partial RoPE (16/64), XSA on all 11 layers, tied embeddings, SP10240 tokenizer.

**Training**: Polar Express NS Muon (5-iter) on matrix params + AdamW on embed/scalar; 4534 steps in ~588s (early stop at MAX_WALLCLOCK_SECONDS=600).
**Quantization**: Mixed GPTQ — int6 attention/MLP matrices, int7 token embeddings.
**Eval**: sliding-window stride 64 on quantized model (PR #1493 legal-TTT line).

## Our novel contributions

1. **SimCTG λ=0.3, margin=0.4 contrastive regularizer** added to the standard CE objective during training — confirmed reproducible across 3 seeds (sliding-window std 0.00230). Adds angular spread on token-level hidden states (off-diagonal cosine²) at no inference cost.
2. **3-seed validation** of this SimCTG setting on the SP10240 base, demonstrating monotonic improvement over the unregularized N9 lineage.

## Compliance

- Trains in 600s on 8×H100 (`MAX_WALLCLOCK_SECONDS=600`). 
- Eval ops < 200s (no PreQuantTTT, no post-quant TTT — pure sliding-window).
- Artifact under 16,000,000 bytes including lzma-compressed code.

## Files

- `final_model.int6.ptz` — brotli-compressed quantized model (~15.93 MB)
- `train_gpt.py` — self-extracting training code (lzma+base85 wrapped, SOTA-standard format, 19,785 bytes)
- `submission.json` — metadata
- `train_seed{42,1337,2025}.log` — 3-seed training logs

## Credits

PR #1855 SOTA stack (Kevin Clark et al.), PR #1413 legal score-first TTT line (dexhunter), PR #1493 sliding-window stride 64 (bigbag), PR #1394 SP-CaseOps tokenizer (clarkkev), PR #287 Partial RoPE (jfprincz), PR #1412 Parallel Residuals (Robby955), PR #549 LeakyReLU(0.5)² (abaybektursun).
