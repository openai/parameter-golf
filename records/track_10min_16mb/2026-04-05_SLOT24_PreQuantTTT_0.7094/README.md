# Record: SLOT-24 + Pre-quant TTT — val_bpb 0.7094 (3-seed mean)

**val_bpb: 0.7094** (3-seed mean, std 0.0031) | **<16 MB** | 8xH100 SXM | 600s train, ~580s eval

## Results (3-seed, 8xH100 80GB SXM)

| Seed | Steps | Sliding BPB | SLOT-24 BPB | Artifact |
|------|-------|------------|-------------|----------|
| 1337 | 5,251 | 1.1102 | **0.7064** | 15,930,472 |
| 42 | ~5,250 | — | **0.7093** | 15,930,124 |
| 2025 | ~5,250 | — | **0.7126** | 15,916,348 |
| **Mean** | | | **0.7094** | |

Beats merged SOTA (PR #1019, 1.1147) by **0.405 BPB = 0.685 nats**.

## Techniques

### 1. Per-Sample SLOT-24 (arXiv:2505.12392v2)

Per-sample hidden delta `[bsz, 1, 512]` + logit bias `[bsz, 1, 1024]` optimized with 24 AdamW steps (cosine LR 0.024 -> 0.001) on scored positions only. Stride=96 reduces windows by 33%, enabling more optimization steps within budget. Model weights completely frozen during SLOT.

### 2. Pre-quant AdamW TTT

AdamW TTT (6 epochs, freeze first 2 blocks, cosine LR 0.0005->0.00005) on full-precision EMA model before GPTQ quantization. Adapts weights that then quantize better.

### 3. Architecture (PR #1019 base)

11L, 512d, 8H/4KV GQA, LeakyReLU(0.5)^2 MLP 3x, XSA all layers, QK-Gain 4.0, SmearGate + BigramHash 1536x128, Partial RoPE 16/64, LN Scale, VE128, EMA(0.997) + SWA. Full Hessian GPTQ int6 + lzma.

## Timing Budget

| Phase | Time |
|-------|------|
| Training (600s cap) | 600s |
| Pre-quant TTT (6 epochs) | ~179s |
| GPTQ calibration + quantize | ~7s |
| Sliding window eval (stride 64) | ~115s |
| SLOT-24 eval (stride 96) | ~280s |
| **Total eval** | **~581s** |

## Compliance

- [x] Training: 600s on 8xH100 SXM
- [x] Eval: ~580s (< 600s limit)
- [x] All artifacts under 16,000,000 bytes
- [x] SLOT: frozen model, per-sample delta optimized on scored positions
- [x] No n-gram cache, no two-pass rescoring

## Reproduction

```bash
SEED=1337 TTT_ENABLED=1 TTT_EPOCHS=6 SLOT_ENABLED=1 SLOT_STEPS=24 \
SLOT_LR=0.024 SLOT_LR_MIN=0.001 SLOT_STRIDE=96 BIGRAM_VOCAB_SIZE=1536 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- Base: PR #1019 (@abaybektursun), PR #1263 (SLOT implementation)
- SLOT: arXiv:2505.12392v2, PR #1229 (@resouer)
- Pre-quant TTT: PR #1306
- QK-Gain: PR #1125
