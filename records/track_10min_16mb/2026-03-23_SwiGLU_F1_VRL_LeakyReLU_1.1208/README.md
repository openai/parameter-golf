# SwiGLU F1 — VRL + LeakyReLU(0.5)² + GPTQ (val_bpb: 1.1208)

## Summary

Quality-maximizing run based on PR #505 (JoeProAI SwiGLU+VE128) with added techniques from v7 SOTA and stacked experiments. **Over 16MB budget (20.6MB)** — intended as raw quality baseline for Frugendorff compression.

## Results

| Metric | Value |
|--------|-------|
| Steps | 4,521 in 600s at 132.7ms/step |
| Pre-quant val_bpb | 1.1410 |
| **Post-GPTQ sliding val_bpb** | **1.1208** |
| Quant gap | 0.0202 |
| Artifact size | 20,645,288 bytes (20.6 MB) |
| Model params | 33,425,507 |
| TTT | None |

## Architecture (PR #505 base)

- 11 layers, dim=512, 8H/8KV (full MHA), head_dim=64
- SwiGLU FFN with **LeakyReLU(0.5)²** (hidden=1792)
- U-Net Skip Gates (5 encoder, 6 decoder)
- XSA4 (last 4 layers)
- Value Embeddings VE128 (layers 9-10)
- BigramHash (8192 buckets, 128-dim)
- **VRL** (Value Residual Learning): sigmoid-gated first-block mixing
- Partial RoPE (16 dims), LN Scale, Logit Softcap 30.0
- Tied embeddings

## Training

- Muon (matrices): lr=0.025, momentum=0.99
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025
- decoder_lr_mult=2.0
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3,500 iters
- EMA: decay=0.997
- Late QAT at scale<0.5

## Quantization

- GPTQ (Hessian-calibrated, 256 training samples)
- int8 for attn.proj (sensitive layer)
- int6 for all other weights
- zstd-22 compression

## What's New vs PR #505

| Technique | Source | Expected Impact |
|-----------|--------|----------------|
| VRL | Stacked experiments | -0.015 BPB |
| LeakyReLU(0.5)² | Stacked experiments | -0.002 to -0.005 |
| Grad clip 0.3 | v7 SOTA | Stability |
| EMA 0.997 | PR #505 matched | — |
| int8 attn.proj | v7 SOTA | -0.001 |

## Next Steps

- Run Frugendorff compression to fit 16MB budget (est. cost: ~0.007 BPB)
- Run with TTT_EVAL_ENABLED=1 for legal score-first TTT boost
- Multi-seed verification
