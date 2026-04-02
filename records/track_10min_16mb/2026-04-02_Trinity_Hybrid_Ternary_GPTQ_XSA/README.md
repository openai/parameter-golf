# Trinity Hybrid: Ternary-Int6 GPTQ Quantization

## Approach

Trinity Hybrid is a mixed-precision post-training quantization strategy that assigns
different bit-widths to different weight categories based on their information density:

- **MLP weights (fc/up + proj/down)**: Ternary quantization {-1, 0, +1}
  - Per-group (group_size=128) absmean scaling
  - Base-3 packing: 5 trits per byte (3^5 = 243 <= 255)
  - Effective ~1.6 bits per weight
  - MLP weights have high redundancy and tolerate aggressive quantization

- **Attention weights (c_q, c_k, c_v, proj)**: Int6 GPTQ
  - Hessian-aware quantization with Cholesky error compensation
  - Per-row scaling with percentile search
  - 6-bit precision preserves attention's directional sensitivity

## Key Insight

MLP weights in transformer models are highly redundant -- they learn sparse, pattern-matching
functions where most weights are near-zero. Ternary quantization captures the essential
sign structure while discarding magnitudes, at ~3.75x compression vs int6.

Attention weights encode precise geometric relationships (queries, keys, values) that require
finer granularity. Int6 GPTQ with Hessian-guided error compensation preserves these.

## Architecture Changes

- **MLP width**: Increased from 3x to 5x model_dim
  - Ternary MLP at 5x width: ~1.6 bits * 5x = 8 bit-equivalents per dim
  - vs Int6 MLP at 3x width: ~6 bits * 3x = 18 bit-equivalents per dim
  - Net effect: more capacity at lower storage cost

## What's Preserved (Unchanged)

- Training loop, optimizer (Muon + Adam), learning rate schedule
- XSA (Cross-head Self-Attention subtraction) on all layers
- BigramHash embedding (2048 vocab, 128-dim)
- Value Embedding injection at layers 9,10
- EMA / SWA weight averaging
- Autoregressive calibration data generation
- Sliding window evaluation

## Base

Built on `2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072` with the following modifications:
1. Added ternary quantization functions (ternary_quantize, pack/unpack_ternary_base3)
2. Replaced mixed_quantize_int6 with mixed_quantize_trinity (hybrid dispatch)
3. Replaced dequantize_mixed_int6 with dequantize_trinity (handles both formats)
4. Changed mlp_mult default from 3.0 to 5.0
5. Updated log messages for Trinity Hybrid branding

## Track

- Track: 10min_16mb (10 minute training, 16MB submission cap)
- Budget: code + compressed model <= 16MB
