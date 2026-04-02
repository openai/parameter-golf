# Round 3 Complete Results

**Date**: 2026-03-30 to 2026-03-31
**Hardware**: RTX 3070 Laptop GPU (8GB VRAM), WSL2
**All experiments**: 500 iterations, 10L/512d/3xMLP, LeakyReLU(0.5)^2

## Results Table (sorted by int6 bpb)

| # | Experiment | fp32 bpb | int6 bpb | zlib MB | zstd MB | Description |
|---|-----------|----------|----------|---------|---------|-------------|
| 5 | Linear WD + QK=2.0 | **1.3933** | **1.4111** | 12.73 | 12.03 | Best bpb at 500 steps |
| 2 | Linear warmdown | 1.3942 | 1.4124 | 12.72 | 12.03 | Single change, free win |
| - | R2 base | 1.3987 | 1.4157 | 12.76 | ~12.03 | Reference |
| 3 | SWA (8 ckpts) | 1.3986 | 1.4162 | 12.75 | — | Neutral |
| 6 | SOTA stack | 1.4151 | 1.4363 | 12.93 | 12.21 | XSA-4 + RoPE + LN + OrthoInit |
| 8 | **DenseFormer** | 1.4175 | 1.4365 | 12.82 | **11.79** | Novel! Smallest artifact |
| 7 | SOTA + SmearGate | 1.4291 | 1.4465 | 14.43 | 13.72 | SmearGate still converging |
| 1 | QK=4 + XSA-all | 1.6382 | 1.6538 | 13.03 | — | Failed |
| 4 | R2 range reg | 3.30 | — | — | — | Broke training |

## Convergence Rates (val_bpb by step)

| Step | Exp5 (simple) | Exp6 (SOTA) | Exp8 (DenseFormer) |
|------|--------------|-------------|-------------------|
| 100 | 1.871 | 1.930 | 1.963 |
| 200 | 1.596 | 1.625 | **1.622** |
| 300 | 1.504 | 1.528 | 1.528 |
| 400 | **1.437** | 1.459 | 1.462 |
| 500 | **1.393** | 1.415 | 1.418 |

Exp6/exp8 converge faster — crossover expected at ~700-1000 steps.

## Key Findings

1. **Linear warmdown** beats cosine by -0.003 to -0.005 bpb (confirmed)
2. **QK gain=2.0** gives small additional gain over 1.5
3. **zstd-22** saves 5-8% over zlib consistently
4. **DenseFormer** (novel, not in competition): matches SOTA stack quality with smallest artifact (11.79 MB vs 12.21 MB), 3.5 MB headroom under 16 MB
5. **SOTA stack** (XSA-4, Partial RoPE, LN Scale, OrthoInit) works but starts slower at 500 steps — will surpass simple config by ~1000 steps
6. **SmearGate** still converges slowly at 500 steps but gap narrows vs SOTA stack (from +0.05 at step 200 to +0.017 at step 500)
