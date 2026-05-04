# Mega-Kernel AutoSearch Results

GPU: NVIDIA H100 80GB HBM3
Torch: 2.9.1+cu128  Triton: 3.5.1
Date: 2026-05-04 22:36:24

## Best Kernels at Full Competition Scale (M=73728)

| Rank | Kernel | Speedup | ms_fused | ms_ref | Notes |
|------|--------|---------|----------|--------|-------|
| 1 | K3_UNIFIED_QKV | 1.356x | 0.205 | 0.278 | WINNER |
| 2 | K1_MLP_AUTOTUNED | 1.325x | 0.583 | 0.772 | WINNER |
| 3 | K2_QKV_AUTOTUNED | 1.224x | 0.223 | 0.273 | WINNER |
| 4 | K2_SCALE | 1.165x | 0.233 | 0.272 | WINNER |
| 5 | BASELINE_MLP | 1.000x | 0.760 | 0.760 |  |
| 6 | BASELINE_QKV | 1.000x | 0.272 | 0.272 |  |

## Scaling Analysis

| M | Kernel | Speedup |
|---|--------|---------|
| 512 | K2_SCALE | 0.477x |
| 512 | K2_2PASS_SCALE | FAIL |
| 1024 | K2_SCALE | 0.441x |
| 1024 | K2_2PASS_SCALE | FAIL |
| 2048 | K2_SCALE | 0.438x |
| 2048 | K2_2PASS_SCALE | FAIL |
| 4096 | K2_SCALE | 0.434x |
| 4096 | K2_2PASS_SCALE | FAIL |
| 8192 | K2_SCALE | 0.425x |
| 8192 | K2_2PASS_SCALE | FAIL |
| 16384 | K2_SCALE | 0.668x |
| 16384 | K2_2PASS_SCALE | FAIL |
| 32768 | K2_SCALE | 1.284x |
| 32768 | K2_2PASS_SCALE | FAIL |
| 65536 | K2_SCALE | 1.269x |
| 65536 | K2_2PASS_SCALE | FAIL |
| 73728 | K2_SCALE | 1.165x |
| 73728 | K2_2PASS_SCALE | FAIL |
| 131072 | K2_SCALE | 1.276x |
| 131072 | K2_2PASS_SCALE | FAIL |

## Recommendation

**USE K3_UNIFIED_QKV** — 1.356x speedup at M=73728

Config: `{}`
