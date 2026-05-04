# Mega-Kernel Search — Live Results

Last update: 2026-05-04 22:36:24

## Winning Configs (speedup > 1.0x)

| Kernel | Config | M | Speedup | ms_fused | ms_ref |
|--------|--------|---|---------|----------|--------|
| K3_UNIFIED_QKV | {} | 16384 | 1.505x | 0.047 | 0.070 |
| K1_MLP_AUTOTUNED | {"best_config": "BLOCK_M: 64, BLOCK_N: 256, BLOCK_K: 64, num | 16384 | 1.471x | 0.125 | 0.184 |
| K3_UNIFIED_QKV | {} | 73728 | 1.356x | 0.205 | 0.278 |
| K1_MLP_AUTOTUNED | {"best_config": "BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 64, nu | 73728 | 1.325x | 0.583 | 0.772 |
| K2_SCALE | {"M": 32768} | 32768 | 1.284x | 0.103 | 0.133 |
| K2_SCALE | {"M": 131072} | 131072 | 1.276x | 0.371 | 0.473 |
| K2_SCALE | {"M": 65536} | 65536 | 1.269x | 0.193 | 0.245 |
| K2_QKV_AUTOTUNED | {"best_config": "BLOCK_M: 128, BLOCK_N: 256, BLOCK_K: 64, nu | 73728 | 1.224x | 0.223 | 0.273 |
| K2_SCALE | {"M": 73728} | 73728 | 1.165x | 0.233 | 0.272 |
| K3_UNIFIED_QKV | {} | 4096 | 1.058x | 0.041 | 0.044 |
| K1_MLP_AUTOTUNED | {"best_config": "BLOCK_M: 64, BLOCK_N: 128, BLOCK_K: 32, num | 4096 | 1.037x | 0.051 | 0.053 |

## All Results

| Kernel | M | Speedup |
|--------|---|---------|
| K3_UNIFIED_QKV | 16384 | 1.505x |
| K1_MLP_AUTOTUNED | 16384 | 1.471x |
| K3_UNIFIED_QKV | 73728 | 1.356x |
| K1_MLP_AUTOTUNED | 73728 | 1.325x |
| K2_SCALE | 32768 | 1.284x |
| K2_SCALE | 131072 | 1.276x |
| K2_SCALE | 65536 | 1.269x |
| K2_QKV_AUTOTUNED | 73728 | 1.224x |
| K2_SCALE | 73728 | 1.165x |
| K3_UNIFIED_QKV | 4096 | 1.058x |
| K1_MLP_AUTOTUNED | 4096 | 1.037x |
| BASELINE_MLP | 4096 | 1.000x |
| BASELINE_QKV | 4096 | 1.000x |
| BASELINE_MLP | 16384 | 1.000x |
| BASELINE_QKV | 16384 | 1.000x |
| BASELINE_MLP | 73728 | 1.000x |
| BASELINE_QKV | 73728 | 1.000x |
| K2_QKV_AUTOTUNED | 16384 | 0.674x |
| K2_SCALE | 16384 | 0.668x |
| K2_SCALE | 512 | 0.477x |
| K2_SCALE | 1024 | 0.441x |
| K2_SCALE | 2048 | 0.438x |
| K2_SCALE | 4096 | 0.434x |
| K2_QKV_AUTOTUNED | 4096 | 0.433x |
| K2_SCALE | 8192 | 0.425x |
| K2_QKV_TMA | 4096 | FAIL |
| K2_QKV_TMA | 16384 | FAIL |
| K2_QKV_TMA | 73728 | FAIL |
| K2_2PASS_QKV | 4096 | FAIL |
| K2_2PASS_QKV | 16384 | FAIL |
| K2_2PASS_QKV | 73728 | FAIL |
| K2_2PASS_SCALE | 512 | FAIL |
| K2_2PASS_SCALE | 1024 | FAIL |
| K2_2PASS_SCALE | 2048 | FAIL |
| K2_2PASS_SCALE | 4096 | FAIL |
| K2_2PASS_SCALE | 8192 | FAIL |
| K2_2PASS_SCALE | 16384 | FAIL |
| K2_2PASS_SCALE | 32768 | FAIL |
| K2_2PASS_SCALE | 65536 | FAIL |
| K2_2PASS_SCALE | 73728 | FAIL |
| K2_2PASS_SCALE | 131072 | FAIL |
