# Experiment 043: Animal 8xH100 — Thunder Config on NVLink (Run 016)

## Status: COMPLETED (sliding eval still running)

## Config
USE_SWIGLU=1, SWIGLU_HIDDEN=672, MATRIX_LR=0.04, SCALAR_LR=0.04, TIED_EMBED_LR=0.05
LAWA_ENABLED=1, LAWA_INTERVAL=100, WARMDOWN_ITERS=3600, EVAL_STRIDE=64, COMPILE_MODE=default
Train data, 8xH100 NV18 NVLink, RUN_ID=043_thunder_config_nvlink

## Results
| Metric | Value |
|--------|-------|
| Steps | 13,248 @ 45.0ms/step |
| LAWA | 36 snapshots averaged |
| Artifact | **16,010,499 bytes ❌ (10KB over)** |
| **Standard eval** | **1.2243 BPB** |
| Sliding eval | *running* |

## Comparison
- Thunder PCIe (8,420 steps): 1.2364 BPB → NVLink (13,248 steps): **1.2243 BPB** — massive improvement from more steps
- Beats baseline (1.2244) by 0.0001 — basically tied
- LAWA with 36 snapshots likely hurt again
- Artifact 10KB over with h=672 — would need h=668

## wandb
- RUN_ID: 043_thunder_config_nvlink
