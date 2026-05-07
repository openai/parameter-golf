# MDLM Masked Diffusion + Depth Recurrence

**val_bpb: 1.3428** (int8+zlib roundtrip) | **14.73 MB** | 8×H100 SXM, 600s | Beats #1403 by 0.0057 BPB

Extends the MDLM baseline (PR #1403) with depth recurrence and quantization improvements.

## Stack

- **Depth recurrence**: physical layers L1–L3 looped 1× extra → 12 effective layers / 9 physical layers
- **QAT (STE)**: straight-through quantization at lr_scale < 0.40 (~last 480 steps of 8,049 total)
- **EMA** (decay=0.997) applied to weights before serialization
- **GPTQ-lite**: 5-candidate percentile clip search (99.9%→100%) per row, min-MSE selection
- **Linear LR → 0** (Muon warmdown), **relu² MLP** (hidden=1024), **Muon WD=0.01**
- **Antithetic sampling** for variance reduction during training
- **U-Net skip connections** (encoder → decoder learned weights)

## Results (8×H100 SXM, seed=1337, 600s wallclock)

| Metric | This | #1403 |
|--------|------|-------|
| Pre-quant val_bpb | 1.3379 | 1.3409 |
| **Post-roundtrip val_bpb** | **1.3428** | 1.3485 |
| Quant penalty | **0.0049** | 0.0076 |
| Artifact | 14.73 MB | 15.63 MB |
| Steps | 8,049 | 11,808 |
| ms/step | 74.6 ms | 50.8 ms |
| Wallclock | 600,028 ms | 600s |

EMA + GPTQ-lite cuts quant penalty from 0.0076 → 0.0049. Depth recurrence gives better pre-quant quality (1.3379 vs 1.3409) even with fewer total steps, due to ~12 effective layers of compute per forward pass.

## Architecture

- 9 physical layers, 512d, 8 heads, GQA kv_groups=4
- Depth recurrence: encoder layers L1–L3 (idx 1, 2, 3) looped 1× extra
- Hidden dim: 1024 (mlp_mult=2, relu²)
- SP1024 vocabulary, seq_len=1024
- int8 per-row quantization + zlib-9 compression

## Run script

```bash
export NUM_LAYERS=9
export MLP_MULT=2
export NUM_KV_GROUPS=4
export RECURRENCE_EXTRA=1
export RECURRENCE_START=1
export RECURRENCE_END=4
export MAX_WALLCLOCK_SECONDS=600
export MIN_LR_RATIO=0.0
export MUON_WEIGHT_DECAY=0.01
export ADAM_WEIGHT_DECAY=0.0
export EMA_DECAY=0.997
export NOISE_EPS=0.05
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

8×H100 SXM 80GB HBM3 (RunPod), 600,028 ms wallclock, seed=1337
