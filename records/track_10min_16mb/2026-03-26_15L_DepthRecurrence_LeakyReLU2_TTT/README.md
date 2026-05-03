# Record: 15L Depth Recurrence + LeakyReLU² (val_bpb=1.1360)

**val_bpb: 1.1360** (seed 1337, pending 3-seed validation) | 15.87 MB | 8xH100 SXM

**Status: Draft — single seed, exploring depth recurrence tradeoffs**

## Results (8xH100 SXM)

| Seed | Steps | Pre-quant BPB | Sliding BPB | Artifact |
|------|-------|---------------|-------------|----------|
| 1337 | 5173 | 1.1515 | **1.1360** | 15.87 MB |

## Key Innovation: BI-Guided Depth Recurrence

We use **Block Influence (BI) scores** (ShortGPT, arXiv:2403.03853) to identify redundant layer positions, then apply **weight tying** (depth recurrence) to share a single block across 5 positions. This gives 15 effective transformer layers from only 11 unique parameter blocks — fitting the same 16MB artifact budget as a standard 11-layer model.

### Block Influence Analysis

We trained a 15-layer model and measured BI (angular distance between each layer's input/output). Layers 9–13 showed the lowest BI scores (0.10–0.16), indicating near-identity transformations:

| Layer | BI Score | Role |
|-------|----------|------|
| 0 | 0.459 | High impact (encoder) |
| 6–8 | 0.230–0.235 | High impact (encoder/decoder boundary) |
| **9–13** | **0.104–0.156** | **Low impact → tied** |
| 14 | 0.203 | Moderate (final decoder) |

### How Tying Works

Layers 9, 10, 11, 12, 13 share **one physical block** (same Q/K/V/MLP weights). Each still participates in the U-Net skip connection flow. The deduplication-aware export stores shared weights once with a reconstruction map.

- 10 unique blocks + 1 shared×5 = 15 virtual layers, ~27M unique params
- Int6 + zstd-22 compression → ~15.9 MB (under 16 MB budget)

### Tradeoff: Depth vs Steps

15L at 116ms/step gets ~5170 steps in 600s, vs 11L at 86ms/step getting ~6975 steps. The depth advantage of 15L does not fully compensate for ~1800 fewer training steps in this wallclock-limited setting. At equal step counts, 15L outperforms 11L.

## Architecture

- 15L (10 unique + 1 shared×5), 512d, 8H/4KV GQA, MLP 3x
- LeakyReLU(0.5)², U-Net skip connections
- XSA last 4, Partial RoPE 16/64, LN Scale, VE128
- SmearGate + BigramHash(2048), OrthoInit
- EMA(0.997) + Tight SWA, Late QAT@0.15
- Int6 + GPTQ-lite + zstd-22, FA3

## Reproduce

```bash
SEED=1337 NUM_LAYERS=15 TIE_LAYERS=9,10,11,12,13 \
  DIFF_ATTN=0 VRES_ENABLED=0 TTT_EPOCHS=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Base: signalrush (PR #374/#414). LeakyReLU²: PR #493 (@parinzee), PR #518 (@sofiabod). Block Influence: ShortGPT (arXiv:2403.03853).
