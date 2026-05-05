# Non-record: Frozen Random Backbone + Rank-304 LoRA Adapters

**val_bpb: 1.3220** (sliding window stride=256, 1xH100) | **13.5MB** artifact (2.5MB headroom) | 1xH100 80GB SXM, 600s

This submission directly addresses the OpenAI wishlist item **"Learning adapters on random linear maps"** — the backbone is never trained or serialized. Instead, frozen weights are reconstructed from a deterministic seed at load time (0 bytes in the artifact). Only LoRA adapter matrices, embeddings, and control tensors are stored.

## Key Innovation

Standard parameter-golf submissions train a full-rank model, then quantize it into 16MB. This approach inverts that: the backbone (67% of total parameters) costs **0 bytes** in the artifact because it is reconstructed from `seed=42` at inference time. The entire 16MB budget goes to low-rank adapter matrices, embeddings, and quantization scales.

At rank 304, each adapter pair (A: dim x rank, B: rank x dim) captures the learned delta from the random backbone. Combined with depth recurrence (reusing adapter weights across 3 passes through layers 3-5), each adapter parameter gets 3x the gradient signal of a standard linear layer.

## Results (1xH100 80GB SXM)

| Eval mode | val_loss | val_bpb |
|-----------|----------|---------|
| Pre-quant (no EMA) | 3.3923 | 1.3133 |
| Quantized (int6+brotli) | 3.4547 | 1.3374 |
| Sliding window (stride=256) | 3.4149 | **1.3220** |

Steps: 439 (1352ms/step). Artifact: 13,512,579 bytes. Quant gap: +0.024 BPB. Peak memory: 70.3 GB / 80 GB.

For comparison, a full-rank run with the same architecture at MLP=3 got 1.3213 BPB sliding window on 1xH100 but required EMA, which adds 0.12 BPB post-quant overhead at this step count. The adapter approach avoids this.

## Architecture

11L x 512d x 8H / 4KV, MLP 3x (LeakyReLU(0.5)^2), tied embeddings.

| Component | Details |
|-----------|---------|
| Frozen backbone | Each linear layer's weight reconstructed from `seed=42 + layer_id`, Kaiming std |
| LoRA rank | 304 on all linear layers: `y = frozen(x) + scale * B(A(x))` |
| Depth recurrence | Layers 3-5 looped 2x (3 passes total), activated at 35% of training |
| XSA | All layers, partial RoPE (16 dims, NTK scaling) |
| U-Net skip connections | Encoder-decoder with learned skip weights |
| Parallel residuals | From layer 7 onward |
| LN scale | 1/sqrt(layer+1) |

## Training

Muon (matrix params) + AdamW (embeddings, scalars), WD=0.095, 786K tokens/step, seq_len=2048, warmdown_frac=0.72.

## Quantization

GPTQ with Fisher-information Hessians (64 calibration batches). Int6 SDClip (k=12.85) for adapter matrices, int8 for embeddings. Brotli-11 with byte-shuffle. Small tensors (adapter_scale, q_gain, attn_scale) stored as float16 passthrough.

## What Didn't Work (Negative Results)

| Experiment | Result | Lesson |
|-----------|--------|--------|
| **EMA on adapters** | +0.47 BPB | adapter_B starts at zero; EMA averages toward zero init for hundreds of steps |
| **AdamW TTT on adapters** | +0.009 BPB at lr=0.0005, 3 epochs | Adapters may need much lower TTT learning rate |
| **MLP 4x** | Fewer steps (413 vs 439) on 1xH100 | MLP=3 was the better speed/quality tradeoff |
| **MODEL_DIM=480** | Crash | head_dim=60 breaks flash_attn (not a multiple of 8) |
| **8xH100 distributed** | Pod killed at step 1000 | 5.6M tok/s confirmed working before credit exhaustion |

## Why This Matters

1. **Zero-cost backbone**: The random backbone consumes 0 bytes of the 16MB budget, leaving the full budget for learned parameters
2. **Depth recurrence is cheap for adapters**: Since adapters are much smaller than full layers, the memory overhead of looping is minimal
3. **EMA incompatibility is a real constraint**: This is a fundamentally different training regime where standard stabilization techniques (EMA, SWA) can harm performance
4. **Competitive with full-rank**: 1.322 BPB vs 1.321 BPB full-rank on the same hardware, while using a radically different parameter allocation strategy

## Run Command

```bash
# 1xH100
USE_RANDOM_ADAPTERS=1 ADAPTER_RANK=304 MLP_MULT=3 python3 train_gpt.py

# 8xH100
USE_RANDOM_ADAPTERS=1 ADAPTER_RANK=304 MLP_MULT=3 torchrun --nproc_per_node=8 train_gpt.py
```

## Credits

- Muon optimizer — modded-nanogpt baseline (kellerjordan)
- XSA — arXiv:2603.09078, PR #265 (@unnir)
- LoRA TTT reference — PR #461 (@Christopher-Lee-McClendon)
