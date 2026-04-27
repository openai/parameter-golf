# The Frugendorff Squared — Fractal Weight Sharing + MLP 4x (val_bpb: 1.1478)

## Summary

Non-record submission exploring a novel approach: **fractal weight sharing** enables MLP 4x expansion within the 16MB artifact budget. 6 unique transformer blocks are looped 2 times each, providing 12 effective layers of depth with only 6 blocks worth of parameters. The freed parameter budget is reinvested into 4x MLP expansion, which provides a significant quality boost over 3x MLP.

## Key Insight

MLP 4x is a powerful quality lever (2%+ relative BPB improvement over 3x), but fitting 12 unique layers with MLP 4x in 16MB is impossible with standard int6 quantization. Fractal weight sharing solves this: 6 unique layers × 2 loops = 12 effective depth at ~60% of the parameter cost. The compression pays for the bigger MLP.

## Architecture

- **6 unique transformer blocks × 2 fractal loops = 12 effective depth**
- dim=640, 10 attention heads, 5 KV heads (GQA 2:1), head_dim=64
- **MLP 4x expansion** (hidden=2560) with relu-squared activation
- Orthogonal loop position embeddings (QR-initialized)
- Partial RoPE (16/64 dims) + NTK-aware scaling
- LN Scale Factor 1/sqrt(layer_idx+1)
- U-Net skip connections within each loop iteration
- SmearGate + BigramHash (2048 buckets, dim=128)
- Shared Value Embedding (dim=128)
- XSA on last 2 unique layers
- Logit softcap 30.0, tied embeddings

## Training

- Muon optimizer (matrices): lr=0.025, momentum=0.99
- AdamW (embeddings): lr=0.035, (scalars): lr=0.025
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3,500 iters (wallclock-based)
- SWA: every 50 steps when scale<0.2
- Late QAT: int6 fake-quantization when LR scale<0.15
- Late Training Replay: 2-epoch replay of last 100 training batches at 10% LR
- Self-distillation: EMA teacher, 50 steps, temp=2.0, alpha=0.7
- EMA: decay=0.997, applied after distillation

## How Fractal Weight Sharing Works

Each training step, the input passes through the 6 unique blocks twice (2 loops). Each loop adds a learned orthogonal position embedding so the shared weights can differentiate which pass they're executing. The U-Net skip connections operate within each loop iteration, providing encoder-decoder structure at each depth level.

This is NOT test-time training on validation data. The loops happen during standard training forward passes. At eval time, the model runs the same 2-loop forward pass deterministically.

## Quantization

- Int6 per-row for MLP + attention weights
- Int8 per-row for embeddings
- Control tensors in fp32
- zstd level 22 compression

## Results

| Metric | Value |
|--------|-------|
| Steps | 4,390 in 600s at 136.7ms/step |
| Pre-quant val_bpb (post-EMA) | 1.1570 |
| Post-quant roundtrip val_bpb | 1.1716 |
| **Sliding window val_bpb** | **1.1478** |
| Quant gap | 0.0146 |
| Artifact size | 15,154,098 bytes (15.15 MB) |
| Model params | 28,224,320 |

## Run

```bash
NUM_LAYERS=6 NUM_LOOPS=2 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4 \
torchrun --nproc_per_node=8 train_gpt_frugendorff_squared.py
```

## No TTT on Validation Data

This submission does not perform test-time training on validation/evaluation tokens. All training (including late replay and distillation) uses training data only. Fully compliant with issue #402.
