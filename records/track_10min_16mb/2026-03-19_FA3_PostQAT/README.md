# FA3 + Post-Training QAT

## Summary

Two changes to the NaiveBaseline (9L/512d/8H/4KV relu^2, 1.2244 BPB):

1. **Flash Attention 3 (Hopper kernels)**: Replaced PyTorch's built-in `scaled_dot_product_attention` (FA2) with FA3's `flash_attn_func` using native BSHD tensor layout. Eliminates transpose operations and leverages H100 TMA hardware for ~8% faster step times (40ms vs 43.5ms). More training steps in the 10-minute window.

2. **Post-Training Quantization-Aware Training**: After the main training loop, reserves 30-45s for a dedicated QAT phase. Each step replaces weight matrices with per-row int8 quantize/dequantize (straight-through estimator), runs forward with quantized weights, restores originals before backward, and updates with Adam (lr=0.001). Reduces int8+zlib quantization penalty from +0.0071 to ~+0.0020 BPB.

FA3 gracefully falls back to PyTorch SDPA if `flash_attn_interface` is not available.

## Architecture

Unchanged from baseline:
- 9 transformer layers, d_model=512, 8 attention heads, 4 KV heads (GQA)
- relu^2 MLP (mlp_mult=2), tied embeddings, vocab=1024
- Muon optimizer for matrix params, Adam for embeddings/scalars
- U-Net skip connections, logit softcapping (30.0)
- RoPE (base=10000), RMSNorm

## Key Hyperparameters

| Parameter | Value |
|---|---|
| matrix_lr (Muon) | 0.04 |
| embed_lr (tied) | 0.05 |
| warmdown_iters | 1200 |
| post_qat_seconds | 30 (default) / 45 (used in runs) |
| post_qat_lr | 0.001 |
| train_batch_tokens | 524,288 |
| seed | 1337 (default) |

## Results

Tested on Vast.ai 8xH100 SXM instance with 10 training shards (competition server uses 25).

Local baseline (unmodified `train_gpt.py` on same machine):
- pre-quant: 1.2229, post-quant: **1.2300**, quant penalty: +0.0071

This submission (FA3 + 45s QAT):

| Seed | Steps | Pre-quant | Post-quant (val_loss) | Post-quant (val_bpb) | Quant penalty |
|------|-------|-----------|----------------------|---------------------|---------------|
| 1337 | 13,835 | 1.2219 | 2.0673 | 1.2244 | +0.0025 |
| 42 | 13,815 | 1.2225 | 2.0677 | 1.2246 | +0.0021 |
| 7 | 13,820 | 1.2220 | 2.0674 | 1.2244 | +0.0024 |
| **mean** | | | **2.0675** | **1.2245** | **+0.0023** |

Local baseline (unmodified `train_gpt.py`, same machine, seed 1337):
- post-quant val_loss: 2.0768, val_bpb: 1.2300, quant penalty: +0.0071

**Improvement vs local baseline: -0.0055 BPB / -0.0093 val_loss nats**

Note: Absolute BPB values differ from the competition leaderboard because our Vast.ai instance has 10 training shards vs the competition server's 25. The relative improvement is data-independent (QAT reduces quantization penalty regardless of training data).

## Implementation Details

### FA3 Integration
- Uses `flash_attn_interface.flash_attn_func` from Dao-AILab/flash-attention Hopper branch
- Native BSHD (batch, seqlen, heads, headdim) tensor layout throughout attention
- Rotary cache stored as `(1, S, 1, D/2)` for BSHD broadcasting
- Falls back to `F.scaled_dot_product_attention` with BHSD layout if FA3 unavailable
- Compatible with `torch.compile(fullgraph=True)`

### Post-Training QAT
- Runs after main training loop completes (wallclock-based)
- Uses uncompiled model to avoid recompilation cost
- Per-row int8 quantization with max-based clipping (fast, not MSE-optimal)
- STE: forward uses quantized weights, backward passes through to original weights
- Fresh Adam optimizer with lr=0.001, 0.9/0.999 betas
- Typically completes 200-350 steps in 30-45 seconds
