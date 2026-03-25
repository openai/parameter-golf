# BitNet b1.58 Ternary Quantization

**val_bpb: 1.2185** (3-seed mean, std 0.0018) | **~14.4 MB** | 8×H100 SXM

## Summary

First submission using BitNet-style ternary quantization {-1, 0, +1} instead of int6. Ternary uses ~1.58 bits/weight vs 6 bits/weight, allowing 2x more parameters in the same size budget.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | **Post-quant bpb** | Artifact |
|------|----------|-------|-------------------|----------|
| 1337 | 214ms | 2,794 | **1.2208** | 14,404,504 |
| 42 | 214ms | ~2,800 | **1.2183** | 14,401,571 |
| 2025 | 214ms | ~2,800 | **1.2163** | 14,399,674 |
| **Mean** | **214ms** | **~2,800** | **1.2185 (std 0.0018)** | |

## Key Innovation: Ternary Quantization

Instead of int6 GPTQ (6 bits/weight), we use ternary weights {-1, 0, +1} (1.58 bits/weight):

```python
# BitNet b1.58 quantization (from Microsoft paper)
gamma = weights.abs().mean()  # global scale per tensor
w_ternary = round(weights / gamma).clamp(-1, 1)  # {-1, 0, +1}

# Straight-Through Estimator for training
w = w + (w_quantized - w).detach()  # forward: quantized, backward: full precision
```

### Parameter Budget

| Quantization | Bits/weight | Params in 16MB |
|--------------|-------------|----------------|
| int6 (SOTA)  | 6 bits      | ~21M           |
| **Ternary**  | **1.58 bits** | **~80M**     |

### Packing Format

5 ternary values per byte using base-3 encoding:
```
Encoding: -1→0, 0→1, +1→2
Pack: v0 + 3*v1 + 9*v2 + 27*v3 + 81*v4
3^5 = 243 < 256, fits in uint8
```

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 18 |
| Model dim | 640 |
| Heads | 8 (4 KV) |
| MLP | 3× |
| XSA | Last 4 layers |
| Parameters | 67.5M |
| Quantization | **Ternary (BitNet 1.58-bit)** |
| EMA | Disabled (incompatible with ternary) |
| SWA | Disabled (incompatible with ternary) |

### Critical Finding: EMA Incompatibility

EMA averages full-precision weights, but forward uses ternary. This creates a mismatch:

| Config | Post-quant val_bpb | Degradation |
|--------|-------------------|-------------|
| With EMA | 3.78 | +0.52 bpb |
| **Without EMA** | **3.27** | **+0.01 bpb** |

**Solution:** Always use `EMA_ENABLED=0` with ternary.

## Run Command

```bash
TERNARY_ENABLED=1 EMA_ENABLED=0 SWA_ENABLED=0 \
NUM_LAYERS=18 MODEL_DIM=640 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Analysis

### Why Ternary is Slower

Our ternary implementation achieves 214ms/step vs SOTA's 84ms/step (~2.5x slower):

1. **Quantization overhead** - Every forward computes `gamma = abs(w).mean()` + `round(w/gamma).clamp(-1,1)`
2. **No tensor core optimization** - Not using FP8/int8 tensor cores
3. **STE overhead** - `w + (w_q - w).detach()` trick

This results in ~2,800 steps vs SOTA's ~7,185 steps in 10 minutes.

### Comparison to SOTA

| Metric | SOTA (1.1194) | Ternary (1.2185) |
|--------|---------------|------------------|
| val_bpb | 1.1194 | 1.2185 |
| Gap | - | +0.099 |
| Steps | ~7,185 | ~2,800 |
| Step time | 84ms | 214ms |
| Quantization | int6 | ternary |
| Parameters | ~33M | 67.5M |

### Potential Improvements

1. **Fused CUDA kernel** for ternary quantization + matmul
2. **Use int8 tensor cores** instead of bf16 with ternary simulation
3. **Pre-compute gamma** instead of recomputing every forward

## References

- [BitNet b1.58 Paper](https://arxiv.org/abs/2402.17764) - Microsoft Research, Feb 2024
- [Parameter Golf Challenge](https://github.com/openai/parameter-golf) - OpenAI
