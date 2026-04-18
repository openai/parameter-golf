# Record: XSA-all + GPTQ + BigramHash 3072 + EMA + FA3 dtype fix

**val_bpb = 1.1161** (3-seed mean, std 0.0009) | **< 16 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Steps | ms/step | Sliding BPB (s64) | Artifact |
|------|-------|---------|-------------------|----------|
| 42   | ~6,900 | ~87ms | **1.1166** | 15,935,215 |
| 314  | ~6,900 | ~87ms | **1.1151** | 15,938,283 |
| 999  | ~6,900 | ~87ms | **1.1165** | 15,933,311 |
| **Mean** | | | **1.1161** | |
| **Std** | | | **0.0009** | |

## Architecture

Built on the PR #1019 stack with one key addition: FA3 dtype compatibility wrapper enabling native Hopper attention.

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers + FA3 with bf16 wrapper |
| BigramHash | 3072 x dim=112 |
| RoPE | Partial (16/64 dims) |
| EMA | decay=0.997, every step |
| Quantization | Full Hessian GPTQ, AR self-gen calibration |
| Compression | int6+lzma |

## Key Contribution

**FA3 dtype compatibility wrapper** — enables Flash Attention 3 Hopper kernels on environments where PyTorch doesn't auto-cast to bf16 for FA3 calls:

```python
from flash_attn_interface import flash_attn_func as _flash_attn_3_raw
def flash_attn_3_func(q, k, v, causal=True):
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    return _flash_attn_3_raw(q, k, v, causal=causal)
```

## Configuration

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=11 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware

- 8x NVIDIA H100 80GB HBM3 SXM (Vast.ai)
- PyTorch 2.9.1+cu128, CUDA 12.8
- Flash Attention 3 (Hopper native)

## Compliance

- Pure neural, no TTT/SLOT/n-gram-cache
- All artifacts under 16,000,000 bytes on all 3 seeds
- Training under 600s on all 3 seeds
- Sliding window eval (stride=64), strictly causal

## Author

Gavin Saunders (@G3sparky)
