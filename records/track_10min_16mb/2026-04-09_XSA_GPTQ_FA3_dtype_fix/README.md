# Record: XSA-all + GPTQ + BigramHash 3072 + EMA + FA3 dtype fix

**val_bpb: 1.1220** (sliding window, stride=64) | **~15.9 MB** artifact | 8×H100 SXM, 600s

## Results

| Seed | Steps | ms/step | Sliding BPB (s64) | Artifact |
|------|-------|---------|-------------------|----------|
| 1337 | 6,244 | 96ms | **1.1220** | ~15.9 MB |

## Architecture

Built on the PR #1019 stack with one key addition: FA3 dtype compatibility wrapper enabling native Hopper attention on PyTorch 2.5.1 (which lacks auto-casting for FA3).

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3× (1536) with LeakyReLU(0.5)² |
| Attention | XSA on all 11 layers + FA3 with bf16 wrapper |
| BigramHash | 3072 × dim=112 |
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

- 8× NVIDIA H100 80GB HBM3 SXM (Vast.ai, Nebraska)
- PyTorch 2.5.1+cu124, CUDA 12.4
- Flash Attention 3 (compiled from source for Hopper)
- 96ms/step average

## Authors

Gavin Saunders & Tron (Claude Code agent)
