# Record: EngramLite + Gated Skips + Full GPTQ + FA3

**val_bpb: 1.1146** (1-seed, pending 3-seed confirmation) | **15.71 MB** | 8×H100 SXM

## Results

| Seed | Steps | ms/step | Sliding BPB (s64) | Artifact |
|------|-------|---------|--------------------|----------|
| 1337 | 6,667 | 87.9 | **1.1146** | 15,711,654 |
| 42 | — | — | pending | — |
| 2025 | — | — | pending | — |

## What's New

This submission combines innovations from PRs #1060, #1072, and #1089 into a single stack:

### From PR #1060 (coprime loader + Full GPTQ)
- **Coprime-stride multi-shard data pipeline** — diverse batches from coprime-stride block sampling
- **Full Hessian GPTQ** — Cholesky error compensation, 64-batch calibration in 6.7s
- **XSA on all 11 layers** — extended from last 4

### From PR #1089 (Turbo-Muon + EngramLite)
- **EngramLite** — multi-head bigram+trigram hash embeddings (8192 buckets, 2 heads, 2 orders, 32 dim/head) with learned sigmoid gate. Replaces BigramHash.
- **Sigmoid-gated skip connections** — `gate = sigmoid(skip_gates[i]); x = lerp(skip_weight*skip, x, gate)`. More expressive than additive skip.
- **LeakyReLU(0.3)²** — negative slope 0.3 instead of 0.5
- **Turbo-Muon** — 4 Newton-Schulz iterations (from 5). Faster optimizer step.
- **LR floor 0.05** — warmdown doesn't reach zero

### Environment
- **FlashAttention 3** (Hopper native): `flash_attn_3-3.0.0` pre-built wheel
- PyTorch 2.9.1+cu128, Triton 3.5.1
- CUDA Driver 580.126.09

## Architecture

- 11L, 512d, 8H/4KV (GQA), MLP 3× LeakyReLU(0.3)²
- XSA on all 11 layers, EngramLite(8192×2×2×32), SmearGate
- Sigmoid-gated U-Net skip connections
- Partial RoPE (16d), LN Scale 1/√(l+1)
- Shared ValueEmbedding (dim=128, layers 9-10)
- EMA (decay=0.997) + Tight SWA (every 50 steps, scale<0.2)
- Parallel Muon (4 NS steps) + Parameter Banking
- Full Hessian GPTQ int6 + LZMA compression

## Timing

| Phase | Time |
|-------|------|
| Training (6,667 steps @ 87.9ms) | 586s |
| GPTQ calibration | 6.7s |
| Sliding window eval (stride=64) | 93s |

## Reproduction

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install sentencepiece huggingface-hub datasets numpy tqdm

SEED=1337 \
NGRAM_BUCKETS=8192 NGRAM_HEADS=2 NGRAM_ORDERS=2 NGRAM_DIM_PER_HEAD=32 \
NEGATIVE_SLOPE=0.3 MUON_BACKEND_STEPS=4 LR_FLOOR=0.05 \
XSA_LAST_N=11 USE_GPTQ=1 GPTQ_RESERVE_MS=14000 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **PR #549** by @abaybektursun (base scaffold: LeakyReLU² + Parallel Muon)
- **PR #1060** by @resouer (coprime loader + Full GPTQ + XSA-all)
- **PR #1089** (Turbo-Muon + EngramLite + gated skips)
- **PR #1072** (fused Triton MLP kernel concept)
