# Record: Fused Triton MLP + Full Hessian GPTQ + Coprime-Stride Loader + XSA-all + BigramHash(2816) (val_bpb 1.1116)

**Author:** @barneywohl  
**Date:** 2026-03-30  
**Hardware:** 8×H100 SXM (GCP)

Built on [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun and [PR #1060](https://github.com/openai/parameter-golf/pull/1060) techniques.

## Results (8×H100 SXM, no TTT)

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 1337 | **1.1110** | 15,977,983 |
| 42 | **1.1121** | ~15,980,000 |
| 2024 | **1.1118** | ~15,980,000 |
| **Mean ± Std** | **1.1116 ± 0.0005** | |

## What is New

This submission combines five orthogonal improvements on the PR #549 scaffold:

### 1. Fused Triton MLP Kernel
Custom Triton kernel fusing `leaky_relu(x, 0.5).square()` into a single pass, replacing two separate PyTorch operations. Saves ~1.8ms/step (~137 extra training steps over 600s budget) with no accuracy change. This is our unique contribution — no other submission uses a fused MLP kernel.

### 2. Full Hessian GPTQ (Frantar et al., ICLR 2023)
Inspired by [PR #634](https://github.com/openai/parameter-golf/pull/634) and [PR #1019](https://github.com/openai/parameter-golf/pull/1019). Replaces GPTQ-lite with full Cholesky error compensation + actorder column permutation + 5-way percentile clip sweep. Calibrates on 64 training batches in ~6.4s.

### 3. Coprime-Stride Multi-Shard Data Loader
Inspired by [PR #726](https://github.com/openai/parameter-golf/pull/726). Replaces sequential token streaming with coprime-stride access patterns across 8-32 shards per batch. Uses `np.memmap` for zero-copy reads and diversity-weighted shard sampling with alpha annealing (0.90→0.50). Forces position-invariant long-range context learning.

### 4. XSA on All 11 Layers
Extends Exclusive Self-Attention from 4 layers to all 11, subtracting each head's self-projection from attention output. Forces every layer to attend to other tokens rather than copying self.

### 5. BigramHash(2816×112)
Enlarged from PR #549's BigramHash(1536×128). Captures more bigram patterns with a slightly narrower projection.

## Stack Summary

- LeakyReLU(0.5)² activation with fused Triton kernel
- Parallel Muon optimizer + AdamW for embeddings
- 11-layer, 512-dim, 8-head transformer with parameter banking
- EMA (decay=0.997) + linear warmdown (3500 iters)
- Full Hessian GPTQ int6 + LZMA compression
- Coprime-stride data loader with shard mixing
- XSA on all 11 layers, BigramHash(2816×112), SmearGate
- Sliding window evaluation (stride=64, seq_len=2048)
- fullgraph=True torch.compile

| Component | Time Budget |
|-----------|------------|
| Training | 590s (7,080-6,900 steps at 85ms/step) |
| GPTQ calibration + quantization | 10s |
| Sliding window eval (stride=64) | 80s |

## Rule Compliance

- ✅ Standard F.cross_entropy scoring (softmax, sum=1)
- ✅ No TTT, no eval-time adaptation, no unnormalized scoring
- ✅ Full `fineweb_val_*` split in canonical sorted order with tokenizer-derived byte accounting
- ✅ Artifact < 16,000,000 bytes (all 3 seeds)
- ✅ Training < 600s, eval < 600s
- ✅ Causal sliding-window evaluation on the full validation split (stride=64)

## Reproduction

```bash
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
FUSED_MLP=1 \
USE_GPTQ=1 \
GPTQ_RESERVE_MS=10000 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2816 \
BIGRAM_DIM=112 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base scaffold**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun
- **Data pipeline**: [PR #726](https://github.com/openai/parameter-golf/pull/726) by @DeepReinforce
- **Full Hessian GPTQ**: [PR #634](https://github.com/openai/parameter-golf/pull/634) by @raahilshah, [PR #1019](https://github.com/openai/parameter-golf/pull/1019) by @abaybektursun
- **XSA**: [PR #287](https://github.com/openai/parameter-golf/pull/287) by @jfprincz
