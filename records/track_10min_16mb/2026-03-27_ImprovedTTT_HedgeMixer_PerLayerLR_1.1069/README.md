# Improved TTT + HedgeMixer: Per-Layer LR Groups + Cosine Schedule

**val_bpb: 1.1069** (3-seed mean, std 0.0202) | 15.66 MB max artifact | 8xH100 SXM, <600s eval

## Results

| Seed | Steps | ms/step | Sliding BPB | TTT BPB | Artifact Bytes | Train Time | TTT Eval Time |
|------|-------|---------|-------------|---------|----------------|------------|---------------|
| 1337 | 5847 | 99.6 | 1.1262 | **1.0952** | 15,661,730 | 582s | 469s |
| 42 | 5847 | 99.6 | 1.1265 | **1.0954** | 15,373,898 | 582s | 469s |
| 2025 | 5855 | 99.4 | 1.1265 | **1.1302** | 15,655,453 | 582s | 475s |
| **Mean** | | | 1.1264 | **1.1069** | | | |

## What's New

This submission improves upon the XSA + HedgeMixer + legal TTT family (building on techniques from PR #549, #720) with a novel TTT recipe that produces significantly better test-time adaptation:

### Novel TTT Recipe (the key contribution)
1. **Per-layer LR groups**: MLP output projections (`mlp.proj`, `attn.proj`) get 3x the base TTT LR; MLP input (`mlp.fc`) gets 0.5x. Output projections are the highest-value adaptation targets since they directly influence the logit distribution.
2. **Cosine LR schedule within TTT**: `lr = base * 0.5 * (1 + cos(pi * chunk/total_chunks))` — starts high, anneals to prevent overfitting on later chunks.
3. **Freeze only 1 block** (vs 2 in prior work) — more adaptation capacity.
4. **3 TTT epochs** with 32K token chunks, AdamW optimizer.

### Base Architecture (from prior work)
- 11 layers, 512 dim, 8 heads, 8 KV heads (full MHA)
- MLP 3.5x expansion, LeakyReLU(0.5)^2 activation
- XSA on all 11 layers
- BigramHash(6144, dim=128) + SmearGate
- Partial RoPE (16/64 dims) + LN Scale + EMA(0.997)
- Int5 GPTQ-lite quantization + zstd compression
- Parameter banking with Parallel Muon optimizer

### Eval-Time Techniques
- LogisticContextMixer: 5 backward-looking experts (Neural, Unigram, Bigram, Trigram, Entropy)
- Score-first legal TTT with the improved recipe above
- Sliding window stride=64

## Architecture Details

- **Parameters**: 33,317,980
- **Quantization**: Int5 GPTQ-lite (per-row clip search, 5 percentiles) + zstd compression
- **Embedding**: Tied, init std=0.005
- **Training**: 786,432 tokens/batch, warmdown 3500 iters, grad clip 0.3
- **Optimizer**: Parallel Muon (lr=0.025, momentum 0.92->0.99 warmup over 1500 steps) + AdamW for embeddings/scalars

## Setup and Run

```bash
# Environment: Python 3.12, PyTorch 2.9.1+cu128, Flash Attention 3
# Required: pip install sentencepiece zstandard

# On GCP with H100s, this NCCL fix is required:
export NCCL_NET=Socket

# Run with specific seed
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py

# TTT configuration (set via env vars, all have defaults in code)
# TTT_EPOCHS=3 TTT_LR=0.0005 TTT_FREEZE_BLOCKS=1 TTT_CHUNK_TOKENS=32768
```

## Compliance

- [x] 3 seeds run on 8xH100 SXM (1337, 42, 2025)
- [x] All seeds train in <=600s (582s)
- [x] All seeds artifact <=16,000,000 bytes (max: 15,661,730)
- [x] All seeds eval in <=600s (max: 561s = 86s sliding + 475s TTT)
- [x] Score-first TTT: every token scored BEFORE any update that could use it
- [x] HedgeMixer uses backward-looking n-gram counts from scored tokens only
- [x] No network calls, external data, or validation leakage during eval
- [x] Reproducible from this directory
