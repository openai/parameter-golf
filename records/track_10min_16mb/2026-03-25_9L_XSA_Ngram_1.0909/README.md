# Record: 9L XSA-all + Multi-order N-gram Backoff + Entropy-Adaptive Alpha

**val_bpb: 1.0238** (2-seed mean) | **14.7 MB** | 8xH100 SXM

## Results

| Seed | Pre-ngram BPB | Post-ngram BPB | N-gram gain | Artifact |
|------|---------------|----------------|-------------|----------|
| 1337 | 1.1696 | **1.0233** | -0.146 | 14.69 MB |
| 42 | 1.1711 | **1.0244** | -0.147 | 14.69 MB |
| **Mean** | **1.1703** | **1.0238** | **-0.147** | |

## Key Techniques

**Training (9L/512d, 17.6M params)**
- 9L transformer, 512d, 8H/4KV GQA, MLP 2x, LeakyReLU(0.5)²
- XSA on all 9 layers, SmearGate, BigramHash(4096)
- OrthoInit, LN Scale, Partial RoPE (25%)
- Int8 per-row quantization + zstd-22

**Eval: Multi-order N-gram Backoff + Entropy-Adaptive Alpha (-0.147 BPB)**
- Orders 2-7 with highest-order-first backoff (separate hash tables per order)
- Entropy-adaptive alpha: `alpha = 0.05 + 0.55 * sigmoid(2*(H-4.0))`
- 4M-bucket hash tables per order
- Score-first, backward-looking, no target-aware gating
- ~157s eval time

## Reproduce

```bash
SEED=1337 NUM_LAYERS=9 MLP_MULT=2 QUANT_BITS=8 GPTQ_ENABLED=0 PRUNE_PCT=0 NGRAM_ENABLED=1 \
  torchrun --nproc_per_node=8 train_gpt.py
```
