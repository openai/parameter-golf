# Record: 1.0722 BPB — Improved TTT + HedgeMixer with Per-Layer LR Groups

**val_bpb: 1.0722** (3-seed mean, std 0.0086) | 15.66 MB max artifact | 8xH100 SXM, <600s eval

Built on PR #720 by @agalimova (XSA + HedgeMixer + legal TTT backbone).

## Results

| Seed | Steps | ms/step | TTT BPB | Eval Time | Artifact Bytes |
|------|-------|---------|---------|-----------|----------------|
| 1337 | 5847 | 99.6 | **1.0726** | 537s | 15,661,730 |
| 42 | 5847 | 99.6 | **1.0635** | 546s | 15,373,898 |
| 2025 | 5855 | 99.4 | **1.0806** | 531s | 15,655,453 |
| **Mean** | | | **1.0722** | | |

## What's New (vs PR #720)

This submission improves upon PR #720's XSA + HedgeMixer + legal TTT stack with a novel TTT recipe:

1. **Per-layer LR groups**: Output projections (`mlp.proj`, `attn.proj`) get 3x the base TTT LR; input projections (`mlp.fc`) get 0.5x. Rationale: output projections directly map representations to logit space and are the highest-value adaptation targets.
2. **Cosine LR schedule within TTT**: `lr = base * 0.5 * (1 + cos(pi * chunk/total_chunks))` — starts high for aggressive early adaptation, anneals to prevent overfitting on later chunks.
3. **4 TTT epochs** (vs 3) — more adaptation passes per chunk.
4. **Freeze only 1 block** (vs 2) — more parameters adapting = more capacity.
5. **Skip standalone sliding window eval** — TTT eval subsumes it, saving ~86s of eval budget to enable the extra epoch.

PR #720 baseline (stride=64): ~1.085 BPB. Our improvement: **1.0722** (-0.013).

## Base Architecture (inherited from PR #720)

- 11 layers, 512 dim, 8 heads, 8 KV heads (full MHA)
- MLP 3.5x expansion, LeakyReLU(0.5)^2 activation
- XSA on all 11 layers
- BigramHash(6144, dim=128) + SmearGate
- Partial RoPE (16/64 dims) + LN Scale + EMA(0.997)
- Int5 GPTQ-lite quantization + zstd compression
- Parameter banking with Parallel Muon optimizer
- LogisticContextMixer: 5 backward-looking experts (Neural, Unigram, Bigram, Trigram, Entropy)

## Setup and Run

```bash
# Environment: Python 3.12, PyTorch 2.9.1+cu128, Flash Attention 3
# Required: pip install sentencepiece zstandard

# On GCP with H100s:
export NCCL_NET=Socket
export SKIP_SLIDING=1  # skip standalone sliding eval to save time for TTT

# Run with specific seed
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py

# TTT configuration (all have defaults in code)
# TTT_EPOCHS=4 TTT_LR=0.0005 TTT_FREEZE_BLOCKS=1 TTT_CHUNK_TOKENS=32768
```

## Compliance

- [x] 3 seeds run on 8xH100 SXM (1337, 42, 2025)
- [x] All seeds train in <=600s (582s)
- [x] All seeds artifact <=16,000,000 bytes (max: 15,661,730)
- [x] All seeds eval in <=600s (max: 546s)
- [x] Score-first TTT: every token scored BEFORE any update that could use it
- [x] HedgeMixer uses backward-looking n-gram counts from scored tokens only
- [x] No network calls, external data, or validation leakage during eval
- [x] Reproducible from this directory
