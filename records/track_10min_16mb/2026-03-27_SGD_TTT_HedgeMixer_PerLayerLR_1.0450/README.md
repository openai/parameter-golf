# Record: 1.0450 BPB — SGD TTT + HedgeMixer with Per-Layer LR Groups

**val_bpb: 1.0450** (3-seed mean, std 0.012) | 15.67 MB max artifact | 8xH100 SXM, <545s eval

Built on PR #720 by @agalimova (XSA-all + HedgeMixer + legal TTT backbone).

## Results

| Seed | TTT BPB | Eval Time | Artifact Bytes |
|------|---------|-----------|----------------|
| 1337 | **1.0312** | 540s | 15,567,538 |
| 42 | **1.0503** | 533s | 15,672,158 |
| 2025 | **1.0535** | 544s | 15,152,756 |
| **Mean** | **1.0450** | | |

## What's New (vs PR #720)

Two key changes to the TTT recipe, both producing large improvements:

### 1. SGD Optimizer for TTT (main win: -0.041 BPB)
Switched TTT from AdamW (lr=0.0005) to **SGD with momentum=0.9** (lr=0.002). SGD provides more aggressive, direct parameter updates that better exploit the short adaptation window. This single change accounts for most of the improvement.

Run with: `TTT_OPTIMIZER=sgd TTT_LR=0.002`

### 2. Per-Layer LR Groups + Cosine Schedule (from our earlier PR)
- Output projections (`mlp.proj`, `attn.proj`): 3x base LR
- Input projections (`mlp.fc`): 0.5x base LR
- Cosine LR decay within TTT: `lr = base * 0.5 * (1 + cos(pi * chunk/total))`
- 4 TTT epochs, zero frozen blocks
- Skip standalone sliding eval to save time for TTT

### Ablation: Mixer vs TTT Attribution

| Config | BPB | Notes |
|--------|-----|-------|
| Full system (mixer + SGD TTT) | **1.0312** | Best (seed 1337) |
| AdamW TTT + mixer (previous PR) | 1.0726 | AdamW is the bottleneck |
| No mixer, AdamW TTT only | 1.1559 | TTT without mixer HURTS |
| No TTT (sliding only) | 1.1210 | Pure neural baseline |

Both mixer and TTT are necessary — the mixer provides the scoring framework that makes TTT adaptation effective.

## Base Architecture (inherited from PR #720)

- 11 layers, 512 dim, 8 heads, 8 KV heads (full MHA)
- MLP 3.5x expansion, LeakyReLU(0.5)^2 activation
- XSA on all 11 layers, BigramHash(6144, dim=128) + SmearGate
- Partial RoPE (16/64 dims) + LN Scale + EMA(0.997)
- Int5 GPTQ-lite + zstd compression, Parallel Muon optimizer
- LogisticContextMixer: 5 backward-looking experts (Neural, Unigram, Bigram, Trigram, Entropy)

## Setup and Run

```bash
export NCCL_NET=Socket
export SKIP_SLIDING=1
export TTT_OPTIMIZER=sgd
export TTT_LR=0.002

SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] 3 seeds on 8xH100 SXM (1337, 42, 2025)
- [x] All train <=600s (582s)
- [x] All eval <=600s (max 544s)
- [x] All artifact <=16,000,000 bytes (max 15,672,158)
- [x] Score-first TTT: every token scored BEFORE any update
- [x] HedgeMixer: backward-looking n-gram counts from scored tokens only
- [x] No network calls, external data, or validation leakage
