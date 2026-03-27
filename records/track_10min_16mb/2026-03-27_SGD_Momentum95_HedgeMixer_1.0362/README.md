# Record: 1.0362 BPB — SGD Momentum 0.95 TTT + HedgeMixer

**val_bpb: 1.0362** (3-seed mean, std 0.006) | 15.67 MB max artifact | 8xH100 SXM, <540s eval

Built on PR #720 by @agalimova. Improves on our earlier submissions (PR #953: 1.0722, PR #xxx: 1.0450).

## Results

| Seed | TTT BPB | Eval Time | Artifact Bytes |
|------|---------|-----------|----------------|
| 1337 | **1.0302** | 513s | 15,567,538 |
| 42 | **1.0365** | 533s | 15,672,158 |
| 2025 | **1.0419** | 539s | 15,152,756 |
| **Mean** | **1.0362** | | |

## What's New

Two key improvements over our previous SGD TTT submission:

### 1. Momentum 0.95 (vs 0.9)
Higher momentum provides smoother gradient accumulation, reducing oscillation during rapid adaptation. This improves both absolute score (-0.009 BPB) and seed variance (range 0.012 vs 0.022).

### 2. Comprehensive hyperparameter validation
We swept: LR (0.001/0.002/0.003), freeze depth (0/1/2), epochs (3/4/5), per-layer LR mult (2x/3x/4x), fc mult (0.3x/0.5x), trigram hash (64K/128K), chunk size (24K/32K), momentum (0.9/0.95).

Best config: SGD lr=0.002, momentum=0.95, 4 epochs, freeze=0, proj=3x, fc=0.5x, 32K chunks.

Run: `TTT_OPTIMIZER=sgd TTT_LR=0.002 TTT_MOMENTUM=0.95 SKIP_SLIDING=1 SEED=1337 torchrun --nproc_per_node=8 train_gpt.py`

## Ablation

| Config | Mean BPB | Range | Notes |
|--------|----------|-------|-------|
| **This (SGD m=0.95)** | **1.0362** | **0.012** | Best |
| SGD m=0.9 (prev PR) | 1.0450 | 0.022 | Higher variance |
| AdamW (first PR) | 1.0722 | 0.017 | AdamW is bottleneck |
| No mixer | ~1.156 | — | Mixer essential |
| No TTT | ~1.121 | — | Neural baseline |

## Base Architecture (from PR #720)
- 11L/512d/8H/8KV, MLP 3.5x, LeakyReLU(0.5)², XSA-all-11
- BigramHash(6144), Partial RoPE, Int5 GPTQ-lite + zstd
- LogisticContextMixer: 5 backward-looking experts

## Compliance
- [x] 3 seeds on 8xH100 SXM, all train ≤600s, eval ≤600s, artifact ≤16MB
- [x] Score-first legal TTT + backward-looking HedgeMixer
- [x] No external data access during eval
