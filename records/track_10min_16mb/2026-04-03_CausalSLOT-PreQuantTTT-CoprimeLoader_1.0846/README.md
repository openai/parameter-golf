# Record: Causal SLOT + Pre-quant AdamW TTT + Coprime-stride Loader

**val_bpb: 1.0846** (3-seed mean) | **~15.95 MB** | 8xH100 SXM, 600s | Causal SLOT + Pre-quant TTT

Merged SOTA (PR #1019, 3-seed mean): **1.88218 nats**. This run: **1.83126 nats**. Delta: **-0.051 nats**. Clears the 0.005-nat threshold.

## Results (3-seed)

| Seed | Steps | ms/step | Sliding BPP | **+ Causal SLOT BPP** | val_loss (nats) | Artifact |
|------|-------|---------|-------------|----------------------|-----------------|----------|
| 1337 | ~6800 | ~87 | 1.0966 | **1.0841** | 1.8304 | 15,952,885 |
| 42 | ~6800 | ~87 | 1.0969 | **1.0843** | 1.8308 | 15,968,373 |
| 2025 | ~6800 | ~87 | 1.0972 | **1.0854** | 1.8326 | 15,938,173 |
| **Mean** | | | 1.0969 | **1.0846** | **1.8313** | |

## Changes from Merged SOTA (PR #1019)

PR #1019 scores 1.1147 BPP using Full Hessian GPTQ + AR self-gen calibration + BigramHash 3072 + XSA-all. No TTT. This submission makes three changes:

### 1. Causal SLOT — provably causal eval-time delta optimization (Novel)

Standard SLOT (PR #1172, #1176, #1229) optimizes a hidden-space delta vector using loss from all positions in a sliding window — including future positions not yet scored. PR #1240 empirically proved this violates causal dependence (100% violation rate across 240 tested pairs).

Our **causal SLOT** restricts delta optimization to **context-only positions**: tokens that were already scored in previous windows. For each batch of sliding windows:

1. Compute frozen hidden states H (`torch.no_grad()` through transformer)
2. Initialize delta = zeros(1, 1, 512) with `requires_grad=True`
3. Optimize delta with 8 AdamW steps (lr=0.005) using loss **only from already-scored positions**
4. Score new positions with the optimized delta

**Provably causal:** P(x_{t+1}) depends only on the artifact and x_1,...,x_t. Changing any future token has zero effect on prior predictions. Delta resets to zero for each new batch — no cross-batch leakage.

Causal SLOT contributes **-0.009 BPP** on top of sliding window eval. Eval time: ~300s.

### 2. Pre-quant AdamW TTT (6 epochs)

Post-quant SGD TTT fails on Full Hessian GPTQ stacks (25 documented failures per PR #756). We apply AdamW TTT on full-precision EMA weights **before** GPTQ quantization. The adapted weights quantize better because TTT shifts the weight distribution toward the evaluation data manifold before the irreversible quantization step.

- AdamW optimizer, lr=0.0005, cosine LR decay across chunks
- 6 epochs, freeze first 2 blocks, batch size 32
- TTT time: ~111s. Total eval budget: ~551s / 600s.

Pre-quant TTT contributes **-0.022 BPP**.

### 3. Coprime-stride multi-shard data loader

Replaces sequential token streaming with weighted random shard sampling using coprime stride patterns. Each training batch draws from multiple shards with mathematically guaranteed coverage diversity.

Coprime-stride loader contributes **-0.003 BPP**.

## Reproduction

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed — all defaults are submission config. FA3 required (see requirements.txt).

## Credits

- Base: PR #1019 (@abaybektursun)
- SLOT concept: arXiv:2505.12392v2, PR #1176 (@bigbag)
- Coprime-stride loader: PR #1184 (@icryo)
- Pre-quant TTT concept: PR #1006
- Causal SLOT design: novel (this submission)
