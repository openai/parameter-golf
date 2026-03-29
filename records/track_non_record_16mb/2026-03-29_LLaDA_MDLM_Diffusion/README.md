# LLaDA-MDLM: Masked Diffusion Language Model

**val_var_bpb: 1.1465** (512 eval steps) | **~33M params** | 1x NVIDIA GB10 (Project DIGITS)

Non-record submission: first discrete diffusion model to beat the AR baseline (1.22 BPB) in parameter-golf.

## Results

| VAR_EVAL_STEPS | val_var_bpb |
|----------------|-------------|
| 64 | 1.1571 |
| 128 | 1.1508 |
| 256 | 1.1482 |
| **512** | **1.1465** |

Comparison:

| Model | BPB |
|-------|-----|
| AR SOTA (merged #1, PR #549) | 1.1194 |
| **This submission (MDLM diffusion)** | **1.1465** |
| AR baseline | 1.2244 |
| PR #820 (MDLM, previous best diffusion) | 1.625 |
| PR #905 (prefix-suffix diffusion) | 1.859 |

## Approach

Bidirectional transformer trained as a masked diffusion language model using the MDLM framework (Sahoo et al., 2024). The model predicts masked tokens given partially corrupted input, with corruption controlled by a log-linear noise schedule.

### Architecture

- 11 layers, 512 dim, 8 heads, MLP 3x (ReLU^2)
- Bidirectional attention (no causal mask)
- adaLN timestep conditioning (sigma embeddings modulate each layer)
- Frozen visible-token logits in `subs_log_probs` (MDLM key technique)
- RoPE positional embeddings
- 32,987,776 parameters

### Training

- MDLM continuous-time ELBO loss with log-linear noise schedule
- dsigma-weighted loss on masked tokens only
- Antithetic time sampling for lower variance
- AdamW optimizer (lr=6e-4, warmup 300, warmdown 1500 steps)
- Sequence length 2048, effective batch size 32 (8 x 4 gradient accumulation)
- 6000 steps on 100M FineWeb SP-1024 tokens
- Training time: ~189 min on 1x NVIDIA GB10

### Evaluation

Discrete absorbing-mask variational ELBO (from the MDLM paper). This is the proper coding-theoretic upper bound:

```
KL(q(x_T | x_0) || p(x_T)) + sum_t E_q KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t))
```

discretized into T steps. More steps = tighter bound. At 512 steps the bound is 1.1465 BPB.

BPB approximation: bits / (tokens x 4.3 bytes/token). The 4.3 factor is the approximate average bytes per SP-1024 token.

## Hyperparameter Search (3 Rounds)

We ran three rounds of systematic sweeps (500 steps each, 27 experiments total) to identify what works for diffusion LMs in the parameter-golf setting:

### Round 1: Architecture + LR + masking eps (12 experiments)

| Finding | Effect |
|---------|--------|
| **eps=0.1 >> eps=0.001** | -0.44 loss (biggest win) |
| Wider (8L 640d) > deeper (14L 384d) | -0.25 loss |
| LR=2e-3 > LR=6e-4 > LR=1e-3 | Non-monotonic |
| ReLU^2 > LeakyReLU(0.5)^2 | -0.16 loss |

### Round 2: AR techniques for diffusion (8 experiments)

| Finding | Effect |
|---------|--------|
| 8L 640d wider + adaLN | Best overall (-0.36 vs baseline) |
| GQA4 + adaLN + bigram | Positive interaction (individually each hurts) |
| adaLN timestep conditioning | Small improvement (-0.005) |
| BigramHash | Hurts for diffusion (unlike AR) |
| eps=0.2 | Too aggressive |

### Round 3: Creative/Karpathy-style (7 experiments)

| Finding | Effect |
|---------|--------|
| **Prefix conditioning (25% always visible)** | **-0.98 loss** (but cheats on eval) |
| Cosine masking schedule | -0.29 loss |
| Self-conditioning | +0.71 loss (hurts) |
| Depth recurrence 2x | +0.11 loss (hurts) |
| Hybrid causal (last 3 layers) | -0.04 loss (marginal) |

Key insight: prefix conditioning dramatically reduces training loss but the model becomes dependent on the prefix and doesn't generalize to the ELBO eval. The proper MDLM training objective with log-linear noise + frozen visible tokens (not uniform masking) is what ultimately works.

## Why This Matters

1. **First diffusion model to beat AR baseline in parameter-golf.** Previous diffusion submissions (PR #820, #904, #905) all scored worse than the naive AR baseline. Our MDLM implementation closes the gap.

2. **The eval method is critical.** Our earlier attempts using MC ELBO sampling gave 2.41 BPB — switching to the proper discrete absorbing-mask ELBO gave 1.15 BPB on the same model. The eval is not just a metric, it determines whether diffusion looks competitive.

3. **Diffusion-specific hyperparameters matter.** Standard AR tricks (LeakyReLU, BigramHash, prefix conditioning) don't transfer directly to diffusion. Higher masking eps (0.1 vs 0.001) and wider architectures are the key levers.

## Hardware

NVIDIA GB10 (Project DIGITS) — Grace Blackwell desktop, 130GB unified memory, CUDA 13.0. Single GPU, no torch.compile (ARM CPU), no Flash Attention 3.

## Credits

- MDLM framework: Sahoo et al. (2024), "Simple and Effective Masked Diffusion Language Models"
- LLaDA: Nie et al. (2025), "Large Language Diffusion with Masking"
- PR #820 (mtybadger): first MDLM implementation in parameter-golf, discrete ELBO eval code
- nanoLLaDA (Lukas Xue): minimal LLaDA implementation
- OpenAI Parameter Golf baseline and community
