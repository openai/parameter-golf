# Masked Diffusion Language Model (MDLM)

**val_var_bpb: 1.1465** (512 eval steps) | **33M params** | 2xH100 80GB HBM3 (TensorPool) | Non-record

First discrete diffusion model to beat the AR baseline (1.22 BPB) in parameter-golf. Developed on NVIDIA GB10 (Project DIGITS), validated on 2xH100. 8xH100 unavailable on RunPod (see #821) and TensorPool; extrapolated training time on 8xH100: ~8 min (within budget).

## Results

| Model | BPB |
|-------|-----|
| AR SOTA (merged #1, PR #549) | 1.1194 |
| **This (MDLM diffusion)** | **1.1465** |
| AR baseline | 1.2244 |
| PR #820 (previous best diffusion) | 1.625 |

Validated on 2xH100 80GB HBM3 (TensorPool): 1.154 BPB, 31 min training at 209K tok/s. Extrapolated to 8xH100: ~8 min, within the 10-min budget. We were unable to provision 8xH100 from RunPod (see #821) or TensorPool (insufficient capacity). We plan to rerun on 8xH100 when available.

## Approach

Bidirectional transformer with MDLM training (Sahoo et al., 2024). The model predicts masked tokens given partially corrupted input, with corruption controlled by a log-linear noise schedule.

- 11 layers, 512 dim, 8 heads, MLP 3x (ReLU^2), RoPE
- adaLN timestep conditioning (sigma embeddings modulate each layer)
- Frozen visible-token logits in `subs_log_probs`
- MDLM continuous-time ELBO loss, antithetic time sampling
- AdamW (lr=6e-4, warmup 300, warmdown 1500), seq len 2048, batch 32
- 6000 steps on 100M FineWeb SP-1024 tokens

## Evaluation

Discrete absorbing-mask variational ELBO, discretized into T steps. More steps = tighter bound:

| Steps | BPB |
|-------|-----|
| 64 | 1.1571 |
| 128 | 1.1508 |
| 256 | 1.1482 |
| 512 | 1.1465 |

BPB uses approximate byte counting (~4.3 bytes/token).

## What We Learned

We ran 27 short experiments (500 steps each) across 3 rounds of sweeps to understand what works for diffusion LMs in this setting.

**What helps:**
- Masking eps=0.1 instead of 0.001 (default). Biggest single improvement.
- Wider models (8L 640d) over deeper (14L 384d) at same param count.
- Proper MDLM loss (log-linear noise + dsigma weighting + frozen visible tokens).
- Discrete ELBO eval. Our MC ELBO gave 2.41 BPB; discrete ELBO gave 1.15 on the same model.

**What doesn't transfer from AR:**
- LeakyReLU(0.5)^2 — worse than ReLU^2 for diffusion.
- BigramHash embeddings — hurts (bigrams assume sequential order; diffusion sees random masked order).
- Prefix conditioning — reduces training loss dramatically but model becomes dependent on it.

## Hardware

Developed on NVIDIA GB10 (Project DIGITS). Validated on 2xH100 80GB HBM3 via TensorPool.

## Credits

- MDLM: Sahoo et al. (2024), "Simple and Effective Masked Diffusion Language Models"
- LLaDA: Nie et al. (2025), "Large Language Diffusion with Masking"
- PR #820 (mtybadger): first MDLM in parameter-golf, discrete ELBO eval
- nanoLLaDA (Lukas Xue): minimal LLaDA reimplementation
