# Non-Record: Polar Express Muon (Negative Result) -- val_bpb 1.0805

**This is a non-record submission documenting a negative result.** It does not beat the current merged SOTA or our best open PR.

**val_bpb: 1.08049** (single seed s42) | **2.79101 nats** | **~15.26 MB** | 8xH100 SXM, 600s | Score-First TTT

## What is Polar Express Muon?

The Muon optimizer uses Newton-Schulz (NS) iteration to approximate the matrix sign function for spectral preconditioning. The standard NS5 iteration uses a fixed 5-step cubic polynomial recurrence.

**Polar Express** (Amsel et al., arXiv:2505.16932) is an alternative set of polynomial coefficients for the same NS iteration. Instead of the standard `(3a, -a^3)` updates, Polar Express uses numerically optimized `(a, b, c)` triples that aim for faster convergence to the orthogonal projection. The coefficients were derived to minimize the spectral approximation error over typical gradient distributions.

The hypothesis was that better spectral approximation would translate to better optimization and lower val_bpb.

## Result

| Seed | Post-EMA BPB | Sliding BPB | **TTT BPB** | val_loss (nats) | Artifact |
|------|-------------|-------------|-------------|-----------------|----------|
| 42   | 1.08800     | 1.08214     | **1.08049** | 2.79101         | 15,998,547 |

**Baseline (standard Muon NS5, same stack, s42):** TTT BPB = 1.08006, val_loss = 2.78991

**Delta: +0.00043 BPB worse** (+0.00111 nats worse). Polar Express is slightly worse than standard Newton-Schulz on this stack.

## Why This Negative Result is Interesting

1. **Standard NS5 is already near-optimal** for this training regime. The Polar Express coefficients, despite being numerically optimized for faster convergence in isolation, do not improve end-to-end training quality on the sp8192 GPT stack.

2. **The optimization landscape matters more than the spectral approximation quality.** Even if Polar Express converges to the orthogonal projection in fewer iterations, the standard NS5 coefficients may interact better with the momentum buffer, learning rate schedule, and gradient noise in practice.

3. **Training throughput was identical** (both reach ~4900 steps in 588s), so the difference is purely in optimization quality, not speed.

## How to Enable

Set `USE_POLAR_EXPRESS=1` in the environment. The implementation adds a `zeropower_via_polar_express` function with pre-computed polynomial coefficients that replace the standard `zeropower_via_newtonschulz5` in the Muon optimizer step.

```bash
SEED=42 TTT_ENABLED=1 MUON_MOMENTUM=0.97 USE_POLAR_EXPRESS=1 \
  PARALLEL_RESIDUAL_START=7 QK_GAIN_INIT=5.0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1394 @clarkkev (SP8192 baseline)
- Polar Express coefficients from Amsel et al. (arXiv:2505.16932)
- Score-first TTT framework from PR #461
