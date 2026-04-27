# PSO: Persistent Spectral Optimizer on PR #1394 stack
val_bpb: TBD | 8×H100 SXM

## Core Idea

Replace Muon's Newton-Schulz polar projection with a **persistent low-rank spectral basis** that is cached and slowly updated across optimizer steps. Updates are computed in r×r persistent coordinates via a noise-aware soft-polar transform, then reconstructed to full rank with an optional residual path outside the basis.

## Theoretical Backing

Three theorems justify the design:

1. **Subspace drift bound.** EMA momentum's dominant left/right singular subspaces drift slowly when the singular gap exceeds the EMA increment: `||P_t - P_{t-1}||_F <= 2*sqrt(2r) * d_t / (g_{t-1} - d_t)`. Caching the basis is provably safe.

2. **Noise-calibrated geometry.** With a precision matrix A, the calibrated steepest-descent selector is `J(g) = A^(1/2) J_gamma(A^(1/2) g)`. The Schatten-infinity endpoint recovers Muon's polar projection. Soft-polar in core coordinates interpolates toward whitening on noisy directions.

3. **Compression.** Total reconstruction error decomposes orthogonally: subspace tail + off-diagonal leakage + scalar quantization error. Once basis drift is bounded, leakage is small.

## Algorithm

For each matrix parameter:
- Maintain EMA momentum buffer `m_t`
- Cache rank-r basis `(U, V)`, refresh every `pso_basis_freq` steps via warm-started alternating block power iteration
- Project: `core = U^T @ m_t @ V` (small r×r matrix)
- Track `var_core` via second-moment EMA
- Soft-polar transform: `core / sqrt(core^2 + lambda * var_core + tau^2)`
- Reconstruct: `update = U @ core_update @ V^T`
- Optional residual: `update += residual_coeff * (m_t - U @ core @ V^T) / rms`
- RMS-normalize, scale by `sqrt(max(1, m/n))`, apply lr

## Hyperparameters

| Variable | Default | Range |
|----------|---------|-------|
| PSO_RANK | 16 | 8 / 16 / 32 |
| PSO_BASIS_FREQ | 8 | 4 / 8 / 16 |
| PSO_POWER_ITERS | 1 | 1 / 2 |
| PSO_BETA2 | 0.98 | 0.95 / 0.98 / 0.99 |
| PSO_NOISE_WEIGHT | 1.0 | 0.5 / 1.0 / 2.0 |
| PSO_RESIDUAL | 0.1 | 0.0 / 0.1 / 0.25 |
| PSO_RENORMALIZE | 1 | 0 / 1 |

## Base Stack (PR #1394)

SP8192 + GPTQ Embeddings + Depth Recurrence (loop layers 4-5 × 2) + SDClip + 11 layers + 512d + 4× MLP + XSA-all + brotli.

## Run Command

```bash
PSO_RANK=16 PSO_BASIS_FREQ=8 PSO_NOISE_WEIGHT=1.0 PSO_RESIDUAL=0.1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
