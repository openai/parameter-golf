# Random Linear Adapter — `TTT_RLA_ENABLED`

OpenAI Requests-for-PRs item: *"Learning adapters on random linear maps."*

## What this is

The standard `BatchedLinearLoRA` in #1855 / #1953 has both `A` (`rank ×
in_features`) and `B` (`out_features × rank`) as learnable parameters.
RLA freezes `A` to a fixed orthonormal random projection (registered as
a buffer, not in the optimizer); only `B` is learnable.

```
LoRA: delta = (B @ A) x        — both A, B trainable
RLA:  delta = (B @ A_frozen) x — A is a fixed random orthonormal projection
```

`A` is initialized via Gaussian QR decomposition (rows orthonormal,
rescaled to LoRA's input-norm bound), shared across the batch slot
dim, and never updated. Implements OpenAI's Requests-for-PRs item
literally.

## Smoke-test verified

On the deployment pod (8×H100, torch 2.9.1+cu128):
- `A` is in `model.buffers()`, not `model.parameters()`.
- Optimizer parameter list excludes `A`.
- `B.grad` flows; `A.requires_grad == False`.
- `A` rows are orthonormal: `max |A A^T - diag| ≈ 9e-10` at rank 16,
  in_features 64; diagonal entries `≈ 1/in_features`.
- `reset()` zeros `B` and leaves `A` untouched.
- Param count drops to ~`B`-only (1/3 of standard LoRA at the same rank).

## Real result on our pod (this competition, single seed)

- LoRA TTT (#1953 standard, rank 80): **1.06600**
- RLA TTT (rank 160 same param budget): **1.07146** (+0.005 regression)

The frozen random `A` doesn't span enough useful adaptation directions
in the per-doc TTT window. Documented as a clean negative result.

## Why it's still here

1. The Requests-for-PRs item is named directly. Even a negative result
   on the literal request is research.
2. The lever might compose better with other techniques here (e.g.
   with SSM blocks where the right adaptation subspace differs from
   attention).
3. There's a parameter-efficient variant — VeRA (Kopiczko et al. 2024)
   — that adds a *learnable diagonal scaling* between fixed-random `A`
   and `B`. Worth trying as a follow-up.

## To make it record-worthy

1. Pretrain `A` on the training data (supervised PCA-like) instead of
   random init.
2. Add VeRA-style learnable diagonals.
3. Sweep the rank — RLA at rank-160 didn't beat LoRA-80, but RLA at
   rank-256+ might (random projections amortize at higher rank).
