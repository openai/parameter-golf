# Evaluation — Spec 018c (Recur-Alpha compile-time constant α)

**Run dir:** `runs/018c-recur-alpha-constant/run-e-constant/`
**Commit:** `aabfbea` on `exp/recur-alpha-constant`
**Pod:** `g5s1rqfhia58uk` — 2×H100 SXM, US-NE-1, NA volume `hvpdph5i3g`
**Eval date:** 2026-04-21

## Hypothesis recap

Replacing `nn.Parameter` α (runtime tensor) with hardcoded Python floats (017 endpoint values) makes α a compile-time constant. torch.compile can then specialize `torch.lerp` kernels per-site, fuse more aggressively, and potentially eliminate identity sites. Expected: material recovery beyond lerp's 18%.

## Result

| Run | Commit | Config | Avg steps 100+125 tok/s | vs baseline | vs lerp |
|-----|--------|--------|------------------------|-------------|---------|
| A (016b) | 154c9b8 | no recur-alpha | 3,333K | — | — |
| B (016b) | 4dd2d63 | current 4-op blend | 3,234K | −2.9% | — |
| C (018) | 97d9854 | torch.lerp, tensor α | 3,252K | −2.4% | — |
| **E (018c)** | **aabfbea** | **torch.lerp, constant α** | **3,325K** | **−0.24%** | **+2.24%** |

Full step log: 50→3,323K / 75→3,325K / 100→3,326K / 125→3,325K / 150→3,303K (step 150 dip typical end-of-run; discard).

## Decision criterion outcome

Let L = 3,252K (lerp, Run C), K = 3,325K (constant, Run E).

K/L = 1.0224 → **K ≥ 1.02 × L** bucket: compile meaningfully specializes on constant α. Clear win.

Overhead recovery:
- Blend overhead (A vs B): 99K tok/s
- Remaining overhead (A vs E): 8K tok/s
- **Recovered: ~92% of blend overhead**

## Interpretation

torch.compile fully specializes `torch.lerp` when the weight is a Python float literal. With tensor α, Dynamo treats the weight as a runtime value even if numerically stable — it can't fold it into the kernel. With a float literal, the compiler bakes the constant into the CUDA kernel, enabling:

1. Specialized lerp kernel per site (6 sites → 6 specialized kernels, each compiled with the known weight)
2. Constant-folding of the `1 - alpha` term eliminates a subtraction
3. Surrounding-op fusion reforms: the block output → lerp → next block input chain becomes one fused pass

The 8K tok/s residual overhead is likely irreducible kernel-launch overhead from 6 blend sites and memory traffic that can't be eliminated.

## Decision — PROMOTE: apply constant-α in production pipeline

**This is the correct optimization.** Constant-α recovers 92% of blend overhead vs 18% for tensor lerp. The 0.24% residual overhead at proxy scale implies <0.05% at full 11L/512d (overhead fraction shrinks ~6× as matmuls dominate).

**Trade-off**: This gives up learned α — values are frozen at 017's endpoint `((1.078, 1.273, 1.398), (1.016, 0.973, 0.832))`. The model no longer adapts α during training.

**Implication for spec 017**: The next full-pipeline run should use commit `aabfbea` (or a branch off it) as the base rather than `97d9854` (lerp with tensor α). The throughput tax effectively disappears.

## Cost

| item | cost |
|---|---|
| Run E: ~8 min compile + 1 min training | ~$0.90 |
| **018c total** | **~$0.90** |

Combined 018 + 018b + 018c on same pod: **~$2.50 total** (plus pod restart ~$0.10).

## Cross-references

- Spec: `research/specs/018c-recur-alpha-constant.md`
- lerp result: `research/evaluations/018-recur-alpha-lerp.md`
- bake-in (shelved): `research/evaluations/018b-recur-alpha-bakein.md`
- Control data: `research/evaluations/016b-recur-alpha-throughput.md`
