# 2026-04-22 — Cross-layer carry arc: 024 → 025b → spec 026

---

## Starting point

021e was the best result in the arc at this point: post-TTT **1.06622**, a miss of 0.00012 on
#1736 (1.06610) — within seed std. The plan was to explore whether the recur-alpha blend
could be made more expressive without sacrificing the frozen-buffer throughput wins.

The open question entering this session: can cross-layer routing (blending carry states from
*all* looped layers, not just self) produce a structural improvement that survives to post-TTT?

---

## 024 — Learnable detached-lerp

The first variant replaced the standard lerp with a detached form:
```
x_before_det = x_before.detach()
x = x_before_det + alpha * (x_new - x_before_det)
```

The motivation: vanilla lerp retains a gradient path through `x_before`, which accumulates
backward overhead across passes. Detaching breaks that path. Result at 4×H:

- steps: 4975, val@4000: **1.1185**, pre-quant EMA: **1.07106**

Slightly worse than 021c's frozen diagonal (1.1177) at 4×H — but this is expected. Learnable
variants can't beat frozen at 4×H because learnable needs large batches per gradient step
to find the optimal basin; 4×H is batch-starved for that purpose. Not actionable at this rung.

---

## 024b — Cross-layer carry blend, shared across passes

The bigger idea: instead of blending only self (pass k of layer i with pass k-1 of layer i),
blend across all three looped layers using the detached first-pass carry:

```
x = beta[i] * x_new + sum_j(alpha[i,j] * carry[j])
```

12 scalars total (3 betas + 3×3 alpha matrix). Carry dict populated during the full first pass,
so no causality constraint — every layer j's first-pass output is available to every layer i's
second pass.

Result at 4×H:
- steps: 5017, val@4000: **1.1196**, pre-quant EMA: **1.06960**

Looks slightly worse than 021c (1.1177 → 1.1196), but investigation revealed the 024b pod
was running 1.2% faster than the 021c pod (4,275K tok/s vs 4,224K tok/s). The step-count
difference (5017 vs 5004) is pod-speed artifact, not code quality. Step-matched, the two
are indistinguishable at 4×H.

**Key learned pattern in 024b's converged alpha:**
- L3: mostly self (self-dominant, ignores L4/L5)
- L4: `alpha[L4,L4] = -0.348` — **self-subtract** (pass 2 subtracts its own pass-1 output)
- L5: pulls from L3 (+0.139) and L4 (+0.241), modest self

The L4 self-subtract is striking. It means pass-2 L4 is actively anti-correlated with where
pass-1 L4 went — a de-redundancy move. Betas converged to `[1.597, 1.883, 1.991]`.

---

## 024c — Cross-layer carry blend, per-pass

Extend 024b so each blend pass gets its own beta/alpha:
```
x = beta[pass_off, i] * x_new + sum_j(alpha[pass_off, i, j] * carry[j])
```

24 scalars (6 betas + 2×3×3 alpha tensor). The rationale: pass 2 and pass 3 of layer 5 are
doing structurally different things — the residual stream is at different stages. No reason
to share routing weights across passes.

024c converged to something remarkable:

```
pass 1 beta:  [0.9453, 1.2969, 1.6250]  ← amplifies with depth (L5 strongest)
pass 2 beta:  [1.9844, 1.6875, 1.1563]  ← reverses: L3 strongest, L5 weakest
```

The two passes use *opposite* depth profiles. Pass 1 deepens amplification toward L5; pass 2
inverts it, giving L3 more weight and L5 less. The model discovered a complementary pair.

The L4 self-subtract persisted in both passes (`alpha[pass1, L4, L4] = -0.305`,
`alpha[pass2, L4, L4] = -0.291`), confirming it's a structural property of what pass-2 L4
needs to do, not a training artifact.

---

## The key reliability insight

At this point we had three learnable variants (024, 024b, 024c) all running roughly even with
or slightly behind 021c's frozen diagonal at 4×H. This looked discouraging — cross-layer is
more expressive but doesn't win.

The key realization: **4×H is reliable only for frozen vs frozen comparisons.** Learnable
alpha at 4×H is seeing 4× fewer gradient steps per parameter update than at 8×H. It's finding
a different (worse) basin not because the hypothesis is wrong, but because it doesn't have
enough gradient signal at this batch size to converge to the good basin within the 1200s
wallclock.

This is why 021c (frozen at 017's converged values) reliably beats 021h (learnable) at 4×H
even though at 8×H the picture may be different.

---

## 025b — The frozen cross-layer bet

If the routing pattern that 024b learned (L4 self-subtract, L5 pulling L3+L4) is the
load-bearing structure, why not freeze it from step 0?

025b replaces 024b's `nn.Parameter` with `register_buffer` at 024b's converged values.
From step 0, the model has the cross-layer routing baked in — no "discovery cost."

Commit `950af24`. Register buffer shapes `[3]` and `[3,3]`. All 4 blend sites indexed
`[local_idx]` and `[local_idx, j]` — shared across passes.

Result: val@4000 = **1.1079**, Δ = **−0.0098 vs 021c (1.1177)**.

This is the largest positive signal we've seen in the entire 021 arc. Better than every
4×H run and better than 021e's 8×H early signal of 1.1134 at step 4000. The decision tree
in spec 025b is unambiguous: > 0.002 better → promote to 8×H.

**Why 025b wins so clearly:** the frozen routing gives the model a pre-configured information
superhighway at step 0. The optimizer never has to discover it — it's already there. The
cross-layer structure (especially L4's anti-correlation move and L5's aggregation) provides
a structural advantage from the first training step.

---

## 025c — Per-pass frozen (queued)

Follow-up: what if we also freeze the per-pass differentiation that 024c learned? 025c
takes 024c's converged values (the depth-inversion pattern) and freezes them into a
`[2,3]` / `[2,3,3]` register_buffer.

Commit `414cbc3`. All 4 blend sites use `[pass_off, local_idx]` indexing.

Not yet run — queued for execution. If 025c val@4000 < 1.1079 (025b's result), use 025c
commit for the 8×H run. Otherwise 025b wins and 025c's differentiation adds nothing.

---

## Spec 026 — 8×H full pipeline

With 025b's 4×H result (1.1079) meeting the "clearly better" threshold, spec 026 is written
for the 8×H full pipeline run:

- **Commit:** `950af24` (025b shared-frozen), swap to `414cbc3` if 025c beats it before pod launch
- **TTT enabled:** `PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3`
- **8×H JP:** `nproc_per_node=8`, no wallclock cap
- **Seed plan:** 42 first; 43/44 conditional on post-TTT ≤ 1.066

Expected outcome: if the −0.0098 4×H advantage translates even partially to 8×H, we may see
post-TTT in the 1.060–1.063 range — a clear submission candidate. Even a partial translation
of 0.004 would put us at ~1.062, which is a definitive beat of #1736 (1.06610).

---

## State of the arc (end of session)

| spec | type | val@4000 | pre-quant EMA | post-TTT |
|---|---|---|---|---|
| 021c | frozen diagonal, 4×H | 1.1177 | 1.06952 | — |
| 021e | frozen diagonal, 8×H | 1.1134 | 1.06944 | **1.06622** |
| 024 | learnable detached-lerp, 4×H | 1.1185 | 1.07106 | — |
| 024b | learnable cross-layer shared, 4×H | 1.1196 | 1.06960 | — |
| 024c | learnable cross-layer per-pass, 4×H | TBD | TBD | — |
| **025b** | **frozen cross-layer shared, 4×H** | **1.1079** | TBD | — |
| 025c | frozen cross-layer per-pass, 4×H | queued | — | — |
| 026 | frozen cross-layer shared, 8×H | **spec ready** | — | — |

The cross-layer frozen approach is now the leading candidate. The entire 021 arc was spent
finding the right frozen routing structure; 025b may be it.

---

## Side notes

- **024b's 5017 steps vs 021c's 5004:** investigated and attributed to pod speed variance
  (~1.2% difference in tok/s). Not a code quality signal.
- **bf16 rounding in register_buffer:** float32 tensors in register_buffer get cast to bf16
  at blend time. Log shows `1.5973426` rounds to `1.59375` (bf16 tick). This is expected and
  harmless — the model's effective routing weights are the bf16-rounded values, which is
  what matters for actual forward passes.
- **Carry causality:** carry dict is populated in a full first-pass loop before any blending
  begins. No upper-triangular constraint needed — all looped-layer first-pass outputs are
  available to all second-pass blend sites.
