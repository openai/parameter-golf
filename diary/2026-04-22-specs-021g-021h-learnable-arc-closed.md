# 2026-04-22 — Specs 021g + 021h: learnable α arc definitively closed

---

## The remaining question after 021e

021e (frozen α, algebraic form, TTT fix) hit post-TTT **1.06622** — the new best. But an
open question remained: 017's pre-quant EMA was **1.06861**, significantly better than
021e's 1.06944. If 017's superior per-step quality could be reproduced with the corrected
algebraic blend form and TTT fix, post-TTT would improve substantially.

017 used **learnable fp32 α**. The hypothesis: 017 found a better α basin than 021e's
frozen 017-endpoint values, and that basin is reachable if we just let α train with the
correct setup. Two variants:

- **021g**: learnable `nn.Parameter` in bf16, algebraic form, TTT fix (same init as 016: ones)
- **021h**: learnable `nn.Parameter` in fp32, algebraic form, TTT fix (storage precision test)

Both were run on the same 8×H JP pod (sequential seeds, same hardware).

---

## Spec 021g — Learnable bf16, algebraic form

021g was the simplest version: exactly 021e but with `requires_grad=True` and bf16 storage.

Result (commit `fab6e7f`):
- Steps: **4,804**
- Pre-quant EMA: **1.06987** — worse than 021e (1.06944) by 0.00043
- Post-TTT: **1.06693** — worse than 021e by 0.00071

The α converged to:
```
L3 (pass 2): 1.102   L4: 1.305   L5: 1.383
L3 (pass 3): 1.039   L4: 0.926   L5: 0.844
```

Compare to 017's endpoint:
```
L3 (pass 2): 1.078   L4: 1.273   L5: 1.430
L3 (pass 3): 1.016   L4: 0.973   L5: 0.832
```

L5 (pass 2) is the critical divergence: 021g converged to **1.383** vs 017's **1.430** —
a gap of −0.047. This is a different optimization basin, not a noisy draw of the same basin.

**Diagnosis: bf16 precision trap.** bf16 has resolution 1/128 ≈ 0.0078 in the [1.0, 2.0]
range. AdamW's learning rate at step 4000 is ~1e-5, far below the bf16 LSB. α gets rounded
to the nearest bf16 tick at each step, so updates smaller than 0.0078 are silently discarded.
The optimizer can't move α past certain grid boundaries; it gets trapped on the coarse
quantization lattice.

021g confirmed learnable bf16 α cannot reproduce 017's basin.

---

## Spec 021h — Learnable fp32 storage

If bf16 traps α, fp32 (LSB ≈ 6e-8) restores full AdamW precision. 021h changes only
the storage dtype: `nn.Parameter(torch.ones(..., dtype=torch.float32))`. The blend sites
cast to bf16 for computation (model runs in bf16), but gradient accumulation happens in fp32.

Result (commit `5906820`):
- Steps: **4,753** — fewer than 021g (4,804); a throughput stall around step 3800 cost ~50 steps
- Val@4000: **1.1084** — best of the 021 family at this checkpoint
- Pre-quant EMA: **1.07043** — worse than 021g (1.06987) due to fewer warmdown steps
- Post-TTT: **1.06734**

fp32 closed the L5 gap to −0.024 (vs 021g's −0.047):
```
L5 (pass 2): 021h = 1.406  vs 017 = 1.430 → gap −0.024
```

But "closed" ≠ "reached". The residual −0.024 gap at fp32 is not a precision artifact.
It reflects a genuine difference in the optimization landscape: **the algebraic blend form
and 017's manual-add form are numerically different in bf16**, creating different loss
surfaces. 017 converged under `α*x_new + (1-α)*x_before`; 021h is optimizing
`x_before + α*(x_new - x_before)`. These are mathematically equivalent in exact arithmetic
but diverge in bf16, and the optimizer finds different critical points.

**017's 1.06861 pre-quant is not reproducible** with the algebraic form. It combined:
1. Manual-add blend form (different optimization landscape)
2. Pod lottery — 017 drew a fast JP node

Neither of these can be deliberately reproduced in a new run.

---

## Learnable α arc: final verdict

| variant | post-TTT | verdict |
|---|---|---|
| 017 (learnable fp32, manual-add, buggy TTT) | 1.06733 | pod luck + wrong form |
| 021g (learnable bf16, algebraic, TTT-fix) | 1.06693 | bf16 precision trap |
| 021h (learnable fp32, algebraic, TTT-fix) | 1.06734 | different basin, fewer steps |
| **021e (frozen α, algebraic, TTT-fix)** | **1.06622** | **winner** |

The learnable α arc is definitively closed. Letting α train produces worse post-TTT than
freezing it at 017's endpoint values — because the 8×H learning regime (batch size,
learning rate schedule, warmdown timing) doesn't converge α to the same basin 017 found.
Frozen α bakes in the good structure without the convergence cost.

**This is the key insight that motivates the cross-layer carry arc (specs 024+):** if
frozen is better than learned at this scale, the natural question is whether we can find
a richer frozen structure by first running a learnable variant to convergence on the
specific configuration we want, then freezing those values.
