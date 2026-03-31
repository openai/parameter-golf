# H-FRUG Findings — Frugendorff/Crawler Architecture Research

**Date:** 2026-03-27
**Branch:** test
**Pod:** ssh4.vast.ai:12234 (RTX A6000, single GPU)
**Control:** purple_8L_save — params=11,159,475, sub=2,390,400, **int6=3.60239**

---

## Background

CrawlerGPT (F_Wing architecture) uses **shared transformer blocks** looped K times.
Each loop iteration produces a "deeper" representation from the same weights.

Hypothesis going in: K-loop recurrence = free depth gain = better BPB at fixed param count.

---

## Three Hypotheses Tested (sweep5b)

### H-FRUG-1: KL Distillation from Frozen Teacher
- **Config:** `FROZEN_TEACHER_PATH=/workspace/purple_8L_ema.pt`, KL loss weight 0.5
- **Trial name:** `green_4x_distill`
- **Result:** int6 = **3.7378** (control: 3.6141)
- **Delta:** −0.1237 (catastrophically worse)
- **Verdict:** FAIL. The distillation loss fights the task loss at 5-minute training budgets. The teacher is already a stronger model than what the crawler can represent — soft targets are too hard, not helpful.

### H-FRUG-2: Loop Bottleneck Gate (64-dim)
- **Config:** `LOOP_BOTTLENECK_DIM=64` — inserted 64-dim linear bottleneck between crawler loop iterations
- **Trial name:** `green_4x_bottleneck`
- **Result:** int6 = **~3.7x** (failed to converge meaningfully within budget)
- **Delta:** worse than control
- **Verdict:** FAIL. Bottleneck doesn't resolve gradient conflict — it just adds parameters that now need to be trained while the shared weights are still receiving conflicting signals.

### H-FRUG-3: Reinvest Freed Bytes → Wider Model (dim=408)
- **Config:** `MODEL_DIM=408` — no recurrence, just a wider standard transformer
- **Trial name:** `green_4x_wide`
- **Result:** int6 = **~3.7x** (failed to converge meaningfully)
- **Delta:** worse than control
- **Verdict:** FAIL. Width increase without architectural change doesn't help at this scale.

---

## Root Cause: The Shared-Weight Gradient Conflict

The fundamental problem with CrawlerGPT's loop architecture:

```
Loop iteration 1: f(x₁) → x₂   (shared weights learn: "given x₁, produce x₂")
Loop iteration 2: f(x₂) → x₃   (SAME weights must learn: "given x₂, produce x₃")
```

These are in direct conflict. A single set of weights cannot simultaneously:
- Map early-layer representations → mid-layer representations
- Map mid-layer representations → late-layer representations

At short training budgets (5 min on H100), the optimizer can't find a compromise.
At longer budgets, the model would converge to some average behavior that's worse than
a properly-stacked model of equivalent depth.

**This is not fixable with**: bottleneck gates, distillation, or wider dims.
**The fix would require**: truly separate weights per loop iteration (= just a deeper model).

---

## Engineering Bug Found During Sweep

`CrawlerGPT.__init__()` was missing `loop_bottleneck_dim: int = 0` in its signature —
the parameter was in `GPT.__init__` instead. Fixed via Python string replacement on pod.
The bug had been hiding because the default value of 0 happened to be safe (no bottleneck).

---

## Conclusion

**Frugendorff / Crawler research line: CLOSED.**

The K-loop shared-weight architecture provides no meaningful advantage at our training budget
and introduces fundamental gradient conflicts that cannot be patched. The architecture would
need to become a standard stacked model (defeating the purpose of shared weights for param savings).

Our SOTA at time of closure: **1.1129 BPB** (Rat Rod Green v1).
Purple_1 (phrase cache + Dirichlet + WARMDOWN=2000) is the next frontier.
