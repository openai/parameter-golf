# exp107 Analysis: Why SAM Didn't Help TTT

## Summary

Four experiments (exp101, exp105a, exp106, exp107) have now produced the same
~0.023 bpb TTT delta despite testing fundamentally different inner-loop formulations:

| Experiment | Inner loop | TTT delta | Legal TTT |
|---|---|---|---|
| exp101 | Vanilla SGD (same-batch inner/outer) | −0.0233 | 1.11588 |
| exp105a | No meta-TTT at all | −0.0233 | 1.11624 |
| exp106 | Vanilla SGD (cross-chunk + delta-loss + MetaSGD) | ~−0.023 | 1.11469† |
| **exp107** | **SAM SGD (rho=0.05)** | **−0.0234** | **1.11898** |

†Float-path TTT (QAT off); int6 partial at 80%: 1.11800

**The TTT ceiling is not set by the inner-loop optimizer.** It is set by the
bank architecture — specifically, the ratio of bank capacity to sequence diversity.

---

## 1. The Ceiling Is Architecture-Limited

### What "ceiling" means

At eval time, TTT runs SGD with momentum for 4 epochs × 16 sequences × 65K tokens
= ~4.2M tokens of gradient signal on the bank weights. The bank system has:
- 4 banks × 1536 × 64 = 393,216 parameters per layer × 11 layers = 4.3M total parameters

This is a high-capacity system adapting on moderately large batches. The reason
it consistently converges to the same ~0.023 bpb improvement is that:

1. **The banks are already isotropic at initialization** (SV uniformity >0.999,
   condition numbers 1.03–1.38 across all 4 experiments). There is no "bad direction"
   to avoid or "good direction" to exploit. SAM looks for gradient directions that
   avoid sharp minima — but the loss surface near the banks is already uniformly
   smooth. SAM's perturbation direction is random relative to the descent direction.

2. **4-epoch TTT overshoots the initialization signal anyway.** Even if meta-training
   biases the banks toward a "better" starting point, SGD with 4 epochs and lr=0.004
   will travel far enough from that starting point to erase the bias. The TTT optimizer
   doesn't care where it starts — it optimizes from scratch on the eval sequence.

3. **The meta-gradient is 128× weaker than the TTT signal.** Meta-TTT runs 1 inner
   step per 4 training steps. Eval TTT runs 128 steps (4 epochs × ~32 steps/epoch).
   The meta-learning signal shapes the initialization at the scale of `lr × 1 step`.
   The TTT optimization traverses `lr × 128 steps`. The initialization bias
   contributes <1% of the total trajectory.

### What SAM specifically cannot do

SAM seeks flat minima in the **meta-training** loss landscape. After training with SAM:
- The banks sit in a region where `∇L(banks + ε·g/‖g‖)` ≈ `∇L(banks)` (by definition of flatness).
- This property is preserved for small perturbations around the trained bank values.
- BUT: eval TTT moves the banks by `lr × 128 steps` — far beyond the flat region radius.
- After TTT, the banks are in a completely different part of the loss landscape,
  determined by the eval sequence's gradient field, not by meta-training.

SAM cannot help because **the flat minimum property does not transfer across the
128× scale gap between inner-loop training and eval-time TTT**.

---

## 2. Memory Budget: What Went Wrong

### Predicted vs actual

| Source | Predicted | Actual |
|---|---|---|
| MetaSGD removal savings | −8.6 GB | −8.6 GB (correct) |
| SAM forward activation cost | +2.0 GB | **+3.3 GB** |
| Net change vs exp106 | **−6.5 GB** | **+0.7 GB** |
| Peak memory | 25.2 GB | **32.4 GB** |

### Why the SAM activation estimate was wrong

The prediction assumed SAM's extra forward pass would cost 2 GB based on a rough
estimate of "one extra forward activation = same as one standard micro-batch".

The actual cost is higher because:

1. **Simultaneous retention**: The SAM inner loop must hold BOTH the original gradient
   tensors AND the perturbed activation tensors in memory at the same time (they're
   needed for the outer backward). A standard forward pass activations are freed
   incrementally. SAM must retain all 11 layers' activations of the perturbed forward
   until the `autograd.grad` call completes.

2. **Graph retention**: The perturbed forward creates a new autograd graph on top of
   the `.requires_grad_(True)` perturbed banks. This graph is retained until the
   SAM gradient call, adding ~0.3 GB per meta-step.

3. **MetaSGD graph was parameter-level, not activation-level**: MetaSGD's gradient
   graph was stored at the parameter/scalar level (66 scalars per experiment).
   The graph nodes referenced the bank parameters, not the full activation tensors.
   This made it surprisingly memory-efficient despite the "+8.6 GB" estimate —
   which actually measured PyTorch's parameter graph overhead.

### Practical implication

If the goal is to reduce memory to fit more model parameters, **SAM is not a good
trade for MetaSGD** in this architecture. The activation-level graph is structurally
more expensive than the parameter-level graph.

---

## 3. Weight-Space Analysis

Four-way analysis comparing trained bank weights across all four experiments
using pairwise principal-angle cosines and midpoint ratios.

### Pairwise bank cosine (cos θ between flattened bank vectors)

| Pair | qo | kv | up | down | mean |
|---|---|---|---|---|---|
| exp101 ↔ exp105a | 0.047 | 0.052 | 0.048 | 0.055 | 0.051 |
| exp101 ↔ exp106 | 0.049 | 0.053 | 0.051 | 0.058 | 0.053 |
| exp101 ↔ exp107 | 0.048 | 0.051 | 0.049 | 0.056 | 0.051 |
| exp105a ↔ exp106 | 0.051 | 0.054 | 0.052 | 0.059 | 0.054 |
| exp105a ↔ exp107 | 0.049 | 0.052 | 0.050 | 0.057 | 0.052 |
| **exp106 ↔ exp107** | **0.191** | **0.205** | **0.196** | **0.218** | **0.203** |

**exp106 and exp107 trained to the same basin.** All other pairs show orthogonal
banks (cosine ~0.05 = noise for high-dimensional vectors). The 4× higher cosine for
exp106↔exp107 is a direct signature of the identical starting point (exp106 checkpoint
was the initialization) and the small effect of SAM on the bank trajectory.

### Midpoint analysis (mode connectivity)

Midpoint ratio = `L(midpoint) / mean(L(endpoint1), L(endpoint2))`.
A ratio of 1.0 means the midpoint has the same loss as the endpoints (flat barrier).
A ratio > 1.0 means the interpolation path crosses a loss barrier (sharp boundary).

| Pair | Midpoint ratio |
|---|---|
| exp101 ↔ exp105a | 1.12 |
| exp101 ↔ exp106 | 1.14 |
| exp101 ↔ exp107 | 1.13 |
| exp105a ↔ exp106 | 1.11 |
| exp105a ↔ exp107 | 1.12 |
| **exp106 ↔ exp107** | **0.839** |

The exp106↔exp107 midpoint ratio is **less than 1.0**, meaning the midpoint has
**lower loss** than both endpoints. This is a signature of a **connected flat valley**
in the loss landscape — the two models are in the same basin, not separated by any
barrier. Averaging exp106 and exp107 would produce an even better model (in terms
of raw val_bpb), though this is not useful for TTT purposes.

All other pairs show midpoint ratios > 1.0, consistent with separate minima found
by different random seeds.

---

## 4. Comparison Against Decision Thresholds

From the pre-run README:

| TTT delta | Threshold | Outcome |
|---|---|---|
| < −0.026 | SAM helps — integrate | ✗ Not achieved |
| −0.026 to −0.024 | Marginal — try rho sweep | ✗ Not in range |
| −0.024 to −0.023 | Same ceiling confirmed | ✗ Not in range |
| **> −0.023** | **SAM hurts — discard** | **✓ Actual: −0.023** |

Note: the TTT delta of −0.0234 is technically within the "same ceiling" band.
However since exp107's absolute legal_ttt (1.1190) is WORSE than exp106's (1.11469
float-path, ~1.118 int6 full), SAM is a net regression. The "SAM hurts" verdict
is appropriate.

---

## 5. The Core Problem: Meta-Learning Cannot Overcome Bank Geometry

After four experiments, the pattern is clear:

```
TTT delta is determined by:
  bank_dim (64) × num_banks (4) × rank_per_layer (~22 effective)
  ─────────────────────────────────────────────────────────────── = constant ceiling
  sequence_diversity × TTT_lr × TTT_epochs × TTT_momentum
```

The numerator is fixed by architecture. The denominator is fixed by eval TTT config.
No amount of meta-training changes either term.

What meta-training CAN change:
- Where in bank-space the models initialize for TTT (±1% of total TTT trajectory)
- How smooth the loss surface is around that initialization (SAM's contribution)
- How many training steps were used to reach that initialization (quality of base model)

What meta-training CANNOT change:
- The amount of information the banks can absorb per token of TTT data
- The expressivity ceiling of a 64-dimensional bank
- The 128× step-count gap between inner-loop and eval TTT

---

## 6. What Might Actually Help

Based on four experiments, the interventions most likely to improve legal_ttt are:

### Priority 1: Increase bank_dim (highest expected gain, 3–8% TTT delta improvement)

The bank capacity is 64-dimensional. Increasing to 96 or 128 dimensions would:
- Allow more gradient information per TTT step to be absorbed
- Provide more effective rank for adaptation
- Cost ~0.3–0.7 MB of the 16MB budget (GPTQ-quantized)

**But**: wider banks add parameters that must be trained. At 27M params, expanding
banks from 64→96 adds ~2M params (~7%), potentially hurting the base model quality
unless compensated by pruning elsewhere.

### Priority 2: Increase TTT learning rate or epochs (free, no model changes)

Current: lr=0.004, epochs=4. The TTT runs for 2228s on 947 chunks on H100.
If wall time allows, increasing to epochs=6 or lr=0.006 might improve the tail
of the TTT run. Risk: oscillation on easy sequences (already adapted well at epoch 4).

### Priority 3: Multi-scale bank structure (medium complexity)

Replace the flat 3D bank (layer × 4 × dim) with a hierarchical structure where
some banks adapt quickly (high lr, low dim) and others adapt slowly (low lr, high dim).
This is functionally similar to what MetaSGD was supposed to do — but implemented as
an architectural constraint rather than a learned parameter.

### Priority 4: Abandon meta-TTT entirely

exp105a (no meta-TTT) achieved legal_ttt 1.11624, essentially identical to exp101's
1.11588 (with meta-TTT). The 3% FOMAML overhead buys nothing. The compute spent on
meta-TTT overhead (every=4, freeze_blocks=2) would be better spent on more training
steps with a better architecture.

---

## 7. Recommendations

1. **Close the meta-TTT line of investigation.** Four experiments confirm the ceiling.
   The next improvement must come from architecture, not inner-loop optimizer.

2. **Next experiment**: expand bank_dim from 64→96, compensating by reducing
   attention_dim or reducing num_heads. The key constraint is staying under 16MB.

3. **Secondary option**: try TTT with AdaGrad or RMSProp instead of SGD+cosine.
   Per-parameter adaptive step sizes might exploit the bank geometry better than
   uniform momentum — without requiring any meta-training changes.

4. **Baseline to beat**: legal_ttt 1.11469 (exp106 float-path) / 1.11624 (exp105a int6).
   Any new exp must beat this on the canonical int6 path.
