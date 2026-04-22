# Idea: Cross-layer carry blend

**Created:** 2026-04-22
**Status:** Parked — run spec 024 first

## What

Extend detached-lerp so each looped layer on pass 2 blends with ALL looped layer outputs from pass 1, not just its own x_before.

**Current (detached-lerp):**
```python
# Layer i, pass 2:
x = x_before_det + alpha[pass, local_i] * (x_new - x_before_det)
# x_before_det is whatever came into layer i on pass 2
```

**Cross-layer carry:**
```python
# First pass: store all looped layer outputs (detached)
carry = {j: x_j_out.detach() for j in [loop_start..loop_end]}

# Layer i, pass 2:
x = x_new + sum(alpha[local_i, local_j] * carry[j] for j in loop_layers)
```

Parameters: 3×3 = 9 scalars (target layer × source layer). All carries detached — no backward overhead.

## Also: parameter over x_new?

Could also add a learnable scalar on x_new itself:

```python
x = beta[local_i] * x_new + sum(alpha[local_i, local_j] * carry[j] for j in loop_layers)
```

That's 3 + 9 = 12 scalars total. `beta` init=1 (normal forward), `alpha` init=0 (no carry at start).

Init choice matters: beta=1, alpha=0 → identical to baseline at start, gradients activate the carry terms if useful.

## Why it's interesting

In the current detached-lerp, layer 5 pass 2 only "remembers" pass 1 implicitly — the pass-1 signal has been transformed through layers 3, 4, 5 again before reaching layer 5's input. The cross-layer form gives each layer a direct shortcut to any looped layer's pass-1 output.

Practical example: if layer 3 computed a strong syntactic feature on pass 1 that got washed out by layers 4+5, layer 5 pass 2 can directly reference it via alpha[2, 0] * carry[3].

## Cost

- 9 or 12 params (negligible for 16MB budget)
- Stores 3 carry tensors [batch, seq, d_model] during forward — same memory as old carry form
- All detached → no backward overhead beyond current

## Prerequisite

Spec 024 (detached-lerp) must pass throughput check first. No point adding cross-layer mixing if the basic form doesn't recover throughput.

## Relationship to prior work

- tashapais carry: `x = x_new + alpha * carry_prev_pass[i]` — carry is previous pass output of same layer (chained). Our version: carry is always pass-1 output of any looped layer (broadcast).
- Cross-layer: extends tashapais's idea to allow explicit cross-layer communication during recurrence.
