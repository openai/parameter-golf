# Idea: Direct-carry free-float calibration for NUM_LOOPS=3

**Created:** 2026-04-23
**Status:** Active

## Context

Spec 029 likely regressed because the frozen cross-layer carry constants from the 025b/025c line were calibrated in the older `NUM_LOOPS=2` regime, then reused under `NUM_LOOPS=3 + depth curriculum`. That changes the loop dynamics enough that the old constants are probably miscalibrated.

Spec 030 showed that the cleaner 025b path with seed 314 and corrected TTT can reach:

- pre-quant EMA: `1.06821629`
- post-TTT: `1.06471941`

That is much better than the 029 stack and only ~`0.00115` behind #1769 seed 314 post-TTT. So the most plausible next lever is not "add more stack blindly," but "recalibrate carry under the actual `NUM_LOOPS=3` regime."

## Core idea

Run a **free-floating calibration** where the carry combiner is learnable again under the true target regime, then freeze whatever it learns into a later submission run.

This is not a submission architecture by itself. It is a measurement run whose job is to answer:

1. What carry structure does `NUM_LOOPS=3` actually want?
2. Do different passes want meaningfully different carry behavior?
3. What frozen constants should we bake in afterward?

## What we are pausing

For this calibration, pause the delayed recurrence-kick idea. Keep the recurrence schedule otherwise simple. The purpose of this run is to isolate carry structure under `NUM_LOOPS=3`, not to stack another scheduling lever on top of it.

## Design candidates discussed

### 1. Re-run old frozen 025b/025c values

Rejected for this purpose. The whole problem is that those values were calibrated in the wrong regime.

### 2. Per-pass alpha/beta carry (024c-style, expanded for depth 4)

Viable and clean. This is the closest extension of the 024c line:

```python
x = beta[pass_off, i] * x_new + sum_j(alpha[pass_off, i, j] * carry[j])
```

Good baseline if we want minimal conceptual drift from 024c.

### 3. Factorized carry: route + carry_gate + self_gate

Most structured design:

```python
output = self_gate * self_signal + carry_gate * sum(route * detached_carry)
```

Pros:
- clean interpretation
- separates "where to read" from "how much carry to use"

Cons:
- adds design assumptions before we know what matters
- less direct first probe

### 4. Direct-carry over explicit source states

Chosen as the first probe.

Instead of factorizing, each allowed carry edge gets its own learned coefficient, plus a self coefficient for the destination.

Conceptually:

```python
output(dst, pass) = self_coeff[pass, dst] * self_signal(dst, pass) \
                  + sum(edge_coeff[pass, dst, src] * detached_source_carry[src])
```

This is the most direct "let the model tell us what it wants" version without adding much parameter count.

## Chosen first probe

### Direct-carry, detached, neutral-init

- `NUM_LOOPS=3`
- detached carry
- no delayed recurrence-kick change
- neutral init
- direct-carry parameterization
- per-pass behavior

### Source availability

For the first probe:

- **Pass 2** destination layers may read all loop-layer carry states from **pass 1**
- **Pass 3** destination layers may read all loop-layer carry states from **pass 1 and pass 2**

Rationale:
- letting pass 3 see only pass 2 hardcodes a specific "only the latest refinement matters" hypothesis
- letting pass 3 see both pass 1 and pass 2 is still small, but more flexible
- training can suppress useless edges by driving them toward zero

### Why direct-carry first

- every learned scalar has a simple meaning
- easiest to inspect after training
- fewer architectural assumptions than route/carry/self
- if it works, we can simplify later

## Parameter count for the chosen probe

Assume 4 loop layers are active in the target regime.

### Carry edges

- pass 2: `4 destinations × 4 pass-1 sources = 16`
- pass 3: `4 destinations × 8 sources (pass1 + pass2) = 32`

Total carry-edge coefficients: **48**

### Self coefficients

- pass 2: 4
- pass 3: 4

Total self coefficients: **8**

### Total

**56 learnable scalars**

This is tiny. Parameter budget is not the issue; the real costs are implementation complexity and whether the parameters stabilize cleanly enough to freeze.

## Neutral initialization

For the direct-carry probe:

- edge coefficients: initialize to `0`
- self coefficients: initialize to `1`

That gives identity-like behavior at start:
- no external carry contribution initially
- destination keeps its own local signal

This avoids anchoring on old `NUM_LOOPS=2` values.

## Runtime / stabilization

This run should be longer than a minimal screen because the goal is not just val_bpb. It is also to observe whether the carry coefficients stabilize enough to freeze.

The run should log:

- `val_bpb`
- full carry coefficient snapshots
- self coefficient snapshots
- simple drift summaries over the last ~1000 steps

Freeze only if the learned pattern is both useful and reasonably stable.

## Decision rule after the free-floating run

If the direct-carry calibration run:

- improves float quality vs the miscalibrated frozen run, and
- yields a stable learned pattern,

then freeze those learned values into a follow-up spec.

If it does not stabilize, either:

- run longer, or
- fall back to a simpler carry parameterization.

If it stabilizes but is hard to interpret, route/carry/self remains a possible second experiment.
