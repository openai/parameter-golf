# Notes on the recurrence band in compressed transformers

A small set of architectural studies on the loop band (layers 3–5) of the
#1736 / 060A baseline. Each section is independent.

---

## Section 1 — Learning mixing parameters in depth-recurrent loops

A depth-recurrent loop runs the canonical Markov iteration through the loop
band (layers 3–5):

```
x_{k+1} = f(x_k)
```

Each pass uses only the previous pass's output. We replace this with a
learned mixing rule, train it end-to-end, and observe that the learned
mixing coefficients converge to a stable, nearly seed-invariant pattern
within a few hundred steps after looping activates. Once stabilized, the
coefficients can be read off the trained model and used as fixed constants
in a fresh training run.

## Recurrent α-β

We add learnable scalars to control how each pass commits to the residual
and to allow detached cross-layer carries within the same pass:

```
x_{k+1} = β_k · f(x_k) + Σ_j α_{k,j} · stop_grad(x_k^{(j)})
```

with `β_k` initialized to 1 and `α_{k,j}` initialized to 0, so the loop
starts from the canonical Markov rule. Across the loop band (layers 3–5,
NL=2) this is a small number of scalars; they are routed to the scalar
optimizer and trained jointly with the rest of the model.

During a full training run on the #1736 base, the scalars drift off their
initialization once looping activates at `frac=0.35`, then plateau. The
final values are reproducible across seeds — for example, layer 4 converges
to a self-subtract pattern at `α ≈ −0.348` (a learned gate), and layer 5
stabilizes into a positive aggregation of the signals from layers 3 and 4.

## Freezing the learned values

We then read the converged values off the trained model and use them as
fixed constants in a new training run from scratch. The optimizer state
and per-step gradient on these scalars are dropped; only the values
survive. Because the loop now starts at the converged mixing pattern
rather than at the canonical Markov rule, the run is no longer
identity-at-init, but training-end quality matches.

This is shipped as PR #1779 on top of #1736:

| Submission | Mixing rule in loop band | val_bpb (3-seed mean) | Δ vs #1736 |
|---|---|---:|---:|
| #1736 (base) | canonical Markov | 1.06549 | — |
| #1779 (frozen α-β) | fixed α-β with cross-layer carry | **1.06421** | **−0.00128** |

3-seed std on #1779 is 0.00023, so the gain is well outside seed noise.
Artifact size is unchanged (the frozen scalars are baked into the model
weights serialized into the 16 MB budget).

The converged values used as fixed constants in #1779 are:

```
β = [1.5973, 1.8828, 1.9922]                          # layers 3, 4, 5

         L3       L4       L5
α = [[ 0.2520, −0.0210, −0.0124],     # L3 contributions
     [ 0.0669, −0.3477,  0.0031],     # L4 contributions
     [ 0.1387,  0.2412,  0.0272]]     # L5 contributions
```

Two patterns stand out. Every β is well above 1, so each pass amplifies
its own block output rather than damping it — the optimizer chose to
overshoot the canonical Markov rule. And the diagonal of α is mixed: L3
adds back ~25% of itself, L4 subtracts ~35% of itself (the learned-gate
self-subtract behavior), L5 leaves itself roughly alone but absorbs ~24%
of L4. The off-diagonal entries in row L5 also confirm L5 acts as an
aggregator over L3 and L4.

## Anderson acceleration with frozen coefficients

The same idea applies to a different mixing rule. Anderson acceleration
replaces the Markov iteration with a length-`m` mix of past iterates,
solved per batch via a small least-squares problem:

```
g_i = f(x_i) − x_i                                     # residuals
α* = argmin_α  ‖Σ_{i=k−m+1..k} α_i · g_i‖²,  Σ α_i = 1
x_{k+1} = Σ α*_i · f(x_i)
```

Trained end-to-end (length-3 Anderson, per-batch LS), the coefficients
land in the noise band of canonical recurrence but pay a ~25% throughput
penalty for the per-batch solve. Inspecting the trained model, the
per-batch α distribution concentrates tightly around

```
α ≈ [+0.55, −0.67, +1.12]
```

Following the same procedure as for α-β, we drop the LS solve and
hardcode these coefficients as constants. The result is a
fixed-coefficient extrapolation across the last three iterates with no
runtime overhead beyond the canonical loop.

| Variant | Mixing rule | Throughput vs canonical | val_bpb (single seed) |
|---|---|---:|---:|
| Canonical | Markov | 1.00× | 1.06108 |
| Anderson, learned per-batch α | length-3 LS | 0.75× | 1.06083 |
| Anderson, frozen α | fixed `[+0.55, −0.67, +1.12]` | 1.00× | 1.05968 |

The frozen-Anderson result is single-seed; multi-seed confirmation has
not been run.

---

## Section 2 — MLP sizing across the three stages

The loop band runs each of layers 3, 4, 5 three times per forward pass
(NL=2). Each pass reads the same FFN weights, so the parameters in the
loop band see roughly 3× the use per token of the FFN parameters in the
non-looped layers. A natural question is whether the loop band deserves
more FFN capacity than the rest of the model at fixed total parameters —
i.e., whether reallocating width from the non-looped layers into the
loop band is a free win.

We split the 11 physical layers into three positional stages and
parameterize the FFN width as a per-stage multiplier of `model_dim`:

```
stage     layers    width multiplier
early     0–2       MLP_EARLY_MULT
middle    3–5       MLP_MIDDLE_MULT     # the loop band
late      6–10      MLP_LATE_MULT
```

The baseline uses `4.0` everywhere, for a total of `11 × 4.0 = 44.0`
width-units. We tried three reallocation schemes that hold the total
fixed at 44.0 width-units while widening the middle stage to 5.0:

| arm | early | middle | late | direction |
|---|---:|---:|---:|---|
| baseline | 4.0 | 4.0 | 4.0 | uniform |
| 040A | 3.625 | 5.0 | 3.625 | shrink both sides evenly |
| 040B | 3.0 | 5.0 | 4.0 | shrink early, keep late |
| 040C | 4.0 | 5.0 | 3.4 | keep early, shrink late |

Single-seed training-only screen on the 038/039 fullfloat research line,
2×H100, 600s wallclock cap, no quantization or TTT. The absolute val_bpb
values are pre-quant post-EMA from this short screen, *not* directly
comparable to the post-quant post-TTT numbers in Section 1 — this is a
relative comparison of training quality between MLP schedules, not an
endpoint number. Pre-quant post-EMA val_bpb on the validation set:

| arm | val_bpb (pre-quant post-EMA) | Δ vs uniform |
|---|---:|---:|
| baseline (uniform 4.0) | 1.16501 | — |
| 040A (3.625 / 5.0 / 3.625) | 1.16742 | +0.00241 |
| 040B (3.0 / 5.0 / 4.0) | 1.16744 | +0.00244 |
| 040C (4.0 / 5.0 / 3.4) | **1.16484** | **−0.00017** |

Three observations:

- **The middle-widen direction is real but small.** 040C is the only
  reallocation that doesn't regress, and the gain is comfortably inside
  single-seed noise (Δ ≈ −0.0002 on a screen with no seed average).
  Treat it as "tied with baseline," not a win.
- **Shrinking the early stage is more expensive than shrinking the
  late stage.** 040B (early shrunk to 3.0, late kept at 4.0) loses
  +0.00244; 040C (early kept at 4.0, late shrunk to 3.4) gains
  −0.00017. A symmetric shrink (040A) lands close to 040B. The early
  layers (0–2) are doing work that doesn't compress; the late layers
  (6–10) tolerate it.
- **The middle-stage gain is bounded above by what the late-shrink
  costs.** Whatever extra capacity the middle stage absorbs from going
  4.0 → 5.0, the late stage gives back roughly the same amount when it
  goes 4.0 → 3.4. The two effects nearly cancel. The implication is that
  the loop band is *not* obviously starved for FFN capacity at the
  uniform baseline.

---

## Section 3 — Sizing the loop band

The canonical 060A loop band is the contiguous set {3, 4, 5} run at
NL=2, so each of layers 3, 4, 5 is visited three times per forward
pass. The full forward does 17 layer-applications, with 9 of them
inside the loop band. Two knobs control the total compute spent inside
the band: which layers form the band (band-set), and how many times
each is visited (NL). We screened both directions on 060A.

| spec | band-set | NL | loop-band passes | description |
|---|---|---:|---:|---|
| 060A canonical | {3,4,5} | 2 | 9 | reference |
| 041B | {3,4,5} | 1 | 3 | half the canonical loop compute |
| 041D | {5} | 2 | 3 | single-layer band, only layer 5 |
| 041H | {4,5} | 2 | 6 | drop the front of the band |
| 070 | {3,4} | 2 | 6 | drop the back of the band |
| 041L | {3,4,5} | 3 | 12 | more visits per layer |
| 041N | {3,4,5} | 4 | 15 | more still |

Same screen protocol throughout: single seed 42, 4×H100, 1200s
wallclock, no TTT. Pre-quant post-EMA val_bpb:

| spec | structure | pre-quant post-EMA | Δ vs canonical |
|---|---|---:|---:|
| 060A canonical | {3,4,5} NL=2 | **1.06358** | — |
| 041B | {3,4,5} NL=1 | 1.06842 | +0.00484 |
| 041D | {5} NL=2 | 1.06993 | +0.00635 |
| 041H | {4,5} NL=2 | 1.06693 | +0.00335 |
| 070 | {3,4} NL=2 | 1.06595 | +0.00237 |
| 041L | {3,4,5} NL=3 | 1.06615 | +0.00257 |
| 041N | {3,4,5} NL=4 | 1.06888 | +0.00530 |

Two observations:

- **Canonical is locally optimal in both directions.** Both shrinking
  (NL=1, single-layer band, drop a layer) and growing (NL=3, NL=4) lose
  to the canonical {3,4,5} NL=2 — the loss is monotonic in how far the
  configuration sits from canonical. NL=3 (+0.00257) is the closest
  miss; NL=4 (+0.00530) loses about as much as halving the loop
  compute.
- **Band shape is roughly position-symmetric.** Dropping layer 3 (041H,
  +0.00335) and dropping layer 5 (070, +0.00237) cost similar amounts.
  Reducing to a single layer (041D, +0.00635) is worse than either, but
  in the same direction. There's no specific layer in {3,4,5} that's
  uniquely load-bearing; the band-as-a-whole is what matters.

The 041L NL=3 result is interesting in isolation — the gap to
canonical (+0.00257) is small enough that with multi-seed averaging
it may close. We did not promote it past the screen.
