# B-WING ENTROPY-SHIFT — Per-Order Center Shift

## Hypothesis
PR #809 shifts the entropy sigmoid center DOWN for higher orders:
  center = entropy_center - 0.25 * (order - min_order)

For order 9, min_order 2: center = 3.0 - 0.25*7 = 1.25
This means even when the model is fairly confident (entropy ~1.5), high-order matches
still get substantial alpha. Our flat center=3.0 for all orders means high-order matches
on confident tokens get almost no alpha boost.

## Changes from X-WING baseline
1. Add per-order entropy center shift: center = ent_center - 0.25*(order - min_order)
2. Keep everything else identical to X-WING baseline

## Expected impact
Should help most on "easy" tokens where the model is confident but an 8/9-gram
match provides even better information. These tokens are currently under-mixed.

## What NOT to change
- Keep alpha range at 0.20-0.75 (isolate this variable)
- Keep cubric 3D
- Keep architecture
