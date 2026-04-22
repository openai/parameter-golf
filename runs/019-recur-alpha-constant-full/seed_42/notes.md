# Spec 019 — execution notes (seed 42)

## Alpha constant correction (caught during interview)

Original commit `2895db3` had pass-2 L5 α = **1.3984375** (016's endpoint value, not 017's).
017's actual `recur_alpha_final` at step 4784 = **1.4296875** for that position.

Bug source: 018c commit was prepared before spec 017 finished; captured an intermediate α value.
Fix: corrected to `1.4296875` in commit `3c3a134`. All other 5 values were correct.

Spec updated to pin `3c3a134`.

## Pod

- **Region:** AP-JP-1
- **Hardware:** 8×H100 SXM
- **Volume:** jlxvxeiol4 → /runpod
- **Seed:** 42
