#!/usr/bin/env python3
"""Reproduce spec 006's step-based LR schedule for normalization.

From train_gpt_sota.py:1235-1240:
    def lr_mul(frac):
        if warmdown_frac <= 0: return 1.0
        if frac >= 1.0 - warmdown_frac: return max((1.0 - frac) / warmdown_frac, min_lr)
        return 1.0

Warmup (20 steps) is a separate pre-training phase — does NOT affect per-step LR
once main training starts. For the checkpoint range (step 100-4500), lr_mul is
just the warmdown function.

With ITERATIONS=4550, WARMDOWN_FRAC=0.72:
    warmdown_start_frac = 1.0 - 0.72 = 0.28
    warmdown_start_step = 0.28 * 4550 = 1274

At step 1274: lr_mul = 1.0
At step 4550: lr_mul = 0.0
Linear decay between.
"""
ITERATIONS = 4550
WARMDOWN_FRAC = 0.72
MIN_LR = 0.0  # or whatever h.min_lr is — check via env/defaults if this matters

def lr_mul(step: int, iterations: int = ITERATIONS, warmdown_frac: float = WARMDOWN_FRAC) -> float:
    """Return the LR multiplier at main-training `step` (0-indexed from after warmup)."""
    if warmdown_frac <= 0:
        return 1.0
    frac = step / max(iterations, 1)
    warmdown_start = 1.0 - warmdown_frac
    if frac >= warmdown_start:
        return max((1.0 - frac) / warmdown_frac, MIN_LR)
    return 1.0

def lr_mul_at_mid_window(start_step: int, end_step: int, iterations: int = ITERATIONS) -> float:
    """Mean LR across a window, approximated via midpoint."""
    mid = (start_step + end_step) / 2
    return lr_mul(int(mid), iterations)

if __name__ == "__main__":
    print(f"ITERATIONS={ITERATIONS}, WARMDOWN_FRAC={WARMDOWN_FRAC}")
    print(f"warmdown_start_step = {int((1.0 - WARMDOWN_FRAC) * ITERATIONS)}")
    for s in [0, 100, 500, 1000, 1274, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 4550]:
        print(f"step {s:>5d}: lr_mul = {lr_mul(s):.4f}")
