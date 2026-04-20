#!/usr/bin/env python3
"""Schedule trace plot for #1736 / spec 015 config.

Shows all training-time rates that change during a ~4800-step run:
  1. LR multiplier (warmdown)       — wallclock-based
  2. Muon momentum (warmup)         — step-based
  3. Loop activation (0/1 switch)   — wallclock-based at 35% frac
  4. Effective LR (MATRIX_LR × mul)

Assumes default #1736 env:
  ITERATIONS=20000   (capped by wallclock)
  WARMDOWN_FRAC=0.75
  MIN_LR=0
  MUON_MOMENTUM=0.97
  MUON_MOMENTUM_WARMUP_START=0.92
  MUON_MOMENTUM_WARMUP_STEPS=1500
  ENABLE_LOOPING_AT=0.35 (wallclock frac)
  MAX_WALLCLOCK_SECONDS=600
  MATRIX_LR=0.026

Empirical step count at wallclock cap (from spec 008 seed 42): 4828.

Run: `python3 research/scripts/spec_015_schedules.py`
Output: `research/scripts/spec_015_schedules.png`
"""
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Defaults from #1736 Hyperparameters class
WARMDOWN_FRAC = 0.75
MIN_LR = 0.0
MUON_MOMENTUM = 0.97
MUON_MOMENTUM_WARMUP_START = 0.92
MUON_MOMENTUM_WARMUP_STEPS = 1500
ENABLE_LOOPING_AT = 0.35
MATRIX_LR = 0.026

# Empirical: at wallclock cap (600s), we hit ~4828 steps (spec 008 seed 42).
TOTAL_STEPS = 4828
MAX_WALLCLOCK_MS = 600_000


def wallclock_ms_at_step(step, total_steps=TOTAL_STEPS, wallclock_ms=MAX_WALLCLOCK_MS):
    """Assume training_frac is wallclock-based. Step → elapsed_ms (linear approx)."""
    return (step / max(total_steps, 1)) * wallclock_ms


def training_frac(step):
    """#1736's training_frac is wallclock-based (elapsed_ms / max_wallclock_ms)."""
    return wallclock_ms_at_step(step) / max(MAX_WALLCLOCK_MS, 1e-9)


def lr_mul(step):
    """Warmdown schedule. Flat at 1.0 until frac=1-warmdown_frac, then linear to min_lr."""
    if WARMDOWN_FRAC <= 0:
        return 1.0
    frac = training_frac(step)
    warmdown_start = 1.0 - WARMDOWN_FRAC
    if frac >= warmdown_start:
        return max((1.0 - frac) / WARMDOWN_FRAC, MIN_LR)
    return 1.0


def muon_momentum_at_step(step):
    """Linear warmup from 0.92 to 0.97 over MUON_MOMENTUM_WARMUP_STEPS, STEP-based."""
    if MUON_MOMENTUM_WARMUP_STEPS <= 0:
        return MUON_MOMENTUM
    frac_mom = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
    return (1 - frac_mom) * MUON_MOMENTUM_WARMUP_START + frac_mom * MUON_MOMENTUM


def looping_active(step):
    """Hard switch at wallclock frac = ENABLE_LOOPING_AT. 0 or 1."""
    return 1 if training_frac(step) >= ENABLE_LOOPING_AT else 0


def main():
    outdir = Path(__file__).parent
    outfile = outdir / "spec_015_schedules.png"

    steps = np.arange(0, TOTAL_STEPS + 1, 10)
    lr_mul_values = np.array([lr_mul(int(s)) for s in steps])
    momentum_values = np.array([muon_momentum_at_step(int(s)) for s in steps])
    loop_values = np.array([looping_active(int(s)) for s in steps])
    effective_lr = lr_mul_values * MATRIX_LR

    warmdown_start_step = int((1.0 - WARMDOWN_FRAC) * TOTAL_STEPS)
    loop_activation_step = next(
        (s for s in steps if looping_active(int(s))), TOTAL_STEPS
    )
    momentum_plateau_step = MUON_MOMENTUM_WARMUP_STEPS

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    # 1. LR multiplier
    ax = axes[0]
    ax.plot(steps, lr_mul_values, color="#1f77b4", lw=2)
    ax.axvline(warmdown_start_step, color="gray", ls=":", alpha=0.6, label=f"warmdown_start @ step {warmdown_start_step}")
    ax.axvline(loop_activation_step, color="#ff7f0e", ls="--", alpha=0.5, label=f"looping_active @ step {loop_activation_step}")
    ax.axvline(momentum_plateau_step, color="#2ca02c", ls="--", alpha=0.4, label=f"muon_momentum plateau @ step {momentum_plateau_step}")
    ax.set_ylabel("lr_mul")
    ax.set_title(f"Spec 015 / #1736 — training schedules over {TOTAL_STEPS} steps ({MAX_WALLCLOCK_MS // 1000}s wallclock)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(-0.05, 1.1)

    # 2. Effective LR (matrix_LR × lr_mul)
    ax = axes[1]
    ax.plot(steps, effective_lr, color="#1f77b4", lw=2)
    ax.axvline(warmdown_start_step, color="gray", ls=":", alpha=0.6)
    ax.axvline(loop_activation_step, color="#ff7f0e", ls="--", alpha=0.5)
    ax.set_ylabel(f"effective LR\n(MATRIX_LR={MATRIX_LR} × lr_mul)")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.001, MATRIX_LR * 1.1)

    # 3. Muon momentum
    ax = axes[2]
    ax.plot(steps, momentum_values, color="#2ca02c", lw=2)
    ax.axvline(momentum_plateau_step, color="#2ca02c", ls="--", alpha=0.5, label=f"plateau @ step {momentum_plateau_step}")
    ax.axvline(loop_activation_step, color="#ff7f0e", ls="--", alpha=0.5)
    ax.axhline(MUON_MOMENTUM_WARMUP_START, color="gray", ls=":", alpha=0.4, label=f"start={MUON_MOMENTUM_WARMUP_START}")
    ax.axhline(MUON_MOMENTUM, color="gray", ls=":", alpha=0.4, label=f"target={MUON_MOMENTUM}")
    ax.set_ylabel("muon momentum")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.91, 0.98)

    # 4. Looping activation (step function)
    ax = axes[3]
    ax.step(steps, loop_values, where="post", color="#ff7f0e", lw=2)
    ax.axvline(loop_activation_step, color="#ff7f0e", ls="--", alpha=0.5, label=f"looping_active @ step {loop_activation_step}")
    ax.set_ylabel("looping_active\n(0 = off, 1 = on)")
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    ax.set_ylim(-0.1, 1.2)

    plt.tight_layout()
    plt.savefig(outfile, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"Wrote {outfile}")
    print()
    print(f"Key schedule events:")
    print(f"  step    0: training begins, lr_mul=1.0, muon_mom={MUON_MOMENTUM_WARMUP_START}, looping=OFF")
    print(f"  step {momentum_plateau_step}: muon_mom plateau at {MUON_MOMENTUM}")
    print(f"  step {loop_activation_step}: looping_active flips to ON (wallclock 35%)")
    print(f"  step {warmdown_start_step}: warmdown begins (wallclock 25%)")
    print(f"  step {TOTAL_STEPS}: wallclock cap, lr_mul=0, training ends")


if __name__ == "__main__":
    main()
