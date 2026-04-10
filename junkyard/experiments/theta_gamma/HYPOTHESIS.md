# Theta-Gamma: Dual EMA Timescales

## Biological inspiration
Hippocampus runs two oscillations simultaneously — slow theta (~8Hz) binds sequences,
fast gamma (~40Hz) encodes items. Two-speed memory. The ratio between time constants
is φ (1.618) by construction.

## Architecture
Two EMA teachers instead of one:
- Fast teacher: τ_fast = 1 - (1/φ²) ≈ 0.618 decay — tracks recent gradient landscape
- Slow teacher: τ_slow = 0.9999 — consolidates long-run patterns
- At each KL step, student pulls toward a gated blend: α * fast_teacher + (1-α) * slow_teacher

The gate α is learned per-layer — some layers anchor to long-term structure (slow),
others track fast signal (fast).

φ bonus: Fast:slow responsiveness ratio = 1:1.618 = 1:φ by construction.

## Key hyperparameters
- THETA_GAMMA_CADENCE (default 4, same role as TORNADO_CADENCE)
- THETA_GAMMA_TAU_FAST = 0.618  (= 1 - 1/φ²)
- THETA_GAMMA_TAU_SLOW = 0.9999
- THETA_GAMMA_KL_WEIGHT = 0.1

## Base
experiments/tornado/train_gpt.py — add second EMA dict (fast_teacher_params),
per-layer learned gate α (nn.Parameter, shape num_layers), blend teachers at KL step.

## Buildability: ★★★★★ — ~20 lines on top of Tornado
Extend tornado: add fast_teacher_params dict, per-layer gate, update both EMAs.
