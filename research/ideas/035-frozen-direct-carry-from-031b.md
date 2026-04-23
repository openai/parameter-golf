# Idea 035 — Frozen gated direct-carry from `031B`

This is the `031B` branch of the same family as:

- `034` from `031A`
- `034b` as TTT adaptation on top of `034`

So:

- `035` should be the frozen `031B` line
- `035b` should be the TTT follow-up on top of `035`

Why this is the right numbering:

- `031A` and `031B` are sibling calibration probes
- `034` / `034b` already cover the `031A` branch
- `035` / `035b` should cover the `031B` branch

Key difference vs `034`:

- `031B` adds a per-destination `carry_gate`

So `035` asks whether that extra frozen gating structure is actually valuable in
the deployment-style line.

We now have a real freeze source for this branch:

- `031B-ratio0272-freshpod-rerun1`
- late snapshot `train_step_5400`

That snapshot looks stable enough to use:

- `self_max_drift = 0.0`
- `gate_max_drift = 0.0039`
- `edge_max_drift ≈ 0.01`
