# 2026-04-20 SP8192 LegalTTT W104 Faithful Replay Candidate

This folder is **not a new submission yet**.

It is a faithful replay / bad-seed-variance reducer candidate based on the current SP8192 + LegalTTT near-SOTA stack (3-layer recurrence, parallel residuals, QK gain 5.25, legal score-first TTT, and quantized+brotli artifact target under 16 MB).

## Intent

- Preserve architecture/compression surface from the SP8192 LegalTTT stack.
- Make key defaults source-visible for evaluator/replay clarity.
- Probe bad-seed behavior starting with seed 314 only.

## Pass condition (seed 314)

- Must improve meaningfully below **1.08168719**.
- Strong pass: **seed314 < 1.0812**.

Only after seed314 passes should seeds **42** and **999** be run.
