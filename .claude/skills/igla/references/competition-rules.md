# IGLA Competition Rules

## Gate-2 Requirements

| Requirement | Value | Notes |
|-------------|---------|--------|
| Target BPB | < 1.85 | Final bpb must be below threshold |
| Minimum Steps | >= 4000 | Training must reach step 4000+ |
| Minimum Samples | >= 5 distinct steps | Need bpb_samples across different checkpoints |
| Eligible Cells | Count toward Gate-2 progression | Only cells meeting all criteria |

## Sanctioned Seeds

| Seed ID | Value | Source |
|----------|--------|---------|
| F_17 | 1597 | Fibonacci sequence |
| F_18 | 2584 | Fibonacci sequence |
| F_19 | 4181 | Fibonacci sequence |
| F_20 | 6765 | Fibonacci sequence |
| F_21 | 10946 | Fibonacci sequence |

## R5 Discipline

- **R5-honest:** No mock training, only real experiments
- Record `not_yet_implemented` for unimplemented formats
- Do not fake numbers for Tier C placeholders

## WAKE_TRIGGERS

| Trigger | Condition | Severity | Action |
|---------|------------|----------|----------|
| W-1 | fleet_alive < 4/6 | CRITICAL | Log, investigate, halt new deploys |
| W-2 | zero bpb_samples (15m) | WARNING | Check TRAINER_KIND configuration |
| W-6 | Φ-1 collapse (5+ identical bpb values) | CRITICAL | Immediate investigation, no new experiments |

## Anti-Collapse Floor

Experiments with bpb < 0.01 are flagged as SUSPECT per R5 discipline.

## Tier Classification

| Tier | Formats | Implementation |
|------|----------|---------------|
| A (real) | GF16, fp16, bf16, fp32 | Native trainer support |
| B (emulated) | GF8, GF12, GF20, GF24 | Quantize-on-the-fly |
| C (placeholder) | GF4, GF32, GF64 | Record as not_yet_implemented |
