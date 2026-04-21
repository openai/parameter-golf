# Three-Seed Results — 2026-04-20/21

## Configuration
- Base: Kevin Clark PR #1394 (SP8192)
- Novel lever: per-layer QK-Gain init schedule
  - `QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5"`
  - Replaces uniform qk_gain_init=4.0 baseline across all 11 attention layers
- Hardware: 1×H100, grad_accum=8 (mathematically equivalent to 8×H100 grad_accum=1)
- Iterations: 20,000 (each run)

## Results (quantized_sliding_window val_bpb)

| Seed | Script | Val BPB | Artifact bytes |
|------|--------|---------|----------------|
| 1337 | train_gpt_sp8192_opt.py | 1.07069667 | 16,009,732 (over cap; retrain pending) |
| 42   | train_gpt_submission.py | 1.07056354 | 15,960,096 |
| 2025 | train_gpt_submission.py | 1.07052821 | 15,965,976 |

**Mean:** 1.07059614
**Std:**  0.00008969 (n=3)
**Range:** 0.00017 BPB

**vs SOTA 1.0810 (bigbag, 2026-04-09):** -0.01040 BPB (2.08× the 0.005 record threshold)

## Status
- [x] Single-hardware (1×H100 grad_accum=8) reproduction confirmed across 3 seeds
- [x] Artifact size under 16MB cap (seeds 42, 2025; seed 1337 retrain pending)
- [ ] 8×H100 eval-environment reproduction — pending grant-funded compute
- [ ] Seed 1337 retrain on train_gpt_submission.py — pending grant
- [ ] Submission PR — pending 8×H100 validation

## Caveats
- Seed 1337 trained on an earlier oversized script (train_gpt_sp8192_opt.py, 64KB).
  Seeds 42 and 2025 trained on the final submission file (train_gpt_submission.py, 15.6KB).
  Consistency across both scripts (delta 0.00017 BPB) suggests the QK-Gain lever is
  the effective variable, but a clean 3-seed submission will retrain seed 1337 on
  the submission codebase.
- 8×H100 evaluation reproducibility not yet confirmed. Request for compute credits
  submitted to enable this verification before opening a record PR.
