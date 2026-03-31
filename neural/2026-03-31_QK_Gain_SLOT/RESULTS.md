# QK_GAIN_SLOT_Gate — Results

**Date:** TBD
**Pod:** TBD
**Seed:** 444

---

## Smoke Test

| step_avg_ms | GPU | NPROC | Status |
|-------------|-----|-------|--------|
| 739ms | H100 80GB HBM3 | 1 | PASSED |

**Key finding:** 739ms/step is correct for NPROC=1. `grad_accum = 8 / world_size = 8` on a single GPU — each logical step processes the same total batch as the 8×H100 run, just in 8 sequential micro-steps. Expected = 91ms × 8 = ~728ms. This is a healthy pod.

---

## Ablation Results

| Case | post_ema_bpb | delta | sliding_bpb | delta | step_avg_ms |
|------|-------------|-------|-------------|-------|-------------|
| baseline | | — | | — | |
| qk_gain4 | | | | | |
| slot_only | | | | | |
| qk_gain4_slot | | | | | |

## Cross-Correlation

| | Value |
|---|---|
| QK_GAIN delta (sliding) | |
| SLOT delta (sliding) | |
| Predicted sum | |
| Actual combo | |
| Interaction residual | |
| Compatible? | |

---

## Decision

- [ ] QK_GAIN signal validated (≥ 0.001 post_ema improvement)
- [ ] SLOT signal validated (≥ 0.003 sliding improvement)
- [ ] Signals additive (interaction < 0.002)
- [ ] Full gate run authorised

**Outcome:** TBD
