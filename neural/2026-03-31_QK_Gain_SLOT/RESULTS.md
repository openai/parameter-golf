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

## Run 1 Results — 2026-03-31 (SLOT crashed, partial)

| Case | post_ema_bpb | delta | sliding_bpb | delta | step_avg_ms |
|------|-------------|-------|-------------|-------|-------------|
| baseline | 1.302300 | — | 1.362200 | — | 746.81 |
| qk_gain4 | 1.303300 | +0.0010 | 1.362500 | +0.0003 | 703.38 |
| slot_only | 1.302900 | +0.0006 | **CRASHED** | — | 703.11 |
| qk_gain4_slot | 1.303600 | +0.0013 | **CRASHED** | — | 711.08 |

**SLOT crash root cause:** `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
SLOT optimization loop was inside `torch.no_grad()` context — gradient tracking was suppressed.
**Fix:** Added `with torch.enable_grad():` wrapping the delta optimization loop. Training ran correctly; only the SLOT eval pass crashed.

**QK_GAIN_INIT=4.0 verdict: DEAD.** +0.0010 post_ema (wrong direction). Not pursuing.

---

## Run 2 — SLOT fix (pending)

Re-run slot_only only: `CASES="slot_only" bash gate.sh`

| Case | post_ema_bpb | delta | sliding_bpb | delta |
|------|-------------|-------|-------------|-------|
| slot_only | | | TBD | |

---

## Decision

- [x] QK_GAIN signal validated → **NO SIGNAL. Drop.**
- [ ] SLOT signal validated (need run 2 sliding_bpb)
- [ ] Full gate authorised

**Outcome:** Pending SLOT re-run.
