# Arch+Sched Sweep — Results

**Date:** 2026-03-31
**Pod:** 4×H100 SXM (Vast.ai)
**Seed:** 444
**MAX_WALLCLOCK_SECONDS:** 600
**NPROC:** 4
**Steps per run:** ~2880–2897

---

## Smoke Test

| step_avg_ms | GPU | NPROC | Status |
|-------------|-----|-------|--------|
| ~207ms | H100 SXM | 4 | PASS (expected 91×8/4 = 182ms; <2.5× threshold = 455ms) |

---

## Sweep Results

| case | post_ema_bpb | delta | sliding_bpb | delta | int6_bpb | quant_gap | size_MB | qat_step | steps |
|------|-------------|-------|-------------|-------|----------|-----------|---------|----------|-------|
| baseline    | 1.176800 | —       | 1.154747 | —       | 1.198524 | +0.0217 | 13.52 | 2376 | 2897 |
| rope_32     | 1.176300 | -0.0005 | 1.154302 | -0.0004 | 1.198813 | +0.0225 | 13.56 | 2355 | 2879 |
| bigram_3072 | 1.176700 | -0.0001 | 1.154759 | 0.0000  | 1.198727 | +0.0220 | 14.30 | 2373 | 2897 |
| bigram_4096 | 1.177300 | +0.0005 | 1.155354 | +0.0006 | 1.200023 | +0.0227 | 14.42 | 2369 | 2893 |
| qat_early   | 1.177100 | +0.0003 | 1.155181 | +0.0004 | 1.199408 | +0.0223 | 14.23 | 2021 | 2894 |
| qat_late    | 1.177200 | +0.0004 | 1.155183 | +0.0004 | 1.199037 | +0.0218 | 14.01 | 2721 | 2895 |
| swa_dense   | 1.177700 | +0.0009 | 1.155744 | +0.0010 | 1.199412 | +0.0217 | 13.60 | 2369 | 2881 |
| gptq (post) | 1.176800 | 0.0000  | 1.154749 | 0.0000  | 1.198524 | +0.0217 | 13.52 | N/A  | N/A  |
| warmdown_4k | 1.180000 | +0.0032 | 1.158120 | +0.0034 | 1.207733 | +0.0277 | 13.79 | 2297 | 2895 |

Note: `gptq_full` (full training + GPTQ) not yet run. See GPTQ bug note below.

---

## Verdicts

### DEAD — no signal at proxy scale
- **bigram_3072**: +0.0000 sliding. Competition target size (14.30MB, fits gate), but zero measured gain. Not pursuing at 8×GPU.
- **bigram_4096**: +0.0006 — hurts. Size risk (14.42MB). Dead.
- **qat_early** (threshold 0.15→0.25): +0.0004 — hurts. QAT fires at step 2021 (355 steps earlier). Dead.
- **qat_late** (threshold 0.15→0.05): +0.0004 — hurts. QAT fires at step 2721 (345 steps later). Dead.
- **swa_dense** (SWA_EVERY 50→10): +0.0010 — hurts. More snapshots = worse. Dead.
- **gptq (post-train)**: 0.0000 delta — GPTQ calibration bug. Only 2 layers hooked, 0 quantized. Mechanically broken; doesn't change model. Not a real test.
- **warmdown_4k** (WARMDOWN_ITERS 3500→4000): **+0.0034 — HURTS SIGNIFICANTLY.** Root cause: time-based schedule means longer warmdown → QAT fires EARLIER (step 2297 vs 2376). At proxy scale this is catastrophic. Dead permanently.

### BORDERLINE — noise level, not worth 8×GPU
- **rope_32** (ROPE_DIMS 16→32): -0.0004 sliding. Below proxy noise floor (~0.001 needed for real signal). Do not promote.

### GPTQ BUG (requires investigation)
- **gptq (post-train SKIP_TRAIN=1)**: calibration log shows `gptq:calibrated 2 layers in 1.9s` → `gptq_quantize: 0 GPTQ layers`.
  - Only 2 of expected ~many layers are hooked during calibration.
  - Quantizer key lookup matches 0 of calibrated layers.
  - Likely cause: `torch.compile` wraps modules with different internal names; hook attachment points don't survive compilation boundary.
  - **gptq_full** (full training with SKIP_GPTQ=0) is queued to test if GPTQ works in full training context (different module graph).

---

## Key Observations

1. **Quantization gap (quant_gap ~+0.022) is the real opportunity.** All cases show ~0.022 BPB gap between float32 and int6. GPTQ, when working, should close most of this. This is bigger than anything the arch/sched sweep found.

2. **warmdown_4k is a trap.** Longer warmdown on time-based schedule causes EARLIER QAT firing, not later. This is backwards from the expected effect. Do not revisit without switching to step-based schedule.

3. **QAT threshold doesn't matter much at 4×GPU.** qat_early and qat_late both show +0.0004 — symmetric and equal hurt. Either the threshold sweet spot is very narrow or QAT signal is weak at proxy scale.

4. **Legal SLOT passed its gate separately** (-0.0057 at 1200-step 1×GPU proxy). That experiment is tracked in `neural/2026-03-31_QK_Gain_SLOT_Legal/`.

---

## Next Steps

1. **Fix GPTQ**: investigate torch.compile hook attachment, or run `gptq_full` case to test in full-training context.
2. **Legal SLOT full run**: gate passed decisively. Prioritize 8×GPU run.
3. **Arch sweep verdict**: all dead. Do not run 8×GPU for any case in this sweep.
