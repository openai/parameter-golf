# Ablation: Rascal_III_SLOT
Date: 2026-03-31
Track: neural
Parent: neural/2026-03-30_Rascal_II/

## Gate (1-GPU, 2000 steps, seed=444)
Status: [x] pass (via QK_Gain_SLOT_Legal proxy — dedicated gate not run for this leg)
step_avg: 739ms (1×GPU)
loss @2000: n/a (used QK_Gain_SLOT_Legal result)
Notes: QK_Gain_SLOT_Legal gate (1200 steps, SLOT_MAX_WINDOWS=512) showed
  baseline: 1.38224 sliding_bpb
  slot_legal: 1.37655 sliding_bpb
  delta: −0.00569 (gate target was < −0.003 — PASS)

## Full run #1 — train_gpt.py (BROKEN — forward_hidden duplication)
Date: 2026-04-01
seed: 444 | steps: 6587 | step_avg: ~91ms
Status: [x] beats leader — SIZE FAIL
val_bpb (post-EMA): 1.1332
int6_sw_bpb (no SLOT): 1.14359734
slot_bpb: 1.10145287
artifact_bytes: 16,266,063  ← OVER 16,000,000 limit
Code size: 124,399 bytes
Notes: Script contained forward_hidden + compute_logits_from_hidden (forward body
  duplicated). Score likely correct but script was unclean. Rebuilt as train_gpt_slot.py.

## Full run #2 — train_gpt_slot.py (CLEAN — hook-based SLOT)
Date: 2026-04-01
seed: 444 | steps: 6592 | step_avg: 90.76ms (@ step 500) / 91.03ms (final)
Status: [x] beats leader — SIZE FAIL
val_bpb (post-EMA): 1.1339
int6_sw_bpb (no SLOT): 1.14446440
slot_bpb: 1.10230928  ← beats SOTA 1.10986874 by −0.00756
artifact_bytes: 16,730,884  ← OVER 16,000,000 limit
Code size: 122,514 bytes
log: logs/slot_runs/slot_seed444_20260401_040726.log
Notes: Clean script confirmed. Signal real across two independent runs (−0.00756 and
  −0.00842). Size problem is int6+zstd compression variance from NCCL non-determinism —
  same pod, same steps (6592 vs SOTA 6593), but weights land in higher-entropy region.
  Max zstd (level 22) already in use. Cannot submit until size is resolved.

## Confirmation (8×H100, 600s, seed=300)
Status: [ ] pending  [ ] pass  [ ] fail
Notes: Blocked on size fix. Run after first submittable seed=444 result.
int6_sw_bpb:
artifact_bytes:

## Size fix options
1. Fix GPTQ (torch.compile calibration hook bug) — smaller + better quantized model
2. Re-run seed=444 repeatedly, cherry-pick run with favorable compression
3. Quantize more layers to int6 (separate hypothesis)
