# Spec 026 screen — seed 314 on 4×H100 (NE-1)

**Pod:** `wlgp6nlp4u1p9e` @ NE-1, 4×H100 SXM, $11.96/hr, image `runpod/parameter-golf:latest`
**Commit:** `d70888f` (exp/recur-alpha-buffer) — 025c per-pass frozen + LoRA warm-start-A
**Seed:** 314
**Launched:** 2026-04-22 15:00:36 UTC
**Stopped (training):** 15:28 UTC (`stopping_early: wallclock_cap` at step 4998, train_time 1196.3s)
**Total pod wall so far:** ~30 min, ~$6

## Result vs spec's screen gates

| metric | 025b seed 42 (baseline) | 026 screen 314 | spec target | verdict |
|---|---|---|---|---|
| steps in 1200s | 4756 | **4998** | ~4750–4870 | ✅ +242 steps |
| val@4000 | 1.1079 | **1.1159** | ≤ 1.105 | ❌ miss (+0.0080) |
| final-step val_bpb | ~1.069 | **1.0682** | — | |
| **pre-quant EMA val_bpb** | **1.06917** | **1.06770** | **< 1.068** | ✅ pass (−0.00147) |
| quantized val_bpb | — | 1.07725 | — | |
| submission size (brotli) | — | 15.97 MB | < 16 MB | ✅ |

**Screen PASSES on the pre-quant EMA gate** → per spec "Pass → proceed to 8×H full pipeline."

## Throughput — matched-step cumulative tok/s vs 025b seed 42

| step | 025b seed 42 | 026 screen 314 | Δ |
|---|---|---|---|
| 500  | 3.98M | 4.21M | +5.7% |
| 1000 | 3.98M | 4.21M | +5.6% |
| 1500 | 3.98M | 4.21M | +5.6% |
| 1700 | 3.98M | 4.20M | +5.6% (post loop-activation ~1665) |
| 1900 | 3.99M | 4.21M | +5.6% |

NE-1 pod was ~5-6% faster than the 025b seed 42 baseline throughout.

## Matched-step train_loss

Δ = 026 − 025b_seed_42; positive = current worse.

| step | 025b_42 | 026_314 | Δ |
|---|---|---|---|
| 100  | 3.6492 | — | |
| 500  | 2.6759 | 2.6636 | −0.0123 |
| 1000 | 2.7667 | 2.7696 | +0.0029 |
| 1500 | 2.5932 | 2.5919 | −0.0013 |
| 2000 | 2.6538 | 2.6558 | +0.0020 |
| 2500 | 2.5000 | 2.5105 | +0.0105 |
| 3000 | 2.5687 | 2.5820 | +0.0133 |
| 3500 | 2.3875 | 2.4021 | +0.0146 |
| 4000 | 2.3758 | **2.3946** | **+0.0188** (val@4000=1.1159) |
| 4500 | 2.3442 | 2.3719 | +0.0277 |
| 4700 | 2.3345 | 2.3571 | +0.0226 |
| 4800 | —      | 2.2752 | — (beyond baseline) |
| 4900 | —      | 2.3290 | — |

Seed 314 ran +0.01–0.03 worse on train_loss through the loop-activation window, but EMA smoothing + the extra ~240 steps closed the gap at the endpoint and crossed the pre-quant gate.

## Incidents worth recording

1. **First launch (commit preflight):** commit `d70888f` was not on fork. Pushed `exp/recur-alpha-buffer` → fork; retry succeeded. Saved memory: always push spec's pinned commit before pod preflight.
2. **Spec text inconsistency (pre-fix):** spec header originally said "025b shared-frozen + LoRA" but commit contained 025c per-pass buffers. Research pushed commit `c429953` correcting header + sanity greps. Pod stopped while research reconciled; resumed same pod after fix.
3. **NE-1 /runpod symlink layout:** `/runpod → /workspace` breaks the data path because NE-1 has CaseOps data at `/workspace/parameter-golf/data/…`, not `/workspace/data/…`. Fixed with per-subdir symlinks (`/runpod/{data, parameter-golf, runs}` → `/workspace/parameter-golf/{data, .}` + `/workspace/runs`). Saved memory `reference_ne1_missing_caseops.md`.
4. **Stock:** JP out of stock at 2/4/8×H during screen attempt; NE-1 4×H landed. JP + NE both out of 8×H when trying to provision for the full pipeline (2026-04-22 ~15:30 UTC).

## Artifacts

- `train.log` — full stdout
- `3cb19989-4662-48ea-a3e0-27336c003d78.txt`, `8c9e158a-…` — per-rank log files
- `launch.out` — empty (setsid detached cleanly)
- Checkpoints on pod container disk only (not rsync'd for screen)

## Decision

Per spec accept criteria: pre-quant EMA passes the gate → **launch 8×H full pipeline** (seed 314 with PHASED_TTT_ENABLED=1, TTT_LORA_ALPHA=144, TTT_WEIGHT_DECAY=1.0). Currently blocked on 8×H stock (JP + NE both unavailable at 15:30 UTC).
