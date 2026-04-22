# Spec 015 seed_42 screen — execution notes

**Pod:** `k9wwhapqaufb0u` (8×H100 SXM, AP-JP-1, $23.92/hr), stopped after rsync.
**Commit:** `a9aa141` (`exp/recur-alpha` on `fork`).
**Env:** standard #1736 screening env + `RECUR_ALPHA_ENABLED=1`, `TRAIN_LOG_EVERY=100`. No `ENABLE_LOOPING_AT` override (kept default 0.35 per spec).
**Status:** **completed — endpoint val_bpb captured, post-training stages skipped per screening-mode memory.**

## Results summary

| measurement | spec 015 | spec 008 | Δ |
|---|---|---|---|
| stopping_early step | 4761 | 4828 | −67 (JP throttled ~98.6%) |
| training wallclock | 596 s | 596 s | ≈ equal |
| endpoint val_bpb | **1.0696** | 1.0697 | **−0.0001** |
| pre-GPTQ post-EMA val_bpb | **1.06916** | 1.06922 | −0.00006 |
| matched-step val_bpb @ 4000 | **1.1078** | 1.1110 | **−0.0032** |
| tok/s late training | ~6,300,000 | ~6,600,000 | pod variance |

**Per spec decision criterion (endpoint Δ):** −0.0001 lands in the *null* bucket (−0.0003, +0.001). But the matched-step @4000 Δ = −0.0032 is a clear improvement, and spec 008 had 67 extra training steps that 015 didn't get (JP hardware variance). Research should interpret.

## α trajectory (the actual story)

Layout: `[[pass2_L3, pass2_L4, pass2_L5], [pass3_L3, pass3_L4, pass3_L5]]` — 2 extra passes × 3 looped layers (3,4,5).

```
step  | pass-2 (earlier extra)        | pass-3 (later extra)
------+-------------------------------+----------------------
2000  | 0.00  0.00  0.00              | 0.00  0.00  0.00       pre-activation
2142  | layer_loop:enabled (frac 0.350)
2100* | 0.03  0.07  0.14              | 0.16  0.24  0.33       just activated
2200  | 0.68  0.90  0.93              | 0.37  0.43  0.58
2300  | 0.91  1.14  1.28              | 0.65  0.61  0.69
2500  | 1.00  1.16  1.37              | 0.85  0.76  0.75
2700  | 1.03  1.16  1.38              | 0.93  0.82  0.75
3000  | 1.04  1.16  1.38              | 0.98  0.86  0.76
3500  | 1.04  1.16  1.38              | 1.00  0.89  0.77
4000  | 1.04  1.16  1.38              | 1.01  0.89  0.77       saturated
4761  | 1.04  1.16  1.38              | 1.01  0.89  0.77       (endpoint)
```

*The 2100 log entry was the first post-activation dump (activation happened at step 2142, recur_alpha log fires at step 2200 log interval).

**Observations:**
1. **Pass-2 (earlier extra pass)** saturates by step ~2500: values [1.04, 1.16, 1.38]. Model wants **>1.0** at layers 4 and 5 — literally amplifying loop contribution, not just standard Loop345.
2. **Pass-3 (later extra pass)** ramps slower, settles by step ~3500: [1.01, 0.89, 0.77]. Deeper in the loop → partial commitment.
3. **Depth gradient asymmetry**: pass-2 α *increases* with layer (L3 1.04 → L5 1.38); pass-3 α *decreases* with layer (L3 1.01 → L5 0.77). The learned shape is non-trivial, non-degenerate.
4. **Matches spec 015's "Mixed / intermediate" diagnostic bucket** — neither all-zero (opt-out) nor all-one (baseline). The model prefers a non-uniform learned-amplitude-per-pass pattern.

## Logging caveat

`recur_alpha grad_norm` printed as `0.000000` every step, including post-activation. This is a **logging bug, not an optimizer failure** — α values are clearly moving (up to 1.38), so the gradient path works. The log computes `grad_norm` after `optimizer.step()` has zeroed the grads. Cosmetic only.

**Therefore the spec 015 stop-early criterion ("α grad_norm exactly 0 for 5+ consecutive entries AFTER activation") cannot be evaluated from the log as-written**; it would always fire. Research should either patch the log to snapshot pre-zero-grad, or remove this criterion.

## Hardware variance note

Training throughput: ~8.1M tok/s initial → ~6.3M tok/s late (normal thermal/cache-pressure drift). Spec 008's pod was faster: 8.06M → 6.6M — got 67 extra steps in the same 596s cap. This ~1.4% throughput delta is well within the spec-008-documented H100 pool variance (spec 000 was 85% of SOTA step rate).

## Deliverables locally

| file | size | notes |
|---|---|---|
| `train.log` | 31 KB | Full training log incl. 50 recur_alpha entries and endpoint val_bpb |
| `final.json` | — | Deliverable metadata |
| `notes.md` | — | This file |

**NOT present** (intentionally skipped):
- `final_model.pt` (pre-GPTQ FP) — training was killed before `serialize()` to avoid the `pyminify` missing-dep crash and TTT burn.
- `final_model.int6.ptz` — same.
- post-quantization diagnostic, sliding-window val, quantized-TTT val — not a screening-mode measurement.

## Cost accounting

| Item | Cost |
|---|---|
| First smoke (JP 8×H100, halted on log-artifact false alarm) | ~$3.50 |
| JP→NA rsync (1×H100 JP + NA prep pod running) | ~$0.85 |
| Earlier pod probes / mis-deletes | ~$0.50 |
| Screen run (JP 8×H100, ~15 min wall) | ~$6.00 |
| **Total spec 015 execution** | **~$10.85** |

## Handback

**Execution recommendation:** research eval this run. The α trajectory is the rich signal here, not the endpoint bpb which landed in the null bucket due to JP step deficit. Open questions:
- Should endpoint val_bpb be re-measured at step-matched ~4828 (would need another ~0.5 min of training — new run) or treated as "~tied" at endpoint?
- Does the learned α pattern (pass-2 amplifies later layers, pass-3 damps them) motivate an architectural follow-up — e.g. fixed-point α matching this shape as a prior, or per-layer loop-budget reallocation?
- If we go to full-pipeline (TTT/GPTQ) run, we first need `pyminify` installed on pod template (missing causes late-stage crash) — flag for EXECUTION.md.
