# Evaluation 021d — Recur-α as nn.Parameter(requires_grad=False), 8×H100 JP

**Spec:** `research/specs/021d-recur-alpha-param-frozen-8xh.md`
**Run:** `runs/021d-recur-alpha-param-frozen-8xh/seed_42/`
**Date:** 2026-04-22
**Commit:** `8b2d791`
**Status:** training + GPTQ completed; TTT skipped (pre-quant EMA sufficient for decision)

---

## Result summary

| metric | value |
|---|---|
| Hardware | 8×H100 SXM AP-JP-1 ($23.92/hr, ~25 min) |
| Final step | 4881 / 20000 (wallclock cap 596s) |
| Loop activation | step 2154 @ frac 0.350 |
| val_bpb @ step 4000 | 1.1135 |
| val_bpb @ step 4881 (in-training) | 1.0701 |
| Pre-quant post-EMA val_bpb | **1.06960** |
| Post-GPTQ val_bpb | 1.07896 |
| Post-TTT val_bpb | not run |

---

## Comparison

| run | commit | hardware | steps | pre-quant EMA | post-TTT |
|---|---|---|---|---|---|
| #1736 reference | — | 8×H JP | — | — | **1.06610** |
| 019b (literal-α) | `e93d77d` | 8×H JP | 4824 | 1.06951 | **1.06628** |
| 021d (param-α bf16) | `8b2d791` | 8×H JP | 4881 | **1.06960** | not run |
| 021 (buggy buffer) | `cb5cd78` | 8×H JP | 4883 | 1.06963 | 1.06900 |

**021d vs 019b pre-quant EMA: +0.00009** — noise level. 57 more steps from cleaner throughput, but those steps didn't move the needle.

---

## Step-matched loss vs 019b

| step | 019b | 021d | Δ |
|---|---|---|---|
| 2200 (post-loop) | 2.5341 | 2.5424 | +0.0083 |
| 2500 | 2.5551 | 2.5606 | +0.0055 |
| 3000 | 2.5613 | 2.5705 | +0.0092 |
| 3500 | 2.5617 | 2.5721 | +0.0104 |
| 4000 | 2.4023 | 2.4125 | +0.0102 |
| 4500 | 2.2632 | 2.2854 | +0.0222 |

The gap opens post-loop and holds at +0.007–0.022 through warmdown. The Parameter trick did not close the per-step loss deficit that 021's buffer showed.

---

## Throughput

| step | tok/s 021d | tok/s 019b |
|---|---|---|
| pre-loop (~1000) | 8.14M | 8.11M |
| 2200 (just post-loop) | 8.04M | 7.98M |
| 3000 | 7.18M | — |
| 4000 | 6.68M | 6.68M |

No Type B compile stalls observed — 021d's throughput profile is as clean as 021-buggy (4881 steps vs 021-buggy's 4883). The Parameter change did eliminate stalls as hypothesised. But at 8×H on this pod, 019b was also clean — so 021d's extra ~57 steps vs 019b are attributable to a slightly faster node, not to stall elimination.

---

## Why the 4×H mini (021c) didn't predict this

021c Arm B (Parameter+bf16, 4×H NE-1) matched Arm A (019b-4H) within +0.00025 pre-quant EMA. At 8×H the gap is +0.00009 EMA but the per-step train_loss is +0.007–0.022. The contradiction resolves: at 4×H, both arms ran ~5004–5034 steps on matched hardware, so the endpoint EMA masked per-step differences. At 8×H, the per-step loss gap is real and persistent — Parameter didn't enable Inductor const-folding at the scale that matters.

---

## Decision

**Arc closed. Shelve Parameter-α and buffer-α variants.**

The full mechanism arc from 021 through 021d has now established:
- Buffer dtype (fp32 → bf16): closed ~half the per-step gap at 8×H.
- α-value fix (dc0b5f8): not the cause of the gap.
- Parameter vs buffer container: does not close the residual gap at 8×H.

No known fix remains. The per-step gap on the 8×H literal-α 019b baseline is load-bearing — the literal Python constant is getting special treatment from Inductor that no buffer/Parameter variant can replicate without rewriting the fusion.

**019b (literal-α, post-TTT 1.06628) remains our best result.** It misses #1736 by 0.00018, within seed std.

### Next steps

1. **019b seed replication (seeds 43/44)** — the 0.00018 miss is within seed variance. A second seed on a fast pod could tie or beat #1736. See 019b-rerun results for calibration.
2. **Spec 022 (extra TTT depth)** — unblocked, separate mechanism entirely.
3. **Spec 023 (TTT LoRA bias)** — unblocked.
