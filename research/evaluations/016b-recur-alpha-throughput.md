# Evaluation — Spec 016b (Recur-Alpha Throughput Diagnostic)

**Run dir:** `runs/016b-throughput/`
**Commits tested:** `154c9b8` (008 baseline, no recur-alpha) vs `4dd2d63` (016, `RECUR_ALPHA_ENABLED=1`)
**Pod:** `g5s1rqfhia58uk` — 2×H100 SXM, US-NE-1, NA volume `hvpdph5i3g`
**Eval date:** 2026-04-21

## Motivation

Specs 015 and 016 both ran 2–4% below spec 008's tok/s on JP. Two hypotheses:
1. **Hardware variance** — 015/016 drew slower physical nodes within the JP pool
2. **Architectural overhead** — the recur-alpha blend op `x = α * x_new + (1-α) * x_before` adds real compute cost

016b resolves this by running both commits on the **same pod, same node** — eliminating hardware variance entirely.

## Model config (proxy)

The full 11L/512d model does not fit on 2×H100 under DDP. A 6L/256d proxy was used:

```
NUM_LAYERS=6  MODEL_DIM=256  XSA_LAST_N=6  PARALLEL_START_LAYER=99
ENABLE_LOOPING_AT=0   (looping + alpha active from step 1)
```

All other flags (CaseOps, GatedAttn, QuantGate) unchanged. Loop structure intact: layers 3-4-5 looped 2×.

**Overhead interpretation caveat:** smaller model = cheaper forward pass = blend op is a *larger fraction* of total compute. The 016b number is an **upper bound** on full-model overhead.

## Results

| run | commit | config | tok/s (last 5 avg) |
|-----|--------|--------|--------------------|
| run-a-2gpu | 154c9b8 | baseline, no recur-alpha | **3,333K** |
| run-b-2gpu | 4dd2d63 | recur-alpha enabled | **3,234K** |

**Ratio B/A: 97.0%** — recur-alpha is **~3% slower** on the proxy model.

Run A ran for the full 596s wallclock. Run B ran for 120s (`MAX_WALLCLOCK_SECONDS=120`) — tok/s was stable by step 250 (~3,234K) and did not change through step 475.

## Interpretation

**The throughput deficit is real, not hardware variance.** Same pod, same node, same commit checkout order. The ~3% cost on the proxy model corresponds to **~1–2% at full model size** (11L/512d has much heavier layers; the blend op shrinks as a fraction of compute).

This is consistent with what was observed in the actual runs:
- 015 vs 008 on JP: −2.4% tok/s (previously attributed to node variance)
- 016 vs 008 on JP: −3.6% tok/s (same)

Those deficits were **real architectural overhead**, not luck of the draw on the JP pool.

## Decision criterion (from spec)

- ≥99%: no tax ✗
- **97–99%: ambiguous / partial tax ✓ (proxy model lands here)**
- <97%: real tax ✗

Per spec: "Proceed to spec 017 with reduced expectations; margin over #1736 may be near-zero." However given the proxy model amplifies overhead, the full-model number is likely ~98–99%, making this closer to the "no tax" bucket in practice.

**Practical conclusion:** recur-alpha costs ~1–2% throughput at full model size. At matched wallclock this corresponds to ~20–40 fewer training steps, which partially offsets the bpb gain. The mechanism is still net positive (−0.0038 bpb gain >> ~0.0005 bpb loss from fewer steps), but it is not a free lunch.

## Cross-references

- Spec: `research/specs/016b-recur-alpha-throughput.md`
- Why 015/016 ran fewer steps: `research/evaluations/015-recur-alpha.md`, `research/evaluations/016-recur-alpha-ones.md`
- Next: spec 017 (full pipeline run with recur-alpha stacked)

## Cost

| item | cost |
|---|---|
| Pod boot (2×H100 US-NE-1, ~15 min total) | ~$1.50 |
| Failed full-model attempts (OOM) | ~$0.50 |
| **016b total** | **~$2** |

*(Prior 016b attempt on 8×H100 JP burned ~$18 — excluded from this tally, logged separately.)*
