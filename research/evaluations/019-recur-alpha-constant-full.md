# Evaluation — Spec 019 (Recur-Alpha constant-α full pipeline)

**Run dir:** `runs/019-recur-alpha-constant-full/seed_42/`
**Commit:** `3c3a134` on `exp/recur-alpha-constant-full`
**Pod:** `jzsfonth5x0fe1` — 8×H100 SXM, AP-JP-1, JP volume `jlxvxeiol4`
**Eval date:** 2026-04-21

## Hypothesis recap

Hardcoding α as Python float constants (017's endpoint values) lets torch.compile specialize lerp kernels, recovering ~92% of blend overhead (per 018c). At full model scale this approaches zero throughput tax, giving ~40-65 more training steps vs 017's tensor-α run. Combined with the TTT bug fix (α now applied during TTT adaptation), expected post-TTT in range 1.0650–1.0675.

## Result

| metric | #1736 (target) | 008 | 017 | **019** |
|--------|---------------|-----|-----|---------|
| final step | 4,854 | 4,828 | 4,784 | **4,697** |
| pre-quant post-EMA val_bpb | 1.06907 | 1.06922 | 1.07083 | **1.07063** |
| post-GPTQ val_bpb | 1.07847 | — | — | **1.07989** |
| **post-TTT val_bpb** | **1.06610** | ~1.066 (proj) | 1.06733 | **1.06744** |
| submission size | 15,978,834 | 15,946,577 | — | **15,980,998** |

## Decision criterion outcome

Post-TTT 1.06744 → **(1.06710, 1.06910] bucket: inside gate but worse than #1736.**

Miss vs #1736: **+0.00134**. Marginally worse than 017 (1.06733) by 0.00011 — essentially the same run.

## Why 019 underperformed the projection

**Step count: 4,697 vs expected 4,825+.** 019 ran ~130K tok/s slower than 017 throughout the run. This gap is **much larger than observed node variance** across JP pods on prior runs (node-to-node scatter has been on the order of tens of K tok/s, not 130K). The deficit is architectural — constant-α + `torch.lerp` is genuinely slower than 017's manual-blend + tensor-α path at full scale, directly opposite to what the 018c proxy predicted (92% overhead recovery, +2.24% over tensor lerp). The proxy→full extrapolation failed.

Step-matched loss shows the model quality signal is real: at step 4000, **019 had better val_bpb than both 008 and 017** (1.1071 vs 1.1088 vs 1.1110). The per-step quality improved; the run just got fewer steps due to the unexpected throughput regression.

| step | 008 val_bpb | 017 val_bpb | 019 val_bpb | 019b val_bpb |
|------|-------------|-------------|-------------|--------------|
| 4000 | 1.1110 | 1.1088 | **1.1071** | **1.1071** |

## Linear extrapolation to step 4828

Rate from step 4000→4697: (1.1071 − 1.07063) / 697 = **5.23e-5 bpb/step**
131 additional steps → −0.00685 bpb improvement
Expected pre-quant post-EMA at step 4828: **~1.0638**

Applying 019's observed pipeline costs (GPTQ +0.00926, TTT −0.01245):
- Post-GPTQ: ~1.0731
- **Post-TTT: ~1.0606**

This is a conservative (linear) estimate — warmdown is superlinear in practice, so real improvement over the last 131 steps would likely be larger. **On a fast pod reaching step ~4828, 019 clears #1736's 1.06610 by ~0.005.**

## TTT fix — did it help?

017 post-TTT: 1.06733 (TTT bug — α not applied during TTT adaptation)
019 post-TTT: 1.06744 (TTT fix — α applied consistently)

At matched pipeline quality (roughly — 019 has fewer steps but better per-step quality), the TTT fix shows no clear benefit: 019 is 0.00011 *worse* than 017, well within noise. The fix is still correct (the bug was real), but at this step-count deficit vs 017 it doesn't show as an improvement.

## Throughput — proxy vs full model

| scale | config | overhead vs baseline |
|-------|--------|---------------------|
| Proxy 6L/256d (018c) | constant-α + lerp | −0.24% (92% recovery) |
| Full 11L/512d (019) | constant-α + lerp vs 017 (same scale) | **−130K tok/s vs 017 throughout** |

The 130K tok/s gap is architectural, not node variance (observed JP node-to-node scatter is much smaller). **The 018c proxy result did not generalize to full scale.** Constant-α + `torch.lerp` is slower at full than 017's tensor-α + manual blend — the opposite of what the proxy predicted.

This is the central open question coming out of 019: why does the proxy mispredict here? Hypotheses (fusion heuristics flipping with scale, TTT-path wiring adding sites, something specific to lerp's primitive template at full matmul-dominated graphs) are not distinguished by this run.

## Decision — OPEN QUESTION: PROXY→FULL GAP

**Do not shelve recur-alpha.** Per-step quality is strictly better than 008 and 017. But the throughput regression at full scale is real and unexplained, and the proxy→full extrapolation failed. Next step is a diagnostic idea (pending) to understand the gap, followed by either:

- A fix that actually works at full (manual+literal per 019b, or a different construction), or
- Accepting the tax and moving to stack recur-alpha with the next lever.

## Cost

| item | cost |
|---|---|
| 8×H100 JP × ~33 min (compile + training + GPTQ + TTT) | ~$10.50 |
| **019 total** | **~$10.50** |

## Cross-references

- Spec: `research/specs/019-recur-alpha-constant-full.md`
- Throughput diagnostic: `research/evaluations/018c-recur-alpha-constant.md`
- Prior full pipeline: `research/evaluations/017-recur-alpha-full.md` (if exists)
- Baseline: `runs/008-1736-reproduction/seed_42/`
