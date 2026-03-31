# FA_Wing_Green_1 — Flow Instructions on A-Wing SOTA Base

## The Claim

Weight sharing in the crawler gives a **structural compression advantage** that flat architectures cannot match. If flow instructions hold the quant gap near zero, we get comparable BPB at roughly half the artifact size — a 2× quality-per-byte ratio vs current SOTA.

---

## Architecture

```
4 flat blocks (unique weights, run once each)
+
1 crawler block (shared weights, run 4 times)
= 8 effective transformer layers of compute
≈ 9.5M unique parameters stored
```

A flat model with equivalent compute depth (8 blocks) stores ~14M unique parameters.

**Structural compression: ~35% smaller artifact, same effective depth.**

---

## What Was Confirmed (FX_Wing_Delta, old eval stack)

| Metric | FX_Wing (static inst) | FX_Wing_Delta (flow inst) |
|--------|----------------------|--------------------------|
| val_bpb (600s) | not measured clean | 1.1996 |
| int6 quant gap | **+2.93 BPB** | **+0.006 BPB** |
| artifact size | ~9.3 MB | ~9.3 MB |

**H0 confirmed**: recomputing `inst_k = up_k(proj(x))` from CURRENT x at each loop, rather than pre-planning all instructions from x_enc before loops, reduces the quantization catastrophe from +2.93 → +0.006 BPB.

Root cause: static instructions force the shared block to serve K distinct activation distributions with a single int6 scale. Flow instructions let each loop's instruction implicitly compensate for quantization-distorted activations — self-healing quant.

---

## This Run: FA_Wing_Green_1

First run of the flow architecture on the **A-Wing SOTA eval stack** (competition-comparable numbers).

**Key question:** What is our BPB on the real eval stack, and what is our artifact size?

Expected:
- val_bpb competitive with A-Wing SOTA (~1.11) given same training budget
- int6 quant gap < 0.01 (H0 carries over)
- Artifact: ~9–10 MB (vs ~14 MB for an equivalent flat model)

---

## The Quality-Per-Byte Case

```
quality-per-byte = final_bpb / artifact_MB
```

| Model | BPB | Artifact | Quality/byte |
|-------|-----|----------|--------------|
| A-Wing SOTA (flat) | ~1.11 | ~14 MB | 0.079 |
| FA_Wing_Green_1 (target) | ~1.13 | ~9.5 MB | **0.119** |

If the numbers land, that is a **~50% better quality-per-byte ratio** than any flat architecture in the competition. The crawler architecture isn't just a curiosity — it's a different point on the Pareto frontier.

---

## Decision Criteria

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| val_bpb ≤ 1.15 AND quant gap < 0.05 | Strong result — crawler competes | Multi-seed, submit |
| val_bpb ≤ 1.15 BUT quant gap > 0.2 | H0 doesn't hold on A-Wing base | Investigate why |
| val_bpb > 1.20 | Architecture needs more training budget | Longer run |
| val_bpb > 1.25 | Crawler is net negative vs flat | Park it |
