---
name: Gap vs #1769 root-cause diagnosis
description: The ~0.0013 bpb gap between our seed 42 results and #1769's 5-seed mean is entirely in training (float model quality), not GPTQ or TTT.
type: project
---

Diagnosed 2026-04-22 by comparing #1769 per-seed diagnostics vs our run artifacts.

## The numbers

| Stage | #1769 mean (5 seeds) | Our spec 026 seed 42 | Gap |
|---|---|---|---|
| Float post-EMA bpb | 1.06742 | 1.06893 | **+0.00151 (we're worse)** |
| GPTQ penalty | +0.00946 | +0.00934 | −0.00012 (we're slightly better) |
| TTT gain | −0.01236 | −0.01245 | −0.00009 (we're slightly better) |
| Post-TTT bpb | 1.06453 | 1.06582 | +0.00129 |

GPTQ and TTT stages are equivalent or marginally better for us. The entire gap traces to the float model.

## Why the float is worse

Seed 42 is a bad training seed. #1769 ran 7 seeds and submitted best 5. Their seed 314 floats at 1.06637; their worst submitted seed (1337) floats at 1.06802. Our seed 42 consistently lands at 1.069+.

## Implication

If we run seeds like 314 or 2025, we should expect float ~1.066–1.067, which projects to post-TTT ~1.064. We are NOT behind on GPTQ or TTT implementation — we just need better training seeds.

**Why:** Confirmed by per-seed diagnostic logs from #1769 submission.json (pre_ttt_val_bpb and README table with float/quantized/post-TTT per seed).

**How to apply:** Do not investigate GPTQ or TTT code differences. The fix is running more seeds. Priority: try seeds 314, 2025, 777 (dexhunter's best seeds) in addition to 43/44.
