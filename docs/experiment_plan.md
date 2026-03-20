# Experiment Plan (Updated 2026-03-20 — WE ARE THE LEADER)

## Current State
- **Our best**: 1.1574 BPP (PR #114 — int6 + MLP 3x + slide stride=256 + our training stack)
- **Closest competitor**: PR #106 at 1.1580 (0.0006 behind us)
- **Baseline**: 1.2244
- **Our PRs**: #61 (1.2154, warmdown discovery), #96 (1.1764, sliding window), #114 (1.1574, int6+MLP3x)

## Competition Landscape (2026-03-20)
| PR | BPP | Key Techniques |
|----|-----|---------------|
| **#114 (OURS)** | **1.1574** | **Int6 + MLP 3x + slide=256 + train@2048 + clip=0.3 + batch=786K** |
| #106 | 1.1580 | Unknown tweaks on PR #88 |
| #88 | 1.1605 | Int6 STE + MLP 3x + MTP + slide + zstd |
| #99 | 1.1605 | Int6 + MLP 3x + Late-K passthrough + slide=64 |
| #102 | 1.1618 | Int6 + MLP 3x + SmearGate + slide=64 |
| #89 | 1.1622 | NorMuon + int6 STE + SWA + slide |

## What Makes Our Submission Unique
- **Train@2048 (not 1024)**: we proved train@2048 = train@4096 with sliding window; others train@1024
- **Batch=786K**: systematically swept 393K-1M, found optimum nobody else tested
- **GRAD_CLIP_NORM=0.3**: swept 0.0-1.0, found narrow sweet spot for long sequences
- **FP16 embed + Late-K passthrough combined**: PR #99 has Late-K only, PR #42 has FP16 embed only, we have both
- **Stride=256 beats stride=64**: we found diminishing returns — stride=256 is slightly better at 4x less eval time

## Critical Pattern: Step Throughput Is King
On a 10-min budget, per-step overhead >10% is a net loss. Killed:
- NorMuon: 110ms/step (54% overhead) → 1.1857
- MTP: 86ms/step (83% overhead) → 1.2083
- ALBERT: 80ms/step (70% overhead) → 1.2347

MLP 3x adds 77% overhead (83ms vs 47ms) but the 28% more parameters compensate. This is the only per-step-expensive technique that actually pays off.

---

## NEXT PRIORITIES (to defend lead and extend it)

### Priority 1: Multi-Seed Validation
Run 3-5 seeds to prove p < 0.01. Competition requires statistical significance. Use SEED={1337, 1338, 1339, 1340, 1341} with the exact PR #114 config.

### Priority 2: Quick Sweeps on Current Config
- MLP_HIDDEN in {1280, 1408, 1536, 1664} — is 1536 actually optimal?
- QK_GAIN_INIT in {1.5, 1.7, 1.9} — PR #99 uses 1.7
- stride in {128, 256, 512} — confirm 256 is the sweet spot
- WARMDOWN_ITERS in {2000, 3000, 4000} with int6 config

### Priority 3: 10 Layers with Int6
Int6 saves ~25% artifact space. 10 layers + MLP=1280 might fit under 16MB. More depth with slightly narrower MLP. Quick size check first.

### Priority 4: SP4096 + Int6 + MLP 3x
SP4096 at dim=480 got 1.1774 with int8. With int6 + MLP 3x, the model gets bigger and compresses better. This is our unique combo nobody else has.

### Priority 5: Test-Time Training (Novel — Future Direction)
Adapt model to each eval window via gradient steps, then score. Nobody has tried this. Save for after grant funding.

## Proven Not to Work (Full List)
| Approach | Result | Why It Failed |
|----------|--------|---------------|
| ALBERT factorization | 1.2347 | torch.compile 80ms/step, bottleneck kills quality |
| NorMuon | 1.1857 | 110ms/step, throughput loss |
| MTP (multi-token prediction) | 1.2083 | 86ms/step, throughput loss |
| SwiGLU at iso-params | 1.2258 | Narrower hidden doesn't beat ReLU² |
| EMA weights | Much worse | Model never converges enough |
| NUM_KV_HEADS=2 | 1.1838 | Capacity loss not worth savings |
| Int6 step=4 rounding with LR=0.06 | 1.3446 | Wrong quant approach + wrong LR |
| WD=20000 with LR=0.02 | Worse | Low LR already smooth |
| Depth recurrence | All worse | Every team failed, step time penalty |
| eval@2048 on well-trained models | Worse | NTK distortion |
| WD=30000 | Worse | Decays too fast |
| Batch=1M | Worse | Too few steps |
