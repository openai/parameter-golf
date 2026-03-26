# X-WING: Shared N-gram Tables + Cubric

## Results

| Seed | Sliding BPB | N-gram + Cubric BPB | Artifact |
|------|-------------|---------------------|----------|
| 1337 | 1.1190 | **0.5640** | 15.59 MB |
| 42 | 1.1193 | **0.5642** | 15.59 MB |
| 300 | 1.1197 | **0.5651** | 15.63 MB |
| **Mean** | **1.1193** | **0.5644** | — |

## What Changed vs Podracing III (#782)

One structural fix to n-gram eval, same training:

1. **Shared n-gram tables (chunk-based)**: Previous podracer gave each GPU rank its own hash tables — with 8 GPUs, each rank only saw 1/8 of val tokens. X-WING groups eval windows into ~1M-token chunks. All ranks score their assigned windows, then ALL ranks update tables with the same chunk tokens. Every rank gets the full 62M-token picture. This is the single change that closes the 0.37 BPB gap between rank-local (0.936) and shared (0.564) tables.

2. **Cubric per-order scaling (retained)**: Same proven adaptive alpha multipliers — suppress noisy orders 2-3, boost reliable orders 5-7. Converged to: `o2:0.45 o3:0.30 o4:0.45 o5:1.94 o6:2.00 o7:2.00`.

## Key Insight

With 8 GPUs and rank-local tables, each rank builds n-gram statistics from only ~7.75M tokens. With shared tables, every rank sees all 62M tokens — 8x more context for probability estimation. Higher-order n-grams (5-7) benefit most because they need large corpora to accumulate meaningful counts.

## Compliance

- Score-first, backward-looking: all windows in a chunk are scored BEFORE that chunk's tokens update the tables
- Chunk boundaries align with scored positions, not window starts
- Alpha depends solely on model's own softmax entropy — no target/label access
- Per-order Cubric multipliers use beat-rate statistics from already-scored tokens
- No oracle selection, no min-NLL comparison
- GPTQ calibration runs inside training phase (before wallclock stop)

## Credits

- Shared n-gram table insight: @deanbrr (PR #779)
- N-gram eval cache concept: @deanbrr (PR #659)
- Multi-order backoff + adaptive alpha: @Asukabot0 (PR #727)
- Per-order adaptive alpha scaling (Cubric): @newjordan (original)
- Base architecture: @signalrush (PR #414)

## Reproduce

```bash
SEED=1337 NPROC_PER_NODE=8 bash concepts/xwing/run.sh
```

8xH100 SXM, 600s training + ~225s eval.
