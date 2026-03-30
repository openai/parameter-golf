# Phase 2: Multiskip Connections

**Dates:** Mar 23, 2026
**Goal:** Test whether multiskip (U-Net-style) connections between layers improve BPB under the same parameter budget.
**Outcome:** Multiskip showed promise as a modifier but wasn't a standalone breakthrough. Established as a component for later architectures.

## Runs

| Run | Config | Notes |
|-----|--------|-------|
| 001-multiskip-x0 | Baseline + multiskip | First test |
| 002-multiskip-n6 | n=6 iterations | Deeper recurrence |
| 003-multiskip-n8 | n=8 iterations | Even deeper |
| 004-frac-warmdown | Fractional warmdown schedule | Training optimization |
| 005-frac-warmdown-multiskip | Combined | Best of both |
| 006-frac-warmdown-multiskip-524k | Batch size 524k | Throughput push |
| 007-multiskip-524k | Multiskip + large batch | Scaling test |
| 008-multiskip-n6-524k-lr04 | n=6, 524k batch, LR=0.04 | Aggressive LR |
| 009-hsm-multiskip-262k | Hierarchical state mixing | Hybrid approach |
| 8xH100/ | Full-scale runs | Select configs on 8 GPUs |

## What Led to Phase 3

Multiskip helped but didn't change the game. The question became: can we replace attention entirely with something faster (SSMs) to see more tokens in 10 minutes?
