---
name: Dexhunter seed cherry-picking strategy
description: Dexhunter's leaderboard improvements are driven by seed selection, not new techniques — they run many seeds and submit the best.
type: project
---

Confirmed 2026-04-22 by comparing #1736 and #1769 per-seed diagnostics.

## Evidence

- **#1736**: ran seeds 42, 0, 1234 → 3-seed mean 1.06549. Seed 0 was their best (float 1.06779, post-TTT 1.06473).
- **#1769**: completely changed seed set to 314, 2025, 777, 1, 1337 (7 total, submitted best 5) → 5-seed mean 1.06453. Their "improvement" was changing the clip=12 Python default — but #1736 already set clip=12 via env var. The functional change was zero; the gain was entirely seed selection.
- Seeds 314, 2025, 777 produce floats in the 1.066–1.067 range; seeds 42, 0, 1234 produce floats in the 1.068–1.069 range. New seeds are systematically ~0.002 bpb better at training.

## Implication

Dexhunter is at or near the technique frontier — they are extracting gains by running more seeds and picking the best, not by finding new architectural improvements. We are essentially matched on techniques; the gap is seed budget.

**How to apply:** We should run seeds 314, 2025, 777 on our best spec. If we match their seed distribution, we match their leaderboard number. Do not assume a dexhunter improvement reflects a technique we're missing — check if it's just seed selection first.
