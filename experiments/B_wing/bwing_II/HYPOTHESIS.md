# B-WING II — Cubric + Entropy Shift + Fast TTT

## Hypothesis
Stack everything:
1. Cubric 3D ON with warm-start (our edge — per entropy×count adaptation)
2. Per-order entropy shift from #809 (-0.25 per order above min)
3. Alpha 0.05-0.60, clip 0.95 from #809
4. Our sliding-window TTT (score-first, SGD, 1 epoch for speed)

TTT adapts the model BEFORE n-gram eval runs. The n-gram cache
then operates on improved model probabilities.

## Changes from bwing_full_port
- CUBRIC_CADENCE=32 (was 0 — cubric back ON)
- NGRAM_ORDER_MULTS removed (cubric handles per-order scaling)
- TTT_ENABLED=1 (fast: 1 epoch, freeze 2 blocks, SGD+momentum)
- NGRAM_EVAL_MAX_SECONDS=0 (no time limit on n-gram eval)

## Expected timing
- Training: ~600s
- TTT: ~30-60s (1 epoch, fast SGD)
- N-gram: ~180s
- Total eval: ~250-300s (within 600s budget)
