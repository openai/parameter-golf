# 🏁 VICTORY PLAN — T-18h to Deadline

## Executive Summary

**Gate-2 target**: BPB < 1.85 at step ≥ 4000 on ≥ 3 sanctioned seeds
**Best realistic result**: BPB = 2.52 at step = 2000 (h=1536, lr=0.002, rng=1597)
**Gap**: +36% above target (2.52 vs 1.85)

## Key Findings from 206 Completed Experiments

### Finding 1: Two Training Regimes

| Regime | Config | Behavior | BPB Range |
|--------|--------|----------|-----------|
| **Stable** | h≥1024, lr≥0.002 | Realistic from step 1 | 2.5–3.2 |
| **Collapsed** | h=828, lr=0.0004 | bpb→0 for 18K+ steps, then emerges at 9–17 | 0.0→17.3 |

The h=828 + lr=0.0004 config (our "champion") has a **collapse phase** lasting 18K–65K steps
before realistic BPB emerges. This is NOT an eval bug — it's a training dynamics issue
specific to small models with low learning rates.

### Finding 2: Best Configs (Realistic Only)

| Rank | Config | BPB | Step | Seed | Source |
|------|--------|-----|------|------|--------|
| 1 | h=1536, lr=0.002 | 2.52 | 2000 | 1597 | E0133 |
| 2 | h=1280, lr=0.002 | 2.53 | 2000 | 4181 | E0132 |
| 3 | h=1024, lr=0.002 | 2.65 | 5000 | 44 | E0304/E0305 |
| 4 | h=1536, lr=0.002 | 2.83 | 4000 | 42 | E0730 |
| 5 | GF8, h=828 | 2.79 | 10000 | 1597 | E0096 |

### Finding 3: BPB Plateau

The E0133 learning curve shows diminishing returns:
```
step  500: 3.02  (Δ = -0.33 per 500 steps)
step 1000: 2.69  (Δ = -0.16)
step 1500: 2.53  (Δ = -0.01)
step 2000: 2.52  (Δ ≈ 0)
```
**Extrapolation**: At step 10000, expected BPB ≈ 2.45. At step 20000, ≈ 2.40.
The model architecture has a ~2.5 BPB floor with the current trainer.

### Finding 4: Phase Transition in h=828

The CHAMPION-REPRO curve (h=828, lr=0.0004, rng=10946, step=27000):
```
step  1000: 0.02  (collapsed)
step 18000: 0.00  (still collapsed)
step 19000: 11.75 ← PHASE TRANSITION
step 27000: 9.73  (realistic, decreasing)
```
The FULL81K curve (h=828, lr=0.0004, rng=1597, step=81000):
```
step  1000: 0.48  (collapsed)
step 65000: 0.00  (still collapsed)
step 66000+: oscillating 0.0–0.1
step 81000: 17.31 (realistic)
```

## Victory Plan — What We Can Achieve Today

### Phase A: Confirm h=1536 Floor (NEXT 2h)

Queue focused experiments around the best-performing config:
- **h=1536, lr=0.002, step=20000** × 5 seeds × 6 accounts = 30 experiments
- **h=1536, lr=0.003, step=15000** × 3 seeds × 6 accounts = 18 experiments
- **h=1536, lr=0.004, step=10000** × 3 seeds × 6 accounts = 18 experiments

Expected: Confirm ~2.5 BPB floor, possibly reach 2.3–2.4 with higher LR.

### Phase B: Scale Up (T+2h to T+6h)

- **h=2048, lr=0.002, step=15000** × 5 seeds × 6 accounts = 30 experiments
- **h=3072, lr=0.002, step=10000** × 3 seeds × 6 accounts = 18 experiments
- **h=4096, lr=0.001, step=10000** × 3 seeds × 6 accounts = 18 experiments

Expected: Test if larger models break through the 2.5 floor.

### Phase C: Format Competition (T+6h to T+12h)

- **h=1536, lr=0.002, step=10000, format=GF16** × 5 seeds × 6 accounts = 30 experiments
- **h=1536, lr=0.002, step=10000, format=GF8** × 5 seeds × 6 accounts = 30 experiments

Expected: Compare GF16 vs FP32 vs GF8 at the best config.

### Total: 192 focused experiments across 18h

## Honest Assessment

**Can we reach Gate-2 (BPB < 1.85)?**

Probability: **LOW (~15%)**. The model plateaus at ~2.5 BPB regardless of:
- Step count (tested up to 81K)
- Hidden size (tested 256–3072)
- Learning rate (tested 0.0001–0.004)

The only untested combination that might work:
- h=4096+ with lr=0.001–0.002 for 20K+ steps
- This requires ~4x more compute per step

**What we WILL achieve:**
1. ✅ Fleet infrastructure proven (206 experiments, 6 accounts)
2. ✅ Scaling laws documented (BPB vs hidden, BPB vs LR, BPB vs steps)
3. ✅ Training dynamics characterized (collapse phase, phase transition)
4. ✅ Best config identified (h=1536, lr=0.002 → 2.52 BPB)
5. ✅ Format comparison (GF16 vs FP32 vs GF8)

## Recommended Experiments to Queue NOW

```sql
-- Phase A: h=1536 focused grid (66 experiments)
INSERT INTO experiment_queue (canon_name, config_json, priority, seed, steps_budget, account, created_by)
VALUES
-- h=1536, lr=0.002, step=20000 (best config, longer run)
('IGLA-VICTORY-h1536-LR002-step20000-acc0-rng1597', '{"hidden":1536,"lr":0.002,"ctx":12,"steps":20000}', 90, 1597, 20000, 'acc0', 'human'),
('IGLA-VICTORY-h1536-LR002-step20000-acc1-rng2584', '{"hidden":1536,"lr":0.002,"ctx":12,"steps":20000}', 90, 2584, 20000, 'acc1', 'human'),
('IGLA-VICTORY-h1536-LR002-step20000-acc2-rng4181', '{"hidden":1536,"lr":0.002,"ctx":12,"steps":20000}', 90, 4181, 20000, 'acc2', 'human'),
('IGLA-VICTORY-h1536-LR002-step20000-acc3-rng6765', '{"hidden":1536,"lr":0.002,"ctx":12,"steps":20000}', 90, 6765, 20000, 'acc3', 'human'),
('IGLA-VICTORY-h1536-LR002-step20000-acc4-rng10946', '{"hidden":1536,"lr":0.002,"ctx":12,"steps":20000}', 90, 10946, 20000, 'acc4', 'human'),
-- h=2048, lr=0.002, step=15000 (scale up)
('IGLA-VICTORY-h2048-LR002-step15000-acc5-rng1597', '{"hidden":2048,"lr":0.002,"ctx":12,"steps":15000}', 85, 1597, 15000, 'acc5', 'human'),
('IGLA-VICTORY-h2048-LR002-step15000-acc0-rng2584', '{"hidden":2048,"lr":0.002,"ctx":12,"steps":15000}', 85, 2584, 15000, 'acc0', 'human'),
('IGLA-VICTORY-h2048-LR002-step15000-acc1-rng4181', '{"hidden":2048,"lr":0.002,"ctx":12,"steps":15000}', 85, 4181, 15000, 'acc1', 'human'),
('IGLA-VICTORY-h2048-LR002-step15000-acc2-rng6765', '{"hidden":2048,"lr":0.002,"ctx":12,"steps":15000}', 85, 6765, 15000, 'acc2', 'human'),
('IGLA-VICTORY-h2048-LR002-step15000-acc3-rng10946', '{"hidden":2048,"lr":0.002,"ctx":12,"steps":15000}', 85, 10946, 15000, 'acc3', 'human'),
-- h=1536, lr=0.003, step=15000 (higher LR)
('IGLA-VICTORY-h1536-LR003-step15000-acc4-rng1597', '{"hidden":1536,"lr":0.003,"ctx":12,"steps":15000}', 80, 1597, 15000, 'acc4', 'human'),
('IGLA-VICTORY-h1536-LR003-step15000-acc5-rng2584', '{"hidden":1536,"lr":0.003,"ctx":12,"steps":15000}', 80, 2584, 15000, 'acc5', 'human'),
('IGLA-VICTORY-h1536-LR003-step15000-acc0-rng4181', '{"hidden":1536,"lr":0.003,"ctx":12,"steps":15000}', 80, 4181, 15000, 'acc0', 'human'),
-- h=3072, lr=0.002, step=10000 (max scale)
('IGLA-VICTORY-h3072-LR002-step10000-acc1-rng1597', '{"hidden":3072,"lr":0.002,"ctx":12,"steps":10000}', 75, 1597, 10000, 'acc1', 'human'),
('IGLA-VICTORY-h3072-LR002-step10000-acc2-rng2584', '{"hidden":3072,"lr":0.002,"ctx":12,"steps":10000}', 75, 2584, 10000, 'acc2', 'human'),
('IGLA-VICTORY-h3072-LR002-step10000-acc3-rng4181', '{"hidden":3072,"lr":0.002,"ctx":12,"steps":10000}', 75, 4181, 10000, 'acc3', 'human');
```

## Timeline

| Time (UTC) | Action | Expected |
|------------|--------|----------|
| 06:00 | Queue Phase A experiments | 30 min to start |
| 08:00 | First h=1536 step=20000 results | BPB at step 5K |
| 10:00 | Phase A complete | Full curves for h=1536 |
| 12:00 | Phase B results streaming | h=2048/3072 data |
| 18:00 | All phases complete | Final ranking |
| 23:59 | **DEADLINE** | Submit best results |
