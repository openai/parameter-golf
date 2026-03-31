# bandit_wagon_choke_battery (BWCB) — Battery on Pyramid

## Background

BWCS established pyramid-512 as the dominant choke shape (-0.01037 vs control, quant_gap
collapsed to -0.0001). The mega ablation (BWB series, flat/no-choke) shows battery alone
has signal but with unexpected scale ordering: 1,2,4 beats 1,3,9 in isolation.

This series answers: **does battery stack with pyramid-512, and which scale wins in combo?**

## References (from prior runs)

| Run | Config | INT6_SW_BPB | Quant Gap |
|-----|--------|-------------|-----------|
| BWCS-00 | flat ctrl (1 shard) | 1.45761 | +0.0013 |
| BWCS-02 | pyramid-512 (1 shard) | 1.44724 | -0.0001 |
| BWC-04 | flat choke=512 (80 shards) | 1.42887 | -0.0009 |
| BWB-01 | battery 1,2,4 flat (80 shards) | 1.43769 | -0.0010 |
| BWB-02 | battery 1,3,9 flat (80 shards) | 1.44470 | +0.0028 |

## Arms

| ID | Shape | Rope Scales | Purpose |
|----|-------|-------------|---------|
| BWCB-00 | pyramid-512 | 1,2,4 | Gentle combo — BWB-01 scale winner on pyramid |
| BWCB-01 | pyramid-512 | 1,3,9 | Core hypothesis combo |
| BWCB-02 | pyramid-512 | 1,5,25 | Aggressive combo |

References from BWCS-00 and BWCS-02 — no control repin needed.

## Results

### Run A — 1 shard (seed=444, 500 steps)

| ID | Scales | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCS-02 |
|----|--------|---------|-------------|-----------|------------|
| BWCS-02 ref | — | 1.4473 | 1.44724 | -0.0001 | — |
| BWCB-00 | 1,2,4 | 1.4473 | 1.44850 | +0.0012 | +0.00126 |
| BWCB-01 | 1,3,9 | 1.4492 | 1.45016 | +0.0010 | +0.00292 |
| BWCB-02 | 1,5,25 | 1.4525 | 1.45534 | +0.0028 | +0.00810 |

Run A conclusion (later revised): ascending battery appears to hurt pyramid.

### Run B — 4 shards (seed=444, 500 steps) ← authoritative

| ID | Scales | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCS-02 |
|----|--------|---------|-------------|-----------|------------|
| BWCS-02 ref | — | 1.4473 | **1.44724** | -0.0001 | — |
| **BWCB-00** | **1,2,4** | **1.4442** | **1.44515** | **+0.0009** | **-0.00210** |
| BWCB-01 | 1,3,9 | 1.4473 | 1.44874 | +0.0014 | +0.00149 |
| BWCB-02 | 1,5,25 | 1.4476 | 1.44864 | +0.0010 | +0.00139 |

## Verdict: 1,2,4 Beats Pyramid — Training Diversity Required

**BWCB-00 (pyramid-512 + 1,2,4) beats pyramid alone by -0.00210 in Run B.**

Run A's "ascending battery hurts pyramid" was a shard-count artifact. With 1 shard, training
data is too narrow for multi-scale attention to specialize — all three loops see the same
patterns at every scale so the battery adds noise. With 4 shards, enough diversity exists
for each loop to find different signal at its causal horizon.

**Why 1,2,4 wins and wider scales don't:**
- 1,2,4 (4× spread): distributions stay close enough for int8 to cover; diversity benefit wins
- 1,3,9 (9× spread): distribution divergence partially offsets the diversity benefit; break-even
- 1,5,25 (25× spread): similar to 1,3,9 in Run B; wider is not better beyond the sweet spot

**Caveat:** BWCS-02 reference is 1-shard. Pyramid-512 at 4 shards might also improve.
Need 4-shard pyramid control to confirm net gain. But the relative advantage of 1,2,4 over
wider battery configs is consistent across both runs.

## Follow-On: BWCD (Descending)

BWCD tests 9,3,1 | 4,2,1 | 9,1,1 | 9,3,9 on pyramid-512. Key question: does descending
(wide→narrow, distribution-converging) do better or worse than 1,2,4 + pyramid?
