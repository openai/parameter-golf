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

| ID | Scales | Raw BPB | INT6_SW_BPB | Quant Gap | vs BWCS-00 | vs BWCS-02 |
|----|--------|---------|-------------|-----------|------------|------------|
| BWCB-00 | 1,2,4 | TBD | TBD | TBD | TBD | TBD |
| BWCB-01 | 1,3,9 | TBD | TBD | TBD | TBD | TBD |
| BWCB-02 | 1,5,25 | TBD | TBD | TBD | TBD | TBD |

## Key Question

Does any combo beat pyramid-512 alone (1.44724)? If yes: battery and pyramid interact
constructively. If no: they're solving different parts of the same problem and don't stack.

Watch quant_gap — should stay near zero or go negative (pyramid alone got -0.0001).
If battery forces it positive, battery is adding quantization stress that pyramid isn't absorbing.
