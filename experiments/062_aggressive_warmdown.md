# Experiment 062: PR135 + NorMuon + QAT + WD=6000

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,328 @ 81.9ms/step |
| Standard eval | 1.1706 BPP |
| Sliding eval | 1.1497 BPP |

## Comparison with 060 (WD=3000)
060 standard: 1.1683, sliding: **1.1474** — WD=3000 is better.
Aggressive warmdown (WD=6000) hurt by 0.002 BPB.

## CONCLUSION: WD=3000 is optimal. Don't increase.
