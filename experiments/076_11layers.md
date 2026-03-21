# Experiment 076: 11 Layers + all features — BEST BPB EVER but over budget

## Results
| Metric | Value |
|--------|-------|
| Sliding eval | **1.1304 BPP** ← BEST EVER by huge margin |
| Standard eval | 1.1516 |
| FLAT+zstd | **18,954,093 ❌ (2.95MB over)** |
| Params | 27,092,057 |
| Steps | 6,073 @ 98.9ms/step |

## Finding
11 layers gives MASSIVE BPB improvement (+0.014 vs 9 layers).
But with BigramHash + SmearGate + 27.1M params, artifact way over budget.
Need to strip features or reduce MLP to fit.
PR179 fits 11 layers at 15.9MB by NOT using BigramHash/SmearGate (~24.5M params).
