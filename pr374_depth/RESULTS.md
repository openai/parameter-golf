# pr374_depth results — 12L/4KV/2.625xMLP + EMA + QAT + warmdown

## Without TTT (seed 1337)
- Post-avg: 1.1429
- Artifact: 15.78MB
- Steps: 7169 at 83.7ms/step

## With TTT (20ep, SAM, freeze=0) on top (seed 1337)
- Sliding BPB: **1.1223** (stride=64)
- Non-sliding: 1.1460
- TTT time: 592.8s (20 epochs)
- TTT loss: 1.9411 → 1.9361

## Notes
- 12L is faster (83.7ms vs 85.6ms) but pre-quant quality is worse than 11L (1.1429 vs 1.1412)
- TTT gave -0.0032 BPB improvement on sliding window
- Extrapolated 11L+TTT would be ~1.1211 (untested)
