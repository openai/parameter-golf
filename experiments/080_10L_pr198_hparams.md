# Experiment 080: 10L + PR198 hparams — still over budget

## Results
- Sliding: 1.1412 (slightly worse than 079's 1.1398 — higher LR needs more steps)
- FLAT+zstd: 16.89MB ❌ (890KB over — higher WD saved only 40KB)
- Params: 24,140,880, Steps: 6,687 @ 89.8ms/step

## Conclusion
10 layers doesn't fit on our platform regardless of hyperparams.
The 890KB gap can't be closed by WD alone — it's a fundamental platform compression difference.
**DECISION: Stick with 9 layers. Apply PR198 hparams (WD=0.04+LR=0.025) to 9-layer config.**
