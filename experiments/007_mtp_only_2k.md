# Experiment 007: Multi-Token Prediction ONLY (no softcap/eps changes)

## Status: COMPLETE (wandb: parameter-golf / exp007_mtp_only_2k)

## Hypothesis
Ablation: test MTP alone without softcap=15 or eps=1e-10.
**Prediction**: val_bpb < 1.295. **Result: 1.3016 post-quant — 0.004 WORSE than baseline.**

## Configuration
- **Architecture**: SAME as baseline (9 blocks, dim=512)
- **Changes**: MTP_ENABLED=1 ONLY (softcap=30, eps=1e-8 — baseline defaults)
- **Script**: train_gpt.py (unnormalized MTP loss — same bug as exp006)

## Results

| Step | val_loss | val_bpb | vs baseline |
|------|----------|---------|-------------|
| 1000 | 2.51     | 1.4852  | +0.11 WORSE |
| 1500 | 2.25     | 1.3325  | +0.006 WORSE |
| 2000 | 2.20     | 1.3002  | +0.004 WORSE |

**Final post-quant: val_bpb = 1.3016** (baseline: 1.2978)

## Conclusion
MTP with unnormalized loss hurts. Same gradient inflation issue as exp006.
