# Experiment 006: Multi-Token Prediction + Softcap15 + Eps1e-10

## Status: COMPLETE (wandb: parameter-golf / exp006_mtp_softcap15_eps10)

## Hypothesis
Stack MTP on exp004's training improvements for richer gradient signal.
**Prediction**: val_bpb < 1.290 at 2000 steps. **WRONG — MTP HURTS.**

## Results

| Step | val_loss | val_bpb | vs baseline |
|------|----------|---------|-------------|
| 500  | 2.86     | 1.6963  | +0.22 WORSE |
| 1000 | 2.51     | 1.4861  | +0.11 WORSE |
| 1500 | 2.25     | 1.3346  | +0.008 WORSE |
| 2000 | 2.20     | 1.3025  | +0.006 WORSE |

**Final post-quant: val_bpb = 1.3059** (baseline: 1.2978 = 0.008 WORSE)

## Analysis
MTP HURTS training at 2K steps. The multi-target loss inflates gradients by ~1.75x without
LR compensation, destabilizing early training (val_bpb 1.70 vs 1.48 at step 500). The model
partially recovers by step 2000 but never catches up.

Root cause: the MTP loss = w0*CE(t+1) + w1*CE(t+2) + w2*CE(t+3) sums multiple CEs without
normalizing by sum(weights). This means the effective learning rate is ~1.75x higher during
MTP phases. modded-nanogpt may handle this via their NorMuon optimizer or different LR schedule.

## Fix for next attempt
- Normalize MTP loss: divide by sum(weights) to keep gradient magnitude constant
- OR reduce LR proportionally during MTP phases
- OR use shorter MTP phase (only first 20% of training)
