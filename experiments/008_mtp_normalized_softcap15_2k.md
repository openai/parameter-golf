# Experiment 008: Normalized MTP + Softcap15 + Eps1e-10

## Status: RUNNING on Instance 0 (wandb: parameter-golf / exp008_mtp_normalized_softcap15)

## Hypothesis
Exp006 showed MTP hurts because the loss sums multiple CEs without normalization, inflating
gradients ~1.75x. Fix: divide loss by sum(weights) to keep gradient magnitude constant.

**Prediction**: val_bpb < 1.295 at 2000 steps if normalization fixes the instability.

## Key Change from Exp006
```python
return loss / mtp_weights.sum()  # normalize to keep gradient magnitude constant
```
