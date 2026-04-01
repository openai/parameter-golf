# Approach H: Focal Loss

## Summary

Builds on Approach B (Int5 GPTQ + 33.6M params + SWA + XSA + VE) by replacing
standard cross-entropy with focal loss during training.

Focal loss: `loss = (1 - p_correct)^gamma * CE_loss` where gamma=2.0 (configurable
via `FOCAL_GAMMA` env var). This down-weights easy tokens the model already predicts
well and focuses gradient signal on hard tokens.

Inspired by PR #1180 which achieved 1.0577 BPB using P2 loss `(1-p)^2` among other
techniques (residual mixing, conv token mixer, wallclock-aware warmdown).

## Key changes from Approach B

1. **Focal loss** (training only): replaces `F.cross_entropy(..., reduction="mean")`
   with `((1 - exp(-ce))^gamma * ce).mean()` in the model's `forward()` method.
2. `FOCAL_GAMMA` env var (default 2.0, set to 0.0 for standard CE).

No eval changes. No architecture changes. No artifact size impact.

## Configuration

```bash
FOCAL_GAMMA=2.0  # default; 0.0 = standard CE
```

All other hyperparameters unchanged from Approach B defaults.

## Expected outcome

Focal loss should improve BPB by focusing training on hard tokens. The
down-weighting factor `(1-p)^2` is strongest early in training when many tokens
are hard, and naturally relaxes as the model improves.
