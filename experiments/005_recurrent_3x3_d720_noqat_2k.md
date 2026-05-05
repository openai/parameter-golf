# Experiment 005: Recurrent 3x3@720 WITHOUT QAT, 2K Steps

## Status: COMPLETE (wandb: parameter-golf / exp005_3x3_d720_noqat_2k)

## Hypothesis
Exp 003 tested 3x3@720 WITH QAT and got val_bpb=1.5097 (worse than baseline 1.4805).
This removes QAT to isolate whether QAT was hurting or weight sharing itself is the issue.

**Prediction**: val_bpb ~1.50. **Result: 1.3426 post-quant — WORSE than baseline 1.2978.**

## Configuration
- **Architecture**: 3 unique blocks x 3 loops = 9 effective layers, dim=720
- **QAT**: DISABLED
- **Other**: logit_softcap=15, adam_eps=1e-10, warmdown=1200

## Results

| Step | val_loss | val_bpb | vs baseline |
|------|----------|---------|-------------|
| 500  | 2.55     | 1.5096  | +0.03 WORSE |
| 1000 | 2.39     | 1.4164  | +0.04 WORSE |
| 1500 | 2.31     | 1.3665  | +0.04 WORSE |
| 2000 | 2.27     | 1.3426  | +0.045 WORSE |

**Final post-quant: val_bpb = 1.3426** (baseline: 1.2978)

## Conclusion
Weight sharing with per-iteration scalars is genuinely worse than unique layers.
QAT was NOT the problem (removing it didn't help). The scalar-only symmetry breaking
is insufficient — confirms need for per-iteration LoRA (DeepMind approach) or just unique layers.
