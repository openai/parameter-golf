# Experiment 013: Recurrence 8x2 d640

## Status: QUEUED (waiting for instance)

## Hypothesis
Most unique blocks (8) with minimal sharing (2x). Tests if more diversity beats more depth.

## Configuration
- **Architecture**: 8 unique × 2 loops = 16 eff, dim=640
- **Training tricks**: logit_softcap=15, adam_eps=1e-10
- **QAT**: disabled
- **wandb run**: exp013_*
