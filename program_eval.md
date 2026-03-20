# Parameter Golf Research Program — Eval-Time Lane

## Objective
Reduce post-quantization `val_bpb` using evaluation-time logic while staying within practical evaluation latency and memory limits.

## Primary Principle
Use evaluation to recover information the model already contains or to score tokens with richer context.

## What We Know
- Post-quantization calibration already produced a large win in this lane.
- The public leaderboard now validates sliding-window evaluation as a top-tier direction.

## Priority Order
1. Sliding-window style evaluation
2. Post-quantization calibration
3. Calibration that is orthogonal to existing temperature/softcap tuning
4. Efficient context-rich scoring

## Preferred Directions
- Sliding-window evaluation with careful stride/batch tuning
- Post-quant logit softcap calibration
- Skip-weight or residual-mix calibration
- Final-head or per-head calibration if cheap enough

## Avoid
- Core-model architecture changes
- Tokenizer or dataset changes
- Expensive eval-time logic that blows the eval budget
- Overlapping calibration tricks that likely do the same thing

## Guidance
- Make one conceptual change at a time
- Keep evaluation time measurable and visible
- Prioritize gains that transfer to strong fast core models
