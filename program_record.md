# Parameter Golf Research Program — Record Track

## Objective
Minimize post-quantization `val_bpb` under the actual record-track constraints:
- 16MB total artifact limit
- 10 minutes training on 8xH100
- competitive evaluation budget

## Primary Principle
For the record track, speed is part of the model.

Do not optimize for best quality-per-step if it materially reduces total steps in 600 seconds.
Optimize for best `val_bpb` per second under the wallclock cap.

## What We Know
- The 12x448 family looked good on 1xH100 and in 1x600 promotion, but on 8xH100 it was too slow and over budget.
- The Modal baseline control shows Modal is somewhat slower than the official reference, but not enough to explain our miss by itself.
- The public frontier now heavily rewards evaluation tricks, especially sliding-window evaluation.
- Recent competitive PRs suggest a stronger record recipe than our current search focus:
  - `train_seq_len=2048`
  - larger batch / more optimizer steps under the wallclock cap
  - `grad_clip_norm=0.3`
  - `muon_momentum=0.99`
  - long warmdown on the true 10-minute recipe
  - sliding-window eval with `stride=256` is a serious contender, not just `64`

## Priority Order
1. Fast record-track training recipe changes that plausibly increase loss-per-second
2. Fast core shapes near baseline throughput
3. Schedule/optimizer improvements with zero throughput cost
4. Export/storage improvements that stay inside the cap
5. Eval-time improvements that fit evaluation budget

## Preferred Directions
- `train_seq_len=2048` style training for the true record-track objective
- tighter grad clipping like `0.3`
- higher Muon momentum like `0.99`
- Fast depth/width tradeoffs near the baseline family
- Warmup/warmdown tuning
- Muon/optimizer refinements with zero step-time cost
- KV-head or attention-layout efficiency changes
- Export-path improvements like FP16 tied embeddings if they help under the byte cap

## Avoid
- Slow deep models that cut step count heavily
- Multi-change speculative rewrites
- Naive width/MLP increases
- Generic activation swaps without a clear speed or export advantage

## Guidance
- Make exactly one conceptual change per experiment
- Favor changes that preserve throughput
- Treat evaluation as part of the system, not an afterthought
- Prefer ideas that are easy to promote into a record-track PR in the next few hours over speculative architecture novelty
