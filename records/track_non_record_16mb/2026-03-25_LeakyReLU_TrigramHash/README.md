# Non-record: LeakyReLU(0.5)^2 + TrigramHash on PR 414 stack

val_bpb: 1.3762 | 1xA100, 7000 iterations

## Built on
PR 414 by signalrush (11L EMA + GPTQ-lite + warmdown3500 + QAT@0.15)

## Changes

### 1. LeakyReLU(0.5)^2
Replaced relu^2 with LeakyReLU(0.5)^2 in MLP.
Motivation: keeps neurons alive during training instead of permanently killing them.

### 2. TrigramHash
Groups 3 consecutive tokens into 8192 buckets before attention.
Gives each layer richer local context.

## Results
Hardware: 1xA100
Iterations: 7000
val_bpb: 1.3762

Note: not comparable to 8xH100 leaderboard submissions.

## Next steps
- Depth-based precision per layer
- Novel layer interaction mechanisms
- Full 8xH100 run once validated
