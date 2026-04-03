# Evolutionary Benchmark Results

This folder contains a lot of queued experiments, but the high-level story is already fairly clear.

## What Failed Cleanly

- Direct crossover in weight or delta space was not a good primitive in the early probes.
  - Short viability runs routinely collapsed.
  - Mutation-only parent-copy loops were effectively flat at the tested operating point.
- Wider earlier was not automatically better.
  - Schedules that spent too much budget on width too early underperformed.
- Funnel schedules are plausible, but they did not beat the best expand-only schedule in the checked-in runs.

Those failures are worth keeping because they explain why the later experiments pivot toward committees, adaptive schedules, recipe genes, and compressed deltas.

## What Worked

- Chunked `vmap` evaluation on a single H100 scaled well enough to make large population search realistic.
  - The checked-in throughput runs reached about `882 models/s` at population `16384` with chunking.
- Committee behavior was the first consistently strong positive signal.
  - Independent branches from the same base, trained on different sampled trajectories, produced useful diversity.
  - Top-8 ensembles improved validation BPB by roughly `0.14-0.15` over the best single branch in the first successful committee probes.
- The best fixed-budget schedule in the checked-in runs is still:
  - `2x120 -> 4x30 -> 8x15`
  - seed `1337`: `val_bpb ~= 2.0305`
  - seed `2025`: `val_bpb ~= 2.1352`

That result is important because it says:

- search narrowly and deeply first
- widen late once the branches have found a good basin
- raw breadth is less useful than staged breadth

## What Is Still Open

- Adaptive widen/narrow control driven by replay disagreement, archive hits, ensemble gain, and pairwise branch distance
- Real tokenizer comparisons beyond byte-level `256`
- Recipe-gene evolution where the genome is the training/config recipe rather than the trained weights
- Compressed committee artifacts, where the submission is `base + shared branch deltas`

## Cleanup Convention

The structured `runs/*.json` files are part of the experimental record and are intentionally kept. Raw launcher logs are treated as transient and ignored.
