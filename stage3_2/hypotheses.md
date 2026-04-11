# Stage 3.2 Hypotheses

These are not static patches.
They are dynamic policy families over the bounded controller DSL.

## H201 Late Deploy Gate

- Mechanism:
  - keep normal training early
  - activate deploy-alignment controls only after the quant-gap proxy crosses a threshold or late phase begins
- Why:
  - many deploy-alignment ideas are helpful late and harmful early
- Dominant lane:
  - deploy / quant damage
- Expected impact:
  - medium to large
- Earliest signal:
  - `180s`, possibly only `600s`
- Failure mode:
  - quant-gap proxy is too noisy, controller flips at the wrong time

## H202 Best-State Controller

- Mechanism:
  - adapt checkpoint capture rate and selection mode by late-run state
- Why:
  - the best deployed checkpoint is unlikely to always be the last one
- Dominant lane:
  - deploy / state selection
- Expected impact:
  - medium
- Earliest signal:
  - `600s`
- Failure mode:
  - extra state tracking is not worth the complexity or the best state still coincides with the last step

## H203 Curriculum-by-State

- Mechanism:
  - change shard order based on progress and learning dynamics
- Why:
  - one global data order is likely false under a short wallclock cap
- Dominant lane:
  - train quality
- Expected impact:
  - medium
- Earliest signal:
  - `180s`
- Failure mode:
  - state signal is too weak and static ordering is enough

## H204 Family-Split Warmdown

- Mechanism:
  - embeddings/head/scalars and trunk matrices get different late multipliers or freeze rules
- Why:
  - different tensor families likely need different late adaptation laws
- Dominant lane:
  - train quality + deploy damage
- Expected impact:
  - medium
- Earliest signal:
  - `600s`
- Failure mode:
  - selective freezing hurts raw fit more than it helps deployed robustness

## H205 Alternating Objective Controller

- Mechanism:
  - sparse microcycles of export-surrogate or late-QAT pressure gated by state
- Why:
  - blended objectives poison the whole trajectory; pulses may internalize deploy loss more cheaply
- Dominant lane:
  - deploy / quant damage
- Expected impact:
  - medium to large
- Earliest signal:
  - `180s` to `600s`
- Failure mode:
  - sparse pulses are too weak, or compute overhead destroys step count

## H206 Systems-Aware Controller

- Mechanism:
  - disable or weaken expensive controls when `step_avg_ms` rises too far
- Why:
  - many good mechanisms lose because they cost too much wallclock
- Dominant lane:
  - throughput / more steps
- Expected impact:
  - small to medium
- Earliest signal:
  - `90s`
- Failure mode:
  - system-aware throttling removes the very mechanism that would have paid off later

## H207 Best-State + Narrow Pre-Quant TTT

- Mechanism:
  - choose the best late deployed state
  - then run pre-quant TTT only on the last 2 blocks
- Why:
  - the best TTT starting point likely matters, and narrow last-block adaptation is one real frontier recipe
- Dominant lane:
  - deploy / TTT
- Expected impact:
  - medium
- Earliest signal:
  - `600s`
- Failure mode:
  - TTT mostly washes away checkpoint choice or last-2-only adaptation is too weak

## H208 Best-State + Broader dTTT Tail

- Mechanism:
  - choose the best late deployed state
  - then run a broader TTT finisher with block-wise LR decay over most of the upper trunk
- Why:
  - the frontier is no longer just “do TTT”; it is “which TTT law wins”
- Dominant lane:
  - deploy / TTT
- Expected impact:
  - medium to large
- Earliest signal:
  - `600s`
- Failure mode:
  - broader adaptation overfits, is too expensive, or does not beat the simpler narrow finisher

## H209 Context Budget Controller

- Mechanism:
  - switch between cheaper and richer context modes by training state
- Why:
  - long-context benefits may not pay uniformly throughout training
- Dominant lane:
  - train quality + throughput
- Expected impact:
  - medium
- Earliest signal:
  - `180s`
- Failure mode:
  - switching context modes introduces instability or wastes specialization

## H210 Composite Late Policy

- Mechanism:
  - jointly control:
    - late deploy alignment
    - checkpoint selection
    - family-specific late freezing
- Why:
  - the final score is decided by late trajectory shape more than early helper gains
- Dominant lane:
  - deploy / state / late consolidation
- Expected impact:
  - large
- Earliest signal:
  - only long run
- Failure mode:
  - too much interaction complexity for the bounded controller to optimize cleanly

## Lead Families

The strongest first-wave families are:

1. `H201 Late Deploy Gate`
2. `H202 Best-State Controller`
3. `H205 Alternating Objective Controller`
4. `H204 Family-Split Warmdown`
5. `H207 Best-State + Narrow Pre-Quant TTT`
6. `H208 Best-State + Broader dTTT Tail`

These should be the center of `stage3_2`.
