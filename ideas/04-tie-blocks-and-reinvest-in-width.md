# 4. Tie Blocks and Reinvest in Width

## Core Thesis

Recurrence or block sharing is the cleanest way to buy more effective capacity per byte in this exact challenge.

## What Bottleneck It Attacks

This attacks the parameter budget. The current model instantiates all transformer blocks separately:

- blocks are created in a `ModuleList`
- first half stores skips, second half reuses skips in reverse order

Relevant code:

- `self.blocks = nn.ModuleList(...)`: `train_gpt.py:674`
- encoder loop: `train_gpt.py:707-709`
- decoder loop: `train_gpt.py:710-713`

## Why It Should Improve `val_bpb`

The artifact budget is extremely tight. The baseline already sits near the cap, so naive expansion is basically unavailable. Shared blocks let you keep roughly the same runtime and unrolled depth while freeing many parameters. Those freed bytes can be reinvested into width, improved embeddings, or other components that may scale better under compression.

This idea also matches the challenge’s intended direction: better compute-per-byte trades rather than only bigger dense models.

## Expected Effect

- Training speed: similar, possibly slightly better depending on implementation details
- Evaluation speed: similar
- Compressed artifact size: much smaller unless you reinvest the savings

## Difficulty

4/5

## Rule-Risk

2/5

## Smallest Decisive Experiment

Start with a mild tying scheme:

- tie every other block
- keep per-position scale vectors or skip vectors untied
- widen `MODEL_DIM` until total artifact returns near the current `15.8` to `15.9` MB zone

Compare exact roundtrip `val_bpb` at fixed 600 seconds.

## Recommendation Bucket

Serious record attempt
