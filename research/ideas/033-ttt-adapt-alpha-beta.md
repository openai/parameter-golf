# Idea 033 — Let TTT adapt frozen alpha/beta

## Thesis

`028` already showed that TTT hyperparameters matter on top of the same frozen `026 seed_42` float checkpoint:

- old TTT: `1.06724`
- new TTT: `1.06649`

But both runs kept the recurrence carry parameters frozen.

The open question is:

- are the training-calibrated frozen `alpha/beta` values still optimal once TTT starts adapting the model?

The clean test is a third `028`-style run:

- same float checkpoint as `028`
- same newer TTT settings as `028B`
- but allow TTT to update `alpha/beta` too

## Key design choice

Do **not** reinitialize `alpha/beta`.

Start from the frozen checkpoint values and let TTT make small adjustments.

That isolates exactly the question we care about:

- does TTT want to move already-good frozen carry values?

## Optimizer stance

Alpha/beta should not share the full LoRA LR.

Use:

- same TTT config as `028B`
- lower LR for `alpha/beta`, likely `0.25x` of `TTT_LORA_LR`

Reason:

- alpha/beta are tiny global structural parameters
- we want gentle adaptation, not to overwrite them in a handful of batches

## Expected outcomes

- if `028C` beats `028B`, then frozen training-time carry is not fully TTT-optimal
- if it ties, then freezing is probably already good enough under TTT
- if it regresses, TTT should leave alpha/beta alone
