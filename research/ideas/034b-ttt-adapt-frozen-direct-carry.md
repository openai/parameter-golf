# Idea 034b — TTT-only adaptation of frozen direct-carry from `034`

## Thesis

`034` should first answer the clean deployment question:

- freeze the `031A` direct-carry tensors
- run the normal live-like 4×H proxy
- save the float checkpoint

Then `034b` should mirror the `033` pattern:

- take the saved `034` float checkpoint
- rerun only the post-training quantized + phased-TTT path
- allow the frozen direct-carry tensors to become trainable during TTT

So this is not a retrain.
It is a TTT-only follow-up from the `034` checkpoint.

## Why this is the right shape

If we retrained again, we would mix two questions:

- was the frozen direct-carry base good?
- did TTT adaptation help?

The `033` line already showed the value of keeping those separate.

So `034b` should be:

- same base checkpoint as `034`
- same quantization / hotstart path
- only change is that TTT may move the direct-carry tensors

## Hypothesis

The older `033` / `033b` line suggested that TTT adaptation of the old shared
alpha/beta basis was either negligible or harmful.

But the frozen `034` direct-carry object is richer:

- pass-specific
- direct edges
- explicit self weights

So it is plausible that this basis is a better target for TTT correction than the
older alpha/beta basis was.

## Mechanism

Start from the frozen `034` checkpoint values.

During TTT only:

- make the frozen direct-carry tensors trainable
- include them in a separate TTT optimizer param group
- log before/live/after snapshots and max drift

The natural trainable set is:

- `direct_carry_self_frozen`
- `direct_carry_edges_frozen_pass1`
- `direct_carry_edges_frozen_pass2`

## Recommendation

This should run immediately after `034` if:

- `034` finishes cleanly
- the float checkpoint exists
- the base result is at least healthy enough to be worth refining

And it should be judged directly against the post-TTT result of `034`, not as a
new standalone training run.
