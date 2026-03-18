# Trap 1. Do Not Untie Embeddings or Grow Vocab First

## Why It Sounds Clever

Untied embeddings or a larger vocabulary are normal levers in language model tuning, and they can improve loss in unconstrained settings.

## Why I Think It Is a Trap Here

This challenge is artifact-limited first. The baseline already uses tied embeddings and is only about `136 KB` under the byte cap. That means a move that directly increases embedding-side parameters is under immediate pressure from the artifact budget.

Also, the current bottleneck does not look like “the vocab is too small for the model to learn anything useful.” It looks much more like:

- not enough effective optimization steps in 10 minutes
- too much performance lost at quantization/export time
- likely suboptimal parameter allocation inside the block stack

So growing vocab or untying embeddings is a bad first bet. It spends bytes in one of the most expensive places before solving the higher-confidence bottlenecks.

## Recommendation

Probably not worth trying early
