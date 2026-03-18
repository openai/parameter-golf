# Trap 3. Do Not Spend Compute on Longer Training Context First

## Why It Sounds Clever

Longer context often helps language models, and the challenge allows flexible evaluation.

## Why I Think It Is a Trap Here

This baseline does not look context-limited first. It looks update-limited. The clearest evidence is that:

- the 10-minute run is still improving at the wallclock stop
- the exact same architecture keeps improving over many more steps in the 4-hour run

That means your first move should usually be to buy more useful optimization within the 600-second budget, not to make each step more expensive by increasing training context.

Longer evaluation context is a different question and may be worth trying. But longer training context as the first lever is a weak bet for this specific script.

## Recommendation

Probably not worth trying early
