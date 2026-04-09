# Practice Record: 12L / 24min / Large-Vocab FlashMuon Branch

## Summary

This is a practice record, not a challenge-targeted submission.

Its purpose is to show how the same main branch improves when we relax the short-budget constraint and allow:

- a deeper model (`12` layers instead of the smaller challenge branch)
- a longer training run (`24` minutes)
- a larger vocabulary branch

The main result is that the branch scales in a healthy way outside the strict challenge budget and reaches roughly `1.10 bpb`, which is meaningfully better than the compressed short-budget versions.

## Why This Record Exists

The challenge runs are heavily constrained by:

- `10` minute wallclock
- `16 MB` final size
- aggressive quantization pressure

That makes it hard to separate:

- real model quality improvements
- from compression and wallclock compromises

This practice record is meant to answer a simpler question:

> If we keep the same general architecture direction, does it continue to improve when we give it more model capacity and more training time?

For this branch, the answer is yes.

## Main Takeaway

The FlashMuon / XSA / RReLU2 branch is not only a challenge-specific compression trick.

With a larger and longer setting:

- the model continues to optimize cleanly
- the same architectural ideas remain useful
- final quality improves to about `1.10 bpb`

So this record should be read as evidence that the branch has real headroom beyond the strict competition setup.

## Configuration

This practice run uses:

- `12` transformer layers
- large vocabulary branch
- Flash Muon optimizer path
- `XSA` on the last `5` layers
- only the final XSA layer gated
- `RReLU2` MLP
- linear phase init for `resid_mix`
- linear-by-depth init for `attn_scale` and `mlp_scale`
- late EMA plus post-train best-choice selection
- mixed-bit `int6_awq + lzma` export

## Quantization

The export path remains mixed-bit AWQ:

- most tensors stay at `int6`
- selected sensitive tensors can be promoted to `int8`
- tensor bit width is stored in `qmeta` so dequantization stays compatible

Default sensitive tensors:

- `tok_emb.weight`
- `lm_head.weight`

This keeps the byte budget focused on the tensors that are most expensive to quantize aggressively.

## Vocabulary and Scaling

This branch also keeps the larger-vocabulary direction.

An important practical observation from earlier sweeps was:

- increasing vocabulary from `1024` to the larger branch cost only about `1 ms` per step in this model family

That made the larger vocabulary a good tradeoff, especially once the model was no longer optimized only for the strict 10-minute track.

## Architecture Notes

The same ideas that helped in the challenge branch still appear useful here:

- late-layer specialization matters
- XSA is more useful near the top of the network than uniformly everywhere
- gating only the final XSA layer remains a good bias
- simple structured initialization works better than more aggressive donor-style initialization

The point of this run is not that every challenge setting should become `12L`, but that the branch itself remains sound when scaled upward.

## Interpretation

This practice result should not be compared only as a submission artifact.

It is better understood as an ablation on scale:

- more layers
- more train time
- same core branch ideas
- better final quality

That makes it a useful reference point when deciding which parts of the branch are genuinely improving the model, versus which parts are only helping under the challenge constraints.

## Final Result

Final metrics are recorded in `submission.json`.

The main headline for this record is:

- this larger and longer practice branch reaches roughly `1.10 bpb`

So even outside the strict challenge setting, the FlashMuon + XSA + RReLU2 + mixed-bit AWQ branch continues to improve rather than saturating immediately.
