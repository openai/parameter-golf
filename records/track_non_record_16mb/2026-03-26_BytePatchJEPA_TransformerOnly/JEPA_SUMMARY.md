# JEPA Attempt Summary

This is the short historical note for the pure-JEPA experiments after cleaning out the old branches. The primary writeup for this folder is in [README.md](README.md).

Repo simple baseline for reference: `1.22436570 val_bpb`.

## Top-Line Result

We did not get pure JEPA close to the repo baseline. The best clean detached-probe result we saw was:

- `2.3839 bpb` with `transformer_rope_gqa_localglobal + slot_ema_teacher`

That was a large improvement over the earlier pure-JEPA runs, but it was still about `+1.16 bpb` above the simple baseline.

## What Counted As "Pure JEPA" Here

- raw `byte260` inputs only
- no tokenizer
- no exact byte-NLL into the backbone
- backbone trained only with JEPA-style latent prediction plus anti-collapse regularization
- exact byte probabilities produced later by a detached Transformer decoder probe on frozen features

So this was a strict test of whether JEPA latents alone could carry enough information for good byte compression.

## Historical Progression

### 2026-03-24 `BytePatchJEPA_PurityFirst`

- Raw-byte JEPA backbone with a coupled exact decoder term
- Best full run reached about `2.8583 bpb`
- Negative: more compute helped, but the coupled byte-loss path was not pure enough and still far from baseline

### 2026-03-25 `BytePatchJEPA_TiedTransformer`

- Early tied-Transformer JEPA retry
- Effectively stalled near uniform-entropy behavior
- Negative: bad Transformer recipe, not a meaningful positive signal

### 2026-03-25 `BytePatchJEPA_DeepGRU`

- Larger recurrent control
- Trained, but stayed weak
- Negative: more GRU was not the answer

### 2026-03-25 `BytePatchJEPA_UncappedValChase`

- Uncapped validation-only chase
- Improved over the earliest pure runs but still did not suggest an easy path to baseline

### 2026-03-26 `BytePatchJEPA_PureProbeScaling`

- First clean frozen-probe pipeline
- Best result was GRU-based at about `3.0774 bpb`
- Data scaling helped, but the first multi-horizon and multi-scale variants hurt
- Negative: detached probing was the right protocol, but the target and early Transformer recipe were still wrong

## Transformer-Only Campaign

This folder kept only the parts that still looked worth pushing:

- Transformer backbones only
- slot-based targets instead of pooled patch regression
- detached Transformer strong probe only
- stronger repo-style Transformer ingredients: RoPE, GQA, RMSNorm, SwiGLU, residual branch scaling, Muon/AdamW split

### Backbone Screen

At the anchor size, with `slot_l2` fixed:

- `transformer_rope_gqa_localglobal`: `2.3889800525604903 bpb`
- `transformer_rope_gqa_base`: `2.389990501438125 bpb`
- `transformer_rope_gqa_convstem`: `2.5803010001832605 bpb`

Takeaway:

- `localglobal` narrowly beat `base`
- `convstem` was a real regression

### Objective Screen

With `transformer_rope_gqa_localglobal` fixed, objective ranking was:

- `slot_ema_teacher`: `2.3839 bpb`
- `slot_cosine`: `2.3885 bpb`
- `slot_l2`: `2.3888 bpb`
- `slot_vicreg`: `2.3918 bpb`
- `masked_slot_jepa`: `2.5098 bpb`

These numbers were recovered from the copied-back live logs because the final `objective_screen/summary.json` was not synced back.

Takeaway:

- `slot_ema_teacher` was the best objective in this family
- objective changes only moved the number by a few thousandths to a few hundredths, except for `masked_slot_jepa`, which was clearly worse
- the main bottleneck did not look like "pick a better JEPA loss" anymore

### Encoder Screen

With `transformer_rope_gqa_localglobal + slot_ema_teacher` fixed and a short equal-budget rerun:

- `conv_patch`: `2.746384624395377 bpb`
- `mlp_baseline`: `2.7525905146099565 bpb`
- `patch_transformer`: `2.8835849452702482 bpb`
- `latent_queries`: `2.899715507869489 bpb`

Takeaway:

- `conv_patch` was the only encoder that slightly beat the baseline MLP, and only by about `0.0062 bpb`
- `patch_transformer` and `latent_queries` were clearly worse and slower
- richer within-patch encoders did not solve the core problem

## Main Negatives

- Pure JEPA remained far above the simple baseline even after moving to the stronger Transformer-only setup.
- Lower JEPA loss did not reliably translate into lower exact byte `bpb`.
- Richer patch encoders were mostly negative.
- The detached exact decoder probe learned fine, but the frozen JEPA features still looked too lossy for byte compression.
- The biggest remaining weakness is probably not raw backbone capacity; it is the latent/interface design, especially how much exact local detail survives into the temporal state.

## Current Best Hypothesis

If pure JEPA is going to work better here, the next gains probably come from changing the latent family and the way the backbone consumes it, not from adding more GRU or just making the patch encoder fancier.

The most plausible next directions are:

- let the backbone consume slot tokens directly instead of mostly reasoning over patch summaries
- redesign the latent target family to preserve more local detail
- keep using a detached exact decoder probe so the experiment stays honest
