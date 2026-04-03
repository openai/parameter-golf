# Junkyard_Rat Triton Track

Date: 2026-03-29

## Mission

Turn `JR-02` from a one-off kernel experiment into a controlled Triton optimization track for the current `JR-01` winner.

This subfolder exists to separate:
- live base-lane winners
- kernel engineering work
- numerics compensation work after kernel changes

## Why This Track Exists

`JR-01` already proved the loader idea is real.

The Triton question is different:
- not "is Triton cool"
- but "can a custom kernel on our exact MLP path improve step time or quality on the winning stack"

This architecture is not exotic, but it is tightly tuned:
- banked FP32 weights
- BF16 runtime math
- `linear -> leaky_relu(0.5) -> square -> linear`
- layerwise residual scaling and mixing

That means a kernel can change:
- dataflow
- launch behavior
- math ordering
- effective numerics in the MLP branch

So Triton needs its own track.

## Current State

### Winner under test

`JR-01`:
- coprime loader
- ~`91.00ms` step time
- `1.11056240` sliding BPB

### Active Triton candidate

`JR-02`:
- `MLP_KERNEL_MODE=triton_act`
- custom Triton activation kernel in the real MLP branch
- same loader winner underneath

### First result: `TR-01` loses, but the surface is live

Measured on the winner stack:

| Variant | Step avg | Post-EMA BPB | Sliding BPB | Decision |
|---|---:|---:|---:|---|
| `JR-01` eager MLP | `91.00ms` | `1.1340` | `1.11056240` | active winner |
| `TR-01` `triton_act` | `91.11ms` | `1.1345` | `1.11099954` | loser |

Interpretation:
- no meaningful speed gain
- slight BPB regression
- but the kernel path is stable and close enough to justify tuning work rather than deleting the track

### Second result: `TR-02` delta sweep found one live compensation knob

We ran a six-way delta sweep on top of `triton_act`.

The original intent was a short `170s` pop-test ladder, but due to a runner override bug the box actually executed full `600s` capped runs. That was expensive, but it also produced stronger signal than a short screen would have.

Cap-time validation ranking from that sweep:

| Variant | Delta | Step avg | Cap-time val BPB | Decision |
|---|---|---:|---:|---|
| `TR-02d` | `attn_scale=1.02` | `91.16ms` | `1.1347` | current Triton leader |
| `TR-02a` | base `triton_act` | `91.09ms` | `1.1349` | baseline |
| `TR-02b` | `mlp_scale=0.98` | `91.12ms` | `1.1351` | loser |
| `TR-02c` | `mlp_scale=1.02` | `91.10ms` | `1.1352` | loser |
| `TR-02e` | `attn_scale=0.98` | `91.13ms` | `1.1354` | loser |
| `TR-02f` | `resid_mix=(0.98,0.02)` | `91.10ms` | `1.1356` | loser |

Interpretation:
- `attn_scale=1.02` is the only completed compensation delta that helped
- the gain is small, but it is directional and consistent enough to warrant a full confirmation run with final eval
- the tested `mlp_scale` and `resid_mix` nudges did not help this kernel path

## Core Hypotheses

### H1: activation-kernel path is a live optimization surface

The first real Triton kernel path is stable and not catastrophically slower.

That means the door is open for:
- block-size tuning
- launch tuning
- broader fusion

### H2: numerics compensation may matter as much as raw speed

Because this stack is tuned, a kernel can shift quality even if wallclock is flat.

Likely compensation surfaces:
- `attn_scale`
- `mlp_scale`
- `resid_mix`
- layerwise norm scaling (`ln_scale_factor`)

Current priority after the first sweep:
1. `attn_scale`
2. layerwise norm scaling (`ln_scale_factor`)
3. deeper or sign-flipped `resid_mix`
4. only then revisit `mlp_scale`

### H3: full mega-fusion is the real upside, but not the first safe step

The current `triton_act` path is a foothold.

If it stays alive, the next serious kernel target is:
- `linear -> leaky_relu -> square -> linear`

implemented around our banked weight layout.

## Track Policy

- Keep `JR-01` as the untouched winner until a Triton candidate clearly beats it.
- Use this folder for Triton-specific runners, notes, and ablation ordering.
- Move losing Triton variants into `triton/losers/` once decided.
- Keep full-run delta results as tracked markdown files in this folder so we do not lose signal when short-screen infrastructure changes.
