# Experiment 0062_swiglu_recur_3x3_mlp8

Parent: 0057_swiglu_recur_3x3 (current winner, val_bpb 2.10275 / 2.10427 mean)

## Question
Does pushing SwiGLU MLP_MULT to 8 (vs 0057's 3) inside K=3 L=3 recurrence pay? The 10 MB freed cap from depth recurrence has been mostly unused; mlp=8 spends ~7 MB of it on per-block hidden width. Records cap at mlp=3-4 but they have many more distinct layers. With recurrence (K=3 unique blocks), wider per-block MLP may compensate.

## Hypothesis [CONJECTURE]
At very wide MLP (mlp=8 → hidden=4096) the SwiGLU gating has more interaction capacity per token. With K=3 L=3 recurrence, this wide MLP is invoked 3× per token → effectively 3 wide gating decisions. If recurrence + width compose well, expected Δ +0.005 to +0.020 vs 0057.

Risk: 200 steps may not converge wide-MLP weights — very wide MLPs typically need more training to use their capacity. Could be neutral or slight loss.

Predicted Δ vs 0057: -0.005 to +0.020.

## Change
Single env-var: `MLP_MULT=8` (up from 3). No code change. Cap math: 21.3M params raw → at our ~6.5× compression ratio → ~13 MB int8+zlib (well under 16 MB).

## Disconfirming
- **Δ ≤ -0.005 vs 0057**: too-wide MLP under-trains at 200 steps; mlp=3 or mlp=4 likely the sweet spot.
- **Δ in [-0.005, +0.005]**: neutral; cap was better spent elsewhere (more loops, more unique blocks).
- **Δ ≥ +0.010 vs 0057**: clear win, promote.
- **Artifact > 16 MB**: cap violation; back off to mlp=6 or 7.

## Notes from execution
