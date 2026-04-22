# Idea: MLPClip12 — GPTQ MLP σ-clip retune (10 → 12)

**Source:** PR #1769 (dexhunter, 2026-04-22)
**Claimed Δ:** −0.00096 bpb (7-seed: 1.06477 vs #1736 1.06549; 5-best-of-7: 1.06453)

## What the lever is

GPTQ int6 calibration uses a per-row outlier clip threshold expressed in standard deviations
(`MLP_CLIP_SIGMAS`). Our #1736 baseline inherited the default `10.0`, which was calibrated
when the MLP was narrower. After widening to 4×MLP, the 10σ clip is too aggressive — it
clips valid tail mass in wide MLP weight rows, leaving a narrower quantization grid that
discards high-magnitude weights carrying disproportionate MLP signal.

Raising to `12.0` admits those tail columns, improving quantization fidelity for the 4×MLP
stack at essentially zero cost. GPTQ calibration runs post-training; TTT runs on the resulting
quantized artifact — TTT cannot absorb this improvement.

## Evidence

dexhunter PR #1769: 7-seed mean 1.06477 (vs #1736 1.06549), Δ = −0.00072 bpb. 5-best-of-7
mean 1.06453. Even the conservative 7-seed estimate clears the baseline. This is ~1× SOTA std,
real but modest signal.

## Implementation

Single env-var: add `MLP_CLIP_SIGMAS=12` to the run invocation. No code change, no new
branch needed.

First verify our baseline default: check
`records/track_10min_16mb/2026-04-19_.../train_gpt.py` for `MLP_CLIP_SIGMAS` default
(should be 10.0 inherited from #1736 before #1769).

## Estimated Δ-bucket

~−0.001 bpb (confirmed empirically).

## Feasibility

Trivial. Zero risk. Should be treated as a baseline hygiene fix rather than a spec — fold
into spec 026's run command if still in progress, else spec as the first item in spec 027.

## Risks

- May already be set to 12.0 in our code — check before speccing.
- Interaction with other GPTQ parameters (ATTN_CLIP_SIGMAS, MATRIX_CLIP_SIGMAS) is unknown;
  those could potentially benefit from similar retuning.
