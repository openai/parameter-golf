## Goal

Test the most portable QLoRA-style lesson from PR `#264` without inheriting its full cost structure.

## Hypothesis

- Mixed quantization by subsystem can buy bytes back without changing the training core.
- A tiny eval-time adaptation set can recover some of the benefit of TTT without paying full-model eval cost.

## Planned branch

- Base: `2026-03-20_LeaderCore10L_ValidEval_TempOnly_Int8Search`
- New branch: `2026-03-20_LeaderCore10L_MixedQuant_MicroTTT`

## Scope

- Keep the same `10L` training recipe.
- Change export only:
  - default `int5` for MLP matrices
  - default `int6` for attention matrices
  - keep small/control tensors in float
- Change eval only:
  - micro-TTT on `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, and `skip_weights`
  - default SGD budget: `32` steps

## What this should falsify

1. Whether mixed export precision materially improves the size/quality frontier on the valid `10L` line.
2. Whether tiny control-only adaptation has any measurable BPB payoff.
3. Whether either change is worth carrying to a full `8xH100` run.

## First ablations

- `base`
- `no_microttt`
- `no_mixedquant`
- `mlp6_attn8`
- `microttt64`
