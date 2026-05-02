# Spec 038 — SmearGate + LQER-asym on top of the full-float sparse-carry `8×H` line

**Slug:** `smear-lqer-asym-8h`
**Created:** 2026-04-24
**Status:** READY
**Branch:** `exp/038-smear-lqer-8h-promotion`
**Commit:** `9636d34`
**Links to:** `research/specs/037-fullfloat-sparse-updated-alpha-beta-8h.md`, `research/specs/035e-sparse-gate-on-1779-family.md`, `runs/035-series-report.md`

## Hypothesis

`037` promotes the current best internal sparse-gate family with the full-float
learned `alpha/beta`. `038` adds the two orthogonal `#1797` levers on top:

- **SmearGate** during training / forward
- **LQER-asym** during GPTQ pack + quantized eval / TTT

Because these hit different parts of the stack, they are worth trying together
directly on the strongest current sparse-family promotion line.

## Baseline

Immediate baseline:

- `037` full-float sparse-updated-`alpha/beta` `8×H` line

External reference points:

- `#1787` seed `42`: pre-quant `1.06764`, quantized `1.07681`, post-TTT `1.06400`
- `#1797` seed `42`: pre-TTT `1.07460`, post-TTT `1.06181`
- `#1797` mean: pre-TTT `1.07443`, post-TTT `1.06157`

## Config diff

Relative to `037`:

- `SMEAR_GATE_ENABLED=1`
- `GATE_WINDOW=12`
- `LQER_ENABLED=1`
- `LQER_RANK=4`
- `LQER_TOP_K=3`
- `LQER_FACTOR_BITS=4`
- `LQER_ASYM_ENABLED=1`
- `LQER_ASYM_GROUP=64`

Everything else stays on the `037` stack:

- sparse gate on
- dense gated-attn off
- full-float frozen updated `035h` carry
- `MIN_LR=0.10`
- `FUSED_CE_ENABLED=1`
- phased LoRA-TTT
- `VAL_LOSS_EVERY=0`
- `MAX_WALLCLOCK_SECONDS=600`

Pinned runnable code source:

- shell/spec branch: `exp/038-smear-lqer-8h-promotion`
- runnable code branch: `exp/038-fullfloat-smear-lqer-asym`
- runnable code commit: `c8620b6`

## Regime

This is a direct `8×H100` full-pipeline promotion.

- no smoke rung
- full quantized eval
- phased LoRA-TTT
- same updated full-float frozen `alpha/beta` as `037`

## Seed policy

Use the public comparison seed family for apples-to-apples checks against
`#1787` / `#1797`:

- `42`
- `0`
- `1234`

Recommended first seed:

- `42`

## Hardware ladder

1. `8×H100` full pipeline, `600s`, first seed `42`

Optional later:

2. additional seeds from the approved shortlist

## Run protocol

Primary rung:

- `038A`
- `8×H100`
- no smoke
- full quantized eval + phased LoRA-TTT
- same `037` sparse-family full-float carry
- add SmearGate + LQER-asym

Execution rule:

- launch from `exp/038-fullfloat-smear-lqer-asym`
- use the pinned runnable code commit
- validate the runnable file at:
  - `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`
- ignore branch-root `./train_gpt.py` on this branch; it is the generic starter
  baseline, not the runnable record-stack file for this spec
- keep the full `037` stack unchanged
- apply only the SmearGate + LQER-asym diffs above
- require `config.json`
- if anything else drifts, the rung is invalid

## Acceptance

Primary target:

- healthy quantized and post-TTT run on the `037` base with no path mismatch

Competitive target:

- post-TTT at or below the `#1787` seed-`42` reference (`1.06400`)

Stretch target:

- enter the low-`1.062` band and become comparable to `#1797`

Failure:

- no measurable improvement vs `037`
- or regression concentrated in quantized / post-TTT stage

## Notes

- SmearGate already exists in the `037` code line and is mirrored into the TTT
  path; this spec simply turns it on.
- LQER-asym is the actual code addition in the runnable branch.
- This is intentionally a direct combined test, not separate ablations.
