# Spec 034b — Frozen `025c`-style carry compressed from `031A`

**Slug:** `frozen-025c-from-031a`
**Created:** 2026-04-23
**Status:** DRAFT
**Branch:** `exp/034b-frozen-025c-from-031a`
**Commit:** `TBD`
**Links to:** `research/ideas/034b-frozen-025c-from-031a.md`, `research/specs/034-frozen-direct-carry-from-031a.md`, `research/specs/025c-cross-layer-carry-frozen-per-pass.md`, `research/specs/031-direct-carry-freefloat-neutral.md`

## Hypothesis

The useful structure learned by `031A` may survive a compression back into a
smaller `025c`-style per-pass `alpha/beta` object.

If so, we get most of the gain of `034` with a simpler frozen carry mechanism.

## Source

Same source snapshot as `034`:

- `031A-ratio0272`
- `val_step_4000`

This spec differs only in the frozen representation.

## Frozen object

Freeze a per-pass `alpha/beta` object:

- `recur_beta_frozen` shape `[2, 3]`
- `recur_alpha_frozen` shape `[2, 3, 3]`

Stored as buffers, not trainable parameters.

## Comparison target

Primary:

- compare against `034` native frozen direct-carry

Secondary:

- compare against historical `025c`-style frozen carry expectations

Interpretation:

- if `034b` ~= `034`, prefer the simpler compressed object
- if `034b` clearly loses, native direct-carry structure mattered

## Regime

Same live-like proxy as `034`:

- `4×H100`
- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`
- full pipeline including quantized and phased TTT
- save persistent artifacts

## Required artifacts

- `final_model.pt`
- `final_model.int6.ptz`
- train log
- diagnostics

## Open implementation point

The exact compression rule from `031A` direct-carry to `025c`-style `alpha/beta`
must be made explicit in code/spec before this can become `READY`.

That is the main unresolved part of this spec.
