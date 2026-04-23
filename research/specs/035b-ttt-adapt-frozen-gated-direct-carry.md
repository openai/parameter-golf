# Spec 035b — TTT adaptation of frozen gated direct-carry from `035`

**Slug:** `ttt-adapt-frozen-gated-direct-carry`
**Created:** 2026-04-23
**Status:** DRAFT
**Branch:** `TBD`
**Commit:** `TBD`
**Links to:** `research/ideas/035b-ttt-adapt-frozen-gated-direct-carry.md`, `research/specs/035-frozen-direct-carry-from-031b.md`, `research/specs/034b-ttt-adapt-frozen-direct-carry.md`

## Hypothesis

If `035` produces a healthy frozen gated direct-carry checkpoint, then TTT may
benefit from adapting that richer gated object.

This is the `031B` / gated analogue of `034b`.

## Base checkpoint

Base artifact will be the saved float checkpoint from `035`:

```text
/workspace/runs/035-frozen-direct-carry-from-031b/seed_314/final_model.pt
```

## Comparison target

This spec should mirror `034b`:

- `035bA`: corrected 3-phase hotstart baseline, frozen gated direct-carry during TTT
- `035bB`: same hotstart path, but gated direct-carry becomes learnable during TTT

Primary comparison:

- `035bB` vs `035bA`

Secondary comparison:

- `035` / `035bA` vs `034` / `034bA`

## Mechanism

Keep the same frozen gated direct-carry base as `035`.

Expected trainable tensors for `035bB`:

- `direct_carry_self_frozen`
- `direct_carry_edges_frozen_pass1`
- `direct_carry_edges_frozen_pass2`
- `direct_carry_gate_frozen`

## Initial LR stance

Match the conservative style used for `034bB`:

- `TTT_DIRECT_CARRY_ENABLED=1`
- `TTT_DIRECT_CARRY_LR_SCALE=0.5`

## TTT settings

Must explicitly use:

- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`

## Run protocol

To be filled after:

1. `035` exists
2. the gated hotstart path is implemented
3. the branch and commit are pinned

## Accept criteria

Strong success:

- `035bB` clearly beats `035bA`

Weak success:

- `035bB` is effectively tied

Failure:

- TTT adaptation of the gated carry object hurts
