# Spec 034c — `MIN_LR` floor on frozen `034`

**Slug:** `min-lr-on-034`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/034-frozen-direct-carry-from-031a`
**Commit:** `c532aea`
**Links to:** `research/ideas/034c-min-lr-on-034.md`, `research/ideas/1779-next-adds-ranked.md`, `research/specs/034-frozen-direct-carry-from-031a.md`

## Hypothesis

`034` is a stable frozen direct-carry deployment line. Adding a nonzero
`MIN_LR` floor is the cheapest plausible schedule improvement on top of it.

Because Parameter Golf runs are short, the default tail may be undercooked. A
hotter late-training floor could improve the endpoint and survive through:

- pre-quant diagnostic
- quantized diagnostic
- phased-TTT eval

## Baseline

Use the corrected `034` line as the direct baseline:

- frozen direct-carry from `031A`
- `NUM_LOOPS=2`
- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`
- explicit 3-phase TTT
- exact code lineage from:
  - branch `exp/034-frozen-direct-carry-from-031a`
  - commit `c532aea`

Reference comparison should center on the corrected post-TTT result from the
same family.

## Config diff

No code change.

Only change from `034`:

- set `MIN_LR` to one of:
  - `0.05`
  - `0.10`
  - `0.15`

Everything else stays pinned to the `034` contract.

That means all of the following must remain identical to `034`:

- branch and pinned commit
- dataset/tokenizer paths
- CaseOps / gated-attn / quant-gate settings
- model width/depth/head counts
- quantization bits and clip sigmas
- TTT settings and phase count
- shard selection and validation token count
- any other env not explicitly changed in this spec

## Recommended run order

If running only one first:

1. `034cB`: `MIN_LR=0.10`

If the first rung is healthy, then run:

2. `034cA`: `MIN_LR=0.05`
3. `034cC`: `MIN_LR=0.15`

## Regime

Pinned intent:

- `DIRECT_CARRY_MODE=frozen_edge_self`
- `NUM_LOOPS=2`
- `MAX_WALLCLOCK_SECONDS=1200`
- `ENABLE_LOOPING_AT=0.35`
- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`
- `DATA_DIR=/workspace/parameter-golf/data`
- `TORCHINDUCTOR_CACHE_DIR=/tmp/...`
- persistent `ARTIFACT_DIR=/workspace/runs/034c-min-lr-on-034/<rung>/seed_<seed>`

## Hardware ladder

1. `4×H100`, `1200s`, full pipeline

No separate smoke rung is required because this is an env-only schedule change.

## Run protocol

Three rung names:

- `034cA` = `MIN_LR=0.05`
- `034cB` = `MIN_LR=0.10`
- `034cC` = `MIN_LR=0.15`

Execution rule:

- launch from `exp/034-frozen-direct-carry-from-031a` at `c532aea`
- apply exactly one env diff:
  - `MIN_LR=<value>`
- if the produced `config.json` differs from `034` on anything else, the rung is
  invalid and must be aborted/relaunched

Pinned entrypoint pattern:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_034c_<rung> \
DATA_DIR=/workspace/parameter-golf/data \
ARTIFACT_DIR=/workspace/runs/034c-min-lr-on-034/<rung>/seed_314 \
MIN_LR=<value> \
DIRECT_CARRY_MODE=frozen_edge_self \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MAX_WALLCLOCK_SECONDS=1200 SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

Recommended immediate launch:

- `034cB`
- `SEED=314`
- `MIN_LR=0.10`

## Required artifacts

Must preserve persistent outputs under `/workspace/runs/...`, including:

- `final_model.pt`
- `final_model.int6.ptz`
- training log
- final metrics JSON
- `config.json`

## Sanity gate before accepting a rung

Before comparing to `034`, execution must verify from `config.json` that the
only intentional diff is:

- `MIN_LR`

If there are any other config-family differences, the run must be marked
invalid for `034c`.

## What to watch

- pre-EMA endpoint loss
- post-EMA pre-quant diagnostic
- quantized diagnostic
- post-TTT `val_bpb`
- any throughput or stability regression

## Accept criteria

Strong success:

- clear post-TTT win over corrected `034`
- no obvious pathology in train or quantized metrics

Weak success:

- one rung is directionally positive and neutral elsewhere

Failure:

- all rungs are flat or worse
- hotter tail helps train-side metrics but harms post-quant or post-TTT results

## Notes

- This is intentionally a simple add-on to `034`, not a new architecture line.
- If this line is flat, next effort should move to `035` rather than more LR
  laddering on `034`.
