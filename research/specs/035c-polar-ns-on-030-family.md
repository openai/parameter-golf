# Spec 035c — Polar NS on the original `030` alpha/beta family

> Obsolete slot. Superseded in the active research numbering by `035d`
> (`#1787`-lite non-sparse bundle) and `035e` (sparse-gate follow-up).

**Slug:** `polar-ns-on-030-family`
**Created:** 2026-04-24
**Status:** OBSOLETE
**Branch:** `exp/035c-polar-ns-on-030-family`
**Commit:** `188ce0b`
**Links to:** `research/ideas/035c-polar-ns-on-030-family.md`, `research/ideas/1779-next-adds-ranked.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

If `Polar NS` is a real optimizer-quality refinement rather than a bundled
artifact from `#1787`, it should transfer to the stronger original `030`
alpha/beta family in `4×H` screen form.

## Baseline

Use the same intended `030` `4×H` screen stack as `035`.

Pinned lineage:

- branch lineage: `exp/029-full-stack`
- runnable code line based on `c3a99b3`
- frozen `025b` carry
- `NUM_LOOPS=2`

Primary `4×H` benchmark:

- `026` screen seed `314`: pre-quant `1.06770372`

Direct siblings:

- `035` (`MIN_LR=0.10`)
- `035b` (loop-onset plateau)

## Config diff

Requires a small Muon optimizer patch on top of the `030` family code line.

Only intended diffs from the intended `030` `4×H` screen stack:

- Polar NS code present via this branch
- optional `MIN_LR` override chosen at launch from the pinned shortlist below

Everything else must remain identical, including:

- `CASEOPS_ENABLED=1`
- `TTT_ENABLED=0`
- `MLP_CLIP_SIGMAS=12.0`
- `ATTN_CLIP_SIGMAS=13.0`
- `EMBED_BITS=7`
- `EMBED_CLIP_SIGMAS=15.0`
- `MATRIX_LR=0.026`
- `GATED_ATTN_ENABLED=1`
- `GATED_ATTN_INIT_STD=0.005`
- `GATED_ATTN_QUANT_GATE=1`
- `RECUR_ALPHA_ENABLED=1`
- `NUM_LOOPS=2`
- `LOOP_START=3`
- `LOOP_END=5`
- `ENABLE_LOOPING_AT=0.35`
- `MUON_BACKEND_STEPS=5`
- `GPTQ_RESERVE_SECONDS=4`
- `GPTQ_CALIBRATION_BATCHES=16`
- `MAX_WALLCLOCK_SECONDS=1200`
- `TRAIN_LOG_EVERY=100`
- `SEED=314`

Runtime-selectable `MIN_LR` shortlist:

- `0.0` (default, pure Polar NS isolation)
- `0.05`
- `0.10`
- `0.15`

Execution may choose one of those values at launch time.
Any other `MIN_LR` value makes the rung invalid.

## Polar NS semantics

Replace stock Muon's repeated fixed coefficients:

- `(3.4445, -4.775, 2.0315)` applied 5 times

with the 5 per-iteration Polar Express tuples from PR `#1344` / `#1787`:

1. `(8.156554524902461, -22.48329292557795, 15.878769915207462)`
2. `(4.042929935166739, -2.808917465908714, 0.5000178451051316)`
3. `(3.8916678022926607, -2.772484153217685, 0.5060648178503393)`
4. `(3.285753657755655, -2.3681294933425376, 0.46449024233003106)`
5. `(2.3465413258596377, -1.7097828382687081, 0.42323551169305323)`

Keep:

- `MUON_BACKEND_STEPS=5`

## Regime

Use a `4×H100` screen-only rung.

Pinned intent:

- exact `030`-family `4×H` screen stack
- pre-quant gate only
- no TTT

## Run protocol

Launch variants:

- `035cA`
- `SEED=314`
- `MIN_LR=0.0` by default

Optional combo-prep variants:

- `035cB`
- same branch/commit, but with `MIN_LR` chosen from the shortlist above

Execution rule:

- launch from `exp/035c-polar-ns-on-030-family`
- use the pinned runnable code commit in this spec
- match the original intended `030` `4×H` screen stack exactly
- apply only the Polar NS code lineage change, plus an optional `MIN_LR`
  override from the pinned shortlist
- if the produced `config.json` differs on anything else, the rung is invalid

Pinned command:

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 188ce0b

if [ -f /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace
elif [ -f /workspace/parameter-golf/data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model ]; then
  export DATA_DIR=/workspace/parameter-golf/data
else
  echo "CaseOps tokenizer not found under either JP or NA layout" >&2
  exit 1
fi

mkdir -p /workspace/runs/035c-polar-ns-on-030-family/run_a/seed_314
mkdir -p /tmp/torch_inductor_cache_035c_a

NCCL_NET=Socket DATA_DIR=$DATA_DIR \
ARTIFACT_DIR=/workspace/runs/035c-polar-ns-on-030-family/run_a/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035c_a \
CASEOPS_ENABLED=1 \
TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
MUON_BACKEND_STEPS=5 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
MIN_LR=${MIN_LR:-0.0} \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035c-polar-ns-on-030-family/run_a/seed_314/train.log 2>&1
```

## Required artifacts

- training log
- `config.json`
- pre-quant metrics in the final output/log

## Sanity gate

Before accepting the result, execution must verify from `config.json` that the
only intentional diff from the intended `030` `4×H` screen stack is the Polar
NS code lineage itself, plus an optional `MIN_LR` value from the pinned
shortlist.

Data-root rule:

- if the CaseOps tokenizer exists under `/workspace/data/...`, use
  `DATA_DIR=/workspace`
- if it exists under `/workspace/parameter-golf/data/...`, use
  `DATA_DIR=/workspace/parameter-golf/data`
- if neither layout exists, abort

## Accept criteria

Strong success:

- pre-quant beats `1.06770372`

Weak success:

- directionally positive enough to justify promoting the chosen `MIN_LR`
  combination or a dedicated follow-up

Failure:

- flat or worse than the `026` `4×H` reference
