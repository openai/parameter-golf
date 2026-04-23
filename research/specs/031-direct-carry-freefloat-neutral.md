# Spec 031 — Direct-carry free-float calibration (`NUM_LOOPS=2`, neutral init)

**Slug:** `direct-carry-freefloat-neutral`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/031-direct-carry-freefloat`
**Commit:** `1cac69b`
**Links to:** `research/ideas/direct-carry-freefloat-num-loops-3.md`, `research/specs/025b-cross-layer-carry-frozen.md`, `research/specs/025c-cross-layer-carry-frozen-per-pass.md`, `research/specs/030-025b-seed314-new-ttt.md`

## Hypothesis

Before revisiting third-pass / recurrence-curriculum complexity, run a cleaner carry recalibration in the simpler `NUM_LOOPS=2` regime.

The `025b/025c` arc already established that cross-layer carry is real signal. What we want here is not another frozen hand-picked matrix, but a free-floating calibration run with a richer direct-carry combiner so we can see what the model actually wants in the cleaner 2-loop setting.

Two variants live under this one spec:

- **031A** — `edge + self`
- **031B** — `edge + self + carry_gate`

Both use:

- `NUM_LOOPS=2`
- detached carry
- neutral init
- same runtime budget
- same recurrence timing

The only difference is whether we add an explicit per-destination carry gate.

## Why `NUM_LOOPS=2` first

We are deliberately **not** using the 3-loop / curriculum regime for this first calibration pass.

Reason:

- keeps the carry experiment close to the successful `025b/025c` family
- avoids confounding carry calibration with third-pass behavior
- avoids mixing carry design with recurrence curriculum
- makes the learned structure easier to interpret

The 3-loop question can come later as a separate follow-up once we understand the cleaner 2-loop version.

## Baselines

| run | regime | pre-quant EMA | post-TTT | note |
|---|---|---|---|---|
| 025b | frozen shared cross-layer carry | 1.06917 | not run | best frozen 4×H screen in the arc |
| 025c | frozen per-pass cross-layer carry | 1.06969 | not run | slightly worse than 025b |
| 030 seed 314 | 025b frozen under healthy 2-loop run | 1.06821629 | 1.06471941 | strong reference |

Primary comparisons for this calibration:

- does free-floating direct-carry learn something better than the old frozen 025b-style structure?
- does adding `carry_gate` help beyond plain `edge + self`?
- do the learned coefficients stabilize enough to freeze?

## Parameterization

The implementation should **derive the carry dimensions from the real loop segment**, not hardcode a toy size.

For current defaults:

- `LOOP_START=3`
- `LOOP_END=5`

so the active loop segment has:

- `num_looped = LOOP_END - LOOP_START + 1 = 3`

Also note the current model repeats the loop segment `NUM_LOOPS + 1` times while looping is active. So with:

- `NUM_LOOPS=2`

there are **two carry-applied later passes** after the first baseline pass.

### Variant 031A: `edge + self`

- edge weights for later pass 1: shape `[3, 3]`
- edge weights for later pass 2: shape `[3, 6]`
- self weights: shape `[2, 3]`

Total: **33 learnable scalars**

### Variant 031B: `edge + self + carry_gate`

- edge weights for later pass 1: shape `[3, 3]`
- edge weights for later pass 2: shape `[3, 6]`
- self weights: shape `[2, 3]`
- carry gates: shape `[2, 3]`

Total: **39 learnable scalars**

## Natural initialization

For both 031A and 031B:

- carry-edge coefficients = `0`
- self coefficients = `1`
- carry gates (031B only) = `1`

This gives identity-like behavior at init:

- no external carry contribution initially
- normal local/self path preserved

## Carry behavior

- all carry sources detached
- later pass 1 reads all loop-layer outputs from the first pass
- later pass 2 reads all loop-layer outputs from the first pass and later pass 1
- no recurrence curriculum in this spec
- no delayed recurrence-kick modification in this spec

## Runtime / schedule

Use the normal 2-loop training structure, but extend the usual 4×H screen budget by about **20%** so the learned coefficients get a longer post-loop tail for stabilization.

### Critical timing rule

This needs to be stated very clearly:

**If the runtime is extended, the recurrence/loop-start ratio must be changed back so loop activation still occurs at roughly the same absolute step as before.**

We want:

- loop start at approximately **step 2100**

If the extended run reaches about:

- **6000 total steps**

then the corrected loop-start ratio must be:

```text
2100 / 6000 = 0.35
```

So the execution instructions must explicitly say:

- do **not** leave the old loop-start ratio unchanged
- after extending runtime, set the loop-start ratio to **0.35** if total steps are ~6000
- more generally:

```text
new_loop_start_ratio = 2100 / new_total_steps
```

The purpose of the extension is to create a **longer post-loop learning window**, not to delay loop activation.

This ratio correction must be called out explicitly in the execution interview so it is not missed.

## Expected signal

This run should tell us:

1. whether direct-carry beats the old frozen 025b-style carry in the clean 2-loop regime
2. whether adding a carry gate helps beyond plain `edge + self`
3. whether the learned coefficients stabilize enough to freeze

## Accept criteria

This is a calibration spec, so acceptance is not just a single bpb threshold.

### Primary accept criteria

- run completes cleanly with no NaN / shape bugs
- float trajectory is competitive with the strong 2-loop references
- coefficients move away from init in a structured way
- the extra 20% tail reduces late-stage drift enough that freezing looks plausible

### Strong success

- pre-quant EMA is competitive with or better than the best relevant 2-loop frozen reference
- one of A/B clearly wins
- learned coefficients look stable enough to freeze directly into a follow-up spec

### Weak / inconclusive

- float improves or looks healthy, but coefficient drift is still material near the end
- action: run longer or modestly adjust carry-param LR

### Failure

- both A and B fail to compete with the relevant 2-loop references
- coefficients remain noisy or near-init
- training becomes unstable

## Hardware ladder

1. **4×H100 calibration run** — this is the first rung
2. **Frozen follow-up run** — separate future spec, only if 031 yields a usable learned pattern

Mini policy:

- if the implementation diff remains a small local carry-combiner change, 4×H can serve as the first smoke
- if the code diff grows beyond that, add a 2×H smoke before launch

Execution order inside this one spec:

1. Run **031A** first
2. Run **031B only if 031A is healthy**

“Healthy” means:

- no shape / indexing bug
- no NaN
- float trajectory competitive with the relevant 2-loop references
- carry coefficients clearly move off init

## Logging requirements

In addition to the normal metrics, emit:

- full carry-edge tensors at every val/log interval
- full self-coefficient tensors at every val/log interval
- carry-gate tensors for 031B
- final snapshots in `final.json` or a sidecar file

Helpful summaries:

- max abs edge weight
- row-wise norms per destination
- drift over the last N snapshots

## Optimizer treatment

Use a separate scalar parameter group for the carry parameters.

Pinned default for this spec:

- carry-parameter LR = **1.5×** the normal scalar LR

Reason:

- this is a calibration run
- the carry block is tiny and starts from a neutral identity-like init
- we want it to move decisively within one extended run
- but we do **not** want an aggressive LR that makes the coefficients noisy or unstable

So:

- base model optimizer groups stay unchanged
- carry params get their own param group
- use **1.5×**, not a larger multiplier, unless execution shows clear under-movement

## Code changes

Planned implementation shape:

- replace the existing carry combiner at the same insertion point; do not invent a new insertion point
- keep carry detached
- derive `num_looped` from `loop_start..loop_end`
- 031A:
  - one edge matrix per later pass, with source width growing with available prior passes
  - one self-weight row per later pass
- 031B:
  - same as 031A
  - one carry-gate row per later pass
- initialize edges to zero and self / carry gates to one

Pinned implementation commit: `1cac69b`

## Run protocol

Both runs use the same code commit and only differ by `DIRECT_CARRY_MODE`.

### Shared execution notes

- use 4×H100
- use `PHASED_TTT_ENABLED=0` for these calibration screens
- extend the normal 20-minute wallclock by 20%:
  - `MAX_WALLCLOCK_SECONDS=1440`
- set:
  - `ENABLE_LOOPING_AT=0.35`

This value is intentional. It is the corrected loop-start ratio to keep loop activation around step ~2100 under the extended runtime. Do **not** leave the old ratio unchanged.

### 031A — `edge + self`

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 1cac69b

mkdir -p /workspace/runs/031-direct-carry-freefloat-neutral/031A
mkdir -p /tmp/torch_inductor_cache_031A

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/031-direct-carry-freefloat-neutral/031A/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/031-direct-carry-freefloat-neutral/031A \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_031A \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
SCALAR_LR=0.02 \
DIRECT_CARRY_MODE=edge_self \
DIRECT_CARRY_LR_SCALE=1.5 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
MAX_WALLCLOCK_SECONDS=1440 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/031-direct-carry-freefloat-neutral/031A/train.log 2>&1

kill $NVSMI_PID
```

### 031B — `edge + self + carry_gate`

Run only if 031A is healthy.

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 1cac69b

mkdir -p /workspace/runs/031-direct-carry-freefloat-neutral/031B
mkdir -p /tmp/torch_inductor_cache_031B

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/031-direct-carry-freefloat-neutral/031B/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/031-direct-carry-freefloat-neutral/031B \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_031B \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
SCALAR_LR=0.02 \
DIRECT_CARRY_MODE=edge_self_carrygate \
DIRECT_CARRY_LR_SCALE=1.5 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
MAX_WALLCLOCK_SECONDS=1440 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/031-direct-carry-freefloat-neutral/031B/train.log 2>&1

kill $NVSMI_PID
```

### Sanity verify before launch

```bash
grep -n "DIRECT_CARRY_MODE" train_gpt.py
grep -n "DIRECT_CARRY_LR_SCALE" train_gpt.py
grep -n "_apply_direct_carry" train_gpt.py
grep -n "direct_carry_summary" train_gpt.py
```

## Cost estimate

Expected:

- two 4×H calibration variants under one spec
- each run uses the extended ~20%-long budget

Still much cheaper than burning another 8×H full-pipeline attempt on a carry design we do not understand.

## Stop-early criteria

- NaN / inf in train loss
- obvious shape or indexing bug in the carry path
- coefficients remain exactly at init long after loop activation, suggesting broken gradient plumbing
- catastrophic float regression relative to the relevant 2-loop references

## Open questions for the spec interview

1. What exact extended wallclock / expected step count should be pinned before launch, so the corrected loop-start ratio can be computed precisely?
2. If the expected total step count differs materially from ~6000 on the chosen hardware draw, what exact corrected loop-start ratio should execution pin before launch?
