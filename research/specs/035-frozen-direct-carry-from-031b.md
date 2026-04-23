# Spec 035 — Frozen gated direct-carry from `031B`

**Slug:** `frozen-direct-carry-from-031b`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/035-frozen-direct-carry-from-031b`
**Commit:** `4919e3b`
**Links to:** `research/ideas/035-frozen-direct-carry-from-031b.md`, `research/specs/031-direct-carry-freefloat-neutral.md`, `research/specs/034-frozen-direct-carry-from-031a.md`

## Hypothesis

`031B` is the gated sibling of `031A`:

- `031A` = `edge + self`
- `031B` = `edge + self + carry_gate`

If the extra per-destination carry gate is genuinely useful, then freezing the
learned `031B` structure as buffers should beat the simpler `034` line that came
from `031A`.

This is the `031B` analogue of `034`.

## Source snapshot

Freeze from the healthy corrected `031B` run:

- source artifact:
  - `runs/031-direct-carry-freefloat-neutral/031B-ratio0272-freshpod-rerun1/carry_snapshots.jsonl`
- pinned late snapshot:
  - `train_step_5400`

Pinned frozen values:

```text
self =
[[1.0234375, 1.4921875, 2.0],
 [2.0,       2.0,       1.3125]]

carry_gate =
[[0.44921875, 0.53515625, 0.6953125],
 [1.046875,   0.6484375,  0.3984375]]

edges_pass1 =
[[ 0.416015625,  -0.04931640625,  0.21875],
 [ 0.36328125,   -0.27734375,    -0.0093994140625],
 [-0.1025390625,  0.345703125,   -0.4296875]]

edges_pass2 =
[[ 0.28125,      -0.09130859375,  0.24609375,   -0.45703125,   -0.06689453125,  0.03271484375],
 [ 0.10205078125,-0.1337890625,  -0.07080078125, 0.42578125,   -0.333984375,    0.01348876953125],
 [-0.06494140625, 0.1904296875,  -0.060791015625, 0.06396484375, 0.478515625,   -0.123046875]]
```

Late drift at this snapshot is small enough to freeze:

- `edge_max_drift = [0.0103, 0.0117]`
- `self_max_drift = 0.0`
- `gate_max_drift = 0.0039`

Pinned frozen objects:

- `self`
- `edges_pass1`
- `edges_pass2`
- `carry_gate`

## Storage semantics

Store the frozen gated direct-carry tensors as `register_buffer(...)`, not
parameters.

Expected pinned objects:

- `direct_carry_self_frozen`
- `direct_carry_edges_frozen_pass1`
- `direct_carry_edges_frozen_pass2`
- `direct_carry_gate_frozen`

Requirements:

- not trainable during training
- not in optimizer param groups
- serialized in `final_model.pt`
- logged repeatedly so execution can confirm they remain fixed

## Comparison targets

Primary structural comparison:

- `035` vs `034`

Practical comparison:

- does the gated frozen direct-carry line compete with or beat the older frozen
  carry baselines (`25b` / `25c`)?

## Regime

Match the live-like proxy shape used for `034`:

- `4×H100`
- `MAX_WALLCLOCK_SECONDS=1200`
- `NUM_LOOPS=2`
- `ENABLE_LOOPING_AT=0.35`
- full pipeline including corrected 3-phase TTT

## TTT settings

Must explicitly use the newer phased TTT path:

- `TTT_ENABLED=1`
- `PHASED_TTT_PREFIX_DOCS=2000`
- `PHASED_TTT_NUM_PHASES=3`

## Checkpoint requirements

Preserve persistent artifacts under `/workspace/runs/...`, including:

- `final_model.pt`
- `final_model.int6.ptz`
- training log
- diagnostics

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 4919e3b

# Sanity verify
grep -n "frozen_edge_self_carrygate" train_gpt.py
grep -n "direct_carry_gate_frozen" train_gpt.py
grep -n "PHASED_TTT_NUM_PHASES" train_gpt.py

mkdir -p /workspace/runs/035-frozen-direct-carry-from-031b/seed_314
mkdir -p /tmp/torch_inductor_cache_035_4h

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/035-frozen-direct-carry-from-031b/seed_314/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/parameter-golf/data \
ARTIFACT_DIR=/workspace/runs/035-frozen-direct-carry-from-031b/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_035_4h \
CASEOPS_ENABLED=1 \
TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 ENABLE_LOOPING_AT=0.35 \
DIRECT_CARRY_MODE=frozen_edge_self_carrygate \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/035-frozen-direct-carry-from-031b/seed_314/train.log 2>&1

kill $NVSMI_PID
```

## Accept criteria

Strong success:

- `035` beats `034`
- `035` is competitive with or better than `25b` / `25c`
- frozen gated direct-carry looks like a better promotion candidate than plain
  frozen direct-carry

Weak success:

- healthy run, but essentially tied with `034`

Failure:

- gate adds no value or makes the frozen line worse
