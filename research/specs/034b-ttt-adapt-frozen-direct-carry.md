# Spec 034b — TTT-only adaptation of frozen direct-carry from `034`

**Slug:** `ttt-adapt-frozen-direct-carry`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/034b-ttt-adapt-frozen-direct-carry`
**Commit:** `5a698a0`
**Links to:** `research/ideas/034b-ttt-adapt-frozen-direct-carry.md`, `research/specs/034-frozen-direct-carry-from-031a.md`, `research/specs/033-ttt-adapt-alpha-beta.md`, `research/specs/033b-ttt-adapt-alpha-beta-high-lr.md`

## Hypothesis

`034b` should answer two tightly coupled questions from the same saved `034`
checkpoint:

- what is the corrected 3-phase hotstart-TTT baseline for `034`?
- does letting TTT adjust the frozen direct-carry object help beyond that?

This is specifically a **TTT-only** spec, not a retrain.

## Base checkpoint

Base artifact is the saved float checkpoint from `034`:

```text
/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/final_model.pt
```

## Comparison target

This spec contains two sibling runs from the same checkpoint:

- `034bA`: corrected hotstart baseline, frozen direct-carry during TTT
- `034bB`: same hotstart path, but direct-carry becomes learnable during TTT

Primary comparison:

- `034bB` vs `034bA`

Secondary comparison:

- `034bA` is also the recovery path if a prior `034` run used the wrong
  1-phase TTT default instead of the intended 3-phase path

Interpretation:

- if `034bB` > `034bA`, then TTT wants to refine direct-carry
- if `034bB` ~= `034bA`, then frozen direct-carry is already good enough
- if `034bB` < `034bA`, then TTT carry adaptation is harmful or unnecessary

## Mechanism

Keep the same frozen direct-carry base as `034`.

For `034bA`:

- load the frozen direct-carry values from the `034` checkpoint
- keep them frozen during TTT

For `034bB`:

- load the same frozen direct-carry values from the same checkpoint
- make them trainable during TTT only
- include them in a separate TTT optimizer param group

Pinned trainable tensors for `034bB`:

- `direct_carry_self_frozen`
- `direct_carry_edges_frozen_pass1`
- `direct_carry_edges_frozen_pass2`

## Initial LR stance

Pinned first learnable probe for `034bB`:

- `TTT_DIRECT_CARRY_ENABLED=1`
- `TTT_DIRECT_CARRY_LR_SCALE=0.5`

Interpretation:

- LoRA keeps the normal `TTT_LORA_LR`
- direct-carry uses `0.5x` of that LR

This is still conservative, but stronger than the original `0.25x` no-op style
probe.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 5a698a0

# Sanity verify
grep -n "frozen_edge_self" train_gpt.py
grep -n "PHASED_TTT_NUM_PHASES" train_gpt.py
grep -n "TTT_DIRECT_CARRY_ENABLED" train_gpt.py
grep -n "TTT_DIRECT_CARRY_LR_SCALE" train_gpt.py
grep -n "ttt_direct_carry:" train_gpt.py

mkdir -p /workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_a_baseline/seed_314
mkdir -p /workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_b_learnable/seed_314
mkdir -p /tmp/torch_inductor_cache_034b

# 034bA: corrected 3-phase hotstart baseline from the 034 checkpoint.
SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/final_model.pt \
ARTIFACT_DIR=/workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_a_baseline/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_034b \
DATA_DIR=/workspace/parameter-golf/data \
CASEOPS_ENABLED=1 \
DIRECT_CARRY_MODE=frozen_edge_self \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TTT_DIRECT_CARRY_ENABLED=0 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_a_baseline/seed_314/ttt.log 2>&1

# 034bB: same hotstart path, but direct-carry is learnable during TTT.
SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/workspace/runs/034-frozen-direct-carry-from-031a/seed_314/final_model.pt \
ARTIFACT_DIR=/workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_b_learnable/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_034b \
DATA_DIR=/workspace/parameter-golf/data \
CASEOPS_ENABLED=1 \
DIRECT_CARRY_MODE=frozen_edge_self \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
TTT_DIRECT_CARRY_ENABLED=1 TTT_DIRECT_CARRY_LR_SCALE=0.5 \
PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=2 LOOP_START=3 LOOP_END=5 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /workspace/runs/034b-ttt-adapt-frozen-direct-carry/run_b_learnable/seed_314/ttt.log 2>&1
```

## Required logging

For `034bA`, must log the normal hotstart diagnostics and final phased-TTT
result.

For `034bB`, must additionally log:

- `ttt_direct_carry: enabled=1 ...`
- `ttt_direct_carry: before_self=...`
- `ttt_direct_carry: before_edges_pass1=...`
- `ttt_direct_carry: before_edges_pass2=...`
- live snapshots during phased TTT, for example:
  - `ttt_direct_carry: live_b17_self=...`
  - `ttt_direct_carry: live_b17_edges_pass1=...`
  - `ttt_direct_carry: live_b17_edges_pass2=...`
  - `ttt_direct_carry: live_b17 ..._max_drift=...`
- `ttt_direct_carry: after_self=...`
- `ttt_direct_carry: after_edges_pass1=...`
- `ttt_direct_carry: after_edges_pass2=...`
- `ttt_direct_carry: after ..._max_drift=...`

## Hotstart validation contract

Use the same style of hotstart validation as the `033` family for both `A` and
`B`:

- verify the checkpoint path exists
- verify the branch/commit contains the TTT direct-carry path
- verify the normal hotstart diagnostics are sane before interpreting TTT delta

If `034` already ran with the wrong 1-phase default, then `034bA` becomes the
canonical corrected post-TTT baseline for the saved `034` checkpoint.

## Stop-early criteria

- NaN in TTT loss → halt
- missing `ttt_direct_carry:` logs in `034bB` → halt
- checkpoint path missing or wrong → halt
- `034bB` clearly worse than `034bA` → flag hard

## Execution order

Only run after:

- `034` completed
- the `034` checkpoint path exists
- the base result is healthy enough to justify refinement

Run order inside `034b`:

1. `034bA` first
2. `034bB` second from the same checkpoint
