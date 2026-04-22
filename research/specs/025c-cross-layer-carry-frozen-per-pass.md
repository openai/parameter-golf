# Spec 025c — Cross-layer carry, frozen per-pass

**Slug:** `cross-layer-carry-frozen-per-pass`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `414cbc3`

## Hypothesis

Freeze 024c's converged per-pass beta/alpha as register_buffers. 024c learned that pass 1
and pass 2 need fundamentally different routing — pass 1 amplifies with depth, pass 2
reverses that pattern. 025b (shared frozen) can't capture this. If the per-pass
differentiation is load-bearing, 025c should beat 025b.

## Hardcoded values (from 024c seed_42 final log)

```
pass 1 beta:  [0.9453, 1.2969, 1.6250]  ← amplifies with depth (L5 strongest)
pass 2 beta:  [1.9844, 1.6875, 1.1563]  ← reverses: L3 strongest, L5 weakest

pass 1 alpha: [[+0.336, +0.004, +0.018],   L3: self-dominant
               [+0.073, -0.305, -0.011],   L4: self-subtract
               [+0.001, +0.279, -0.326]]   L5: L4-add, self-subtract

pass 2 alpha: [[+0.316, -0.059, +0.158],   L3: self + L5 pull
               [+0.041, -0.291, +0.019],   L4: self-subtract
               [+0.057, +0.131, +0.073]]   L5: balanced
```

## Baselines (4×H100)

| run | val@4000 | pre-quant EMA |
|---|---|---|
| 021c frozen diagonal | 1.1177 | 1.06952 |
| 025b frozen cross (shared) | TBD (live) | TBD |

## Accept criteria

Primary: steps ≥ 5000.
Secondary: val@4000 < 025b val@4000 (per-pass better than shared).
Stretch: val@4000 < 1.1177 (beats frozen diagonal).

## Config diff vs 025b

| | 025b | 025c |
|---|---|---|
| Commit | `950af24` | `414cbc3` |
| recur_beta shape | [3] | [2, 3] |
| recur_alpha shape | [3, 3] | [2, 3, 3] |

## Hardware ladder

4×H100 mini, MAX_WALLCLOCK_SECONDS=1200.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 414cbc3

# Sanity verify
grep "0.9453125" train_gpt.py              # pass 1 beta[L3]
grep "1.984375" train_gpt.py               # pass 2 beta[L3]
grep -c "recur_beta\[pass_off" train_gpt.py   # must be 4
grep -c "recur_alpha\[pass_off" train_gpt.py  # must be 4

mkdir -p /workspace/runs/025c-cross-layer-carry-frozen-per-pass/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/025c-cross-layer-carry-frozen-per-pass/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/025c-cross-layer-carry-frozen-per-pass/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_025c_4h \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /workspace/runs/025c-cross-layer-carry-frozen-per-pass/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Note for JP pod:** replace `/workspace` with `/runpod` throughout.

## Cost estimate

~$3–4 (4×H100, 20 min mini)

## Decision tree

| 025c vs 025b val@4000 | Action |
|---|---|
| 025c clearly better (>0.002) | Per-pass routing load-bearing → use 025c for 8×H |
| Roughly equal | Prefer 025b (simpler) for 8×H |
| 025c worse | Shared is sufficient — 025b for 8×H |
