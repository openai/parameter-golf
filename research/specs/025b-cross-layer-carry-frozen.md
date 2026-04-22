# Spec 025b — Cross-layer carry, frozen

**Slug:** `cross-layer-carry-frozen`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `950af24`

## Hypothesis

Hardcode 024b's converged beta/alpha as frozen register_buffers. If the routing structure
(L4 self-subtract, L5 cross-referencing L3/L4) is load-bearing, frozen cross-layer should
beat frozen diagonal (021e, best single-seed post-TTT 1.06622).

Same logic as 021e: 017 learned good alpha → freeze it → best result. Now: 024b learned
good cross-layer routing → freeze it → potentially better result. Zero throughput overhead
since both beta and alpha are frozen buffers.

## Hardcoded values (from 024b seed_42 final log)

```python
recur_beta  = [1.5973426, 1.8826205, 1.9906198]   # [L3, L4, L5] — shared across passes
recur_alpha = [[+0.2520, -0.0210, -0.0124],        # L3: mostly self, ignore L4/L5
               [+0.0669, -0.3477, +0.0031],        # L4: self-subtract, small L3 add
               [+0.1387, +0.2412, +0.0272]]        # L5: pulls from L3 + L4
```

## Baselines (4×H100, 4H wallclock)

| run | steps | val@4000 | pre-quant EMA |
|---|---|---|---|
| 021c (frozen diagonal) | 5004 | 1.1177 | 1.06952 |
| 024 (learnable diagonal) | 4975 | 1.1185 | 1.07106 |
| 024b (learnable cross-layer) | 5017 | 1.1196 | 1.06960 |

## Accept criteria

Primary: steps ≥ 5000 (throughput — frozen buffer should have zero overhead).
Secondary: val@4000 ≤ 1.1177 (match or beat frozen diagonal 021c).

If 025b val@4000 < 021c val@4000 → routing structure load-bearing → promote to 8×H.
If 025b val@4000 ≈ 021c val@4000 → routing structure neutral → skip 8×H for this variant.

## Config diff vs 024b

| | 024b | 025b |
|---|---|---|
| Commit | `89367c5` | `950af24` |
| recur_beta | nn.Parameter([2,3] ones) | register_buffer([3] hardcoded) |
| recur_alpha | nn.Parameter([2,3,3] zeros) | register_buffer([3,3] hardcoded) |
| Optimizer | scalar_params includes beta+alpha | no recur params in optimizer |

## Hardware ladder

4×H100 mini, MAX_WALLCLOCK_SECONDS=1200. Same as 024/024b.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 950af24

# Sanity verify
grep "1.5973426" train_gpt.py        # must match beta[0]
grep "\-0.34765625" train_gpt.py     # must match alpha[L4,L4] self-subtract
grep -c "recur_beta\[local_idx\]" train_gpt.py    # must be 4
grep -c "recur_alpha\[local_idx, j\]" train_gpt.py  # must be 4
grep "recur_beta.*requires_grad\|recur_alpha.*requires_grad" train_gpt.py  # must be empty

mkdir -p /workspace/runs/025b-cross-layer-carry-frozen/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/025b-cross-layer-carry-frozen/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/025b-cross-layer-carry-frozen/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_025b_4h \
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
  > /workspace/runs/025b-cross-layer-carry-frozen/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Note for JP pod:** replace `/workspace` with `/runpod` throughout.

## Monitoring

Same as 024. Key signal: `recur_alpha: beta=[1.597, 1.883, 1.991] alpha=[[...]]` should appear
at every TRAIN_LOG_EVERY step after loop activation, with values exactly matching the hardcoded
constants (no drift — these are frozen).

## Cost estimate

~$3–4 (4×H100, 20 min mini)

## Decision tree

| 025b val@4000 vs 021c (1.1177) | Action |
|---|---|
| Clearly better (< 1.115) | Cross-layer routing load-bearing → spec 026 (8×H) |
| Roughly equal (1.115–1.120) | Neutral — frozen diagonal (021e) remains best bet for 8×H |
| Worse (> 1.120) | Routing structure hurts — kill cross-layer arc |
