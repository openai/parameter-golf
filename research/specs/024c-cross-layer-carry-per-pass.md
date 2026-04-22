# Spec 024c — Cross-layer carry blend, per-pass parameterization

**Slug:** `cross-layer-carry-per-pass`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `ba957e1`
**Depends on:** spec 024 passing throughput check

## Hypothesis

Extend 024b's cross-layer carry blend so each blend pass gets independent weights:

```python
x = beta[pass_off, i] * x_new + sum_j(alpha[pass_off, i, j] * carry[j])
```

- `beta[pass_off, local_idx]` — shape [2, 3] = 6 params
- `alpha[pass_off, local_idx, source_idx]` — shape [2, 3, 3] = 18 params
- Total: **24 scalars** (negligible vs 16MB)

Pass 2 and pass 3 through the looped layers can learn completely different cross-layer routing. There's no reason to share — the model state is different at each pass, different information may be useful.

Init: beta=ones, alpha=zeros — identical to baseline at start.

## Why per-pass matters

In 024b, `alpha[i,j]` was shared across passes. But pass 2 of layer 5 is doing something fundamentally different than pass 3 of layer 5 — the residual stream has been processed more. Giving each pass its own mixing matrix costs nothing (24 scalars total) and removes an arbitrary constraint.

## Baselines (4×H100, 4H wallclock)

| run | steps | val@4000 | pre-quant EMA |
|---|---|---|---|
| 021c (frozen param) | 5004 | 1.1177 | 1.06952 |
| 024 (detached-lerp) | TBD | TBD | — |
| 024b (cross-layer, shared) | TBD | TBD | — |

## Accept criteria

Same as 024b. Primary: steps ≥ 5000 (throughput). Secondary: val@4000 ≤ 1.1207.

Compare directly to 024b — if 024c val@4000 is meaningfully better, per-pass parameterization is load-bearing.

## Config diff vs 024b

| | 024b | 024c |
|---|---|---|
| Commit | `89367c5` | `ba957e1` |
| beta shape | [3] | [2, 3] |
| alpha shape | [3, 3] | [2, 3, 3] |
| Total params | 12 | 24 |

## Hardware ladder

4×H100 mini, 4H wallclock. Same as 024 and 024b.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout ba957e1

# Sanity verify
grep "h.num_loops, num_looped, num_looped" train_gpt.py    # alpha [2,3,3]
grep "h.num_loops, num_looped" train_gpt.py | grep beta    # beta [2,3]
grep -c "recur_beta\[pass_off" train_gpt.py                # must be 4
grep -c "recur_alpha\[pass_off, local_idx, j\]" train_gpt.py  # must be 4

mkdir -p /workspace/runs/024c-cross-layer-carry-per-pass/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/024c-cross-layer-carry-per-pass/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/024c-cross-layer-carry-per-pass/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_024c_4h \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
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
  > /workspace/runs/024c-cross-layer-carry-per-pass/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Note for JP pod:** replace `/workspace` with `/runpod` throughout.

## Monitoring

Same as 024. Extra signal: after loop activation, `recur_alpha: beta=... alpha=...` should show two separate [3] beta rows and two separate [3,3] alpha matrices (pass 2 and pass 3 diverging independently).

## Cost estimate

~$3–4 (4×H100, 4H mini)

## Decision tree

| 024c vs 024b val@4000 | Action |
|---|---|
| 024c clearly better (>0.001) | Per-pass parameterization load-bearing — use 024c for 8×H |
| Roughly equal | Either works — prefer 024c (more expressive, same cost) |
| 024c worse | Shared weights were better — use 024b for 8×H |
