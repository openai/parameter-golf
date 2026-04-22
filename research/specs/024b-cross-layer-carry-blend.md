# Spec 024b — Cross-layer carry blend

**Slug:** `cross-layer-carry-blend`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `89367c5`
**Depends on:** spec 024 passing throughput check (launch 024b only after 024's first ~10 min look healthy)

## Hypothesis

Instead of blending only with x_before (spec 024 detached-lerp), give each looped layer on pass 2 direct access to ALL looped layer outputs from pass 1:

```python
x = beta[i] * x_new + sum_j(alpha[i,j] * carry[j])
```

- `beta[i]` (3 params, init=1): scale on current pass output
- `alpha[i,j]` (9 params, init=0): weight for layer j's pass-1 output added to layer i's pass-2 output
- `carry[j]` = detached first-pass output of looped layer j

At init: beta=1, alpha=0 → identical to baseline. Gradients activate carry terms where useful.

**Why more expressive than 024:** layer 5 pass 2 can directly reference layer 3's pass-1 output, even though that signal has been transformed through layers 4 and 5 again before reaching layer 5's input. Cross-layer shortcuts may preserve information that sequential recurrence washes out.

**Cost:** same as 024 — all carries detached, no backward retention overhead.

## Baselines (4×H100, 4H wallclock)

| run | steps | val@4000 | pre-quant EMA |
|---|---|---|---|
| 019b ref (frozen constant) | 5034 | 1.1190 | 1.06927 |
| 021c (frozen param) | 5004 | 1.1177 | 1.06952 |
| 024 (detached-lerp) | target ≥5000 | target ≤1.1207 | — |

024b should match 024 on throughput and ideally beat it on val@4000.

## Accept criteria

| Steps at 4H | val@4000 | Decision |
|---|---|---|
| ≥ 5000 | < 024's val@4000 | 024b better — use 024b for 8×H run |
| ≥ 5000 | ≈ 024's val@4000 | Equivalent — prefer 024b (more expressive) |
| ≥ 5000 | > 024 + 0.003 | Quality regression — investigate beta/alpha init |
| < 4900 | any | Carry overhead not eliminated — debug |

## Config diff vs 024

| var | 024 | 024b |
|---|---|---|
| Commit | `613bc8e` | `89367c5` |
| Blend form | `x_before_det + alpha*(x_new - x_before_det)` | `beta*x_new + sum(alpha[i,j]*carry[j])` |
| Params | 6 (alpha [2,3]) | 12 (beta [3] + alpha [3,3]) |
| Carry dict | no | yes (all detached) |

## Hardware ladder

**4×H100 mini, 4H wallclock.** Same as spec 024. Run in parallel with 024 after confirming 024 starts cleanly.

## Seed plan

Seed 42 only. 3-seed conditional on 8×H promoting.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 89367c5

# Sanity verify
grep "torch.zeros.*num_looped.*num_looped" train_gpt.py    # alpha [3,3] zeros
grep "torch.ones.*num_looped" train_gpt.py                 # beta [3] ones
grep -c "beta \* x_new" train_gpt.py                       # must be 4
grep -c "carry\[self.loop_start" train_gpt.py              # must be 4

mkdir -p /workspace/runs/024b-cross-layer-carry-blend/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/024b-cross-layer-carry-blend/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/024b-cross-layer-carry-blend/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_024b_4h \
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
  > /workspace/runs/024b-cross-layer-carry-blend/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Note for JP pod:** replace `/workspace` with `/runpod` throughout.

## Benchmarks

### Throughput (steps at 4H cap, 4×H100)

| run | steps | Δ |
|---|---|---|
| 019b ref | 5034 | — |
| 021c frozen | 5004 | -30 |
| **024b target** | **≥ 5000** | **≤ -34** |

### Quality (val@4000)

| run | val@4000 |
|---|---|
| 021c frozen | 1.1177 |
| 024 detached-lerp | TBD |
| **024b target** | **better than or ≈ 024** |

## Monitoring protocol

Same as spec 024. Key additional signal:

- **After loop activation (~step 2200):** first `recur_alpha: beta=... alpha=...` log line
  - beta values should be near 1.0
  - alpha values should be near 0.0 and gradually diverging
- **By step 3500:** are any alpha[i,j] terms nonzero? Which cross-layer connections are being learned?

## Stop-early criteria

- NaN/inf → halt
- Step time > 1.5× 021c → halt
- `layer_loop_enabled_at_step` outside [2000, 2400] → halt
- Any beta value < 0.1 after step 3000 → flag (model collapsing x_new contribution)

## Cost estimate

~$3–4 (4×H100, 4H). Run in parallel with 024 for ~$7 total.

## Open questions for executor

1. **Carry populated before blend?** All looped layers must be visited (and carry stored) before any blend site executes. With encoder sequence [3,4,5,3,4] and decoder [5,3,4,...], carry[3] and carry[4] populated in encoder steps 3-4, carry[5] in encoder step 5, all before any blend. Verify `carry` dict has 3 keys at first blend site.
2. **Beta/alpha log format:** `recur_alpha: beta=[...] alpha=[[...]]` — confirm this appears at TRAIN_LOG_EVERY steps after loop activation.
