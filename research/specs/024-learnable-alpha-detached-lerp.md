# Spec 024 — Learnable α detached-lerp (throughput fix)

**Slug:** `learnable-alpha-detached-lerp`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/recur-alpha-buffer`
**Commit:** `613bc8e`

## Context

All learnable-α runs (015–017, 021g, 021h) show ~1.4% throughput overhead vs frozen α — roughly 80–120 fewer steps at 8×H100 8H wallclock. The original blend form:

```python
x = x_before + alpha * (x_new - x_before)
```

keeps `x_before` (live tensor) in the computation graph. PyTorch must save it for backward at each of 4 blend sites. The fix is to detach x_before at blend time:

```python
x_before_det = x_before.detach()
x = x_before_det + alpha * (x_new - x_before_det)
```

`x_before_det` is now a constant — no backward save. `alpha` still gets gradients via `d_loss/d_alpha = upstream * (x_new - x_before_det)`, which reuses `x_new` already saved by block internals. Net extra saves at blend sites: ~0.

Alpha init changed from zeros → ones (lerp identity = full transformation at start).

## Hypothesis

Detached-lerp recovers most or all of the throughput overhead while preserving alpha's ability to learn. With throughput matching frozen α, learnable α gets the same number of steps — removing the systematic disadvantage in pre-quant EMA.

## Baselines (4×H100, 4H wallclock)

| run | steps | val@4000 | pre-quant EMA |
|---|---|---|---|
| 019b ref (frozen constant) | 5034 | 1.1190 | 1.06927 |
| 021c (frozen nn.Parameter) | 5004 | 1.1177 | 1.06952 |

These are the throughput and quality bars to match.

## Expected Δ

| Signal | Expected |
|---|---|
| Steps at 4H | ≥ 5000 (matching frozen) |
| val@4000 | ~1.117–1.119 (same ballpark as 021c) |
| pre-quant EMA | competitive with 021c (1.06952) |

## Accept criteria

| Steps at 4H | Decision |
|---|---|
| ≥ 5000 | Throughput recovered — proceed to 8×H full run (spec 025) |
| 4900–4999 | Partial recovery — investigate before promoting |
| < 4900 | Detach didn't eliminate overhead — kill learnable-α arc |

Quality gate: val@4000 must be within 0.003 of 021c (i.e., ≤ 1.1207) to confirm no regression from init-ones change.

## Config diff vs 021h

| var | 021h | 024 |
|---|---|---|
| Code | `5906820` (carry form, zeros init) | `613bc8e` (detached-lerp, ones init) |
| `WARMDOWN_FRAC` | default (0.75) | default (0.75) |

All other env vars identical to 021h.

## Code changes

**Commit `613bc8e` on `exp/recur-alpha-buffer`.** Diff vs `5906820`:

```python
# __init__: init ones (identity)
self.recur_alpha = nn.Parameter(
    torch.ones(h.num_loops, num_looped, dtype=torch.float32),
    requires_grad=True,
)

# All 4 blend sites (forward_logits enc/dec, forward_ttt enc/dec):
# From:
x = x_new + alpha * carry[i]
# First pass stored: carry[i] = x_new.detach()

# To:
x_before_det = x_before.detach()
x = x_before_det + alpha * (x_new - x_before_det)
# No carry dict needed
```

## Hardware ladder

**4×H100 mini — NA or JP, 4H wallclock cap.** This is a throughput + early quality signal run. Proceed to 8×H only if accept criteria pass.

## Seed plan

Seed 42 only (mini). 3-seed conditional on 8×H promoting.

## Run protocol

```bash
python -c "import brotli"

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 613bc8e

# Sanity verify
grep "torch.ones.*num_loops" train_gpt.py                   # must match
grep -c "x_before_det = x_before.detach()" train_gpt.py    # must be 4
grep -c "x_before_det + alpha" train_gpt.py                 # must be 4
grep "carry" train_gpt.py | grep -v "#"                     # must be empty

mkdir -p /workspace/runs/024-learnable-alpha-detached-lerp/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /workspace/runs/024-learnable-alpha-detached-lerp/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/024-learnable-alpha-detached-lerp/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_024_4h \
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
  > /workspace/runs/024-learnable-alpha-detached-lerp/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Note for JP pod:** replace `/workspace` with `/runpod` throughout.

## Artifacts

Standard: `train.log`, `diag_nvsmi.csv`. No checkpoint needed (mini run).

**Key signals to capture:**
- Total steps at wallclock cap (throughput comparison vs 021c: 5004 steps)
- `val_bpb` at step 4000 (quality comparison vs 021c: 1.1177)
- `recur_alpha: values=...` log lines — do alpha values drift from 1.0 at all?
- `layer_loop_enabled_at_step` — should be ~2200 as usual

## Stop-early criteria

- NaN/inf in train_loss → halt
- Step time > 1.5× 021c's → halt (something wrong with detach implementation)
- `layer_loop_enabled_at_step` outside [2000, 2400] → halt

## Cost estimate

~$3–4 (4×H100, 4H, mini only)

## Decision tree after 024

| Steps | val@4000 | Action |
|---|---|---|
| ≥ 5000 | ≤ 1.1207 | Throughput fixed + quality ok → spec 025 (8×H full run) |
| ≥ 5000 | > 1.1207 | Throughput fixed but quality regressed → investigate init |
| < 4900 | any | Detach didn't help → submit 021e 3-seed |

## Benchmarks

Primary comparison is throughput (steps at 4H cap) and early quality (val@4000). This run is a mini so no TTT; full-pipeline comparison deferred to spec 025.

### Throughput benchmark (steps at 4H wallclock cap, 4×H100)

| run | spec | α type | steps | Δ vs frozen |
|---|---|---|---|---|
| 019b ref | frozen constant | 5034 | — |
| 021c frozen param | frozen nn.Parameter | 5004 | -30 (-0.6%) |
| **024 detached-lerp** | **learnable, detached** | **target ≥ 5000** | **≤ -34** |
| (021h equivalent at 4H) | learnable, live | ~4900 est. | ~-134 (-2.7%) |

**Pass bar:** ≥ 5000 steps — matches frozen. **Fail bar:** < 4900 — overhead persists.

### Quality benchmark (val@4000, 4×H100)

| run | val@4000 |
|---|---|
| 019b ref | 1.1190 |
| 021c frozen | 1.1177 |
| 021h (8×H, comparable stage) | 1.1084 ← best learnable ever |
| **024 target** | **1.117–1.119** |

**Pass bar:** ≤ 1.1207 (within 0.003 of 021c). A val@4000 better than 1.1177 would be a bonus signal that learnable α with full-LR is finding a better basin than frozen.

### Full-pipeline context (8×H, for spec 025 planning)

| run | pre-quant EMA | post-TTT | steps |
|---|---|---|---|
| 021e (frozen, best frozen) | 1.06944 | **1.06622** | 4863 |
| 021h (learnable fp32) | 1.07043 | 1.06734 | 4780 |
| 017 (learnable, best pre-quant) | **1.06861** | 1.06733* | 4726* |
| #1736 target | — | 1.06610 | — |

*017 TTT was buggy; post-TTT not comparable.

If 024 recovers throughput (≥5000 steps at 4H), spec 025 would rerun on 8×H expecting pre-quant EMA near 017's 1.06861 — the hypothesis being throughput was what held learnable α back.

## Monitoring protocol

Poll `train.log` every 30s during the run. Key checkpoints:

### Phase 1 — pre-loop (steps 0 → ~2200)
| What to check | Expected | Action if wrong |
|---|---|---|
| `train_loss` decreasing | Smooth, ~9.0 → ~2.8 | NaN/stuck → halt |
| Step time (from log timestamps) | ~0.24s/step (matches 021c) | >0.35s → halt, check compile |

### Phase 2 — loop activation (~step 2200)
| What to check | Expected | Action if wrong |
|---|---|---|
| `layer_loop_enabled_at_step` in log | Step 2000–2400 | Outside range → halt |
| Step time after activation | Should NOT jump >1.5× pre-loop | >1.5× → detach may not be working |
| First `recur_alpha: values=...` line | All values near 1.0 | Any value > 2.0 or < 0 → flag |

### Phase 3 — post-activation (steps 2200 → 4000)
| What to check | Expected | Action if wrong |
|---|---|---|
| `4000/20000 val_bpb:` line | ≤ 1.1207 | > 1.13 → quality regression, flag |
| Alpha values drifting from 1.0 | Gradual drift OK, wild oscillation not | |value| > 3.0 → flag |

### Phase 4 — wallclock cap (~step 5000+)
| What to check | Expected | Action if wrong |
|---|---|---|
| Final step number | ≥ 5000 | < 4900 → throughput not fixed |
| `stopping_early: wallclock_cap` line | Present | If training ended for other reason → investigate |
| `pre-quantization post-ema val_bpb:` | ≤ 1.0700 | > 1.072 → quality issue |

### Every poll — also check
```bash
runpodctl pod list  # confirm pod running, cost/hr reasonable
```

## Open questions for executor

1. **Verify carry is gone.** `grep "carry" train_gpt.py | grep -v "#"` must return empty.
2. **Watch alpha values.** First `recur_alpha: values=...` log line should be all 1.0 at activation (~step 2200). Do they move meaningfully before wallclock cap?
3. **Step count at cap.** Primary signal. Log the final step number explicitly.
