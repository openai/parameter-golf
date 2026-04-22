# Spec 029 — Full stack: 025b frozen + LoRA warm-start-A + depth curriculum

**Slug:** `full-stack-025b`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/029-full-stack`
**Commit:** `c3a99b3`
**Links to:** `research/specs/026-cross-layer-carry-frozen-8xh.md`, `research/specs/027-lora-warmstart-depth-curriculum.md`

## Hypothesis

Stack all three validated levers on the clean 025b shared-frozen baseline:

1. **025b cross-layer carry frozen** — L4 self-subtract, L5 aggregates L3+L4 (Δ −0.0098 at val@4000 vs 021c). Shared indexing, 1D beta, 2D alpha. Better than 025c.
2. **LoRA warm-start-A** — keep A across TTT batch resets, scale by alpha/rank. Δ ~−0.001. Validated by #1767/#1771.
3. **Depth curriculum** — training phases 1→3→4 (depth upgrades at 67%), eval/TTT always at depth=4. Δ ~−0.0004 to −0.001. Validated by #1756/#1771.
4. **Seed 314** — dexhunter's best seed in #1769 (float 1.06637, post-TTT 1.06357). Seed 42 is a known mediocre seed.

Projected post-TTT: **~1.060–1.063** — beats #1769 mean (1.06453) and potentially their best seed (1.06357).

## Baselines

| run | val@4000 | pre-quant EMA | post-TTT |
|---|---|---|---|
| 025b seed 42 (4×H) | 1.1079 | 1.06917 | — |
| 026 seed 42 (8×H, 025b only) | 1.1128 | 1.06893 | 1.06582 |
| #1769 seed 314 (8×H) | — | 1.06637 | **1.06357** |
| #1769 mean 5-seed | — | 1.06742 | **1.06453** |

## Accept criteria

Benchmark: **#1769** (5-seed mean 1.06453, seed 314 best = 1.06357).

| post-TTT bpb | Bucket | Action |
|---|---|---|
| < 1.060 | Clear new SOTA, beats #1769 seed 314 | Run seeds 2025 + 777 → 3-seed submission |
| [1.060, 1.064] | Matches or beats #1769 mean | Run seeds 2025 + 777 to confirm |
| (1.064, 1.068] | Depth curriculum not contributing as expected | Check val@4000; compare to 026 seed 314 |
| > 1.068 | Regression | Kill; debug curriculum interaction |

Mini pass/fail:
- `loop_depth:upgraded` fires at ~20% (LOOP_DEPTH_UPGRADE_AT=0.20 override in mini)
- `layer_loop:enabled` fires at ~10% (ENABLE_LOOPING_AT=0.10 override)
- No NaN; throughput ≥ 600 tok/s on 2×H

4×H screen pass/fail (seed 314):
- pre-quant EMA < 1.068 (beats 025b seed 42 baseline of 1.06917)
- val@4000 ≤ 1.108

## Config diff vs 026 seed 42

| env var | 026 seed 42 | spec 029 |
|---|---|---|
| `NUM_LOOPS` | 2 | **3** |
| `LOOP_DEPTH_UPGRADE_AT` | (absent) | **0.67** |
| `TTT_LORA_ALPHA` | (absent, default 96) | **144** |
| `TTT_WEIGHT_DECAY` | 0.5 | **1.0** |
| `SEED` | 42 | **314** |

Commit `75722d3` vs `950af24` (026 seed 42): adds LoRA warm-start-A + depth curriculum on top of 025b. No 025c changes.

## Hardware ladder

1. **2×H mini** (~$3, ~25 min) — validates depth curriculum fires with 025b architecture. Required (training-path change).
2. **4×H screen seed 314** (~$4) — float check, no TTT. Pass if pre-quant EMA < 1.068.
3. **8×H full pipeline seed 314** (~$12) — when available.
4. **Seeds 2025 + 777** (~$24) — conditional on seed 314 post-TTT ≤ 1.064.

## Run commands

### Mini (2×H JP)

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 75722d3

# Sanity verify
grep "1.5973426" train_gpt.py                              # beta[L3] 025b value
grep "\-0.34765625" train_gpt.py                           # alpha[L4,L4] 025b self-subtract
grep -c "recur_beta\[local_idx\]" train_gpt.py             # must be 4 (shared indexing)
grep -c "recur_alpha\[local_idx, j\]" train_gpt.py         # must be 4
grep "loop_depth_upgrade_at" train_gpt.py                  # must be present
grep "loop_depth:upgraded" train_gpt.py                    # must be present
grep "warm-start A" train_gpt.py                           # must be present
grep "_scale = alpha / rank" train_gpt.py                  # must be present
grep "_decoder_alpha_info_int" train_gpt.py                # must be present (curriculum fix)

mkdir -p /runpod/runs/029-full-stack-025b/mini_seed_314
mkdir -p /tmp/torch_inductor_cache_029_mini

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/029-full-stack-025b/mini_seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_029_mini \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=3 LOOP_DEPTH_UPGRADE_AT=0.20 ENABLE_LOOPING_AT=0.10 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /runpod/runs/029-full-stack-025b/mini_seed_314/train.log 2>&1
```

**Mini checks:**
1. `layer_loop:enabled` fires at ~10%
2. `loop_depth:upgraded` fires at ~20% — **required; fail spec if absent**
3. `loop_warmup:depth_upgraded` fires during startup warmup (before step 0)
4. No NaN
5. No mid-run recompiles — all 3 graph states pre-warmed before step 0; `TORCH_LOGS=recompiles` should show 0 recompiles after warmup

### 4×H screen (seed 314, no TTT)

```bash
mkdir -p /runpod/runs/029-full-stack-025b/screen_seed_314
mkdir -p /tmp/torch_inductor_cache_029_screen

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/029-full-stack-025b/screen_seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_029_screen \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=3 LOOP_DEPTH_UPGRADE_AT=0.67 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /runpod/runs/029-full-stack-025b/screen_seed_314/train.log 2>&1
```

### 8×H full pipeline (seed 314)

```bash
mkdir -p /runpod/runs/029-full-stack-025b/seed_314
mkdir -p /tmp/torch_inductor_cache_029_8h

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/029-full-stack-025b/seed_314/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/029-full-stack-025b/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_029_8h \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=3 LOOP_DEPTH_UPGRADE_AT=0.67 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/029-full-stack-025b/seed_314/train.log 2>&1

kill $NVSMI_PID
```

## Stop-early criteria

- NaN/inf in train_loss → halt
- `loop_depth:upgraded` absent by step 2500 (75% of training) on 8×H → halt
- Step time > 2.0 s/step at 8×H (depth=4 adds ~10% compute; normal ~1.3 s/step)
- `layer_loop:enabled` fires without subsequent `loop_depth:upgraded` by step 2500 → halt

## Cost estimate

| item | cost |
|---|---|
| 2×H mini (~25 min) | ~$3 |
| 4×H screen seed 314 (~25 min) | ~$4 |
| 8×H seed 314 full pipeline (~28 min) | ~$12 |
| Conditional 8×H seeds 2025 + 777 | ~$24 |
| **Max total** | **~$43** |

## Open questions for executor interview

1. **JP stock?** Provision with `--template-id y5cejece4j`. Do not use other regions.
2. **Recompile count:** expect at most 3 events (initial, loop activation, depth upgrade). More than 5 → halt and report.
3. **Monitoring cadence?** Ask user: poll every 30s or leave to notification?
