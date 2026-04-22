# Spec 030 — 025b seed 314 + new TTT (correct PHASED_TTT + warm-start-A)

**Slug:** `025b-seed314-new-ttt`
**Created:** 2026-04-23
**Status:** READY
**Commit:** `c3a99b3` (exp/029-full-stack — 025b values, warm-start-A in TTT path)
**Links to:** `research/specs/026-cross-layer-carry-frozen-8xh.md`, `research/specs/028-ttt-only-026-seed42.md`

## Hypothesis

Spec 026 seed 42 (025b arch, 8×H) produced pre-quant EMA 1.06893, post-TTT 1.06582. Two
independent improvements have been validated since:

1. **Seed 314** is empirically better than seed 42 — dexhunter's #1769 shows seed 314 gives
   pre-quant EMA ~1.06637 vs seed 42's mediocre draw. Expected ~−0.002 on the float.

2. **Correct PHASED_TTT_ENABLED=3** — spec 026's 8×H command used `PHASED_TTT_ENABLED=1`
   (slow single-phase TTT), not the fast 3-phase path. Spec 028 showed the fast path at
   α=144/WD=1.0 reaches 1.06649 on 4×H vs the old path's 1.06724. The improvement on 8×H
   should be larger still.

3. **Warm-start-A** — commit c3a99b3 keeps LoRA A across TTT batch resets; validated in
   #1767/#1771, Δ ~−0.001.

Training is identical to 026 (025b arch, NUM_LOOPS=2, same frozen α/β). The only changes
are seed and the TTT path.

## Baselines

| run | pre-quant EMA | post-TTT | notes |
|---|---|---|---|
| 026 seed 42 (8×H, PHASED_TTT=1) | 1.06893 | 1.06582 | reference |
| 028 Run B (4×H, PHASED_TTT=3) | 1.06893 | 1.06649 | same float, fast path |
| #1769 seed 314 (dexhunter) | 1.06637 | 1.06357 | target to beat |
| #1769 5-seed mean | 1.06742 | 1.06453 | secondary benchmark |

## Expected delta

- Seed 314 vs seed 42: ~−0.002 pre-quant EMA (based on #1769 empirical data)
- PHASED_TTT=3 + warm-start-A vs old path: ~−0.001 post-TTT
- Projected post-TTT: **~1.060–1.062**

## Accept criteria

Benchmark: **#1769 seed 314** post-TTT 1.06357.

| post-TTT bpb | verdict | action |
|---|---|---|
| < 1.062 | Beats or matches #1769 seed 314 | Run seeds 2025 + 777 → 3-seed submission |
| [1.062, 1.065] | Matches #1769 mean; above their best | Run seeds 2025 + 777 to confirm mean |
| (1.065, 1.068] | Seed 314 didn't help or TTT still broken | Debug TTT path; check val@4000 |
| > 1.068 | Regression vs 026 | Kill |

Screen gate (4×H, no TTT): pre-quant EMA < 1.068.

## Config diff vs 026 seed 42 actual run

| env var | 026 seed 42 | spec 030 | note |
|---|---|---|---|
| `SEED` | 42 | **314** | better seed |
| `PHASED_TTT_ENABLED` | 1 (slow) | **3** (fast) | critical fix |
| `TTT_LORA_ALPHA` | 144 | 144 | same — 026 already had this |
| `TTT_WEIGHT_DECAY` | 1.0 | 1.0 | same — 026 already had this |
| Commit | `950af24` | **`c3a99b3`** | adds warm-start-A |

`NUM_LOOPS=2`, no `LOOP_DEPTH_UPGRADE_AT` — depth curriculum code in c3a99b3 is dormant.

## Hardware ladder

Mini: **skip** — training path at c3a99b3 with NUM_LOOPS=2 (no LOOP_DEPTH_UPGRADE_AT) is
equivalent to 026 seed 42. No new training logic is activated. Cite spec 026.

1. **4×H screen seed 314, no TTT** (~$4, ~25 min) — float gate only. If pre-quant EMA < 1.068, proceed.
2. **8×H full pipeline seed 314** (~$12, ~28 min) — main result with PHASED_TTT=3.
3. **Seeds 2025 + 777 at 8×H** (~$24) — conditional on seed 314 post-TTT ≤ 1.064.

## Run commands

### Screen (4×H, seed 314, no TTT)

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout c3a99b3

# Sanity verify — 025b values + warm-start-A
grep "1.5973426" train_gpt.py           # 025b beta[L3]
grep "\-0.34765625" train_gpt.py        # 025b alpha[L4,L4] self-subtract
grep "warm-start A" train_gpt.py        # warm-start-A must be present
# Confirm depth curriculum is dormant: must NOT see LOOP_DEPTH_UPGRADE_AT fire without us setting it
grep "loop_depth_upgrade_at" train_gpt.py  # code present but will not activate

mkdir -p /runpod/runs/030-025b-seed314-new-ttt/screen_seed_314
mkdir -p /tmp/torch_inductor_cache_030_screen

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/030-025b-seed314-new-ttt/screen_seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_030_screen \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=4 train_gpt.py \
  > /runpod/runs/030-025b-seed314-new-ttt/screen_seed_314/train.log 2>&1
```

Screen pass/fail:

| metric | 026 seed 42 | target | action if missed |
|---|---|---|---|
| val@4000 | 1.1128 | ≤ 1.110 | note; float is the gate |
| pre-quant EMA | 1.06893 | < 1.068 | try seed 2025 screen |

### Full pipeline (8×H, seed 314)

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout c3a99b3

# Sanity verify
grep "1.5973426" train_gpt.py           # 025b beta[L3]
grep "\-0.34765625" train_gpt.py        # 025b alpha[L4,L4]
grep "warm-start A" train_gpt.py        # warm-start-A
grep "_scale = alpha / rank" train_gpt.py

mkdir -p /runpod/runs/030-025b-seed314-new-ttt/seed_314
mkdir -p /tmp/torch_inductor_cache_030_8h

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/030-025b-seed314-new-ttt/seed_314/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/030-025b-seed314-new-ttt/seed_314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_030_8h \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=3 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
TRAIN_LOG_EVERY=100 \
SEED=314 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/030-025b-seed314-new-ttt/seed_314/train.log 2>&1

kill $NVSMI_PID
```

## Stop-early criteria

- NaN/inf in train_loss → halt
- Screen pre-quant EMA ≥ 1.068 → stop, try seed 2025 before 8×H
- `loop_depth:upgraded` fires unexpectedly during training → halt (depth curriculum should be dormant)
- Step time > 2.0 s/step at 8×H → flag

## Cost estimate

| item | cost |
|---|---|
| 4×H screen seed 314, no TTT (~25 min) | ~$4 |
| 8×H seed 314 full pipeline (~28 min) | ~$12 |
| Seeds 2025 + 777 at 8×H (conditional) | ~$24 |
| **Total if 3-seed** | **~$40** |

## Open questions for executor interview

1. **JP stock?** Provision with `--template-id y5cejece4j`. Do not use other regions.
2. **c3a99b3 on fork remote?** Run `git ls-remote fork exp/029-full-stack` to confirm before launch.
3. **PHASED_TTT_ENABLED=3 confirmed** — screen uses `=0` (training only); 8×H uses `=3`.
