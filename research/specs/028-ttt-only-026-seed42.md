# Spec 028 — TTT-only: 026 seed 42 float, old vs new TTT settings

**Slug:** `ttt-only-026-seed42`
**Created:** 2026-04-23
**Status:** READY
**Float checkpoint:** `026/seed_42/final_model.pt` on JP volume (025b arch, 8×H trained, commit `950af24`)

## Hypothesis

The 026 seed 42 float (025b arch, pre-quant EMA 1.06893) was TTT'd inline during the 8×H training
pipeline with old settings (α=96, WD=0.5) → post-TTT 1.06582. Running it standalone via
`spinquant_hotstart.py` on 4×H should reproduce that result (run A), then the new settings
(α=144, WD=1.0 + LoRA warm-start-A) should improve it (run B).

This isolates the TTT settings improvement in pure form: same float, same architecture, only
TTT hyperparams change.

## Baselines

| run | pre-quant EMA | post-TTT | TTT settings |
|---|---|---|---|
| 026 seed 42 inline (8×H, old) | 1.06893 | **1.06582** | α=96, WD=0.5, no warm-start-A |
| #1769 5-seed mean | 1.06742 | 1.06453 | unknown |
| #1769 seed 314 best | 1.06637 | 1.06357 | unknown |

## Two runs

### Run A — old TTT settings (sanity check)

Reproduce the inline 026 TTT result via standalone spinquant_hotstart.py.

- **Commit:** `950af24` (exp/026-cross-layer-carry-frozen-8xh — original 026 seed 42, no warm-start-A)
- **TTT:** `TTT_LORA_ALPHA=96 TTT_WEIGHT_DECAY=0.5`
- **Expected:** ~1.06582 ± 0.0003 (standalone vs inline may differ slightly)

### Run B — new TTT settings

- **Commit:** `c3a99b3` (exp/029-full-stack — 025b + LoRA warm-start-A)
- **TTT:** `TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0`
- **Expected:** < 1.065 (new settings + warm-start-A should improve by ~0.001–0.002)

## Accept criteria

| post-TTT bpb (run B) | verdict |
|---|---|
| < 1.063 | Strong TTT gain — new settings clearly worth it |
| [1.063, 1.065] | Modest gain vs run A — confirm direction |
| ≥ 1.065 | No improvement — debug α/WD interaction |

Run A fail condition: if run A lands > 1.062 or < 1.068 (outside ±0.003 of 1.06582) → pipeline
difference is larger than expected, flag before trusting run B.

## Float checkpoint location (JP volume)

```
/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt
```

Confirmed in `final.json`: pre_quant_ema_val_bpb = 1.06893.

## Hardware

2 × (1×4×H JP) — run A then run B sequentially on the same pod. ~$2 each, ~$4 total, ~15 min each.

## Run commands

### Run A (old TTT, sanity check)

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 950af24

# Sanity verify
grep "1.5973426" train_gpt.py          # 025b beta[L3]
grep "TTT_LORA_ALPHA\|ttt_lora_alpha" train_gpt.py  # must be present

mkdir -p /runpod/runs/028-ttt-only-026-seed42/run_a_old_ttt
mkdir -p /tmp/torch_inductor_cache_028a

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt \
ARTIFACT_DIR=/runpod/runs/028-ttt-only-026-seed42/run_a_old_ttt \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_028a \
DATA_DIR=/runpod/data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
TTT_LORA_ALPHA=96 TTT_WEIGHT_DECAY=0.5 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /runpod/runs/028-ttt-only-026-seed42/run_a_old_ttt/ttt.log 2>&1
```

### Run B (new TTT settings)

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout c3a99b3

# Sanity verify
grep "1.5973426" train_gpt.py          # 025b beta[L3]
grep "warm-start A\|warm_start_a" train_gpt.py  # warm-start-A must be present

mkdir -p /runpod/runs/028-ttt-only-026-seed42/run_b_new_ttt
mkdir -p /tmp/torch_inductor_cache_028b

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt \
ARTIFACT_DIR=/runpod/runs/028-ttt-only-026-seed42/run_b_new_ttt \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_028b \
DATA_DIR=/runpod/data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
NUM_LOOPS=2 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
GPTQ_RESERVE_SECONDS=0 GPTQ_CALIBRATION_BATCHES=16 \
SEED=42 \
torchrun --standalone --nproc_per_node=4 spinquant_hotstart.py \
  > /runpod/runs/028-ttt-only-026-seed42/run_b_new_ttt/ttt.log 2>&1
```

**Key checks after each run:**
1. `diagnostic pre-quantization` bpb ≈ 1.06893 (same float loaded both runs)
2. `quantized_ttt_phased val_bpb` — the gate number
3. Submission size ≤ 16,000,000 bytes

## Stop-early criteria

- NaN in TTT loss → halt
- `diagnostic pre-quantization` bpb differs from 1.06893 by > 0.001 → wrong checkpoint, halt
- Run A post-TTT outside [1.062, 1.068] → flag to user before starting run B

## Cost estimate

| item | cost |
|---|---|
| Run A: 4×H JP TTT-only (~15 min) | ~$2 |
| Run B: 4×H JP TTT-only (~15 min) | ~$2 |
| **Total** | **~$4** |

## Open questions for executor interview

1. **JP volume mounted?** Verify `/runpod/runs/026-cross-layer-carry-frozen-8xh/seed_42/final_model.pt` exists before launching.
2. **Run sequentially on same pod** — no need to stop/restart between A and B.
