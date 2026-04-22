# Spec 028 — TTT-only: 026 screen seed 314 float + new TTT settings

**Slug:** `ttt-only-026-screen-seed314`
**Created:** 2026-04-23
**Status:** READY
**Branch:** `exp/026-cross-layer-carry-frozen-8xh` (commit `d70888f`)
**No training code change** — uses `spinquant_hotstart.py` in baseline mode.

## Hypothesis

The 026 screen seed 314 float (pre-quant EMA 1.06770, 025c architecture, 4×H 1200s) was
run with TTT disabled. Re-running GPTQ + phased TTT on that float using the improved TTT
settings (TTT_LORA_ALPHA=144, TTT_WEIGHT_DECAY=1.0, warm-start-A already in d70888f)
should produce a post-TTT score directly comparable to spec 029's 4×H screen — but at
~$1 instead of ~$6.

Expected post-TTT: ~1.062–1.065 (seed 314 float 1.06770, TTT gain ~−0.012 per spec 026
seed 42 baseline, new settings add ~−0.001 on top).

## Baselines

| run | pre-quant EMA | post-TTT |
|---|---|---|
| 026 seed 42 (8×H, old TTT α=96 WD=0.5) | 1.06893 | 1.06582 |
| 026 screen seed 314 (4×H, no TTT) | 1.06770 | — ← this spec fills this |
| #1769 seed 314 | 1.06637 | 1.06357 |
| #1769 5-seed mean | 1.06742 | 1.06453 |

## Accept criteria

| post-TTT bpb | Verdict |
|---|---|
| ≤ 1.062 | Clear new SOTA signal — proceed to spec 029 8×H immediately |
| (1.062, 1.064] | Matches or beats #1769 mean — spec 029 4×H should confirm |
| (1.064, 1.066] | TTT settings helping but seed 314 float is limiting — check gap to #1769 |
| > 1.066 | Regression vs spec 026 seed 42 — debug TTT_LORA_ALPHA / TTT_WEIGHT_DECAY |

## What this isolates

- **Seed 314 post-TTT** — we only have seed 42 post-TTT from 026; this fills the seed 314 gap
- **New TTT settings** (α=144, WD=1.0) on 025c architecture — contribution isolated from training changes
- **No depth curriculum, no 025b change** — those are tested in spec 029

## Config vs 026 seed 42 TTT

| param | 026 seed 42 | spec 028 |
|---|---|---|
| float checkpoint | 026/seed_42 (025c, seed 42, 8×H) | **026/screen_seed_314 (025c, seed 314, 4×H)** |
| `TTT_LORA_ALPHA` | 96 (default) | **144** |
| `TTT_WEIGHT_DECAY` | 0.5 | **1.0** |
| `TTT_LORA_WARM_START_A` | yes (in d70888f) | yes |

## Float checkpoint location (JP volume)

```
/runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314/final_model.pt
```

Confirmed present in train.log config block:
```
model_path: /runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314/final_model.pt
```

## Hardware

1×2×H JP (~$1, ~15 min) — GPTQ + phased TTT only, no training.

## Run command

```bash
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout d70888f

# Sanity verify
grep "TTT_LORA_ALPHA" train_gpt.py                         # must be present
grep "warm-start A\|warm_start_a\|keep.*A\|zero.*B" train_gpt.py  # warm-start A present
grep "0.9453125\|pass_off, local_idx" train_gpt.py         # 025c constants (per-pass frozen)

mkdir -p /runpod/runs/028-ttt-only-026-screen-seed314
mkdir -p /tmp/torch_inductor_cache_028

SPINQUANT_MODE=baseline \
HOTSTART_FP_CKPT=/runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314/final_model.pt \
ARTIFACT_DIR=/runpod/runs/028-ttt-only-026-screen-seed314 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_028 \
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
SEED=314 \
torchrun --standalone --nproc_per_node=2 spinquant_hotstart.py \
  > /runpod/runs/028-ttt-only-026-screen-seed314/ttt.log 2>&1
```

**Key checks after run:**
1. `diagnostic pre-quantization` bpb ≈ 1.06770 (matches original screen run — confirms same float loaded)
2. `quantized_ttt_phased val_bpb` — the gate number
3. Submission size ≤ 16,000,000 bytes

## Stop-early criteria

- NaN in TTT loss → halt
- `diagnostic pre-quantization` bpb differs from 1.06770 by > 0.001 → wrong checkpoint loaded, halt

## Cost estimate

| item | cost |
|---|---|
| 2×H JP TTT-only (~15 min) | ~$1 |
| **Total** | **~$1** |

## Open questions for executor interview

1. **JP volume mounted?** Verify `/runpod/runs/026-cross-layer-carry-frozen-8xh/screen_seed_314/final_model.pt` exists before launching.
2. **NUM_LOOPS=2** — the 026 screen ran with NUM_LOOPS=2 (026 config). Verify from the screen run's train.log config block.
