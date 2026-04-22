# Spec 021 — recur_alpha as register_buffer (full 8×H100 submission validation)

**Slug:** `recur-alpha-buffer`
**Created:** 2026-04-21, updated 2026-04-22
**Status:** READY (full 8×H100 arm, post-021-on-4×H100)
**Links to:** `research/ideas/throughput-alpha-proxy-gap.md`

## Context: the 4×H100 predecessor and why we need 8×H100

021 was originally written for 8×H100 JP. At launch time JP and NE-1 8×H100 were dry; we substituted with 4×H100 NE-1 at 2× wallclock (1196s). That run (`runs/021-recur-alpha-buffer/seed_42/` @ commit `cb5cd78`) finished training cleanly:
- Final step 4736 (target range 4700-4800 ✓)
- Zero Type B compile stalls, zero post-val recompile clusters ✓
- Pre-quant post-EMA: **1.07095**

The pre-quant number was ~0.001 behind 019b's 1.06951 because of the **4×H100 batch-size tax** — per-GPU activation memory and collective patterns differ from 8×H100, costing ~0.001-0.002 bpb. This isn't a buffer-α failure; it's a hardware asymmetry. **GPTQ compression crashed on missing `brotli`**, so no post-TTT number for the 4×H100 run.

**Mechanism fully validated by 020b + 021-on-4×H100** — buffer-α has zero stalls at full scale. What's missing is the actual submission-comparable number on matched hardware.

This spec is the 8×H100 arm. Clean config, no instrumentation, same throughput schedule as 017/019/019b.

## Hypothesis

Buffer-α on 8×H100 at standard config should:
- Match or beat 019b's 1.06628 post-TTT (because no compile stalls → ~136 more steps than 019b)
- Linear-projection estimate: **post-TTT ~1.0591-1.066** (discounting optimism)
- If the projection holds → first submission candidate solidly beating #1736 (1.06610)

## Baseline and projection

Full comparison table:

| metric | #1736 | 008 | 017 | 019b | **021 projected** |
|---|---|---|---|---|---|
| final step | 4854 | 4828 | 4784 | 4716 | **~4852** |
| pre-quant post-EMA | 1.06907 | 1.06922 | 1.07083 | 1.06951 | **~1.06237** |
| post-quant (GPTQ int6) | 1.07847 | 1.08010 | — | 1.07877 | **~1.07163** |
| **post-TTT** | **1.06610** | 1.06728 | 1.06733 | 1.06628 | **~1.05914** |

Projection math: 019b's +136 step recovery (from stall elimination) × linear slope of −5.25e-5 bpb/step → −0.00714 pre-quant improvement → same quant/TTT stack.

**Honest discount range:** projections stack three optimistic assumptions (linear slope, full stall recovery, constant TTT gain). Even halving the expected improvement: post-TTT ~1.0627 — still beats #1736.

## Accept criteria

**Primary (post-TTT):**

| Post-TTT | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 by ≥0.002 | **3-seed confirmation → submission candidate** |
| (1.06400, 1.06610] | Beats #1736 but tighter | 3-seed to resolve |
| (1.06610, 1.06710] | Ties/misses #1736 by ≤0.001 | 3-seed; borderline — compare with 019b |
| (1.06710, 1.06910] | Inside gate but misses #1736 | Recur-α arc closed; pivot |
| > 1.06910 | Outside gate | Investigate |

**Secondary (throughput sanity):**

| Final step | Interpretation |
|---|---|
| ≥ 4820 | Matches projection; stall-free throughput confirmed at 8×H100 |
| 4750-4820 | Some variance but broadly on track |
| < 4750 | Unexpected slowdown; investigate |

## Config diff

| var | value | note |
|---|---|---|
| `RECUR_ALPHA_ENABLED` | `1` | buffer-α path |
| `ENABLE_LOOPING_AT` | `0.35` (default, omit) | standard activation — matches 017/019/019b |
| `MATRIX_LR` | `0.026` | same as 019b |
| `PHASED_TTT_ENABLED` | `1` | yes, we want post-TTT |
| `PHASED_TTT_PREFIX_DOCS` | `2000` | same as #1736 |
| `PHASED_TTT_NUM_PHASES` | `3` | same as #1736 |
| `THROUGHPUT_DIAG` | *unset* | **instrumentation OFF** — this is the submission run |

**No `MAX_WALLCLOCK_SECONDS` override.** Default 596s wallclock cap.

## Code changes

**Branch:** `exp/recur-alpha-buffer` forking from `4dd2d63` (017's commit).
**Commit:** **`cb5cd78`** — buffer-α only, no instrumentation. Clean.

25-LOC diff vs 017:
1. `self.recur_alpha = nn.Parameter(torch.ones(...))` → `self.register_buffer("recur_alpha", <017 endpoint tensor>)`.
2. Optimizer append guarded by `isinstance(..., nn.Parameter)` so a buffer is skipped.

017's endpoint table (baked in):
```python
[[1.078125, 1.2734375, 1.4296875],     # pass-2 L3, L4, L5
 [1.015625, 0.96484375, 0.83203125]]   # pass-3 L3, L4, L5
```

## Hardware ladder

**8×H100 JP preferred.** NE-1 acceptable if JP unavailable.

**Do NOT substitute 4×H100 again** — the 4×H100 run is already in hand (`runs/021-recur-alpha-buffer/seed_42/`) and its pre-quant was ~0.001 worse than 019b due to batch-size tax. Another 4×H100 run would replicate the tax without answering the submission question. If 8×H100 stays unavailable, either wait or pivot to post-hoc TTT recovery on the 4×H100 `final_model.pt`.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610** (clear beat or tie #1736).

## Inputs

- Data: CaseOps dataset, JP `/runpod/data/...` or NE-1 `/workspace/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

```bash
# Preflight: install brotli explicitly (both prior NE-1 runs hit ModuleNotFoundError)
pip install brotli

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout cb5cd78

mkdir -p /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42
mkdir -p /runpod/.torch_inductor_cache_021_8xh100

# nvsmi sidecar (thermal/clock telemetry, zero GPU overhead)
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021-recur-alpha-buffer-8xh100/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache_021_8xh100 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/021-recur-alpha-buffer-8xh100/seed_42/train.log 2>&1

kill $NVSMI_PID
```

Substitute `/workspace` for `/runpod` on NE-1.

## Checkpoints / artifacts to emit

- `final_model.pt` — post-EMA FP state dict (backup in case anything post-training crashes)
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, loss, TTT progress
- `diag_nvsmi.csv` — per-GPU per-second telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_buffer: true`, `recur_alpha_values_hardcoded`
- `notes.md`

## Stop-early criteria

- NaN/inf in train_loss → halt
- Step time > 2× spec 017 (> ~240ms) → halt
- `layer_loop_enabled_at_step` < 2000 or > 2300 → halt (activation didn't fire at ~frac 0.35 as expected)

## Cost estimate

| item | cost |
|---|---|
| 8×H100 × ~12 min (compile + 596s training + GPTQ + TTT) | ~$10 |
| Rsync + pod stop | ~$0.10 |
| **Single-seed total** | **~$10** |
| (Conditional) 3-seed × 2 additional runs | ~$20 |
| **If 3-seed promotes** | **~$30** total |

## Extra artifacts

None beyond standard. Instrumentation intentionally off for submission.

## Open questions for interview (execution)

1. **Brotli preflight is non-negotiable.** Both 020b (NE-1) and 021-4×H100 (NE-1) hit `ModuleNotFoundError: No module named 'brotli'` at `_compress()` after training completed, costing the submission artifact. Add `pip install brotli` to preflight before launch. Verify with `python -c "import brotli"` before `torchrun`.
2. **If JP 8-GPU unavailable:** wait for availability rather than substituting 4×H100 again. The existing 4×H100 run's `final_model.pt` can provide a post-hoc TTT number if needed (but with the batch-size tax).
3. **3-seed gating:** do NOT launch 3-seed if single-seed post-TTT misses the promote bucket (> 1.06710). Small chance of being worth it for reproducibility data, but submission window is tight (deadline 2026-04-30); prioritize pivot over cost.
