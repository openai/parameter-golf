# Spec 021d — Recur-α as nn.Parameter(requires_grad=False), 8×H100 JP single-seed

**Slug:** `recur-alpha-param-frozen-8xh`
**Created:** 2026-04-22
**Status:** READY
**Links to:** `research/specs/021c-recur-alpha-param-frozen-mini.md`, `research/specs/021b-recur-alpha-buffer-bf16.md`, `research/specs/021-recur-alpha-buffer.md`, `research/evaluations/021-recur-alpha-buffer.md`

## Context

Buffer-α debugging arc on 8×H100 has tested three variants sequentially, each closing some of the per-step train-loss gap vs 019b:
- `cb5cd78` buffer fp32, buggy α: +0.010-0.012 gap
- `dc0b5f8` buffer fp32, α fix: +0.006-0.011 (α-typo wasn't the cause)
- `d070df3` buffer bf16, α fix, dropped cast: +0.002-0.007 (halved the gap)

Final mechanism candidate: **nn.Parameter(requires_grad=False) replacing register_buffer.** Inductor may treat Parameter tensors as compile-time-constant-like (parameters change slowly via optimizer.step; buffers are treated as potentially-mutable runtime inputs). If so, frozen Parameter enables the const-folding that literal-α gets, closing the residual gap.

**4×H100 NE-1 mini (spec 021c) Arm B** of this commit matched Arm A (019b-literal) exactly on training loss (Δ ±0.003, jitter-level) with ~1% slower tok/s. This is the expected outcome of "mechanism not broken" at a scale where literal-α doesn't have throughput chaos. The real test — whether Parameter+bf16 wins at 8×H where literal-α has ~10% tok/s σ from Type B compile stalls — can only be answered by promoting to 8×H.

## Hypothesis

At 8×H, Parameter+bf16 α:
1. **Eliminates Type B compile stalls** (buffer-like throughput profile → +160 steps over 019b).
2. **Const-folds in Inductor** → fused blend kernel → tracks 019b per-step loss.

Combined result: pre-quant post-EMA gains ~0.007 from extra steps over 019b's trajectory; post-TTT projection ~1.058-1.063.

## Baseline

- **Primary:** 019b @ commit `e93d77d` (post-TTT **1.06628**, step 4716, pre-quant 1.06951).
- **Target to beat:** #1736 post-TTT **1.06610**.
- **Prior 8H buffer-α variants:** 021-buggy 1.06900, 021-fix and 021-bf16 (halfway closures).

## Expected Δ

Credence distribution based on 4×H data + prior 8×H trends:

| Bucket | Probability | Predicted post-TTT |
|---|---|---|
| Clean beat | 30% | 1.058-1.063 |
| Marginal beat | 45% | 1.064-1.067 |
| Arc closes | 25% | 1.068-1.070 |

## Accept criteria

**Primary (post-TTT):**

| Post-TTT bpb | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 by ≥ 0.002 | **3-seed 43/44 on same pod/commit → submission candidate** |
| (1.06400, 1.06610] | Tight beat | 3-seed |
| (1.06610, 1.06710] | Borderline (tie/miss by ≤ 0.001) | Compare to 019b; may skip 3-seed and pivot to 019b 3-seed replication |
| > 1.06710 | Arc closed | Shelve buffer-α arc; pivot to 019b seeds 43/44 |

**Secondary (throughput sanity):**

| Final step | Interpretation |
|---|---|
| ≥ 4850 | Stall-free throughput confirmed (like 021-buggy's 4883) |
| 4720-4850 | Parameter didn't fully eliminate stalls |
| < 4720 | Throughput regressed below 019b — unexpected |

## Early-signal checkpoints (informational ONLY, NOT abort criteria)

Policy: **let the full pipeline complete regardless of mid-training signal.** These thresholds exist for expectation calibration and pre-planning the next move (3-seed launch vs pivot), not for early abort. The post-TTT number is the submission-grade answer; the ~$2 saved by aborting isn't worth losing the definitive data point.

| Step 3000 train_loss | Likely bucket |
|---|---|
| ≤ 2.565 | Clean beat (30%) |
| 2.565-2.570 | Marginal (45%) |
| > 2.570 | Arc closes (25%) |

| Step 4000 val_bpb | Meaning |
|---|---|
| ≤ 1.109 | Good signal |
| 1.109-1.112 | Marginal |
| > 1.112 | Off-track |

## Config diff

Identical to spec 021 — same env block, same `RECUR_ALPHA_ENABLED=1`, `MATRIX_LR=0.026`, `ENABLE_LOOPING_AT` default (0.35), default 596s wallclock cap.

Only change is the code commit.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`8b2d791`** (already pushed)

Cumulative diff vs spec 021's original commit `cb5cd78`:
1. `dc0b5f8`: pass-3 L4 α `0.96484375` → `0.97265625` (matches 017 endpoint).
2. `d070df3`: buffer dtype fp32 → bfloat16; drop `.to(x_new.dtype)` cast.
3. `8b2d791`: `register_buffer("recur_alpha", ...)` → `nn.Parameter(..., requires_grad=False)`; optimizer guard tightened to also check `requires_grad`.

## Hardware ladder

**8×H100 AP-JP-1 required.** All 8×H reference runs (019b, 021-family) are on JP — region-matched comparison. JP 8×H stock was "Low" at last probe; provision with ceiling price ~$24/hr.

**Do NOT substitute.** If JP unavailable, STOP and ask.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610** (clear beat or tie #1736). Same pod for all three seeds to share inductor cache and avoid re-lottery risk.

## Inputs

- Data: CaseOps dataset, JP `/runpod/data/...`
- Tokenizer: `fineweb_8192_bpe.model` bundled
- Hotstart: none

## Run protocol

```bash
# Preflight (non-negotiable — prior runs lost submission artifact to missing brotli)
pip install --break-system-packages brotli
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 8b2d791

# Sanity-verify
grep "nn.Parameter" train_gpt.py | grep recur_alpha     # must match
grep "dtype=torch.bfloat16" train_gpt.py | head -1      # must match
grep "\.to(x_new\.dtype)" train_gpt.py                  # must return nothing
grep "0.97265625" train_gpt.py                          # must match (pass-3 L4 fix)

mkdir -p /runpod/runs/021d-recur-alpha-param-frozen-8xh/seed_42
mkdir -p /tmp/torch_inductor_cache_021d_8h_jp

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021d-recur-alpha-param-frozen-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021d-recur-alpha-param-frozen-8xh/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_021d_8h_jp \
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
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/021d-recur-alpha-param-frozen-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Checkpoints / artifacts to emit

- `final_model.pt` — post-EMA FP state dict (backup)
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU per-second telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_parameter: true`, `recur_alpha_requires_grad: false`, `recur_alpha_dtype: "bfloat16"`, `recur_alpha_values_hardcoded`

## Stop-early criteria (hard safety — NOT the soft thresholds above)

- NaN/inf in train_loss → halt
- Step time > 2× 019b's (> ~240ms) → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt (activation didn't fire at frac 0.35)

## Cost estimate

| item | cost |
|---|---|
| 8×H JP × ~25 min (compile + 596s training + GPTQ + TTT) | ~$10 |
| Rsync + pod stop | ~$0.20 |
| **Single-seed total** | **~$10** |
| (Conditional) 3-seed × 2 additional on same pod | ~$20 |
| **Seed 42 + seeds 43/44 if promoted** | **~$30** |

## Extra artifacts

None beyond standard. Instrumentation intentionally off for submission-grade run.

## Open questions for executor interview

1. **JP 8×H availability.** If dry, STOP and ask — do NOT substitute with 4×H or NE-1. 4×H loses the throughput-chaos regime we're testing the fix against; NE-1 was dry at last probe.
2. **Inductor cache location.** Spec sets `/tmp/torch_inductor_cache_021d_8h_jp` (tmpfs-backed) per memory note `feedback_inductor_cache_on_tmp`. Do NOT set to `/runpod/...` (NFS FUSE → stale-file-handle race).
3. **Brotli preflight.** Non-negotiable. Verify `python -c "import brotli"` succeeds BEFORE launch.
4. **Monitor interview:** 30-second polling on train.log. Surface train_loss at step 3000 and val_bpb at step 4000 as information-only checkpoints (see "Early-signal checkpoints" — NOT abort criteria).
5. **If the 8H JP pod variance is bad (e.g., throughput is <6M tok/s post-loop)** — still run to completion. Variance is noise we absorb; no retry on same commit/seed.
6. **Parallel 019b 3-seed on separate 8×H JP pod** — if stock allows, launch simultaneously. Insurance submission regardless of 021d outcome. Separate spec forthcoming.

## What this does NOT test

- TTT-specific mechanisms (depth, LoRA bias) — that's specs 022/023.
- Non-seed-42 α basin behavior — tested only if 021d single-seed promotes to 3-seed.
- Longer wallclock / tighter warmdown variants — out of scope.
