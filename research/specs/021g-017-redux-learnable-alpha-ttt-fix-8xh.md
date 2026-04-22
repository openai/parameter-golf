# Spec 021g — 017 redux: learnable α + TTT fix + algebraic form, 8×H100 JP

**Slug:** `017-redux-learnable-alpha-ttt-fix-8xh`
**Created:** 2026-04-22
**Status:** READY
**Links to:** `research/specs/017-recur-alpha-full.md`, `research/specs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh.md`, `research/evaluations/017-recur-alpha-full.md` (if exists)

## Context

**017's pre-quant post-EMA (1.06861) is the best of any 8H run we've produced in this session.** Frozen-α variants (019b at 1.06951, 021e at 1.06944, etc.) all land ~0.0008-0.0011 worse. The mechanism appears to be weights-α co-evolution during training producing a better checkpoint than frozen-from-step-0.

017's headline post-TTT (1.06733) was artificially bad because of the TTT α bug — α was applied during `forward_logits` (training) but NOT during `forward_ttt` (eval). TTT delta was only −0.01048 vs 019b's fixed −0.01249, leaving ~0.002 of TTT recovery on the table.

021g combines 017's learnable-α recipe with the 021e stack's two improvements (TTT α fix and algebraic blend form).

## Hypothesis

With:
- Learnable `nn.Parameter` α (init=1.0), like 017 → weights-α co-evolution → best pre-quant (~1.0686)
- TTT α fix from 931bd7c → full TTT delta (−0.01249 like 019b) instead of buggy −0.01048
- Algebraic blend form (x = x_before + α·(x_new − x_before)) → bf16 fusion consistent with 019b-submission

**Projected post-TTT: ~1.06532.** Decisive beat of #1736 (1.06610) by 0.00078 and 019b (1.06628) by 0.00096.

## Baseline

- **Primary:** 017 (commit `4dd2d63`) on 8H JP: pre-quant 1.06861, post-quant 1.07781, **post-TTT 1.06733 (buggy TTT)**.
- **Target:** #1736 post-TTT 1.06610.
- **Best prior:** 019b-original post-TTT 1.06628.

## Expected Δ

Credence distribution:

| Bucket | Probability | Predicted post-TTT |
|---|---|---|
| Clean beat | **50%** | 1.0640-1.0660 (decisive) |
| Marginal beat | 25% | 1.0655-1.0665 (ties #1736 within noise) |
| Parity | 15% | 1.0665-1.067 (marginal miss) |
| Regress | 10% | > 1.067 (017 advantage was pod luck) |

Higher credence than 021e because 017's pre-quant advantage is empirical (not projected), and we're fixing the TTT bug that was demonstrably costing 017 in the post-TTT measurement.

## Accept criteria

**Primary (post-TTT):**

| Post-TTT bpb | Bucket | Next action |
|---|---|---|
| ≤ 1.06400 | Clear beat #1736 by ≥ 0.002 | **3-seed 43/44 immediately on same pod** → submission |
| (1.06400, 1.06610] | Beats #1736 | 3-seed for confirmation |
| (1.06610, 1.06700] | Borderline | Compare to 021e; 3-seed if 021g > 021e |
| > 1.06700 | Disappointing | 017 pre-quant advantage didn't reproduce; pivot |

## Early-signal checkpoints (informational only)

Policy: let full pipeline complete regardless.

| Step 3000 train_loss | Bucket |
|---|---|
| ≤ 2.558 | Tracking 017-like or better trajectory |
| 2.558-2.565 | Marginal |
| > 2.565 | Tracking 019b-like (not 017 advantage) |

| Pre-quant post-EMA | Meaning |
|---|---|
| ≤ 1.069 | 017 advantage reproduced (best case) |
| 1.069-1.0695 | Tracks 019b/021e (marginal) |
| > 1.07 | Regressed |

## Config diff

Identical to 021e — same env block, same `RECUR_ALPHA_ENABLED=1`, `MATRIX_LR=0.026`, default `ENABLE_LOOPING_AT` (0.35), 596s wallclock.

## Code changes

**Branch:** `exp/recur-alpha-buffer`
**Commit:** **`fab6e7f`** (already pushed)

Stack on top of spec 021 base (`cb5cd78`):
1. `dc0b5f8`: α pass-3 L4 fix (doesn't matter — 021g's α isn't hardcoded)
2. `d070df3`: bf16 dtype
3. ~~`8b2d791`~~ / ~~`0ad5269`~~ (frozen container — replaced)
4. `931bd7c`: TTT α bug fix
5. `d761a22`: Algebraic blend form
6. `fab6e7f`: **Learnable Parameter α (requires_grad=True), init=1.0** ← the 017 revival

Code delta from 021f (0ad5269):
```python
# Was (021f, register_buffer with hardcoded 017 endpoint):
self.register_buffer("recur_alpha", _recur_alpha_017_endpoint)

# Now (021g, learnable ones-init):
self.recur_alpha = nn.Parameter(
    torch.ones(h.num_loops, num_looped, dtype=torch.bfloat16),
    requires_grad=True,
)
```

Optimizer guard already handles this case (checks `requires_grad` — now True → α joins scalar_params optimizer group).

## Hardware ladder

**8×H100 AP-JP-1.** Same pod as 021e preferred for pod-variance-matched comparison. Sequential after 021e completes.

**Do NOT substitute** other hardware.

## Seed plan

Seed 42 first. **3-seed (42/43/44) conditional on post-TTT ≤ 1.06610.**

## Run protocol

```bash
# Preflight
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout fab6e7f

# Sanity-verify all changes present
grep "nn.Parameter" train_gpt.py | grep recur_alpha       # must match
grep "torch.ones" train_gpt.py | grep -v "#"              # should include recur_alpha init
grep "requires_grad=True" train_gpt.py | grep -v "#"      # must have at least this one
grep "dtype=torch.bfloat16" train_gpt.py | head -1        # must match
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py  # must be 4

mkdir -p /runpod/runs/021g-017-redux-learnable-alpha-ttt-fix-8xh/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/021g-017-redux-learnable-alpha-ttt-fix-8xh/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/021g-017-redux-learnable-alpha-ttt-fix-8xh/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_021g_8h_jp \
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
  > /runpod/runs/021g-017-redux-learnable-alpha-ttt-fix-8xh/seed_42/train.log 2>&1

kill $NVSMI_PID
```

**Expected log patterns during training:**
- α values should start near 1.0 (from init)
- `recur_alpha: values=...` log entries should show α drifting after loop activation (~step 2150)
- α should converge toward 017-endpoint-like values (~[[1.08, 1.27, 1.43], [1.02, 0.97, 0.83]]) by step ~3500
- If α stays at 1.0s or diverges wildly, something's broken in the optimizer integration

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace, recur_alpha values
- `diag_nvsmi.csv`
- `final.json` — must include `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, `val_bpb_post_ttt`, `stopping_early_at_step`, `layer_loop_enabled_at_step`, `recur_alpha_is_learnable: true`, `recur_alpha_final_values` (for audit vs 017's endpoint)

## Stop-early criteria (hard safety)

- NaN/inf in train_loss → halt
- Step time > 2× 019b's → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt
- α divergence (any |value| > 3.0 after step 3000) → halt (sign that learnable α lost stability)

## Cost estimate

| item | cost |
|---|---|
| 8×H JP × ~25 min (compile + 596s train + GPTQ + TTT) | ~$10 |
| Conditional 3-seed × 2 additional on same pod | ~$20 |
| **If 3-seed promotes** | **~$30** |

## Dependencies

1. **021e completes first** (captured post-TTT, same-pod comparison available).
2. **JP 8×H stock** available at launch time.
3. **Brotli preflight confirmed.**

## Open questions for executor

1. **α should be in optimizer.** Verify at launch: log should show α values starting at 1.0 and drifting after step 2150. If α is stuck at 1.0 with grad_norm=0 past step 2200, the optimizer guard didn't include it correctly — halt and fix.
2. **α divergence guard.** If |α| > 3.0 at any point after step 3000, halt. Means learnable α is destabilizing.
3. **Same-pod comparison to 021e.** After both complete, the 021g vs 021e comparison shows whether learnable α advantage over frozen α is real at 8H (as it was at 017's original 8H).
4. **What if 021g ties or loses to 021e?** Means 017's pre-quant advantage was mostly pod luck, not mechanism. Buffer/Parameter/learnable containers are equivalent at scale once TTT fix is applied. Pivot.

## Relationship to other specs

- **021e** = frozen Parameter α (019b-equivalent post-TTT projection ~1.0661)
- **021f** = frozen register_buffer α (confirmatory to 021e)
- **021g** = this spec, learnable Parameter α (017-equivalent projection ~1.0653)

If 021g wins, **021e/021f become superseded**. If 021e wins decisively but 021g misses, shipping 021e 3-seed is the submission.

## What this does NOT test

- **α updating during TTT.** Current code freezes all base model params for TTT (`base_model.parameters().requires_grad_(False)`). If you want α to also update per-document during TTT, that's a follow-up 021h variant (keep α.requires_grad=True through the TTT freeze).
- **α=0 init (015 style).** If 021g wins and we want to squeeze another 0.0005, 015's ZERO-init was actually +0.0006 worse than 016 (ones-init) on 4000-val but 017 used ones-init. Not worth chasing unless 021g misses.

## Priority justification

021g is the **highest-EV remaining variant** because:
1. 017 had the best pre-quant of any 8H run — empirical, not projected.
2. The TTT bug cost 017 ~0.002 of TTT delta — we can recover that with 931bd7c's fix.
3. All other variables (container, dtype, blend form) have been tested and found marginal.
4. If 017's pre-quant advantage reproduces, we have a decisive beat of #1736.
5. Code change is minimal (3 lines) and well-motivated.
