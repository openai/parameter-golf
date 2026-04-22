# Spec 024 — Learnable α with delayed warmdown (loop-before-warmdown fix)

**Slug:** `learnable-alpha-warmdown`
**Created:** 2026-04-22
**Status:** SHELVED — 2026-04-22. Superseded by `024-learnable-alpha-detached-lerp.md`. Warmdown delay is a valid idea but deprioritised in favour of fixing the throughput overhead first. Revisit post-deadline if detached-lerp arc succeeds.
**Links to:** `research/specs/021h-learnable-alpha-fp32-8xh.md`, `research/specs/021e-recur-alpha-param-bf16-algebraic-ttt-fix-8xh.md`

## Context

In all prior learnable-α runs (015–017, 021g, 021h), warmdown starts at training frac=0.25 (step ~1216) but the recurrence loop doesn't activate until frac=0.35 (step ~1702). **α never sees full LR.** It starts learning into an already-decaying schedule from its very first gradient update.

021h showed the best val@4000 of the entire 021 family (1.1084) — α was genuinely learning better mid-training representations. But its pre-quant EMA (1.07043) was the worst, because warmdown had already been compressing LR for ~500 steps before α even activated.

## Hypothesis

Delay warmdown to WARMDOWN_FRAC=0.60 so that warmdown starts at frac=0.40 (step ~1920), giving the loop ~240 steps at full LR after activation. α gets a proper exploration phase at full learning rate before decay begins, enabling it to reach a better basin than prior learnable runs.

**Expected results:**
- Pre-quant EMA closer to 017's 1.06861 than 021h's 1.07043
- Post-TTT: beats 021e (1.06622) — best of frozen-α arc — by meaningful margin

## Baseline

| run | pre-quant EMA | post-TTT |
|---|---|---|
| 021e (frozen α, WARMDOWN_FRAC=0.75) | 1.06944 | **1.06622** ← best |
| 021h (learnable fp32 α, WARMDOWN_FRAC=0.75) | 1.07043 | 1.06734 |
| 017 (learnable α, manual-add, WARMDOWN_FRAC=0.75) | **1.06861** | 1.06733 (buggy TTT) |
| #1736 target | — | 1.06610 |

## Expected Δ

| Bucket | Probability | Post-TTT |
|---|---|---|
| Clear beat (warmdown fix unlocks 017-like pre-quant) | 40% | ≤ 1.0655 |
| Marginal beat (better than 021e, misses 017 basin) | 35% | 1.0655–1.0662 |
| Ties 021e (warmdown delay doesn't help α enough) | 20% | 1.0662–1.0668 |
| Regression | 5% | > 1.0668 |

## Accept criteria

| Post-TTT bpb | Decision |
|---|---|
| ≤ 1.06400 | Clear beat — 3-seed immediately |
| (1.06400, 1.06610] | Beats #1736 — 3-seed for confirmation |
| (1.06610, 1.06622] | Ties or marginal beat of 021e — run spec 025 (frozen α + same WARMDOWN_FRAC) to isolate |
| > 1.06622 | Warmdown delay didn't help — submit 021e 3-seed |

**Scientific signal — did delaying warmdown help α?**

| α at step ~3500 vs 017 endpoint | Interpretation |
|---|---|
| Matches 017 within 0.01 | Warmdown ordering was the root cause |
| Still offset by 0.02+ | Algebraic blend landscape is the fundamental blocker |

## Config diff

Only change from 021h:

| var | 021h | 024 |
|---|---|---|
| `WARMDOWN_FRAC` | not set (default 0.75) | **0.60** |

All other env vars identical to 021h.

## Code changes

**No code change.** Same commit as 021h: **`5906820`** on branch `exp/recur-alpha-buffer`.

Stack: fp32 learnable α + TTT α fix + algebraic blend form.

## Hardware ladder

**Skip mini — config-only change on a validated commit.** Directly to 8×H100 JP.

## Seed plan

Seed 42 first. 3-seed (42/43/44) conditional on post-TTT ≤ 1.06610.

## Run protocol

```bash
python -c "import brotli"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 5906820

# Sanity-verify (same checks as 021h)
grep "nn.Parameter" train_gpt.py | grep recur_alpha
grep "torch.ones.*dtype=torch.float32" train_gpt.py
grep -c "alpha = self.recur_alpha\[.*\].to(x_new.dtype)" train_gpt.py  # must be 4
grep -c "x_before + alpha \* (x_new - x_before)" train_gpt.py          # must be 4

mkdir -p /runpod/runs/024-learnable-alpha-warmdown/seed_42

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/024-learnable-alpha-warmdown/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/024-learnable-alpha-warmdown/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_024_8h_jp \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
RECUR_ALPHA_ENABLED=1 \
WARMDOWN_FRAC=0.60 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/024-learnable-alpha-warmdown/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Artifacts

Standard: `final_model.pt`, `final_model.int6.ptz`, `train.log`, `diag_nvsmi.csv`, `final.json`.

**Critical:** capture `recur_alpha: values=...` log lines across the full run for α trajectory comparison vs 017 and 021h at matched steps. Key signal: does α reach 017's basin (L5 ~1.43) or stays offset (L5 ~1.38 like 021g/021h)?

`final.json` must include `recur_alpha_final_values`, `warmdown_frac: 0.60`, `layer_loop_enabled_at_step`.

## Stop-early criteria

- NaN/inf in train_loss → halt
- Step time > 2× 019b's → halt
- `layer_loop_enabled_at_step` outside [2000, 2300] → halt
- α divergence (|value| > 3.0 after step 3000) → halt

## Cost estimate

~$10 single-seed. ~$30 if 3-seed promotes.

## Dependencies

1. JP 8×H stock available
2. Brotli preflight confirmed

## Open questions for executor

1. **Watch loop_enabled_at_step.** With WARMDOWN_FRAC=0.60, the warmdown boundary moves but ENABLE_LOOPING_AT=0.35 stays the same. Loop should still activate around step 2000-2200 as normal. Verify `layer_loop_enabled_at_step` is in [2000, 2300].

2. **Watch α trajectory from first activation.** At full LR (before warmdown at ~step 1920), α should move more aggressively than in 021h. If α values at step 2200-2500 are similar to 021h's, the fix had no effect.

3. **Compare val@4000 to 021h (1.1084).** If 024 matches or beats 021h at step 4000, the per-step quality is maintained. If it regresses, delaying warmdown somehow hurt.

## What this does NOT test

- Frozen α with WARMDOWN_FRAC=0.60 (that's spec 025 if needed — to isolate warmdown vs α effect)
- WARMDOWN_FRAC values other than 0.60

## Decision tree after 024

| 024 post-TTT | Action |
|---|---|
| ≤ 1.064 | Clear beat. 3-seed. Submit. |
| 1.064–1.0661 | Beats #1736. 3-seed. |
| 1.0661–1.0662 | Ties 021e. Run spec 025 (frozen α + 0.60) to isolate. |
| > 1.0662 | Warmdown delay didn't unlock α. Submit 021e 3-seed. |
