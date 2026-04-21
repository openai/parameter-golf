# Spec 019b — Recur-Alpha manual+constant, full-pipeline on 8×H100

**Slug:** `recur-alpha-manual-constant-full`
**Created:** 2026-04-21
**Status:** **READY** — skipping 018d proxy validation (user decision: direct full-scale test is the question that matters)
**Links to:** spec 019 (lerp+constant, regressed −2.7% at full), `research/evaluations/018c-recur-alpha-constant.md`

## Hypothesis

Spec 019 used `torch.lerp(x_before, x_new, α_literal)` and **regressed −2.7% vs 008** at full scale, despite 018c (the same recipe at proxy) recovering 92% of blend overhead. The most likely culprit: `torch.lerp` is a primitive op whose fusion with surrounding pointwise code depends on Inductor heuristics that flip with scale (matmul-dominated graphs at full vs blend-dominated at proxy).

**Manual addition with literal α** (`α * x_new + (1−α) * x_before`) replaces lerp's primitive template with elementary pointwise arithmetic. After Dynamo's constant-folding, it's `mul, mul, add` with two literal scalars per site. Pointwise fusion is template-independent and should chew this into the surrounding block residual sum at any scale.

If 018d confirms this works at proxy (Run F ≥ 0.97 × Run E), we test it at full scale here.

**Target post-TTT:** ≤ 1.06610 (beat #1736).

Predicted full-scale throughput: somewhere between 017's −0.9% and ~flat. Could plausibly be the best variant tested.

## Baseline

Primary comparison: **#1736's 1.06610** (canonical target).
Secondary: **019's 1.06744** (same constant α, lerp version) and **017's 1.06733** (manual + tensor α, with TTT bug).

Expected post-TTT range: 1.064–1.067.

## Accept criteria

- Training completes without NaN/divergence
- `final_model.pt`, `final_model.int6.ptz` both emit
- Phased-TTT val_bpb captured (the submission-gate number)

**Decision criterion (post-TTT val_bpb):**
| Post-TTT | Bucket | Next action |
|---|---|---|
| ≤ 1.06550 | Clear beat #1736 | 3-seed confirmation (~$30) → submission |
| (1.06550, 1.06710] | Close, within seed std | 3-seed to resolve |
| (1.06710, 1.06910] | Inside gate but worse than #1736 | Recur-alpha exhausted; shelve and pivot |
| > 1.06910 | Outside gate | Investigate; manual+literal hurt training quality |

**Decision criterion (throughput):**
| 019b final step | vs 008 (4828) | Interpretation |
|---|---|---|
| ≥ 4800 | ≤ −0.6% tax | Manual+literal won — best variant |
| 4750–4800 | −0.6% to −1.6% tax | Comparable to 017; literal-α gave no extra win at full |
| < 4750 | > −1.6% tax | Manual+literal also disrupts fusion at full; constant-α path is dead |

## Code changes

**Branch:** `exp/recur-alpha-manual-constant-full` forking from `3c3a134` (019).
**Commit:** `e93d77d` on `exp/recur-alpha-manual-constant-full`.

Diff scope: replace `torch.lerp` with manual addition at **4 sites** (vs 2 sites in 018d, since 019 has the TTT-fix wiring):
- `forward_logits` encoder loop
- `forward_logits` decoder loop
- `forward_ttt` encoder loop
- `forward_ttt` decoder loop

Per site:
```python
# OLD (019):
x = torch.lerp(x_before, x_new, alpha)
# or in TTT:
x = torch.lerp(x_before, x, alpha)

# NEW (019b):
x = alpha * x_new + (1.0 - alpha) * x_before
# or in TTT:
x_new_local = x  # the just-computed _block_with_lora output
x = alpha * x_new_local + (1.0 - alpha) * x_before
```

Everything else stays from 019: same `_ALPHA_CONSTANTS_017` table (with corrected pass-2 L5 = 1.4296875), same `self.recur_alpha = None`, same TTT-fix path wiring.

## Hardware ladder

- **Skip smoke** if 018d passed cleanly (cite 018d as the proxy validation; the surgical change is ~8 LOC of basic arithmetic, no new functionality).
- **8×H100, region = whichever has capacity.** NA preferred for cleaner throughput vs 019's JP run; JP acceptable.
- **Seed 42** first. 3-seed (42/43/44) conditional on clear-promote bucket.

## Seed plan

Single seed (42) first. 3-seed if results promote.

## Inputs

- Data: CaseOps dataset. On JP `/runpod/data/...`, on NA `/workspace/data/...`.
- Tokenizer: `fineweb_8192_bpe.model`, bundled.
- Hotstart: **none — fresh from-scratch training with hardcoded α from step 1.**

## Execution protocol

Standard #1736 full pipeline. Example for NA:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout e93d77d

mkdir -p /workspace/runs/019b-recur-alpha-manual-constant-full/seed_42
mkdir -p /workspace/.torch_inductor_cache

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/019b-recur-alpha-manual-constant-full/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/workspace/.torch_inductor_cache \
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
  > /workspace/runs/019b-recur-alpha-manual-constant-full/seed_42/train.log 2>&1
```

Substitute `/runpod` for `/workspace` on JP. pyminify must be installed (preflight).

## Expected throughput

Per 018d's result (assuming it lands in ≥0.97×K bucket): full-scale should match or beat 017's −0.9% throughput tax. Predicted final step ≥ 4780 (vs 008's 4828).

Tok/s snapshot logging in final.json required (compare at steps 100/1000/2000/3000/4000/4500 vs 017 and 019).

## Checkpoints / artifacts to emit

Inherited baseline:
- `final_model.pt` — pre-GPTQ FP state dict, post-EMA
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — full pipeline sequence
- `final.json` — must include `val_bpb`, `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, tok/s snapshots, `recur_alpha_values_hardcoded`
- `notes.md`

## Stop-early criteria

Unconditional:
- NaN/inf in train_loss → halt
- Step time > 2× spec 008 → halt

Conditional on `looping_active=True`:
- Training loss > 008's matched-step loss + 0.03 for 5+ consecutive log entries → halt

## Cost estimate

| item | cost |
|---|---|
| 8×H100 × ~25 min (compile + training + full pipeline) | ~$10 |
| Rsync + pod stop | ~$0.10 |
| **Single-seed total** | **~$10–12** |
| (Conditional) 3-seed × 2 additional runs | ~$20–24 |
| **If 3-seed promotes** | **~$30–36** |

## Open questions for interview

1. **Same-pod chain with 019?** If 019's JP pod is still up, run there for cache-warm + same-hardware comparison vs 019. Otherwise spin fresh — NA preferred.
2. **What if 019b lands at ~017's throughput but with fixed TTT path?** Then we have 017's quality + 019's TTT-fix + lower throughput tax than 019. Submission candidate.
3. **Diagnostic logging:** add per-step tok/s logging for steps 2100–2400 (around loop activation) so we can directly compare 019 vs 019b for the post-activation gap pattern. ~5 LOC.

## Sequencing

- **No precondition.** 018d (proxy validation) was drafted but intentionally skipped — the question we care about is whether manual+literal works at full scale, and proxy results have already mis-extrapolated once (018c → 019). Direct full-scale test is the decisive experiment.
- Run on whatever region has capacity. NA preferred over JP (019's regression may have been partly JP pod variance; NA gives cleaner comparison vs 008's baseline throughput).
- If 019b promotes (final step ≥ 4780 AND post-TTT ≤ 1.06710): 3-seed confirmation → submission candidate.
- If 019b regresses (final step < 4750): constant-α at full is fundamentally broken; pivot to **017's recipe with TTT fix only** (manual + tensor α, no constant α experiment) as the safe submission baseline.

## What 019b does NOT do

- Does not learn α (same as 019)
- Does not test α=1 identity elimination
- Does not run smoke (trusting 018d's proxy validation + surgical 8-LOC arithmetic change)
- Does not investigate WHY 019's lerp regressed at full (answer is implicit in the comparison; deeper investigation needs TORCH_LOGS=output_code on a separate diagnostic spec)
