# Spec 019 — Recur-Alpha with hardcoded α, full-pipeline on 8×H100

**Slug:** `recur-alpha-constant-full`
**Created:** 2026-04-21
**Links to:** spec 018c (throughput diagnostic), `research/evaluations/018c-recur-alpha-constant.md`, `research/ideas/beating-1736-note.md`

## Hypothesis

Spec 018c showed that hardcoding α as Python float constants (017's endpoint values) allows torch.compile to specialize the lerp kernel and recover **92% of blend overhead** at proxy scale. At full model scale this approaches zero throughput tax.

This spec tests the training-quality trade-off: **does losing α's adaptive learning (frozen at 017's values from step 1) hurt post-TTT val_bpb?**

Works backward from target:
- Target post-TTT: ≤ 1.06610 (beat #1736)
- 017's actual post-TTT with learnable α: 1.06733 (missed by 0.00123) — but 017 had buggy TTT path (α not applied during TTT)
- 019 recovers ~44 steps from throughput savings (~0.002 bpb training endpoint) + fixes TTT bug (unknown direction, probably helps)
- Expected post-TTT range: 1.0650 – 1.0675

## Baseline

Primary comparison: **#1736's 1.06610** (canonical target).
Secondary comparison: **017's 1.06733** (same recur-alpha mechanism, different α values & buggy TTT).

## Accept criteria

- Training completes without NaN/divergence
- `final_model.pt`, `final_model.int6.ptz` both emit
- Post-GPTQ val_bpb captured
- **Phased-TTT val_bpb captured** (the submission-gate number)

**Decision criterion (post-TTT val_bpb):**
| Post-TTT | Bucket | Next action |
|---|---|---|
| ≤ 1.06550 | Clear beat #1736 | 3-seed confirmation (~$30) then submission |
| (1.06550, 1.06710] | Close, within seed std | 3-seed to resolve |
| (1.06710, 1.06910] | Inside gate but worse than #1736 | Shelve recur-alpha for submission; still a mechanistic finding. |
| > 1.06910 | Outside gate | Investigate; likely hardcoded-α hurt too much |

## Code changes

**Branch:** `exp/recur-alpha-constant-full` forking from `aabfbea` (018c's commit) + additional TTT wiring.
**Commit:** `3c3a134` on `fork/exp/recur-alpha-constant-full`.
*(Corrected from original `2895db3`: pass-2 L5 α was 1.3984375 — 016's endpoint — instead of 1.4296875 — 017's actual recur_alpha_final. Bug introduced when 018c commit was prepared before 017 finished. Fixed during spec 019 execution interview.)*

Key properties:
- **No learnable α** — values hardcoded at `((1.078125, 1.2734375, 1.4296875), (1.015625, 0.97265625, 0.83203125))` from 017 endpoint
- **torch.compile sees α as compile-time constants** in both `forward_logits` and `forward_ttt` lerp sites
- **TTT bug fixed** — recur-alpha applies during TTT adaptation + eval (was missing in 015/016/017)
- `self.recur_alpha = None` — not a Parameter, no gradient tracking, no optimizer state

## Hardware ladder

- **Skip smoke** — cite 018c's mini-model throughput run (commit `aabfbea`) + the new TTT wiring as surgical change that doesn't affect training-path graph.
- **8×H100, region = whichever has capacity**. NA preferred (cleaner throughput), JP acceptable fallback (same constant-α benefit).
- **Seed 42** first. 3-seed (42/43/44) conditional on clear-promote bucket.

## Seed plan

Single seed (42) first. 3-seed if results promote.

## Inputs

- Data: CaseOps dataset. On JP `/runpod/data/...`, on NA `/workspace/data/...`.
- Tokenizer: `fineweb_8192_bpe.model`, bundled.
- Hotstart: **none — fresh from-scratch training with hardcoded α from step 1.**

## Execution protocol

Standard #1736 full pipeline. Example for JP:

```bash
cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git checkout 2895db3

mkdir -p /runpod/runs/019-recur-alpha-constant-full/seed_42
mkdir -p /runpod/.torch_inductor_cache

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/019-recur-alpha-constant-full/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/runpod/.torch_inductor_cache \
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
  > /runpod/runs/019-recur-alpha-constant-full/seed_42/train.log 2>&1
```

Substitute `/workspace` for `/runpod` on NA. pyminify must be installed (preflight).

## Expected throughput

Per 018c's 92% recovery at proxy scale, 019 on 8×H100 full model should run at nearly baseline (008) throughput. Expected endpoint step count: ~4825-4850 (vs 017's 4784 with tensor α). Gains: ~40-65 more training steps vs 017 = ~0.002 bpb of training-endpoint improvement.

Tok/s snapshot logging in final.json required (compare at steps 100/1000/2000/3000/4000/4500 vs 017).

## Checkpoints / artifacts to emit

Inherited baseline:
- `final_model.pt` — pre-GPTQ FP state dict, post-EMA
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — full pipeline sequence
- `final.json` — must include `val_bpb`, `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, **`val_bpb_post_ttt`**, `stopping_early_at_step`, `layer_loop_enabled_at_step`, tok/s snapshots, `recur_alpha_values_hardcoded` (the constants from the table for audit)
- `notes.md`

## Stop-early criteria

Unconditional:
- NaN/inf in train_loss → halt
- Step time > 2× spec 008 → halt

Conditional on `looping_active=True`:
- Training loss > 008's matched-step loss + 0.03 for 5+ consecutive log entries → halt (hardcoded α might be badly off for this seed's trajectory)

## Cost estimate

| item | cost |
|---|---|
| 8×H100 × ~25 min (compile + training + full pipeline) | ~$10 |
| Rsync + pod stop | ~$0.10 |
| **Single-seed total** | **~$10-12** |
| (Conditional) 3-seed × 2 additional runs | ~$20-24 |
| **If 3-seed promotes** | **~$30-36** |

## Open questions for interview

1. **Is hardcoded α at 017's values appropriate for seed 42?** 017 itself was seed 42, so the α values were learned on this exact seed. Should reproduce well. For seeds 43/44, hardcoded α would be a different-seed transplant; could be fine (α shape reproduces across seeds per 016/017 finding) or a slight mismatch.
2. **What if α=017's-values produces worse val_bpb than 017's learnable trajectory?** Then we know α co-evolution with weights matters — pivot to Path B (spec 020: learn then freeze).
3. **TTT fix included — is there any risk it interacts badly with recur-alpha?** Should be fine: TTT now applies α consistently with training. But it's unmeasured at full pipeline scale.

## Sequencing

- Run **after** 018c (already done, validated throughput).
- Run **before** any Path-B (learn-then-freeze) experiment — 019's simpler approach is worth testing first.
- If 019 promotes: 3-seed confirmation → submission candidate.
- If 019 is null/regression vs 017: 020 (Path B learn-then-freeze) becomes the next test.

## What 019 does NOT do

- Does not learn α (that's the whole point)
- Does not test Path B (learn then freeze) — that's deferred conditional on 019's outcome
- Does not run smoke (trusting 018c's validation of constant-α path + surgical TTT extension)
- Does not attempt freeze-plus-constant-fold mid-training (too expensive: recompile cost exceeds savings for 10-min run)
