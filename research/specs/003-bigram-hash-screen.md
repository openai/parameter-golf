# Spec 003 — BigramHash signal screen (match Exp 24 baseline)

**Slug:** `bigram-hash-screen`
**Created:** 2026-04-19
**Links to idea:** `research/ideas/bigram-hash.md`

## Hypothesis
Adding BigramHash (3072 buckets × 112 dims ≈ 344K params) to our stack improves pre-quant val_bpb. This is a **signal screen** — we don't test the 16MB fit here (artifact will be ~16.2 MB, oversized but fine for screening). Budget-fit engineering is deferred to spec 004 (only if this wins).

**Comparison via training-loss trajectory at matched steps**, plus final pre-quant val_bpb. BigramHash activates from step 1 (`train_gpt_sota.py:474, 553-554`, zero-init → no disruption), so same-seed runs give a deterministic train_loss comparison.

## Baseline — REUSE EXP 24
Exp 24 is already a 2×H100 40-min SOTA-code training run with `BIGRAM_VOCAB_SIZE=0`. Log at `logs/exp24_sota_sp8192_2xh100_40m.log`. We **don't rerun the control** — we match Exp 24's config exactly (below) in the variant and compare variant's train_loss against Exp 24's log at matched steps.

Cost savings: ~$4 and ~45 min vs the original paired-runs plan.

### Exp 24 reference milestones (from `logs/exp24_sota_sp8192_2xh100_40m.log`)

| Step | train_loss | val_bpb (if sampled) |
|---|---|---|
| 500 | 3.2805 | — |
| 1000 | 3.2346 | 1.2424 |
| 1500 | 3.1254 | — |
| 2000 | ? (interview) | 1.2?? |
| 3000 | ? | 1.1?? |
| 4531 (final) | ? | **pre-quant 1.08670** |

Final artifact: 15.989 MB, post-quant val_bpb 1.09850. No TTT.

## Expected Δ
+0.002 to +0.005 bpb at end-of-training pre-quant (per original BigramHash submission's claim). Train_loss curves should diverge below Exp 24's from ~step 500 onward if the signal is real.

## Accept criteria
- **Validity:** variant's training trajectory shape (step 0 → first 200-500 steps) roughly matches Exp 24's. If training explodes or wildly diverges early, kill and investigate — probably a config plumbing issue.
- **Signal:** variant ≤ Exp 24 at ≥3 of last 4 train_loss milestones AND final pre-quant val_bpb ≤ **1.0847** (Δ ≤ −0.002 vs Exp 24's 1.08670).

**Important caveat:** This screens BigramHash on `QK_GAIN_INIT=5.0` (Exp 24's config), not our spec-000 baseline's 5.25. The two interventions are orthogonal (bigram is embedding-layer; QK gain is attention-layer), so a positive signal here should transfer to QK=5.25 — but it's not bulletproof. Spec 004 (if we promote) is a proper full-stack 8×H100 run with `QK_GAIN_INIT=5.25`.

## Config diff
**Match Exp 24 exactly.** Only difference: `BIGRAM_VOCAB_SIZE`.

| Env var | Value | Source |
|---|---|---|
| `BIGRAM_VOCAB_SIZE` | **3072** | the variable under test |
| `BIGRAM_DIM` | 112 | hyperparameter default matches Exp 24 implicit value (Exp 24 didn't have the option; the code now does) |
| `QK_GAIN_INIT` | **5.0** | Exp 24's value (code default; current spec 000 uses 5.25 but we match Exp 24 for clean comparison) |
| `TTT_ENABLED` | **0** | Exp 24 didn't have TTT on; we don't enable it either |
| `SEED` | **1337** | Exp 24's seed |
| `TRAIN_LOG_EVERY` | **100** | Exp 24's value |
| `MAX_WALLCLOCK_SECONDS` | 2400 | Exp 24's 40-min cap |

Leave everything else at code defaults. **Single run, no paired control run** — Exp 24's log is the control.

## Code changes
- Branch: `research`
- Commit: `77085f6`
- Diff: **none.** BigramHash is already implemented (`train_gpt_sota.py:432-460, 474, 553-554`). Hyperparam-only spec.

## Hardware ladder
- [ ] 2×H100 NA-1 — **only rung**. Match Exp 24's harness.
- [ ] 8×H100 — not used. Screen only.

Note: different physical pod than Exp 24 (inevitable — Exp 24's pod is long gone). Step count may differ from Exp 24's 4531 due to pod-to-pod throughput variance. Compare at matched **step numbers** (not wall-clock times); train_loss at step N reflects the model's state after N steps regardless of pod speed.

## Seed plan
Single seed (1337, matching Exp 24). Same-seed + same-config-except-bigram gives deterministic data ordering, so any divergence in train_loss is attributable to BigramHash.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/`
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: **none** — from-scratch training.
- Exp 24 reference log (for comparison, read-only): `logs/exp24_sota_sp8192_2xh100_40m.log` (in-repo, ~4 KB).
- Base repo commit: `77085f6` on `research`.

## Execution protocol

```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 \
QK_GAIN_INIT=5.0 TTT_ENABLED=0 SEED=1337 \
TRAIN_LOG_EVERY=100 MAX_WALLCLOCK_SECONDS=2400 \
torchrun --standalone --nproc_per_node=2 train_gpt_sota.py \
  > /workspace/runs/003-bigram-hash-screen/variant_train.log 2>&1
```

Run to wallclock cap (~40 min). Post-training pre-quant eval is automatic. No TTT eval (disabled). No sliding eval needed for screen signal.

## Stop-early criteria
Applied to the variant run, comparing against Exp 24's log at matched steps:

| At variant step N | Condition | Action |
|---|---|---|
| 100-500 | Variant train_loss within ~±0.1 of Exp 24 at matched step | Normal — early steps are noisy, keep going |
| 1000 | Variant > Exp 24[step 1000] + 0.03 (Exp 24 was 3.2346 → kill if variant > 3.26) | **Kill.** BigramHash is hurting. Save ~30 min pod time. |
| 1500 | Variant > Exp 24[step 1500] − 0.01 (Exp 24 was 3.1254 → variant should be ≤ 3.115 to show signal) | Weak — flag, continue |
| 2000+ | Variant ≤ Exp 24 − 0.02 at same step | Strong — run to end |

Standard: NaN / step-time > 2× expected / divergence → kill.

## Cost estimate
- 2×H100 NA-1 at ~$6/hr.
- Single 40-min run = **~$4**.
- Early-kill at step 1000: **~$2**.
- No second run because Exp 24 is the control.

## Extra artifacts
- `runs/003-bigram-hash-screen/variant_train.log` — full stdout (train_loss at every 100 steps, per Exp 24's cadence).
- `runs/003-bigram-hash-screen/final.json` — variant's end metrics (pre-quant bpb, post-quant bpb, step count, throughput) + Δ vs Exp 24's 1.08670 pre-quant.
- `runs/003-bigram-hash-screen/loss_compare.md` — **primary artifact.** Matched-step train_loss table with columns: step, exp24_train_loss, variant_train_loss, Δ. Include ALL milestones from both logs.
- `runs/003-bigram-hash-screen/notes.md` — execution narrative.

No train checkpoints (hotstart from this trajectory isn't planned). No `.ptz` retained (screen model, not submission candidate).

## Open questions for interview
- Confirm `MAX_WALLCLOCK_SECONDS=2400` matches Exp 24's timing; if Hyperparameters default differs from 2400, explicit override required.
- Confirm `logs/exp24_sota_sp8192_2xh100_40m.log` is accessible on pod (in-repo at checkout, should be present).
- Pod variance note: if our pod is materially slower than Exp 24's (e.g. trains only 3500 steps in the 40 min), the final pre-quant eval will be on an under-trained model and the Δ vs Exp 24 is confounded by training completeness. Interview surfacing: what should execution do if step count at 40 min is < 4000? Recommend: **extend wall cap to match Exp 24's step count**, not its wall time — set `MAX_WALLCLOCK_SECONDS` high enough that variant reaches step 4500. Would cost extra ~$1-2 but gives apples-to-apples comparison.
