# Spec 012 — Training-time bundle (tapered WD + GradPower + softer QK-gain)

**Slug:** `training-bundle`
**Created:** 2026-04-20
**Supersedes:** spec 011 (tapered-wd alone). Spec 011 design doc kept for reference; it is not run as a standalone.
**Links to ideas:** `research/ideas/1736-improvement.md`, `gradpower-muon.md`, `per-layer-qk-gain.md`.

## Hypothesis

Three **training-time, upstream-of-TTT** levers stack additively on #1736:

1. **Tapered Muon WD** (port #1729): full WD early → half WD after 70% of steps.
2. **GradPower p=0.9** for Muon (port #1682): softens pre-orthogonalization gradient magnitudes.
3. **Softer uniform QK-gain init=2.5** (port #1648, simplified): reduces over-sharp attention at init.

Each is env-gated and independently isolable by a single env-var flip. All three default to no-op when unset → train_gpt.py is byte-compatible with spec 008.

## Baseline

Spec 008's seed-42 val_bpb (`runs/008-1736-reproduction/seed_42/final.json`). Comparison is Δ vs that number.

## Expected Δ

- Tapered WD alone: −0.0005 to −0.002 (thin)
- GradPower alone: −0.001 to −0.003
- QK-gain 5.0 → 2.5 alone: **unknown; could be −0.002 to −0.005 OR could regress**
- Bundled: conservatively −0.001 to −0.004 (some interaction / non-additive absorption expected)

**Headline risk:** QK-gain change is the biggest unknown. Softer init is plausible per #1648's convergence evidence but may interact badly with #1736's specific stack. Isolation follow-up run needed if bundle regresses.

## Accept criteria

- Training completes without NaN / divergence.
- Post-quant post-TTT val_bpb measured.
- Artifact < 16 MB, within time budget.
- **Decision criterion:**
  - Δ ≤ −0.002 → promote, run isolation (three single-flag runs) to attribute.
  - Δ ∈ (−0.002, −0.0005] → promote cautiously, isolate the positive flag(s).
  - Δ ∈ (−0.0005, +0.001) → null bundle; run flag-isolation mini to find if any single flag helps alone.
  - Δ > +0.001 → regression (likely QK-gain). Run with `QK_GAIN_INIT=5.0` restored (WD+GradPower only) as second attempt.

## Config diff vs spec 008

```
WD_TAPER_START_FRAC=0.70
WD_TAPER_FINAL_MULT=0.50
MUON_GRAD_POWER=0.9
QK_GAIN_INIT=2.5
```

Everything else identical.

## Code changes

- **Branch:** `exp/training-bundle` (worktree at `worktrees/training-bundle/`).
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** Add four new `Hyperparameters` fields (with no-op defaults) + three small insertion points:
  - `WD_TAPER_*` — mutate `group["weight_decay"]` in the training loop when `step >= start_step`.
  - `MUON_GRAD_POWER` — apply `g = sign(g) * g.abs().pow(p)` at line ~1593 in `Muon.step()`, gated behind `if p != 1.0`.
  - `QK_GAIN_INIT` — existing env var; default shifts from 5.0 to 2.5 only when explicitly set.
  - `QK_GAIN_PER_LAYER` — new env var; if set, overrides each block's `attn.q_gain` after block construction.
- **Default-off invariant:** if `WD_TAPER_START_FRAC=0` AND `MUON_GRAD_POWER=1.0` AND `QK_GAIN_INIT=5.0` AND `QK_GAIN_PER_LAYER` unset → forward and optimizer are byte-identical to spec 008's code path.

## Hardware ladder

- [x] **8×H100 full training run, seed 42.** Same as spec 008. ~30 min wall + TTT eval. ~$20.
- Optional mini on 2×H100 with reduced steps only if the code patch lands late and we need a quick NaN check. Skip if patch is clean.

## Seed plan

Single seed (42) for screen. 3-seed confirmation only if bundle lands and we commit to it.

## Inputs

- **Data:** same CaseOps dataset as spec 008 (persistent volume).
- **Tokenizer:** bundled with #1736 submission dir.
- **Hotstart:** none, full from-scratch training.

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/012-training-bundle/seed_42

NCCL_NET=Socket DATA_DIR=./data \
ARTIFACT_DIR=/workspace/runs/012-training-bundle/seed_42 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
WD_TAPER_START_FRAC=0.70 \
WD_TAPER_FINAL_MULT=0.50 \
MUON_GRAD_POWER=0.9 \
QK_GAIN_INIT=2.5 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/012-training-bundle/seed_42/train.log 2>&1
```

Expected log lines at start:
- `TRAINING_BUNDLE: wd_taper=0.70→0.50, muon_grad_power=0.9, qk_gain=2.5`

## Checkpoints to emit

- `final_model.pt` (pre-GPTQ FP) — reusable for future stacked experiments.
- Submission artifact + `final.json` as usual.

## Stop-early criteria

- NaN in train_loss → halt.
- Step time > 2× spec 008 → halt.
- Artifact > 16 MB → flag.

## Cost estimate

~$20 (same as spec 008). Full training run.

## Follow-up runs (if bundle lands)

- **Isolation run #1:** QK_GAIN_INIT=5.0 (restore), WD_TAPER + GradPower only → attribute the QK contribution.
- **Isolation run #2:** MUON_GRAD_POWER=1.0, WD_TAPER + QK only → attribute GradPower.
- **Isolation run #3:** WD_TAPER_START_FRAC=0, GradPower + QK only → attribute WD taper.

Three single-flag runs = ~$60 additional spend. Only do this if bundle lands Δ ≤ −0.002.

## Open questions for interview

1. Should the first pass use `QK_GAIN_INIT=2.5` (uniform softer) or the full 11-value `QK_GAIN_PER_LAYER` list? Plan uses uniform for simplicity — per-layer requires convergence runs we haven't done.
2. Should tapered WD apply to Muon only (per #1729) or also Adam? Plan uses Muon only.
3. If bundle regresses, do we rerun immediately with `QK_GAIN_INIT=5.0` restored, or pause for analysis? Plan: rerun immediately, QK is the most likely culprit.

## What this spec does NOT do

- Does not include xIELU activation or symmetric resid_mix from #1648 — deferred to potential spec 013.
- Does not include Tap-In (#1555) — eval-time lever, separate spec.
- Does not include trajectory-state readout (#1676) — deferred.
- Does not run 3-seed — single-seed screen only.
