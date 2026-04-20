# Spec 012 — Training-time bundle (tapered WD + GradPower)

**Slug:** `training-bundle`
**Created:** 2026-04-20
**Supersedes:** spec 011 (tapered-wd alone). Spec 011 design doc kept for reference; it is not run as a standalone.
**Links to ideas:** `research/ideas/1736-improvement.md`, `gradpower-muon.md`, `per-layer-qk-gain.md`.

## Hypothesis

Two **training-time, upstream-of-TTT** levers stack additively on #1736:

1. **Tapered Muon WD** (port #1729): full WD early → half WD after 70% of steps.
2. **GradPower p=0.9** for Muon (port #1682): softens pre-orthogonalization gradient magnitudes.

Each is env-gated and independently isolable. Both default to no-op when unset → train_gpt.py is byte-compatible with spec 008.

**Dropped from first-pass:** softer QK_GAIN (port #1648). Highest-risk lever; deferred to spec 013 if 012 lands cleanly. Keeps attribution between WD and GradPower only, and avoids the scenario where QK regression masks a WD or GradPower win.

## Baseline

Spec 008's seed-42 val_bpb (`runs/008-1736-reproduction/seed_42/final.json`). Comparison is Δ vs that number.

## Expected Δ

- Tapered WD alone: −0.0005 to −0.002 (thin)
- GradPower alone: −0.001 to −0.003
- Bundled (WD + GradPower): **−0.0015 to −0.004** (roughly additive, some overlap possible)

**Headline risk:** Both levers are training-time, upstream of TTT. Our spec 010/010b finding shows TTT absorbs upstream deltas unevenly; post-TTT Δ may be smaller than pre-TTT Δ.

## Accept criteria

- Training completes without NaN / divergence.
- Post-quant post-TTT val_bpb measured.
- Artifact < 16 MB, within time budget.
- **Decision criterion:**
  - Δ ≤ −0.002 → promote, run two single-flag isolation runs (WD-only, GradPower-only) to attribute.
  - Δ ∈ (−0.002, −0.0005] → promote cautiously, isolate.
  - Δ ∈ (−0.0005, +0.001) → null bundle; consider flag-isolation mini or kill both.
  - Δ > +0.001 → regression; investigate before retrying.

## Config diff vs spec 008

```
WD_TAPER_START_FRAC=0.70
WD_TAPER_FINAL_MULT=0.50
MUON_GRAD_POWER=0.9
```

Everything else identical. `QK_GAIN_INIT` left at the default 5.0 (not changed for this run).

## Code changes

- **Branch:** `exp/training-bundle` (worktree at `worktrees/training-bundle/`).
- **Commit:** `8d54854`.
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** Add four new `Hyperparameters` fields (with no-op defaults) + three small insertion points:
  - `WD_TAPER_*` — mutate `group["weight_decay"]` in the training loop when `step >= start_step`.
  - `MUON_GRAD_POWER` — apply `g = sign(g) * g.abs().pow(p)` at line ~1593 in `Muon.step()`, gated behind `if p != 1.0`.
  - `QK_GAIN_INIT` — existing env var; default shifts from 5.0 to 2.5 only when explicitly set.
  - `QK_GAIN_PER_LAYER` — new env var; if set, overrides each block's `attn.q_gain` after block construction.
- **Default-off invariant:** if `WD_TAPER_START_FRAC=0` AND `MUON_GRAD_POWER=1.0` AND `QK_GAIN_INIT=5.0` AND `QK_GAIN_PER_LAYER` unset → forward and optimizer are byte-identical to spec 008's code path.

## Hardware ladder

- [x] **Smoke test: 2×H100, short run (~5 min, ~$1).** Purpose: confirm the code patch doesn't crash, NaN, or divergence-bomb. **Do NOT read val_bpb from this rung** — batch-size regime differs from 8×H100, so any Δ is ambiguous. Pass criterion: 500+ training steps complete, train_loss curve smooth (no NaN, not diverging). Typical invocation: `ITERATIONS=500 torchrun --nproc_per_node=2 train_gpt.py`.
- [x] **8×H100 full training run, seed 42.** Same as spec 008. ~30 min wall + TTT eval. ~$20. Read post-TTT val_bpb from `final.json` here.

### Early-stop guidance (on the 8×H100 rung)

**Protocol:** executor + user both monitor `train.log` via `tail -f` once training is underway. Every `TRAIN_LOG_EVERY` steps the run emits `{step}/{iterations} train_loss: X.XXXX`. Compare against spec 008's log at matched step.

**Kill-the-pod decision is a joint call, not an automatic trigger.** Things to flag for discussion:
- Consistently worse than spec 008 (by any visible margin) across multiple late-training log entries, trend not improving.
- train_loss plateau that looks qualitatively different from spec 008's curve.
- NaN, inf, or step-time blow-up → automatic kill (these are bugs, not regressions).

**Default if in doubt:** let it finish. The $4 saved by early-terminating isn't worth killing a run that might still deliver a real post-TTT delta. Spec 010/010b showed TTT absorbs some upstream deltas; a "neutral train_loss" can still produce a non-trivial post-TTT win.

**Caveats:**
1. Train_loss is on training data, not val. Lower train_loss can coexist with higher val_bpb (overfitting), though in a 600s data-bound run this is unlikely.
2. Treat train_loss as a *lower bound* on the bad case — clearly worse → discuss killing; ambiguous → finish.

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
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/012-training-bundle/seed_42/train.log 2>&1
```

Expected log lines at start:
- `training_bundle: wd_taper_start_frac=0.7 wd_taper_final_mult=0.5 muon_grad_power=0.9 qk_gain_init=5.0 qk_gain_per_layer=''`

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

- **Isolation run #1:** MUON_GRAD_POWER=1.0, WD_TAPER on → attribute GradPower.
- **Isolation run #2:** WD_TAPER_START_FRAC=0, GradPower on → attribute WD taper.

Two single-flag runs = ~$40 additional spend. Only do this if bundle lands Δ ≤ −0.002.

**Spec 013 (queued):** softer QK_GAIN (5.0 → 2.5 uniform, then per-layer if uniform wins). Was part of 012's first draft but deferred as highest-regression-risk lever. Run only after 012 lands and we've confirmed the stack is healthy.

## Open questions for interview

1. Should tapered WD apply to Muon only (per #1729) or also Adam? Plan uses Muon only.
2. Is the WD taper linear (current) or cosine? #1729's README implies linear; easy to swap if cosine preferred.

## What this spec does NOT do

- Does not change `QK_GAIN_INIT` (deferred to spec 013).
- Does not include xIELU activation or symmetric resid_mix from #1648 — deferred.
- Does not include Tap-In (#1555) — eval-time lever, separate spec.
- Does not include trajectory-state readout (#1676) — deferred.
- Does not run 3-seed — single-seed screen only.
