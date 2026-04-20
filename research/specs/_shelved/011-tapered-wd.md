# Spec 011 — Tapered weight decay (SHELVED 2026-04-20)

**Status:** SHELVED. Superseded by `research/specs/011-training-bundle.md` which bundles tapered WD with GradPower (port #1682). This standalone design was never run. Kept as a reference for design intent.

**Slug:** `tapered-wd`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md`.

## Hypothesis

Late-training weight-decay taper (full WD early → half WD after 70% of training) improves final val_bpb. Witnessed on PR #1729 (romeerp) at claimed small-but-real Δ on top of a #1626-adjacent base. Intuition: full WD early maintains regularization / compression pressure when it matters most (weights still moving); reduced WD late lets the model converge to a lower loss in the settled regime without fighting regularization.

## Baseline

Spec 008's seed-42 val_bpb (#1736 reproduction), measured as spec 009's `baseline` mode once that runs. Comparison is Δ vs the no-taper number.

## Expected Δ

−0.0005 to −0.002 bpb vs baseline. This is a thin lever — independent witnesses on #1729 report small gains, and #1736's WD values are already well-tuned for its schedule.

**Null/positive Δ** → taper is redundant with #1736's existing WD schedule, shelve the idea. Not a bug, just a null result.

## Accept criteria

- Training completes without NaN / divergence.
- Post-quant post-TTT val_bpb measured.
- Artifact < 16 MB, within time budget (600s train + 600s eval).
- **Decision criterion:** Δ ≤ −0.001 vs baseline → promote, consider stacking with spec 009 winner (if any). Δ > +0.0005 → shelve. Otherwise inconclusive, revisit if budget permits.

## Config diff

Two new env vars (implement as patch to `train_gpt.py`):

```
WD_TAPER_START_FRAC=0.70       # begin taper at 70% of iterations
WD_TAPER_FINAL_MULT=0.50       # WD at end = WD_initial * 0.50
```

Everything else identical to spec 008's env block.

Taper schedule applied to **Muon WD and Adam WD** (both of which #1729 applies to); embed WD left untouched (separate optimizer, not the focus of the lever).

## Code changes

- **Branch:** `research` (same as spec 008 — this is a lever we either promote to the baseline or shelve, per the baseline-migration convention in CLAUDE.md).
- **Commit:** TBD after implementation.
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** add env-var-gated taper logic in the training loop. Approximate sketch:

  ```python
  # In Hyperparameters:
  wd_taper_start_frac = float(os.environ.get("WD_TAPER_START_FRAC", "0.0"))  # 0 disables
  wd_taper_final_mult = float(os.environ.get("WD_TAPER_FINAL_MULT", "1.0"))

  # In the training loop, per step `step` of `total_steps`:
  if h.wd_taper_start_frac > 0.0:
      start_step = int(h.wd_taper_start_frac * total_steps)
      if step >= start_step:
          # Linear taper from 1.0 at start_step to h.wd_taper_final_mult at total_steps.
          progress = (step - start_step) / max(1, total_steps - start_step)
          mult = 1.0 - progress * (1.0 - h.wd_taper_final_mult)
          for group in muon_optimizer.param_groups:
              group["weight_decay"] = h.muon_wd * mult
          for group in adam_optimizer.param_groups:
              if "weight_decay" in group and group["weight_decay"] != 0.0:
                  group["weight_decay"] = h.adam_wd * mult
      else:
          # Make sure WD is at initial values (in case schedule re-enters).
          ...
  ```

  Env-gated: default `WD_TAPER_START_FRAC=0.0` disables the taper entirely → script is a no-op change when the env var is absent. This keeps the patch compatible with spec 008 (if execution re-runs spec 008 for any reason, it won't pick up taper behavior by default).

- **Reference:** `romeerp/parameter-golf-caseops-v1`'s companion repo and PR #1729 body for the exact schedule form.

## Hardware ladder

- [x] **8×H100 full training run, single seed (42).** Same hardware and time budget as spec 008. ~30 min wall + eval + TTT. Total ~20 min GPU + eval.

## Seed plan

Single seed (42). If signal is positive and we want to stack with spec 009's winning mode, that stack runs as a separate spec.

## Inputs

- **Data:** same CaseOps dataset as spec 008 (on persistent volume, already prepared).
- **Tokenizer:** bundled with #1736 submission dir.
- **Hotstart checkpoint:** none — full-from-scratch training.

## Execution protocol

Single pod, single run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/011-tapered-wd/seed_42

NCCL_NET=Socket DATA_DIR=./data \
ARTIFACT_DIR=/workspace/runs/011-tapered-wd/seed_42 \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
WD_TAPER_START_FRAC=0.70 \
WD_TAPER_FINAL_MULT=0.50 \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/011-tapered-wd/seed_42/train.log 2>&1
```

Verify in log: `muon_wd` value at step >= 0.7×total_steps should show the ramp. Add a one-time log line at the start of the taper zone:

```
log(f"WD_TAPER: start_step={start_step} total_steps={total_steps} "
    f"muon_wd_init={h.muon_wd} adam_wd_init={h.adam_wd} final_mult={h.wd_taper_final_mult}")
```

## Checkpoints to emit

**Exactly one:** `runs/011-tapered-wd/seed_42/final_model.pt` — auto-saved by `serialize()` before GPTQ. Same convention as spec 008. Reusable for future quant-family experiments (SpinQuant, per-group bit, AR-selfgen) on top of tapered-WD weights if this lever lands.

Plus the submission `.ptz` artifact and `final.json` as usual.

## Stop-early criteria

- NaN in train_loss at any step → halt.
- Step time > 2× expected → halt, investigate.
- Artifact > 16 MB → halt, flag (shouldn't change from spec 008).
- Post-TTT val_bpb > spec 008 baseline + 0.003 (clear regression) → flag, but DON'T auto-halt (the full run has already completed at that point; just write the eval).

## Cost estimate

| Item | Cost |
|---|---|
| Pod spin-up | $2 |
| Full training run (8×H100, ~30 min wall) | $10 |
| Eval + TTT (~10 min) | $3 |
| Buffer for debug | $5 |
| **Total** | **~$20** |

Same rough cost as spec 008, since it's a full retrain with a tiny config change.

## Extra artifacts

- `runs/011-tapered-wd/seed_42/train.log` — full training log
- `runs/011-tapered-wd/seed_42/final_model.pt` — pre-GPTQ FP checkpoint
- `runs/011-tapered-wd/seed_42/final_model.int6.ptz` — quantized submission artifact
- `runs/011-tapered-wd/seed_42/final.json` — post-TTT val_bpb, Δ vs spec 008, wall times
- `runs/011-tapered-wd/seed_42/notes.md` — execution narrative

## Open questions for interview

1. **Which optimizer(s) get the taper?** PR #1729's body suggests their taper applied to *Muon WD only*. Our implementation should probably follow that — the lever as they measured it is Muon-specific. Adam WD can be left at 0.02 throughout. Confirm at interview; if unclear, run Muon-only for the first pass.
2. **Parallel to spec 009?** Yes — spec 009 hotstarts off spec 008's `pre_gptq.pt` on one pod; spec 011 retrains on a separate pod. Independent. Total combined cost ~$35 if run simultaneously, vs ~$35 sequentially anyway — simultaneity just parallelizes wall time.
3. **Is the taper linear or cosine?** PR #1729's README implies linear from start_frac to end. If cosine decay is preferred, we can change to `mult = h.wd_taper_final_mult + 0.5 * (1 - h.wd_taper_final_mult) * (1 + cos(pi * progress))`. For the first pass, linear is simpler and cheaper to reason about.
4. **Does WD taper interact with MATRIX_LR decay?** #1736 already has a cosine LR schedule during warmdown. Tapering WD on top is an additional schedule — need to verify no weird interaction (e.g., LR near-zero + reduced WD = almost no parameter movement, which shouldn't matter but worth glancing at training log).

## What this spec does NOT do

- Does not sweep `WD_TAPER_START_FRAC` or `WD_TAPER_FINAL_MULT` values — single config (0.70 / 0.50) matching #1729's report. If signal, a follow-up spec can sweep.
- Does not change any non-WD hyperparameter.
- Does not touch the SpinQuant-family code path. Completely independent of spec 009.
- Does not run 3-seed — single seed, screening. 3-seed confirmation only if lever lands and we want to submit.
- Does not stack with spec 009's winner automatically. If both land positive, a separate follow-up spec runs tapered-WD training AND whichever SpinQuant mode won.
