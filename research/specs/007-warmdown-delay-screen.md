# Spec 007 — delayed-warmdown hotstart screen

**Slug:** `warmdown-delay-screen`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/training-dynamics.md` (post-spec-006 section)

## Hypothesis

Spec 006 showed the steepest val_bpb descent (−0.012 per 100 steps) at steps 1700-1900, immediately after the recurrence-activation bump. In the current schedule, LR is already decaying through this window (LR_mul 0.87→0.78). Holding peak LR until a later warmdown_start — e.g. step 2300 — should extract more descent from this high-velocity window.

Screen tests the aggressive end of the delayed-warmdown hypothesis: warmdown_start pushed to step ~2300 (i.e., no warmdown within the screen window). If this doesn't show signal, the hypothesis is weakened.

## Baseline

Spec 006's val_bpb at step 2300 = 1.1685 (from `runs/006-dense-ckpts/analysis/val_loss.csv`). This is the comparison target.

Phase 1 (variant A) establishes the *hotstart-based* baseline, which may differ slightly from spec 006's 1.1685 due to RNG replay subtleties in hotstart fast-forward. Variant C compares directly against Phase 1's output, not against 1.1685.

## Expected Δ

- **Strong signal:** val_bpb_C − val_bpb_A ≤ −0.005 → delayed warmdown extracts meaningful post-recurrence value → escalate to full-run spec 008.
- **Weak/null:** |Δ| ≤ 0.003 → inconclusive; try variant B (warmdown_start ≈ 1700 or 2000) next.
- **Negative:** Δ ≥ +0.003 → delayed warmdown hurts at screen horizon; shelve.

## Accept criteria

### Phase 1 (variant A — reproduction)
- Training reaches step 2300 without NaN / divergence.
- Both auto-final checkpoints land (`ckpt_final_pre_ema_step2300.pt`, `ckpt_final_post_ema_step2300.pt`).
- val_bpb at step 2300 within ±0.003 of spec 006's 1.1685.
  - **If this fails** (>0.003 off): halt. Do NOT launch Phase 2. Flag to research.

### Phase 2 (variant C — delayed warmdown, runs only if Phase 1 passes)
- Training reaches step 2300 without NaN / divergence.
- Both auto-final checkpoints land.
- val_bpb at step 2300 is reported (success criterion is evaluated by research, not execution).

## Config diff

Hotstart-based, step-based schedule. Single 8×H100 pod runs both phases sequentially.

### Phase 1 (variant A)

| Env var | spec 006 | **Phase 1** |
|---|---|---|
| `WARMDOWN_FRAC` | 0.72 | **0.72** (same) |
| `ITERATIONS` | 4550 | **4550** (same) |
| `MAX_WALLCLOCK_SECONDS` | 0 | **0** (same) |
| `TRAIN_LOG_EVERY` | 5 | **5** (same) |
| `VAL_LOSS_EVERY` | 100 | **100** (same) |
| `CKPT_STEPS` | 100,200,...,4500 | **empty** (only auto-finals) |
| `CKPT_DIR` | ...006... | `/workspace/runs/007a-hotstart-reproduce/checkpoints` |
| All others | same | same |

### Phase 2 (variant C)

| Env var | Phase 1 | **Phase 2** |
|---|---|---|
| `WARMDOWN_FRAC` | 0.72 | **0.4945** (→ warmdown_start ≈ step 2300) |
| `CKPT_DIR` | ...007a... | `/workspace/runs/007c-delayed-warmdown/checkpoints` |
| All others | same | same |

Full env (Phase 2 example):

```
BIGRAM_VOCAB_SIZE=0
QK_GAIN_INIT=5.25
TTT_ENABLED=1
SEED=42
ITERATIONS=4550
MAX_WALLCLOCK_SECONDS=0
WARMDOWN_FRAC=0.4945
TRAIN_LOG_EVERY=5
VAL_LOSS_EVERY=100
CKPT_STEPS=
CKPT_DIR=/workspace/runs/007c-delayed-warmdown/checkpoints
```

## Code changes

- Branch: `exp/hotstart-tail` (reused, commit `2cd0ff7`)
- Diff: **none additional.** Reuses existing `hotstart.py::cmd_resume` infrastructure.
- No new logging, no schedule code changes.

## Hardware ladder

- [ ] 8×H100 — only rung. Screen is cheap; no 2×H100 mini-test needed.

## Seed plan

Single seed (42). Screen is single-variant comparison vs single-variant baseline; multi-seed deferred to full-run spec if this shows signal.

## Inputs

- **Hotstart checkpoint:** `/workspace/runs/006-dense-ckpts/checkpoints/ckpt_event_step1200.pt`
  - From spec 006, on JP volume (`jlxvxeiol4`)
  - Contains model_state_dict + optimizer_states + ema_state (verified via spec 006 notes)
- **Data:** `/workspace/data/datasets/fineweb10B_sp8192/` (JP volume)
- **Tokenizer:** `/workspace/data/tokenizers/fineweb_8192_bpe.model` (JP volume)
- **Base repo commit:** `2cd0ff7` on `exp/hotstart-tail`

## Execution protocol

Single 8×H100 pod, sequential two-phase execution. Kill after Phase 2.

### Phase 1 launch

```bash
mkdir -p /workspace/runs/007a-hotstart-reproduce
cd /workspace/parameter-golf

BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1 SEED=42 \
ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_FRAC=0.72 \
TRAIN_LOG_EVERY=5 VAL_LOSS_EVERY=100 \
CKPT_STEPS= \
CKPT_DIR=/workspace/runs/007a-hotstart-reproduce/checkpoints \
torchrun --standalone --nproc_per_node=8 hotstart.py resume \
  --ckpt /workspace/runs/006-dense-ckpts/checkpoints/ckpt_event_step1200.pt \
  --steps 2300 \
  > /workspace/runs/007a-hotstart-reproduce/train.log 2>&1
```

### Phase 1 gate

After Phase 1 completes:
1. Extract final val_bpb from `/workspace/runs/007a-hotstart-reproduce/train.log` (line matching `2300/4550 val_loss: ... val_bpb: ...`).
2. Check: `abs(val_bpb - 1.1685) <= 0.003`.
3. **If PASS:** proceed to Phase 2 below.
4. **If FAIL:** do NOT launch Phase 2. Kill pod. Post to Discord with observed val_bpb and trajectory summary. Flag for research.

### Phase 2 launch (only if Phase 1 passes)

```bash
mkdir -p /workspace/runs/007c-delayed-warmdown
cd /workspace/parameter-golf

BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1 SEED=42 \
ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_FRAC=0.4945 \
TRAIN_LOG_EVERY=5 VAL_LOSS_EVERY=100 \
CKPT_STEPS= \
CKPT_DIR=/workspace/runs/007c-delayed-warmdown/checkpoints \
torchrun --standalone --nproc_per_node=8 hotstart.py resume \
  --ckpt /workspace/runs/006-dense-ckpts/checkpoints/ckpt_event_step1200.pt \
  --steps 2300 \
  > /workspace/runs/007c-delayed-warmdown/train.log 2>&1
```

### Kill protocol

After Phase 2's `ckpt_final_post_ema_step2300.pt` saves (watch log), `runpodctl pod stop $POD_ID`.

If Phase 1 fails, kill after Phase 1.

## Checkpoints to emit

| Source | Files |
|---|---|
| Phase 1 auto final_pre_ema | `007a-hotstart-reproduce/checkpoints/ckpt_final_pre_ema_step2300.pt` |
| Phase 1 auto final_post_ema | `007a-hotstart-reproduce/checkpoints/ckpt_final_post_ema_step2300.pt` |
| Phase 2 auto final_pre_ema | `007c-delayed-warmdown/checkpoints/ckpt_final_pre_ema_step2300.pt` |
| Phase 2 auto final_post_ema | `007c-delayed-warmdown/checkpoints/ckpt_final_post_ema_step2300.pt` |

~4 × 313 MB = ~1.3 GB total. Retention: keep through record-track push (2026-04-30).

## Stop-early criteria

- NaN in train_loss at any step → kill immediately, mark failed.
- Step time > 2× expected (~300ms/step vs expected ~150ms) → kill, investigate.
- DDP hang (no log activity for >60s after startup completes) → kill, investigate.
- Phase 1 val_bpb > 0.003 off spec 006's 1.1685 → stop the spec entirely (see Phase 1 gate).

## Cost estimate

| item | cost |
|---|---|
| Pod spinup | ~$2.00 |
| Phase 1 training (1100 steps, ~3 min wall) | ~$1.20 |
| Phase 2 training (same) | ~$1.20 |
| Inter-phase + finalization | ~$0.50 |
| **Total (both phases)** | **~$5** |

If Phase 1 fails and Phase 2 is skipped: ~$3.20 saved.

## Extra artifacts

- `train.log` files for each phase
- `notes.md` — execution narrative per phase (Phase 1 PASS/FAIL with val_bpb, any DDP anomalies, Phase 2 observations)
- Side-by-side val_bpb table at every 100 steps (1200-2300) for both phases — research will compile.

## Open questions for interview

- Confirm 8×H100 availability (any region — JP preferred since that's where spec 006 ckpts are).
- Confirm `ckpt_event_step1200.pt` is present and loadable on JP volume.
- Confirm `exp/hotstart-tail` branch can be checked out on the pod (commit `2cd0ff7`).
- Confirm DDP launch path uses `torchrun --standalone --nproc_per_node=8 hotstart.py resume` (not `python hotstart.py`). This is a small modification from the 1×H100 smoke test — execution should sanity-check that `cmd_resume` survives torchrun wrapping.
- Phase 1 gate logic: does execution have a clean way to grep val_bpb from the log and gate Phase 2 on it? (If not, spec to it manually — Phase 1 observation, ping research, research decides Phase 2.)

## What this spec does NOT do

- Does not run to step 4550 — screening only.
- Does not produce a `.ptz` submission artifact.
- Does not test variant B (warmdown_start ∈ {1700, 2000}). That's a follow-up if C signals positive-but-too-aggressive, or null.
- Does not emit intermediate checkpoints — only auto finals at step 2300.
- Does not compare against full-run end-of-training val_bpb (which is the actual record metric).
