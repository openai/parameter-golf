# Spec 006 — dense-checkpoint full-curve run for flat-zone and training-dynamics analysis

**Slug:** `dense-ckpts`
**Created:** 2026-04-20
**Links to idea:** (flat-zone investigation — idea file will be written after analysis)

## Hypothesis
We can diagnose the reproducible step-2000-2500 flat zone (and more broadly understand full training dynamics) by collecting model checkpoints at 100-step resolution across the whole training run. Windowed weight-delta analysis on these checkpoints will distinguish:
- Cause A (post-recurrence loop-layer adaptation)
- Cause B (LR-schedule artifact at specific LR regimes)
- Cause C (data-order coincidence)

Task A's coarse analysis (2 windows, 500-step resolution) was inconclusive: showed loop layers move ~7-8% more but couldn't cleanly isolate the flat-zone window. Dense checkpoints fix the resolution gap.

## Baseline
Not a bpb-comparison spec. No val_bpb target. Existing spec 000's checkpoints at steps 1500, 2275, 3412 will be the coarse-resolution reference; this run extends to 100-step resolution.

## Expected Δ
N/A (data-gathering spec, no model-quality metric).

## Accept criteria
- All ~49 checkpoints land on NA-1 volume (45 explicit + 4 auto).
- `train.log` contains per-step train_loss (TRAIN_LOG_EVERY=1) and per-layer grad norms at every step.
- Training reaches step 4550 (natural completion via step-based schedule).
- No NaN, no obvious anomaly.
- Pod killed immediately after `ckpt_final_post_ema_step4550.pt` save (skip post-training eval).

## Config diff
**Step-based schedule (not wallclock-based)** — this is the key config change from spec 000.

| Env var | Spec 000 | **Spec 006** |
|---|---|---|
| `ITERATIONS` | 20000 (default, capped by wallclock) | **4550** (match SOTA's actual step count) |
| `MAX_WALLCLOCK_SECONDS` | 600 | **0** (disables wallclock cap → step-based `frac`) |
| `CKPT_STEPS` | 455,1137,2275,3412 (4 sparse) | **100,200,...,4500** (45 every-100) |
| `TRAIN_LOG_EVERY` | 500 (default) | **1** (per-step loss, micro-structure) |
| All others | (spec 000 values) | same as spec 000 |

Full env:
```
BIGRAM_VOCAB_SIZE=0
QK_GAIN_INIT=5.25
TTT_ENABLED=1
SEED=42
ITERATIONS=4550
MAX_WALLCLOCK_SECONDS=0
TRAIN_LOG_EVERY=1
VAL_LOSS_EVERY=4000
CKPT_DIR=/workspace/runs/006-dense-ckpts/checkpoints
CKPT_STEPS=100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500
```

**Schedule events will fire at step-based fractions:**
- `warmdown_start`: step ~1274 (frac=0.28)
- `recurrence activates`: step ~1593 (frac=0.35)

These shift from spec 000's wallclock-based positions (1048, 1378). Flat zone location likely shifts proportionally — dense CKPT_STEPS capture it regardless of where it lands.

## Code changes
- Branch: `exp/dense-ckpts-grad-logging` (forked from `research` at `503a116`)
- Tiny diff (~5 lines) in `train_gpt_sota.py` around line 1262:
  - Capture the return value of `torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)` (this is the pre-clip total grad norm).
  - Additionally compute per-layer grad norms (loop over `base_model.transformer.h` — sum of param.grad.norm()^2 per block, sqrt at the end).
  - Log both to `train.log` at every step where `should_log_train` fires (which, with TRAIN_LOG_EVERY=1, is every step).
- Execution: please confirm the diff is ≤10 lines, no control-flow changes, purely observational. Pin the resulting commit hash in the run's `notes.md`.

## Hardware ladder
- [ ] 8×H100 NA-1 — only rung.

## Seed plan
Single seed (42), matching spec 000.

## Inputs
- Data: `/workspace/data/datasets/fineweb10B_sp8192/`
- Tokenizer: `/workspace/data/tokenizers/fineweb_8192_bpe.model`
- Hotstart: none (fresh from-scratch)
- Base repo commit: `503a116` on `research`

## Execution protocol
```bash
mkdir -p /workspace/runs/006-dense-ckpts
cd /workspace/parameter-golf

BIGRAM_VOCAB_SIZE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1 SEED=42 \
ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=4000 \
CKPT_DIR=/workspace/runs/006-dense-ckpts/checkpoints \
CKPT_STEPS=100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500 \
torchrun --standalone --nproc_per_node=8 train_gpt_sota.py \
  > /workspace/runs/006-dense-ckpts/train.log 2>&1 &
```

### Kill protocol (CRITICAL — saves ~$4 of unneeded post-training eval)
Monitor `train.log`. Kill pod when this appears:
```
Checkpoint saved: .*ckpt_final_post_ema_step4550.pt
```

One-liner for execution:
```bash
tail -f /workspace/runs/006-dense-ckpts/train.log | \
  grep --line-buffered "ckpt_final_post_ema" | \
  (read line && echo "stopping pod" && runpodctl pod stop $POD_ID)
```

**Safety valve:** if `ckpt_final_post_ema` doesn't appear within 2 min of `peak memory allocated`, flag research, **don't auto-kill**. Something would be wrong.

## Checkpoints to emit
~49 total. All saved to `/workspace/runs/006-dense-ckpts/checkpoints/` on NA-1 volume.

| Source | Files |
|---|---|
| CKPT_STEPS (explicit, 45 files) | `ckpt_event_step{100,200,...,4500}.pt` |
| Auto: warmdown_start | `ckpt_warmdown_start_step{~1274}.pt` |
| Auto: pre_recurrence | `ckpt_pre_recurrence_step{~1593}.pt` |
| Auto: final_pre_ema | `ckpt_final_pre_ema_step4550.pt` |
| Auto: final_post_ema | `ckpt_final_post_ema_step4550.pt` ← kill trigger |

Each ~300 MB. Total ~14 GB. Retention: keep through record-track push (through 2026-04-30).

## Stop-early criteria
- NaN in train_loss at any step → kill, mark failed
- Step time > 2× expected (~300ms/step) → kill, investigate
- No checkpoints appearing at CKPT_STEPS values in the log → CKPT_DIR broken, kill and fix
- Standard: as above

## Cost estimate
- 8×H100 NA-1 at ~$23.92/hr
- Training compute: 4550 steps × ~150ms = ~682s
- Save overhead: 45 saves × ~3s = ~135s
- Provisioning overhead: ~5 min
- **Total wall: ~13-14 min = ~$5.50-6.00**
- Total including provisioning: **~$7.50**

## Extra artifacts
- `train.log` — full stdout with per-step train_loss + per-layer grad norms (~4550 rows)
- `notes.md` — execution narrative (was the kill trigger hit cleanly? any anomalies? pinned commit hash of the grad-logging diff)

No `final.json` (we killed before post-training eval).

## Open questions for interview
- Confirm 8×H100 NA-1 availability
- Confirm NA-1 volume has ≥15 GB free for checkpoint set
- Confirm kill protocol understood (watch log for final_post_ema, then stop pod)
- Confirm `ITERATIONS=4550` overrides the code default of 20000 (it's an env var)
- Confirm `MAX_WALLCLOCK_SECONDS=0` properly disables the wallclock cap (code reads `if h.max_wallclock_seconds > 0` → returns None)
- Confirm the per-layer grad-norm logging diff is acceptable and fits in ≤10 lines; share the patch back to research before launch if anything looks different than described

## What this spec does NOT do
- Does not produce a val_bpb — no eval runs
- Does not produce a `.ptz` submission artifact — no quantization runs
- Does not create an `exp/*` branch — hyperparam-only
- Does not test any intervention — pure data-collection for analysis
- Does not commit to a follow-up spec — spec 007 (weight-delta analysis on dense checkpoints) comes after artifacts are reviewed
