# Execution notes — spec 004

## Outcome
Ambiguous. Phase 1 (A/B triage) showed QK=6.0 at step 1000 was **Δ −0.109** vs spec 000 with matched RNG. Phase 2 (004b full run) showed endpoint val_bpb Δ **−0.00040** (tied) vs spec 000 — but with `VAL_LOSS_EVERY=200` vs spec 000's 4000, so RNG differed and the comparison was confounded. Verification run at matched VAL=4000 + full 10 min would cost ~$5 to resolve.

## Timeline (UTC, 2026-04-19)

### Phase 1 — A/B triage (same pod `ccy8ciuq8zkcxc`)

- `~01:10` pod created, preflight, git checkout feaf45e
- `~01:11` Run A launched: `QK_GAIN_INIT=6.0 TTT_ENABLED=1 SEED=42 TRAIN_LOG_EVERY=200 MAX_WALLCLOCK_SECONDS=300`
- `~01:11:45` step 200 hit: 3.6744 (variant) vs no spec 000 data at step 200. Spec's "kill at step 200 if >3.5" gate fired — but threshold was mis-specified (Exp 24 step 200 was 3.6762, so 3.67 is normal). Ignored.
- `~01:16` Run A stopping_early at step 1700 / 288s train
  - Step 1000: 3.1394 (vs spec 000 3.2487 → Δ **−0.109**)
  - Step 1500 interp: 3.0321 (vs 3.0884 → Δ **−0.056**)
- `~01:17` Run B launched: `QK_GAIN_INIT=5.5 ...`
- `~01:22` Run B stopping_early at step 2246 / 288s train (benefits from warm compile cache)
  - Step 1000: 3.2321 (Δ **−0.017**)
  - Step 2000: 2.8529 (Δ **−0.090**)
- `~01:23` pod stopped + deleted

Phase 1 cost: **$5.70**. At this point we thought QK=6.0 was a monotonic win and drafted a spec 005.

### Phase 2 — full QK=6.0 run, ad-hoc (pod `waby1c846taown`)

Requested by user mid-session: "run QK=6.0 with checkpoints, only training no post-training, a good 10 minutes, log the end, I want more detail logs."

Launch config (deviations from Phase 1 Run A highlighted):
- `QK_GAIN_INIT=6.0 SEED=42 BIGRAM_VOCAB_SIZE=0` ✓ same as Run A
- `TTT_ENABLED=0` ← disabled (user asked to skip post-training)
- `TRAIN_LOG_EVERY=100` ← denser logging for visibility
- **`VAL_LOSS_EVERY=200`** ← denser val logs too — THIS WAS THE MISTAKE
- `MAX_WALLCLOCK_SECONDS=600` ← full 10 min
- `CKPT_DIR=...` `CKPT_STEPS=455,1137,2275,3412` ← phase-boundary checkpoints

The `VAL_LOSS_EVERY=200` deviation from spec 000's default of 4000 broke the RNG match. At step 1000, 004b saw different training batches than spec 000 (and different from Run A). Step-matched comparison becomes noise-confounded. Caught this too late — should have grepped spec 000's log for `val_loss_every` before launch.

Kill-after-prequant watcher was also set up to skip GPTQ+quant+sliding, but polled every 15s — too coarse. Post-training stages flew through the 15s window, so quant + sliding ran anyway. Useful data (final quantized bpb 1.10407, sliding 1.08754) but not per original plan.

Pod vanished from runpod's API some time after the script finished. Couldn't confirm what killed it — possibly runpod auto-reclamation, possibly pod-level failure, possibly my earlier `pod list` call racing with a manual termination elsewhere. Spent $8.34 on the run before it disappeared. Spun a $0.03 recovery pod to rsync the log + confirm checkpoints on volume.

### Phase 2 results

- Step 3876 at 587.972s training time (vs spec 000's step 3849 at 588.040s — 27 steps further)
- Pre-quant post-EMA val_bpb: **1.09249499** (vs spec 000's 1.09288539, Δ **−0.00040**)
- Quantized val_bpb: 1.10406539 (vs 1.10430016, Δ −0.00023)
- Quantized + sliding-window: 1.08753702 (vs 1.08774104, Δ −0.00020)
- Artifact size: **16,046,371 bytes** — **46,371 over the 16,000,000-byte leaderboard cap**

Phase 2 cost: **$8.37** (pod + recovery).

## Key errors I made

1. **Set `VAL_LOSS_EVERY=200` without checking spec 000's value.** Spec 000 used the code default (4000). Result: broke the RNG-matched comparison between 004b and spec 000.

2. **Kill watcher polled every 15s** — too coarse to fire between `pre-quantization post-ema` (eval_time 6s) and GPTQ start. Should've been 2-3s poll if we wanted tight control. Pipeline ran quant+sliding before kill fired. Not catastrophic (got more data), but wasted ~$1-2 of pod time.

3. **Wrong spec-000 reference numbers in monitor scripts** for steps 2500/3000/3500 (used values ~0.03-0.05 lower than real). Made 004b look WORSE in mid-training than it really was. User caught this.

4. **Spec 004's step-200 stop-early threshold was too aggressive** (set to 3.5, but reality at step 200 across the architecture is ~3.67). I ignored it at runtime but worth flagging back to research.

## Running cost total

- Phase 1: $5.70
- Phase 2: $8.37
- Total spec 004: **$14.07**
- Balance now: $56.10 (there was a Runpod refund or top-up mid-session — from $6.22 → $56.10)

## Artifacts synced back to repo

In `runs/004-qk-gain-extension/`:
- `qk_6.0_train.log` (Run A, Phase 1, 5-min cap)
- `qk_5.5_train.log` (Run B, Phase 1, 5-min cap)

In `runs/004b-qk6-full/`:
- `train.log` — full 10-min Phase 2 run, with post-training eval output

On NA-1 volume (not synced):
- `/workspace/runs/004b-qk6-full/checkpoints/` — 9 phase-boundary checkpoints (~2.7 GB). Hotstart-ready for any future QK=6.0-based experiment.

## Lessons I'll carry forward

- **Always grep the baseline log for ALL env vars** (esp. `val_loss_every`, `train_log_every`, `max_wallclock_seconds`, `train_batch_tokens`) before launching a comparison run. Don't rely on the spec or memory — read the log.
- **Kill watchers should poll at ~2-3 second cadence** if they need to catch narrow post-training windows. 15s is fine for long training phases; not for post-training pipeline transitions.
- **Bake spec-000 reference values into comparison tooling via `grep` at build time, not typed from memory.** My monitor script had hardcoded dict values I'd typed by hand that turned out to be slightly wrong.
- **Add a `VAL_LOSS_EVERY` override to spec 000's `final.json`** so future specs can read it programmatically. (Research's call, but would help.)

## Handback

Research: spec 004 evaluation is your call (see `summary.md`). If you want to resolve the A/B-vs-004b conflict, verification run is ~$5. Otherwise "size overage + tied endpoint → kill" is defensible.
