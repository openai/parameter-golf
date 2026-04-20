# Spec 015 — Recur-Alpha learnable per-pass blending (port from #1714)

**Slug:** `recur-alpha`
**Created:** 2026-04-21
**Links to idea:** `research/ideas/recurrence-parallel-literature.md`.

---

## ⚠️ CRITICAL FOR EXECUTION — READ BEFORE ACTING ON ANY LOG

**α is architecturally out-of-circuit until looping activates at `training_frac ≥ 0.35`.**

Concretely: at default config (ITERATIONS=20000, wallclock cap 600s, giving ~4828 actual steps):
- Steps **0 → ~1690**: looping_active=False, α is NOT in the forward graph, α.grad is None, **α grad_norm=0.0 is EXPECTED AND NORMAL.**
- Step **~1690**: looping activates, α enters the forward graph from this step onward
- Steps **~1690 → ~4828**: α.grad is non-None, α grad_norm should become non-zero if plumbing is correct

**DO NOT HALT OR FLAG THE RUN** during steps 0-1690 because of α grad_norm=0.0. That is expected behavior, not a bug. Only treat α grad_norm=0 as a failure signal **AFTER step ~1690** (post-looping-activation).

### Applies specifically to:

| Situation | α grad_norm=0 interpretation |
|---|---|
| Smoke with `ITERATIONS=500` and default `ENABLE_LOOPING_AT=0.35` | **NORMAL.** Smoke never crosses the 0.35 threshold at 500 steps. Expected. Do not halt. |
| Real 8H screen, steps 0-1690 | **NORMAL.** Pre-looping phase. Do not halt. |
| Real 8H screen, steps 1700+ with all-zero grad_norm for 5+ log entries | **BROKEN plumbing.** Halt and investigate. |
| Smoke with `ENABLE_LOOPING_AT=0` (override for plumbing check) | Should see **NON-ZERO** grad_norm almost immediately. If zero here, plumbing is broken. |

**Prior incident (2026-04-21):** A first smoke at `ITERATIONS=500, ENABLE_LOOPING_AT=0.35` was incorrectly flagged as failing the stop-early criterion. α grad_norm=0 was expected at that phase; the spec's stop-early wording didn't condition on looping_active. The smoke was actually FINE (500 iters no NaN, identity-at-init preserved). The experiment workflow was unnecessarily halted due to spec ambiguity. Fixed in this version of the spec — do not repeat.

---

## Hypothesis

In #1736's Loop345 (layers 3-5 × 3 passes = 17 virtual layers), every pass fully commits its block output to the residual stream — there's no learned control over how much each pass contributes. Recur-Alpha adds a learnable blend scalar per (non-first pass, looped layer), initialized to zero:

```
y = block(x_current)
x_new = α × y + (1 − α) × x_current
```

- At α=0: pass is pure passthrough (block output ignored, gradient blocked)
- At α=1: standard Loop345 behavior
- At α∈(0,1): partial commitment

The model learns α via gradient descent. If extra passes carry useful signal, α moves toward 1; if not, α stays near 0 (effectively "opting out" of recurrence on that pass).

**Source:** #1714 (Anakintano, Apr 18) tested this on a simpler pre-#1736 stack and got 1.0857 pre-TTT. Their compute grant ran out before phased TTT eval — so **Recur-Alpha's composition with #1736's full phased-TTT / CaseOps / gates / XSA stack has never been measured.** We are uniquely positioned to fill this gap.

## Baseline

Spec 008's seed-42 val_bpb (`runs/008-1736-reproduction/seed_42/final.json`) = **1.0697** (endpoint bare, screening mode).

## Expected Δ

Asymmetric outcome distribution:

- **−0.001 to −0.003 bpb** (best case): α moves to useful values, recurrence becomes more efficient
- **Null (±0.0005)**: α stays near 0 or 1 depending on what's optimal; no effective change
- **Very unlikely**: regression — identity-at-init + small param count blocks catastrophic pathways

Rationale for the thin range: #1714 got 1.0857 on a ~1.08 baseline, gap ~0.002. Porting onto #1736's stronger base retains at most half that gain due to (a) other architectural levers already capturing some benefit, (b) TTT absorbing upstream deltas per spec 010 finding.

## Thoughts (rationale + risks)

### Why this is the strongest remaining recurrence lever

Per `research/ideas/recurrence-parallel-literature.md`, three previously-proposed recurrence experiments have been **directly tested on this stack** and shelved:

- Earlier activation (#1726): 0.15 → +0.050 worse; #1739 step-0 catastrophic
- Smooth-vs-hard schedule (#1663): no difference
- Position shift / range expansion (#1726): layers 5-6 +0.006, 2-7 +0.163 worse

Recur-Alpha is the one recurrence-class lever that (a) has positive evidence elsewhere (#1714), (b) has NOT been composed with #1736's full stack, (c) has identity-at-init safety properties.

### What makes it safe

1. **Identity at init.** All 6 alphas start at 0. The first forward pass is behaviorally equivalent to baseline with all extra passes producing zero contribution (passthrough). If the model never learns to move α, worst case is "training under effective NUM_LOOPS=0 regime" — a known benign state.
2. **Small parameter count.** 6 scalars total, 24 bytes quantized. No artifact budget concern.
3. **No compute overhead.** Blend is 6 scalar-multiplies into the residual stream per forward. Negligible.
4. **Torch.compile friendly.** Python-level conditionals on precomputed static lists; integers known at trace time.

### What makes it interesting

Recur-Alpha turns "do we loop?" from a training-time hyperparameter (currently forced: NUM_LOOPS=2) into a *learned decision* per-pass. Three diagnostic outcomes:

| Final α pattern | Interpretation |
|---|---|
| All near 0 | Model opted out of recurrence; extra passes contribute nothing useful |
| All near 1 | Standard Loop345 behavior is already optimal; α flexibility unused |
| Mixed / intermediate | Model finds partial-commitment useful on some passes |

All three outcomes teach us something concrete about whether our current recurrence config is tuned.

### Optional p2p cosine diagnostic

Separately env-gated (`RECUR_DIAG_P2P_COS=1`). Computes cosine similarity between consecutive pass deltas for each looped layer, logged alongside α. Tells us whether pass outputs are pointing in similar directions (redundancy case → cross-pass XSA is the next research question) or diverse directions (no redundancy → different research direction).

**Off by default for this run.** Reason: the diagnostic uses a Python dict (`self._diag_prev_deltas`) mutated inside `forward_logits`, which may not compose cleanly with `torch.compile(fullgraph=True)`. If we want this data, we'd either flip the flag and accept potential compile fallback, or do a follow-up run.

## Accept criteria

- Training completes without NaN / divergence.
- α gradient norms are non-zero (optimizer is actually updating α).
- Endpoint bare val_bpb measured at `stopping_early: wallclock_cap`.
- **Decision criterion:**
  - Δ ≤ −0.001 → promote; optionally enable p2p diagnostic in follow-up; consider 3-seed confirmation + full TTT run
  - Δ ∈ (−0.001, −0.0003] → weak positive; weigh vs cost of 3-seed
  - Δ ∈ (−0.0003, +0.001) → null; shelve for this push, document α trajectory
  - Δ > +0.001 → regression (unexpected given identity-at-init); investigate

## Config diff vs spec 008

```
RECUR_ALPHA_ENABLED=1
TRAIN_LOG_EVERY=100   # increased from default 500 for diagnostic resolution
```

Optional:
```
RECUR_DIAG_P2P_COS=1  # off by default for this run, see reasoning above
```

No other changes.

## Code changes

- **Branch:** `exp/recur-alpha` (worktree at `worktrees/recur-alpha/`).
- **Commit:** `a9aa141`.
- **Patch target:** `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`.
- **Patch scope:** 132 insertions, 3 deletions (3 from cosmetic restructure of `x → x_before, x_new`). Components:
  - 2 new `Hyperparameters` fields
  - `GPT.__init__`: recur_alpha Parameter + precomputed encoder/decoder alpha_info lists (using shared visit-count state spanning encoder + decoder)
  - `forward_logits`: encoder loop + decoder single-lane path apply alpha-blending when configured
  - `Optimizers.__init__`: recur_alpha routed to scalar AdamW
  - Per-step logging of α values / grad norm / p2p cos (when enabled)
  - Startup log echoes config
- **Default-off invariant:** with `RECUR_ALPHA_ENABLED=0` (unset), all new code paths guard on `self.recur_alpha is None` and fall through to baseline logic. Verified byte-equivalent to original.

## Hardware ladder

- [x] **2×H100 smoke** (~5 min, ~$1): correctness only, 500 steps, watch for NaN + confirm α grad norms are non-zero. `ITERATIONS=500 RECUR_ALPHA_ENABLED=1 torchrun --nproc_per_node=2 train_gpt.py`. Do NOT read val_bpb.
- [x] **8×H100 screening run** (~$5, seed 42): endpoint bare val_bpb, no TTT/GPTQ/sliding. Primary measurement.
- [x] **(Conditional)** If screen shows Δ ≤ −0.001 → **8×H100 full run** (~$20) for TTT confirmation + proper submission number.

### Pre-registered expectations

Unlike BigramHash (zero-init projection, late divergence), Recur-Alpha's α starts at zero and:

| Step range | Expected behavior |
|---|---|
| 0–300 | train_loss near-identical to spec 008 at matched step (α=0 means no recurrence contribution) |
| 300–1500 | If recurrence is useful, α starts drifting from 0; small grad norms build up |
| 1500–3500 | First real signal on whether model wants α>0. Check α values + p2p cos. |
| 3500–4500 | α trajectory stabilizes; final values inform interpretation |
| Endpoint | Δ measured against spec 008's 1.0697 |

**Surprising would indicate a bug:**
- α moves to negative values (should converge positive if anything)
- α grad norms exactly zero for many steps (optimizer not registering)
- train_loss significantly worse than spec 008 in first 500 steps (identity-at-init should prevent)

### Early-stop guidance

Same joint-executor+user pattern as prior specs. Automatic kill on NaN / inf / step-time blow-up. Joint decision on "train_loss much worse than spec 008 across multiple late-training log entries." Default to finish when ambiguous — α=0 staying is an INFORMATIVE null, not a failure.

## Seed plan

Single seed (42) for screen. 3-seed confirmation only if Δ ≤ −0.001.

## Inputs

- Data: same CaseOps dataset as spec 008
- Tokenizer: bundled with #1736 submission dir
- Hotstart: none, full from-scratch training

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/015-recur-alpha/seed_42

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/015-recur-alpha/seed_42 \
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
  > /workspace/runs/015-recur-alpha/seed_42/train.log 2>&1
```

Expected startup log line:
`recur_alpha: enabled=True num_loops=2 loop_start=3 loop_end=5 diag_p2p_cos=False`

Expected log line at step 2000 (~example):
```
2000/20000 train_loss: 2.85 train_time: 4.2m tok/s: 120000
recur_alpha: values=[[0.02, 0.01, 0.03], [0.00, 0.01, 0.02]] grad_norm=0.0008
```

## Checkpoints / artifacts to emit

- `final_model.pt` (pre-GPTQ FP) — standard, reusable for analysis
- `train.log` (~50 log lines with α trajectory)
- `screen_endpoint.txt` snapshot
- `notes.md` execution narrative

**No intermediate model checkpoints** for this first run. Can add if α trajectory reveals something requiring mid-training inspection.

## Stop-early criteria (ALL conditioned on looping being active)

Unconditional (always halt):
- NaN / inf in train_loss → halt
- Step time > 2× spec 008 → halt (indicates compile failure / unexpected overhead)

Conditional on `looping_active=True` (roughly step ≥ 1700 at default 0.35):
- α grad_norm exactly 0.0 for 5+ consecutive log entries **AFTER looping activates** → halt, optimizer routing broken

**Pre-looping-activation (steps 0 to ~1690): do NOT halt based on α grad_norm.** Zero is expected. See the ⚠️ CRITICAL banner at the top of this spec.

## Smoke protocol (updated 2026-04-21 after first-smoke issue)

**The smoke must use `ENABLE_LOOPING_AT=0` to force looping active from step 1.** Otherwise α is out-of-circuit for the entire 500-iter smoke window and no α plumbing information is gained.

Smoke command:
```
ITERATIONS=500 RECUR_ALPHA_ENABLED=1 ENABLE_LOOPING_AT=0 TRAIN_LOG_EVERY=50 ...
```

Smoke pass criteria:
- No NaN / inf
- α grad_norm **non-zero** from early log entries (confirms autograd edge exists)
- α values slightly drifted from 0 (confirms optimizer applies updates)
- 500 iters complete cleanly

**Real screen** keeps `ENABLE_LOOPING_AT=0.35` unchanged — do NOT propagate the smoke override to the real run. Prior PG evidence:
- #1739 (step-0 activation): **1.3936 bpb catastrophic**
- #1726 (`ENABLE_LOOPING_AT=0.15`): +0.050 worse than 0.35
- The 0.35 default is empirically tuned and load-bearing for the main run.

## Alternative: skip second smoke, go direct to screen

Given the first smoke already verified compile + no-NaN + param count + identity-at-init, the α-plumbing question can be answered during the real screen itself:
- Watch α grad_norm across the first ~1800 steps (expected 0)
- Around step 1700, looping activates; α grad_norm should become non-zero within the next few log entries
- If α grad_norm stays 0 AFTER step 1700 (5+ consecutive log entries), plumbing is broken → halt

This path costs ~$5-6 for the screen vs ~$5-7 for another smoke + screen. Either path is valid.

## Cost estimate

| Item | Cost |
|---|---|
| 2×H100 smoke | ~$1 |
| 8×H100 screening run | ~$5 |
| **First-pass total** | **~$6** |
| (Conditional) 8×H100 full run with TTT | ~$20 |
| (Conditional) 3-seed confirmation | ~$30 additional |

## Open questions for interview

1. **Should we enable `RECUR_DIAG_P2P_COS=1` in this run?** Plan: off first time. Enabling might cause torch.compile fullgraph to fall back to eager mode (~15% slower). If we absolutely want the cross-pass XSA data from this run, flip it on and accept potential slowdown.
2. **Does `_diag_prev_deltas` dict-mutation work under torch.compile?** Unknown until tested. Smoke will reveal. If it crashes, disable p2p diag for the run.
3. **Is the compile time going to surprise us?** This patch adds a few Python-level conditionals to the forward. torch.compile has to re-trace. First-step compile time might be ~2x normal. Not a correctness issue; just a wallclock minute or two eaten upfront.

## What this spec does NOT do

- Does not change recurrence position (proven bad by #1726)
- Does not change activation schedule (proven bad/irrelevant by #1663/#1726)
- Does not implement cross-pass XSA (deferred as follow-up if p2p cos diagnostic reveals stationarity)
- Does not touch parallel residuals or skip connections
- Does not change the main model architecture in any way beyond adding 6 scalars + the blend op
- Does not run 3-seed (single-seed screen only)
