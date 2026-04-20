# Spec 016 — Recur-Alpha with α=1 initialization

**Slug:** `recur-alpha-ones`
**Created:** 2026-04-21
**Links to idea:** `research/ideas/recurrence-parallel-literature.md`; follow-up to spec 015.

---

## Hypothesis

Spec 015 showed learnable α works (matched-step @4000 Δ = −0.0032 vs spec 008). But α=0 init imposes a ~400-step catch-up handicap post-looping-activation: α has to ramp from 0 to useful values before the extra loop passes contribute to loss reduction. During those ~400 steps, the model trains with effective NUM_LOOPS=0 on the looped layers.

Changing the init from `torch.zeros(...)` to `torch.ones(...)` eliminates the handicap: at looping-activation, behavior equals standard Loop345 (α=1 ≡ full-commitment extra passes, matching the baseline non-learnable recurrence). Model can then drift α upward (toward 015's pass-2 saturations of 1.04/1.16/1.38) or downward (toward pass-3's 1.01/0.89/0.77) from a no-handicap starting point.

## Baseline

Spec 008 seed-42 val_bpb = **1.0697** endpoint, matched-step @4000 = 1.1110 (`runs/008-1736-reproduction/seed_42/final.json`).
Spec 015 seed-42 val_bpb = 1.0696 endpoint (null — short 67 steps), matched-step @4000 = **1.1078** (`runs/015-recur-alpha/seed_42/final.json`).

Primary comparison: **spec 016 matched-step @4000 vs spec 015 matched-step @4000** (both same-recur-alpha-mechanism, differ only in init). Secondary: endpoint val_bpb vs spec 008.

## Expected Δ

- **−0.001 to −0.002 vs 015 @4000** (best case): catch-up handicap recovered, plus ramp-wallclock re-routed to converged training.
- **Null (±0.0005 vs 015)**: init doesn't matter much — α trajectory reaches same plateau either way.
- **Regression vs 015**: surprising. Would indicate α=0 → α=learned has some useful training-dynamics side effect (e.g. initial zero-contribution phase lets non-looped layers specialize on the low-capacity regime).

## Thoughts

### Expected α trajectory vs 015

Starting at α=1.0 everywhere instead of 0:
- pass2_L3 (015 final 1.04): barely moves (~+0.04).
- pass2_L4 (015 final 1.16): small upward drift (~+0.16).
- pass2_L5 (015 final 1.38): larger upward drift (~+0.38).
- pass3_L3 (015 final 1.01): near-noise motion.
- pass3_L4 (015 final 0.89): small downward drift (~−0.11).
- pass3_L5 (015 final 0.77): larger downward drift (~−0.23).

Gradient magnitudes should be smaller overall than 015's initial ramp (since we're starting closer to the optimum), so motion should be gentler and plateau reached sooner (expected ~step 2300, vs 015's ~step 2500).

### What makes this safe

- **α=1 ≡ baseline Loop345.** Worst-case trajectory: α stays at 1.0 forever → run is literally spec 008 behavior. The α=0 run's worst case was "effective NUM_LOOPS=0" (a novel unstable configuration); α=1's worst case is the known-good baseline.
- Same 6 scalars, same optimizer routing, same logging — only the init value changes.

### Diagnostic value

Three outcomes and their interpretations:
| Outcome | Interpretation |
|---|---|
| Δ < −0.001 vs 015 @4000 | Handicap was real and removable. α-init matters. |
| Δ ≈ 0 vs 015 | Trajectory converges regardless of init. The plateau is the attractor. |
| Δ > +0.001 vs 015 | Unexpected — α=0 starting phase has some useful side effect worth investigating. |

Any outcome is informative.

## Accept criteria

- Training completes without NaN / divergence.
- Endpoint val_bpb captured.
- `final.json` includes top-level `recur_alpha_final` field (convention codified this spec onward).
- **Decision criterion (matched-step @4000 Δ vs spec 015):**
  - Δ ≤ −0.001 → promote to 3-seed + full-pipeline (TTT+GPTQ) for submission-quality number
  - Δ ∈ (−0.001, +0.001) → null, α-init doesn't matter; document and move on
  - Δ > +0.001 → regression, investigate before further α-init experiments

## Config diff vs spec 015

None. Same env vars. The only change is a one-line code patch: α init `torch.zeros(...)` → `torch.ones(...)`.

## Code changes

- **Branch:** `exp/recur-alpha-ones` (worktree at `worktrees/recur-alpha-ones/`). Forks from `research` (commit `304c552`, includes serialize-fix so `final_model.pt` lands unconditionally) + cherry-pick of `a9aa141` (spec 015's recur-alpha patch) + new commit `4dd2d63` with the init change and grad-norm logging fix.
- **Commit:** `4dd2d63` on `fork/exp/recur-alpha-ones`.
- **Patch scope:** 1 line change vs spec 015's `a9aa141`.
  ```python
  # In GPT.__init__:
  - self.recur_alpha = nn.Parameter(torch.zeros(num_loops, num_looped, dtype=torch.float32))
  + self.recur_alpha = nn.Parameter(torch.ones(num_loops, num_looped, dtype=torch.float32))
  ```

### Emit `recur_alpha_final` in final.json

Convention codified this spec onward. The run's final α values should appear as a top-level field in `final.json`:
```json
"recur_alpha_final": [[1.04, 1.16, 1.38], [1.01, 0.89, 0.77]]
```
This happens in execution's post-run final.json composition (not in train_gpt.py). Execution already did this for spec 015; we're standardizing it.

## Hardware ladder

- [ ] **2×H100 smoke** — SKIP (can cite spec 015's smoke `a9aa141` as last-clean validation; only the init value differs, no new code paths). If we want a smoke anyway for safety, run with `ITERATIONS=500 ENABLE_LOOPING_AT=0` (forces looping active from step 1 so α trajectory is observable in smoke window).
- [ ] **8×H100 screening run** (~$5, seed 42): endpoint + matched-step @4000 val_bpb, no TTT/GPTQ. Primary measurement.
- [ ] **(Conditional)** If screen shows Δ ≤ −0.001 vs spec 015 @4000 → **3-seed confirmation** (~$15) + **8×H100 full run with TTT/GPTQ** (~$20) for submission-quality number.

## Seed plan

Single seed (42) for screen. 3-seed (42/43/44) confirmation only if screen beats spec 015 by ≥0.001 @4000.

## Inputs

- Data: same CaseOps dataset as spec 008/015.
- Tokenizer: bundled with #1736 submission dir.
- Hotstart: none, full from-scratch training.

## Execution protocol

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/016-recur-alpha-ones/seed_42
mkdir -p /workspace/.torch_inductor_cache

NCCL_NET=Socket DATA_DIR=/workspace/data \
ARTIFACT_DIR=/workspace/runs/016-recur-alpha-ones/seed_42 \
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
  > /workspace/runs/016-recur-alpha-ones/seed_42/train.log 2>&1
```

Expected startup log line (unchanged from spec 015):
`recur_alpha: enabled=True num_loops=2 loop_start=3 loop_end=5 diag_p2p_cos=False`

Expected at step ~2200 (first log entry post-activation):
`recur_alpha: values=[[1.01, 1.03, 1.06], [1.00, 0.98, 0.94]] grad_norm=...`
(small motion already underway, not the rapid ramp seen in 015.)

## Checkpoints / artifacts to emit

Inherited from baseline (post-commit `304c552`):
- `final_model.pt` — pre-GPTQ FP state dict, post-EMA. Standard, reusable for post-hoc TTT/GPTQ.
- `train.log` — includes per-step α trajectory.
- `final.json` — must include `recur_alpha_final` top-level field.
- `notes.md` — execution narrative.

No intermediate model checkpoints.

## Stop-early criteria

Unconditional (always halt):
- NaN / inf in train_loss → halt.
- Step time > 2× spec 008 → halt (compile failure / unexpected overhead).

Conditional on `looping_active=True` (step ≥ ~1700 at default 0.35):
- Training loss > spec 008's matched-step loss + 0.03 for 5+ consecutive log entries → halt. (Unlike α=0, α=1-init should match or beat baseline from activation onward; a sustained loss gap indicates the one-line change somehow broke convergence.)

**Notes on logging bug from spec 015:** `recur_alpha grad_norm` was logged as 0.0 every step because the log fires after `optimizer.zero_grad()`. This is cosmetic, not broken plumbing (α values clearly moved in 015). Do NOT halt based on grad_norm=0 in this run either. If there's time to patch the log to snapshot grad pre-zero, do so in a separate cleanup commit before 016; if not, tolerate the cosmetic 0.

## Cost estimate

| Item | Cost |
|---|---|
| 8×H100 screen (primary) | ~$5 |
| **First-pass total** | **~$5** |
| (Conditional) 3-seed confirmation | ~$15 |
| (Conditional) 8×H100 full run with TTT/GPTQ | ~$20 |

## Open questions for interview

1. **Should we patch the grad_norm logging bug before launching?** ~5 min research. Clean-up; not load-bearing for this spec's decision. Probably yes.
2. **Do we want to piggyback a 015 seed-43/44 confirmation pod in parallel?** Two seeds * $5 = $10, gives us 015 3-seed shape confirmation alongside 016's init-test, totals ~$15 vs $5 alone but much higher information yield.
3. **If 016 matches 015 exactly (null):** do we bother with Option B (fixed α at learned shape, removing the 6 scalars as a zero-param architectural move)? Only worth it if 015 3-seed shape is consistent.

## What this spec does NOT do

- Does not change recurrence position, schedule, or layer range.
- Does not add parameters (still 6 α scalars, same as spec 015).
- Does not implement cross-pass XSA (deferred to a follow-up spec if this run + 015 3-seed motivate it).
- Does not run TTT/GPTQ on the primary screen — those are screening-mode skipped per `feedback_screen_via_training_endpoint_val.md`. Full pipeline is conditional on screen outcome.
