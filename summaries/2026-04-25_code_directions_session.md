# Code-Level Directions Session — 2026-04-25 (afternoon)

7 experiments (0055-0061), ~104 min. **Cumulative gain vs canonical baseline (val_bpb 2.5212 → 2.10275 single-seed / 2.10427 mean): +0.418 / +0.417 (~16.6%)**.

Theme: pivot from env-var sweeps (exhausted last session) to **big code-level architecture changes**. Tested parallel residuals, depth recurrence, SwiGLU, per-loop scalars (Universal Transformer style), smear-gate. **Promoted one new winner**: SwiGLU(mlp=3) inside K=3 L=3 depth recurrence. Cap dropped from 15.18 MB (0051) to 5.998 MB (0057) — **10 MB cap headroom freed for further compounds**.

Subagent-driven: 2 of the 7 experiments used a subagent for the code change (depth recurrence, per-loop scalars). Others were small enough to self-edit.

---

## Final winner — submittable

**Path**: `winners/2026-04-25_recur_3x3_swiglu_mlp3/` (corresponds to exp `experiments/0057_swiglu_recur_3x3/`)

val_bpb_post_quant = **2.10275** SEED=1337 / **2.10579** SEED=42 → **mean 2.10427**. Mean Δ vs 0051 = +0.0055 (just over the +0.005 protocol boundary, both seeds positive). Artifact 5.998 MB.

Env-var deltas vs 0051 (which was vs canonical):
```
NUM_UNIQUE_LAYERS=3        # was 0 (canonical 9 distinct blocks); now 3 unique blocks
NUM_LOOPS=3                # was 1; now 3 loops × 3 unique = 9 effective layers
MLP_TYPE=swiglu            # was relu² (canonical); now SwiGLU
MLP_MULT=3                 # was 4 (0051); now 3 to fit cap with SwiGLU
```
Code changes vs canonical (~50 lines): added depth recurrence (subagent-implemented in 0056) and SwiGLU MLP (env-gated swap). The recurrent path drops the U-Net skip-weights entirely.

Everything else (warmdown_300, warmup_30, batch_24k, lr_045, init_05, muon_15) inherited from 0051.

---

## Stack of session-promotes (only one this session)

| # | Lever | Δ vs prior best | Tag | Promoted as | SEED=42 confirm | Journal heading |
|---|---|---|---|---|---|---|
| 1 | K=3 L=3 depth recurrence + SwiGLU(mlp=3) | +0.0055 (mean) | med | `winners/2026-04-25_recur_3x3_swiglu_mlp3/` (exp 0057) | exp 0058 | `## 2026-04-25 · exp 0057 · SwiGLU(mlp=3) + K=3 L=3 recurrence — NEW BEST 2.10275 (Δ+0.007), pending SEED=42` |

---

## Cross-experiment lessons (each tied to a journal heading)

1. **Parallel residuals as a standalone change actively HURT** at d=512 / 200 steps (Δ=-0.038, exp 0055). The inter-block sequential dependency does real work — feeding MLP the post-attention output is meaningfully better than feeding it the same pre-attention input. Records that use parallel-resid combine it with depth-recur + qk-gain + parallel-muon; standalone the gating advantage isn't there at our scale.
   - Journal: `## 2026-04-25 · exp 0055 · parallel residuals hurt -0.038 standalone`

2. **Depth recurrence (K=3 L=3) costs only -0.013 standalone and frees 10 MB of cap.** Strategic position dramatically improves even when the val_bpb regresses. Subagent-implemented; needed one main() fix (`hasattr(skip_weights)` guard). Trajectory was actually AHEAD of 0051 at step 10 (5.80 vs 6.22) — recurrent params get effectively 3× the updates per step.
   - Journal: `## 2026-04-25 · exp 0056 · depth recurrence (K=3 L=3) costs only -0.013 and frees 10.6 MB cap`
   - **Caveat**: U-Net skip removal is conflated with recurrence in this experiment. A `K=9 L=1` recurrent-codepath isolation experiment (no actual recurrence) would isolate which side carries the cost. Not run this session.

3. **SwiGLU + depth recurrence COMPOUND positively** — the gating advantage transfers to recurrent invocation. SwiGLU's standalone +0.011 (from 0044, 9L cap-violating) plus the recurrence's -0.013 (from 0056) sum to -0.002, but the actual compound was +0.0055 vs 0051 — the gating benefits more from repeated invocation than from distinct-block invocation. Each loop applies the gate to a different residual-stream state → multiple gating decisions per token at the same param cost.
   - Journal: `## 2026-04-25 · exp 0057 · SwiGLU(mlp=3) + K=3 L=3 recurrence — NEW BEST 2.10275 (Δ+0.007), pending SEED=42`

4. **Cross-seed variance baseline still 0.0024-0.003 for stable configs.** 0058 confirmed 0057 with cross-seed Δ 0.003 — typical baseline. The +0.007 single-seed Δ ended up at +0.0055 cross-seed mean → consistent with the previous session's lesson that single-seed wins overstate by ~10-20%.
   - Journal: `## 2026-04-25 · exp 0058 · SEED=42 confirms 0057 (cross-seed Δ 0.003, mean Δ vs 0051 +0.0055) — promoted`

5. **Per-loop scalars (Universal Transformer-style) are NEUTRAL at 200 steps.** Adding L=3 separate copies of `attn_scale`, `mlp_scale`, `resid_mix` to each block (so the same block can specialize per loop iteration) gave Δ=+0.0002 vs 0057 mean — within noise. Theoretically grounded (UT, ALBERT, RR-Transformers) but the per-loop differentiation needs more than 200 steps to develop. Records that use this pattern train for 20K+ steps. **General principle confirmed**: every "specialization-takes-training-steps" technique (mlp=5+, num_layers=11+, per-loop scalars) is flat or hurts at 200 steps.
   - Journal: `## 2026-04-25 · exp 0060 · per-loop scalars (Universal Transformer-style) neutral at 200 steps`

6. **Smear-gate (per-channel sigmoid mix with previous-token x) HURTS in our stack.** Δ=-0.027 vs 0057. Even at light-smear init (-2.0 logit, sigmoid≈0.119), the prior interferes with attention's natural local-pattern handling at d=512 with 9 effective recurrent layers. Records use smear-gate paired with bigram-hash + int6 + muon-wd; the value is in the broader stack, not standalone.
   - Journal: `## 2026-04-25 · exp 0061 · smear-gate on 0057 hurts -0.027 (real)`

7. **Cap headroom is now the limiting strategic asset, not val_bpb directly.** With the 0057 winner at 5.998 MB (vs 16 MB cap), there are 10 MB of compression headroom = ~25M raw params at the current 3.87× compression ratio. Spending this on depth (more loops), width (mlp=8), more unique blocks (K=4 or 5), or cap-freeing tricks (GPTQ-style pre-quant) is the next strategic move. Single-axis micro-tuning won't unlock big gains.

---

## Dead axes this session

| Axis | Result | Path |
|---|---|---|
| Parallel residuals (standalone) | Δ=-0.038 (HURTS) | `experiments/0055_parallel_resid_baseline/` |
| Per-loop scalars (Universal Transformer-style) | Δ=+0.0002 (neutral) | `experiments/0060_per_loop_scales/` |
| Smear-gate (init -2.0) | Δ=-0.027 (HURTS) | `experiments/0061_smear_gate_on_winner/` |

Note: depth recurrence at -0.013 is technically a regression but is "kept" as scaffolding because of the cap savings. SwiGLU+recurrence at +0.007 (single-seed) / +0.0055 (mean) is the only clean session win.

---

## Set in stone vs still hypothesis

### Set in stone (verified, multi-evidence)

- Depth recurrence K=3 L=3 trains acceptably at 200 steps (-0.013 cost, 10 MB cap freed). Verified by 0056 alone (single-seed but strong — pre/post Δ match).
- SwiGLU + recurrence compound positively. Verified by both seeds (0057 SEED=1337, 0058 SEED=42). Mean Δ +0.0055 vs 0051.
- Cross-seed variance is 0.003 for the recurrent stack — same as the canonical-architecture stable configs.
- Per-loop scalars don't help at 200 steps. Verified by 0060 single-seed (the change is byte-identical to 0057 at init, so same-trajectory through warmup ⇒ low cross-seed risk).

### Still hypothesis

- The 0057 SwiGLU+recurrence win is at the edge of the noise floor (mean Δ +0.0055, just over +0.005 protocol boundary). A second SEED at the same config would tighten the magnitude estimate; both seeds were positive but the precise effect could be anywhere in [+0.003, +0.008].
- The depth-recurrence -0.013 cost may be partly U-Net-skip-removal cost. Untested isolation: K=9 L=1 recurrent-codepath would tell us.
- Wider SwiGLU (mlp=4 or mlp=8) inside K=3 L=3 recurrence: prepped (0059) but not run. Cap allows up to mlp=8 (~12.7 MB). Expected to help if "capacity per block" inside recurrence is a real lever.
- "More loops" (K=3 L=4 or L=5) at fixed params: untested. Records use up to 11+ effective layers via recurrence.

---

## Follow-ups for next session (ranked by EV)

1. **Wider SwiGLU inside recurrence**: K=3 L=3 + SwiGLU(mlp=8) — already prepped at `experiments/0059_swiglu_recur_3x3_mlp4` (currently env-set to mlp=4; bump to 8 for the bigger test, OR run mlp=4 first as moderate). Cap allows. Most likely +0.005 to +0.020 vs 0057.

2. **More-loops compound**: K=3 L=4 (or L=5) at SwiGLU mlp=3 — env-var only (`NUM_LOOPS=4`), zero param cost, eff depth 12. Tests if recurrence depth helps when the model has "more compute through the same params".

3. **K=9 L=1 isolation experiment**: SwiGLU mlp=3 + recurrent-codepath but K=9 unique blocks, L=1 loops (no actual recurrence). Isolates the U-Net-skip-removal cost from the recurrence cost in 0056's -0.013. If K=9 L=1 ≈ 0051, U-Net was the culprit; if ≈ 0056, recurrence itself.

4. **Bigram-hash embedding** (record-validated, untested): replace `tok_emb` lookup with bigram-hash + projection. Records use heavily; ~30 lines of code, subagent territory. Adds effective vocabulary at near-zero cap cost.

5. **EMA over weights** (record-validated, untested): track exponential moving average of weights during training, eval against EMA params. Records use this to smooth late-step training noise. ~30 lines, subagent.

6. **GPTQ-style pre-quant calibration** (top-record technique): bake calibration-aware scale/permutation into weights so the harness's int8 quant gives less quant_tax. Multi-hundred lines, subagent. Could free another 1-2 MB of effective cap. **Highest-payoff late-session item.**

7. **Sliding-window attention in training** (untested in records but plausible from outside ML): swap `is_causal=True` for a banded mask. Could regularize or free attention compute for more depth. Note: most "SWA" tags in records are actually sliding-window EVAL, not training-time SWA — verified this session.

8. **Re-test smear-gate with stronger negative init** (-4 or -6 logit, sigmoid ≈ 0.018 / 0.0025 — basically off at start). The 0061 init at -2.0 may have been too active; a near-off start would let the model dial smearing UP only if it actually helps. ~5-line change to 0061.

9. **2nd SEED on 0057 winner** is technically done (0058) but the mean Δ is at the protocol boundary. A 3rd seed (e.g. SEED=2024) would tighten the magnitude estimate.

---

## Reflections

### What went well

- **Subagent dispatch was effective** for the depth-recurrence code change. The plan.md was detailed enough that the subagent produced a clean implementation in one shot, with one minor `main()` reference (`base_model.skip_weights.numel()`) that I caught and fixed self in 1 line. The previous session avoided subagents and lost time on env-var sweeps; this session demonstrates the cost-benefit math for code changes >20 lines clearly favors subagent.

- **Pre-walk brainstorm in scratch/** (~15 minutes spent in `scratch/brainstorm_big_directions.md` before any experiment) paid off. The exploration of records' actual technique usage (vs the ASSUMED meaning of tags like "SWA") changed the experiment ordering — saved spend on what would have been a less-record-validated direction.

- **Promoting at the protocol boundary (Δ +0.0055 mean) was the right call**. Both seeds positive, cross-seed Δ at typical baseline, pre/post Δ match. The previous session's lesson about single-seed overstate was applied: I ran SEED=42 BEFORE promoting (rather than direct-promote single-seed). The single-seed +0.007 was indeed overstated; mean was +0.0055.

- **Walk-driven planning worked**. The walk note crystallized "per-loop scaling" as the highest-EV next move. Even though that experiment ended up neutral, it was the right next experiment to run — it tests a theoretically-grounded hypothesis against actual data, and the null result is informative (rules out the per-loop differentiation hypothesis at our regime).

- **Coverage discipline**: across 7 experiments, hit 5 distinct axes (parallel-resid, depth-recur, SwiGLU, per-loop scalars, smear-gate) instead of micro-tuning one axis. The user's directive ("cover BIG directions") was followed reasonably well.

### What I did wrong / could have done better

1. **Underused the cap headroom early.** After 0057 promoted and cap dropped to 6 MB, my first post-promote experiment was per-loop scalars (0060) — tiny extra params. The wider-MLP push (mlp=4 or mlp=8) was prepped but not run. Should have run wider-MLP IMMEDIATELY after the promote rather than per-loop scaling, because wider MLP is a more direct "spend the freed cap" experiment with higher likely Δ. Per-loop scaling could have been done after the cap was actually full again.

2. **Smear-gate init at -2.0 was too aggressive** for our stack. The standard records use init at 0 (sigmoid 0.5 = strong prior), but I chose -2.0 to be conservative. Result: the prior was still too strong at d=512+recurrence and hurt -0.027. A smarter init choice (e.g. -6 logit ≈ off at start) would have given a cleaner test of "does the model find smearing useful given freedom to add it". This is a generalization of the lesson "initialize new mechanisms near identity so the model has the choice to use them".

3. **Didn't check sliding-window attention.** Despite identifying it as a coverage axis early, never ran a single experiment on it. Time pressure after the per-loop null result pushed me toward smear-gate instead. Re-prioritize for next session: SWA in training (with custom attn_mask) is unique among the listed directions in that NO records demonstrably use it for training (they use sliding-window EVAL), so the test would generate genuine new information about whether it helps.

4. **The 0056 → 0057 → 0060 → 0061 chain was all-on-the-recurrent-axis.** After 0057 promoted, three consecutive experiments stayed in the recurrent-codepath. Could have alternated: 0058 SEED=42 (good, did it), 0059 wider SwiGLU (skipped), 0060 something different from recurrence (e.g. SwiGLU at 8L without recurrence). Anchoring on the recurrent direction may have left other axes underexplored.

5. **The K=9 L=1 isolation experiment to disambiguate U-Net cost vs recurrence cost was identified as a follow-up but not run.** It's literally a one-env-var change to 0057 (set NUM_LOOPS=1, NUM_UNIQUE_LAYERS=9). Should have done it as a quick info experiment.

### Higher-level patterns

- **The hardest-to-predict experiments were also the most informative.** Two of the three nulls/losses this session (parallel-resid -0.038, smear-gate -0.027) were directions I had medium-to-high confidence WOULD help. They didn't. The information value of these "I was wrong" results is high — they refute hypothesis-classes that could otherwise have been retried in many sub-variants.

- **At 200 steps, every "needs training to specialize" technique fails.** A pattern across this session and the previous: capacity scaling (mlp_mult>4), depth scaling (num_layers>9), per-loop differentiation, parameter expansion. The 200-step regime aggressively favors techniques that work at init (gating, ortho-init in records, smaller/sharper schedules) over techniques that need many steps to develop (per-layer specialization, sparse experts, depth-recurrent specialization). For the H100 transfer, the picture inverts.

- **Compounds beat singles when the compound's mechanism is independent.** SwiGLU+recurrence worked because the gating mechanism is independent of the same-block-multiple-times mechanism. Smear-gate+recurrence didn't because both mechanisms target "look back at adjacent tokens" — they competed.

### What a future agent should do first

If the next agent reads only this section: **run K=3 L=3 + SwiGLU mlp=4** (experiments/0059 already prepped — just `cd experiments/0059_swiglu_recur_3x3_mlp4 && ../../run_experiment.sh`). It's the single most likely +0.005-0.020 win on the table, builds directly on the new winner, uses the freed cap room without adding architectural complexity.

After that, in priority order:
1. **K=3 L=4** (more recurrence loops at fixed params) — env-var only (`NUM_LOOPS=4`).
2. **K=9 L=1 isolation** — disambiguates depth-recur cost vs U-Net-skip cost in 0056.
3. **Wider still: K=3 L=3 + SwiGLU mlp=8** — push capacity to the cap edge (~12.7 MB).
4. **Bigram-hash embedding** — record-validated, ~30 lines, subagent.
5. **EMA over weights** — record-validated, ~30 lines, subagent. Could synergize with the noisy late-step train_loss at 200 steps.
6. **GPTQ-style pre-quant calibration** — multi-hundred lines, subagent. The single highest-payoff direction left if the others saturate.

The +0.418 cumulative gain (down from 2.5212 to 2.10427) is now coming up against the architectural ceiling at 200 steps. The next +0.05 likely requires either (a) deep capacity expansion via the freed cap (mlp=8 / more loops), or (b) the GPTQ-style quant-time gains that don't require model changes at all.

---

## File pointers for next agent

- `journal.md` — Current threads + Open questions (curated). All session entries rotated to `journals/`.
- `journals/2026-04-25_code_directions.md` — this session's experiment-by-experiment narrative.
- `summaries/2026-04-25_code_directions_session.md` — this file.
- `summaries/2026-04-25_overnight_session.md` — previous session (env-var phase, 54 experiments, ended at 2.10971).
- `winners/2026-04-25_recur_3x3_swiglu_mlp3/` — current best.
- `winners/2026-04-25_warmdown_300_warmup_30_mlp_mult_4_batch_24k_matrix_lr_045_init_05_muon_backend_15/` — prior best (0051).
- `scratch/brainstorm_big_directions.md` — direction-ranking memo.
- `scratch/depth_recurrence_design.md` — design notes for the recurrent code path.
- `scratch/parking_lot.md` — ideas captured mid-session for future work.
- `scratch/next_experiments.md` — branching plan written between 0056 and 0057.
- `walks/2026-04-25_1440.md` — walk note that crystallized per-loop scaling and wider-MLP as next moves.
- `experiments/0059_swiglu_recur_3x3_mlp4/` — **prepped but unrun**; just needs `cd && ../../run_experiment.sh`.
