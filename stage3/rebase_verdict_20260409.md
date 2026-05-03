# Rebase Verdict — 2026-04-09

This is a concrete triage of the currently planned experiment families after the April 6 PR update in [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md).

## Rule

Each planned family gets one of four tags:

- `keep`
  - still worth running in essentially the same form
- `demote`
  - still real, but no longer a lead frontier hypothesis
- `kill`
  - should stop being scheduled in its current form
- `rebase_required`
  - mechanism still makes sense, but only after moving onto the new frontier base

## Frontier Update

The new evidence is clear:

- the frontier is now `SP4096/SP8192 + full GPTQ + SDClip + GPTQ embeddings + depth recurrence + MuonEq-R`, then optional `TTT/ETLB/parallel residuals`
- our planning was still mostly centered on `SP1024` + old 11L static trunks

The most important lines in the PR analysis are:

- [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L56)
- [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L67)
- [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L184)
- [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L197)

## Hailmary

### Base Verdict

`rebase_required`

Reason:

- the active default is still `VOCAB_SIZE=1024` and `SP1024` data/tokenizer paths in [hailmary/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json#L15) and [hailmary/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json#L22)
- the PR analysis says tokenizer size alone is about `0.02-0.03` BPB at [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L197)
- that means the whole stage is starting from a base that is too far behind

### Pack Verdicts

- `phase_split`: `rebase_required`
  - The process-split idea still makes sense.
  - It should not be run on SP1024 as a lead pack.
- `checkpoint_selection`: `keep`
  - This still fits the new frontier because pre-quant TTT, ETLB, and deploy scoring all make checkpoint/state choice more important.
- `staged_curriculum`: `demote`
  - Loader/shuffle changes exist in the frontier, but they are not the main story.
- `alternating_objective`: `rebase_required`
  - Still plausible as a deploy-alignment helper, but secondary to full GPTQ + SDClip + recurrence.
- `moonshot_core`: `demote`
  - Too tied to the older SP1024 trunk.
- `moonshot_second_wave`: `demote`
  - Same issue; many ideas are still real, but no longer lead-stage.
- `moonshot_geometry`: `kill`
  - Geometry-only moonshots are too small relative to the new missing mass.
- `moonshot_throughput`: `demote`
  - Throughput matters, but now mainly as an enabler for recurrence, bigger vocab, and TTT.
- `parameter_family_split`: `keep`
  - Still a reasonable second-order lane after rebasing.
- `context_stage`: `demote`
  - The new PR analysis does not make context-stage switching look like a lead frontier story.

## Stage 3.2

### Base Verdict

`rebase_required`

Reason:

- defaults still point at `SP1024` in [stage3_2/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_2/run_configs.json#L13) and [stage3_2/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_2/run_configs.json#L24)
- the controller mechanisms are still conceptually good, but the stage base is outdated

### Family Verdicts

- `H201 Late Deploy Gate`: `keep`
  - Still matches the frontier because late deploy pressure and pre-quant TTT are clearly real.
- `H202 Best-State Controller`: `keep`
  - Still strong because the new frontier increasingly cares about deployed state, EMA, and pre-quant/post-quant staging.
- `H202B Best-State Raw`: `demote`
  - Still useful as a falsifier, but not a lead mechanism.
- `H204 Family-Split Warmdown`: `demote`
  - Plausible, but second-order relative to tokenizer, recurrence, and full GPTQ.
- `H205 Alternating Objective Controller`: `keep`
  - Still one of the better process-level ideas because it attacks late deploy alignment without poisoning the whole run.
- `H206 Systems-Aware Controller`: `demote`
  - Useful only as a support mechanism to keep expensive frontier mechanisms affordable.
- `H203 Curriculum-by-State`: `demote`
  - No longer a lead family under the current PR evidence.
- `H207 Context Budget Controller`: `kill`
  - No current support in the PR analysis as a strong frontier path.
- `H208 Composite Late Policy`: `rebase_required`
  - Still a good composite target, but only after the base is moved to Era 6.

## Stage 3.5

### Base Verdict

`rebase_required`

Reason:

- defaults still point at `SP1024` in [stage3_5/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_5/run_configs.json#L25)
- the new frontier is dominated by bigger vocab, recurrence, and stronger quantization, not by branch-tournament logic

### Family Verdicts

- `H501 Adaptive Tri Portfolio`: `demote`
  - Good idea, but now clearly a second-order exploitation stage.
- `H502 Scale-Gated Dual Deep`: `demote`
  - Same story; branch-depth tuning is not the main missing mass.
- `H503 Plateau-Gated Aggressive`: `keep`
  - Still interesting as a rebased stage because the new frontier explicitly supports harder late swings via better quantization and TTT.
- `H504 State-Style Tournament`: `keep`
  - Export-state choice still looks real and should matter more, not less, on a full GPTQ stack.
- `H505 Family-vs-Deploy Event Duel`: `demote`
  - Interesting diagnostic, but not a lead frontier mechanism.
- `H506 Failsafe Event Tri`: `demote`
  - More about trigger robustness than missing mass.

## What Is Actually Dead

These should stop consuming lead-stage attention in their current form:

- any SP1024-first frontier search
- geometry-only moonshot packs
- context-stage as a lead family
- branch-tournament logic as a replacement for tokenizer/recurrence/GPTQ modernization

## What Still Works

These mechanism classes still survive the new evidence:

- deploy-state selection
- late deploy gating
- alternating late objective pressure
- export-state tournaments
- aggressive late finishers, but only after rebasing

## Practical Priority

If the goal is to catch the frontier, the order should now be:

1. rebase the default stack to SP4096/SP8192 + full GPTQ + SDClip + GPTQ embeddings + depth recurrence + MuonEq-R
2. keep `stage3_2` controller families as support mechanisms on top of that base
3. keep `stage3_5` only as a late exploitation stage on top of that base
4. stop treating `hailmary` in its current SP1024 form as a lead moonshot

## Short Verdict

- `hailmary`: `rebase_required`
- `stage3_2`: `rebase_required`, but strongest surviving process stage
- `stage3_5`: `rebase_required`, with a few families worth preserving
- old SP1024-first frontier planning: effectively failed as a lead strategy

## Stage 3.1

### Base Verdict

`rebase_required`

Reason:

- defaults still point at `SP1024` in [stage3_1/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L18)
- the stage benchmarks are still anchored to the old `1.1194` frontier in [stage3_1/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_1/run_configs.json#L4)
- the main live lane is export/process ideas over an outdated base, while the new frontier moved hard on tokenizer, recurrence, GPTQ embeddings, MuonEq-R, and SDClip

### What Still Holds

- the hypothesis bar itself is still good and should be preserved
  - [hypothesis_stage_bar.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_1/hypothesis_stage_bar.md)
- the `Lane B export-only` framing is still conceptually useful

### Lane / Family Verdicts

- `H1 companding_mu100`: `demote`
  - Still a valid export-only curiosity.
  - No evidence from the latest PRs that companding is where the current missing mass is.
- `H2 fisher_bit_allocation`: `keep`
  - This is still one of the few `stage3_1` ideas that fits the Era 6 story because it attacks export allocation directly.
  - It should be rebased onto full GPTQ + embedding GPTQ + SDClip, not old int6 assumptions.
- `H3 sparsify_5pct`: `kill`
  - The latest frontier is winning on better modeling plus better GPTQ, not pre-quant sparsification.
  - This is too small and too likely to hurt quality.
- `H1H3 companding_plus_sparsify`: `kill`
  - Compound on top of a dead/weak parent.
- training-side cross-field slots `H4-H8`: `demote`
  - The stage currently frames them as interesting process imports, but none of them attack the newly dominant missing mass.
  - They are no longer credible lead-stage candidates.

### Rework Needed

`stage3_1` should only survive if it is narrowed into:

1. true export-only bakeoffs on the new base
2. byte-neutral export allocation ideas
3. direct comparisons against full GPTQ + SDClip + embedding GPTQ

Otherwise it should not be scheduled as a main stage.

## Stage 3.3

### Base Verdict

`rebase_required`

Reason:

- defaults still point at `SP1024` in [stage3_3/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_3/run_configs.json#L18)
- the stage is built around state-dependent hyperparameter schedules over the old 11L/SP1024 frontier
- the latest PRs do support some state-dependent schedule ideas, but only after adopting the stronger base

### Family Verdicts

- `H1 step_lr_schedule`: `demote`
  - Interesting, but there is no new evidence that schedule-shape alone is where the frontier gap now lives.
- `H2 throughput_ns_steps`: `kill`
  - The PR analysis points to bigger throughput levers like fused kernels and parallel residuals, not Newton-Schulz retiming on this old base.
- `H3 three_phase_seq_curriculum`: `kill`
  - Context/seq-length curriculum is not supported as a strong current frontier path.
- `H4 per_family_warmdown`: `demote`
  - Still plausible, but second-order.
- `H5 velocity_warmdown_gate`: `keep`
  - This is one of the better survivors because the latest frontier does increasingly care about when late behavior changes.
- remaining adaptive-hyperparam families in this stage: `demote` unless they directly support deploy-state timing or recurrence timing

### Rework Needed

`stage3_3` should be rebuilt around:

1. state-conditioned controls for a rebased Era 6 stack
2. schedule gating specifically for recurrence / GPTQ / TTT transitions
3. throughput gating only where it preserves stronger mechanisms

As a pure “adaptive hyperparams on SP1024” stage, it is behind.

## Stage 3.4

### Base Verdict

`rebase_required`

Reason:

- defaults still point at `SP1024` in [stage3_4/run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3_4/run_configs.json#L25)
- the stage premise is still a late-branching exploitation stage over a static old trunk
- the latest PRs imply that the main missing gains are earlier and more structural: bigger vocab, recurrence, better quantization, and deploy-aware TTT

### Family Verdicts

- `H401 tri_branch_default`: `demote`
  - Sensible branch baseline, but not a lead frontier attack now.
- `H402 earlier_tri_branch`: `demote`
  - Mostly timing refinement inside a now-second-order stage.
- `H403 late_dual_branch`: `demote`
  - Good ablation, not a lead path.
- `H404 deploy_vs_family_duel`: `demote`
  - Useful diagnostic only.
- `H405 raw_ema_deploy_triple`: `keep`
  - Export-state style still looks real and should matter more on a stronger GPTQ stack.
- `H406 aggressive_tri_branch`: `keep`
  - Still worth preserving because the new frontier does support harder late swings when quantization is strong enough.

### Rework Needed

`stage3_4` should not be deleted, but it should be repositioned:

1. make it a late exploitation stage on top of a rebased SP4096/SP8192 + recurrence + full GPTQ trunk
2. narrow it to export-state and aggressive late-branch families
3. stop treating branch timing/breadth variants as if they are frontier-catching by themselves

## Summary For These Three Stages

- `stage3_1`: mostly too old; keep only the strongest export-allocation ideas
- `stage3_3`: keep only the timing/gating ideas that can be rebased onto the Era 6 stack
- `stage3_4`: keep as a late exploitation stage, not a lead frontier stage
