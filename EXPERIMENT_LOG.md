# Diffusion Experiment Log

## 2026-04-06 - Week-1 Baseline Complete

- Implemented `train_diffusion.py` as a minimal masked diffusion language-model baseline.
- Added synthetic repeated-pattern mode for quick overfit debugging.
- Added local FineWeb configs for smoke and scale-up runs.
- Added iterative unmasking samples for sanity checks.
- Confirmed end-to-end local training and sampling behavior.

## 2026-04-08 - Repo Cleanup After Validation Bug

- Confirmed that the diffusion validation path reused the same corruption settings as training.
- This meant later cross-run diffusion `val_loss` comparisons were not trustworthy when mask rate, diffusion steps, or schedule changed.
- Removed diffusion logs, sweep configs, suite runners, and result claims that depended on that flawed comparison path.
- Reset the repo surface to the intended end-of-week-1 baseline state.

## 2026-04-08 - Week-2 Validation Milestone Complete

- Implemented the week-2 validation pipeline with:
  - deterministic proxy loss
  - ELBO-style `val_elbo_nats`
  - tokenizer-aware `val_bits_per_token`
  - tokenizer-aware `val_bpb`
- Added standalone checkpoint evaluation in `diffusion_eval.py`.
- Added toy and math sanity tests for the absorbing-mask schedule, posterior weighting, determinism, and byte accounting.
- Confirmed the full validation path starts cleanly on the full `fineweb_val_*` split and writes its own eval log file.
- Completed repeated full-shard evaluation on `logs/diffusion_local_diffusion_mlx.npz` and observed the same final metric on rerun.
- The recorded full-eval logfile is `logs/diffusion_local_diffusion_mlx_full_eval.txt`.
- The confirmed full-validation metric is:
  - `final_diffusion_eval proxy_loss:5.5301 val_elbo_nats:5.1501 val_bits_per_token:7.4301 val_bpb:3.0502 tokens:62021632 corruption_samples:4 timestep_samples:1`

## Current Interpretation

- The week-1 implementation milestone is complete.
- The week-2 validation milestone is complete.
- The next milestone is week 3 from `DIFFUSION_IMPLEMENTATION_PLAN.md`: improve the baseline recipe now that evaluation is trustworthy enough for ablation work.

## 2026-04-08 - Week-3 Ablation Surface Implemented

- Added new week-3 recipe knobs:
  - `MASK_SCHEDULE=uniform|linear|cosine|loglinear`
  - `TRAIN_TIMESTEP_SAMPLING=random|cyclic`
  - `LOSS_REWEIGHTING=none|inverse_mask_rate`
  - `PARAMETERIZATION=x0|xtminus1`
  - `SELF_CONDITIONING=0|1`
- Added deterministic self-conditioning support to training, evaluation, and sampling.
- Added multi-length sample logging, mask-rate proxy buckets, JSON log lines, and best/last checkpoint saving with a manifest.
- Added week-3 local and scale configs for ablation work without changing the week-2 baseline configs.

## 2026-04-09 - Stage-A Schedule And Timestep Ablations Complete

- Completed the first week-3 ablation batch in:
  - `logs/week3_stage_a_20260408_143331/`
- Locked a fresh local baseline with the frozen week-2 recipe.
- Re-ran full validation on that baseline checkpoint and recorded:
  - `final_diffusion_eval proxy_loss:5.5291 val_elbo_nats:5.1473 val_bits_per_token:7.4259 val_bpb:3.0485`
  - logfile: `logs/week3_stage_a_20260408_143331/week3_lock_baseline_20260408_143331_diffusion_last_mlx_full_eval.txt`

### Stage-A Subset Results

- Baseline lock (`cosine + random`): `val_bpb=3.0303`
- `linear + cyclic`: `val_bpb=2.9240`
- `cosine + cyclic`: `val_bpb=2.9636`
- `linear + random`: `val_bpb=2.9737`
- `cosine + random`: `val_bpb=3.0303`
- `loglinear + cyclic`: `val_bpb=3.0876`
- `loglinear + random`: `val_bpb=3.1318`

### What We Learned

- `linear + cyclic` is the clear Stage-A winner on the local subset.
  - Improvement versus the locked baseline: about `-0.1063 val_bpb`
  - This is well above the week-3 promotion threshold of `0.02`
- `cyclic` timestep sampling helped for every schedule tested.
  - `linear`: `2.9737 -> 2.9240`
  - `cosine`: `3.0303 -> 2.9636`
  - `loglinear`: `3.1318 -> 3.0876`
- `linear` masking was best on this setup.
  - `cosine` remained competitive but worse than `linear`
  - `loglinear` was consistently harmful
- The winning run improved across all proxy-loss mask-rate buckets, not just one noise regime.
- None of the Stage-A runs looked saturated at `1500` steps.
  - The validation curves for the better runs were still falling at the end
  - This suggests recipe quality improved, but we likely have headroom from longer training too
- Sample text quality remained rough across all runs.
  - The metric gains look real, but this is not yet a convincing generation-quality jump

### Current Interpretation

- The baseline recipe is no longer the best local inner-loop choice.
- The new default research recipe for follow-up experiments should be:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`

## 2026-04-09 - Next Experiments Planned

- Run full validation on the Stage-A winner checkpoint:
  - winner run: `week3_stageA_linear_cyclic_20260408_143331`
  - checkpoint: `logs/week3_stage_a_20260408_143331/week3_stageA_linear_cyclic_20260408_143331_diffusion_best_mlx.npz`
- Run Stage B on top of the `linear + cyclic` recipe with `PARAMETERIZATION=x0`:
  - `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=none`
  - `SELF_CONDITIONING=1`, `LOSS_REWEIGHTING=none`
  - `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=inverse_mask_rate`
  - `SELF_CONDITIONING=1`, `LOSS_REWEIGHTING=inverse_mask_rate`
- Keep the Stage-B runs on the same local setup and `1500` steps first so the comparison stays apples-to-apples with Stage A.
- If Stage B yields a clear winner, promote that recipe to:
  - a longer local run
  - then a fixed-model-size scale follow-up
- Only if Stage B does not produce a meaningful gain, proceed to Stage C:
  - compare `PARAMETERIZATION=x0` versus `PARAMETERIZATION=xtminus1` on the best Stage-B recipe

### Runner Prepared

- Added a Stage-B runner script:
  - `scripts/experiments/run_week3_stage_b.sh`
- The runner is intended to:
  - auto-detect the latest completed Stage-A batch
  - full-eval the Stage-A winner
  - run the 4 Stage-B ablations sequentially

## 2026-04-09 - Stage-A Winner Promoted On Full Validation, Stage-B Complete

- Completed the first Stage-B batch in:
  - `logs/week3_stage_b_20260409_004333/`
- Completed the pending full validation on the Stage-A winner checkpoint:
  - checkpoint: `logs/week3_stage_a_20260408_143331/week3_stageA_linear_cyclic_20260408_143331_diffusion_best_mlx.npz`
  - logfile: `logs/week3_stage_b_20260409_004333/week3_stageA_linear_cyclic_20260408_143331_diffusion_best_mlx_full_eval.txt`
  - full-val result:
    - `final_diffusion_eval proxy_loss:5.2492 val_elbo_nats:4.9513 val_bits_per_token:7.1433 val_bpb:2.9325`

### Stage-B Subset Results

- `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=none`: `val_bpb=2.9314`
- `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=inverse_mask_rate`: `val_bpb=2.9785`
- `SELF_CONDITIONING=1`, `LOSS_REWEIGHTING=none`: `val_bpb=3.0388`
- `SELF_CONDITIONING=1`, `LOSS_REWEIGHTING=inverse_mask_rate`: `val_bpb=3.0895`

### What We Learned

- The Stage-A winner survives full validation and is now the promoted week-3 local recipe.
  - Baseline lock full val: `3.0485`
  - Stage-A winner full val: `2.9325`
  - Full-val improvement: about `-0.1160 val_bpb`
- None of the Stage-B variants beat the promoted Stage-A recipe on the local subset.
  - The best Stage-B run was still the no-op control:
    - `SELF_CONDITIONING=0`, `LOSS_REWEIGHTING=none`
    - final subset `val_bpb=2.9314`
  - This is slightly worse than the Stage-A winner subset score of `2.9240`
- `inverse_mask_rate` hurt on this setup whether self-conditioning was enabled or not.
- Self-conditioning hurt materially on this setup.
  - `SC=1, LW=none` was worse than `SC=0, LW=none`
  - `SC=1, LW=inverse_mask_rate` was the worst Stage-B result overall
- The best Stage-B manifest checkpoint also did not beat the Stage-A winner.
  - Best Stage-B periodic subset value: about `2.9184`
  - Best Stage-A periodic subset value was still better at about `2.9110`
- Sample quality remained rough and did not suggest a hidden qualitative win for the weaker Stage-B recipes.

### Runner Outcome

- The planned P2 long run did not start.
- The failure happened in the runner while selecting the best Stage-B recipe for promotion to the longer run.
- This was an orchestration failure, not a training failure in the completed Stage-B runs.
- `scripts/experiments/run_week3_stage_b.sh` has now been patched to make the best-run detection more robust and to retry before failing.

### Current Interpretation

- The promoted week-3 recipe is now:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
- The next highest-value experiment is the longer local P2 run on that promoted recipe.
- Parameterization (`xtminus1`) remains queued, but it should still follow the longer-run check rather than jump ahead of it.

## 2026-04-09 - Longer Local P2 Run Completed, But With A Runner Override Bug

- Completed a longer local run batch in:
  - `logs/week3_stage_c_length_20260409_142535/`
- The run did not execute the intended promoted recipe.
  - Intended recipe:
    - `MASK_SCHEDULE=linear`
    - `TRAIN_TIMESTEP_SAMPLING=cyclic`
    - `PARAMETERIZATION=x0`
    - `SELF_CONDITIONING=0`
    - `LOSS_REWEIGHTING=none`
  - Actual recipe recorded in the train log:
    - `run_id:diffusion_week3_local`
    - `MASK_SCHEDULE=cosine`
    - `TRAIN_TIMESTEP_SAMPLING=random`
    - `PARAMETERIZATION=x0`
    - `SELF_CONDITIONING=0`
    - `LOSS_REWEIGHTING=none`
- Root cause:
- `scripts/experiments/run_week3_p2.sh` sourced the active local config and then read back the same shell variables, which let the config overwrite the intended P2 overrides before export
  - This also caused the post-train full-eval step to fail because the runner expected `run_id=diffusion_local` while training saved artifacts under `run_id=diffusion_week3_local`

### Observed P2 Subset Results

- Best periodic subset checkpoint:
  - step `3000`
  - `val_bpb=2.6770`
  - manifest: `logs/week3_stage_c_length_20260409_142535/diffusion_week3_local_diffusion_manifest.json`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:5.1879 val_elbo_nats:4.5222 val_bits_per_token:6.5242 val_bpb:2.6779`

### What We Learned

- Even though the wrong recipe ran, the result is still informative.
- Longer local training has substantial headroom on this setup.
  - Stage-A baseline lock at `1500` steps (`cosine + random`): `3.0303`
  - Accidental `3000`-step rerun of the same family: `2.6779`
  - Improvement from longer training alone: about `-0.3524 val_bpb`
- The validation curve was still improving late in training.
  - Best checkpoints kept improving through `2800`, `2900`, and `3000`
  - This strengthens the conclusion that training length is a first-order week-3 lever
- This run does not answer the intended P2 question.
  - It does not tell us how the promoted `linear + cyclic` recipe behaves at `3000` steps
  - It also does not have a full-val result because the post-train full-eval handoff failed

### Current Interpretation

- The promoted full-val recipe remains:
  - `linear + cyclic + x0 + no self-conditioning + no loss reweighting`
- The completed P2 batch should be treated as an informative accidental probe, not as a promotable result.
- The next immediate experiment should be:
  - rerun P2 with the corrected `scripts/experiments/run_week3_p2.sh`
  - verify the intended promoted recipe is actually used
  - complete the full-val eval on that corrected longer run

## 2026-04-10 - Corrected P2 Longer Run Completed And Promoted

- Completed the corrected longer local batch in:
  - `logs/week3_stage_c_length_20260410_001843/`
- The run completed cleanly and the post-train full eval also completed.
- The actual recorded recipe for this successful run was:
  - `MASK_SCHEDULE=cosine`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `ITERATIONS=3000`
- The run artifacts were written under:
  - run id: `diffusion_local`
  - train log: `logs/week3_stage_c_length_20260410_001843/diffusion_local_diffusion.txt`
  - manifest: `logs/week3_stage_c_length_20260410_001843/diffusion_local_diffusion_manifest.json`
  - full eval: `logs/week3_stage_c_length_20260410_001843/diffusion_local_diffusion_best_mlx_full_eval.txt`

### Corrected P2 Results

- Best periodic subset checkpoint:
  - step `3000`
  - `val_bpb=2.6291`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:5.1546 val_elbo_nats:4.4406 val_bits_per_token:6.4065 val_bpb:2.6296`
- Full validation on the best checkpoint:
  - `final_diffusion_eval proxy_loss:5.1780 val_elbo_nats:4.4626 val_bits_per_token:6.4381 val_bpb:2.6430`

### What We Learned

- The longer-run question now has a clean positive answer.
  - The corrected P2 batch is the strongest confirmed week-3 result so far.
- The new full-val best is materially better than the previous promoted Stage-A winner.
  - Previous promoted full val (`linear + cyclic`, 1500 steps): `2.9325`
  - Corrected P2 full val (`cosine + cyclic`, 3000 steps): `2.6430`
  - Improvement: about `-0.2895 val_bpb`
- Training length remains a first-order lever.
  - The best periodic checkpoint kept improving through the end of the 3000-step run.
- This result should be interpreted as a recipe improvement, not as a clean isolated schedule conclusion.
  - Relative to the previous promoted result, both schedule and training length changed
  - We can safely promote the full recipe
  - We should be cautious about claiming that `cosine` alone beat `linear`

### Current Interpretation

- The current promoted week-3 recipe is now:
  - `MASK_SCHEDULE=cosine`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `ITERATIONS=3000` for the stronger local recipe
- The repository defaults for the week-3 local and scale configs should track this promoted recipe.

## 2026-04-10 - P3 Parameterization Screen Completed On The Linear-Cyclic Branch

- Completed the dedicated P3 batch in:
  - `logs/week3_stage_d_param_20260410_012750/`
- The runner completed successfully and made the intended screen decision.
- This batch tested parameterization on the then-current `linear + cyclic` branch:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `ITERATIONS=1500`

### P3 Results

- `x0`
  - best periodic subset checkpoint: `val_bpb=2.9188`
  - final subset eval: `val_bpb=2.9320`
- `xtminus1`
  - best periodic subset checkpoint: `val_bpb=3.2008`
  - final subset eval: `val_bpb=3.2119`
- The long `xtminus1` follow-up was correctly skipped.
  - `xtminus1` was not within the configured `0.02 val_bpb` margin of `x0`

### What We Learned

- `xtminus1` is clearly worse than `x0` on the `linear + cyclic` branch.
- This is a trustworthy rejection for that branch.
  - The gap is much larger than normal screening noise
  - The runner's best-checkpoint selection and skip logic worked as intended
- This batch does not fully answer parameterization on the new promoted P2 recipe.
  - The promoted best recipe changed overnight to `cosine + cyclic` at `3000` steps
  - P3 still provides a useful negative result, but only for the `linear + cyclic` branch

### Current Interpretation

- `x0` remains the default parameterization for all current week-3 work.
- If we want the P3 question answered strictly on the latest promoted recipe, the clean follow-up is:
  - rerun the parameterization screen on `cosine + cyclic`
  - only launch a longer `xtminus1` follow-up there if it is competitive

## 2026-04-10 - Clean Linear-Cyclic P2 Rerun Completed And Reclaimed Best Overall

- Completed the intended long-run comparison batch in:
  - `logs/week3_stage_c_length_20260410_074124/`
- This batch cleanly executed the intended recipe:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `ITERATIONS=3000`
- The runner log and train log agree on the requested recipe.
- The run artifacts are:
  - train log: `logs/week3_stage_c_length_20260410_074124/week3_p2_long_linear_cyclic_paramx0_sc0_lwnone_20260410_074124_diffusion.txt`
  - manifest: `logs/week3_stage_c_length_20260410_074124/week3_p2_long_linear_cyclic_paramx0_sc0_lwnone_20260410_074124_diffusion_manifest.json`
  - full eval: `logs/week3_stage_c_length_20260410_074124/week3_p2_long_linear_cyclic_paramx0_sc0_lwnone_20260410_074124_diffusion_best_mlx_full_eval.txt`

### Clean Linear-Cyclic P2 Results

- Best periodic subset checkpoint:
  - step `3000`
  - `val_bpb=2.5648`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:4.8271 val_elbo_nats:4.3572 val_bits_per_token:6.2861 val_bpb:2.5802`
- Full validation on the best checkpoint:
  - `final_diffusion_eval proxy_loss:4.8507 val_elbo_nats:4.3656 val_bits_per_token:6.2983 val_bpb:2.5856`

### What We Learned

- The intended `linear + cyclic` 3000-step rerun is now the strongest confirmed week-3 result.
- It beats the previously promoted `cosine + cyclic` 3000-step run on both subset and full validation.
  - Previous promoted full val (`cosine + cyclic`, 3000 steps): `2.6430`
  - Clean linear-cyclic full val (`linear + cyclic`, 3000 steps): `2.5856`
  - Improvement: about `-0.0574 val_bpb`
- This resolves the temporary ambiguity from the earlier cosine-cyclic promotion.
  - The cosine-cyclic result was real and strong
  - But once we ran the intended apples-to-apples long comparison, `linear + cyclic` came out ahead
- The Stage-A direction was therefore stable across training length.
  - `linear + cyclic` was best at `1500` steps
  - `linear + cyclic` remains best among the tested schedule/timestep recipes at `3000` steps
- Sample quality still improved only gradually.
  - The metric win looks credible
  - But generation quality is still rough enough that week-3 selection should remain metric-led

### Current Interpretation

- The promoted week-3 recipe is now back to:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `ITERATIONS=3000` for the stronger local recipe
- This also makes the completed P3 result more directly useful again.
  - P3 already screened `x0` versus `xtminus1` on the same `linear + cyclic` branch
  - `xtminus1` was clearly worse there

## 2026-04-11 - P4 Process Sweep Ran Partially, Revealed One Invalid Candidate, And Did Not Produce A Clear New Winner

- Completed a partial P4 process batch in:
  - `logs/week3_stage_e_process_20260410_142713/`
- Five screen runs completed successfully:
  - diffusion steps `{16, 32, 64}` at default mask bounds
  - mask-bound variants `(0.0, 1.0)` and `(0.05, 1.0)` at the winning step count
- One planned mask-bound run failed immediately:
  - `MIN_MASK_RATE=0.0`, `MAX_MASK_RATE=0.95`
  - failure reason: ELBO validation requires the final absorbing state to be fully masked, but this recipe produced `m_T=0.95`

### P4 Screen Results

- `NUM_DIFFUSION_STEPS=16`, bounds `(0.0, 1.0)`
  - best subset checkpoint `val_bpb=2.9856`
- `NUM_DIFFUSION_STEPS=32`, bounds `(0.0, 1.0)`
  - best subset checkpoint `val_bpb=2.9118`
- `NUM_DIFFUSION_STEPS=64`, bounds `(0.0, 1.0)`
  - best subset checkpoint `val_bpb=2.9296`
- On the winning `32`-step branch:
  - bounds `(0.0, 1.0)` best subset checkpoint `val_bpb=2.9117`
  - bounds `(0.05, 1.0)` best subset checkpoint `val_bpb=2.9108`

### What We Learned

- `32` diffusion steps remain best on this local `1500`-step screen.
  - `16` was clearly worse
  - `64` was also worse
- Raising the minimum mask rate to `0.05` may help slightly on the `32`-step branch.
  - best valid P4 screen winner: `NUM_DIFFUSION_STEPS=32`, `MIN_MASK_RATE=0.05`, `MAX_MASK_RATE=1.0`
  - improvement versus previous `1500`-step Stage-A SOTA (`2.910961...`): about `-0.00011 val_bpb`
- That improvement is extremely small.
  - it technically beats the previous `1500`-step best checkpoint under a strict less-than rule
  - but it is not a convincing margin by itself
- `MAX_MASK_RATE<1.0` is not currently compatible with the ELBO validation path.
  - this is not just a bad result; it is an invalid evaluation recipe for the current setup

### Runner Fix

- `scripts/experiments/run_week3_p4.sh` has now been patched to:
  - validate ELBO compatibility before launching each mask-bound candidate
  - mark invalid recipes such as `MAX_MASK_RATE=0.95` as skipped/invalid instead of aborting the whole batch
  - preserve the intended screen -> promote -> long-run -> full-eval workflow for valid candidates

### Current Interpretation

- P4 did not produce a clear new recipe winner yet.
- The most interesting valid follow-up remains:
  - rerun P4 with the patched runner
  - let the valid best screen winner promote to `3000` steps
  - then decide based on that long-run result rather than the tiny `1500`-step edge alone

## 2026-04-12 - P4 Long Rerun Finished, Improved Subset Metrics, But Did Not Beat The Current Full-Val Champion

- Completed the patched P4 continuation in:
  - `logs/week3_stage_e_process_20260411_225310/`
- The runner reused all already-completed `1500`-step P4 screen runs from the earlier batch.
- It correctly skipped the ELBO-invalid candidate:
  - `MIN_MASK_RATE=0.0`, `MAX_MASK_RATE=0.95`
- It promoted the best valid screen winner to a `3000`-step rerun:
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.05`
  - `MAX_MASK_RATE=1.0`
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`

### Long-Run Result

- Promoted P4 `3000`-step manifest best:
  - subset best checkpoint `val_bpb=2.5675`
  - best step: `3000`
- Final subset eval at step `3000`:
  - `val_bpb=2.5782`

### Full-Val Comparison

- Full validation on the promoted P4 best checkpoint:
  - `val_bpb=2.5868`
  - `val_elbo_nats=4.3676`
- Current promoted P2 full validation:
  - `val_bpb=2.5856`
  - `val_elbo_nats=4.3656`
- Difference:
  - P4 is better on subset best-checkpoint screening than the old P2 subset result
  - but P4 is slightly worse on full validation by about `+0.0012 val_bpb`

### What We Learned

- Raising `MIN_MASK_RATE` from `0.0` to `0.05` does help the local subset metric at `3000` steps.
- That gain does not survive as a full-val promotion.
- The result is close enough to count as a real candidate, but not strong enough to replace the current base recipe.

### Current Interpretation

- Keep the promoted week-3 base recipe unchanged:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
- P4 is now resolved enough to move on.
- The next high-value batch should be `P5`, using the unchanged base recipe.

## 2026-04-12 - P5 Optimizer Sweep Produced A New Best Week-3 Recipe

- Completed the optimizer sweep batch in:
  - `logs/week3_stage_f_optim_20260412_003530/`
- This batch ran the full staged search:
  - learning-rate sweep
  - weight-decay sweep
  - grad-clip sweep
  - warmup sweep
  - beta2 sweep
  - promoted `3000`-step rerun
  - full validation on the promoted best checkpoint

### P5 Stage Winners

- Learning rate:
  - winner `LEARNING_RATE=0.0004`
  - best screen checkpoint `val_bpb=2.9070`
- Weight decay:
  - winner `WEIGHT_DECAY=0.0`
  - `0.01` and `0.03` were dramatically worse, both around `3.57 val_bpb`
- Gradient clip:
  - winner `GRAD_CLIP_NORM=0.3`
  - best screen checkpoint `val_bpb=2.8383`
  - previous `1.0` was clearly worse at `2.9055`
- Warmup:
  - winner `WARMUP_STEPS=20`
  - best screen checkpoint `val_bpb=2.8234`
  - `0` and `5` were both slightly worse
- Beta2:
  - winner remained `BETA2=0.95`
  - `0.98` and `0.99` both regressed

### Promoted Long-Run Result

- Promoted optimizer recipe:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0004`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMUP_STEPS=20`
- Promoted `3000`-step manifest best:
  - subset best checkpoint `val_bpb=2.4786`
  - best step: `3000`
- Local final subset eval at step `3000`:
  - `val_bpb=2.4952`

### Full-Val Comparison

- Full validation on the promoted P5 best checkpoint:
  - `val_bpb=2.5005`
  - `val_elbo_nats=4.2219`
- Previous promoted P2 full validation:
  - `val_bpb=2.5856`
  - `val_elbo_nats=4.3656`
- Improvement:
  - about `-0.0851 val_bpb`

### What We Learned

- The current week-3 recipe was significantly under-optimized.
- The highest-value optimizer changes were:
  - a higher learning rate (`4e-4`)
  - stronger clipping (`0.3`)
  - longer warmup (`20`)
- Weight decay is harmful in the current local regime.
- Increasing `beta2` above `0.95` did not help.

### Current Interpretation

- P5 is a real promotion, not a noisy local-only win.
- The week-3 default recipe should now include the optimizer changes above.
- This is strong enough to unblock `P6` fixed-model-size scale work on the new recipe.

## 2026-04-12 - Dynamic Optimizer Boundary Search Produced A Second P5 Promotion

- Completed the dynamic optimizer follow-up batch in:
  - `logs/week3_stage_f_optim_20260412_103410/`
- This batch started from the then-current promoted P5 recipe:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0004`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.3`
  - `WARMUP_STEPS=20`
- The dynamic search logic ran a fresh `1500`-step control, then searched in incumbent-updating passes over:
  - `grad_clip_norm`
  - `warmup_steps`
  - `beta2`
  - `learning_rate`

### Dynamic Search Results

- Fresh control at `1500` steps:
  - best subset checkpoint `val_bpb=2.8281`
- Accepted changes:
  - `grad_clip_norm 0.3 -> 0.2`
    - best subset checkpoint `val_bpb=2.8239`
  - `learning_rate 0.0004 -> 0.0005`
    - best subset checkpoint `val_bpb=2.7847`
  - `learning_rate 0.0005 -> 0.0007`
    - best subset checkpoint `val_bpb=2.6866`
  - `learning_rate 0.0007 -> 0.0011`
    - best subset checkpoint `val_bpb=2.6308`
  - `learning_rate 0.0011 -> 0.0012`
    - best subset checkpoint `val_bpb=2.6205`
- Rejected changes:
  - `grad_clip_norm 0.2 -> 0.05`
  - `warmup_steps 20 -> 40`
  - `beta2 0.95 -> 0.92`
  - second-pass probes on `grad_clip_norm`, `warmup_steps`, and `beta2` after the stronger learning-rate recipe had already been promoted
- The converged screen recipe was:
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`

### Promoted Long-Run Result

- Promoted `3000`-step best subset checkpoint:
  - `val_bpb=2.3636`
- Local final subset eval at step `3000`:
  - `val_bpb=2.3823`
- Full validation on the promoted best checkpoint:
  - `val_bpb=2.3900`
  - `val_elbo_nats=4.0678`

### What We Learned

- The previous optimizer recipe still had substantial headroom in learning rate.
- The dynamic search found a much stronger regime than the fixed P5 sweep.
- The gain is not a local-only artifact.
  - prior promoted full val: `2.5005`
  - dynamic-search promoted full val: `2.3900`
  - improvement: about `-0.1105 val_bpb`
- Only two optimizer changes survived promotion in the final recipe:
  - `grad_clip_norm=0.2`
  - `learning_rate=0.0012`
- `warmup_steps=20` and `beta2=0.95` remained the right choices once the stronger learning-rate branch was active.

### Current Interpretation

- The promoted week-3 local recipe is now:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`
- The next highest-value experiment remains `P6`:
  - run the promoted recipe on the scale config
  - keep model size fixed so the result stays aligned with the final artifact constraint while still allowing seq/batch scaling

## 2026-04-12 - Fixed-Size P6 Scale Run Improved Again

- Completed the fixed-size scale batch in:
  - `logs/week3_stage_g_scale_20260412_154123/`
- This run kept model size fixed relative to the promoted local recipe:
  - `NUM_LAYERS=6`
  - `MODEL_DIM=256`
- It used the promoted dynamic optimizer recipe:
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`
- The scale-context training changes were:
  - `TRAIN_SHARDS=2`
  - `TRAIN_SEQ_LEN=512`
  - `TRAIN_BATCH_TOKENS=32768`
  - `GRAD_ACCUM_STEPS=4`
  - `ITERATIONS=3000`

### P6 Results

- Best periodic subset checkpoint:
  - step `3000`
  - `val_bpb=2.3102`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:4.5369 val_elbo_nats:4.0564 val_bits_per_token:5.8547 val_bpb:2.3183`
- Full validation on the best checkpoint:
  - `final_diffusion_eval proxy_loss:4.5644 val_elbo_nats:3.9255 val_bits_per_token:5.6633 val_bpb:2.3249`

### What We Learned

- The fixed-size scale branch is a real promotion over the best local branch.
  - prior promoted full val (`F1`): `2.3900`
  - fixed-size scale full val (`G`): `2.3249`
  - improvement: about `-0.0651 val_bpb`
- The scale run did not look saturated at `3000` steps.
  - periodic subset values kept improving through `2200`, `2400`, `2600`, `2800`, and `3000`
  - the manifest best checkpoint was the final step
- The gain appears to come from the scale-context training regime, not from a larger model.
  - parameter count stayed fixed at about `3.42M`

### Current Interpretation

- The best confirmed week-3 recipe is now the fixed-size scale branch:
  - same `6L x 256d` model size
  - same promoted dynamic optimizer recipe
  - scale-context training settings (`TRAIN_SHARDS=2`, `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=32768`, `GRAD_ACCUM_STEPS=4`)
- The next highest-value experiment is to stay on this branch and increase data exposure and/or training length.
- Reopening local optimizer search is lower value than extending this still-improving fixed-size scale run.

## 2026-04-12 - P7 Warm-Start Continuation Reached A Much Stronger Local-Device Result

- Completed the weights-only continuation batch in:
  - `logs/week3_stage_h_continue_20260412_200615/`
- This run warm-started from the promoted `P6` best checkpoint:
  - `logs/week3_stage_g_scale_20260412_154123/diffusion_week3_scale_diffusion_best_mlx.npz`
- It kept the same fixed-size scale recipe:
  - `NUM_LAYERS=6`
  - `MODEL_DIM=256`
  - `TRAIN_SHARDS=2`
  - `TRAIN_SEQ_LEN=512`
  - `TRAIN_BATCH_TOKENS=32768`
  - `GRAD_ACCUM_STEPS=4`
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`
- The continuation run actually used:
  - `ITERATIONS=7000`
  - `EARLY_STOP_PATIENCE=10`
  - `EARLY_STOP_METRIC=val_bpb`
- Early stopping never fired because the run kept finding new improvements before patience was exhausted.

### P7 Results

- Starting point from the loaded checkpoint at continuation step `0`:
  - `val_bpb=2.3102`
- Best periodic subset checkpoint:
  - step `7000`
  - `val_bpb=2.1093`
- Final logged eval from the continuation run:
  - `final_diffusion_eval proxy_loss:4.3237 val_elbo_nats:3.5729 val_bits_per_token:5.1546 val_bpb:2.1158`
- Manifest result:
  - best checkpoint remained the final-step checkpoint
  - manifest best metric value: `2.109334907496521`

### What We Learned

- The fixed-size scale branch still had substantial headroom on local hardware.
  - prior promoted scale full val (`G`): `2.3249`
  - continuation subset best (`H`): `2.1093`
  - gap: about `-0.2156 val_bpb`
- Improvement was not perfectly monotonic, but the run kept making fresh lows late into training.
  - for example, `5200 -> 5400` regressed slightly (`2.1287 -> 2.1302`)
  - then the run improved again to `2.1193`, `2.1106`, and finally `2.1093`
- Because this was a weights-only continuation, it is not a clean apples-to-apples replacement for the fresh `P6` baseline.
  - optimizer state, schedule state, and data order were reset
  - this makes `H` a strong achieved-quality reference, not a strict fresh-run control

### Current Interpretation

- The best confirmed full-val baseline still remains `G` until `H` gets its own standalone full eval.
- The strongest achieved local-device quality result is now `H`.
- The right next baseline-setting run is a fresh `10000`-step training run on the same fixed-size scale recipe, without warm-starting from `H`.

## 2026-04-13 - P8 Fresh 10000-Step Scale Run Confirmed The New Clean Champion

- Completed the fresh long fixed-size scale batch in:
  - `logs/week3_stage_i_scale_long_20260413_134901/`
- This run kept the same promoted fixed-size scale recipe as `P6/H`:
  - `NUM_LAYERS=6`
  - `MODEL_DIM=256`
  - `TRAIN_SHARDS=2`
  - `TRAIN_SEQ_LEN=512`
  - `TRAIN_BATCH_TOKENS=32768`
  - `GRAD_ACCUM_STEPS=4`
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`
  - `ITERATIONS=10000`
- The batch runner originally marked the run failed because the config re-sourced `RUN_ID=diffusion_scale_long` and the verification step looked for the synthetic `week3_p8_...` train-log path.
- Training itself completed cleanly, and the standalone full eval later completed on:
  - checkpoint: `logs/week3_stage_i_scale_long_20260413_134901/diffusion_scale_long_diffusion_best_mlx.npz`
  - full eval: `logs/week3_stage_i_scale_long_20260413_134901/diffusion_scale_long_diffusion_best_mlx_full_eval.txt`

### P8 Results

- Best periodic subset checkpoint:
  - step `10000`
  - `val_bpb=2.1097`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:4.3279 val_elbo_nats:3.5761 val_bits_per_token:5.1593 val_bpb:2.1177`
- Full validation on the best checkpoint:
  - `final_diffusion_eval proxy_loss:4.3322 val_elbo_nats:3.5923 val_bits_per_token:5.1826 val_bpb:2.1276`

### What We Learned

- The strong `H` continuation result was not a continuation-only artifact.
  - `P8` reproduced essentially the same quality in a fresh run
  - `H` subset best: `2.1093`
  - `P8` subset best: `2.1097`
- The long fixed-size scale branch is a clear promotion over the previous clean full-val champion.
  - previous promoted full val (`G`): `2.3249`
  - fresh `10000`-step full val (`P8`): `2.1276`
  - improvement: about `-0.1973 val_bpb`
- The best checkpoint again arrived at the very end of training.
  - this branch still looked productive through `10000` steps
- The original `P8` failure was orchestration-only.
  - it does not invalidate the checkpoint or the full-val result

### Current Interpretation

- The promoted clean week-3 baseline is now the fresh `P8` scale-long run.
- This becomes the new control for follow-up scale-context experiments.
- The next highest-value probe is data exposure on the same branch:
  - first try `TRAIN_SHARDS=4`
  - keep the rest of the recipe unchanged so the comparison stays clean

## 2026-04-14 - P9 4-Shard Data-Exposure Probe Completed, But Did Not Beat P8

- Completed the first 4-shard follow-up batch in:
  - `logs/week3_stage_j_scale_long_4shards_20260414_013602/`
- This run kept the promoted `P8` fixed-size scale recipe and changed:
  - `TRAIN_SHARDS=2 -> 4`
- Everything else stayed fixed:
  - `NUM_LAYERS=6`
  - `MODEL_DIM=256`
  - `TRAIN_SEQ_LEN=512`
  - `TRAIN_BATCH_TOKENS=32768`
  - `GRAD_ACCUM_STEPS=4`
  - `MASK_SCHEDULE=linear`
  - `TRAIN_TIMESTEP_SAMPLING=cyclic`
  - `PARAMETERIZATION=x0`
  - `SELF_CONDITIONING=0`
  - `LOSS_REWEIGHTING=none`
  - `NUM_DIFFUSION_STEPS=32`
  - `MIN_MASK_RATE=0.0`
  - `MAX_MASK_RATE=1.0`
  - `LEARNING_RATE=0.0012`
  - `WEIGHT_DECAY=0.0`
  - `BETA2=0.95`
  - `GRAD_CLIP_NORM=0.2`
  - `WARMUP_STEPS=20`
  - `ITERATIONS=10000`

### P9 Results

- Best periodic subset checkpoint:
  - step `9800`
  - `val_bpb=2.1112`
- Final subset eval:
  - `final_diffusion_eval proxy_loss:4.3274 val_elbo_nats:3.5761 val_bits_per_token:5.1592 val_bpb:2.1177`
- Full validation on the best checkpoint:
  - `final_diffusion_eval proxy_loss:4.3328 val_elbo_nats:3.5925 val_bits_per_token:5.1829 val_bpb:2.1277`

### What We Learned

- Increasing `TRAIN_SHARDS` from `2` to `4` did not produce a meaningful gain at fixed `10000` steps.
  - `P8` full val: `2.1276`
  - `P9` full val: `2.1277`
  - difference: about `+0.0001 val_bpb`
- The two runs are effectively tied on the local subset too.
  - `P8` subset best: `2.1097`
  - `P9` subset best: `2.1112`
- Extra data exposure by itself is therefore not enough to promote at the current training budget.

### Current Interpretation

- Keep `P8` as the promoted clean full-val champion.
- Do not promote the 4-shard variant.
- If we revisit higher data exposure, we should change another budget dimension too:
  - longer training
  - or a larger token budget
  - rather than only increasing shard count
