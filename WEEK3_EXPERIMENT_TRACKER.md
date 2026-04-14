# Week 3 Diffusion Experiment Tracker

Operator index for week-3 status. Detailed milestone narrative and postmortems live in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Status legend: `planned`, `running`, `completed`, `analyzed`, `promoted`, `rejected`, `blocked`

## Current State

### Aliases

| Alias | Meaning |
|---|---|
| `R0` | `cosine + random + x0 + sc0 + lw:none` |
| `R1` | `linear + cyclic + x0 + sc0 + lw:none` |
| `R2` | `cosine + cyclic + x0 + sc0 + lw:none` |

### Batch Handles

- `A` = [logs/week3_stage_a_20260408_143331](logs/week3_stage_a_20260408_143331)
- `B` = [logs/week3_stage_b_20260409_004333](logs/week3_stage_b_20260409_004333)
- `C0` = [logs/week3_stage_c_length_20260409_142535](logs/week3_stage_c_length_20260409_142535)
- `C1` = [logs/week3_stage_c_length_20260410_001843](logs/week3_stage_c_length_20260410_001843)
- `C2` = [logs/week3_stage_c_length_20260410_074124](logs/week3_stage_c_length_20260410_074124)
- `D` = [logs/week3_stage_d_param_20260410_012750](logs/week3_stage_d_param_20260410_012750)
- `E0` = [logs/week3_stage_e_process_20260410_142713](logs/week3_stage_e_process_20260410_142713)
- `E1` = [logs/week3_stage_e_process_20260411_225310](logs/week3_stage_e_process_20260411_225310)
- `F` = [logs/week3_stage_f_optim_20260412_003530](logs/week3_stage_f_optim_20260412_003530)
- `F1` = [logs/week3_stage_f_optim_20260412_103410](logs/week3_stage_f_optim_20260412_103410)
- `G` = [logs/week3_stage_g_scale_20260412_154123](logs/week3_stage_g_scale_20260412_154123)
- `H` = [logs/week3_stage_h_continue_20260412_200615](logs/week3_stage_h_continue_20260412_200615)
- `I` = [logs/week3_stage_i_scale_long_20260413_134901](logs/week3_stage_i_scale_long_20260413_134901)
- `J` = [logs/week3_stage_j_scale_long_4shards_20260414_013602](logs/week3_stage_j_scale_long_4shards_20260414_013602)

### Live Summary

- Current promoted recipe: `R1` process with optimizer `lr=1.2e-3`, `wd=0.0`, `beta2=0.95`, `grad_clip=0.2`, `warmup=20`, plus scale-context training `TRAIN_SHARDS=2`, `TRAIN_SEQ_LEN=512`, `TRAIN_BATCH_TOKENS=32768`, `GRAD_ACCUM_STEPS=4`, `ITERATIONS=10000`
- Current best confirmed full-val run: `diffusion_scale_long`
- Best subset best-checkpoint result: `val_bpb=2.1093`
- Best clean fresh-run subset best-checkpoint result: `val_bpb=2.1097`
- Best full-val result: `val_bpb=2.1276`
- Best achieved local-device checkpoint: [H best ckpt](logs/week3_stage_h_continue_20260412_200615/diffusion_week3_scale_diffusion_best_mlx.npz)
- Promoted full-val checkpoint: [I best ckpt](logs/week3_stage_i_scale_long_20260413_134901/diffusion_scale_long_diffusion_best_mlx.npz)
- Promoted full eval: [I full eval](logs/week3_stage_i_scale_long_20260413_134901/diffusion_scale_long_diffusion_best_mlx_full_eval.txt)
- Best clean 1500-step control: `R1` in `A`, subset `2.9240`, full val `2.9325`
- Promotion threshold:
  - screen winner should beat the incumbent by about `0.02 val_bpb`, unless a smaller gain is validated by a clearly better full-val result
  - any promoted recipe must have a full-val result
- Next planned batch: none locked yet; `P9` did not beat `P8`, so keep `P8` as the clean control and only revisit higher data exposure if another budget dimension changes too

## Batch Board

| Batch | Question | Recipe Base | Status | Winner / Result | Next Action | Ref |
|---|---|---|---|---|---|---|
| `P0` | Does the Stage-A winner survive full validation? | `R1 @ 1500` | `completed` | full val `2.9325`; promote `R1` over `R0` | done | [B](logs/week3_stage_b_20260409_004333) |
| `P1` | Do self-conditioning or inverse-mask-rate help on `R1`? | `R1 @ 1500` | `analyzed` | best was control at subset `2.9314`; no gain | no promotion | [B](logs/week3_stage_b_20260409_004333) |
| `P2` | Does the best local recipe improve at `3000` steps? | `R1 @ 3000` | `promoted` | subset `2.5802`, full val `2.5856` | current default | [C2](logs/week3_stage_c_length_20260410_074124) |
| `P3` | Can `xtminus1` compete once schedule/timestep are fixed? | `R1 @ 1500` | `analyzed` | `x0` `2.9320`; `xtminus1` rejected | keep `x0` | [D](logs/week3_stage_d_param_20260410_012750) |
| `P4` | Is the process bottlenecked by diffusion steps or mask bounds? | `R1 @ 1500` screen | `analyzed` | promoted `min_mask=0.05` improved subset to `2.5675` at `3000` steps but full val was `2.5868`, slightly worse than `P2` | keep the base recipe unchanged and move to `P5` | [E1](logs/week3_stage_e_process_20260411_225310) |
| `P5` | Can optimizer tuning unlock another gain after process stabilizes? | `R1 @ 1500` screen | `promoted` | dynamic search winner: `lr=1.2e-3`, `wd=0`, `beta2=0.95`, `clip=0.2`, `warmup=20`; long-run subset `2.3636`, full val `2.3900` | current default optimizer recipe | [F1](logs/week3_stage_f_optim_20260412_103410) |
| `P6` | Does the best local recipe hold up when we keep model size fixed on the scale config? | `R1` fixed-size scale config | `promoted` | subset `2.3183`, full val `2.3249` on `TRAIN_SHARDS=2`, `seq_len=512`, `batch_tokens=32768` | promote this fixed-size scale branch and extend training/data exposure next | [G](logs/week3_stage_g_scale_20260412_154123) |
| `P7` | Does a weights-only continuation show more local-device headroom on the promoted fixed-size scale branch? | `P6` best checkpoint warm start | `completed` | best subset `2.1093`, final subset-style eval `2.1158`; no new full val yet | use as achievable local-device quality reference, but keep `P6` as the clean fresh-run control until full eval lands | [H](logs/week3_stage_h_continue_20260412_200615) |
| `P8` | Does a fresh `10000`-step scale run confirm the warm-start gains as a clean baseline? | `P6/H` recipe at `10000` steps | `promoted` | subset best `2.1097`, final subset `2.1177`, full val `2.1276` on `TRAIN_SHARDS=2` | promote as the new clean full-val champion and use as the control for data-exposure probes | [I](logs/week3_stage_i_scale_long_20260413_134901) |
| `P9` | Does increasing training shards from `2` to `4` help on the promoted `P8` branch? | `P8` recipe + `TRAIN_SHARDS=4` | `analyzed` | subset best `2.1112`, final subset `2.1177`, full val `2.1277`; effectively tied but slightly worse than `P8` | no promotion; keep `P8` as the clean control | [J](logs/week3_stage_j_scale_long_4shards_20260414_013602) |

## Completed Snapshot

| Batch | Best Result | Decision | Ref |
|---|---|---|---|
| `A` | `R1` subset `2.9240`, full val `2.9325` | promote `R1` over `R0`; strongest clean 1500-step branch | [A](logs/week3_stage_a_20260408_143331) |
| `B` | Stage-B control on `R1` subset `2.9314` | no promotion; self-conditioning and inverse-mask-rate hurt | [B](logs/week3_stage_b_20260409_004333) |
| `C0` | wrong-recipe probe subset `2.6779` | informative only; not promotable | [C0](logs/week3_stage_c_length_20260409_142535) |
| `C1` | `R2 @ 3000` subset `2.6296`, full val `2.6430` | formerly promoted, now superseded by `C2` | [C1](logs/week3_stage_c_length_20260410_001843) |
| `C2` | `R1 @ 3000` subset `2.5802`, full val `2.5856` | current best confirmed recipe | [C2](logs/week3_stage_c_length_20260410_074124) |
| `D` | `x0` subset `2.9320`; `xtminus1` subset `3.2119` | reject `xtminus1` on the `R1` branch | [D](logs/week3_stage_d_param_20260410_012750) |
| `E0` | valid screens: `steps=32,min=0.05,max=1.0` best checkpoint `2.910849`; invalid candidate `(min=0.0,max=0.95)` | no promotion yet; patch runner to skip ELBO-invalid recipes and rerun `P4` | [E0](logs/week3_stage_e_process_20260410_142713) |
| `E1` | `P4` long rerun on `steps=32,min=0.05,max=1.0`: subset `2.5675`, full val `2.5868` | informative but not promotable; keep `R1` base unchanged because full val is slightly worse than `C2` | [E1](logs/week3_stage_e_process_20260411_225310) |
| `F` | promoted `P5` optimizer recipe: subset best `2.4786`, local final subset `2.4952`, full val `2.5005` | promote optimizer changes over `C2`; next step is scale-up | [F](logs/week3_stage_f_optim_20260412_003530) |
| `F1` | dynamic `P5` search winner: subset best `2.3636`, local final subset `2.3823`, full val `2.3900` | promote dynamic optimizer changes over `F`; next step is scale-up | [F1](logs/week3_stage_f_optim_20260412_103410) |
| `G` | fixed-size `P6` scale run: subset best `2.3102`, final subset `2.3183`, full val `2.3249` | promote scale-context recipe over `F1`; next step is more data and/or longer training on this branch | [G](logs/week3_stage_g_scale_20260412_154123) |
| `H` | warm-start `P7` continuation: subset best `2.1093`, final subset-style eval `2.1158`, best checkpoint at step `7000` | strong achievable-quality reference on local device; still awaiting full eval before replacing `G` as the formal confirmed champion | [H](logs/week3_stage_h_continue_20260412_200615) |
| `I` | fresh `P8` scale-long run: subset best `2.1097`, final subset `2.1177`, full val `2.1276` | promote over `G`; establishes the new clean fixed-size scale champion | [I](logs/week3_stage_i_scale_long_20260413_134901) |
| `J` | `P9` 4-shard rerun: subset best `2.1112`, final subset `2.1177`, full val `2.1277` | no promotion; extra shard exposure at fixed `10000` steps did not beat `P8` | [J](logs/week3_stage_j_scale_long_4shards_20260414_013602) |

## Rules

- Use subset validation for screening and full validation for promotion.
- Compare saved best checkpoints for decisions; use final subset evals for short summaries.
- Do not start a new broad sweep until the previous batch has a written decision here.
- Keep process, optimization, and scale-up sweeps separate.
- If a result needs narrative context, failure analysis, or lessons learned, put that in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), not here.
