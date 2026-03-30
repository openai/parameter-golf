# TR-02 Delta Sweep

Date: 2026-03-29

## Goal

Screen simple numerics-compensation deltas on top of:
- `JR-01` loader winner
- `MLP_KERNEL_MODE=triton_act`

The original plan was to use short `170s` pop tests.

What actually happened:
- the runner on the box still had the old hardcoded `600s` training cap
- each delta executed as a full capped run
- this was a runner bug, but the resulting data is still useful and stronger than a short screen

## Fixed Context

- seed: `1337`
- loader: `coprime`
- kernel: `triton_act`
- compile mode: `default`
- fullgraph: `1`
- train cap actually observed: `600s`
- final eval: skipped for this sweep
- comparison metric: cap-time validation BPB

## Results

| Rank | Variant | Delta | Step avg | Cap-time val BPB |
|---|---|---|---:|---:|
| 1 | `delta04_attn_scale_102` | `ATTN_SCALE_INIT=1.02` | `91.16ms` | `1.1347` |
| 2 | `delta00_base` | baseline `triton_act` | `91.09ms` | `1.1349` |
| 3 | `delta01_mlp_scale_098` | `MLP_SCALE_INIT=0.98` | `91.12ms` | `1.1351` |
| 4 | `delta02_mlp_scale_102` | `MLP_SCALE_INIT=1.02` | `91.10ms` | `1.1352` |
| 5 | `delta03_attn_scale_098` | `ATTN_SCALE_INIT=0.98` | `91.13ms` | `1.1354` |
| 6 | `delta05_residmix_098_002` | `RESID_MIX_X_INIT=0.98`, `RESID_MIX_X0_INIT=0.02` | `91.10ms` | `1.1356` |

## Readout

- `ATTN_SCALE_INIT=1.02` is the only tested delta that improved the Triton path.
- The gain is small but real enough to justify one full confirmation run with final eval.
- The two tested `mlp_scale` nudges both lost.
- The tested `resid_mix` nudge lost worst, so that exact direction should be treated as a loser.
- Throughput stayed effectively flat across the sweep, so this was a quality ranking, not a speed ranking.

## Comparison To Existing Triton Baseline

Earlier full-run `TR-01` result with final eval:
- step avg: `91.11ms`
- post-EMA BPB: `1.1345`
- sliding BPB: `1.11099954`

This sweep suggests the best next confirmation candidate is:

```bash
ATTN_SCALE_INIT=1.02 bash experiments/Junkyard_Rat/triton/run_jr02_triton_act.sh
```

That is the only remaining question for this branch:
- does `triton_act + attn_scale=1.02` beat `JR-01`'s `1.11056240`
- or does the Triton branch stay archived as an interesting but losing surface

## Runner Note

The short-screen runner bug has already been fixed on `test`, but that fix was not present on the box that produced this sweep.

Expected corrected short-screen signature for future use:
- `warmup_steps:0`
- `max_wallclock_seconds:170.000`
