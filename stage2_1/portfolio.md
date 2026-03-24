# Stage 2_1 Portfolio

This is the revised `stage2_1` portfolio after the 2026-03-23 `stage3` attack-surface update.

The earlier `stage2_1` slate was useful as a first pass, but it is no longer strong enough. The frontier has moved. The strongest no-TTT submissions now cluster around:

- 11L frontier-aligned env settings
- GPTQ-class deployment
- LeakyReLU(0.5)^2
- EMA
- XSA4
- MuonWD=0.04

So the portfolio shifts from an older "missing community tricks" screen to a frontier-template screen.

## Base Control Shift

The training control is no longer the old root-like recipe. It is the stronger frontier-aligned base:

- `NUM_LAYERS=11`
- `MLP_MULT=3`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `MATRIX_LR=0.025`
- `SCALAR_LR=0.025`
- `TIED_EMBED_LR=0.035`
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_START=0.92`
- `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `GRAD_CLIP_NORM=0.3`
- `WARMDOWN_ITERS=3500`
- `VAL_LOSS_EVERY=4000`
- `EVAL_STRIDE=64`

If this base is not the control, then `stage2_1` is spending most of its budget relearning old baseline fixes rather than attacking the current frontier gap.

## Active Top 8

| Slot | Role | Family | Why it is in the top set now | Main lane |
| --- | --- | --- | --- | --- |
| `P1` | exploit | Full GPTQ export | biggest known no-TTT unlock; must be tested before almost anything else | deployment |
| `P2` | exploit | LeakyReLU(0.5)^2 | one-line, repeated frontier win | training |
| `P3` | exploit | EMA(0.997) | now more frontier-aligned than SWA-first | training to deployment |
| `P4` | exploit | XSA4 | common across nearly every very strong no-TTT stack | architecture/context |
| `P5` | exploit | MuonWD=0.04 | still a real helper and easy to compose | training to deployment |
| `P6` | scout | VRL | new medium-high structural lever with real evidence | architecture |
| `P7` | scout | Partial RoPE 16/64 + LN Scale | low-cost refinement bundle from merged SOTA | architecture refinement |
| `P8` | exploit | Sliding eval plus doc-aware child | still a real late-stage score lever, but no longer the main story | evaluation |

## Demoted From The Old Pack

These are still live ideas, but they are no longer primary `stage2_1` slots:

- NorMuon
- OrthoInit + muP
- solo SmearGate
- solo BigramHash
- FA3-first throughput reclaim
- label smoothing
- MTP

Reason:

- the new frontier evidence does not support them as the best next explanation of the remaining gap
- some are helper ideas, but not the current bottleneck
- some have been overtaken by stronger substitutes such as GPTQ and EMA

## Pack A: Shared-Checkpoint Deployment Lane

Use one strong trained checkpoint and only change export/eval behavior.

| Slot | Type | Control | Purpose |
| --- | --- | --- | --- |
| `A0` | control | none | frontier-aligned shared-checkpoint deployment control |
| `A1` | control | `A0` | repeat for readout noise |
| `A2` | candidate | `A0` | sliding eval `stride=64` |
| `A3` | candidate | `A2` | doc-isolated sliding child |
| `A4` | candidate | `A0` | GPTQ-lite clip-search export |
| `A5` | candidate | `A0` | full GPTQ int6 export |
| `A6` | candidate | `A5` | full GPTQ + zstd-22 |
| `A7` | candidate | `A5` | full GPTQ + fp16 embedding passthrough |

Ranking for Pack A:

1. deployed `score_bpb`
2. deployment gain vs control on the same checkpoint
3. artifact legality margin

## Pack B: Frontier Training Screen

Use short wallclock-aligned training screens, but judge against the stronger frontier-aligned base.

| Slot | Type | Control | Purpose |
| --- | --- | --- | --- |
| `B0` | control | none | frontier-aligned training base |
| `B1` | control | `B0` | repeat control |
| `B2` | candidate | `B0` | MuonWD=0.04 |
| `B3` | candidate | `B0` | LeakyReLU(0.5)^2 |
| `B4` | candidate | `B0` | EMA(0.997) |
| `B5` | candidate | `B0` | XSA4 |
| `B6` | candidate | `B0` | VRL |
| `B7` | candidate | `B0` | Partial RoPE 16/64 + LN Scale |

Ranking for Pack B:

1. `delta_pre_quant_bpb`
2. `delta_post_quant_bpb`
3. `delta_step_avg_ms`
4. whether the gain plausibly survives deployment

## Pack C: Frontier Child Stacks

Only run this after one Pack A winner and at least one Pack B winner are real.

| Slot | Type | Control | Purpose |
| --- | --- | --- | --- |
| `C0` | control | none | promoted frontier base |
| `C1` | control | `C0` | repeat control |
| `C2` | candidate | `C0` | MuonWD + LeakyReLU^2 |
| `C3` | candidate | `C0` | MuonWD + EMA |
| `C4` | candidate | `C0` | MuonWD + XSA4 |
| `C5` | candidate | `C0` | MuonWD + LeakyReLU^2 + EMA + XSA4 |
| `C6` | candidate | `C5` | best core stack + best deployment lane |
| `C7` | candidate | `C5` | best core stack + best architecture helper |

Architecture helpers for `C7`:

- VRL if it wins solo
- VE128 if GPTQ headroom exists
- BigramHash only if budget remains and the gain is still competitive

## Promotion Rules

- Promote deployment moves only if they win on the same checkpoint.
- Promote training moves only if they beat the stronger frontier control, not just the old baseline.
- Promote child stacks only if the parent mechanisms are distinct and their single-effect wins are already real.

## What This Portfolio Is Trying To Do

It is no longer trying to explain rank-19 with older public tricks.

It is trying to build a credible no-TTT route from `1.1631` toward the `1.12` band by centering the search on the mechanisms that now dominate the leaderboard.
