# Stage 2 Experiments

Stage 2 is no longer about broad family search.

It is about disciplined `8xH100` experiments built on the public record frontier:

- validate which pieces of the record SOTA are actually carrying the win,
- invalidate weak stories quickly,
- explicitly test whether lower `ms/step` buys enough extra steps to win,
- and test only a small number of orthogonal extensions on top of that baseline.

## Current Execution Model

The active execution model is now budget-aware, not symmetry-for-symmetry's sake.

Use the same `8` GPUs in three horizons:

- `sanity`: `8 x 1xH100` in parallel for `90s`
- `screen`: `8 x 1xH100` in parallel for `180s`
- `final_single`: top survivor on `1xH100` for `600s`

Only later, if the `1xH100` finalist is strong enough, promote it to optional `8xH100` confirmation.

The active matrix for this is:

- [portfolio.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/h100_matrix_r2/portfolio.md)
- [run_configs.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/h100_matrix_r2/run_configs.json)
- [orchestrate_stage2.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2/h100_matrix_r2/orchestrate_stage2.py)

## Two Baselines

Stage 2 needs two different baselines.

### 1. Public Record Target

Use the current best public record in the repo as the score target:

Authoritative local source:

- [submission.json](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/submission.json)
- [train.log](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train.log)
- [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_WarmdownQuantization/train_gpt.py)

Important repo-state caveat:

- the README in that folder is stale
- the actual record metadata is in `submission.json` and `train.log`

This defines the score to beat, but it is not the best controlled training trunk because it bundles:

- int6 export
- fp16 embedding passthrough
- late-K passthrough
- longer-context training
- and sliding-window eval

### 2. Training Trunk Baseline

Use the strongest clean training-side public record as the Stage 2 trunk:

- [README.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/README.md)
- [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/train_gpt.py)

Reason:

- it is the cleanest strong training recipe
- it is much easier to attribute gains on top of it
- it avoids baking eval-only tricks into the baseline itself

## Training Trunk Spec

Treat `TrainingOptSeq4096 v1` as the baseline to replay first on `8xH100`.

- score target: `1.20143417`
- train geometry:
  - `TRAIN_SEQ_LEN=4096`
  - `TRAIN_BATCH_TOKENS=393216`
- eval geometry:
  - fixed-chunk eval first
  - then always also record `EVAL_STRIDE=64` as a secondary readout for comparability
- model:
  - `9` layers
  - `512` width
  - `KV=4`
- optimizer:
  - `TIED_EMBED_LR=0.03`
  - `MATRIX_LR=0.02`
  - `SCALAR_LR=0.02`
  - `MUON_MOMENTUM=0.99`
  - `MUON_MOMENTUM_WARMUP_START=0.92`
  - `MUON_MOMENTUM_WARMUP_STEPS=1500`
  - `WARMDOWN_ITERS=3000`
- reported public result:
  - `1.20143417`

## Record Stack Reference

Treat the current best public record stack as the later composition target, not the attribution baseline.

- score target: `1.15744040`
- train geometry:
  - `TRAIN_SEQ_LEN=2048`
  - `TRAIN_BATCH_TOKENS=786432`
- eval geometry:
  - `EVAL_SEQ_LEN=2048`
  - `EVAL_STRIDE=256`
- model:
  - `9` layers
  - `512` width
  - `KV=4`
  - `MLP 3x` expansion per record blurb
  - `21,778,504` params from the run log
- optimizer:
  - `TIED_EMBED_LR=0.03`
  - `MATRIX_LR=0.02`
  - `SCALAR_LR=0.02`
  - `MUON_BACKEND_STEPS=5`
- export:
  - int6 quantization
  - fp16 tied embedding passthrough
  - late-K passthrough
- final logged metrics:
  - plain post-quant eval at `2048`: `1.17890201`
  - sliding-window eval at `2048, stride=256`: `1.15744040`

## Stage 2 Rules

- `S2-B0` trunk replay is mandatory. If it does not reproduce closely, stop and debug before any mutation runs.
- Each experiment must answer one main causal question.
- Prefer ablations and single extensions over broad composites.
- Use short `1xH100` runs for elimination.
- Use full `1xH100` reruns only for experiments that beat their matched control by enough margin to matter.
- Use `8xH100` only for the final confirmed survivor.
- For every training-side experiment, record both:
  - fixed-chunk final eval
  - sliding-window final eval at `EVAL_STRIDE=64`
- For every training-side experiment, record all of:
  - `step_avg ms`
  - `steps reached before wallclock cap`
  - pre-quant `val_bpb`
  - post-quant fixed-chunk `val_bpb`
  - post-quant sliding `val_bpb`
- Every experiment must name both:
  - what would validate the idea
  - what would falsify the idea

## Core Questions

1. Which training-side mutations move the strongest clean public trunk?
2. Which export submoves are actually carrying the current record SOTA?
3. Is eval policy near-optimal, or just the first strong public choice?
4. Which of our internal wins transfer from A100 proxy to real `8xH100`?
5. Can we reduce `ms/step` enough to buy more loss reduction within the same 10-minute budget?

## Stage 2 Portfolio

### Wave A: Trunk Replay And Clean Extensions

These runs establish the strongest controlled training trunk.

| ID | Role | Starting point | Mutation | Why this is strong | Validate if | Invalidate if | Stackability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2-B0 | gate | `TrainingOptSeq4096 v1` | exact replay | without this, every downstream conclusion is weak | score lands near `~1.2014`, step time is in-family, and the run is stable | miss by `>0.005` BPB or strong throughput drift | trunk only |
| S2-E1 | extension | `S2-B0` | add `EVAL_STRIDE=64` sliding eval | checks whether Stage 1 underpriced eval-only gains on a stronger checkpoint | sliding eval materially improves score | gain is small or eval cost becomes operationally bad | yes |
| S2-E2 | extension | `S2-B0` | add fp16 tied-embedding export, trim `MLP_HIDDEN=992` if needed | directly tests whether the cleanest export trick helps on the seq4096 trunk | better final score and/or smaller post-quant gap | negligible gain or bad size trade | yes |
| S2-E3 | extension | `S2-B0` | `INT8_CLIP_PERCENTILE=99.99995` and `INT8_PER_ROW_SCALE_DTYPE=float32` | directly tests whether our strongest A100 quant-fidelity win survives on real `8xH100` | post-quant gap shrinks and final score improves | gap barely moves | yes |

### Wave B: Public-Recipe Transfers

These runs test whether adjacent public recipe components transfer to the strong trunk.

| ID | Role | Starting point | Mutation | Why this is strong | Validate if | Invalidate if | Stackability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2-E4 | extension | best survivor from `S2-E1` to `S2-E3` | add `MUON_WEIGHT_DECAY=0.02` | public top-2 record suggests Muon WD may improve generalization and export robustness | score improves or quant gap shrinks | no real gain | yes |
| S2-E5 | extension | best survivor from `S2-E1` to `S2-E4` | add overtone / spectral embedding init and phase-transition `resid_mix` init | explicitly tests whether the public top-2 init story is portable | better early curve and better final score | flat curves and flat final score | yes |
| S2-E6 | extension | best survivor from `S2-E1` to `S2-E5` | add adaptive Muon schedule | directly resolves the Stage 1 R1/R2 disagreement in the exact target environment | clear gain on the stacked baseline | flat or negative again | yes, if positive |

### Wave B2: Throughput-Reclaim Branch

These runs explicitly combine the public record path with our own step-time-focused winners.

The logic is:

- the public record teaches what a strong stack looks like
- our internal winners suggest there may still be cheap ways to reduce `ms/step`
- if lower `ms/step` produces materially more updates before the wallclock cap, it can reduce loss even when per-step quality is unchanged

| ID | Role | Starting point | Mutation | Why this is strong | Validate if | Invalidate if | Stackability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2-E6b | throughput | best export/eval-aware stack so far, ideally record-like `2048` branch | adaptive Muon with the strongest safe low-cost schedule | this is the cleanest direct test of whether step-time reduction buys extra loss reduction | `step_avg` drops, steps reached rises, and final score improves or holds | `step_avg` barely moves or score regresses | yes |
| S2-E6c | throughput | record-like `2048` branch plus `S2-E3` if positive | record stack + our quant-quality win + adaptive Muon | this is the strongest “record plus our winners” derivative branch | hybrid beats the plain record-like branch on final score and/or post-quant gap | no gain over the simpler stack | conditional |

### Wave C: Architecture Challenge Run

Do this only after at least one Wave B survivor.

| ID | Role | Starting point | Mutation | Why this is strong | Validate if | Invalidate if | Stackability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S2-E7 | composite challenge | best stack from Waves A/B | increase to `NUM_LAYERS=10` at width `512` | this is the fair test of whether the public 10-layer story transfers, unlike our earlier naked depth test | score improves despite fewer steps | still loses, implying 10L is not worth pursuing on our path | conditional |
| S2-E8 | branch challenge | `S2-B0` plus export improvements | warmdown branch: `WARMDOWN_ITERS=20000`, higher LR, clip `1.0`, optional `EVAL_SEQ_LEN=1408` | resolves the biggest disagreement between proxy postmortem and public `8xH100` wins | smaller quant gap offsets any pre-quant loss and final score improves | instability or worse final score again | conditional |

## Promotion Logic

Promote from Wave A to Wave B only if:

- the effect size is meaningful relative to record noise,
- the artifact stays legal,
- and the change has a plausible causal story.

Promote from Wave B to three-seed confirmation only if:

- the candidate beats `S2-B0`,
- and the improvement is not obviously explainable by eval-only loophole risk or export accounting weirdness.

## What The Stage 1 Postmortem Changes

The postmortem in [postmortem_r1_r2.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage1/postmortem_r1_r2.md) gives three hard constraints for Stage 2:

- `seq2048` is a real positive prior
- `adaptive_muon` is unstable enough that it must be retested only on the exact strong stack
- `always-decay` as tested is not promotion-worthy and should not get an early Stage 2 slot

That means:

- use the clean training trunk for most Stage 2 attribution
- export and eval attribution come before more schedule speculation
- naked depth changes stay out until the public-stack prerequisites are present
- the absolute record SOTA remains the benchmark target, but not the attribution baseline

## Recommended Order

1. `sanity`: `S2-C0`, `S2-C1`, `S2-E1`, `S2-E2`, `S2-E3`, `S2-E4`, `S2-R0`, `S2-R1`
2. `screen`: the same eight-way pack in parallel
3. `final_single`: top promoted survivor only, default `top_k=1`
4. `champion_8x`: optional only after the single-GPU finalist looks real

## Default Interpretation Rules

- If `S2-E1` wins clearly: eval remains its own optimization frontier even after moving to a stronger trunk.
- If `S2-E2` wins: fp16 embedding export is portable and should move into the trunk.
- If `S2-E3` wins: our quant-quality work transfers and should be stacked.
- If `S2-E4` wins: Muon WD is not just a 10-layer trick.
- If `S2-E5` wins: overtone / phase-structured init is real in the 10-minute regime.
- If `S2-E6` loses again: adaptive Muon should be retired from the mainline path.
- If `S2-E6b` wins: the remaining frontier still has exploitable step-time slack.
- If `S2-E6c` wins: our Stage 1 winners are not separate from the public frontier; they are derivative improvements on top of it.

## Stage 2 Interpretation

The right mental model is:

- public record recipes tell us what already works
- our Stage 1 winners tell us where there may still be unclaimed slope

So Stage 2 should build on both:

- `record baseline` for realism
- `our winners` for derivative improvements

Especially:

- `quant_quality` is likely a derivative improvement on the current record export path
- `adaptive_muon` is likely a derivative improvement on the current record training-speed path

That is why Stage 2 needs both:

- attribution runs on public baselines
- and hybrid runs that test whether our winners still add value after adopting the record stack
