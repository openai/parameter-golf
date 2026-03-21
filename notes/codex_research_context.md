# Codex Research Context

Last updated: 2026-03-18

## Objective

This repo is for the Parameter Golf challenge:
- optimize final compressed-model performance under a strict `16,000,000` byte artifact cap
- leaderboard runs must finish within `10` minutes on `8xH100`
- current working assumption for architecture research: keep tokenizer fixed for now

For fixed tokenizer experiments:
- use `val_loss` as the main local metric
- `val_bpb` is equivalent up to a constant on the fixed validation split
- use `train_loss` only as a sanity/debug signal, not as the main ranking metric

## Current Local Setup

Primary local research path:
- [train_gpt_mlx.py](/Users/soumil/Desktop/parameter-golf/train_gpt_mlx.py)

Important local adjustments already made:
- added `VAL_MAX_TOKENS` to support a mini-val path on a fixed prefix of the validation split
- added configurable AttnRes modes:
  - `ATTNRES_MODE=full`
  - `ATTNRES_MODE=block`
  - `ATTNRES_MODE=hybrid`
- added AttnRes knobs:
  - `ATTNRES_BLOCK_SIZE`
  - `ATTNRES_LOCAL_BLEND`

Why mini-val was added:
- full local validation was too slow for architecture screening
- one full-val control run spent more than ~4m45s in final eval after training had finished
- mini-val makes local iteration practical

Current mini-val protocol:
- `VAL_MAX_TOKENS=4194304`
- `ITERATIONS=200` for short screens
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_LOSS_EVERY=0`
- `VAL_BATCH_SIZE=8192`
- single local train shard currently available
- for local Mac research runs, set `MAX_WALLCLOCK_SECONDS=0`

Note on baseline controls:
- AttnRes lives in the working-tree version of `train_gpt_mlx.py`
- for apples-to-apples baseline controls, a temporary baseline copy was generated from `HEAD`
- if needed again, recreate a clean baseline script from git `HEAD` rather than manually editing the working tree

## AttnRes Hypothesis

Question being tested:
- can a constrained AttnRes variant show enough local promise to justify further investment?

Important framing:
- AttnRes is *not* currently assumed to be the main source of leaderboard gains
- it is being treated as a possible helper or stabilizer, especially for more aggressive backbones later

## AttnRes Results So Far

Short local screen, `200` steps, mini-val, seed `1337`:

- baseline mini-val roundtrip `val_loss`: `3.97771597`
- block AttnRes roundtrip `val_loss`: `4.03348017`
- hybrid AttnRes roundtrip `val_loss`: `3.90412021`

Interpretation:
- naive `full` depth AttnRes is already considered a weak direction
- `block` lost clearly in the first screen and is deprioritized for now
- `hybrid` is the only AttnRes variant that has shown positive signal so far

Tradeoff observed:
- `hybrid` improved mini-val loss at `200` steps
- `hybrid` also appears slower than baseline
- next question is whether the gain survives longer training

Current/next AttnRes check:
- `1000`-step mini-val baseline vs `1000`-step mini-val hybrid
- same seed, same local protocol

Stage-1 result on the first `1000`-step comparison:
- baseline completed all `1000` steps and reached roundtrip mini-val `val_loss: 3.29495144`
- hybrid hit the default wallclock cap at step `859`, but still reached roundtrip mini-val `val_loss: 3.25895929`
- this is promising enough to keep hybrid alive, but the next local runs should be uncapped so comparisons are not truncated by the default `600s` limit

Decision rule:
- if `hybrid` still wins at `1000` steps, keep AttnRes alive
- if the gain disappears, deprioritize AttnRes and move on

## Main Architecture Priorities

Current belief about likely primary gain sources:

1. Effective depth with parameter sharing / recurrence
- best fit to the challenge
- spends compute instead of bytes
- shared parameters often compress better

2. Latent memory / eval-time memory
- good match for a byte-constrained challenge
- likely most useful once attached to a recurrent or shared-depth backbone

3. Compressibility-aware architecture
- repeated structure, shared matrices, low-rank/shared parameterization, and small control tensors should help both model quality per byte and final zlib artifact size

4. Residual routing / AttnRes
- secondary lever
- more interesting if it helps a deeper tied/recurrent model work better

## Main Research Families

### Family A: Residual Routing / AttnRes

Variants already considered:
- baseline control
- full AttnRes
- block AttnRes
- hybrid AttnRes

Potential further AttnRes variants if hybrid survives:
- selective AttnRes only in deeper layers
- AttnRes only before MLP
- hybrid with different `ATTNRES_LOCAL_BLEND`
- block + hybrid only if hybrid looks strong enough to justify one more branch

### Family B: Effective Depth / Weight Sharing / Recurrence

This is currently the highest-priority non-AttnRes family.

Planned variants:
- one shared block repeated many times
- small cycle of unique blocks repeated, e.g. 3 blocks repeated over depth
- separate encoder-shared and decoder-shared stacks
- shared-depth with iteration/depth-step conditioning

Why this matters:
- likely the strongest path to "more model per byte"
- likely more important than AttnRes alone

### Family C: Latent Memory

This should be explored after or alongside a promising recurrent/shared-depth backbone.

Planned variants:
- recurrent memory tokens carried across segments
- compressed latent summary passed between segments
- tiny memory cross-attention path with a small fixed latent bank
- only later: a Titans-lite memory update scheme if simpler memory shows signal

Why not jump straight to full Titans:
- too much complexity as a first proof step
- simpler memory mechanisms are better filters

## Other Potential Ideas Not Yet Fully Discussed

These are plausible future directions if the main families do not dominate:

- low-rank factorization of larger shared matrices
- compressibility-aware training or QAT-style export alignment
- learned iteration embeddings for recurrent/shared-depth models
- selective untied "control tensors" on top of heavily shared weights
- chunkwise recurrent evaluation with longer effective context
- memory tokens plus shared-depth recurrence
- stronger skip/bridge structure in the hourglass backbone
- evaluation-time recurrence schedules that spend more compute at test time than train time

Lower priority for now:
- tokenizer changes
- broad hyperparameter brute force
- anything that confounds architecture research before we know the backbone direction

## Rough Plan

### Phase 0: Local Harness
- baseline sanity runs
- mini-val support
- fast local screen protocol

Status:
- mostly complete

### Phase 1: Local Architecture Screen

Goal:
- kill bad ideas cheaply on the Mac

Current order:
1. close out AttnRes decision
2. move to recurrence / shared-depth
3. add latent memory on top of the best backbone

Promotion rules from local screen:
- better or equal mini-val loss than baseline
- no major throughput collapse
- stable training behavior

### Phase 2: CUDA / H100 Pilot

Goal:
- verify that local winners survive the real competition stack

Promote only a small shortlist:
- best AttnRes candidate if any
- best recurrence/shared-depth candidate
- best latent memory candidate
- best combo only if it has already shown signal locally

### Phase 3: 8xH100 Funnel

Goal:
- spend expensive runs only on serious contenders

Sequence:
- one leaderboard-style calibration run per top candidate
- a small focused sweep around the best one
- repeated final verification runs for variance/significance

## Compute Request Timing

Do not wait for every local experiment to finish.

Submit the compute request once we have:
- a stable local harness
- a written plan
- at least one credible non-baseline direction

Reason:
- request/approval may take time
- the plan is already legitimate once a first shortlist exists

## Immediate Next Steps

1. finish the `1000`-step mini-val baseline vs hybrid AttnRes comparison
2. if hybrid survives, run one more seed or one blend ablation
3. then start Family B: recurrence/shared-depth
4. once a good shared-depth backbone exists, attach Family C: latent memory
5. then decide which candidates deserve CUDA/H100 pilots

## Working Belief

Most likely eventual leaderboard candidate:
- a shared-depth / recurrent backbone
- possibly with a small latent memory path
- with AttnRes only if it proves helpful as a supporting mechanism
