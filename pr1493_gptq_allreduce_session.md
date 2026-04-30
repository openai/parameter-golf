# PR1493 — GPTQ All-Reduce + Damp/Block Sweep Session

Session 2026-04-30. Picks up after `pr1493_wd_paired_session.md` left off with
q_ttt = 1.07974 / 1.07976 (session log / HF metadata) and a 0.0006-ish BPB gap to
the leaderboard target (~1.07910).

The headline plan we executed was the user's: implement GPTQ Hessian all-reduce
across ranks (the rank-0-only calibration was leaving 7/8 of compute on the table),
then sweep `GPTQ_DAMP` and `GPTQ_BLOCK_SIZE` to see if quantization knobs can
recover the rest.

## TL;DR

- **All-reduce works as a fix and saturates at 128 shards**: at full calibration data
  the AR delta is ≈ −7e-6 BPB on q_ttt — within machine noise. At reduced
  calibration (16 shards) the same fix is worth ~−0.00084 BPB and recovers almost
  the full 128-shard rank-0 result. So AR is a real correctness/efficiency fix,
  but **not** the lever that closes the leaderboard gap on its existing setup.
- **`GPTQ_DAMP`**: clean unimodal U-curve over `{0.005, 0.01, 0.02, 0.03, 0.05}`.
  **Default 0.01 is optimal**; the worst point (0.05) is +8e-5 BPB on q_sw, the
  closest neighbours are within +4e-6. No room to mine.
- **`GPTQ_BLOCK_SIZE`**: `{64, 128, 256}` all tied within 3e-6 BPB on q_sw. blk=64
  is marginally best; the lead is below within-config noise.
- **Best q_ttt achievable from GPTQ tuning alone**: 1.07975 (128-shard AR, damp=0.01).
  This effectively reproduces the prior wd_paired result. The remaining 0.0006 BPB
  gap to target will not close from GPTQ knobs.
- **Code shrink remains existential**: every quant blob in this session is
  16,030–16,034 KB (without code). Adding the ~55 KB script puts the submission
  ~30 KB over the 16 MB cap. Solving BPB without solving size yields nothing.

## Code changes (committed in this session)

### `train_pr1493.py`

1. **GPTQ Hessian all-reduce.** `collect_hessians` now `dist.all_reduce(SUM)` the
   per-rank Hessians and divides by `n_calibration_batches × world_size`. Gated
   by `GPTQ_ALL_REDUCE` (default on). Smoking-gun log line:
   `gptq:all-rank Hessian averaging across 8 ranks (denom=512)` when on,
   `gptq:per-rank Hessian (no all-reduce, denom=64)` when off. Sorted-key iteration
   so all ranks reduce in the same order (no deadlock risk even if iteration order
   ever drifted).
2. **`GPTQ_DAMP`, `GPTQ_BLOCK_SIZE` env vars.** Plumbed through `Hyperparameters`,
   `gptq_quantize_weight` signature, and `gptq_mixed_quantize` call site.
   New log line records what was actually used: `gptq:damp_frac=X block_size=Y`.

### `safe_launch.sh`

- Repo path made script-relative (was hard-coded `/workspace/parameter-golf`,
  wrong on a fresh container).
- `REQUIRED_SYMBOLS` extended with `'gptq:all-rank Hessian'`, `gptq_damp`,
  `gptq_block_size` so a rolled-back source file fails loudly before any GPU
  work happens.

### New: `requant_eval.py`

Standalone torchrun script that loads a saved FP `state_dict`, calls
`configure_eval_model` (otherwise `looping_active=False` and pre-quant val_bpb
explodes to ~1.27 — discovered the hard way), runs `serialize` (collect_hessians
→ GPTQ → brotli) into a custom `OUT_PTZ`, then runs the standard quantized,
sliding-window, and TTT evals. Reuses `train_pr1493`'s functions verbatim — no
code duplication, sweeps test the actual production path. Each cell is ~3 min
with TTT, ~2 min with `TTT_ENABLED=0` (TTT alone is ~8 min of the wallclock).

### New: `run_damp_sweep.sh`, `run_block_sweep.sh`

Driver scripts that loop the GPTQ knobs at fixed checkpoint, fixed seed, fixed
calibration shard set. TTT off for speed; only worth a TTT pass if q_sw shows
a real gap. (Nothing did.)

## Environment

- 8 × NVIDIA H100 80GB HBM3, fresh container.
- torch 2.9.1+cu128, flash_attn_3 cu128_torch291, brotli installed at session start
  (`brotli` was missing from `requirements.txt` and from the system pip).
- Dataset: `kevclark/parameter-golf` SP8192, downloaded via
  `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128`.
  16 train shards downloaded first for a small-data test; bumped to 128 once
  the calibration-quantity effect was apparent.
- Checkpoint: `models/pr1493_wd_paired_s42.pt` from
  `shikhar007/parameter-golf-gram-ns` HF repo (the root `final_model.pt` in that
  repo is a misnamed leftover from an old 1024-vocab experiment — wasted one A/B
  cell figuring this out). 135 MB, 8192-vocab, 11 blocks, skip_weights[8, 512].

## Result matrices

All numbers from `requant_eval.py` runs at `SEED=42 QK_GAIN_INIT=5.25
TTT_LR=0.007 TTT_EPOCHS=5 WD_SCHEDULE_ENABLED=1 PAIRED_HEAD_MUON_ENABLED=1`,
loading `wd_paired_s42.pt`. Reproducibility check: 128-shard no-AR damp=0.01
reproduced HF metadata (q_ttt 1.07976130) within 5e-7 BPB.

### A/B: all-reduce × calibration shard count (damp=0.01, blk=128)

| Calib shards | All-reduce | pre-quant | q | q_sw | **q_ttt** | size (B) |
|---|---|---|---|---|---|---|
| 16 | OFF | 1.08610 | 1.10117 | 1.08437 | **1.08060** | 16,034,159 |
| 16 | **ON** | 1.08610 | 1.09855 | 1.08178 | **1.07977** | 16,032,979 |
| 128 | OFF | 1.08610 | 1.09852 | 1.08175 | **1.07976** | 16,032,799 |
| 128 | **ON** | 1.08610 | 1.09850 | 1.08174 | **1.07975** | 16,032,588 |

**Δ all-reduce at 16 shards**: −0.00084 BPB on q_ttt. Substantial.
**Δ all-reduce at 128 shards**: −0.0000076 BPB on q_ttt. Within noise.

The 16-shard AR run almost exactly matches the 128-shard no-AR run on every
post-quant metric (q within 5e-5, q_sw within 5e-6, q_ttt within 7e-6). This
confirms the theory the user proposed: averaging Hessians across W ranks is
worth roughly W× more rank-0 calibration data. With 128 shards × 8 ranks already
giving rank 0 a 16-shard slice (which we now know is enough at the AR-aware
margin), the saturation point is reached with the existing data plumbing.

### Damp sweep (128 shards, AR=on, blk=128)

| damp | q | q_sw |
|---|---|---|
| 0.005 | 1.09851124 | 1.08174856 |
| **0.01** | **1.09849901** | **1.08173646** |
| 0.02 | 1.09851329 | 1.08174082 |
| 0.03 | 1.09851902 | 1.08175254 |
| 0.05 | 1.09857635 | 1.08181617 |

Clean unimodal: 0.01 minimum, 0.005 and 0.02 within 1.2e-5 BPB, 0.05 +8e-5
BPB worse. Default is essentially optimal — no room to mine. The GPTQ paper's
damp=0.01 transfers correctly to int6 + paired-head.

### Block-size sweep (128 shards, AR=on, damp=0.01)

| blk | q | q_sw |
|---|---|---|
| **64** | 1.09849940 | **1.08173441** |
| 128 | 1.09849901 | 1.08173646 |
| 256 | 1.09849755 | 1.08173731 |

All three within 3e-6 BPB on q_sw. blk=64 marginally best but well below
within-config noise (≈ 1e-5 BPB based on damp-sweep neighbour spread). Not
worth a TTT confirmation pass — even if blk=64's q_sw lead transferred 1:1
to q_ttt, we'd be at 1.07975, identical to the default.

## What this session does NOT settle

- **Cross-seed variance** of the 1.07975 result. Single-seed all the way through.
  Going for leaderboard p<0.01 needs 3 seeds; if cross-seed σ is ≥0.0004 (the
  prior data implies that ballpark), our mean-across-seeds is unlikely to clear
  the 0.005-nat bar even if the single-seed point is exactly on target.
- **Higher calibration-batch counts** (`GPTQ_CALIBRATION_BATCHES > 64`).
  At AR=on, going to 128 batches × 8 ranks = 1024 effective is in
  diminishing-returns territory — but I didn't actually measure it.
- **`MATRIX_CLIP_SIGMAS`** sweep. Currently 12.85; this is the int6 clip range
  knob and is plausibly more impactful than damp/block. Not run.
- **Code shrink.** Untouched. Submission still 30 KB over cap regardless of BPB.
- **Methodology changes**: `QK_GAIN_INIT` retune, `EMA_DECAY` retune, fresh
  wd_paired training. These are the most plausible remaining levers but require
  retraining and weren't in scope for this session's "GPTQ-only fix" pass.

## Critical assessment

The user's framing of all-reduce as "highest-confidence remaining technical fix"
was directionally correct (it IS a real fix that closes a real calibration gap)
but the magnitude prediction was wrong at the configuration we're actually running.
At 128-shard rank-stripe calibration on this specific model, the calibration
distribution per rank is already close to saturated for GPTQ's purposes; the
fix recovers the full 1.07976 baseline but doesn't push past it. The path to
leaderboard acceptance is **not** in GPTQ-knob land. It's in:

1. **Code shrink (mandatory).** Without 30 KB out, nothing else matters.
2. **Methodology change**: a fresh wd_paired training with one of the small-hparam
   tweaks the user listed (QK_GAIN_INIT={5.0, 5.5}, EMA_DECAY={0.9960, 0.9970},
   WD_SCHED_LOW={0.50}, WD_SCHED_HIGH={1.50}). Each is one full 10-min run.
   The plan rates these as "small full-run hparams only if GPTQ fails" — that
   condition is now met.
3. **3-seed proof** is downstream of (2) producing a single-seed result around
   1.0792. Until then, seed runs only confirm noise-bounds, not a win.

## Files added/modified in this session

Modified:
- `train_pr1493.py` — all-reduce path + new env vars
- `safe_launch.sh` — host-portable + extended REQUIRED_SYMBOLS

New:
- `requant_eval.py` — standalone GPTQ-eval driver
- `run_damp_sweep.sh`, `run_block_sweep.sh` — sweep drivers
- `pr1493_gptq_allreduce_session.md` — this file
- `logs/requant_baseline_noar.{stdout,txt}` — 16-shard no-AR
- `logs/requant_ar.{stdout,txt}` — 16-shard AR
- `logs/requant_128_noar.{stdout,txt}` — 128-shard no-AR (reproduction)
- `logs/requant_128_ar.{stdout,txt}` — 128-shard AR
- `logs/requant_d{0_005,0_02,0_03,0_05}.{stdout,txt}` — damp sweep
- `logs/requant_b{64,256}.{stdout,txt}` — block-size sweep
- `logs/{damp,block}_sweep.driver.log` — sweep driver outputs

Not committed (and shouldn't be):
- `wd_paired_s42.pt`, `requant_*.int6.ptz` — model + quantized blobs (gitignored)
- `data/datasets/fineweb10B_sp8192/*.bin` — 24 GB shards (gitignored)
