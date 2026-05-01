# HopperRegenAdapterB2B - Documented Failed Direction

**TL;DR - the regenerated-adapter idea failed in its current form.**
The best regen-adapter ablation here reached `val_bpb = 2.05623253` on a capped
1xH100 validation run. A matched base-only ablation reached `1.68527241` at the
same 297-step cap. That is a `0.37096012 bpb` regression for the best filtered
adapter setup, so this is not a leaderboard contender.

This is submitted to `records/track_non_record_16mb` as a compact negative
result. The reusable idea is a zero-byte regenerated adapter: frozen ternary
low-rank factors are generated from seeds and do not get serialized into the
artifact.

## Evidence

All headline runs used one Runpod H100 80GB HBM3, seed `1337`, SP1024 FineWeb,
`TRAIN_BATCH_TOKENS=524288`, `TRAIN_SEQ_LEN=1024`, and capped validation
(`EVAL_MAX_TOKENS=524288`). No 8xH100 result or SOTA claim is made.

| run | adapter config | steps | strict `val_bpb` | total bytes | note |
|---|---|---:|---:|---:|---|
| `train_runpod_1h100_smart_roles_ablation_297step.log` | `USE_REGEN_ADAPTER=1`, `REGEN_INCLUDE_ROLES=attn_c_v,attn_proj,mlp_fc,mlp_proj`, backend off | 297 | `2.05623253` | `8,099,240` | best regen variant |
| `train_runpod_1h100_native_10m.log` | all-role regen, `USE_HOPPER_REGEN_GEMM=native` | 297 | `2.28083634` | `7,439,076` | native bridge trainer gate |
| `train_runpod_1h100_base_ablation_297step.log` | `USE_REGEN_ADAPTER=0` | 297 | `1.68527241` | `8,323,872` | matched base-only control |

Skipping Q/K helps substantially versus all-role regen (`2.28083634` to
`2.05623253`), but the filtered adapter still loses clearly to base-only.

## Architecture

Each replaced projection computes:

```text
Y = X @ W.T + alpha * (X @ A.T) @ B.T
```

`W` is the normal learned projection. `A` and `B` are frozen ternary matrices in
`{-1, 0, +1}` regenerated from a 64-bit master seed with Rule-30 plus an
Achlioptas-style ternary expansion. They are non-persistent buffers, so the
artifact stores the normal model weights and tiny learned adapter scales, not the
materialized adapter matrices.

The trainer supports:

- pure PyTorch adapter math (`USE_HOPPER_REGEN_GEMM=off`)
- a Triton adapter wrapper (`USE_HOPPER_REGEN_GEMM=triton`)
- a small SM90 native extension bridge (`USE_HOPPER_REGEN_GEMM=native`)

The native bridge is included only as a correctness/repro dependency for the
submitted native log. It is not a final fused CUTLASS/WGMMA/TMA backend.

## Negative Results

- **All-role rank-16 scalar-alpha regen hurts badly.** The native all-role run
  reached `2.28083634` strict roundtrip `val_bpb`.
- **Skipping Q/K helps but does not fix it.** The smart-role run reached
  `2.05623253`, still `0.37096012 bpb` worse than base-only.
- **The native bridge is not a speed win.** The all-role native run took
  `601.473s` for 297 steps, while the matched base-only control took `261.165s`.
- **This needs redesign before more 8xH100 spend.** The next useful work is a
  targeted adapter-structure sweep, not a blind leaderboard run.

## Reproduction

Use the official Parameter Golf Runpod template and SP1024 cached FineWeb:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
cd records/track_non_record_16mb/2026-04-28_HopperRegenAdapterB2B
```

Best regen ablation:

```bash
RUN_ID=hopper_regen_smart_roles_ablation_297step \
SEED=1337 \
USE_REGEN_ADAPTER=1 \
USE_HOPPER_REGEN_GEMM=off \
REGEN_INCLUDE_ROLES=attn_c_v,attn_proj,mlp_fc,mlp_proj \
ITERATIONS=297 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_MAX_TOKENS=524288 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Base-only control:

```bash
RUN_ID=hopper_base_1h100_ablation_297step \
SEED=1337 \
USE_REGEN_ADAPTER=0 \
USE_HOPPER_REGEN_GEMM=off \
ITERATIONS=297 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_MAX_TOKENS=524288 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Files

- `README.md` - this writeup
- `submission.json` - compact metadata and comparisons
- `train_gpt.py` - submitted trainer, 1440 lines locally
- `requirements.txt` - root requirements copy
- `python/gemm_hopper.py` plus the small native-extension dependency files used
  by the optional native bridge path
- three `train_runpod_1h100_*.log` evidence logs

## Compliance

- [x] Non-record submission; no SOTA claim.
- [x] No 8xH100 claim.
- [x] Best submitted regen artifact is under the 16,000,000-byte decimal cap:
  `8,099,240` bytes.
- [x] `train_gpt.py` is under the 1500-line hard cap (`1440` lines locally).
- [x] No tokenizer or dataset scoring change is claimed.
- [x] No paid-prefix tricks; validation tokens are not compressed into the
  artifact.

## Compute Note

I am interested in compute support only if reviewers think this primitive is
worth developing further. The useful next spend would be targeted H100 ablations
and a real fused-kernel validation pass, not an immediate 8xH100 leaderboard run.

