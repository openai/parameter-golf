# MergedTop3_v3 clean H100 rerun

This folder is the clean rerun candidate for the merged top-stack recipe that previously produced a strong recovered 8x H100 result. It keeps the same model recipe as `2026-03-24_MergedTop3_v1_prep`, but strips out the recovery-specific framing and adds only the runtime controls needed for a fresh uninterrupted 8x H100 attempt.

## Recipe

- 11 layers
- XSA on the last 4 layers
- EMA only
- 3x MLP
- SmearGate
- BigramHash with 2048 buckets
- mixed int6 quantization + zstd
- seq_len 2048
- Muon/AdamW weight decay 0.04
- sliding eval stride 64
- Partial RoPE with `ROPE_DIMS=16`
- layerwise LN scaling
- GPTQ-lite clip search
- `WARMDOWN_ITERS=3500`

## What changed relative to `v1`

- Added `WALLCLOCK_BUFFER_SECONDS` so a clean run stops before the strict 600 second cap.
- Default one-shot script now forces `AUTO_RESUME=0`.
- Default one-shot script now sets `VAL_LOSS_EVERY=0` so the 600 second budget is spent on training, with final exact/export only at the end.
- Added volume-backed environment bootstrap and strict remote preflight.
- Added a distributed stability probe that checks:
  - flash attention availability
  - NCCL collectives
  - checkpoint save/load on attached volume

## Intentionally not changed

- No late-QAT threshold path.
- No recurrence, groupfield, or other RIOM-only feature.
- No new bucket-size sweep.

The reason is simple: this folder is for a clean reproduction of the strongest H100 direction we have actually seen, not for new feature exploration.

## Expected remote flow

From inside this folder:

```bash
bash bootstrap_remote_env.sh
NPROC_PER_NODE=1 bash preflight_remote_strict.sh
NPROC_PER_NODE=8 bash preflight_remote_strict.sh
bash run_8xh100_one_shot.sh
bash collect_artifacts.sh
```

## Notes on preflight

`preflight_remote_strict.sh` is intentionally strict:

- it requires the tokenizer and full dataset staging to be present
- it requires enough visible GPUs for the requested launch width
- it requires flash attention to be usable unless explicitly disabled
- it runs both single-process and distributed stability probes before training

If this preflight fails, do not spend 8x H100 time on the run.

## Clean run result

Fresh uninterrupted 8x H100 run completed on 2026-03-25 with:

- `step_stop=5347`
- `train_time=580.213s`
- `final_int6_roundtrip_exact val_loss=1.96565872`
- `final_int6_roundtrip_exact val_bpb=1.16417381`
- `eval_time=44.398s`
- `bytes_model_int6_zstd=15,562,277`
- `bytes_code=72,924`
- `bytes_total=15,635,201`

This run stayed under the strict 600 second training cap and under the 16,000,000 byte artifact cap.

## Notes

- This is a single-seed clean run, not a multi-seed statistical record claim.
- The final metrics were first captured from the streamed run output and then cross-checked after reopening SSH and retrieving the original remote files.
- `train_seed1337.log` in this folder is the original remote run log retrieved after the run.
