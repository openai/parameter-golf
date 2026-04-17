# 11L XSA-All + EMA + Legal GPTQ on 8xH100

**Track:** Non-record / Unlimited Compute (`16MB`)
**Author:** Rtx09x
**GitHub ID:** [@Rtx09x](https://github.com/Rtx09x)
**Base:** `2026-04-04_11L_XSA11_EMA_GPTQ_1xH100_PCIe`, rerun legally on `8x H100`
**Hardware:** `8x H100 80GB HBM3`
**Cloud:** RunPod Parameter Golf template
**Date:** 2026-04-17

## Result

- **Final exact sliding-window score:** `1.11355040 bpb`
- **Final exact sliding-window loss:** `1.88017824`
- **Final exact post-quant roundtrip score:** `1.13694354 bpb`
- **Final exact post-quant roundtrip loss:** `1.91968154`
- **Post-EMA pre-quant score:** `1.1248 bpb`
- **Post-EMA pre-quant loss:** `1.8991`
- **Final compressed artifact size:** `15,249,936 bytes`
- **Total submission size:** `15,353,950 bytes`
- **Train stop:** `6460` steps at `600.119s`

## Key Log Lines

```text
step:6460/20000 val_loss:1.9010 val_bpb:1.1259 train_time:600119ms step_avg:92.90ms
DIAGNOSTIC post_ema val_loss:1.8991 val_bpb:1.1248 eval_time:2227ms
Serialized model int6+lzma: 15249936 bytes
Total submission size int6+lzma: 15353950 bytes
final_int6_roundtrip_exact val_loss:1.91968154 val_bpb:1.13694354
final_int6_sliding_window_exact val_loss:1.88017824 val_bpb:1.11355040
final_int8_zlib_roundtrip_exact val_loss:1.88017824 val_bpb:1.11355040
```

## What This Run Is

- Legal `600s` run on `8x H100`
- Same `11L XSA-all + EMA + legal GPTQ` stack first screened on `1x H100 PCIe`
- Same export hardening:
  - stronger Cholesky damping retries
  - percentile int6 fallback if Hessian factorization still fails
- Explicit pre-quant checkpoint saved before export:
  - `final_model_pre_quant.pt`

## Configuration Notes

- tokenizer/data: `sp1024`
- `80` train shards
- `XSA` on all `11` layers
- `EMA`
- `MLP_MULT=4.0`
- `QK_GAIN_INIT=4.0`
- wallclock cap: `600s`
- legal autoregressive self-generated GPTQ calibration:
  - `64` sequences
  - `2048` tokens each
  - `temp=0.8`

## Reproduction Notes

This run was launched on the official RunPod Parameter Golf template with:

```bash
python3 runpod/notebook_run.py --profile safe-final-auto
```

Effective setup:

- `DATA_PATH=./data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
- `VOCAB_SIZE=1024`
- `MAX_WALLCLOCK_SECONDS=600`
- `TRAIN_LOG_EVERY=250`
- `VAL_LOSS_EVERY=10000`
- `GPTQ_CALIB_BATCHES=64`
- `SEED=1337`

## Artifact Notes

Local large artifacts are stored under [`artifacts/`](./artifacts) but git-ignored:

- `artifacts/final_model_pre_quant.pt`
- `artifacts/final_model.pt`
- `artifacts/final_model.int6.ptz`

Checksums and byte counts are recorded in:

- `artifacts_manifest.local.json`
