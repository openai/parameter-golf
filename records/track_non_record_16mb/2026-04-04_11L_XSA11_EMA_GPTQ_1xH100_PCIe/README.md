# 11L XSA-All + EMA + Legal GPTQ on 1xH100 PCIe

**Track:** Non-record / Unlimited Compute (`16MB`)
**Author:** Rtx09x
**GitHub ID:** [@Rtx09x](https://github.com/Rtx09x)
**Base:** `PR #1019` lineage, forked into `working/current_frontier/train_gpt.py`
**Hardware:** `1x H100 PCIe 80GB`
**Cloud:** RunPod Community, Parameter Golf PROTEUS+STYX template
**Date:** 2026-04-04

## Result

- **Final exact post-quant score:** `1.15466807 bpb`
- **Final exact post-quant loss:** `1.94960867`
- **Post-EMA pre-quant score:** `1.1416 bpb`
- **Post-EMA pre-quant loss:** `1.9275`
- **Final compressed artifact size:** `15,139,756 bytes`
- **Total submission size:** `15,243,770 bytes`
- **Train stop:** `4216` steps at `4800.442s`

This is a **non-record** submission. It is not eligible for the 10-minute-on-8xH100 main leaderboard because the training run used a single `H100 PCIe` for `~80 minutes`.

## Key Log Lines

```text
step:4216/20000 val_loss:1.9286 val_bpb:1.1422 train_time:4800442ms step_avg:1138.62ms
DIAGNOSTIC post_ema val_loss:1.9275 val_bpb:1.1416 eval_time:28741ms
Serialized model int6+lzma: 15139756 bytes
Total submission size int6+lzma: 15243770 bytes
final_int6_roundtrip val_loss:1.9496 val_bpb:1.1547 eval_time:54869ms
final_int6_roundtrip_exact val_loss:1.94960867 val_bpb:1.15466807
```

## What This Run Changed

- Started from the merged `#1019` stack rather than the OpenAI starter baseline.
- Used the branch script with:
  - `11` layers
  - all-layer `XSA`
  - `EMA`
  - `MLP_MULT=4.0`
  - `QK_GAIN_INIT=4.0`
  - higher weight decay defaults
  - `sp1024` dataset/tokenizer path for the first long screening run
- Hardened GPTQ export to avoid crashing on non-positive-definite Hessians:
  - retries Cholesky with stronger diagonal damping
  - falls back to percentile int6 quantization if Hessian factorization still fails
- Saved an explicit pre-quant checkpoint before export:
  - `final_model_pre_quant.pt`

## Artifact Notes

The downloaded model files are kept locally under [`artifacts/`](./artifacts) for submission packaging, but they are git-ignored so the PR does not try to commit large binaries:

- `artifacts/final_model_pre_quant.pt`
- `artifacts/final_model.pt`
- `artifacts/final_model.int6.ptz`

Checksums and exact byte counts for those local files are recorded in:

- `artifacts_manifest.local.json`

## Reproduction Notes

The run was launched from the custom RunPod notebook launcher in the companion fork, using the equivalent of:

```bash
python3 runpod/notebook_run.py --profile branch-screen
```

Important effective settings:

- `DATA_PATH=./data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`
- `VOCAB_SIZE=1024`
- `TRAIN_SHARDS=80`
- `MAX_WALLCLOCK_SECONDS=4800`
- `VAL_LOSS_EVERY=4000`
- `BIGRAM_VOCAB_SIZE=0`
- `SMEAR_ENABLED=0`

## Interpretation

This run is useful for two reasons:

1. It establishes a valid non-record result with a final exact score under the 16MB cap.
2. It shows the branch is strong enough to justify a true next iteration toward the April frontier stack rather than more infrastructure work.

The next steps should be architectural, not operational:

- port depth recurrence from the April PR implementations
- port parallel residual routing
- revisit tokenizer/vocab upgrades for a real leaderboard attempt
