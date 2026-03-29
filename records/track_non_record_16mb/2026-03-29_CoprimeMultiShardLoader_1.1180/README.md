# Coprime Multi-Shard Loader Delta

**Track:** Non-record 16MB submission  
**Final score:** `final_int6_sliding_window_exact val_bpb = 1.11796805`  
**Artifact size:** `15,871,257` bytes total (`15,753,772` model bytes + `117,485` code bytes)  
**Hardware:** 8xH100 SXM, 600s wallclock  

## Overview

This folder contains a complete non-record 16MB submission package for a coprime multi-shard loader result on a clean no-TTT trunk.

The packaged result is:

- compliant under the exact `16,000,000`-byte cap
- free of pre-quant TTT
- complete with exact terminal metrics
- self-contained for audit and reproduction

## Method

The packaged script corresponds to source commit `0d6643923382fd124b41cd37d70e44855925b9bc` and uses the following boundary:

- `train_loader:mode:coprime_multi_shard_phase2_no_prefetch`
- `TTT_ENABLED=0`
- `TARGET_TOTAL_BYTES=16000000`
- `BIGRAM_VOCAB_SIZE=3072`
- `BIGRAM_DIM=112`
- `XSA_LAST_N=11`
- `MOUSSE=0`
- no mixed export
- no pre-quant prune

`train_gpt.py` in this folder is the exact script snapshot used for this package.

## Results

### Compliance

- `TIMING:pre_quant_ttt=0.0s enabled=False`
- `selective_prune_size_budget total_cap_bytes=16000000 code_bytes=117485 target_model_bytes=15882515 unpruned_model_bytes=15753772 unpruned_total_bytes=15871257 giveback_needed_bytes=0 ... mode=post_quant`
- `selective_prune: already fits, no pruning needed headroom_bytes=128743`
- `Total submission size int6+lzma: 15871257 bytes`
- `Serialized model int6+lzma: 15753772 bytes`
- `TIMING:ar_selfgen_gptq=329.9s`
- `TIMING:final_eval=140.7s`

### Quality

- `DIAGNOSTIC post_ema val_loss:1.9202 val_bpb:1.1373`
- `final_int8_zlib_roundtrip_exact val_loss:1.88763723 val_bpb:1.11796805`
- `final_int6_sliding_window_exact val_loss:1.88763723 val_bpb:1.11796805`
- `final_int6_roundtrip_exact val_loss:1.92728468 val_bpb:1.14144655`

## Reproduction

The packaged run used:

```bash
SEED=1337
TTT_ENABLED=0
BIGRAM_VOCAB_SIZE=3072
BIGRAM_DIM=112
XSA_LAST_N=11
WARMDOWN_ITERS=4000
TARGET_TOTAL_BYTES=16000000
MOUSSE=0
ROPE_DIMS=16
LN_SCALE=1
VE_ENABLED=1
VE_DIM=128
VE_LAYERS=9,10
TRIGRAM=0
GATED_ATTENTION=0
VALUE_RESIDUAL=0
DATA_PATH=./data/datasets/fineweb10B_sp1024
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
```

The packaged script hash is:

- `train_gpt.py` SHA-256: `02bb2c458c0a1a310e8b2f888f1034cb18062a1f632c7129970d28b0f2036b1f`

## Files

- `submission.json`: machine-readable metadata for the packaged run
- `train.log`: chronological training/eval log
- `train_seed1337.log`: seed-specific training/eval log
- `train_gpt.py`: exact script snapshot for this package
- `requirements.txt`: reference dependency list copied alongside the script
