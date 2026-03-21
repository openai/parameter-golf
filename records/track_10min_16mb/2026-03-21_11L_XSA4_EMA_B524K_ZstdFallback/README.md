# Record: 11L XSA4 + EMA + Batch524K + zstd Fallback

**val_bpb = 1.1357** (sliding window, stride=64) | **15.67 MB** artifact | 8xH100 SXM, ~600s

Single-seed submission using an 11-layer int6 MLP3x model with XSA on the last 4 layers, EMA averaging, SmearGate, BigramHash, and a 524K fixed-time batch setting.

## Result

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 1.1529 |
| Int6 roundtrip val_bpb | 1.1580 |
| **Int6 sliding val_bpb (stride 64)** | **1.1357** |
| Model bytes (int6+zstd) | 15,603,062 |
| Code bytes | 66,891 |
| **Total submission bytes** | **15,669,953** |

This is below the current merged README SOTA (`1.1428`) but it is not a 3-seed validated record claim.

## What's New

| Change | Impact |
|--------|--------|
| `TRAIN_BATCH_TOKENS=524288` | Better fixed-budget step count than the larger-batch 11-layer XSA+EMA setting |
| SDPA fallback for `flash_attn_interface` | Runs cleanly when FA3 Python bindings are unavailable in the official image |
| `torch.compile` behind an env flag | Reliable eager smoke tests, faster compiled full run |
| `zstd` Python-or-CLI fallback | Keeps int6 export under 16MB without depending on a specific Python package in the image |

## Configuration

```bash
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
XSA_LAST_N=4 EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 WARMUP_STEPS=20 ENABLE_TORCH_COMPILE=1 \
MAX_WALLCLOCK_SECONDS=600 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Run Details

| Metric | Value |
|--------|-------|
| Steps reached | 8,202 |
| Average train step time | 73.37 ms |
| Peak memory allocated | 13,879 MiB |
| Peak memory reserved | 14,004 MiB |
| Final eval mode | Sliding window, stride 64 |

## Included Files

- `train_gpt.py` — training, export, and eval script
- `run_hybrid_attempt.sh` — launch wrapper used for the run
- `train.log` — full log from the validated 600s attempt
- `submission.json` — metadata for the submission
