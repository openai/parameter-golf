# Non-Record Submission: 1.20289664 BPB — Mixed-Int6 LZMA9 B3072 Warm5000

**EMA + XSA(last-4) + BigramHash3072 + warmdown5000 + LeakyReLU^2 + mixed-int6 export**

**val_bpb: 1.20289664** (sliding, seed=42) | **15,991,188 bytes** artifact | single-GPU unlimited-compute run (~16.1h)

> **This is a non-record unlimited-compute submission.** Training ran for about 16.1 hours on a single GPU, so this is not a 10-minute leaderboard attempt. The main result is a stronger legal artifact than the listed 4-hour non-record baseline, using a known flat-transformer recipe plus longer single-GPU training and a broad mixed-int6/LZMA9 export path.

## Result

| Metric | Value |
|--------|-------|
| Sliding BPB | **1.20289664** |
| Sliding val_loss | **1.99963255** |
| Pre-quant sliding BPB | **1.16618894** |
| Pre-quant sliding val_loss | **1.93861159** |
| Steps | **16,000** |
| Training time | **57,979.039s** (~16.1h) |
| Artifact | **15,991,188 bytes** |
| Code bytes | **110,016** |
| Compressed model bytes | **15,881,172** |
| Parameters | **27,124,828** |

## Positioning

This is **not** claiming a new SOTA technique. EMA, XSA, BigramHash, LeakyReLU^2, int6-style quantization, LZMA compression, and sliding evaluation are all established in prior submissions.

The useful contribution is a non-record data point showing that the flat EMA/XSA/BigramHash family still improves under longer single-GPU training, and that a broad `mlp;attn;embed` mixed-int6 export with LZMA9 can keep the resulting 27.1M-parameter checkpoint legal under the 16MB artifact cap.

| Reference | BPB | Notes |
|-----------|----:|-------|
| This submission | **1.20289664** | 16.1h single-GPU, non-record, 15,991,188 bytes |
| 4-hour non-record baseline | 1.20737944 | Listed unlimited-compute baseline |
| 1-bit non-record | 1.1239 | Stronger non-record result; this submission does not beat it |
| Current 10-minute record | 1.11473509 | Stronger 10-minute SOTA; this submission is not a record attempt |

## What Changed

- **Longer training on the flat stack:** `BIGRAM_VOCAB_SIZE=3072`, XSA on the last 4 layers, EMA, LeakyReLU^2, and `WARMDOWN_ITERS=5000` produced a raw sliding score of **1.16618894 BPB** before legal export.
- **Broad mixed-int6 export:** `QUANT_INT6_CATS=mlp;attn;embed`, `INT8_KEEP_FLOAT_MAX_NUMEL=32768`, and LZMA9 extreme compression produced a legal artifact at **15,991,188 bytes**.
- **Export separation:** the raw checkpoint was preserved, then re-exported and evaluated independently. The export found **3,894,003** candidate `+-1` int6 entries, but the compressed artifact already fit the target byte cap, so no entries had to be pruned.

## Training Configuration

- 8 FineWeb SP1024 training shards
- EMA enabled, decay `0.997`
- XSA active on the last 4 layers
- BigramHash `3072 x 128`
- `leaky_relu2` activation with slope `0.5`
- `TRAIN_BATCH_TOKENS=262144`
- `TRAIN_SEQ_LEN=2048`
- `ITERATIONS=16000`
- `WARMDOWN_ITERS=5000`
- `MAX_WALLCLOCK_SECONDS=64800`

The exact training log is included in [train.log](./train.log).

## Export Configuration

- Eval-only export from the preserved raw checkpoint
- `QUANT_SCHEME=mixed_int6`
- `QUANT_INT6_CATS=mlp;attn;embed`
- `INT8_KEEP_FLOAT_MAX_NUMEL=32768`
- `QUANT_SELECTIVE_PRUNE=1`
- `QUANT_TARGET_TOTAL_BYTES=16000000`
- `QUANT_LZMA_PRESET=9`
- `QUANT_LZMA_EXTREME=1`

The exact export/eval log is included in [export_eval.log](./export_eval.log).

## Files

- `train_gpt.py`: exact script for this submission branch
- `requirements.txt`: environment reference
- `train.log`: training log
- `export_eval.log`: export and final eval log

## Compliance

- [x] Artifact <= 16,000,000 bytes
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] Self-contained script included
- [x] Non-record unlimited-compute track
