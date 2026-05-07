# 11L INT7 + MuonWD + SWA

## Key Innovation: INT7 Quantization

INT7 (7-bit, 127 levels) is a quantization sweet spot between INT8 and INT6:

| Quant | Post-quant degradation | Size (11L) |
|-------|----------------------|------------|
| INT8  | 0.009 bpb            | 13.1 MB    |
| **INT7** | **0.020 bpb**     | **10.5 MB** |
| INT6  | 0.116 bpb            | 7.7 MB     |

INT7 saves 2.6 MB vs INT8 with minimal quality loss, enabling deeper architectures within the 16 MB budget. INT6 degrades too much (0.12 bpb) even with QAT and percentile clipping.

## Architecture

- 11 unique transformer layers at d=512, 8 heads, 4 KV heads
- Tied embeddings, 2x MLP expansion
- ~20M parameters

## Training Modifications

- **Muon weight decay**: WD=0.04 applied to matrix parameters, reduces weight magnitude and improves quantization
- **SWA (Stochastic Weight Averaging)**: Checkpoint averaging over last 25% of training
- **INT7 per-row quantization**: Percentile clipping (99.99984th) + zlib compression

## Preliminary Results (1xH100, 5 min, 710 steps)

- Pre-quant: val_bpb = 1.3672
- Post-quant INT7+zlib: val_bpb = 1.3874
- Total size: 10.5 MB (5.5 MB headroom for deeper models)

## Planned 8xH100/10min Configuration

With ~12,000 steps available on 8xH100, the optimal config shifts to 13-15 layers:
- 13-15L d=512 INT7: fits in 16 MB, more depth = better loss
- Expected val_bpb: ~1.15-1.20 range (extrapolating from scaling behavior)

## Command

```bash
RUN_ID=11L_int7_wd_swa \
NUM_UNIQUE_LAYERS=11 \
NUM_RECURRENCE_ITERS=1 \
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
QAT_BITS=7 \
MUON_WEIGHT_DECAY=0.04 \
SWA_START_PCT=2.0 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Experiment Log (17 runs on 1xH100)

| Config | Steps | Post-quant bpb | Size |
|--------|-------|---------------|------|
| 9L INT8 baseline | 775 | 1.3932 | 11.5MB |
| 9L INT8 +WD | 762 | 1.3775 | 11.0MB |
| **11L INT8 +WD +SWA** | **712** | **1.3737** | **13.1MB** |
| **11L INT7 +WD** | **710** | **1.3874** | **10.5MB** |
| 13L INT8 +WD | 599 | 1.3921 | 14.6MB |
| 13L INT7 +WD | 602 | 1.4135 | 11.4MB |
| 5x2 recurrence INT6 | 785 | 1.5421 | 3.9MB |
| 11L INT6 QAT | 702 | 1.4880 | 7.7MB |
