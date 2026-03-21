This record captures the **Standard 10-Layer Transformer with BigramHash, SWA, Late QAT, and Weight Decay**.

## Approach

A clean 10-layer standard transformer (no depth-recurrence) with several quality-of-life improvements over the baseline:

- **10 independent layers** instead of 9 — extra capacity at the same step budget
- **MLP 3× expansion** (hidden dim = 3 × model_dim = 1248) — wider MLP outperforms wider attention per parameter
- **BigramHash(4096, dim=64)** — maps (prev_token, curr_token) pair to a small embedding added before the first layer, providing bigram context with minimal size cost (~384 KB quantized)
- **Late QAT** (activates at 85% of wallclock) — model trains freely in bf16 first, then STE fake-quantize fine-tunes for int8 tolerance in the final 15% of training
- **SWA** (Stochastic Weight Averaging from 40% of training, every 50 steps) — averages snapshots over the second half of training for a smoother loss landscape
- **Weight decay 0.04** on Adam optimizers — regularizes weights toward smaller magnitudes, improving int8 quantization quality
- **Extended RoPE base = 50000** — better long-range positional encoding
- **Gradient clipping = 1.0** — prevents NaN divergence
- **GQA** (4 KV heads, 8 query heads) — reduces KV projection cost

## Configuration

```
NUM_LAYERS=10  MODEL_DIM=416  NUM_HEADS=8  NUM_KV_HEADS=4  MLP_MULT=3
BIGRAM_BUCKETS=4096  BIGRAM_DIM=64
ROPE_BASE=50000  LOGIT_SOFTCAP=30.0
QAT_ENABLED=1  QAT_START_FRAC=0.85
SWA_ENABLED=1  SWA_START_FRAC=0.4  SWA_INTERVAL=50
EMBED_LR=0.05  MATRIX_LR=0.025  SCALAR_LR=0.04  WEIGHT_DECAY=0.04
GRAD_CLIP_NORM=1.0
TRAIN_BATCH_TOKENS=524288  TRAIN_SEQ_LEN=1024  MAX_WALLCLOCK_SECONDS=600
```

## Command

```bash
torchrun --standalone --nproc_per_node=8 our_train_gpt.py
```

(Run via Modal with 8×H100 SXM5)

## Key metrics

- Timed training stopped due to wallclock cap at 10 min
- Pre-quant eval: `val_loss:2.9329`, `val_bpb:1.7370`  ← first run (dim=512, too large — model didn't save)
- Post-quant roundtrip: TBD (pending successful run with dim=416 fix)
- Step avg: ~52ms/step on 8×H100
- Peak memory: ~12308 MiB

## Included files

- `train_gpt.py` — training script
- `submission.json` — leaderboard metadata
- `README.md` — this file
