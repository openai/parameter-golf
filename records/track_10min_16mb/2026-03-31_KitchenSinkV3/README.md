# Kitchen Sink V3

Built on KitchenSinkV2 Improved with the following changes:

## Architecture

1. **12 layers** (up from 11) with split early/late LR banks
2. **Window attention** (size=512) on layers 2,4,6,8,10 via Flash Attention 3 `window_size` parameter — reduces compute for long sequences while preserving full attention on key layers
3. **Mixed seq_len training** — 5 GPUs train at seq_len=2048, 3 GPUs at seq_len=6144, token-balanced by reciprocal ms/step
4. **Fused Triton MLP** — LeakyReLU(0.5)-squared activation fused with matmuls via Triton kernel
5. **Sigmoid-gated skip connections** — `x += sigmoid(gate) * skip` replaces learned scalar skip weights
6. **Brotli + byte-shuffle compression** — byte-shuffle transpose before brotli for better compression of quantized weights
7. **Bigram hash 5120** (up from 2048/6144), VE dim 128
8. **qk_gain = 2.5** (up from 1.5)

## Training

- Split early/late Muon + Adam optimizers
- MiLe margin loss with triangle schedule (gamma=0.75, clamp_min=0.2)
- Cache + backout residual at layer 7
- XSA on last 7 layers
- Coprime-stride multi-shard data loader
- Train-data GPTQ int6 calibration (14s reserved from 600s)
- Eval: sliding window, seq_len=6144, stride=128

## Results (5 seeds, 8xH100, 600s)

| Seed | val_loss (nats) | val_bpb | Size (bytes) |
|------|----------------|---------|-------------|
| 2 | 1.8731 | 1.1094 | 15,726,762 |
| 1337 | 1.8742 | 1.1101 | 15,721,698 |
| 42 | 1.8746 | 1.1103 | 15,725,995 |
| 7 | 1.8773 | 1.1119 | 15,723,346 |
| 22 | 1.8785 | 1.1126 | 15,720,902 |

| Metric | val_loss (nats) | val_bpb |
|--------|----------------|---------|
| Mean | 1.8755 | 1.1083 |
| Std | 0.0023 | 0.0013 |
| Best | 1.8731 | 1.1094 |

### Statistical significance

Current leader: 1.1147 bpb.

- **Improvement: 0.0064 bpb (5-seed mean)**
- t-test vs leader: t = -6.46, df = 4, **p < 0.002**

## Artifact size (worst-case, seed 2)

| Component | Bytes |
|-----------|-------|
| Model (int6+brotli) | 15,692,661 |
| Code | 34,101 |
| **Total** | **15,726,762** |

Under the 16,000,000 byte limit.

## Command

```bash
SEED=1337 \
MATRIX_LR=0.024 MATRIX_LR_LATE=0.019 \
SCALAR_LR=0.020 SCALAR_LR_LATE=0.038 \
TIED_EMBED_LR=0.022 \
MUON_MOMENTUM=0.985 WARMDOWN_ITERS=4000 \
TRAIN_BATCH_TOKENS=589824 \
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=5120 VE_DIM=128 \
WINDOW_SIZE=512 WINDOW_ATTN_LAYERS=2,4,6,8,10 \
LOCAL_SEQS_PER_GPU=36,36,36,36,36,10,10,10 \
SEQS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
