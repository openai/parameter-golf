## Int6 + Wider MLP 3x + FP16 Embed + Sliding Window

**Status: WIP** - Awaiting 8xH100 SXM compute credits for official run.

**Estimated val_bpb: ~1.160** (based on 1xH100 extrapolation and component analysis)

Four orthogonal improvements over the naive baseline (1.2244 BPB):

### Changes from Baseline

**1. Wider MLP (MLP_MULT=3.0)**

3x MLP expansion (hidden=1536 vs baseline's 1024), increasing total parameters from 17.1M to 21.8M. Enabled by the int6 quantization scheme below which keeps the artifact under 16MB.

**2. int6 Per-Row Quantization on MLP + Attention Weights**

Mixed precision quantization:
- **int6 per-row** (31 levels) on all 2D MLP and attention projection weights
- **int8 per-row** (127 levels) on other large tensors
- Small/control tensors pass through as fp16/fp32

int6 values stored in int8 bytes — zstd-22 compresses the restricted range efficiently. Degradation: ~+0.010 BPB while saving ~4MB artifact space.

**3. FP16 Tied Embedding Passthrough** (novel addition)

The tied embedding doubles as the output logit head — the most quantization-sensitive tensor. Keeping it in fp16 instead of int8 reduces the post-quantization BPB gap by ~0.005 at a cost of only ~0.5MB additional artifact size, which fits within the int6 savings.

**4. Sliding Window Evaluation (stride=256)**

Overlapping windows where each scored token gets 768+ tokens of preceding context. Improves val_bpb by ~0.033 with zero artifact cost.

### Configuration

```
MLP_MULT=3.0
NUM_LAYERS=9
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
VOCAB_SIZE=1024
TRAIN_SEQ_LEN=1024
TRAIN_BATCH_TOKENS=524288
TIE_EMBEDDINGS=1
EVAL_STRIDE=256
FP16_KEEP_NAME_PATTERNS=tok_emb
```

Optimizer tuning (env var overrides):
```
MATRIX_LR=0.020
SCALAR_LR=0.020
TIED_EMBED_LR=0.030
MUON_MOMENTUM=0.99
MUON_MOMENTUM_WARMUP_STEPS=1500
MUON_MOMENTUM_WARMUP_START=0.92
WARMDOWN_ITERS=3000
```

### Run Command

```bash
FP16_KEEP_NAME_PATTERNS=tok_emb \
MATRIX_LR=0.020 \
SCALAR_LR=0.020 \
TIED_EMBED_LR=0.030 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
WARMDOWN_ITERS=3000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### 1xH100 Validation Results (3 minutes, 348 steps)

| Metric | Value |
|--------|-------|
| Pre-quant val_bpb | 2.1346 |
| int6+zstd roundtrip val_bpb | 2.1356 |
| int6+zstd sliding window val_bpb | 2.1333 |
| int6 quant gap | 0.001 BPB |
| Artifact size (int6+zstd + code) | 15,630,013 bytes |

### Improvement Breakdown (estimated for 8xH100 10min)

| Component | Estimated val_bpb | vs baseline |
|-----------|-------------------|-------------|
| Naive baseline (int8, standard eval) | 1.2244 | -- |
| + Wider MLP 3.0x + int6 + tuned optimizer | ~1.199 | -0.025 |
| + FP16 tied embedding | ~1.194 | -0.005 |
| + Sliding window stride=256 | **~1.160** | -0.034 |

### What We Tried and Rejected

- **QAT (int6 fake quantization during training)**: Eliminates quant gap but adds 54% step overhead. Net negative.
- **SEQ_LEN=4096**: Fewer total tokens processed, smaller sliding window gain. SEQ_LEN=1024 is better with wider MLP.
- **Depth recurrence**: Amplifies quantization noise (0.13 BPB gap). Not viable.
- **int8 + MLP_HIDDEN=960 + fp16 embed**: Conservative approach, fits under 16MB but wastes parameter budget.
