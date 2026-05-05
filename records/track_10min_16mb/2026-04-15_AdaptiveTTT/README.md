# Adaptive TTT: Difficulty-Aware Epoch Allocation

**Adaptive Test-Time Training on the SOTA #1 stack (PR #1493)**

**Baseline reproduction (non-adaptive): 1.0832 bpb** (quantized+TTT, seed=42) | **15.28 MB** artifact | 8×H100 SXM

## Technique: Adaptive TTT

Current SOTA applies uniform 3 epochs of SGD to every 32K validation chunk during test-time training. But chunks vary in difficulty — some are easy text the model handles well, others are hard (rare topics, unusual structure) where additional adaptation helps most.

**Adaptive TTT** dynamically allocates training epochs per chunk based on difficulty:

```
After scoring each chunk (score-first protocol):
  chunk_nll = mean NLL of scored tokens in chunk
  running_mean = EMA(chunk_nll, alpha=0.3)
  difficulty = chunk_nll / running_mean
  epochs = clamp(round(base_epochs * difficulty), min=1, max=5)
```

- **Hard chunks** (NLL > running mean) → up to 5 epochs
- **Easy chunks** (NLL < running mean) → as few as 1 epoch
- **Average chunks** → ~3 epochs (same as baseline)

This focuses compute where it matters, uses the ~139s of eval time headroom, and costs **zero additional model parameters or artifact size**.

### New Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TTT_ADAPTIVE` | 1 | Enable adaptive epoch allocation |
| `TTT_MAX_EPOCHS` | 5 | Maximum epochs for hardest chunks |
| `TTT_MIN_EPOCHS` | 1 | Minimum epochs for easiest chunks |
| `TTT_ADAPT_EMA` | 0.3 | EMA smoothing for running mean NLL |

## Results

### Run 1: SOTA Reproduction (non-adaptive baseline)

| Eval Stage | val_loss | val_bpb | Time |
|---|---|---|---|
| Pre-quantization post-EMA | 2.8146 | 1.0896 | 7,442ms |
| Quantized | 2.8449 | 1.1013 | 24,637ms |
| Quantized + Sliding Window | 2.8019 | 1.0847 | 125,145ms |
| **Quantized + TTT** | **2.7980** | **1.0832** | **376,575ms** |

- Seed: 42
- Training: 4587 steps in 588s (wallclock capped)
- Artifact: 16,022,779 bytes (15,972,076 model + 50,703 code)
- Total eval time: ~534s (66s headroom under 600s)

### Run 2: Adaptive TTT (epoch-based)

Same model as Run 1 (same seed=42); only the TTT eval protocol differs.

**Key finding**: chunk-level NLL variance across validation data is very narrow (±6% around the running mean). With `difficulty = chunk_NLL / running_mean_NLL`, all 1,238 chunks received `round(3 × difficulty) = 3` epochs — identical to baseline. The difficulty range (0.93–1.06) never crosses the ±17% threshold needed for integer rounding to produce 2 or 4 epochs.

**Result: 1.08319 bpb** (identical to baseline 1.08320, confirming the analysis).

Eval time: 1,274,666ms on 1×H100 (would be ~377s on 8×H100, well under 600s limit).

### Proposed Fix: Continuous LR-Scaling (future work)

Instead of discrete epoch allocation (which suffers from integer rounding), scale the TTT learning rate continuously per chunk:
```
lr_scale = 1.0 + amplification * (chunk_nll - running_mean) / running_mean
chunk_lr = base_lr * clamp(lr_scale, 0.3, 2.0)
```
This eliminates the rounding problem entirely — every chunk gets a different learning rate proportional to its difficulty, even with narrow NLL variance.

## Architecture (inherited from SOTA #1, PR #1493)

- 11L × 512d × 8H/4KV, MLP 4x, LeakyReLU(0.5)²
- Partial RoPE 16/64, QK-Gain 5.25
- 3-layer depth recurrence (loops 3-5, activated at 35% training)
- Parallel residuals (L7+)
- XSA all layers, skip gates, layerwise LN scale
- 35.9M parameters

### Training
- MuonEq-R optimizer, WD=0.095, MLR=0.022
- EMA decay=0.9965, warmdown_frac=0.72
- ~4550 steps in 588s on 8×H100

### Quantization
- Full-Hessian GPTQ: int6 matrices (SDClip k=12.85), int8 embeddings (k=20.0)
- Byte-shuffle + Brotli-11 compression

### TTT
- Legal score-first protocol
- 32K-token chunks, SGD (lr=0.005, momentum=0.9)
- Baseline: 3 epochs/chunk, cosine LR decay
- **Adaptive: 1-5 epochs/chunk based on difficulty**

## Compliance

- [x] Training under 600s (588s)
- [x] Artifact under 16MB (15.28MB)
- [ ] Eval under 600s (534s non-adaptive; adaptive TBD)
- [x] No SLOT
- [x] No pre-quant TTT
- [x] No ETLB
- [x] No n-gram cache
- [x] Score-first TTT
- [ ] Three seeds (budget-limited, 1 seed completed)

## Reproduction

```bash
# Clone and enter repo
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download SP8192 dataset from kevclark's repo
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10

pip install brotli sentencepiece

# Baseline (non-adaptive TTT)
VOCAB_SIZE=8192 TTT_ENABLED=1 COMPRESSOR=brotli \
  TTT_ADAPTIVE=0 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

# Adaptive TTT
VOCAB_SIZE=8192 TTT_ENABLED=1 COMPRESSOR=brotli \
  TTT_ADAPTIVE=1 TTT_MAX_EPOCHS=5 TTT_MIN_EPOCHS=1 TTT_ADAPT_EMA=0.3 \
  SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

- Base stack: @bigbag — SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT (PR #1493)
- SP8192 + GPTQ + SDClip: @clarkkev (PR #1394)
- Depth recurrence: @dexhunter (PR #1331, #1437)
- Parallel residuals: @Robby955 (PR #1412), @msisovic (PR #1204)
- Legal TTT framework: @abaybektursun (PR #549), @dexhunter (PR #1413)
- **Adaptive TTT**: @kunwar-vikrant (novel technique, this PR)
