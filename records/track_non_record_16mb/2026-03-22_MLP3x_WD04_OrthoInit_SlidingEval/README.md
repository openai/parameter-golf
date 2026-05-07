This record captures a non-record local development run on an RTX 3060 (1000 steps, 65K tokens/step). The purpose is to document techniques that individually and collectively improve post-quantization BPB, validated through 38+ controlled experiments. These techniques are ready for H100-scale verification.

## Changes from Baseline

The training script (`train_gpt.py`, snapshot of `train_gpt_exp.py`) adds the following features, all toggled via environment variables and off by default:

### 1. MLP Width Multiplier (`MLP_MULT=3`)
Widens the feedforward hidden dimension from `2 * model_dim` to `3 * model_dim`, increasing model capacity within the same layer count. Every top leaderboard entry uses this. Adds ~3.5M params (19.4M → 21.8M). The single most impactful architecture change.

### 2. Decoupled Weight Decay (`WEIGHT_DECAY=0.04`)
Applies decoupled weight decay in the Muon optimizer: `p.mul_(1.0 - lr * wd)` after the Newton-Schulz update. This was the largest single training improvement found (+0.007 BPB) and critically reduces the quantization gap from ~0.004 to ~0.001 BPB — weight decay regularizes weight magnitudes, making them more quantization-friendly.

### 3. Orthogonal Initialization (`ORTHO_INIT=1`)
Replaces default initialization of 2D weight matrices in `CastedLinear` with `nn.init.orthogonal_()`, preserving `_zero_init` markers. Small but consistent improvement (+0.003 BPB).

### 4. Sliding Window Evaluation (`EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64`)
Evaluates with overlapping windows: each 1024-token window shares context with the previous one, and only fresh (non-context) tokens contribute to the loss. This is implemented by feeding the full window through the model and masking context targets with `ignore_index=-100` in `cross_entropy`. Stride 64 yields ~0.035 BPB improvement over standard evaluation, for free at inference time.

### 5. Additional Implemented Features (not used in this run)
- **Int6 quantization** (`QUANT_BITS=6`): Clamps weights to [-31, 31] for smaller artifacts (~25% more params fit in 16 MB)
- **QAT** (`QAT=1`): Straight-Through Estimator fake quantization during training
- **Bigram hash embeddings** (`BIGRAM_HASH_SIZE=N`): Hash-based bigram context, zero-initialized as residual
- **zstd compression** (`COMPRESSOR=zstd`): Level 22, ~40% smaller artifacts than zlib
- **SWA** (`SWA_START_FRAC`): EMA weight averaging — **found to be quantization-hostile**, never used in best configs

## Experiment Summary (38 experiments)

| Finding | Impact | Experiments |
|---------|--------|-------------|
| MLP_MULT=3 | +0.04 BPB | exp11, exp12, exp13 |
| WEIGHT_DECAY=0.04 | +0.007 BPB, quant gap 0.001 | exp22, exp28, exp29 |
| Sliding eval stride=64 | +0.035 BPB (free) | exp30, exp38 |
| OrthoInit | +0.003 BPB | exp23, exp28 |
| Int8 > Int6 when fits | 0.001 vs 0.025 quant gap | exp12 vs exp11, exp24 |
| zstd vs zlib | 40% size reduction | exp37 |
| SWA: quantization-hostile | -0.12 BPB post-quant | exp25, exp31, exp32, exp33 |
| Layer sharing: dead end | no improvement in 5 runs | exp2-5, exp7-8 |
| 11L best pre-quant val | 1.4450 (but tested with SWA) | exp36 |

## Configuration

- Track: `non-record`, local development, under the `16,000,000` byte artifact cap
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Training: `WEIGHT_DECAY=0.04 ORTHO_INIT=1`
- Batching: `TRAIN_BATCH_TOKENS=65536 TRAIN_SEQ_LEN=1024` (local GPU constraint)
- Eval: `EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64`

Command:
```bash
MLP_MULT=3 WEIGHT_DECAY=0.04 ORTHO_INIT=1 \
EVAL_SLIDING_WINDOW=1 EVAL_WINDOW_STRIDE=64 \
ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=200 WARMUP_STEPS=5 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Key Metrics (from `train.log`)

- Training stopped at `1000/1000` steps (local run, no wallclock cap)
- Pre-quant eval: `val_loss:2.4521`, `val_bpb:1.4522`
- Post-quant roundtrip eval: `val_loss:2.4543`, `val_bpb:1.4536`
- Quantization gap: `0.0014 BPB` (excellent — weight decay effect)
- Exact printed metric: `final_int8_zlib_roundtrip_exact val_bpb:1.45360348`
- Model params: `21,778,504`
- Train time: `662069ms` (`step_avg:662.07ms`, single RTX 3060)
- Serialized model int8+zlib: `14,595,701 bytes`
- Code size: `64,829 bytes`
- Total submission size int8+zlib: `14,660,530 bytes` (well under 16 MB cap)

Training volume:
- Batch: `65,536` tokens/step (local constraint; H100 uses 524,288)
- Total train tokens seen: `65,536,000` (vs ~7.2B on H100 in 10 min)

## H100 Projections

Based on local scaling curves and the baseline's known H100 performance (1.2244 BPB at 13,780 steps × 524K batch):
- Our config at H100 scale is projected to reach **~1.18–1.21 BPB**
- Deeper variants (10–11L) may push to **~1.16–1.19 BPB**
- The 14.6 MB artifact at Int8+zlib leaves ~1.4 MB headroom, enough for 10–11 layers with zstd

## Included Files

- `train_gpt.py` — code snapshot (1500 lines, all features env-var toggled)
- `train.log` — exact local training log
- `submission.json` — leaderboard metadata
