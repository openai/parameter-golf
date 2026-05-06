# Non-Record Submission: SP8192 Tempered BPB-Weighted Loss + Easy-Chunk TTT + Late Loop Onset

**Author:** Artem Buldin ([@Buld1n](https://github.com/Buld1n))  
**Track:** `non_record_16mb`  
**Status:** completed, negative result

## What This Is

This experiment keeps our strongest valid `loop2600` SP8192 stack almost unchanged and modifies only the training loss:

1. **Late step-based loop onset** at `2600`
2. **Pass-gated recurrence** in the looped band
3. **Easy-chunk legal TTT**
4. **Tempered BPB-weighted training loss**
5. **Control-int8 packing**

The idea is simple: the leaderboard metric is bits per byte, but standard cross-entropy weights tokens uniformly. During training, this variant reweights token loss by the tokenizer byte count so the optimization target is closer to the final evaluation metric. Unlike the raw weighting variant, it tempers the byte weights with a power transform and a clip so a narrow set of long-byte tokens cannot dominate the gradient.

## Key Change

When `BPB_WEIGHTED_LOSS=1`, train-time CE becomes:

```python
per_token_loss = cross_entropy(..., reduction="none")
byte_weights = base_bytes_lut[target_ids].clamp_min(1.0)
byte_weights = byte_weights.pow(BPB_WEIGHT_POWER)
byte_weights = byte_weights.clamp_max(BPB_WEIGHT_CLIP)
byte_weights = byte_weights / byte_weights.mean()
loss = (per_token_loss * byte_weights).mean()
```

The byte lookup already exists in the codebase for BPB evaluation, so this adds no model parameters and negligible runtime overhead.

## Observed Result

Server run: `sp8192_tbpbw_l2600_s42_r1`

- `3713` steps in the effective training window
- `3713/20000 val_bpb = 1.1092`
- `pre-quantization post-ema val_bpb = 1.10931410`
- `quantized val_bpb = 1.11968686`
- `Serialized model quantized+brotli = 15,967,346 bytes`
- `Total submission size quantized+brotli = 16,065,507 bytes`

This was a large improvement over the raw byte-weighted loss variant, but it is still far from the strong `loop2600` baseline family and also misses the 16 MB size limit. Tempering fixed the worst train-side instability, but not enough to make the direction competitive.

## Planned Recipe

- SP8192 tokenizer / dataset
- 11 layers, 512 dim, 8 heads / 4 KV heads
- looped band over layers `3..5`
- `ENABLE_LOOPING_AT_STEP=2600`
- `RECUR_ATTN_GATE=1`
- `BPB_WEIGHTED_LOSS=1`
- `BPB_WEIGHT_POWER=0.5`
- `BPB_WEIGHT_CLIP=2.0`
- `TTT_ENABLED=1`
- `TTT_PARAM_MODE=full`
- `TTT_EASY_CHUNK_RATIO=0.998`
- `TTT_EASY_CHUNK_EPOCHS=1`
- `TTT_OUTLIER_DROP_FRACTION=0.03`
- `TTT_SCORE_WEIGHT_POWER=0.5`

## Setup

```bash
pip install brotli sentencepiece numpy
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

## Reproduction

```bash
SEED=42 \
QK_GAIN_INIT=5.0 \
QK_GAIN_DEPTH_RAMP=0.5 \
PARALLEL_RESIDUAL_START=6 \
ENABLE_PARALLEL_RESIDUAL_AT_STEP=0 \
ENABLE_LOOPING_AT_STEP=2600 \
RECUR_ATTN_GATE=1 \
RECUR_ATTN_GATE_SCALE=0.5 \
BPB_WEIGHTED_LOSS=1 \
BPB_WEIGHT_POWER=0.5 \
BPB_WEIGHT_CLIP=2.0 \
TTT_ENABLED=1 \
TTT_PARAM_MODE=full \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_EASY_CHUNK_RATIO=0.998 \
TTT_EASY_CHUNK_EPOCHS=1 \
TTT_OUTLIER_DROP_FRACTION=0.03 \
TTT_SCORE_WEIGHT_POWER=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Included Files

- `README.md`
- `requirements.txt`
- `train_gpt.py`
- `train_seed42.log`
- `submission.json`
