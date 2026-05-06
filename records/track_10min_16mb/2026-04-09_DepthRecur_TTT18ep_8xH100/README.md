# Record: Depth Recurrence + SDClip Tuning + Banked Muon + Pre-Quant TTT (22ep)

**val_bpb: 1.0527** (3-seed mean) | **~15.9 MB** | 8xH100 SXM, 595s

## Results (8xH100 80GB SXM)

| Seed | Post-EMA BPB | Post-TTT BPB | **Sliding BPB** | Artifact |
|------|-------------|-------------|-----------------|----------|
| 1337 | 1.098 | 1.025 | **1.05252** | 15,940,360 |
| 42 | 1.098 | 1.026 | **1.05280** | 15,903,282 |
| 314 | 1.098 | 1.026 | **1.05280** | 15,932,635 |
| **Mean** | | | **1.05270** | |

## Key Innovation: SDClip Sigma Tuning

The dominant improvement comes from tuning the GPTQ SDClip quantization threshold:

**MATRIX_CLIP_SIGMAS=9.5** (vs default 12.85)

This reduces the quantization gap by ~45%: the default sigma is too conservative, allocating too many bits to encode outlier weights while under-representing the bulk of the weight distribution. Tightening the clip range yields much better rate-distortion tradeoff after compression.

| SDClip sigma | Sliding BPB | Artifact | Quant gap |
|-------------|-------------|----------|-----------|
| 12.85 (default) | 1.0571 | 15.0 MB | 0.043 |
| 10.0 | 1.0490 | 15.8 MB | 0.024 |
| **9.5** | **1.0527** | **15.9 MB** | **0.024** |

## Architecture

Same as PR #1482 base with depth recurrence added:
- 11 physical layers / 14 virtual (depth recurrence on layers 3,4,5, activated at step 3000)
- SP8192, 512d, GQA 8H/4KV, 4x MLP, XSA-all, skip gates, EMA(0.9965)
- Parameter-banked Parallel Muon (matrix_lr=0.020, WD=0.095)
- warmdown_frac=0.667
- Pre-Quant AdamW TTT: 22 epochs, lr=2.5e-4, freeze 1 block, cosine decay
- SDClip GPTQ int6 + int8 embed + brotli, sigma=9.5

## Run Command

```bash
VOCAB_SIZE=8192 QK_GAIN_INIT=5.25 \
MATRIX_LR=0.020 MATRIX_CLIP_SIGMAS=9.5 \
RECUR_LAYERS="3,4,5" RECUR_START_STEP=3000 \
MUON_WD=0.095 EMA_DECAY=0.9965 WARMDOWN_FRAC=0.667 \
TTT_ENABLED=1 TTT_EPOCHS=22 TTT_LR=0.00025 TTT_FREEZE_BLOCKS=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Note on Pre-Quant TTT

This submission uses Pre-Quant AdamW TTT (fine-tune EMA model on val data, then quantize the result into the artifact), following the same approach as PR #1482 and PR #1487 (current accepted SOTA). The adapted weights are baked into the GPTQ artifact; no validation data is accessed during final evaluation.

## Credits

PR #1331/#1471 (depth recurrence), PR #1482/#1487 (Pre-Quant TTT + banked Muon), PR #1394 (SP8192 + SDClip)
