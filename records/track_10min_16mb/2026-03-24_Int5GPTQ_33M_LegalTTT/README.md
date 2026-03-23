# Record: int5 GPTQ + 33.6M model (3-seed mean val_bpb=1.1179)

## Summary

**3-seed mean val_bpb = 1.1179 (std 0.0008)**

Breakthrough compression: int5 quantization (values in [-15, 15], 31 unique levels) with GPTQ Hessian-aware error compensation. This enables a 33.6M parameter model to fit under 16MB — something impossible with int6 (which gives ~19.7MB at this model size).

GPTQ makes int5 nearly lossless: only 0.001 BPB quant tax vs int6's 0.006 BPB. The bigger model (33.6M vs 27M) more than compensates for the lower bit-width.

## Key Innovations

1. **int5 quantization** — 31 unique values ([-15,15]) stored as int8, ~0.46 bytes/param after zstd vs int6's ~0.58. Lower entropy = better compression ratio.
2. **GPTQ error compensation** — Hessian-aware column reordering + Cholesky error redistribution makes int5 nearly lossless. 256-sample calibration.
3. **33.6M param model** — MHA 8/8 (full attention), BigramHash 8192, MLP 3.5x (1792), enabled by int5 compression.
4. **Early QAT 0.5** — QAT clipping matched to int5 range (0.9995 percentile / 15.0), ~1750 QAT steps.
5. **EMA 0.997** — tuned from 0.9985.
6. **Legal score-first TTT** — every token scored BEFORE any gradient update using it.

## Architecture

- 11 layers, model_dim=512, 8 heads / 8 KV heads (MHA), MLP 3.5x relu²
- XSA on all 11 layers
- Partial RoPE 16/64, LN Scale (1/√(layer+1))
- SmearGate + OrthoInit
- BigramHash 8192, Shared VE128 (layers 9,10)
- Tight SWA (every 50) + EMA 0.997
- Muon lr=0.025, WD=0.04
- FA3 Hopper, ~99ms/step → ~6060 steps in 600s
- **33.6M params**, int5 GPTQ + zstd-22, 2% magnitude pruning

## Quantization Pipeline

1. **Early QAT** (threshold 0.5): QAT-aware training with int5 clipping (scale = row_clip / 15.0, clamp [-16, 15])
2. **GPTQ** (post-training): 256-sample Hessian calibration, per-row optimal scales (5-percentile search), column reordering by Hessian diagonal, block-128 Cholesky error compensation
3. **int5 quantization** (range [-15, 15], 31 levels) stored as int8
4. **zstd-22** compression
5. **2% magnitude pruning**

## Legal Score-First TTT

- Val data split into 131072-token chunks (474 chunks)
- For each chunk: **score first** (sliding window stride=32, inference_mode), **then** adapt
- AdamW (lr=0.0001, wd=0.0), 2-3 epochs per chunk, cosine LR across chunks
- Last 2 blocks + norms + lm_head unfrozen (~5.8M / 33.6M params)
- Last chunk never trained on
- Every token scored BEFORE any gradient update using it
- Manual grad all_reduce (no DDP wrapper)

## Results

| Seed | Sliding BPB | TTT BPB | Artifact | TTT Epochs |
|------|-------------|---------|----------|------------|
| 1337 | 1.1244 | **1.1170** | 15.53 MB | 3 |
| 42 | 1.1249 | **1.1182** | 15.36 MB | 3 |
| 7 | 1.1250 | **1.1184** | 15.28 MB | 2 |
| **Mean** | **1.1248** | **1.1179** | | |
| **Std** | | **0.0008** | | |

## Reproduction

```bash
# On 8xH100 SXM:
pip install --break-system-packages zstandard
pip install --break-system-packages flash-attn --no-build-isolation
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 PRUNE_PCT=0.02 TTT_EPOCHS=3 TTT_LR=0.0001 \
TTT_OPTIMIZER=adamw TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=131072 \
EVAL_STRIDE=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
