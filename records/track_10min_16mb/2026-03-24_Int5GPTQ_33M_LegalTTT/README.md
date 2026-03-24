# Record: int5 GPTQ + 33.6M model + Soft-Round QAT + Legal Score-First TTT

## Summary

**3-seed mean val_bpb = 1.1162 (std 0.0006)**

int5 GPTQ quantization (values in [-15, 15], 31 unique levels) with Hessian-aware error compensation enables a 33.6M parameter model to fit under 16MB. Soft-Round QAT replaces STE hard rounding with differentiable tanh-based rounding (alpha annealing 1→16) for better training quality at zero cost. Combined with early QAT at threshold 0.5, EMA 0.997, and legal score-first AdamW TTT with cosine LR decay across chunks.

## Key Innovations

1. **int5 quantization** — 31 unique values ([-15,15]) stored as int8, ~0.46 bytes/param after zstd. Lower entropy = better compression ratio than int6.
2. **GPTQ error compensation** — Hessian-aware column reordering + Cholesky error redistribution. 256-sample calibration on training data.
3. **33.6M param model** — MHA 8/8 (full attention), BigramHash 8192, MLP 3.5x (1792), enabled by int5 compression.
4. **Soft-Round QAT** — Differentiable rounding `s_α(y) = floor(y) + 0.5 * tanh(α·r) / tanh(α/2) + 0.5` replaces STE. Alpha anneals from 1→16 during QAT steps. Better gradient flow = better training quality at zero computational cost.
5. **Early QAT 0.5** — QAT clipping matched to int5 range (0.9995 percentile / 15.0), ~1750 QAT steps.
6. **EMA 0.997** — Exponential moving average of weights, tuned from 0.9985.
7. **Legal score-first TTT** — every token scored BEFORE any gradient update using it. Cosine LR decay across chunks.

## Architecture

- 11 layers, model_dim=512, 8 heads / 8 KV heads (MHA), MLP 3.5x relu²
- XSA on all 11 layers
- Partial RoPE 16/64, LN Scale (1/√(layer+1))
- SmearGate + OrthoInit
- BigramHash 8192, Shared VE128 (layers 9,10)
- Tight SWA (every 50) + EMA 0.997
- Muon lr=0.025, WD=0.04
- FA3 Hopper, ~98ms/step → ~6120 steps in 600s
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
- AdamW (lr=0.0001, wd=0.0), 3 epochs per chunk, cosine LR across chunks
- Last 2 blocks + norms + lm_head unfrozen (~5.8M / 33.6M params)
- Last chunk never trained on
- Every token scored BEFORE any gradient update using it
- Manual grad all_reduce (no DDP wrapper)

## Results

| Seed | TTT BPB | Artifact |
|------|---------|----------|
| 1337 | **1.1155** | 15,822,078 bytes |
| 42 | **1.1163** | 15,415,405 bytes |
| 7 | **1.1167** | 15,368,627 bytes |
| **Mean** | **1.1162** | |
| **Std** | **0.0006** | |

## Reproduction

```bash
# On 8xH100 SXM:
pip install --break-system-packages zstandard
# Build FA3 Hopper (see repo README for instructions)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

SEED=1337 SKIP_SLIDING=1 PRUNE_PCT=0.02 \
SOFT_ROUND_QAT=1 \
TTT_EPOCHS=3 TTT_LR=0.0001 TTT_OPTIMIZER=adamw \
TTT_FREEZE_BLOCKS=2 TTT_CHUNK_TOKENS=131072 \
TTT_TEMPERATURE=0.98 INT6_LAST_N=0 \
PPM_ALPHA=1.0 BYTE_WEIGHTED_TTT=0 USE_CACHE=0 \
ADAPTIVE_LR=0 USE_MIXER=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
