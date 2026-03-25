# 11L + LeakyReLU(0.5)² + Legal TTT

## What's new vs #2 entry (1.1233 bpb)

| Change | Expected Δ bpb |
|--------|---------------|
| LeakyReLU(0.5)² in MLP | −0.003 |
| Legal score-first TTT | −0.002 |
| **Total** | **−0.005** → ~1.118 |

## Architecture (inherited from PR #414 stack)
- 11 transformer layers
- XSA (Cross-Self Attention) on last 4 layers
- Partial RoPE (16 dims out of 64)
- LN-scale (layer-depth residual scaling)
- Value Embedding at layers 9–10 (dim 128)
- BigramHash embedding (vocab 2048, dim 128)
- GQA: 8 query heads / 2 KV heads
- FlashAttention 3 (falls back to SDPA if unavailable)
- EMA (decay 0.997) + SWA averaging
- GPTQ-lite int6 quantization + zstd compression

## Key additions

### LeakyReLU(0.5)²
`F.leaky_relu(x, negative_slope=0.5).square()` instead of `relu(x).square()`.
Preserves gradient flow through negative activations → fewer dead neurons.

### Legal TTT (Test-Time Training, PR #461 recipe)
Score-first protocol: every validation token is scored by a model that has
**never trained on it**. Process:
1. Divide validation into 32k-token chunks
2. For each chunk: **score first** (inference_mode), **then train** on it
3. Cosine LR decay over chunks: `lr = 0.002 × 0.5 × (1 + cos(π·i/N))`
4. SGD with momentum=0.9, gradient clip=1.0, 3 epochs per chunk
5. Last chunk is only scored (no update)

**Why it's legal**: evaluator cannot peek at the chunk before scoring.

## Run instructions

```bash
cd /workspace
git clone https://github.com/StolbaJ/parameter-golf.git
cd parameter-golf

pip install -q zstandard sentencepiece huggingface-hub datasets tqdm

# Download all data (80 train shards + val + tokenizer)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run on 8×H100
OMP_NUM_THREADS=28 RUN_ID=leakyrelu2_ttt SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-23_11L_LeakyReLU2_TTT/train_gpt.py
```

## Expected performance
- Pre-TTT: ~1.120–1.122 bpb (from LeakyReLU + #2 base)
- Post-TTT: ~1.118–1.120 bpb
- SOTA to beat: **1.1194 bpb**

## Step time
- With FA3: ~83 ms/step (~7500 steps in 10 min)
- Without FA3: ~100 ms/step (~6000 steps)
- TTT adds ~3–5 min at the end (outside 10-min training window)
