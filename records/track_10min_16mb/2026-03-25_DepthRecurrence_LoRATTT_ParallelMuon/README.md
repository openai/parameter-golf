# Submission: Depth Recurrence + LoRA TTT + BigramHash(3072)

Built on PR #549 (LeakyReLU² + Legal TTT + Parallel Muon, 1.1194 BPB) with:

## Key Improvements

1. **Depth Recurrence** (layers 4,5): Repeat layers 4 and 5 → 13 virtual layers from 11 physical. Zero parameter cost, just extra compute per step.

2. **LoRA TTT** (rank 8): Replace full-param SGD TTT with LoRA adapters on Q/V projections. Uses Adam optimizer instead of SGD. ~24x more effective in score-first framework per community benchmarks.

3. **BigramHash(2048)**: Keep proven bucket count to stay within 16MB artifact budget.

4. **SDPA Fallback**: Automatically falls back to PyTorch SDPA when FlashAttention 3 is not available.

## Architecture

- 11 physical layers (13 virtual via depth recurrence on layers 4,5)
- 512d, 8H/4KV GQA, 3x MLP with LeakyReLU(0.5)²
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale 1/√(layer+1)
- BigramHash(2048, dim=128) + SmearGate + ValueEmbedding(layers 9,10)
- Parameter Banking + Parallel Muon optimizer
- EMA(0.997) + Tight SWA weight averaging
- Int6 GPTQ-lite quantization + lzma compression
- Legal score-first LoRA TTT (rank 8, Adam, 3 epochs per 32K chunk)

## Quick Test (1xH100)

```bash
cd /workspace/parameter-golf

# Copy submission script
cp submission/train_gpt.py train_gpt_sub.py

# Download data if not already done
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Quick smoke test (1000 steps, ~10 min on 1xH100)
RUN_ID=test_v1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
DEPTH_RECURRENCE=4,5 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt_sub.py
```

## Full Run (8xH100)

```bash
RUN_ID=submission_v1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
DEPTH_RECURRENCE=4,5 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_USE_LORA=1 TTT_LORA_RANK=8 TTT_LORA_LR=0.01 \
TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt_sub.py
```
