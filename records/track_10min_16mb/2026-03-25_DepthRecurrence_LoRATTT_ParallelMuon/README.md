# Record: 1.1182 BPB — Depth Recurrence + SGD TTT

**val_bpb: 1.1182** | **~15.93 MB** | 8×H100 SXM | Seed 1337

Built on PR #549 (LeakyReLU² + Legal TTT + Parallel Muon, 1.1194 BPB) with:

## Key Innovation: Depth Recurrence

Repeat layers 4 and 5 → 13 virtual layers from 11 physical. Zero parameter cost, just extra compute per step. First successful use of depth recurrence on the leaderboard — previously dismissed as "needs way more steps than 10 min allows."

## Results (8×H100 80GB SXM, seed 1337)

| Metric | Value |
|--------|-------|
| step_avg | 98.23ms |
| Steps completed | 6,109 |
| Pre-TTT int6 sliding window BPB | 1.1208 |
| **Post-TTT BPB** | **1.1182** |
| TTT gain | -0.0026 |
| TTT time | 454s |
| Artifact size | 15,930,486 bytes |

## Architecture

- 11 physical layers (13 virtual via depth recurrence on layers 4,5)
- 512d, 8H/4KV GQA, 3x MLP with LeakyReLU(0.5)²
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale 1/√(layer+1)
- BigramHash(2048, dim=128) + SmearGate + ValueEmbedding(layers 9,10)
- Parameter Banking + Parallel Muon optimizer
- EMA(0.997) + Tight SWA weight averaging
- Int6 GPTQ-lite quantization + lzma compression
- Legal score-first SGD TTT (3 epochs per 32K chunk, all blocks unfrozen)
- SDPA fallback when FlashAttention 3 is not available

## TTT Protocol

Backward-looking, score-first TTT following PR #461's framework:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |

### Timing Budget

| Phase | Time |
|-------|------|
| Training | 600s (≤10 min) |
| Standard eval (int6 roundtrip + sliding window) | ~93s |
| Legal TTT (score-first sliding + adaptation) | ~454s |
| **Total eval** | **~547s (< 10 min)** |

## Run Command

```bash
RUN_ID=8gpu_depth_sgd_ttt_v1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
DEPTH_RECURRENCE=4,5 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_USE_LORA=0 \
TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt_sub.py
```

## Credits

- **Depth Recurrence**: First successful application on this leaderboard
- **Base model + LeakyReLU² + TTT + Parallel Muon**: PR #549 by @abaybektursun
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **Architecture stack**: PR #414 by @signalrush
- **SDPA fallback**: Added for compatibility with non-FA3 environments
