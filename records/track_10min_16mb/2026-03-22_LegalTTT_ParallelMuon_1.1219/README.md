# Legal Score-First TTT + Parallel Muon + Parameter Banking

**val_bpb: 1.1214** (legal TTT, 3-seed mean, std 0.0009) | **~16.0 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT eval time | Artifact |
|------|----------|-------|-------------|-----------------|----------|---------------|----------|
| 1337 | 82.5ms | 7,273 | 1.1224 | **1.1204** | -0.0020 | 413s | 15,979,188 |
| 42 | 82.5ms | 7,278 | 1.1237 | **1.1216** | -0.0021 | 406s | 15,987,108 |
| 2025 | 82.5ms | 7,270 | 1.1240 | **1.1221** | -0.0019 | 405s | 15,988,312 |
| **Mean** | **82.5ms** | **7,274** | **1.1233** | **1.1214 (std 0.0009)** | **-0.0019** | **~408s** | |

### Timing Budget

| Phase | Time | Constraint |
|-------|------|-----------|
| Training | 600s | ≤600s on 8×H100 |
| Standard eval (int6 roundtrip + sliding window) | ~120s | — |
| **Legal TTT (score-first sliding + adaptation)** | **~408s** | — |
| **Total eval** | **~528s (< 10 min)** | Eval ≤10 min |

Training completes in 600s (10 min). TTT runs during the evaluation phase — the same phase where other TTT submissions (#398, #442, #461) run their adaptation. Total eval is ~528s (~8.8 min).

## TTT Legality

This submission uses **backward-looking, score-first TTT** as established by PR #461. The implementation guarantees that every token is scored before the model can adapt on it:

### Code Flow (see `eval_val_sliding_ttt()`)

1. **Chunk partitioning**: Val tokens (62M) are divided into 1,893 non-overlapping 32K-token chunks.

2. **For each chunk** (in order):
   - **Phase 1 — SCORE** (`base_model.eval()` + `torch.inference_mode()`):
     Sliding window eval computes per-token NLL. `inference_mode()` is a PyTorch context manager that **disables all gradient tracking and prohibits in-place weight mutation**. No model weights change during scoring.
   - **Phase 2 — TRAIN** (only if NOT the last chunk):
     `base_model.train()` enables gradients. SGD updates model weights using only the tokens from this already-scored chunk. These updates improve predictions for **future** chunks only.

3. **Last chunk is never trained on**: The final chunk is scored but no adaptation follows, eliminating any edge case.

4. **Causal guarantee**: Chunk N is scored by a model that has adapted on chunks 0..N-1 only. Within each chunk, autoregressive causal masking ensures each token only attends to past context.

This is the same legal TTT framework validated in PR #461 (which was accepted as a non-record submission).

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | SGD + momentum(0.9) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 3 |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |
| Batch size | 32 sequences |

## Training Architecture

Built on PR #414's stack with Parameter Banking + Parallel Muon optimizer (first introduced in PR #399):

- 11L, 512d, 8H/4KV, MLP 3× (relu²)
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale
- SmearGate, BigramHash(3072), VE128 on layers 9-10
- EMA(0.997) + Tight SWA(every 50)
- GPTQ-lite int6 quantization + lzma compression
- **Parameter Banking**: 4 contiguous 3D banks replace 66 nn.Linear weights
- **Parallel Muon**: No DDP for banks. Post-backward reduce-scatter → local NS → all-gather

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Optimizer (Parameter Banking + Parallel Muon)**: PR #399 by @abaybektursun
- **TTT recipe**: PR #461 by @anantdgoel — legal score-first TTT with SGD+momentum, selective freezing
- **Base model**: PR #414 by @signalrush — GPTQ-lite, VE128, Tight SWA, warmdown=3500
