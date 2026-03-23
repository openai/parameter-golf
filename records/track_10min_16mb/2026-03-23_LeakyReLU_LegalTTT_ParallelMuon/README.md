# LeakyReLU² + Legal Score-First TTT + Parallel Muon

**val_bpb: 1.1194** (3-seed mean, std 0.0006) | **~15.95 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 83.3ms | 7,179 | 1.1217 | **1.1192** | -0.0025 | 410s | 15,977,386 |
| 42 | 83.4ms | 7,182 | 1.1227 | **1.1200** | -0.0027 | 408s | 15,876,510 |
| 2025 | 83.4ms | 7,193 | 1.1212 | **1.1189** | -0.0023 | 408s | 15,990,006 |
| **Mean** | **83.4ms** | **7,185** | **1.1218** | **1.1194 (std 0.0006)** | **-0.0025** | **~409s** | |

## Key Innovation: LeakyReLU(0.5)²

One-line activation change that delivers -0.003 BPB:

```python
# Standard (relu²)
x = torch.relu(self.fc(x)).square()

# This submission (leaky relu²)
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

LeakyReLU with slope 0.5 preserves negative gradient flow through the MLP, allowing the model to learn from both positive and negative pre-activations. The squaring step still produces non-negative outputs, maintaining the relu² inductive bias while eliminating dead neurons.

This activation is used in PR #493 (ablated at -0.003 BPB) and PR #518 (part of their 1.0622 record submission).

## Legal TTT Protocol

Backward-looking, score-first TTT following PR #461's framework:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. **For each chunk**:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation possible
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

`inference_mode()` is a PyTorch context manager that disables gradient tracking and prohibits in-place weight mutation, providing a hard guarantee that scoring is stateless.

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
| Standard eval (int6 roundtrip + sliding window) | ~120s |
| Legal TTT (score-first sliding + adaptation) | ~410s |
| **Total eval** | **~530s (< 10 min)** |

## Training Architecture

PR #414 stack with Parameter Banking + Parallel Muon (PR #399):

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.5)²** |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |

### Parameter Banking + Parallel Muon

First introduced in [PR #399](https://github.com/openai/parameter-golf/pull/399):

- 4 contiguous 3D `nn.Parameter` banks replace 66 separate `nn.Linear` weights
- Batched Newton-Schulz orthogonalization via `torch.bmm`
- DDP removed for banks; async reduce-scatter → local NS → async all-gather
- 83.3ms/step vs ~85ms baseline

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
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

## Ablation

Incremental contribution of each technique (all seed 1337):

| Change | Pre-TTT bpb | Post-TTT bpb | Delta |
|--------|-------------|-------------|-------|
| PR #414 base (relu², BIGRAM=2048) | 1.1234 | — | — |
| + Parameter Banking + Parallel Muon | 1.1234 | — | ±0.0000 |
| + Legal TTT (3ep, freeze=2) | — | 1.1217 | -0.0017 |
| + TTT freeze=0 (all blocks) | — | 1.1213 | -0.0004 |
| + BigramHash 2048→3072 | — | 1.1204 | -0.0009 |
| + **LeakyReLU(0.5)²** | 1.1213 | **1.1183** | **-0.0021** |

## Credits

- **LeakyReLU² activation**: PR #493 by @jxnl, PR #518
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @anantdgoel
- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
