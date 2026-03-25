# Fixed Bank QAT + XSA5 + Label Smoothing (Non-Record)

**val_bpb: 1.1352** (single seed) | **~15.44 MB** | 8xH100 SXM

This is a **non-record experimental submission** exploring whether fixing the broken QAT mechanism and tuning hyperparameters could improve over the SOTA (1.1194 BPB). It did not beat the SOTA.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

### Run 1: With QAT fix enabled

| Seed | Steps | Step Avg | Pre-TTT sliding BPB | Post-TTT BPB |
|------|-------|----------|---------------------|-------------|
| 1337 | 6,719 | 89.3ms | — | 1.1376 |

QAT recompilation cost ~50s + 5ms/step STE overhead, losing ~460 training steps vs SOTA's 7,179.

### Run 2: Without QAT (XSA5 + label smoothing + TTT tuning only)

| Seed | Steps | Step Avg | Sliding BPB (s64) | Post-TTT BPB |
|------|-------|----------|-------------------|-------------|
| 1337 | 7,062 | 85.0ms | 1.1621 | **1.1352** |

## Key Findings

### 1. QAT Fix: Sound Idea, Too Expensive

The SOTA has a **0.0083 BPB quantization gap** (pre-quant 1.1369 → post-quant 1.1452) because QAT is completely non-functional:

- Bank parameters (~96% of model weights) bypass `CastedLinear` — they use raw `F.linear()`
- `torch.compile` constant-folds the `CastedLinear._qat_enabled` flag at first trace

Our fix implements STE int6 fake-quantization directly in `GPT.forward()` for all bank parameters, using a plain Python bool `_qat_active` that gets constant-folded by torch.compile after `torch._dynamo.reset()` + recompile.

**However**, the recompilation costs ~50 seconds of wall-clock time plus ~5ms/step STE overhead, resulting in 460 fewer training steps. The lost steps hurt more than QAT helps. A future approach might:
- Apply QAT from the start (no recompile needed) — but STE overhead for all 7000 steps = ~35s lost
- Use a cheaper fake-quantization method
- Find a way to make STE work within torch.compile without overhead

### 2. Label Smoothing: Counterproductive

Label smoothing (0.05) weakened gradient signals. The model is compute-limited (only 7000 steps), not overfitting. Regularization hurts undertrained models.

### 3. XSA5: Neutral to Slightly Negative

Expanding XSA from last 4 to last 5 layers did not help. The SOTA's choice of XSA4 appears well-calibrated.

### 4. TTT Hyperparameters: Original was Better

TTT_LR=0.003/TTT_MOMENTUM=0.95 performed worse than SOTA's 0.002/0.9.

## Architecture

Same as SOTA (LeakyReLU + Legal TTT + Parallel Muon) with:

| Component | SOTA | This Submission |
|-----------|------|-----------------|
| XSA | Last 4 layers | Last 5 layers |
| Label Smoothing | 0 | 0.05 |
| TTT LR | 0.002 | 0.003 |
| TTT Momentum | 0.9 | 0.95 |
| Bank QAT | Broken (dead code) | Fixed but too expensive |

## Run Command (Run 2, no QAT)

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=5 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.003 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.95 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 LABEL_SMOOTHING=0.05 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base model**: LeakyReLU + Legal TTT + Parallel Muon by @abaybektursun (PR #549)
- **QAT bug diagnosis**: torch.compile constant-folding documented in PR #374 README
