# Late Soft-Round QAT + Score-First Backward-Looking TTT

**val_bpb: 1.1178** (3-seed mean, std 0.0005) | **~15.75 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-----------------|----------|----------|----------|
| 1337 | 85.3ms | 7,030 | 1.1201 | **1.1176** | -0.0025 | 405s | 15,700,318 |
| 42 | 85.3ms | 7,025 | 1.1209 | **1.1183** | -0.0026 | ~405s | 15,850,153 |
| 7 | 85.3ms | 7,035 | 1.1200 | **1.1174** | -0.0026 | ~405s | 15,706,617 |
| **Mean** | **85.3ms** | **7,030** | **1.1203** | **1.1178 (std 0.0005)** | **-0.0026** | **~405s** | |

## Architecture

Built on PR #414 stack with LeakyReLU(0.5)² from PR #493 and score-first backward-looking TTT from PR #461.

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with LeakyReLU(0.5)² (PR #493) |
| BigramHash | 3072 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | **Late Soft-Round QAT** + GPTQ-lite int6 + zstd |
| TTT Burst | 2 epochs, 100 steps, 0.1× LR |

## Novel Contribution: Late Soft-Round QAT

Standard STE quantization-aware training uses hard rounding in the forward pass and an identity-style surrogate in the backward pass, which provides no bin-aware gradient signal near quantization boundaries. We replace that late in training with a temperature-controlled soft-round surrogate.

When the warmdown schedule scale drops below 0.02, the quantizer keeps the hard quantized forward pass but uses a sigmoid-interpolated soft-round surrogate in the backward pass:

```python
frac = x / scale - floor(x / scale)
soft = floor(x / scale) + sigmoid((frac - 0.5) / tau)  # tau=0.1
```

This gives the optimizer a non-zero, bin-aware gradient signal that encourages weights to settle onto nearby int6 grid points just before EMA/SWA finalization.

## Score-First Backward-Looking TTT

Backward-looking adaptation following PR #461:

1. Validation tokens split into ~1,893 non-overlapping 32K-token chunks
2. For each chunk:
   - **SCORE**: Sliding window eval under `torch.inference_mode()` (stride=64, seq_len=2048)
   - **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

`inference_mode()` guarantees scoring is stateless — no gradients, no weight mutation.

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
| Standard sliding window eval | ~75s |
| Score-first backward-looking TTT | ~405s |
| **Total eval** | **~480s (< 10 min)** |

## Ablation

Incremental contribution (seed 1337):

| Change | Pre-TTT bpb | Post-TTT bpb | Delta |
|--------|-------------|-------------|-------|
| PR #414 base (relu²) | 1.1234 | — | — |
| + Late Soft-Round QAT + training stack | 1.1217 | — | -0.0017 |
| + Legal Score-First TTT (PR #461) | — | 1.1195 | -0.0022 |
| + LeakyReLU(0.5)² (PR #493) | 1.1201 | **1.1176** | -0.0021 |

## Reproduction

```bash
pip install -r requirements.txt
# Build FA3 Hopper kernels (required, ~10 min compile)
cd /tmp && git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention/hopper && python setup.py install

# Run all 3 seeds
for SEED in 1337 42 7; do
  SEED=$SEED RUN_ID=seed_${SEED} MAX_WALLCLOCK_SECONDS=600 TTT_ENABLED=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

- **Base model**: PR #414 by @signalrush
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **LeakyReLU² activation**: PR #493 by @parinzee
