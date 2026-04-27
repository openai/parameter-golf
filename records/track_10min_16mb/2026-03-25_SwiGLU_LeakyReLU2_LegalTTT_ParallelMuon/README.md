# SwiGLU + Legal Score-First TTT + Parallel Muon

**val_bpb: TBD** (3-seed mean) | **~15.9 MB** | 8×H100 SXM

Built on the PR #549 SOTA stack (LeakyReLU² + Legal TTT + Parallel Muon). Single change: replace the LeakyReLU(0.5)² activation with **SwiGLU** gating.

## Key Innovation: SwiGLU MLP

SwiGLU is a gated linear unit (Shazeer 2020) used in LLaMA, Mistral, Gemma, and PaLM.
It replaces the non-gated squared activation with a multiplicative gate:

```python
# Before (SOTA): non-gated LeakyReLU²
x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5).square()
out = F.linear(x, down_w)

# After (this submission): SwiGLU gated activation
half = up_w.shape[0] // 2
gate = F.silu(F.linear(x, up_w[:half]))   # learned gate
up   = F.linear(x, up_w[half:])           # content projection
out  = F.linear(gate * up, down_w)
```

### Parameter Neutrality

The bank shapes change to preserve the exact same parameter count per layer:

| Tensor | Old shape | New shape | Params |
|--------|-----------|-----------|--------|
| `mlp_up_bank[i]` | (1536, 512) | (2048, 512) | stores gate\|\|up |
| `mlp_down_bank[i]` | (512, 1536) | (512, 1024) | down projection |
| **Total MLP/layer** | **1,572,864** | **1,572,864** | ✓ same |

Proof: `2 × 512 × 1536 = 3 × 512 × 1024 = 1,572,864`.

### Why SwiGLU over LeakyReLU²

- **Input-dependent gating**: the sigmoid gate (`silu`) selects which features to pass through, allowing the MLP to learn feature-selective routing rather than a uniform nonlinearity.
- **No dead neurons**: `silu` has non-zero gradient everywhere, unlike the positive half of relu².
- **Literature**: SwiGLU improves over relu²/gelu in LLaMA (Touvron et al. 2023), PaLM (Chowdhery et al. 2022), and Gemma. This is the first application of SwiGLU in this challenge.

## Training Architecture (identical to PR #549)

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| **MLP activation** | **SwiGLU** (gate\|\|up bank → half each, down 1024) |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/√(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma |
| Optimizer | Parameter Banking + Parallel Muon |
| Legal TTT | score-first, SGD(lr=0.002, mom=0.9), 3ep, all blocks |

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
USE_SWIGLU=1 SWIGLU_HALF_DIM=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **SOTA stack**: PR #549 by @abaybektursun (LeakyReLU² + Legal TTT + Parallel Muon)
- **TTT recipe**: PR #461 by @Christopher-Lee-McClendon
- **SwiGLU**: Shazeer (2020), widely used in LLaMA, Mistral, Gemma
